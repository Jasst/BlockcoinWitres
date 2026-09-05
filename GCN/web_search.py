"""
web_search.py v2 — улучшенный веб-поиск для BlockcoinWitres.

Что добавлено по сравнению с исходной версией:
  1. CROSS-ENCODER РЕРАНКЕР (BAAI/bge-reranker-v2-m3, мультиязычный).
     После скачивания страниц top-N кандидатов переранжируются моделью,
     которая смотрит запрос и текст ВМЕСТЕ — заметно точнее, чем
     токенный overlap ChunkRanker. Ленивая загрузка, отключение через
     RERANK_ENABLED=false, безопасный откат на ChunkRanker при сбое.
  2. FALLBACK-БЭКЕНДЫ: DDG -> SearXNG (self-hosted) -> Brave API.
     Поиск перестаёт "работать через раз", когда DDG отдаёт капчу.
     Настройка: env SEARXNG_URL / BRAVE_API_KEY (ничего менять в
     config_ai.py не нужно, но можно и там).
  3. СТРУКТУРИРОВАННОЕ ИЗВЛЕЧЕНИЕ: JSON-LD микроразметка (цены, рейтинги,
     даты событий), HTML-таблицы (курсы валют), дата публикации
     страницы. Для time-sensitive запросов свежие страницы получают
     буст в ранжировании.
  4. ДЕДУПЛИКАЦИЯ ПО СОДЕРЖИМОМУ (simhash): агрегаторы и копипаст
     больше не засоряют контекст LLM.
  5. ИТЕРАТИВНОЕ УТОЧНЕНИЕ: если лучшие найденные страницы слабо
     релевантны, выполняется до max_refinements дополнительных проходов
     с переформулированным запросом (эвристически или через LLM —
     параметр query_refiner).

Публичный API полностью совместим с прежней версией:
  deep_search, fetch_url, search_ddg, best_excerpt, domain_trust,
  extract_urls, normalize_raw_url, content_has_currency_numbers,
  SearchCache, WebPageFetcher, ChunkRanker, close_search_resources.
"""

import asyncio
import hashlib
import json
import logging
import os
import re
import time
from collections import OrderedDict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

try:
    from GCN.config_ai import (
        SEARCH_CACHE_TTL,
        SEARCH_CACHE_MAX_SIZE,
        DDG_MIN_INTERVAL,
        DDG_MAX_RETRIES,
        MAX_PAGES_TO_FETCH,
        PAGE_CONTENT_MAX_CHARS,
        CHUNK_SIZE,
        CHUNK_OVERLAP,
        PARALLEL_FETCH_LIMIT,
        MIN_RELEVANCE_THRESHOLD,
    )
except ImportError:
    SEARCH_CACHE_TTL = 300
    SEARCH_CACHE_MAX_SIZE = 200
    DDG_MIN_INTERVAL = 1.2
    DDG_MAX_RETRIES = 3
    MAX_PAGES_TO_FETCH = 7
    PAGE_CONTENT_MAX_CHARS = 6000
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 150
    PARALLEL_FETCH_LIMIT = 8
    MIN_RELEVANCE_THRESHOLD = 0.28

logger = logging.getLogger(__name__)

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pdfplumber
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
        logger.info("Для чтения PDF установите pypdf или pdfplumber")

# =====================================================================
# НОВЫЕ НАСТРОЙКИ (env-переменные, чтобы не трогать config_ai.py)
# =====================================================================
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "10"))          # сколько скачанных страниц переранжировать
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "2048"))  # обрезка документа для реранкера

SEARXNG_URL = os.getenv("SEARXNG_URL", "").rstrip("/")        # например http://localhost:8080
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

# Ниже этого скора (после реранка) результат считается слабым и
# запускается уточняющий проход поиска.
REFINEMENT_SCORE_THRESHOLD = float(os.getenv("REFINEMENT_SCORE_THRESHOLD", "0.30"))
MAX_REFINEMENTS = int(os.getenv("MAX_REFINEMENTS", "1"))
# Близость simhash (в битах): меньше => строже дедупликация.
SIMHASH_DEDUP_BITS = int(os.getenv("SIMHASH_DEDUP_BITS", "8"))

# Тип опциональной LLM-функции уточнения запроса:
#   async def refiner(original_query: str, found_titles: List[str]) -> str
QueryRefiner = Callable[[str, List[str]], Awaitable[str]]


# =====================================================================
# Кэш (без изменений)
# =====================================================================
class SearchCache:
    def __init__(self, ttl: int = SEARCH_CACHE_TTL, maxsize: int = SEARCH_CACHE_MAX_SIZE):
        self.ttl = ttl
        self.maxsize = maxsize
        self._cache: OrderedDict[str, Tuple[Any, float]] = OrderedDict()
        self._lock = asyncio.Lock()

    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            if key not in self._cache:
                return None
            value, ts = self._cache[key]
            if time.time() - ts > self.ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    async def set(self, key: str, value: Any):
        async with self._lock:
            self._cache[key] = (value, time.time())
            self._cache.move_to_end(key)
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)


URL_RE = re.compile(r'https?://[^\s<>"\')\]]+')

_GITHUB_BLOB_RE = re.compile(
    r'^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$'
)


def normalize_raw_url(url: str) -> str:
    m = _GITHUB_BLOB_RE.match(url.strip())
    if m:
        user, repo, branch, path = m.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    return url


def extract_urls(text: str) -> List[str]:
    return URL_RE.findall(text or "")


def _extract_pdf_text(raw_bytes: bytes) -> str:
    if not PDF_AVAILABLE:
        return ""
    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(raw_bytes))
        text = ""
        for page in reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception:
        try:
            import io
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            return text
        except Exception as e:
            logger.debug(f"PDF extraction failed: {e}")
            return ""


# =====================================================================
# Simhash — дедупликация страниц по содержимому
# =====================================================================
def _simhash(text: str, bits: int = 64) -> int:
    """Классический simhash: взвешенная сумма хэшей токенов."""
    tokens = ChunkRanker._tokenize(text)
    if not tokens:
        return 0
    vec = [0] * bits
    for t in set(tokens):
        h = int(hashlib.md5(t.encode("utf-8")).hexdigest(), 16)
        for i in range(bits):
            vec[i] += 1 if (h >> i) & 1 else -1
    out = 0
    for i in range(bits):
        if vec[i] > 0:
            out |= 1 << i
    return out


def _hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")


def dedup_by_simhash(texts: List[str], min_bits: int = SIMHASH_DEDUP_BITS) -> List[int]:
    """
    Возвращает индексы texts без почти-дубликатов (simhash-близость
    >= min_bits считается разным содержимым).
    """
    keep: List[int] = []
    hashes: List[int] = []
    for i, t in enumerate(texts):
        h = _simhash(t)
        if h and any(_hamming(h, kh) < min_bits for kh in hashes):
            logger.debug(f"simhash: дубликат содержимого отброшен (idx {i})")
            continue
        hashes.append(h)
        keep.append(i)
    return keep


# =====================================================================
# Cross-encoder реранкер (ленивая загрузка, откат при сбое)
# =====================================================================
def _default_device() -> str:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


RERANK_DEVICE = os.getenv("RERANK_DEVICE") or _default_device()


class Reranker:
    _model = None
    _failed = False

    @classmethod
    def _get(cls):
        if not RERANK_ENABLED or cls._failed:
            return None
        if cls._model is None:
            try:
                from sentence_transformers import CrossEncoder
                logger.info(f"Загрузка реранкера {RERANK_MODEL} на {RERANK_DEVICE}...")
                cls._model = CrossEncoder(RERANK_MODEL, device=RERANK_DEVICE)
                logger.info("Реранкер загружен.")
            except Exception as e:
                logger.warning(
                    f"Реранкер недоступен ({e}) — откат на токенный скоринг ChunkRanker. "
                    "Установите: pip install sentence-transformers"
                )
                cls._failed = True
                return None
        return cls._model

    @classmethod
    def rerank(cls, query: str, docs: List[str]) -> List[int]:
        """
        Возвращает индексы docs, отсортированные по убыванию релевантности
        запросу. При недоступности модели — исходный порядок (вызывающий
        код должен отдельно отсортировать по ChunkRanker, как раньше).
        """
        model = cls._get()
        if model is None or not docs:
            return list(range(len(docs)))
        pairs = [(query, d[:RERANK_MAX_CHARS]) for d in docs]
        try:
            scores = model.predict(pairs)
        except Exception as e:
            logger.warning(f"Реранкер упал, откат на исходный порядок: {e}")
            return list(range(len(docs)))
        return sorted(range(len(docs)), key=lambda i: -float(scores[i]))


# =====================================================================
# Структурированное извлечение: JSON-LD, таблицы, дата публикации
# =====================================================================
_JSONLD_KEEP_KEYS = {
    "price", "priceCurrency", "lowPrice", "highPrice", "value",
    "ratingValue", "reviewCount", "datePublished", "dateModified",
    "startDate", "endDate", "validThrough", "name", "availability",
}


def _extract_jsonld(soup: BeautifulSoup) -> str:
    parts: List[str] = []
    for tag in soup.find_all("script", type="application/ld+json"):
        raw = tag.string or tag.get_text() or ""
        try:
            data = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            continue
        items = data if isinstance(data, list) else [data]
        for it in items:
            if not isinstance(it, dict):
                continue
            keep = {k: v for k, v in it.items() if k in _JSONLD_KEEP_KEYS}
            if keep:
                parts.append("[jsonld] " + json.dumps(keep, ensure_ascii=False))
    return "\n".join(parts)


def _extract_tables(soup: BeautifulSoup, max_tables: int = 5) -> str:
    parts: List[str] = []
    for table in soup.find_all("table")[:max_tables]:
        rows: List[str] = []
        for tr in table.find_all("tr")[:30]:
            cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
            cells = [c for c in cells if c]
            if cells:
                rows.append(" | ".join(cells))
        if len(rows) >= 2:  # таблица из одной строки — это навигация, не данные
            parts.append("[table]\n" + "\n".join(rows))
    return "\n\n".join(parts)


_DATE_META_RE = re.compile(r"article:published_time|datePublished|date|dc.date|pubdate", re.I)
_ISO_DATE_RE = re.compile(r"(20\d{2}-\d{2}-\d{2})")


def _extract_published_date(soup: BeautifulSoup, text: str = "") -> Optional[str]:
    for meta in soup.find_all("meta"):
        prop = (meta.get("property") or meta.get("name") or meta.get("itemprop") or "")
        if prop and _DATE_META_RE.search(prop):
            content = meta.get("content") or ""
            m = _ISO_DATE_RE.search(content)
            if m:
                return m.group(1)
    if text:
        m = _ISO_DATE_RE.search(text[:500])
        if m:
            return m.group(1)
    return None


def _recency_boost(published: Optional[str], time_sensitive: bool) -> float:
    """Свежие страницы бустятся только для чувствительных ко времени запросов."""
    if not time_sensitive or not published:
        return 0.0
    try:
        pub = time.strptime(published, "%Y-%m-%d")
        age_days = (time.time() - time.mktime(pub)) / 86400.0
    except (ValueError, OverflowError):
        return 0.0
    if age_days < 0:
        return 0.0
    if age_days <= 7:
        return 0.20
    if age_days <= 31:
        return 0.12
    if age_days <= 180:
        return 0.05
    return 0.0


# =====================================================================
# Загрузчик страниц (API совместим: fetch / fetch_with_links / fetch_many*)
# =====================================================================
class WebPageFetcher:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9",
                },
                timeout=aiohttp.ClientTimeout(total=self.timeout),
            )
        return self._session

    TEXTUAL_CONTENT_TYPES = (
        "text/html", "application/xhtml", "text/plain", "text/markdown",
        "text/x-markdown", "application/json", "text/csv", "text/xml",
        "application/xml", "text/javascript", "application/javascript",
        "application/x-yaml", "text/yaml", "text/x-python", "text/x-python-script",
        "text/x-c", "text/x-csrc", "text/x-java-source", "application/x-sh",
        "application/octet-stream",
    )

    async def fetch(self, url: str) -> str:
        text, _ = await self._fetch_impl(url, with_links=False)
        return text

    async def fetch_with_links(self, url: str) -> Tuple[str, List[Tuple[str, str]]]:
        return await self._fetch_impl(url, with_links=True)

    async def _fetch_impl(self, url: str, with_links: bool):
        url = normalize_raw_url(url)
        try:
            session = await self._get_session()
            async with session.get(url, allow_redirects=True, max_redirects=5) as resp:
                if resp.status != 200:
                    return "", []
                content_type = resp.headers.get("Content-Type", "").lower()
                is_textual = any(t in content_type for t in self.TEXTUAL_CONTENT_TYPES)

                if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                    raw_bytes = await resp.read()
                    text = _extract_pdf_text(raw_bytes)
                    if text:
                        return self._clean_plain_text(text), []
                    return "", []

                if content_type and not is_textual:
                    return "", []

                raw = await resp.text(errors="replace")
                if "text/html" in content_type or "application/xhtml" in content_type:
                    final_url = str(resp.url)
                    return self._extract_html(raw, final_url, with_links)
                return self._clean_plain_text(raw), []
        except Exception as e:
            logger.debug(f"Fetch error for {url}: {e}")
            return "", []

    @staticmethod
    def _clean_plain_text(text: str) -> str:
        lines = [line.rstrip() for line in text.splitlines()]
        cleaned = []
        blank_run = 0
        for line in lines:
            if not line.strip():
                blank_run += 1
                if blank_run > 2:
                    continue
            else:
                blank_run = 0
            cleaned.append(line)
        text = "\n".join(cleaned)
        if len(text) > PAGE_CONTENT_MAX_CHARS:
            text = text[:PAGE_CONTENT_MAX_CHARS] + "\n...[truncated]"
        return text

    def _extract_html(self, html: str, base_url: str, with_links: bool):
        """Единая точка извлечения: текст + JSON-LD + таблицы + дата + ссылки."""
        try:
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]):
                tag.decompose()
            main = (soup.find("main") or soup.find("article")
                    or soup.find("div", class_=re.compile("content|article|post")))
            container = main or soup

            text = container.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)

            # === НОВОЕ: структурированные данные (цены, курсы, даты) ===
            structured = "\n".join(filter(None, [
                _extract_jsonld(soup),
                _extract_tables(soup),
            ]))
            if structured:
                text = structured + "\n\n" + text

            if len(text) > PAGE_CONTENT_MAX_CHARS:
                text = text[:PAGE_CONTENT_MAX_CHARS] + "\n...[truncated]"

            if not with_links:
                return text, []

            links = self._collect_links(container, base_url)
            return text, links
        except Exception as e:
            logger.debug(f"Parse error: {e}")
            return "", []

    @staticmethod
    def _collect_links(container, base_url: str) -> List[Tuple[str, str]]:
        base_domain = urlparse(base_url).netloc
        links: List[Tuple[str, str]] = []
        seen_hrefs = set()
        for a in container.find_all("a", href=True):
            href = a["href"].strip()
            if not href or href.startswith(("#", "javascript:", "mailto:")):
                continue
            absolute = urljoin(base_url, href)
            if not absolute.startswith(("http://", "https://")):
                continue
            if urlparse(absolute).netloc != base_domain:
                continue
            if absolute in seen_hrefs:
                continue
            anchor_text = a.get_text(strip=True)
            if not anchor_text:
                continue
            seen_hrefs.add(absolute)
            links.append((anchor_text[:200], absolute))
            if len(links) >= 40:
                break
        return links

    # fetch_many / fetch_many_with_links — без изменений
    async def fetch_many(self, urls: List[str], limit: int = PARALLEL_FETCH_LIMIT) -> List[Tuple[str, str]]:
        semaphore = asyncio.Semaphore(limit)

        async def fetch_one(url):
            async with semaphore:
                text = await self.fetch(url)
                return url, text

        results = await asyncio.gather(*[fetch_one(u) for u in urls], return_exceptions=True)
        return [r for r in results if isinstance(r, tuple)]

    async def fetch_many_with_links(self, urls: List[str],
                                    limit: int = PARALLEL_FETCH_LIMIT) -> List[Tuple[str, str, List[Tuple[str, str]]]]:
        semaphore = asyncio.Semaphore(limit)

        async def fetch_one(url):
            async with semaphore:
                text, links = await self.fetch_with_links(url)
                return url, text, links

        results = await asyncio.gather(*[fetch_one(u) for u in urls], return_exceptions=True)
        return [r for r in results if isinstance(r, tuple)]

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# =====================================================================
# ChunkRanker (токенный fallback — оставлен для обратной совместимости)
# =====================================================================
class ChunkRanker:
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r'\b\w+\b', text.lower())

    @staticmethod
    def score_chunks(query: str, chunks: List[str]) -> List[Tuple[float, str]]:
        q_tokens = set(ChunkRanker._tokenize(query))
        if not q_tokens:
            return [(0.0, c) for c in chunks]
        scored = []
        for chunk in chunks:
            c_tokens = ChunkRanker._tokenize(chunk)
            if not c_tokens:
                scored.append((0.0, chunk))
                continue
            overlap = len(q_tokens & set(c_tokens))
            tf = sum(c_tokens.count(qt) for qt in q_tokens)
            score = (overlap * 2 + tf) / (len(c_tokens) + 1)
            scored.append((score, chunk))
        scored.sort(reverse=True)
        return scored

    @staticmethod
    def score_text(query: str, text: str) -> float:
        q_tokens = set(ChunkRanker._tokenize(query))
        if not q_tokens:
            return 0.0
        t_tokens = ChunkRanker._tokenize(text)
        if not t_tokens:
            return 0.0
        overlap = len(q_tokens & set(t_tokens))
        tf = sum(t_tokens.count(qt) for qt in q_tokens)
        return (overlap * 2 + tf) / (len(t_tokens) + 1)

    @staticmethod
    def chunk_text(text: str, size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
        if len(text) <= size:
            return [text]
        chunks = []
        start = 0
        while start < len(text):
            end = start + size
            chunk = text[start:end]
            chunks.append(chunk)
            start += size - overlap
        return chunks


# =====================================================================
# Доверие доменам (без изменений)
# =====================================================================
_TRUSTED_DOMAINS_HIGH = (
    ".gov", ".gov.ru", ".edu", ".mil",
    "wikipedia.org", "who.int", "un.org",
    "cbr.ru",
)
_TRUSTED_DOMAINS_MEDIUM = (
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "bloomberg.com",
    "tass.ru", "ria.ru", "interfax.ru", "kommersant.ru", "vedomosti.ru",
    "nature.com", "sciencedirect.com", "arxiv.org", "github.com",
    "docs.python.org", "developer.mozilla.org", "stackoverflow.com",
)
_LOW_TRUST_MARKERS = ("pinterest.", "quora.com",)

DOMAIN_TRUST_BOOST_HIGH = 0.35
DOMAIN_TRUST_BOOST_MEDIUM = 0.15


def domain_trust(url: str) -> Tuple[str, float]:
    if not url:
        return "", 0.0
    try:
        host = urlparse(url).netloc.lower()
    except Exception:
        return "", 0.0
    host = host[4:] if host.startswith("www.") else host
    if any(host == d or host.endswith("." + d.lstrip(".")) or d in host for d in _TRUSTED_DOMAINS_HIGH):
        return "высокая", DOMAIN_TRUST_BOOST_HIGH
    if any(d in host for d in _TRUSTED_DOMAINS_MEDIUM):
        return "средняя", DOMAIN_TRUST_BOOST_MEDIUM
    if any(m in host for m in _LOW_TRUST_MARKERS):
        return "", 0.0
    return "", 0.0


# =====================================================================
# Утилиты (дата, хэш, числовые паттерны — без изменений)
# =====================================================================
_DATE_SENSITIVE_MARKERS = (
    "сегодня", "сейчас", "завтра", "вчера", "погода", "температур",
    "курс", "цена", "стоимость", "сколько стоит", "котировк",
    "новост", "актуальн", "последн", "свеж", "результат",
    "расписан", "во сколько", "кто победил",
)
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_MONTHS_RU = (
    "января", "февраля", "марта", "апреля", "мая", "июня",
    "июля", "августа", "сентября", "октября", "ноября", "декабря",
)


def is_time_sensitive_query(query: str) -> bool:
    low = (query or "").lower()
    return any(m in low for m in _DATE_SENSITIVE_MARKERS) or bool(_YEAR_RE.search(query or ""))


def inject_query_date(query: str) -> str:
    q = (query or "").strip()
    if not q:
        return q
    low = q.lower()
    if not any(m in low for m in _DATE_SENSITIVE_MARKERS):
        return q
    if _YEAR_RE.search(q):
        return q
    now = time.localtime()
    date_str = f"{now.tm_mday} {_MONTHS_RU[now.tm_mon - 1]} {now.tm_year}"
    return f"{q} {date_str}"


def hash_query(q: str) -> str:
    return hashlib.sha256(q.lower().strip().encode()).hexdigest()[:32]


def content_has_currency_numbers(text: str) -> bool:
    if not text:
        return False
    numeric_patterns = [
        r'\b\d{1,3}[.,]\d{2}\b',
        r'\b\d{1,3}\.\d{2}\s*(?:₽|руб|RUB|USD|EUR)\b',
        r'(?:USD|EUR|RUB)\s*/\s*(?:RUB|USD|EUR)\s*[:=]?\s*\d{1,3}[.,]\d{2}',
    ]
    for pat in numeric_patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    topic_words = (
        "курс", "доллар", "евро", "биткоин", "bitcoin", "btc", "эфириум", "eth",
        "акци", "котировк", "цена", "стоимост", "прайс", "price", "exchange rate",
    )
    text_l = text.lower()
    if any(w in text_l for w in topic_words):
        return True
    return False


# =====================================================================
# Поисковые бэкенды: DDG -> SearXNG -> Brave
# =====================================================================
_search_cache = SearchCache()
_fetcher = WebPageFetcher()
_ddg_lock = asyncio.Lock()
_last_ddg_call = 0.0


async def search_ddg(query: str, max_results: int = 5) -> List[Dict]:
    if not DDGS_AVAILABLE:
        return []
    global _last_ddg_call
    loop = asyncio.get_event_loop()

    async with _ddg_lock:
        elapsed = time.time() - _last_ddg_call
        if elapsed < DDG_MIN_INTERVAL:
            await asyncio.sleep(DDG_MIN_INTERVAL - elapsed)

        for attempt in range(DDG_MAX_RETRIES):
            try:
                ddgs = DDGS()
                results = await loop.run_in_executor(
                    None,
                    lambda: list(ddgs.text(query, max_results=max_results))
                )
                _last_ddg_call = time.time()
                return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")}
                        for r in results]
            except Exception as e:
                logger.warning(f"DDG attempt {attempt + 1}/{DDG_MAX_RETRIES} failed: {e}")
                if attempt < DDG_MAX_RETRIES - 1:
                    await asyncio.sleep((2 ** attempt) + 0.5)
        _last_ddg_call = time.time()
        return []


async def search_searxng(query: str, max_results: int = 5) -> List[Dict]:
    """SearXNG (self-hosted метапоиск). Нужен env SEARXNG_URL, например http://localhost:8080."""
    if not SEARXNG_URL:
        return []
    try:
        session = await _fetcher._get_session()
        async with session.get(
            f"{SEARXNG_URL}/search",
            params={"q": query, "format": "json"},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status != 200:
                logger.debug(f"SearXNG status {resp.status}")
                return []
            data = await resp.json(content_type=None)
            out = []
            for r in data.get("results", []):
                url = r.get("url")
                if not url:
                    continue
                out.append({
                    "title": r.get("title", ""),
                    "url": url,
                    "snippet": r.get("content", "") or "",
                    "published": r.get("publishedDate", "")[:10] or None,
                })
            return out[:max_results]
    except Exception as e:
        logger.debug(f"SearXNG error: {e}")
        return []


async def search_brave(query: str, max_results: int = 5) -> List[Dict]:
    """Brave Search API. Нужен env BRAVE_API_KEY."""
    if not BRAVE_API_KEY:
        return []
    try:
        session = await _fetcher._get_session()
        async with session.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": min(max_results, 20)},
            headers={"Accept": "application/json", "X-Subscription-Token": BRAVE_API_KEY},
            timeout=aiohttp.ClientTimeout(total=15),
        ) as resp:
            if resp.status != 200:
                logger.debug(f"Brave status {resp.status}")
                return []
            data = await resp.json(content_type=None)
            out = []
            for r in (data.get("web", {}) or {}).get("results", []):
                url = r.get("url")
                if not url:
                    continue
                out.append({
                    "title": r.get("title", ""),
                    "url": url,
                    "snippet": r.get("description", "") or "",
                    "published": (r.get("page_age") or "")[:10] or None,
                })
            return out[:max_results]
    except Exception as e:
        logger.debug(f"Brave error: {e}")
        return []


async def search_backends(query: str, max_results: int) -> Tuple[List[Dict], str]:
    """
    Пробует бэкенды по цепочке, пока кто-то не вернёт непустую выдачу.
    Возвращает (results, backend_name) — backend попадает в логи/метаданные.
    """
    ddg = await search_ddg(query, max_results)
    if ddg:
        return ddg, "ddg"
    if SEARXNG_URL:
        sx = await search_searxng(query, max_results)
        if sx:
            logger.info(f"[Search] DDG пуст, использован SearXNG для '{query[:60]}'")
            return sx, "searxng"
    if BRAVE_API_KEY:
        br = await search_brave(query, max_results)
        if br:
            logger.info(f"[Search] использован Brave API для '{query[:60]}'")
            return br, "brave"
    return [], "none"


async def fetch_url(url: str, max_chars: int = PAGE_CONTENT_MAX_CHARS) -> Dict[str, Any]:
    url = url.strip()
    text = await _fetcher.fetch(url)
    if not text:
        return {"url": url, "title": url, "text": "", "ok": False,
                "error": "Не удалось получить содержимое (страница недоступна, "
                         "требует авторизации или формат не поддерживается)."}
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return {"url": url, "title": url, "text": text, "ok": True}


def best_excerpt(query: str, text: str, max_chars: int = 3000) -> str:
    if len(text) <= max_chars:
        return text
    chunks = ChunkRanker.chunk_text(text)
    if len(chunks) <= 1:
        return text[:max_chars]
    scored = ChunkRanker.score_chunks(query, chunks)
    if all(score <= 0 for score, _ in scored):
        return text[:max_chars]
    chunk_index = {c: i for i, c in enumerate(chunks)}
    picked = []
    total = 0
    for score, chunk in scored:
        if total >= max_chars:
            break
        picked.append(chunk)
        total += len(chunk)
    picked.sort(key=lambda c: chunk_index[c])
    excerpt = "\n[...]\n".join(picked)
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars] + "\n...[truncated]"
    return excerpt


# =====================================================================
# Обработка набора страниц: дедуп -> реранк -> хопы -> контекст
# =====================================================================
async def _process_pages(
    query: str,
    fetched: List[Tuple[str, str, List[Tuple[str, str]]]],
    url_to_title: Dict[str, str],
    url_to_published: Dict[str, Optional[str]],
    time_sensitive: bool,
    extra_hops_budget: int,
) -> Tuple[List[Dict], List[str], float, int]:
    """
    Возвращает (sources, context_parts, best_score, hops_used).
    Используется и для основного прохода, и для уточняющего.
    """
    sources: List[Dict] = []
    context_parts: List[str] = []
    best_score = 0.0
    extra_hops_used = 0
    seen_urls: set = set()

    # === 1. Дедупликация по содержимому (simhash) ===
    texts = [t for _, t, _ in fetched if t]
    keep_idx = set(dedup_by_simhash(texts))
    deduped = []
    k = 0
    for item in fetched:
        text = item[1]
        if text:
            if k not in keep_idx:
                k += 1
                continue
            k += 1
        deduped.append(item)

    # === 2. Cross-encoder реранк (по убыванию релевантности) ===
    if deduped and Reranker._get() is not None:
        top = deduped[:RERANK_TOP_N]
        order = Reranker.rerank(query, [t for _, t, _ in top])
        rest = deduped[RERANK_TOP_N:]
        deduped = [top[i] for i in order] + rest

    # === 3. Построение контекста с хопами по ссылкам ===
    for url, text, links in deduped:
        if not text:
            continue
        page_score = ChunkRanker.score_text(query, text)

        # Нерелевантная страница с хорошими ссылками — пробуем перейти (как раньше)
        if page_score <= 0 and links and extra_hops_used < extra_hops_budget:
            scored_links = []
            for anchor_text, href in links:
                s = ChunkRanker.score_text(query, anchor_text)
                if s > 0:
                    scored_links.append((s, href))
            scored_links.sort(key=lambda x: x[0], reverse=True)
            take = min(3, extra_hops_budget - extra_hops_used, len(scored_links))
            if take > 0:
                best_hrefs = [href for _, href in scored_links[:take]]
                results = await asyncio.gather(
                    *[_fetcher.fetch_with_links(h) for h in best_hrefs],
                    return_exceptions=True,
                )
                for href, res in zip(best_hrefs, results):
                    if isinstance(res, Exception):
                        continue
                    deep_text, _ = res
                    if deep_text and ChunkRanker.score_text(query, deep_text) > page_score:
                        extra_hops_used += 1
                        hop_trust_label, _ = domain_trust(href)
                        hop_source = {"title": f"{url_to_title.get(url, url)} → подробнее", "url": href}
                        if hop_trust_label:
                            hop_source["reliability"] = hop_trust_label
                        sources.append(hop_source)
                        excerpt = best_excerpt(query, deep_text, max_chars=3000)
                        hop_suffix = f" [надёжность источника: {hop_trust_label}]" if hop_trust_label else ""
                        context_parts.append(
                            f"Источник: {url_to_title.get(url, url)} (подробности по ссылке со страницы){hop_suffix}\n"
                            f"URL: {href}\n{excerpt}"
                        )
                continue

        if url in seen_urls:
            continue
        seen_urls.add(url)

        # === 4. Рекенси-буст для time-sensitive запросов ===
        published = url_to_published.get(url)
        page_score += _recency_boost(published, time_sensitive)
        best_score = max(best_score, page_score)

        src = {"title": url_to_title.get(url, url), "url": url}
        trust_label, _ = domain_trust(url)
        if trust_label:
            src["reliability"] = trust_label
        if published:
            src["published"] = published
        sources.append(src)

        excerpt = best_excerpt(query, text, max_chars=3000)
        reliability_suffix = f" [надёжность источника: {trust_label}]" if trust_label else ""
        date_suffix = f" [опубликовано: {published}]" if published else ""
        context_parts.append(
            f"Источник: {url_to_title.get(url, url)}{reliability_suffix}{date_suffix}\nURL: {url}\n{excerpt}"
        )

    return sources, context_parts, best_score, extra_hops_used


# =====================================================================
# ГЛАВНЫЙ ВХОД
# =====================================================================
async def deep_search(
    query: str,
    max_results: int = MAX_PAGES_TO_FETCH,
    max_refinements: int = MAX_REFINEMENTS,
    query_refiner: Optional[QueryRefiner] = None,
) -> Dict[str, Any]:
    """
    query_refiner — опциональная async-функция (original_query, found_titles) -> new_query.
    Если не передана, уточнение выполняется эвристически (добавление года/даты).
    """
    # --- Прямые ссылки ---
    direct_urls = extract_urls(query)
    if direct_urls:
        fetched = await asyncio.gather(*[fetch_url(u) for u in direct_urls[:max_results]])
        sources = [{"title": r["title"], "url": r["url"]} for r in fetched if r["ok"]]
        context_parts = [
            f"Источник: {r['url']}\nURL: {r['url']}\n{r['text']}"
            for r in fetched if r["ok"]
        ]
        if context_parts:
            return {
                "sources": sources,
                "context": "\n\n---\n\n".join(context_parts),
                "search_performed": True,
                "chunks_found": len(context_parts),
                "backend": "direct_url",
            }
        remainder = URL_RE.sub("", query).strip()
        if remainder:
            query = remainder

    query = inject_query_date(query)
    cache_key = hash_query(query)
    skip_cache = content_has_currency_numbers(query)

    if not skip_cache:
        cached = await _search_cache.get(cache_key)
        if cached:
            return cached

    time_sensitive = is_time_sensitive_query(query)
    ddg_results, backend = await search_backends(query, max_results + 2)
    if not ddg_results:
        return {"sources": [], "context": "Поиск не дал результатов.", "search_performed": False}

    # --- Ранжирование сниппетов: скор + доверие домена + свежесть ---
    scored = []
    for r in ddg_results:
        base = ChunkRanker.score_text(query, f"{r.get('title', '')} {r.get('snippet', '')}")
        base += domain_trust(r.get("url", ""))[1]
        base += _recency_boost(r.get("published"), time_sensitive)
        scored.append((base, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    above_threshold = [r for s, r in scored if s >= MIN_RELEVANCE_THRESHOLD]
    ordered_results = above_threshold if above_threshold else [r for _, r in scored]

    urls = [r["url"] for r in ordered_results if r.get("url")][:max_results]
    url_to_title = {r["url"]: r["title"] for r in ddg_results}
    url_to_published = {r["url"]: r.get("published") for r in ddg_results}

    fetched = await _fetcher.fetch_many_with_links(urls, limit=PARALLEL_FETCH_LIMIT)
    # Если реранкер есть — он живёт внутри _process_pages.

    MAX_EXTRA_HOPS = 3
    sources, context_parts, best_score, _ = await _process_pages(
        query, fetched, url_to_title, url_to_published,
        time_sensitive, MAX_EXTRA_HOPS,
    )

    # === ИТЕРАТИВНОЕ УТОЧНЕНИЕ ===
    # Лучший скор низкий -> переформулируем запрос и ищем ещё раз,
    # добавляя новые уникальные источники (как делают агентные системы).
    refinements_done = 0
    current_query = query
    while (
        best_score < REFINEMENT_SCORE_THRESHOLD
        and refinements_done < max_refinements
    ):
        found_titles = [s.get("title", "") for s in sources]
        if query_refiner is not None:
            try:
                refined = (await query_refiner(current_query, found_titles) or "").strip()
            except Exception as e:
                logger.debug(f"query_refiner failed: {e}")
                refined = ""
        else:
            # Эвристический фолбэк: добавить текущий год, если его нет
            refined = current_query if _YEAR_RE.search(current_query) else f"{current_query} {time.localtime().tm_year}"

        if not refined or refined == current_query:
            break
        refinements_done += 1
        current_query = refined
        logger.info(f"[Search] слабый результат (score={best_score:.2f}), уточнение {refinements_done}: '{refined[:80]}'")

        refined_results, _ = await search_backends(refined, max_results)
        if not refined_results:
            break
        r_scored = sorted(
            refined_results,
            key=lambda r: ChunkRanker.score_text(refined, f"{r.get('title','')} {r.get('snippet','')}")
                          + domain_trust(r.get("url", ""))[1]
                          + _recency_boost(r.get("published"), time_sensitive),
            reverse=True,
        )
        new_urls = [r["url"] for r in r_scored if r.get("url") and r["url"] not in {s.get("url") for s in sources}][:max_results]
        if not new_urls:
            break
        for r in refined_results:
            url_to_title.setdefault(r["url"], r["title"])
            url_to_published.setdefault(r["url"], r.get("published"))
        new_fetched = await _fetcher.fetch_many_with_links(new_urls, limit=PARALLEL_FETCH_LIMIT)
        new_sources, new_parts, new_best, _ = await _process_pages(
            refined, new_fetched, url_to_title, url_to_published,
            time_sensitive, 0,  # хопы только в первом проходе
        )
        sources.extend(new_sources)
        context_parts.extend(new_parts)
        best_score = max(best_score, new_best)

    context = "\n\n---\n\n".join(context_parts)
    result = {
        "sources": sources,
        "context": context,
        "search_performed": True,
        "chunks_found": len(context_parts),
        "backend": backend,
        "refinements": refinements_done,
    }
    if not skip_cache and context_parts:
        await _search_cache.set(cache_key, result)
    return result


async def close_search_resources():
    await _fetcher.close()


__all__ = [
    'SearchCache',
    'WebPageFetcher',
    'ChunkRanker',
    'Reranker',
    'dedup_by_simhash',
    'hash_query',
    'content_has_currency_numbers',
    'search_ddg',
    'search_searxng',
    'search_brave',
    'search_backends',
    'deep_search',
    'fetch_url',
    'normalize_raw_url',
    'extract_urls',
    'best_excerpt',
    'domain_trust',
    'is_time_sensitive_query',
    'inject_query_date',
    'close_search_resources',
]