"""
web_search.py v3 — улучшенный веб-поиск для BlockcoinWitres.

Обратная совместимость: публичный API v2 полностью сохранён
(deep_search, fetch_url, search_ddg, best_excerpt, domain_trust, и т.д.).

Что нового в v3 (см. ИЗМЕНЕНИЯ ниже):

  1. BM25-скоринг вместо голого токенного overlap.
     ChunkRanker.score_text раньше делал (overlap*2 + tf) / (len+1) —
     длинные страницы систематически занижались, короткие сниппеты
     завышались, и ранжирование сниппетов/страниц/чанков было
     несопоставимо между собой. Теперь везде, где документы сравниваются
     между собой, используется классический BM25 (k1=1.5, b=0.75) с
     корпусной IDF. score_text оставлен для совместимости, но тоже
     улучшен (BM25-стиль TF-сатурация + бонус за точную фразу).

  2. Порядок бэкендов настраивается (env SEARCH_BACKENDS_ORDER,
     по умолчанию "ddg,searxng,brave"). Если поднят self-hosted SearXNG,
     имеет смысл поставить его первым — DDG часто отдаёт капчу, и логично
     сначала пробовать собственный стабильный бэкенд.

  3. Сниппеты как fallback. Раньше если страница не скачалась (таймаут,
     JS-рендер, 403) — источник терялся полностью, хотя поисковик уже
     дал релевантный сниппет. Теперь при недоступности страницы контекст
     строится из сниппета (с явной пометкой "[сниппет поисковика]").

  4. Общий бюджет времени (env SEARCH_TIME_BUDGET, по умолчанию 30с).
     deep_search раньше мог растягиваться непредсказуемо: DDG-ретраи с
     экспоненциальными паузами + до 7 страниц × 15с + хопы + refinement.
     Теперь между стадиями проверяется дедлайн; при нехватке времени
     возвращается лучший из накопленного (вплоть до контекста только из
     сниппетов), а не пустой результат.

  5. Частичные результаты при дедлайне. fetch_many* принимает deadline
     и возвращает успевшие загрузиться страницы вместо потери всей
     пачки из-за отмены gather.

  6. Хук параллельного расширения запроса — query_expander
     (async def(query) -> List[str], например через LLM: синонимы,
     английский вариант, декомпозиция). Несколько запросов уходят в
     бэкенды параллельно, результаты сливаются с приоритетом базового
     запроса и дедупликацией по URL. Без expander поведение прежнее.

  7. Чистка URL (utm_*/fbclid/gclid/фрагменты/www) — меньше мусорных
     дублей в кэше, seen-сетах и источниках.

  8. Ссылки на папки GitHub (/tree/...) теперь читаются через GitHub
     Contents API и возвращаются аккуратным листингом (blob-ссылки
     по-прежнему конвертируются в raw).

  9. Заголовок страницы (<title>) используется как название источника,
     если поисковик не дал своего.

 10. simhash-дедупликация переписана без хрупкого счётчика индексов
     (было: ручной инкремент k, легко рассинхронизировать при правках).
"""

import asyncio
import hashlib
import json
import logging
import math
import os
import re
import time
from collections import Counter, OrderedDict
from typing import Any, Awaitable, Callable, Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse

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
    import pypdf  # noqa: F401
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pdfplumber  # noqa: F401
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
        logger.info("Для чтения PDF установите pypdf или pdfplumber")

# =====================================================================
# НАСТРОЙКИ (env-переменные, чтобы не трогать config_ai.py)
# =====================================================================
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "true").lower() in ("1", "true", "yes")
RERANK_MODEL = os.getenv("RERANK_MODEL", "BAAI/bge-reranker-v2-m3")
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "10"))
RERANK_MAX_CHARS = int(os.getenv("RERANK_MAX_CHARS", "2048"))

SEARXNG_URL = os.getenv("SEARXNG_URL", "").rstrip("/")
BRAVE_API_KEY = os.getenv("BRAVE_API_KEY", "")

REFINEMENT_SCORE_THRESHOLD = float(os.getenv("REFINEMENT_SCORE_THRESHOLD", "0.30"))
MAX_REFINEMENTS = int(os.getenv("MAX_REFINEMENTS", "1"))
SIMHASH_DEDUP_BITS = int(os.getenv("SIMHASH_DEDUP_BITS", "8"))

# v3: порядок бэкендов. Если SearXNG поднят — разумно поставить его первым:
# SEARCH_BACKENDS_ORDER="searxng,ddg,brave"
SEARCH_BACKENDS_ORDER = [
    s.strip() for s in os.getenv("SEARCH_BACKENDS_ORDER", "ddg,searxng,brave").split(",")
    if s.strip()
]
# v3: общий бюджет времени на deep_search (секунды)
SEARCH_TIME_BUDGET = float(os.getenv("SEARCH_TIME_BUDGET", "30"))
# v3: максимум запросов при расширении (1 базовый + N дополнительных)
MAX_EXPANDED_QUERIES = int(os.getenv("MAX_EXPANDED_QUERIES", "3"))

QueryRefiner = Callable[[str, List[str]], Awaitable[str]]
# v3: расширитель запроса — async def(query: str) -> List[str]
QueryExpander = Callable[[str], Awaitable[List[str]]]


def _loop_time() -> float:
    return asyncio.get_event_loop().time()


# =====================================================================
# Кэш
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
# v3: ссылки на папки GitHub читаем через Contents API
_GITHUB_TREE_RE = re.compile(
    r'^https?://github\.com/([^/]+)/([^/]+)/tree/([^/]+)/?(.*)$'
)

# v3: трекинговые параметры, портящие дедупликацию и кэш
_TRACKING_PARAMS = {
    "fbclid", "gclid", "yclid", "dclid", "gclsrc", "mc_cid", "mc_eid",
    "spm", "msclkid", "ref", "ref_src",
}
_TRACKING_PREFIXES = ("utm_",)


def clean_url(url: str) -> str:
    """
    v3: нормализует URL — убирает трекинговые параметры, фрагмент, www,
    хвостовой слэш. Канонический вид URL используется в seen-сетах, кэше
    и source-списках, чтобы одна страница не мельтешила в выдаче как
    несколько разных источников.
    """
    try:
        p = urlparse(url.strip())
    except Exception:
        return url
    host = (p.netloc or "").lower()
    if host.startswith("www."):
        host = host[4:]
    qs = [
        (k, v) for k, v in parse_qsl(p.query, keep_blank_values=True)
        if k.lower() not in _TRACKING_PARAMS
        and not any(k.lower().startswith(pref) for pref in _TRACKING_PREFIXES)
    ]
    path = p.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    return urlunparse((p.scheme.lower() or "https", host, path, "", urlencode(qs), ""))


def normalize_raw_url(url: str) -> str:
    m = _GITHUB_BLOB_RE.match(url.strip())
    if m:
        user, repo, branch, path = m.groups()
        url = f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    return clean_url(url)


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
# BM25 (v3) — корпусный скоринг вместо голого overlap
# =====================================================================
def _tokenize(text: str) -> List[str]:
    return re.findall(r'\b\w+\b', (text or "").lower())


class _BM25:
    """Классический BM25 (Robertson). Потокобезопасен после fit()."""

    def __init__(self, docs_tokens: List[List[str]], k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.dl = [len(d) for d in docs_tokens]
        self.avgdl = (sum(self.dl) / len(self.dl)) if docs_tokens else 0.0
        df: Counter = Counter()
        for d in docs_tokens:
            for t in set(d):
                df[t] += 1
        n = len(docs_tokens)
        self.idf = {
            t: math.log(1.0 + (n - freq + 0.5) / (freq + 0.5))
            for t, freq in df.items()
        }

    def score(self, query_tokens: List[str], doc_idx: int) -> float:
        doc = self._docs[doc_idx]
        if not doc:
            return 0.0
        dl = self.dl[doc_idx]
        score = 0.0
        freqs = Counter(doc)
        for qt in set(query_tokens):
            idf = self.idf.get(qt)
            if not idf or idf <= 0:
                continue
            tf = freqs.get(qt, 0)
            if tf == 0:
                continue
            denom = tf + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1.0))
            score += idf * (tf * (self.k1 + 1)) / denom
        return score

    # docs запоминаем отдельно, чтобы score() не требовал передачи токенов
    _docs: List[List[str]] = []

    @classmethod
    def fit(cls, docs_tokens: List[List[str]], **kwargs) -> "_BM25":
        bm = cls(docs_tokens, **kwargs)
        bm._docs = docs_tokens  # instance attr shadows class attr
        return bm


def bm25_scores(query: str, docs: List[str]) -> List[float]:
    """v3: BM25-скоры документов относительно друг друга (с IDF по корпусу)."""
    if not docs:
        return []
    query_tokens = _tokenize(query)
    docs_tokens = [_tokenize(d) for d in docs]
    bm = _BM25.fit(docs_tokens)
    return [bm.score(query_tokens, i) for i in range(len(docs))]


# =====================================================================
# Simhash — дедупликация страниц по содержимому (v3: без счётчика индексов)
# =====================================================================
def _simhash(text: str, bits: int = 64) -> int:
    tokens = _tokenize(text)
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
    """Возвращает индексы texts без почти-дубликатов (API v2 сохранён)."""
    keep: List[int] = []
    hashes: List[int] = []
    for i, t in enumerate(texts):
        h = _simhash(t)
        if h and any(_hamming(h, kh) < min_bits for kh in hashes):
            logger.debug(f"simhash: дубликат содержимого отброшен (idx {i})")
            continue
        if h:
            hashes.append(h)
        keep.append(i)
    return keep


def dedup_pages(
    fetched: List[Tuple[str, str, List[Tuple[str, str]]]],
    min_bits: int = SIMHASH_DEDUP_BITS,
) -> List[Tuple[str, str, List[Tuple[str, str]]]]:
    """v3: та же дедупликация, но сразу над списком (url, text, links)."""
    keep: List[Tuple[str, str, List[Tuple[str, str]]]] = []
    hashes: List[int] = []
    for item in fetched:
        text = item[1]
        if not text:
            keep.append(item)
            continue
        h = _simhash(text)
        if h and any(_hamming(h, kh) < min_bits for kh in hashes):
            continue
        if h:
            hashes.append(h)
        keep.append(item)
    return keep


# =====================================================================
# Cross-encoder реранкер (без изменений)
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
                    f"Реранкер недоступен ({e}) — откат на BM25-скоринг. "
                    "Установите: pip install sentence-transformers"
                )
                cls._failed = True
                return None
        return cls._model

    @classmethod
    def rerank(cls, query: str, docs: List[str]) -> List[int]:
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
        if len(rows) >= 2:
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
# Загрузчик страниц
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
        text, _, _ = await self._fetch_impl(url, with_links=False)
        return text

    async def fetch_with_links(self, url: str) -> Tuple[str, List[Tuple[str, str]]]:
        text, links, _ = await self._fetch_impl(url, with_links=True)
        return text, links

    async def _fetch_github_tree(
        self, session: aiohttp.ClientSession, owner: str, repo: str, branch: str, path: str
    ) -> Optional[Tuple[str, List[Tuple[str, str]], str]]:
        """v3: листинг папки GitHub через Contents API (публичные репозитории)."""
        api_url = f"https://api.github.com/repos/{owner}/{repo}/contents/{path.strip('/')}?ref={branch}"
        try:
            async with session.get(
                api_url,
                headers={"Accept": "application/vnd.github+json"},
                timeout=aiohttp.ClientTimeout(total=15),
            ) as resp:
                if resp.status != 200:
                    return None
                data = await resp.json(content_type=None)
                if isinstance(data, dict):
                    data = [data]
                shown = path.strip("/") or "/"
                lines = [
                    f"[github] Содержимое папки {shown} "
                    f"(репозиторий {owner}/{repo}, ветка {branch}):"
                ]
                for it in data:
                    if it.get("type") == "dir":
                        lines.append(f"  📁 {it.get('name', '?')}/")
                    else:
                        lines.append(f"  📄 {it.get('name', '?')} ({it.get('size', 0)} байт)")
                lines.append(
                    "\nДля чтения файла передай ссылку на файл (blob) или используй "
                    "инструмент fetch_github_file."
                )
                return "\n".join(lines), [], api_url
        except Exception as e:
            logger.debug(f"GitHub tree fetch failed: {e}")
            return None

    async def _fetch_impl(self, url: str, with_links: bool) -> Tuple[str, List[Tuple[str, str]], str]:
        """Возвращает (text, links, title). title может быть пустым."""
        url = normalize_raw_url(url)

        # v3: ссылка на папку GitHub — отдаём листинг через API
        tree = _GITHUB_TREE_RE.match(url)
        if tree:
            session = await self._get_session()
            owner, repo, branch, path = tree.groups()
            result = await self._fetch_github_tree(session, owner, repo, branch, path)
            if result is not None:
                return result

        try:
            session = await self._get_session()
            async with session.get(url, allow_redirects=True, max_redirects=5) as resp:
                if resp.status != 200:
                    return "", [], ""

                content_type = resp.headers.get("Content-Type", "").lower()
                is_textual = any(t in content_type for t in self.TEXTUAL_CONTENT_TYPES)

                if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                    raw_bytes = await resp.read()
                    text = _extract_pdf_text(raw_bytes)
                    if text:
                        return self._clean_plain_text(text), [], ""
                    return "", [], ""

                if content_type and not is_textual:
                    return "", [], ""

                raw = await resp.text(errors="replace")
                if "text/html" in content_type or "application/xhtml" in content_type:
                    final_url = str(resp.url)
                    return self._extract_html(raw, final_url, with_links)
                return self._clean_plain_text(raw), [], ""
        except Exception as e:
            logger.debug(f"Fetch error for {url}: {e}")
            return "", [], ""

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
        """Единая точка извлечения: текст + JSON-LD + таблицы + дата + ссылки + <title>."""
        try:
            soup = BeautifulSoup(html, "lxml")
            title = ""
            if soup.title and soup.title.string:
                title = soup.title.string.strip()[:200]
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]):
                tag.decompose()
            main = (soup.find("main") or soup.find("article")
                    or soup.find("div", class_=re.compile("content|article|post")))
            container = main or soup

            text = container.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)

            structured = "\n".join(filter(None, [
                _extract_jsonld(soup),
                _extract_tables(soup),
            ]))
            if structured:
                text = structured + "\n\n" + text

            if len(text) > PAGE_CONTENT_MAX_CHARS:
                text = text[:PAGE_CONTENT_MAX_CHARS] + "\n...[truncated]"

            links = self._collect_links(container, base_url) if with_links else []
            return text, links, title
        except Exception as e:
            logger.debug(f"Parse error: {e}")
            return "", [], ""

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

    async def fetch_many(self, urls: List[str], limit: int = PARALLEL_FETCH_LIMIT,
                         deadline: Optional[float] = None) -> List[Tuple[str, str]]:
        semaphore = asyncio.Semaphore(limit)

        async def fetch_one(url):
            async with semaphore:
                try:
                    if deadline is not None:
                        remaining = deadline - _loop_time()
                        if remaining <= 0:
                            return url, ""
                        text = await asyncio.wait_for(
                            self.fetch(url), timeout=min(self.timeout + 5, remaining)
                        )
                    else:
                        text = await self.fetch(url)
                    return url, text
                except Exception:
                    return url, ""

        results = await asyncio.gather(*[fetch_one(u) for u in urls])
        return [r for r in results if isinstance(r, tuple)]

    async def fetch_many_with_links(
        self, urls: List[str],
        limit: int = PARALLEL_FETCH_LIMIT,
        deadline: Optional[float] = None,
    ) -> List[Tuple[str, str, List[Tuple[str, str]]]]:
        """v3: deadline — общий дедлайн (loop.time()); при его наступлении
        возвращаются успевшие загрузиться страницы, а не пустой список."""
        semaphore = asyncio.Semaphore(limit)

        async def fetch_one(url):
            async with semaphore:
                try:
                    if deadline is not None:
                        remaining = deadline - _loop_time()
                        if remaining <= 0:
                            return url, "", []
                        text, links, _ = await asyncio.wait_for(
                            self._fetch_impl(url, with_links=True),
                            timeout=min(self.timeout + 5, remaining),
                        )
                    else:
                        text, links, _ = await self._fetch_impl(url, with_links=True)
                    return url, text, links
                except Exception:
                    return url, "", []

        results = await asyncio.gather(*[fetch_one(u) for u in urls])
        return [r for r in results if isinstance(r, tuple)]

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# =====================================================================
# ChunkRanker (API v2 сохранён; скоринг улучшен)
# =====================================================================
class ChunkRanker:
    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return _tokenize(text)

    @staticmethod
    def score_chunks(query: str, chunks: List[str]) -> List[Tuple[float, str]]:
        """v3: BM25 по корпусу чанков вместо попарного overlap."""
        if not chunks:
            return []
        scores = bm25_scores(query, chunks)
        scored = list(zip(scores, chunks))
        scored.sort(key=lambda x: -x[0])
        return scored

    @staticmethod
    def score_text(query: str, text: str) -> float:
        """Скоринг одиночного текста (BM25-стиль TF + бонус за фразу).

        Используется там, где документ не сравнивается с корпусом
        (например, anchor-тексты ссылок). Нормализация по длине мягкая
        (sqrt), чтобы не занижать длинные релевантные тексты так сильно,
        как это делало старое деление на len+1.
        """
        q_tokens = set(_tokenize(query))
        if not q_tokens:
            return 0.0
        t_tokens = _tokenize(text)
        if not t_tokens:
            return 0.0
        freqs = Counter(t_tokens)
        tf_sat = sum(math.log1p(freqs.get(qt, 0)) for qt in q_tokens)
        overlap = len(q_tokens & set(t_tokens)) / len(q_tokens)
        phrase = 1.0 if (query or "").strip().lower() in (text or "").lower() else 0.0
        score = (overlap * 2.0 + tf_sat + phrase * 3.0) / (math.sqrt(len(t_tokens)) + 1.0)
        return score

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
# Поисковые бэкенды: DDG / SearXNG / Brave
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
                    "published": (r.get("publishedDate") or "")[:10] or None,
                })
            return out[:max_results]
    except Exception as e:
        logger.debug(f"SearXNG error: {e}")
        return []


async def search_brave(query: str, max_results: int = 5) -> List[Dict]:
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


_BACKENDS = {
    "ddg": search_ddg,
    "searxng": search_searxng,
    "brave": search_brave,
}


async def search_backends(query: str, max_results: int) -> Tuple[List[Dict], str]:
    """
    v3: порядок бэкендов берётся из SEARCH_BACKENDS_ORDER (env).
    Первый непустой результат побеждает; исключения бэкенда не роняют
    цепочку. Если бэкенд не сконфигурирован — его функция сама вернёт [].
    """
    for name in SEARCH_BACKENDS_ORDER:
        fn = _BACKENDS.get(name)
        if fn is None:
            continue
        try:
            results = await fn(query, max_results)
        except Exception as e:
            logger.warning(f"Backend '{name}' failed for '{query[:60]}': {e}")
            continue
        if results:
            return results, name
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
    """v3: выбор чанков — через BM25 по корпусу чанков (раньше попарный
    overlap, что давало нестабильный порядок на длинных текстах)."""
    if len(text) <= max_chars:
        return text
    chunks = ChunkRanker.chunk_text(text)
    if len(chunks) <= 1:
        return text[:max_chars]
    scores = bm25_scores(query, chunks)
    if all(s <= 0 for s in scores):
        return text[:max_chars]
    ranked = sorted(zip(scores, range(len(chunks))), key=lambda x: -x[0])
    picked_idx = set()
    total = 0
    for score, idx in ranked:
        if total >= max_chars:
            break
        if score <= 0:
            break
        picked_idx.add(idx)
        total += len(chunks[idx])
    if not picked_idx:
        return text[:max_chars]
    picked = [chunks[i] for i in sorted(picked_idx)]
    excerpt = "\n[...]\n".join(picked)
    if len(excerpt) > max_chars:
        excerpt = excerpt[:max_chars] + "\n...[truncated]"
    return excerpt


# =====================================================================
# Обработка набора страниц: дедуп -> BM25 -> реранк -> хопы -> контекст
# =====================================================================
async def _process_pages(
    query: str,
    fetched: List[Tuple[str, str, List[Tuple[str, str]]]],
    url_to_title: Dict[str, str],
    url_to_snippet: Dict[str, str],
    url_to_published: Dict[str, Optional[str]],
    time_sensitive: bool,
    extra_hops_budget: int,
    deadline: Optional[float] = None,
) -> Tuple[List[Dict], List[str], float, int]:
    """
    Возвращает (sources, context_parts, best_score, hops_used).
    v3:
      - BM25-скоры по корпусу страниц (сопоставимо между страницами);
      - fallback на сниппет поисковика, если страница не прочитана;
      - <title> страницы как название источника при отсутствии своего;
      - дедлайн: хопы по ссылкам прекращаются, когда время вышло.
    """
    sources: List[Dict] = []
    context_parts: List[str] = []
    best_score = 0.0
    extra_hops_used = 0
    seen_urls: set = set()

    deduped = dedup_pages(fetched)
    if not deduped:
        return sources, context_parts, best_score, extra_hops_used

    texts = [t for _, t, _ in deduped]
    bm_scores = bm25_scores(query, texts)

    items = list(zip(deduped, bm_scores))

    # Cross-encoder реранк топ-N (по убыванию релевантности)
    if Reranker._get() is not None and len(items) > 1:
        top_n = min(RERANK_TOP_N, len(items))
        order = Reranker.rerank(query, texts[:top_n])
        items = [items[i] for i in order] + items[top_n:]

    for (url, text, links), base_score in items:
        # --- v3: сниппет как fallback, если страница не прочитана ---
        if not text:
            snippet = (url_to_snippet.get(url) or "").strip()
            if snippet and url not in seen_urls:
                seen_urls.add(url)
                src = {"title": url_to_title.get(url, url), "url": url}
                trust_label, _ = domain_trust(url)
                if trust_label:
                    src["reliability"] = trust_label
                sources.append(src)
                context_parts.append(
                    f"Источник: {url_to_title.get(url, url)}\nURL: {url}\n"
                    f"[сниппет поисковика — страница не прочитана]\n{snippet[:1200]}"
                )
                best_score = max(best_score, ChunkRanker.score_text(query, snippet))
            continue

        page_score = base_score

        # Нерелевантная страница с хорошими ссылками — пробуем перейти
        if page_score <= 0 and links and extra_hops_used < extra_hops_budget:
            if deadline is not None and _loop_time() > deadline:
                pass  # время вышло — хопы не делаем
            else:
                scored_links = []
                for anchor_text, href in links:
                    s = ChunkRanker.score_text(query, anchor_text)
                    if s > 0:
                        scored_links.append((s, href))
                scored_links.sort(key=lambda x: x[0], reverse=True)
                take = min(3, extra_hops_budget - extra_hops_used, len(scored_links))
                if take > 0:
                    hop_deadline = (deadline - 3.0) if deadline is not None else None
                    results = await asyncio.gather(
                        *[_fetcher.fetch_with_links(h) for h in [h for _, h in scored_links[:take]]],
                        return_exceptions=True,
                    )
                    for href, res in zip([h for _, h in scored_links[:take]], results):
                        if hop_deadline is not None and _loop_time() > hop_deadline:
                            break
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
    query_expander: Optional[QueryExpander] = None,
    time_budget: float = SEARCH_TIME_BUDGET,
) -> Dict[str, Any]:
    """
    v3:
      - time_budget — общий дедлайн на всю операцию (loop-time, секунды).
        При его исчерпании возвращается лучший накопленный контекст
        (вплоть до сниппетов), а не пустой результат;
      - query_expander — опциональный async hook(query) -> List[str]:
        дополнительные формулировки, ищутся параллельно с базовым запросом
        и сливаются с приоритетом базового (например, LLM-варианты:
        синонимы, английская формулировка, декомпозиция);
      - query_refiner — прежний хук итеративного уточнения (без изменений).
    """
    deadline = _loop_time() + max(5.0, float(time_budget))

    # --- Прямые ссылки ---
    direct_urls = [clean_url(u) for u in extract_urls(query)]
    if direct_urls:
        fetch_deadline = min(deadline, _loop_time() + 20.0)
        fetched = await _fetcher.fetch_many(direct_urls[:max_results], deadline=fetch_deadline)
        sources = []
        context_parts = []
        for url, text in fetched:
            if not text:
                continue
            sources.append({"title": url, "url": url})
            context_parts.append(f"Источник: {url}\nURL: {url}\n{text}")
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

    # --- v3: расширение запроса (параллельные формулировки) ---
    queries = [query]
    if query_expander is not None:
        try:
            extra = await asyncio.wait_for(query_expander(query), timeout=8.0)
            if extra:
                for q in extra:
                    q = (q or "").strip()
                    if q and q.lower() != query.lower() and q not in queries:
                        queries.append(q)
                    if len(queries) >= MAX_EXPANDED_QUERIES:
                        break
        except Exception as e:
            logger.debug(f"query_expander failed: {e}")

    # --- Поиск по всем формулировкам параллельно ---
    per_query = max(3, max_results // len(queries) + 1)
    search_results = await asyncio.gather(
        *[search_backends(q, per_query + 2) for q in queries],
        return_exceptions=True,
    )

    merged_results: List[Dict] = []
    seen: set = set()
    backend_used = "none"
    for q, res in zip(queries, search_results):
        if isinstance(res, Exception):
            logger.warning(f"deep_search: запрос '{q[:60]}' упал: {res}")
            continue
        items, backend = res
        if items and backend_used == "none":
            backend_used = backend
        for r in items:
            url = clean_url(r.get("url", ""))
            if not url or url in seen:
                continue
            seen.add(url)
            r["url"] = url
            merged_results.append(r)

    if not merged_results:
        return {"sources": [], "context": "Поиск не дал результатов.", "search_performed": False}

    # --- Ранжирование сниппетов: BM25 + доверие домена + свежесть ---
    snippet_texts = [f"{r.get('title', '')} {r.get('snippet', '')}" for r in merged_results]
    bm = bm25_scores(query, snippet_texts)
    scored = []
    for r, base in zip(merged_results, bm):
        base += domain_trust(r.get("url", ""))[1]
        base += _recency_boost(r.get("published"), time_sensitive)
        scored.append((base, r))
    scored.sort(key=lambda x: x[0], reverse=True)
    above_threshold = [r for s, r in scored if s >= MIN_RELEVANCE_THRESHOLD]
    ordered_results = above_threshold if above_threshold else [r for _, r in scored]

    urls = [r["url"] for r in ordered_results if r.get("url")][:max_results]
    url_to_title = {r["url"]: r["title"] for r in merged_results}
    url_to_snippet = {r["url"]: (r.get("snippet") or "") for r in merged_results}
    url_to_published = {r["url"]: r.get("published") for r in merged_results}

    # --- Чтение страниц с учётом оставшегося бюджета ---
    remaining = deadline - _loop_time()
    MAX_EXTRA_HOPS = 3
    if remaining <= 2.0:
        # Времени на чтение страниц нет — контекст из сниппетов
        sources, context_parts, best_score, _ = await _process_pages(
            query, [(u, "", []) for u in urls], url_to_title, url_to_snippet,
            url_to_published, time_sensitive, 0, deadline=deadline,
        )
    else:
        fetch_deadline = deadline - 2.0  # резерв на хопы/сборку
        fetched = await _fetcher.fetch_many_with_links(
            urls, limit=PARALLEL_FETCH_LIMIT, deadline=fetch_deadline
        )
        sources, context_parts, best_score, _ = await _process_pages(
            query, fetched, url_to_title, url_to_snippet, url_to_published,
            time_sensitive, MAX_EXTRA_HOPS, deadline=fetch_deadline,
        )

    # === ИТЕРАТИВНОЕ УТОЧНЕНИЕ ===
    refinements_done = 0
    current_query = query
    while (
        best_score < REFINEMENT_SCORE_THRESHOLD
        and refinements_done < max_refinements
        and (deadline - _loop_time()) > 8.0  # v3: только если есть время на полный круг
    ):
        found_titles = [s.get("title", "") for s in sources]
        if query_refiner is not None:
            try:
                refined = (await query_refiner(current_query, found_titles) or "").strip()
            except Exception as e:
                logger.debug(f"query_refiner failed: {e}")
                refined = ""
        else:
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
        new_urls = [clean_url(r["url"]) for r in r_scored
                    if r.get("url") and clean_url(r["url"]) not in {s.get("url") for s in sources}][:max_results]
        if not new_urls:
            break
        for r in refined_results:
            u = clean_url(r.get("url", ""))
            if not u:
                continue
            url_to_title.setdefault(u, r["title"])
            url_to_snippet.setdefault(u, r.get("snippet") or "")
            url_to_published.setdefault(u, r.get("published"))
        new_fetched = await _fetcher.fetch_many_with_links(
            new_urls, limit=PARALLEL_FETCH_LIMIT, deadline=deadline - 1.0
        )
        new_sources, new_parts, new_best, _ = await _process_pages(
            refined, new_fetched, url_to_title, url_to_snippet, url_to_published,
            time_sensitive, 0, deadline=deadline - 1.0,
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
        "backend": backend_used if len(queries) == 1 else f"{backend_used}+expand",
        "refinements": refinements_done,
        "expanded_queries": queries[1:] if len(queries) > 1 else [],
        "elapsed_seconds": round(_loop_time() - (deadline - max(5.0, float(time_budget))), 2),
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
    'bm25_scores',
    'clean_url',
    'dedup_by_simhash',
    'dedup_pages',
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