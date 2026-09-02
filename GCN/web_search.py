import asyncio
import hashlib
import logging
import re
import time
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urljoin, urlparse

import aiohttp
from bs4 import BeautifulSoup

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
    MIN_RELEVANCE_THRESHOLD
)

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

# === ДОБАВЛЕНО: попытка импорта библиотек для PDF ===
try:
    import pypdf
    PDF_AVAILABLE = True
except ImportError:
    try:
        import pdfplumber
        PDF_AVAILABLE = True
    except ImportError:
        PDF_AVAILABLE = False
        logger = logging.getLogger(__name__)
        logger.info("Для чтения PDF установите pypdf или pdfplumber")

logger = logging.getLogger(__name__)


# ---------- Кэш ----------
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

# github.com/<user>/<repo>/blob/<branch>/<path> -> raw.githubusercontent.com/...
_GITHUB_BLOB_RE = re.compile(
    r'^https?://github\.com/([^/]+)/([^/]+)/blob/([^/]+)/(.+)$'
)


def normalize_raw_url(url: str) -> str:
    """Приводит GitHub blob-ссылки к raw-варианту. Остальные URL не трогает."""
    m = _GITHUB_BLOB_RE.match(url.strip())
    if m:
        user, repo, branch, path = m.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{branch}/{path}"
    return url


def extract_urls(text: str) -> List[str]:
    return URL_RE.findall(text or "")


# === ДОБАВЛЕНО: функция извлечения текста из PDF ===
def _extract_pdf_text(raw_bytes: bytes) -> str:
    """Извлекает текст из PDF, если доступна одна из библиотек."""
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
            import pdfplumber
            with pdfplumber.open(io.BytesIO(raw_bytes)) as pdf:
                text = "\n".join(page.extract_text() for page in pdf.pages if page.extract_text())
            return text
        except Exception as e:
            logger.debug(f"PDF extraction failed: {e}")
            return ""


# ---------- Загрузчик страниц ----------
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

    # Расширенный список текстовых типов (добавлен application/octet-stream)
    TEXTUAL_CONTENT_TYPES = (
        "text/html", "application/xhtml", "text/plain", "text/markdown",
        "text/x-markdown", "application/json", "text/csv", "text/xml",
        "application/xml", "text/javascript", "application/javascript",
        "application/x-yaml", "text/yaml", "text/x-python", "text/x-python-script",
        "text/x-c", "text/x-csrc", "text/x-java-source", "application/x-sh",
        "application/octet-stream",  # <-- добавлено для GitHub raw и др.
    )

    async def fetch(self, url: str) -> str:
        url = normalize_raw_url(url)
        try:
            session = await self._get_session()
            async with session.get(url, allow_redirects=True, max_redirects=5) as resp:
                if resp.status != 200:
                    return ""
                content_type = resp.headers.get("Content-Type", "").lower()
                is_textual = any(t in content_type for t in self.TEXTUAL_CONTENT_TYPES)

                # === НОВОЕ: обработка PDF ===
                if 'application/pdf' in content_type or url.lower().endswith('.pdf'):
                    raw_bytes = await resp.read()
                    text = _extract_pdf_text(raw_bytes)
                    if text:
                        return self._clean_plain_text(text)
                    return ""

                # Если Content-Type вообще не задан (бывает у некоторых raw-раздач),
                # не отбрасываем сразу — пробуем прочитать как текст.
                if content_type and not is_textual:
                    return ""

                raw = await resp.text(errors="replace")
                if "text/html" in content_type or "application/xhtml" in content_type:
                    return self._extract_text(raw)
                # Не-HTML текст (raw-файлы с гитхаба, json, markdown, код и т.п.) —
                # используем как есть, без попытки распарсить как HTML.
                return self._clean_plain_text(raw)
        except Exception as e:
            logger.debug(f"Fetch error for {url}: {e}")
            return ""

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

    def _extract_text(self, html: str) -> str:
        try:
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]):
                tag.decompose()
            main = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile("content|article|post"))
            if main:
                text = main.get_text(separator="\n", strip=True)
            else:
                text = soup.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)
            if len(text) > PAGE_CONTENT_MAX_CHARS:
                text = text[:PAGE_CONTENT_MAX_CHARS] + "\n...[truncated]"
            return text
        except Exception as e:
            logger.debug(f"Parse error: {e}")
            return ""

    def _extract_text_and_links(self, html: str, base_url: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Как _extract_text, но дополнительно собирает ссылки из того же
        содержательного блока (main/article/...), которые ведут на тот же домен."""
        try:
            soup = BeautifulSoup(html, "lxml")
            for tag in soup(["script", "style", "nav", "footer", "header", "aside", "form", "iframe", "noscript"]):
                tag.decompose()
            main = soup.find("main") or soup.find("article") or soup.find("div", class_=re.compile("content|article|post"))
            container = main or soup
            text = container.get_text(separator="\n", strip=True)
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            text = "\n".join(lines)

            base_domain = urlparse(base_url).netloc
            links: List[Tuple[str, str]] = []
            seen_hrefs = set()
            for a in container.find_all("a", href=True):
                href = a["href"].strip()
                if not href or href.startswith("#") or href.startswith("javascript:") or href.startswith("mailto:"):
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

            if len(text) > PAGE_CONTENT_MAX_CHARS:
                text = text[:PAGE_CONTENT_MAX_CHARS] + "\n...[truncated]"
            return text, links
        except Exception as e:
            logger.debug(f"Parse error: {e}")
            return "", []

    async def fetch_with_links(self, url: str) -> Tuple[str, List[Tuple[str, str]]]:
        """Как fetch(), но для HTML-страниц дополнительно возвращает ссылки."""
        url = normalize_raw_url(url)
        try:
            session = await self._get_session()
            async with session.get(url, allow_redirects=True, max_redirects=5) as resp:
                if resp.status != 200:
                    return "", []
                content_type = resp.headers.get("Content-Type", "").lower()
                is_textual = any(t in content_type for t in self.TEXTUAL_CONTENT_TYPES)

                # PDF – возвращаем текст без ссылок
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
                    return self._extract_text_and_links(raw, final_url)
                return self._clean_plain_text(raw), []
        except Exception as e:
            logger.debug(f"Fetch error for {url}: {e}")
            return "", []

    async def fetch_many(self, urls: List[str], limit: int = PARALLEL_FETCH_LIMIT) -> List[Tuple[str, str]]:
        semaphore = asyncio.Semaphore(limit)

        async def fetch_one(url):
            async with semaphore:
                text = await self.fetch(url)
                return url, text

        tasks = [fetch_one(u) for u in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = []
        for r in results:
            if isinstance(r, tuple):
                out.append(r)
        return out

    async def fetch_many_with_links(self, urls: List[str],
                                     limit: int = PARALLEL_FETCH_LIMIT) -> List[Tuple[str, str, List[Tuple[str, str]]]]:
        semaphore = asyncio.Semaphore(limit)

        async def fetch_one(url):
            async with semaphore:
                text, links = await self.fetch_with_links(url)
                return url, text, links

        tasks = [fetch_one(u) for u in urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = []
        for r in results:
            if isinstance(r, tuple):
                out.append(r)
        return out

    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()


# ---------- Ранжировщик ----------
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


# ---------- Оценка доверия к источнику ----------
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


# ---------- Утилиты ----------
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


# ---------- Модульные синглтоны ----------
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


# ---------- Извлечение наиболее релевантного отрывка ----------
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


async def deep_search(query: str, max_results: int = MAX_PAGES_TO_FETCH) -> Dict[str, Any]:
    # Обработка прямых ссылок
    direct_urls = extract_urls(query)
    if direct_urls:
        fetched = await asyncio.gather(*[fetch_url(u) for u in direct_urls[:max_results]])
        sources = [{"title": r["title"], "url": r["url"]} for r in fetched if r["ok"]]
        context_parts = [
            f"Источник: {r['url']}\nURL: {r['url']}\n{r['text']}"
            for r in fetched if r["ok"]
        ]
        if context_parts:
            result = {
                "sources": sources,
                "context": "\n\n---\n\n".join(context_parts),
                "search_performed": True,
                "chunks_found": len(context_parts),
            }
            return result
        remainder = URL_RE.sub("", query).strip()
        if remainder:
            query = remainder

    cache_key = hash_query(query)
    skip_cache = content_has_currency_numbers(query)

    if not skip_cache:
        cached = await _search_cache.get(cache_key)
        if cached:
            return cached

    ddg_results = await search_ddg(query, max_results=max_results + 2)
    if not ddg_results:
        return {"sources": [], "context": "Поиск не дал результатов.", "search_performed": False}

    # Сортировка с учётом доверия к домену
    scored = [
        (ChunkRanker.score_text(query, f"{r.get('title', '')} {r.get('snippet', '')}")
         + domain_trust(r.get('url', ''))[1], r)
        for r in ddg_results
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    above_threshold = [r for s, r in scored if s >= MIN_RELEVANCE_THRESHOLD]
    ordered_results = above_threshold if above_threshold else [r for _, r in scored]

    urls = [r["url"] for r in ordered_results if r.get("url")][:max_results]
    fetched = await _fetcher.fetch_many_with_links(urls, limit=PARALLEL_FETCH_LIMIT)

    url_to_title = {r["url"]: r["title"] for r in ddg_results}
    sources = []
    context_parts = []

    # === УЛУЧШЕНИЕ: множественные переходы по ссылкам ===
    MAX_EXTRA_HOPS = 3
    extra_hops_used = 0

    for url, text, links in fetched:
        if not text:
            snippet = next((r["snippet"] for r in ddg_results if r["url"] == url), "")
            if snippet:
                text = f"{url_to_title.get(url, '')}\n{snippet}"
            else:
                continue

        page_score = ChunkRanker.score_text(query, text)

        # Если страница нерелевантна, но имеет ссылки – пробуем перейти по нескольким лучшим
        if page_score <= 0 and links and extra_hops_used < MAX_EXTRA_HOPS:
            # Ранжируем ссылки по релевантности анкора
            scored_links = []
            for anchor_text, href in links:
                s = ChunkRanker.score_text(query, anchor_text)
                if s > 0:
                    scored_links.append((s, href))
            scored_links.sort(key=lambda x: x[0], reverse=True)
            # Берём до 3 лучших (но не больше оставшихся hop'ов)
            take = min(3, MAX_EXTRA_HOPS - extra_hops_used, len(scored_links))
            if take > 0:
                best_hrefs = [href for _, href in scored_links[:take]]
                # Параллельно скачиваем их
                tasks = [_fetcher.fetch_with_links(h) for h in best_hrefs]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                for (href, deep_text, deep_links), res in zip(best_hrefs, results):
                    if isinstance(res, Exception):
                        continue
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
                # после обработки ссылок пропускаем добавление самой нерелевантной страницы
                continue

        # Обычная обработка релевантной страницы
        sources.append({"title": url_to_title.get(url, url), "url": url})
        excerpt = best_excerpt(query, text, max_chars=3000)
        trust_label, _ = domain_trust(url)
        if trust_label:
            sources[-1]["reliability"] = trust_label
        reliability_suffix = f" [надёжность источника: {trust_label}]" if trust_label else ""
        context_parts.append(f"Источник: {url_to_title.get(url, url)}{reliability_suffix}\nURL: {url}\n{excerpt}")

    context = "\n\n---\n\n".join(context_parts)
    result = {
        "sources": sources,
        "context": context,
        "search_performed": True,
        "chunks_found": len(context_parts)
    }
    if not skip_cache:
        await _search_cache.set(cache_key, result)
    return result


async def close_search_resources():
    await _fetcher.close()


__all__ = [
    'SearchCache',
    'WebPageFetcher',
    'ChunkRanker',
    'hash_query',
    'content_has_currency_numbers',
    'search_ddg',
    'deep_search',
    'fetch_url',
    'normalize_raw_url',
    'extract_urls',
    'best_excerpt',
    'domain_trust',
    'close_search_resources'
]