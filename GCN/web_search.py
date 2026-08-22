import asyncio
import hashlib
import logging
import re
import time
from collections import OrderedDict
from typing import List, Dict, Any, Optional, Tuple

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

    async def fetch(self, url: str) -> str:
        try:
            session = await self._get_session()
            async with session.get(url, allow_redirects=True, max_redirects=5) as resp:
                if resp.status != 200:
                    return ""
                content_type = resp.headers.get("Content-Type", "")
                if "text/html" not in content_type and "application/xhtml" not in content_type:
                    return ""
                html = await resp.text()
                return self._extract_text(html)
        except Exception as e:
            logger.debug(f"Fetch error for {url}: {e}")
            return ""

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


# ---------- Утилиты ----------
def hash_query(q: str) -> str:
    return hashlib.sha256(q.lower().strip().encode()).hexdigest()[:32]


def content_has_currency_numbers(text: str) -> bool:
    if not text:
        return False
    patterns = [
        r'\b\d{1,3}[.,]\d{2}\b',
        r'\b\d{1,3}\.\d{2}\s*(?:₽|руб|RUB|USD|EUR)\b',
        r'(?:USD|EUR|RUB)\s*/\s*(?:RUB|USD|EUR)\s*[:=]?\s*\d{1,3}[.,]\d{2}',
        r'курс.*\d{1,3}[.,]\d{2}',
        r'доллар.*\d{1,3}[.,]\d{2}',
        r'евро.*\d{1,3}[.,]\d{2}',
    ]
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False


# ---------- Модульные синглтоны (были локальными — кеш и коннекшн-пул
# создавались заново на каждый вызов deep_search и не работали вообще) ----------
_search_cache = SearchCache()
_fetcher = WebPageFetcher()
_ddg_lock = asyncio.Lock()
_last_ddg_call = 0.0


async def search_ddg(query: str, max_results: int = 5) -> List[Dict]:
    """DDG-поиск с троттлингом и ретраями (раньше это было только в неиспользуемом
    CognitiveController.search_ddg — реальный путь вызова шёл в обход)."""
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


async def deep_search(query: str, max_results: int = MAX_PAGES_TO_FETCH) -> Dict[str, Any]:
    cache_key = hash_query(query)

    # Курсы/цены устаревают быстро — для таких запросов кеш сознательно
    # обходим, чтобы не отдавать протухшие цифры (раньше это никогда не
    # срабатывало, т.к. кеш был всегда пуст).
    skip_cache = content_has_currency_numbers(query)

    if not skip_cache:
        cached = await _search_cache.get(cache_key)
        if cached:
            return cached

    ddg_results = await search_ddg(query, max_results=max_results + 2)
    if not ddg_results:
        return {"sources": [], "context": "Поиск не дал результатов.", "search_performed": False}

    urls = [r["url"] for r in ddg_results if r.get("url")]
    fetched = await _fetcher.fetch_many(urls[:max_results], limit=PARALLEL_FETCH_LIMIT)

    url_to_title = {r["url"]: r["title"] for r in ddg_results}
    sources = []
    context_parts = []
    for url, text in fetched:
        if not text:
            snippet = next((r["snippet"] for r in ddg_results if r["url"] == url), "")
            if snippet:
                text = f"{url_to_title.get(url, '')}\n{snippet}"
            else:
                continue
        sources.append({"title": url_to_title.get(url, url), "url": url})
        context_parts.append(f"Источник: {url_to_title.get(url, url)}\nURL: {url}\n{text[:1000]}")

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
    """Вызывать при остановке приложения — закрывает переиспользуемую aiohttp-сессию."""
    await _fetcher.close()


__all__ = [
    'SearchCache',
    'WebPageFetcher',
    'ChunkRanker',
    'hash_query',
    'content_has_currency_numbers',
    'search_ddg',
    'deep_search',
    'close_search_resources'
]