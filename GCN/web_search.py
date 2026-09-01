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
# Обычная HTML-страница blob'а у GitHub — это React-приложение, реальный код
# лежит внутри вложенного JSON внутри <script>, а не в видимом тексте страницы,
# поэтому BeautifulSoup с неё практически ничего полезного не вытаскивал.
# Raw-хост отдаёт файл как чистый текст — именно то, что нужно для чтения кода.
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

    # Раньше сюда попадал только "text/html"/"application/xhtml" — из-за этого
    # молча отбрасывалось всё, что не рендерит HTML: raw.githubusercontent.com
    # (text/plain), .py/.md/.json/.yaml файлы, gist-раблы и т.д. Именно это
    # объясняло "не читает гит и файлы там" — запрос доходил до сервера,
    # получал 200 OK, и текст просто выкидывался на этой проверке.
    TEXTUAL_CONTENT_TYPES = (
        "text/html", "application/xhtml", "text/plain", "text/markdown",
        "text/x-markdown", "application/json", "text/csv", "text/xml",
        "application/xml", "text/javascript", "application/javascript",
        "application/x-yaml", "text/yaml", "text/x-python", "text/x-python-script",
        "text/x-c", "text/x-csrc", "text/x-java-source", "application/x-sh",
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
        # схлопываем длинные последовательности пустых строк, но не режем контент
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
        """
        Как _extract_text, но дополнительно собирает ссылки из того же
        содержательного блока (main/article/...), которые ведут на тот же домен.
        Нужно для "второго прыжка": когда сама страница малоинформативна
        (например, это оглавление/лендинг), но по ссылкам с неё можно дойти
        до страницы, которая реально отвечает на запрос — вместо того, чтобы
        просто отдать модели верхнеуровневый нерелевантный текст.
        Ограничено тем же доменом намеренно — не хотим по ссылке с одной
        страницы улетать на случайный сторонний сайт без запроса пользователя.
        """
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
        """Как fetch(), но для HTML-страниц дополнительно возвращает ссылки
        с той же страницы (см. _extract_text_and_links) — основа для перехода
        "вглубь", когда верхний уровень оказался нерелевантным."""
        url = normalize_raw_url(url)
        try:
            session = await self._get_session()
            async with session.get(url, allow_redirects=True, max_redirects=5) as resp:
                if resp.status != 200:
                    return "", []
                content_type = resp.headers.get("Content-Type", "").lower()
                is_textual = any(t in content_type for t in self.TEXTUAL_CONTENT_TYPES)
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
        """
        Та же формула, что и в score_chunks, но для одного куска текста целиком
        (сниппета DDG, текста анкора ссылки, всей страницы) — раньше эта оценка
        считалась только внутри чанков одной уже скачанной страницы. Используется
        для отбора/сортировки результатов ПЕРЕД скачиванием (по сниппету) и для
        решения "стоит ли переходить по ссылке вглубь" (см. deep_search).
        """
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


# ---------- Оценка доверия к источнику (пункт №3 плана улучшений) ----------
# Раньше результаты DDG сортировались/фильтровались только по текстовому
# совпадению сниппета с запросом (ChunkRanker.score_text) — авторитетность
# домена никак не учитывалась, а .gov/.edu/крупное СМИ и случайный форум/
# сайт-агрегатор были равнозначны. Это не полноценная проверка фактов
# (нужен был бы отдельный сервис), а лёгкий, бесплатный (без доп. запросов)
# сигнал, который двигает более надёжные источники выше в очереди на
# скачивание и помечает их для модели прямо в тексте контекста.
_TRUSTED_DOMAINS_HIGH = (
    ".gov", ".gov.ru", ".edu", ".mil",
    "wikipedia.org", "who.int", "un.org",
    "cbr.ru",  # Банк России — официальный источник курсов валют
)
_TRUSTED_DOMAINS_MEDIUM = (
    "reuters.com", "apnews.com", "bbc.com", "bbc.co.uk", "bloomberg.com",
    "tass.ru", "ria.ru", "interfax.ru", "kommersant.ru", "vedomosti.ru",
    "nature.com", "sciencedirect.com", "arxiv.org", "github.com",
    "docs.python.org", "developer.mozilla.org", "stackoverflow.com",
)
# Явно ненадёжные/спамные домены-агрегаторы контента — не блокируем (страница
# может быть единственным результатом по редкому запросу), но не даём буст.
_LOW_TRUST_MARKERS = ("pinterest.", "quora.com",)

DOMAIN_TRUST_BOOST_HIGH = 0.35
DOMAIN_TRUST_BOOST_MEDIUM = 0.15


def domain_trust(url: str) -> Tuple[str, float]:
    """Возвращает (метка_для_человека, буст_к_скору) на основе домена URL.
    Метка пустая и буст 0.0, если домен не входит ни в один список — это
    НЕ штраф, просто отсутствие дополнительного сигнала доверия."""
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
    """
    Определяет, стоит ли считать текст (обычно — поисковый запрос) достаточно
    "чувствительным ко времени" (курс валюты, цена), чтобы обходить кеш в
    deep_search.

    ИСПРАВЛЕНИЕ: раньше все паттерны требовали, чтобы в самом тексте уже было
    отформатированное число вида "95.40" — то есть текст должен был уже
    СОДЕРЖАТЬ курс/цену. Но deep_search вызывает эту функцию на пользовательском
    ЗАПРОСЕ (query), а не на найденном тексте — а в запросе вида "какой сегодня
    курс доллара" числа по определению ещё нет: пользователь как раз его
    спрашивает. Из-за этого функция практически никогда не возвращала True на
    реальных вопросах о курсах/ценах, skip_cache молча не срабатывал, и
    deep_search мог до 5 минут (SEARCH_CACHE_TTL) отдавать устаревший курс на
    повторный вопрос — ровно то, что комментарий над вызовом в deep_search
    обещал предотвращать. Теперь дополнительно ловим сами ТЕМЫ курса/цены по
    ключевым словам, не требуя, чтобы число уже было в тексте; числовые
    паттерны оставлены как есть — они по-прежнему полезны, когда текст уже
    содержит цифры (например, при проверке уже найденного контента).
    """
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
    # Тематические маркеры "текущего курса/цены" — срабатывают даже если
    # числа в тексте ещё нет (типичный случай для самого запроса).
    topic_words = (
        "курс", "доллар", "евро", "биткоин", "bitcoin", "btc", "эфириум", "eth",
        "акци", "котировк", "цена", "стоимост", "прайс", "price", "exchange rate",
    )
    text_l = text.lower()
    if any(w in text_l for w in topic_words):
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


async def fetch_url(url: str, max_chars: int = PAGE_CONTENT_MAX_CHARS) -> Dict[str, Any]:
    """
    Прямое чтение конкретной ссылки (без похода в DDG) — используется, когда
    пользователь сам прислал URL (например, ссылку на файл в гитхабе) и хочет,
    чтобы ассистент прочитал именно её, а не то, что найдётся по текстовому
    поиску по этой ссылке как по строке.
    """
    url = url.strip()
    text = await _fetcher.fetch(url)
    if not text:
        return {"url": url, "title": url, "text": "", "ok": False,
                "error": "Не удалось получить содержимое (страница недоступна, "
                         "требует авторизации или формат не поддерживается)."}
    if len(text) > max_chars:
        text = text[:max_chars] + "\n...[truncated]"
    return {"url": url, "title": url, "text": text, "ok": True}


# ---------- Извлечение наиболее релевantного отрывка ----------
def best_excerpt(query: str, text: str, max_chars: int = 3000) -> str:
    """
    Раньше на этом месте был плоский text[:3000] — если релевантная часть
    страницы лежала не в начале (частый случай для длинных статей/доков),
    в контекст модели попадал нерелевантный кусок (шапка, вступление, меню),
    а ChunkRanker/chunk_text были объявлены и никогда не вызывались.
    Разбиваем текст на чанки, ранжируем по совпадению с запросом и берём
    столько лучших чанков, сколько влезает в max_chars, сохраняя их исходный
    порядок в странице (для связности), а не порядок по убыванию скора.
    """
    if len(text) <= max_chars:
        return text
    chunks = ChunkRanker.chunk_text(text)
    if len(chunks) <= 1:
        return text[:max_chars]
    scored = ChunkRanker.score_chunks(query, chunks)
    # если ни один чанк не пересёкся с запросом лексически (score==0 везде),
    # ранжирование бессмысленно — возвращаемся к началу текста как и раньше
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
    # Если в запросе есть прямые ссылки — читаем их напрямую, а не гоняем как
    # текст через DDG (где сама ссылка, скорее всего, не совпадёт с чужими
    # проиндексированными страницами и просто ничего полезного не найдёт).
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
        # Ссылка есть, но прочитать не удалось (например, требует JS) —
        # падаем обратно на обычный поиск по остальному тексту запроса,
        # а не молча возвращаем пустоту.
        remainder = URL_RE.sub("", query).strip()
        if remainder:
            query = remainder

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

    # НОВОЕ: сортируем результаты DDG по релевантности сниппета запросу ДО того,
    # как тратить сетевые запросы на скачивание страниц. Раньше страницы
    # скачивались строго в порядке выдачи DDG без какого-либо анализа того,
    # отвечает ли сниппет вообще на запрос — MIN_RELEVANCE_THRESHOLD импортировался
    # из конфига, но нигде не использовался. Совсем нерелевантные по тексту
    # сниппета результаты (0 общих слов с запросом) уходят в конец очереди и не
    # съедают бюджет max_results, если есть более релевантные варианты.
    # ДОБАВЛЕНО (пункт №3 плана улучшений): к текстовой релевантности добавляется
    # буст за авторитетность домена (domain_trust) — при близких по тексту
    # сниппетах официальный/крупный источник (.gov, Reuters, cbr.ru и т.п.)
    # уходит на скачивание раньше случайного сайта-агрегатора. Это буст, а не
    # фильтр: нерелевантный, но "надёжный" домен всё равно не обгонит явно
    # релевантный результат с большим текстовым совпадением.
    scored = [
        (ChunkRanker.score_text(query, f"{r.get('title', '')} {r.get('snippet', '')}")
         + domain_trust(r.get('url', ''))[1], r)
        for r in ddg_results
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    # MIN_RELEVANCE_THRESHOLD импортировался из конфига, но нигде не применялся —
    # отфильтровываем сниппеты, которые совсем не пересекаются по смыслу с
    # запросом, НО только если после фильтра хоть что-то остаётся: для редких/
    # узкоспециальных запросов сниппеты DDG иногда формулируются совсем другими
    # словами, чем сам запрос, и жёсткий порог без запасного варианта оставил бы
    # поиск ни с чем вместо результата похуже, но хоть какого-то.
    above_threshold = [r for s, r in scored if s >= MIN_RELEVANCE_THRESHOLD]
    ordered_results = above_threshold if above_threshold else [r for _, r in scored]

    urls = [r["url"] for r in ordered_results if r.get("url")][:max_results]
    fetched = await _fetcher.fetch_many_with_links(urls, limit=PARALLEL_FETCH_LIMIT)

    url_to_title = {r["url"]: r["title"] for r in ddg_results}
    sources = []
    context_parts = []
    # НОВОЕ: "второй прыжок" — не более нескольких доп. переходов за один вызов
    # deep_search, чтобы не превращать поиск в неограниченный обход сайта.
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
        # Страница скачалась, но по тексту ни одного общего слова с запросом —
        # частый случай для лендингов/оглавлений документации, где реальный
        # ответ лежит на подстранице. Вместо того чтобы отдать модели заведомо
        # нерелевантный текст, пробуем один переход по самой релевантной ссылке
        # с этой же страницы (по совпадению текста ссылки с запросом).
        if page_score <= 0 and links and extra_hops_used < MAX_EXTRA_HOPS:
            best_href, best_link_score = None, 0.0
            for anchor_text, href in links:
                s = ChunkRanker.score_text(query, anchor_text)
                if s > best_link_score:
                    best_link_score, best_href = s, href
            if best_href and best_link_score > 0:
                extra_hops_used += 1
                deep_text, _ = await _fetcher.fetch_with_links(best_href)
                if deep_text and ChunkRanker.score_text(query, deep_text) > page_score:
                    logger.info(f"deep_search: {url} нерелевантна, перешёл по ссылке -> {best_href}")
                    hop_trust_label, _ = domain_trust(best_href)
                    hop_source = {"title": f"{url_to_title.get(url, url)} → подробнее", "url": best_href}
                    if hop_trust_label:
                        hop_source["reliability"] = hop_trust_label
                    sources.append(hop_source)
                    excerpt = best_excerpt(query, deep_text, max_chars=3000)
                    hop_suffix = f" [надёжность источника: {hop_trust_label}]" if hop_trust_label else ""
                    context_parts.append(
                        f"Источник: {url_to_title.get(url, url)} (подробности по ссылке со страницы){hop_suffix}\n"
                        f"URL: {best_href}\n{excerpt}"
                    )
                    continue  # верхнеуровневую нерелевантную страницу отдельно не добавляем

        sources.append({"title": url_to_title.get(url, url), "url": url})
        # Раньше здесь стояло text[:1000] (потом text[:3000]) — плоская обрезка,
        # игнорирующая релевантность. Теперь выбираем чанки, реально относящиеся
        # к запросу (см. best_excerpt/ChunkRanker), вместо первых символов страницы.
        excerpt = best_excerpt(query, text, max_chars=3000)
        # ДОБАВЛЕНО (пункт №3): помечаем в тексте контекста и в метаданных
        # источника его уровень доверия по домену, если он определён — модель
        # видит это прямо рядом с текстом (не нужно самой гадать по URL), а
        # вызывающий код (search_meta["sources"]) получает поле "reliability"
        # для отображения в клиенте, если понадобится.
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
    'fetch_url',
    'normalize_raw_url',
    'extract_urls',
    'best_excerpt',
    'domain_trust',
    'close_search_resources'
]