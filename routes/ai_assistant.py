"""
Когнитивный ассистент с интеграцией CognitiveMemory, планированием, автономностью.
"""
import sys
import uuid
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import logging
import json
import asyncio
import time
import hashlib
import re
from typing import Dict, Optional, Any, List, Tuple
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
import numpy as np
import aiohttp
from bs4 import BeautifulSoup
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from GCN.GCN import AIAdapter, KnowledgeObject, KnowledgeType, MemoryScope
from GCN.memory_graph import CognitiveMemory, Fact, Episode, Goal, GCNMemoryRouter

try:
    from GCN.config_ai import *
except ImportError:
    # fallback (все необходимые переменные)
    LM_STUDIO_URL = "http://localhost:1234/v1/chat/completions"
    LM_STUDIO_API_KEY = "lm-studio"
    LM_STUDIO_TIMEOUT = 160
    LM_STUDIO_STREAM_TIMEOUT = 500
    LM_STUDIO_USE_STREAM = True
    LM_STUDIO_VISION_SUPPORTED = False
    MEMORY_BASE_DIR = Path("ai_memory_v3")
    MEMORY_BASE_DIR.mkdir(exist_ok=True)
    MAX_MESSAGE_LENGTH = 10000
    MIN_MESSAGE_LENGTH = 1
    DEEP_SEARCH_TOTAL_BUDGET = 15
    REFLECTION_INTERVAL = 3600 * 4
    REFLECTION_ERROR_THRESHOLD = 0.6
    REFLECTION_HISTORY_SIZE = 100
    REFLECTION_LLM_TEMP = 0.5
    REFLECTION_LLM_MAX_TOKENS = 300
    CONSOLIDATION_INTERVAL = 3600 * 2
    DEEP_CONSOLIDATION_INTERVAL = 3600 * 8
    CURIOSITY_RESEARCH_INTERVAL = 600
    LONG_TERM_PLANNER_INTERVAL = 3600 * 6
    DDG_MIN_INTERVAL = 1.2
    DDG_MAX_RETRIES = 3
    SEARCH_CACHE_TTL = 300
    SEARCH_CACHE_MAX_SIZE = 200
    PAGE_CONTENT_MAX_CHARS = 6000
    MAX_PAGES_TO_FETCH = 7
    MIN_RELEVANCE_THRESHOLD = 0.28
    CHUNK_SIZE = 1200
    CHUNK_OVERLAP = 150
    PARALLEL_FETCH_LIMIT = 8
    MAX_SEARCH_ATTEMPTS = 3
    ENABLE_QUERY_REWRITE = True
    EXTRACT_FACTS_FROM_SEARCH = True
    EXTRACT_FACTS_WITH_LLM = True
    EASYDIFFUSION_ENABLED = True
    EASYDIFFUSION_URL = "http://localhost:9000"
    EASYDIFFUSION_TIMEOUT = 120
    EASYDIFFUSION_DEFAULT_STEPS = 20
    EASYDIFFUSION_DEFAULT_WIDTH = 512
    EASYDIFFUSION_DEFAULT_HEIGHT = 512
    STREAM_CHAR_BY_CHAR = False
    STREAM_CHAR_DELAY = 0.02
    MAX_IMAGE_SIZE_BASE64 = 5 * 1024 * 1024
    MEMORY_CONTROL_COMMANDS = {
        "запомни": "store",
        "забудь": "forget",
        "что ты знаешь о": "recall"
    }

logger = logging.getLogger(__name__)

try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False
    logger.warning("⚠️ ddgs not installed")

try:
    from dependencies import require_auth
except ImportError:
    async def require_auth():
        return "anonymous"


def _now() -> float:
    return time.time()


def _hash_query(q: str) -> str:
    """SHA-256 хэш запроса (32 символа — без коллизий)."""
    return hashlib.sha256(q.lower().strip().encode()).hexdigest()[:32]


# =====================================================================
# 1. Поисковый кэш (LRU + TTL)
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
            if _now() - ts > self.ttl:
                del self._cache[key]
                return None
            self._cache.move_to_end(key)
            return value

    async def set(self, key: str, value: Any):
        async with self._lock:
            self._cache[key] = (value, _now())
            self._cache.move_to_end(key)
            while len(self._cache) > self.maxsize:
                self._cache.popitem(last=False)


# =====================================================================
# 2. Загрузчик страниц
# =====================================================================
class WebPageFetcher:
    def __init__(self, timeout: int = 15):
        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
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
                return self._extract_text(html, url)
        except Exception as e:
            logger.debug(f"Fetch error for {url}: {e}")
            return ""

    def _extract_text(self, html: str, url: str) -> str:
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
            logger.debug(f"Parse error for {url}: {e}")
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


# =====================================================================
# 3. Ранжирование чанков
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
# 4. Умный триггер поиска
# =====================================================================
SEARCH_TRIGGER_KEYWORDS = [
    'сегодня', 'сейчас', 'новости', 'курс', 'погода', 'свежие',
    'последние', 'завтра', 'найди', 'поищи', 'информацию', 'актуальные',
    '2024', '2025', '2026', 'сколько стоит', 'какой сейчас', 'последние данные',
    'факт', 'статистика', 'результаты', 'кто победил', 'когда выйдет',
    'где находится', 'как делается', 'пошагово', 'инструкция', 'рецепт',
    'сравнение', 'обзор', 'анализ', 'докажи', 'проверь', 'правда ли',
]


def needs_search_heuristic(message: str) -> bool:
    msg_lower = message.lower()
    if re.search(r'https?://\S+', msg_lower):
        return True
    if any(kw in msg_lower for kw in SEARCH_TRIGGER_KEYWORDS):
        return True
    if re.search(r'\b(сейчас|сегодня|вчера|завтра|этот год|этот месяц)\b', msg_lower):
        return True
    return False


def is_factual_query(message: str) -> bool:
    patterns = [
        r'\b\d+[.,]?\d*\s*(?:USD|EUR|RUB|₽|$|€|%|кг|км|г|м|см|мм|MB|GB|TB)\b',
        r'\b\d{1,2}[./-]\d{1,2}(?:[./-]\d{2,4})?\b',
        r'\b(?:курс|цена|стоимость|тариф|скорость|температура|вес|рост|расстояние)\b'
    ]
    for pat in patterns:
        if re.search(pat, message, re.IGNORECASE):
            return True
    return False


# =====================================================================
# 5. Переписывание запроса
# =====================================================================
async def rewrite_query(llm_caller, original: str) -> str:
    if not ENABLE_QUERY_REWRITE:
        return original
    prompt = (
        f"Перепиши вопрос в виде ОДНОГО короткого поискового запроса (3-10 слов), "
        f"оптимизированного для DuckDuckGo. Убери лишнее. Ответь ТОЛЬКО запросом, без пояснений.\n\n"
        f"Вопрос: {original}"
    )
    try:
        rewritten = await llm_caller([{"role": "user", "content": prompt}], temp=0.3, max_tokens=80)
        rewritten = rewritten.strip().strip('"').strip("'")
        if rewritten and len(rewritten) >= 5:
            logger.info(f"[QueryRewrite] '{original[:50]}...' -> '{rewritten[:80]}'")
            return rewritten
    except Exception as e:
        logger.debug(f"Query rewrite failed: {e}")
    return original


# =====================================================================
# 5b. Промпты для программно-парсимых LLM-вызовов (строгий JSON)
# =====================================================================
# Все три промпта ниже используются там, где ответ LLM парсится кодом, а не
# показывается пользователю напрямую. Поэтому: temp=0.0-0.2, явная JSON-схема,
# запрет на пояснения/markdown, и safe-парсинг с фолбэком на эвристику при сбое.

ROUTER_PROMPT = """Ты — модуль планирования когнитивного ассистента. Проанализируй запрос пользователя и контекст.
Верни ТОЛЬКО валидный JSON, без пояснений, без markdown-разметки, без ```:

{{
  "needs_web_search": true/false,
  "search_query": "переписанный поисковый запрос (3-10 слов) или null",
  "is_factual_time_sensitive": true/false,
  "answer_strategy": "direct" | "search_then_answer" | "recall_then_answer" | "clarify"
}}

Правила:
- needs_web_search=true, если для точного ответа нужны свежие/актуальные/числовые данные
  (курсы, цены, новости, даты, "сейчас", "сегодня"), которых нет в истории диалога.
- search_query — короткий запрос для поисковика, а не сам вопрос пользователя дословно.
- is_factual_time_sensitive=true для вопросов с числами, единицами измерения, курсами, датами.
- answer_strategy="clarify" только если вопрос пользователя действительно неоднозначен
  настолько, что угадать намерение нельзя.

Последние реплики диалога:
{history_tail}

Активные цели пользователя: {goals}

Запрос пользователя: {message}
"""

REFLECTION_PROMPT = """Ты — модуль саморефлексии когнитивного ассистента. Ниже темы, где предсказания модели чаще всего ошибались (ошибка > {threshold}).
Верни ТОЛЬКО валидный JSON, без пояснений, без markdown-разметки, без ```:

{{
  "weight_adjustments": {{"semantic": 0.0, "graph": 0.0, "freshness": 0.0, "evidence": 0.0, "confidence": 0.0}},
  "topics_to_research": ["тема1", "тема2"]
}}

Каждое значение в weight_adjustments — дельта в диапазоне [-0.05, 0.05] (0, если менять не нужно).
topics_to_research — темы, по которым стоит провести дополнительный автономный поиск (максимум 3).

Темы с ошибками:
{topics}
"""

CONTRADICTION_VERIFY_PROMPT = """Ты — верификатор фактов в системе памяти AI-ассистента. Даны два утверждения, помеченные как противоречащие друг другу.
Верни ТОЛЬКО валидный JSON, без пояснений, без markdown-разметки, без ```:

{{
  "relation": "true_contradiction" | "false_positive" | "both_partially_true",
  "keep": "A" | "B" | "both" | "neither",
  "reason": "краткое обоснование в одном предложении"
}}

"false_positive" — утверждения на самом деле не противоречат друг другу (например, относятся
к разным моментам времени, разным объектам, или совпадение ключевых слов случайно).
"both_partially_true" — оба верны в своём контексте, keep="both".

Утверждение A: {text_a}
Утверждение B: {text_b}
"""


def parse_llm_json(raw: str) -> Optional[Dict]:
    """
    Безопасный парсинг JSON из ответа LLM. Локальные модели (особенно через
    LM Studio) часто оборачивают JSON в ```json ... ``` или добавляют текст
    до/после — эта функция вытаскивает первый валидный JSON-объект.
    Возвращает None при неудаче — вызывающий код обязан иметь фолбэк.
    """
    if not raw:
        return None
    text = raw.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass
    # Последняя попытка: вырезать самый внешний {...} блок
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None


# =====================================================================
# 6. КОГНИТИВНЫЙ КОНТРОЛЛЕР
# =====================================================================
class CognitiveController:
    """
    Управляет когнитивным циклом: восприятие, память, предсказание,
    принятие решений, обучение.
    """

    def __init__(self, user_id: str):
        self.user_id = user_id
        self.user_dir = MEMORY_BASE_DIR / user_id
        self.user_dir.mkdir(parents=True, exist_ok=True)

        # ---- GCN-память: роутер (личная + глобальная + общая) ----
        self.router = GCNMemoryRouter(user_id, MEMORY_BASE_DIR)
        self.router.set_llm_caller(self._call_llm)  # передаём метод для извлечения фактов

        # Для обратной совместимости: старый код использует self.memory
        # Теперь self.memory указывает на личную память
        self.memory = self.router.private_memory

        # AIAdapter использует store из личной памяти (для публикации)
        embedder_func = (
            (lambda text: self.memory.embedder.encode(text, convert_to_numpy=True).tolist())
            if self.memory.use_embeddings and self.memory.embedder is not None
            else None
        )
        self.ai_adapter = AIAdapter(self.memory.store, user_id, embedder_func=embedder_func)

        self.history: List[Dict] = []
        self.max_history = 20
        self._load_history()

        self._searcher = None
        self._last_ddg_call = 0.0
        self.search_cache = SearchCache()
        self.web_fetcher = WebPageFetcher()
        self.chunk_ranker = ChunkRanker()

        self._consolidation_task = None
        self._planner_task = None
        self._research_task = None
        self._reflection_task = None
        self._start_background_tasks()

        self.current_working_memory: List[str] = []
        self.current_goals: List[Goal] = []
        self.last_prediction_error = 0.0
        self._last_prepare_meta: Dict = {}

        # ---- Рефлексия (самообучение) ----
        self.prediction_history: List[Dict] = []
        self.reflection_interval = REFLECTION_INTERVAL
        self._last_reflection_time = time.time()

        logger.info(f"CognitiveController (GCN) initialized for {user_id[:16]}")

    @property
    def searcher(self):
        if self._searcher is None and DDGS_AVAILABLE:
            self._searcher = DDGS()
        return self._searcher

    def _load_history(self):
        history_path = self.user_dir / "history.json"
        if history_path.exists():
            try:
                with open(history_path, "r", encoding="utf-8") as f:
                    self.history = json.load(f)[-self.max_history:]
            except Exception:
                pass

    def _save_history(self):
        history_path = self.user_dir / "history.json"
        try:
            with open(history_path, "w", encoding="utf-8") as f:
                json.dump(self.history[-self.max_history:], f, ensure_ascii=False)
        except Exception:
            pass

    def _start_background_tasks(self):
        loop = asyncio.get_event_loop()
        if loop.is_running():
            self._consolidation_task = asyncio.create_task(self._periodic_consolidation())
            self._planner_task = asyncio.create_task(self._periodic_planning())
            self._research_task = asyncio.create_task(self._periodic_research())
            self._reflection_task = asyncio.create_task(self._periodic_reflection())

    async def _periodic_consolidation(self):
        while True:
            await asyncio.sleep(CONSOLIDATION_INTERVAL)
            try:
                await self.memory.light_consolidation()
            except Exception as e:
                logger.error(f"Light consolidation error: {e}")
            try:
                await self._verify_pending_contradictions()
            except Exception as e:
                logger.error(f"Contradiction verification error: {e}")
            await asyncio.sleep(DEEP_CONSOLIDATION_INTERVAL - CONSOLIDATION_INTERVAL)
            try:
                await self.memory.deep_consolidation()
            except Exception as e:
                logger.error(f"Deep consolidation error: {e}")

    async def _periodic_planning(self):
        while True:
            await asyncio.sleep(LONG_TERM_PLANNER_INTERVAL)
            try:
                await self._plan_goals()
            except Exception as e:
                logger.error(f"Planning error: {e}")

    async def _periodic_research(self):
        while True:
            await asyncio.sleep(CURIOSITY_RESEARCH_INTERVAL)
            try:
                await self._auto_research()
            except Exception as e:
                logger.error(f"Auto research error: {e}")

    async def _route(self, message: str) -> Dict:
        """
        Заменяет собой связку rewrite_query() + is_factual_query()-эвристику
        одним LLM-вызовом с temp=0.0 и строгим JSON-выводом. Учитывает не
        только текущее сообщение, но и хвост диалога и активные цели —
        rewrite_query() видел только исходную строку.
        Вызывается из _prepare_messages ТОЛЬКО когда решение "искать" уже
        принято (см. needs_search_heuristic/явный флаг), поэтому не вносит
        дополнительный LLM round-trip в обычные (без поиска) реплики.
        """
        # Исправлено: используем правильную структуру истории
        history_tail = "\n".join(
            f"{item['role'].capitalize()}: {item['content'][:200]}"
            for item in self.history[-4:]
        ) if self.history else "(диалог только начался)"

        active_goals = [obj for obj in self.memory.store._objects.values()
                        if obj.type == KnowledgeType.HYPOTHESIS and obj.object.get("status") == "active"]
        goals_str = "; ".join(g.subject for g in active_goals[:3]) or "нет"

        prompt = ROUTER_PROMPT.format(history_tail=history_tail, goals=goals_str, message=message)
        try:
            raw = await self._call_llm([{"role": "user", "content": prompt}], temp=0.0, max_tokens=200)
            result = parse_llm_json(raw)
            if result and "search_query" in result:
                return result
            logger.warning(f"Router: bad/incomplete JSON, falling back to heuristics: {raw[:200]!r}")
        except Exception as e:
            logger.warning(f"Router LLM call failed, falling back to heuristics: {e}")

        # Фолбэк на regex-эвристику при сбое LLM/парсинга — offline safety net.
        return {
            "needs_web_search": True,
            "search_query": None,
            "is_factual_time_sensitive": is_factual_query(message),
            "answer_strategy": "search_then_answer",
        }

    async def _plan_goals(self):
        if len(self.history) < 5:
            return
        history_summary = "\n".join([f"User: {item['user']}\nAI: {item['assistant']}" for item in self.history[-10:]])
        prompt = (
            f"На основе диалогов с пользователем сформулируй 1-3 долгосрочные цели. "
            f"Ответь в виде списка целей (каждая с новой строки).\n\n{history_summary}"
        )
        try:
            goals_text = await self._call_llm([{"role": "user", "content": prompt}], temp=0.7, max_tokens=200)
            goals = [g.strip("-• ").strip() for g in goals_text.split('\n') if g.strip()]
            for g in goals:
                # Только add_goal – без ручного создания
                await self.memory.add_goal(g, priority=0.5)
            await self.memory._schedule_save()
            logger.info(f"[Planner] Generated goals: {goals}")
        except Exception as e:
            logger.error(f"Planning error: {e}")

    async def _auto_research(self):
        # Ищем активные цели (объекты типа HYPOTHESIS с object="active")
        active_goals = [obj for obj in self.memory.store._objects.values()
                        if obj.type == KnowledgeType.HYPOTHESIS and obj.object.get("status") == "active"]
        for goal_obj in active_goals:
            if goal_obj.confidence < 0.5:
                logger.info(f"Auto-research triggered for goal: {goal_obj.subject}")
                await self.research(goal_obj.subject)
                goal_obj.confidence = min(1.0, goal_obj.confidence + 0.2)
                self.memory.store.update(goal_obj.id, {"confidence": goal_obj.confidence}, self.user_id)
                self.memory._sync_goal_from_gcn(goal_obj.id)  # <--- добавить
        await self.memory._schedule_save()

    async def _call_llm(self, messages, temp=0.7, max_tokens=2048, retries=3):
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LM_STUDIO_API_KEY}"}
        payload = {"model": "local-model", "messages": messages, "temperature": temp, "max_tokens": max_tokens}
        for attempt in range(retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(LM_STUDIO_URL, json=payload, headers=headers, timeout=LM_STUDIO_TIMEOUT) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                        else:
                            error_text = await resp.text()
                            logger.error(f"LLM error {resp.status}: {error_text[:200]}")
                            if resp.status >= 500 and attempt < retries - 1:
                                await asyncio.sleep(2 ** attempt)
                                continue
                            return ""
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                if attempt < retries - 1:
                    await asyncio.sleep(2 ** attempt)
                    continue
                return ""
        return ""

    async def _call_llm_stream(self, messages, temp=0.7, max_tokens=2048):
        headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LM_STUDIO_API_KEY}"}
        payload = {
            "model": "local-model",
            "messages": messages,
            "temperature": temp,
            "max_tokens": max_tokens,
            "stream": True,
        }
        timeout = aiohttp.ClientTimeout(total=LM_STUDIO_STREAM_TIMEOUT)
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(LM_STUDIO_URL, json=payload, headers=headers, timeout=timeout) as resp:
                    logger.info(f"[Stream] Status: {resp.status}")
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(f"Stream error {resp.status}: {error_text[:200]}")
                        yield "[Ошибка LLM]"
                        return
                    async for line in resp.content:
                        line = line.decode('utf-8').strip()
                        if not line:
                            continue
                        if line.startswith('data: '):
                            data = line[6:]
                            if data == '[DONE]':
                                break
                            try:
                                chunk = json.loads(data)
                                delta = chunk.get('choices', [{}])[0].get('delta', {})
                                content = delta.get('content', '')
                                if content:
                                    yield content
                            except json.JSONDecodeError:
                                continue
        except asyncio.CancelledError:
            logger.debug("Stream cancelled")
            raise
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"[Ошибка: {e}]"

    async def search_ddg(self, query: str, max_results: int = 5) -> List[Dict]:
        if not self.searcher:
            return []
        elapsed = _now() - self._last_ddg_call
        if elapsed < DDG_MIN_INTERVAL:
            await asyncio.sleep(DDG_MIN_INTERVAL - elapsed)
        loop = asyncio.get_event_loop()
        for attempt in range(DDG_MAX_RETRIES):
            try:
                results = await loop.run_in_executor(
                    None,
                    lambda: list(self.searcher.text(query, max_results=max_results))
                )
                self._last_ddg_call = _now()
                return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in results]
            except Exception as e:
                logger.warning(f"DDG attempt {attempt + 1} failed: {e}")
                if attempt < DDG_MAX_RETRIES - 1:
                    await asyncio.sleep((2 ** attempt) + 0.5)
                else:
                    logger.error("DDG failed after all retries")
        return []

    # =====================================================================
    # РЕФЛЕКСИЯ (самообучение на ошибках)
    # =====================================================================

    def _compute_prediction_error(self, predicted: List[str], actual: str) -> float:
        """
        Оценивает, насколько предсказанные фразы (список) похожи на реальный ответ.
        Возвращает 0 (идеально) … 1 (полное несовпадение).
        Использует схожесть по ключевым словам из memory_graph.
        """
        if not predicted or not actual:
            return 1.0
        pred_text = " ".join(predicted)
        sim = self.memory._compute_similarity(pred_text, actual)
        error = 1.0 - min(1.0, sim * 1.5)
        return max(0.0, min(1.0, error))

    async def _periodic_reflection(self):
        while True:
            await asyncio.sleep(self.reflection_interval)
            try:
                await self._run_reflection()
            except Exception as e:
                logger.error(f"Reflection error: {e}")

    async def _run_reflection(self):
        if len(self.prediction_history) < 10:
            return

        errors_by_keyword = defaultdict(list)
        for entry in self.prediction_history[-REFLECTION_HISTORY_SIZE:]:
            if entry["error"] > REFLECTION_ERROR_THRESHOLD:
                kw = CognitiveMemory._extract_keywords(entry["query"])
                for word in kw:
                    errors_by_keyword[word].append(entry["error"])

        if not errors_by_keyword:
            return

        worst_topics = sorted(errors_by_keyword.items(), key=lambda kv: sum(kv[1]) / len(kv[1]), reverse=True)[:3]
        topics_str = "\n".join(f"- {topic} (средняя ошибка: {sum(err)/len(err):.2f})" for topic, err in worst_topics)
        prompt = REFLECTION_PROMPT.format(threshold=REFLECTION_ERROR_THRESHOLD, topics=topics_str)

        try:
            raw = await self._call_llm(
                [{"role": "user", "content": prompt}],
                temp=REFLECTION_LLM_TEMP,
                max_tokens=REFLECTION_LLM_MAX_TOKENS
            )
        except Exception as e:
            logger.warning(f"Reflection LLM call failed: {e}")
            return

        result = parse_llm_json(raw)
        if not result:
            logger.warning(f"Reflection: bad JSON from LLM, skipping this cycle: {raw[:200]!r}")
            return

        # Верхние границы на веса — не даём рефлексии "разогнать" один вес
        # в стену за счёт остальных. Значения подобраны так, чтобы ни один
        # компонент не мог перекрыть больше половины итогового скора.
        bounds = {"semantic": 0.6, "graph": 0.35, "freshness": 0.35, "evidence": 0.25, "confidence": 0.20}
        adjustments = result.get("weight_adjustments", {})
        if isinstance(adjustments, dict):
            for key, delta in adjustments.items():
                if key not in self.memory._dynamic_weights or not isinstance(delta, (int, float)):
                    continue
                delta = max(-0.05, min(0.05, float(delta)))
                if abs(delta) < 1e-6:
                    continue
                old_val = self.memory._dynamic_weights[key]
                new_val = max(0.01, min(bounds.get(key, 0.5), old_val + delta))
                self.memory._dynamic_weights[key] = new_val
                logger.info(f"[Reflection] {key}: {old_val:.3f} -> {new_val:.3f} (Δ{delta:+.3f})")

        topics_to_research = result.get("topics_to_research", [])
        if isinstance(topics_to_research, list):
            for topic in topics_to_research[:3]:
                if isinstance(topic, str) and topic.strip():
                    logger.info(f"[Reflection] Auto-research for topic: {topic}")
                    asyncio.create_task(self.research(topic.strip()))

        self.prediction_history.clear()
        self._last_reflection_time = time.time()

    async def _quick_correction(self, query: str, predicted: List[str], actual: str):
        logger.info(f"[QuickCorrection] High error detected for: {query[:50]}...")
        await self.research(query)

    def _content_has_currency_numbers(self, text: str) -> bool:
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

    def _generate_alternative_queries(self, original: str, attempt: int) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if attempt == 1:
            return f"{original} {today}"
        elif attempt == 2:
            return f"курс {original} покупка продажа сегодня"
        elif attempt == 3:
            return f"{original} сайт банки.ру"
        else:
            return f"{original} котировка"

    async def deep_search(self, query: str, max_results: int = MAX_PAGES_TO_FETCH) -> Dict[str, Any]:
        all_sources = []
        all_context_parts = []
        queries_to_try = [query]
        for i in range(1, MAX_SEARCH_ATTEMPTS):
            alt = self._generate_alternative_queries(query, i)
            if alt not in queries_to_try:
                queries_to_try.append(alt)

        fetched_total = 0
        budget = getattr(self, '_deep_search_budget', DEEP_SEARCH_TOTAL_BUDGET)

        for q in queries_to_try:
            if fetched_total >= budget:
                break
            cache_key = _hash_query(q)
            cached = await self.search_cache.get(cache_key)
            if cached:
                if self._content_has_currency_numbers(cached.get("context", "")):
                    return cached
                all_sources.extend(cached.get("sources", []))
                all_context_parts.append(cached.get("context", ""))
                continue

            ddg_results = await self.search_ddg(q, max_results=max_results + 2)
            if not ddg_results:
                continue

            urls = [r["url"] for r in ddg_results if r.get("url")]
            remaining = budget - fetched_total
            to_fetch = urls[:min(max_results, remaining)]
            fetched = await self.web_fetcher.fetch_many(to_fetch, limit=PARALLEL_FETCH_LIMIT)
            fetched_total += len(fetched)

            documents = []
            url_to_title = {r["url"]: r["title"] for r in ddg_results}
            for url, text in fetched:
                if text:
                    full_text = f"{url_to_title.get(url, '')}\n{text}"
                    documents.append({"url": url, "title": url_to_title.get(url, url), "text": full_text})
                else:
                    snippet = next((r["snippet"] for r in ddg_results if r["url"] == url), "")
                    if snippet:
                        documents.append({"url": url, "title": url_to_title.get(url, url), "text": snippet})

            if not documents:
                continue

            all_chunks = []
            for doc in documents:
                chunks = self.chunk_ranker.chunk_text(doc["text"])
                for ch in chunks:
                    all_chunks.append({"chunk": ch, "url": doc["url"], "title": doc["title"]})

            scored = self.chunk_ranker.score_chunks(q, [c["chunk"] for c in all_chunks])
            top_chunks = []
            sources_used = set()
            for score, chunk_text in scored:
                if score < MIN_RELEVANCE_THRESHOLD:
                    continue
                meta = next((c for c in all_chunks if c["chunk"] == chunk_text), None)
                if not meta:
                    continue
                top_chunks.append({"score": round(score, 3), **meta})
                sources_used.add(meta["url"])
                if len(top_chunks) >= max_results * 2:
                    break

            context_parts = []
            for i, ch in enumerate(top_chunks, 1):
                context_parts.append(
                    f"[{i}] Источник: {ch['title']}\nURL: {ch['url']}\nРелевантность: {ch['score']}\n{ch['chunk'][:600]}"
                )
            context = "\n\n---\n\n".join(context_parts)
            sources = [{"title": url_to_title.get(u, u), "url": u} for u in sources_used]

            result_item = {
                "sources": sources,
                "context": context,
                "search_performed": True,
                "chunks_found": len(top_chunks),
            }
            await self.search_cache.set(cache_key, result_item)

            if self._content_has_currency_numbers(context):
                return result_item

            all_sources.extend(sources)
            all_context_parts.append(context)

        combined_context = "\n\n".join(filter(None, all_context_parts))
        seen = set()
        unique_sources = []
        for s in all_sources:
            key = s["url"]
            if key not in seen:
                seen.add(key)
                unique_sources.append(s)

        return {
            "sources": unique_sources,
            "context": combined_context,
            "search_performed": True,
            "chunks_found": len(combined_context) // 500,
        }

    async def _prepare_messages(self, message: str, web_search: bool = False,
                                image_base64: Optional[str] = None,
                                image_mime: Optional[str] = None,
                                reasoning: bool = False) -> Tuple[List[Dict], Dict]:
        auto_search = False
        if AUTO_SEARCH_ENABLED and not web_search and needs_search_heuristic(message):
            web_search = True
            auto_search = True

        search_meta = {"web_search_used": False, "auto_triggered": auto_search, "sources": []}
        search_context = ""
        sources = []

        if web_search and self.searcher:
            # Единый вызов вместо раздельных rewrite_query() + is_factual_query():
            # роутер одновременно переписывает поисковый запрос и определяет,
            # время-чувствительный ли вопрос (влияет на глубину поиска), с учётом
            # контекста диалога и активных целей — не только последнего сообщения.
            # Вызывается ТОЛЬКО когда поиск уже решено делать (auto-heuristic или
            # явный флаг), поэтому не добавляет LLM round-trip к обычным репликам.
            route = await self._route(message)
            search_query = route.get("search_query") or message
            max_res = 7 if route.get("is_factual_time_sensitive") else MAX_PAGES_TO_FETCH
            search_data = await self.deep_search(search_query, max_results=max_res)
            if search_data["search_performed"]:
                search_meta["web_search_used"] = True
                search_meta["sources"] = search_data["sources"]
                sources = search_data["sources"]
                search_context = search_data["context"] or "Поиск выполнен, но полезный текст извлечь не удалось."
            if EXTRACT_FACTS_FROM_SEARCH and search_data.get("context"):
                try:
                    if EXTRACT_FACTS_WITH_LLM:
                        facts = await self._extract_facts_llm(search_data["context"])
                    else:
                        facts = self._extract_facts_from_text(search_data["context"])
                    for f in facts:
                        # Публикуем через AIAdapter в GCN (добавляет в GCN и синхронизирует кэши)
                        self.ai_adapter.publish({
                            "subject": f,
                            "predicate": "is_fact",
                            "object": "true",
                            "type": "claim",
                            "confidence": 0.6
                        })
                    await self.memory._schedule_save()
                    logger.info(f"Extracted {len(facts)} facts from web search")
                except Exception as e:
                    logger.warning(f"Fact extraction error: {e}")

        relevant = await self.router.retrieve(message, top_k=7, include_private=True)
        memory_context = ""
        if relevant:
            lines = []
            for fact in relevant:
                text = fact["text"][:300]
                conf = fact.get("confidence", 0.5)
                lines.append(f"- {text} (уверенность: {conf:.2f}, важность: {fact.get('importance', 1.0):.2f})")
            memory_context = "=== КОНТЕКСТ ИЗ ДОЛГОСРОЧНОЙ ПАМЯТИ ===\n" + "\n".join(lines) + "\n\n"
            self.current_working_memory = [f["text"] for f in relevant[:3]]

        predictions = await self.memory.predict_next(self.current_working_memory) if self.current_working_memory else []

        uncertainty = 0.0
        if relevant:
            avg_conf = sum(f["confidence"] for f in relevant) / len(relevant)
            uncertainty = 1.0 - avg_conf
        if search_context:
            uncertainty *= 0.7

        # Активные цели (из GCN)
        active_goals = [obj for obj in self.memory.store._objects.values()
                        if obj.type == KnowledgeType.HYPOTHESIS and obj.object.get("status") == "active"]
        goal_hint = ""
        if active_goals:
            goal_hint = "Активные цели: " + ", ".join([g.subject for g in active_goals[:2]])

        messages = self._build_messages(
            message=message,
            web_search=web_search,
            search_context=search_context,
            memory_context=memory_context,
            image_base64=image_base64,
            image_mime=image_mime,
            reasoning=reasoning,
            uncertainty=uncertainty,
            predictions=predictions,
            goal_hint=goal_hint
        )

        self._last_prepare_meta = {
            "search_meta": search_meta,
            "sources": sources,
            "memory_context": memory_context,
            "predictions": predictions,
            "uncertainty": uncertainty,
            "active_goals": active_goals,
            "relevant": relevant,
            "message": message,
            "web_search": web_search,
            "reasoning": reasoning,
            "image_base64": image_base64,
            "image_mime": image_mime,
        }
        return messages, search_meta

    async def process_input(self, message: str, web_search: bool = False,
                            image_base64: Optional[str] = None,
                            image_mime: Optional[str] = None,
                            reasoning: bool = False) -> Tuple[str, Dict]:
        cmd_response = await self._handle_memory_command(message)
        if cmd_response:
            return cmd_response[0], cmd_response[1]

        messages, search_meta = await self._prepare_messages(
            message, web_search, image_base64, image_mime, reasoning
        )

        response = await self._call_llm(messages)

        self.history.append({"role": "user", "content": message})
        if response:
            self.history.append({"role": "assistant", "content": response})
        self._save_history()

        if response:
            uncertainty = self._last_prepare_meta.get("uncertainty", 0.5)
            salience = 1.0 - uncertainty
            await self.memory.add_episode(message, response, salience=salience)

            # --- НОВОЕ: обновляем рабочую память ---
            relevant = self._last_prepare_meta.get("relevant", [])
            for fact_dict in relevant[:3]:
                gcn_id = fact_dict.get("gcn_id")
                if gcn_id:
                    self.memory.hierarchy.add_to_working(gcn_id)

        # ---- Рефлексия: запоминаем предсказание и ошибку ----
        predictions = self._last_prepare_meta.get("predictions", [])
        if predictions and response:
            error = self._compute_prediction_error(predictions, response)
            self.prediction_history.append({
                "query": message,
                "predicted": predictions,
                "actual": response,
                "error": error,
                "timestamp": time.time()
            })
            if len(self.prediction_history) > REFLECTION_HISTORY_SIZE:
                self.prediction_history.pop(0)
            if error > 0.85:
                asyncio.create_task(self._quick_correction(message, predictions, response))

        return response, search_meta

    async def _extract_facts_llm(self, text: str) -> List[str]:
        prompt = (
            "Извлеки из текста ниже список коротких, самодостаточных фактических утверждений "
            "(проверяемые факты, а не мнения или вода). Каждый факт — отдельным пунктом, "
            "без нумерации, без пояснений. Если фактов нет — верни пустую строку.\n\n"
            f"ТЕКСТ:\n{text[:4000]}"
        )
        try:
            raw = await self._call_llm([{"role": "user", "content": prompt}], temp=0.2, max_tokens=500)
            if not raw or not raw.strip():
                return []
            facts = []
            for line in raw.split('\n'):
                line = line.strip().strip('-•*').strip()
                if 15 < len(line) < 400:
                    facts.append(line[:300])
            return facts[:20]
        except Exception as e:
            logger.warning(f"LLM fact extraction failed, falling back to regex: {e}")
            return self._extract_facts_from_text(text)

    async def _verify_pending_contradictions(self, max_checks: int = 5):
        """
        Реальная верификация противоречий через LLM + GCN-провенанс.
        Раньше это была заглушка (pass) — вызывалась в каждом цикле
        консолидации, но ничего не делала. _detect_contradictions() в
        memory_graph.py помечает пары фактов грубой эвристикой (наличие
        отрицания + пересечение ключевых слов), давая много ложных
        срабатываний. Здесь эти пары прогоняются через LLM-судью с temp=0.0
        и строгим JSON-выводом, а решение фиксируется в GCN через уже
        существующий, но ранее не задействованный MemoryStore.verify().
        """
        pairs = self.memory.get_unverified_contradictions(limit=max_checks)
        if not pairs:
            return

        actor = f"reflection:{self.user_id}"
        resolved = 0
        for fact_a, fact_b in pairs:
            prompt = CONTRADICTION_VERIFY_PROMPT.format(text_a=fact_a.text, text_b=fact_b.text)
            try:
                raw = await self._call_llm([{"role": "user", "content": prompt}], temp=0.0, max_tokens=150)
            except Exception as e:
                logger.warning(f"Contradiction verify LLM call failed ({fact_a.id},{fact_b.id}): {e}")
                continue

            verdict = parse_llm_json(raw)
            if not verdict or "relation" not in verdict:
                logger.warning(f"Contradiction verify: bad JSON from LLM: {raw[:200]!r}")
                continue

            relation = verdict.get("relation")
            reason = verdict.get("reason", "")

            if relation == "false_positive":
                # Удаляем противоречие из графа GCN
                try:
                    self.memory.store._graph.remove_relation(fact_a.gcn_id, "CONTRADICTS", fact_b.gcn_id)
                    self.memory.store._graph.remove_relation(fact_b.gcn_id, "CONTRADICTS", fact_a.gcn_id)
                    # Обновляем confidence
                    fact_a.confidence = min(1.0, fact_a.confidence + 0.05)
                    fact_b.confidence = min(1.0, fact_b.confidence + 0.05)
                    self.memory.store.update(fact_a.gcn_id, {"confidence": fact_a.confidence}, self.user_id)
                    self.memory.store.update(fact_b.gcn_id, {"confidence": fact_b.confidence}, self.user_id)
                except Exception as e:
                    logger.debug(f"Failed to remove contradiction edges: {e}")
                # Также удаляем из локальных множеств
                fact_a.contradicts.discard(fact_b.id)
                fact_b.contradicts.discard(fact_a.id)
            else:
                keep = verdict.get("keep")
                if keep == "A":
                    self._demote_or_retract(fact_b, actor, reason)
                elif keep == "B":
                    self._demote_or_retract(fact_a, actor, reason)
                elif keep == "neither":
                    self._demote_or_retract(fact_a, actor, reason)
                    self._demote_or_retract(fact_b, actor, reason)
                # keep == "both" (both_partially_true) — оставляем оба как есть,
                # но всё равно фиксируем VERIFY-событие в GCN ниже.
                # Раз мы вынесли явный вердикт — снимаем пару из "необработанных",
                # чтобы не гонять её через LLM повторно каждый цикл.
                fact_a.contradicts.discard(fact_b.id)
                fact_b.contradicts.discard(fact_a.id)

            for fact in (fact_a, fact_b):
                if fact.gcn_id:
                    try:
                        self.memory.store.verify(fact.gcn_id, verifier="llm_reflection",
                                                  status=relation, actor=actor)
                    except Exception as e:
                        logger.debug(f"GCN verify() failed for {fact.gcn_id}: {e}")

            resolved += 1
            self.memory._dirty = True

        if resolved:
            logger.info(f"[ContradictionVerify] Resolved {resolved}/{len(pairs)} pending pairs")
            await self.memory._schedule_save()

    def _demote_or_retract(self, fact: 'Fact', actor: str, reason: str):
        """Понижает доверие к факту; при падении ниже порога — ретрактит в GCN."""
        fact.confidence *= 0.5
        if fact.confidence < 0.15 and fact.gcn_id:
            try:
                self.memory.store.retract(fact.gcn_id, actor, reason=f"contradiction: {reason}"[:200])
                logger.info(f"[ContradictionVerify] Retracted fact {fact.id}: {reason}")
            except Exception as e:
                logger.debug(f"Retract failed for {fact.gcn_id}: {e}")

    def _extract_facts_from_text(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]', text)
        facts = []
        for s in sentences:
            s = s.strip()
            if len(s) > 30 and re.search(r'\b(?:является|составляет|равен|находится|имеет|будет|был|стал)\b', s):
                facts.append(s[:300])
        return facts[:20]

    async def _handle_memory_command(self, message: str) -> Optional[Tuple[str, Dict]]:
        lower_msg = message.lower()
        for cmd, action in MEMORY_CONTROL_COMMANDS.items():
            if lower_msg.startswith(cmd):
                rest = message[len(cmd):].strip()
                if not rest:
                    continue

                # ------------------------------------------------------------
                # 1. Команда "запомни" – сохраняет факт и генерирует ответ через LLM
                # ------------------------------------------------------------
                if action == "store":
                    fid = self.memory._add_fact(rest, 'command', confidence=1.0, importance=1.5)
                    # --- НОВОЕ: добавить в рабочую память ---
                    fact = self.memory.facts_by_id.get(fid)
                    if fact and fact.gcn_id:
                        self.memory.hierarchy.add_to_working(fact.gcn_id)
                    await self.memory._schedule_save()

                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "Ты — AI-ассистент с когнитивной памятью. Пользователь попросил запомнить информацию. "
                                "Подтверди, что ты запомнил, кратко и естественно, возможно, с уточнением или перефразировкой, "
                                "чтобы показать понимание."
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Запомни: {rest}"
                        }
                    ]
                    response = await self._call_llm(messages, temp=0.5, max_tokens=150)
                    if response:
                        return response, {"memory": "stored", "id": fid}
                    else:
                        return f"Запомнил: {rest}", {"memory": "stored", "id": fid}

                # ------------------------------------------------------------
                # 2. Команда "забудь" – удаляет факты и генерирует ответ через LLM
                # ------------------------------------------------------------
                elif action == "forget":
                    to_remove_ids = {f.id for f in self.memory.semantic_facts if rest.lower() in f.text.lower()}
                    if to_remove_ids:
                        removed = self.memory._remove_facts(to_remove_ids)
                        await self.memory._schedule_save()

                        messages = [
                            {
                                "role": "system",
                                "content": (
                                    "Ты — AI-ассистент с когнитивной памятью. Пользователь попросил забыть информацию. "
                                    "Подтверди, что ты удалил соответствующие факты, кратко и естественно."
                                )
                            },
                            {
                                "role": "user",
                                "content": f"Забудь: {rest} (удалено {removed} фактов)"
                            }
                        ]
                        response = await self._call_llm(messages, temp=0.5, max_tokens=150)
                        if response:
                            return response, {"memory": "forgot", "count": removed}
                        else:
                            return f"Удалено {removed} фактов о '{rest}'", {"memory": "forgot"}
                    else:
                        return "Ничего не найдено для удаления.", {"memory": "no_match"}

                # ------------------------------------------------------------
                # 3. Команда "что ты знаешь о" – улучшенная версия с LLM
                # ------------------------------------------------------------
                elif action == "recall":
                    facts = await self.memory.retrieve_hybrid(rest, top_k=7, use_graph=True)

                    if not facts:
                        return "Ничего не найдено по вашему запросу.", {"memory": "no_recall"}

                    context_lines = []
                    for f in facts[:5]:
                        context_lines.append(f"- {f['text']}")
                    context = "\n".join(context_lines)

                    messages = [
                        {
                            "role": "system",
                            "content": (
                                "Ты — AI-ассистент с когнитивной памятью. На основе предоставленных фактов дай связный, "
                                "естественный ответ на русском языке. Не перечисляй факты списком, а объедини их в единое "
                                "объяснение. Если фактов недостаточно или они не относятся к вопросу, честно скажи об этом."
                            )
                        },
                        {
                            "role": "user",
                            "content": f"Вопрос: {rest}\n\nФакты из памяти:\n{context}"
                        }
                    ]

                    response = await self._call_llm(messages, temp=0.6, max_tokens=500)

                    if not response:
                        answer = "Вот что я знаю:\n" + "\n".join(
                            f"- {f['text']} (уверенность: {f.get('confidence', 0.5):.2f})" for f in facts[:5]
                        )
                        return answer, {"memory": "recalled_fallback"}

                    return response, {"memory": "recalled"}

        return None

    def _build_messages(self, message: str, web_search: bool, search_context: str,
                        memory_context: str, image_base64: Optional[str],
                        image_mime: Optional[str], reasoning: bool,
                        uncertainty: float, predictions: List[str], goal_hint: str) -> List[Dict]:
        system_parts = [
            "Ты — AI-ассистент с когнитивной памятью и доступом к интернету.",
            "Используй предоставленный контекст из памяти и результаты поиска."
        ]
        if uncertainty > 0.6:
            system_parts.append(f"Твоя уверенность в ответе низкая ({uncertainty:.2f}). Если не знаешь – скажи об этом.")
        if predictions:
            system_parts.append(f"Возможное продолжение темы: {', '.join(predictions[:3])}.")
        if goal_hint:
            system_parts.append(f"Учитывай активные цели: {goal_hint}.")
        if reasoning:
            system_parts.append("Перед ответом покажи рассуждения: начни с 💭 РАССУЖДЕНИЕ: и заканчивая ---, затем финальный ответ.")
        if web_search:
            system_parts.append("Ты выполнил поиск в интернете, используй полученные данные как основной источник фактов.")

        system_content = "\n\n".join(system_parts)
        messages = [{"role": "system", "content": system_content}]

        for item in self.history[-self.max_history:]:
            if item.get("role") != "system":
                messages.append(item)

        user_blocks = []
        if memory_context:
            user_blocks.append(memory_context)
        if search_context:
            user_blocks.append(
                f"=== ДАННЫЕ ИЗ ИНТЕРНЕТА (актуальны на {datetime.now(timezone.utc).strftime('%Y-%m-%d')}) ===\n\n"
                f"{search_context}\n\n=== КОНЕЦ ДАННЫХ ==="
            )
        user_blocks.append(f"Вопрос пользователя: {message}")
        user_text = "\n\n".join(user_blocks)

        if image_base64 and LM_STUDIO_VISION_SUPPORTED:
            if not image_base64.startswith("data:image"):
                if image_mime:
                    image_url = f"data:{image_mime};base64,{image_base64}"
                else:
                    image_url = f"data:image/png;base64,{image_base64}"
            else:
                image_url = image_base64
            user_content = [{"type": "text", "text": user_text}, {"type": "image_url", "image_url": {"url": image_url}}]
            messages.append({"role": "user", "content": user_content})
        else:
            messages.append({"role": "user", "content": user_text})

        return messages

    async def research(self, goal: str) -> Dict[str, Any]:
        prompt = f"Сформулируй 3 гипотезы по вопросу: {goal}"
        hypotheses_text = await self._call_llm([{"role": "user", "content": prompt}], temp=0.8)
        hypotheses = [h.strip("-• ").strip() for h in hypotheses_text.split('\n') if h.strip()][:3]
        if not hypotheses:
            hypotheses = ["Не удалось сгенерировать гипотезы"]

        all_evidence = []
        queries = [goal] + hypotheses[:2]
        for q in queries:
            try:
                data = await self.deep_search(q, max_results=3)
                for src in data.get("sources", []):
                    all_evidence.append({"source": src.get("url", ""), "title": src.get("title", ""), "query": q})
            except Exception as e:
                logger.debug(f"Research search error for '{q}': {e}")

        evidence_text = "\n".join([f"- {e['title']}: {e['source']} (запрос: {e['query']})" for e in all_evidence[:6]])
        context = f"Вопрос: {goal}\nГипотезы: {', '.join(hypotheses)}\nИсточники:\n{evidence_text}"

        answer_prompt = (
            f"На основе гипотез и источников дай развёрнутый ответ. "
            f"Укажи уверенность (0-1) и аргументы.\n\n{context}"
        )
        answer = await self._call_llm([{"role": "user", "content": answer_prompt}], temp=0.6)
        return {"answer": answer, "confidence": 0.7, "hypotheses": hypotheses, "evidence": all_evidence}

    async def get_response(self, message: str, web_search: bool = False,
                           image_base64: str = None, image_mime: str = None,
                           reasoning: bool = False):
        return await self.process_input(message, web_search, image_base64, image_mime, reasoning)

    async def stream_response(self, message: str, web_search: bool = False,
                              image_base64: str = None, image_mime: str = None,
                              reasoning: bool = False, char_by_char: bool = None):
        cmd_response = await self._handle_memory_command(message)
        if cmd_response:
            yield f"data: {json.dumps({'token': cmd_response[0]})}\n\n"
            yield "data: [DONE]\n\n"
            return

        messages, search_meta = await self._prepare_messages(
            message, web_search, image_base64, image_mime, reasoning
        )

        if search_meta.get("sources"):
            yield f"data: {json.dumps({'sources': search_meta['sources']})}\n\n"

        full_response = ""
        try:
            if LM_STUDIO_USE_STREAM:
                async for token in self._call_llm_stream(messages):
                    full_response += token
                    yield f"data: {json.dumps({'token': token})}\n\n"
                    await asyncio.sleep(0)
            else:
                response = await self._call_llm(messages)
                full_response = response
                if char_by_char is None:
                    char_by_char = STREAM_CHAR_BY_CHAR
                if char_by_char:
                    for ch in response:
                        yield f"data: {json.dumps({'token': ch})}\n\n"
                        await asyncio.sleep(STREAM_CHAR_DELAY)
                else:
                    for word in response.split():
                        yield f"data: {json.dumps({'token': word + ' '})}\n\n"
        except Exception as e:
            logger.error(f"Stream error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            return

        self.history.append({"role": "user", "content": message})
        if full_response:
            self.history.append({"role": "assistant", "content": full_response})
            self._save_history()
            uncertainty = self._last_prepare_meta.get("uncertainty", 0.5)
            salience = 1.0 - uncertainty
            await self.memory.add_episode(message, full_response, salience=salience)

            # Обновление целей (из GCN)
            active_goals = self._last_prepare_meta.get("active_goals", [])
            for goal_obj in active_goals:
                if goal_obj.subject.lower() in full_response.lower():
                    goal_obj.confidence = min(1.0, goal_obj.confidence + 0.1)
                    if goal_obj.confidence >= 0.9:
                        new_obj = goal_obj.object.copy() if isinstance(goal_obj.object, dict) else {}
                        new_obj["status"] = "completed"
                        self.memory.store.update(goal_obj.id, {"object": new_obj, "confidence": goal_obj.confidence},
                                                 self.user_id)
                    else:
                        self.memory.store.update(goal_obj.id, {"confidence": goal_obj.confidence}, self.user_id)
                    self.memory._sync_goal_from_gcn(goal_obj.id)
            await self.memory._schedule_save()

            # --- НОВОЕ: обновляем рабочую память (аналогично process_input) ---
            relevant = self._last_prepare_meta.get("relevant", [])
            for fact_dict in relevant[:3]:
                gcn_id = fact_dict.get("gcn_id")
                if gcn_id:
                    self.memory.hierarchy.add_to_working(gcn_id)

        # Рефлексия: запоминаем предсказание и ошибку
        predictions = self._last_prepare_meta.get("predictions", [])
        if predictions and full_response:
            error = self._compute_prediction_error(predictions, full_response)
            self.prediction_history.append({
                "query": message,
                "predicted": predictions,
                "actual": full_response,
                "error": error,
                "timestamp": time.time()
            })
            if len(self.prediction_history) > REFLECTION_HISTORY_SIZE:
                self.prediction_history.pop(0)
            if error > 0.85:
                asyncio.create_task(self._quick_correction(message, predictions, full_response))

        yield "data: [DONE]\n\n"

    async def enhance_prompt(self, prompt: str) -> str:
        enhancement = await self._call_llm([
            {"role": "system", "content": "Ты — эксперт по улучшению промптов. Добавь детали, стиль, освещение, сохрани суть."},
            {"role": "user", "content": prompt}
        ], temp=0.9)
        return enhancement.strip() if enhancement else prompt

    async def generate_image(self, prompt: str) -> Optional[str]:
        if not EASYDIFFUSION_ENABLED:
            return None
        try:
            async with aiohttp.ClientSession() as session:
                payload = {
                    "prompt": prompt,
                    "steps": EASYDIFFUSION_DEFAULT_STEPS,
                    "width": EASYDIFFUSION_DEFAULT_WIDTH,
                    "height": EASYDIFFUSION_DEFAULT_HEIGHT,
                }
                async with session.post(f"{EASYDIFFUSION_URL}/generate", json=payload,
                                        timeout=aiohttp.ClientTimeout(total=EASYDIFFUSION_TIMEOUT)) as resp:
                    if resp.status == 200:
                        data = await resp.json()
                        return data.get("image_base64")
                    else:
                        logger.error(f"EasyDiffusion error: {resp.status}")
                        return None
        except Exception as e:
            logger.error(f"Image generation failed: {e}")
            return None

    def get_stats(self):
        memory_stats = self.memory.get_stats() if hasattr(self, 'memory') else {}
        return {
            "history_len": len(self.history),
            "max_history": self.max_history,
            "current_working_memory": len(self.current_working_memory),
            "last_prediction_error": self.last_prediction_error,
            "working_memory": len(self.memory.hierarchy.working_memory),
            **memory_stats
        }

    async def shutdown(self):
        await self.web_fetcher.close()
        if self._consolidation_task:
            self._consolidation_task.cancel()
        if self._planner_task:
            self._planner_task.cancel()
        if self._research_task:
            self._research_task.cancel()
        if self._reflection_task:
            self._reflection_task.cancel()
        await self.memory.shutdown()


# =====================================================================
# Фабрика ассистентов
# =====================================================================
_assistants: Dict[str, CognitiveController] = {}
_assistants_lock = asyncio.Lock()


async def get_assistant(user_id: str):
    async with _assistants_lock:
        if user_id not in _assistants:
            _assistants[user_id] = CognitiveController(user_id)
            logger.info(f"Создан когнитивный ассистент для {user_id[:16]}")
        return _assistants[user_id]


# =====================================================================
# FastAPI роутер
# =====================================================================
router = APIRouter(prefix='/ai', tags=['ai'])


class AIRequest(BaseModel):
    message: str = Field(..., min_length=MIN_MESSAGE_LENGTH, max_length=MAX_MESSAGE_LENGTH)
    stream: bool = True
    web_search: bool = False
    image_base64: Optional[str] = Field(None, max_length=MAX_IMAGE_SIZE_BASE64 * 2)
    image_mime: Optional[str] = None
    reasoning: bool = False
    char_by_char: Optional[bool] = None


class ResearchRequest(BaseModel):
    goal: str = Field(..., min_length=1, max_length=2000)


class ImageGenRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=2000)


class EnhanceRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=5000)


@router.post("/chat")
async def chat_with_ai(body: AIRequest, address: str = Depends(require_auth)):
    logger.info(f"Запрос от {address[:16]}, web={body.web_search}, reasoning={body.reasoning}")
    assistant = await get_assistant(address)
    if body.stream:
        return StreamingResponse(
            assistant.stream_response(
                message=body.message,
                web_search=body.web_search,
                image_base64=body.image_base64,
                image_mime=body.image_mime,
                reasoning=body.reasoning,
                char_by_char=body.char_by_char
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache,no-store,must-revalidate", "X-Accel-Buffering": "no"}
        )
    response, meta = await assistant.get_response(
        message=body.message,
        web_search=body.web_search,
        image_base64=body.image_base64,
        image_mime=body.image_mime,
        reasoning=body.reasoning
    )
    return {"reply": response, "meta": meta}


@router.post("/search")
async def direct_search(body: dict, address: str = Depends(require_auth)):
    query = body.get("query", "").strip()
    if not query:
        return {"error": "query required"}
    assistant = await get_assistant(address)
    result = await assistant.deep_search(query, max_results=5)
    return {"type": "search", "query": query, **result}


@router.post("/research")
async def research_endpoint(body: ResearchRequest, address: str = Depends(require_auth)):
    assistant = await get_assistant(address)
    try:
        result = await assistant.research(body.goal)
        return result
    except Exception as e:
        logger.error(f"Research failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/generate_image")
async def generate_image_endpoint(body: ImageGenRequest, address: str = Depends(require_auth)):
    assistant = await get_assistant(address)
    try:
        image_b64 = await assistant.generate_image(body.prompt)
        if image_b64:
            return {"image_base64": image_b64}
        else:
            raise HTTPException(status_code=503, detail="Image generation failed")
    except Exception as e:
        logger.error(f"Image gen failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/enhance_prompt")
async def enhance_prompt_endpoint(body: EnhanceRequest, address: str = Depends(require_auth)):
    assistant = await get_assistant(address)
    try:
        enhanced = await assistant.enhance_prompt(body.prompt)
        return {"enhanced": enhanced}
    except Exception as e:
        logger.error(f"Enhance prompt failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/global_stats")
async def global_stats(address: str = Depends(require_auth)):
    assistant = await get_assistant(address)
    return assistant.get_stats()


@router.post("/force_merge")
async def force_merge(address: str = Depends(require_auth)):
    return {"status": "no-op", "message": "Global merge disabled"}


@router.post("/apply_global")
async def apply_global(address: str = Depends(require_auth)):
    return {"status": "no-op", "message": "Global apply disabled"}


def start_global_merge_task():
    logger.info("Global merge task disabled")


async def shutdown_all():
    for uid, assistant in _assistants.items():
        try:
            await assistant.shutdown()
        except Exception:
            pass


import atexit


def _shutdown():
    try:
        loop = asyncio.get_event_loop()
        if loop.is_closed():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
        loop.run_until_complete(shutdown_all())
    except Exception as e:
        logger.warning(f"Shutdown error: {e}")


atexit.register(_shutdown)