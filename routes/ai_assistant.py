"""
Когнитивный ассистент с интеграцией CognitiveMemory, планированием, автономностью.
Рефакторинг: вынесены общие утилиты в GCN.llm_client и GCN.web_search.
Добавлены: классификация намерений, автоматическое извлечение фактов, улучшенные команды.
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
from GCN.mcp_client_manager import MCPToolManager

from GCN.GCN import AIAdapter, KnowledgeObject, KnowledgeType, MemoryScope
from GCN.memory_graph import CognitiveMemory, Fact, Episode, Goal, GCNMemoryRouter

from GCN.llm_client import call_llm, call_llm_raw, call_llm_stream
from GCN.web_search import deep_search, fetch_url
from GCN.image_utils import enhance_prompt, generate_image
from GCN.tool_router import ToolRegistry, ToolRouter, build_tool_trace_context

from GCN.config_ai import *

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


# =====================================================================
# 1. Умный триггер поиска (оставлен)
# =====================================================================
SEARCH_TRIGGER_KEYWORDS = [
    # Раньше список включал 'рецепт', 'инструкция', 'пошагово', 'сравнение',
    # 'обзор', 'анализ', 'докажи', 'проверь', 'правда ли', 'как делается' —
    # это форсировало веб-поиск ДО того, как LLM-роутер (_route/ROUTER_PROMPT)
    # успевал решить сам. При этом сам ROUTER_PROMPT в качестве примера явно
    # учит модель НЕ искать по запросу "Как приготовить борщ?" — то есть
    # эвристика и роутер прямо противоречили друг другу, и эвристика всегда
    # побеждала первой, вызывая лишние поисковые запросы на обычные
    # инструкции/рецепты/сравнения, не требующие свежих данных из интернета.
    # Оставлены только маркеры, для которых нужны действительно свежие/точные
    # данные (даты, курсы, новости, актуальные события).
    'сегодня', 'сейчас', 'новости', 'курс', 'погода', 'свежие',
    'последние', 'завтра', 'найди', 'поищи', 'актуальные',
    '2024', '2025', '2026', 'сколько стоит', 'какой сейчас', 'последние данные',
    'статистика', 'результаты', 'кто победил', 'когда выйдет',
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
# 2. Промпты для строгого JSON (без изменений)
# =====================================================================
ROUTER_PROMPT = """Ты — модуль планирования когнитивного ассистента. Проанализируй запрос пользователя и контекст.
Верни ТОЛЬКО валидный JSON, без пояснений, без markdown-разметки, без ```.

Примеры правильных ответов:
- Запрос: "Курс доллара сегодня" -> {{"needs_web_search": true, "search_query": "курс доллара сегодня", "is_factual_time_sensitive": true, "answer_strategy": "search_then_answer"}}
- Запрос: "Что такое теория относительности?" -> {{"needs_web_search": false, "search_query": null, "is_factual_time_sensitive": false, "answer_strategy": "recall_then_answer"}}
- Запрос: "Как приготовить борщ?" -> {{"needs_web_search": false, "search_query": null, "is_factual_time_sensitive": false, "answer_strategy": "direct"}}

Правила:
- needs_web_search=true, если для точного ответа нужны свежие/актуальные/числовые данные (курсы, цены, новости, даты, "сейчас", "сегодня"), которых нет в истории диалога.
- search_query — короткий запрос для поисковика (3-10 слов), а не сам вопрос пользователя дословно.
- is_factual_time_sensitive=true для вопросов с числами, единицами измерения, курсами, датами, текущими событиями.
- answer_strategy="clarify" только если вопрос пользователя действительно неоднозначен настолько, что угадать намерение нельзя.

Последние реплики диалога:
{history_tail}

Активные цели пользователя: {goals}

Запрос пользователя: {message}
"""

REFLECTION_PROMPT = """Ты — модуль саморефлексии когнитивного ассистента. Ниже темы, где предсказания модели чаще всего ошибались (ошибка > {threshold}).
Верни ТОЛЬКО валидный JSON, без пояснений, без markdown-разметки, без ```.

Пример корректного ответа:
{{
  "weight_adjustments": {{"semantic": 0.02, "graph": 0.0, "freshness": -0.01, "evidence": 0.0, "confidence": 0.0}},
  "topics_to_research": ["квантовая физика", "нейросети"],
  "propose_concepts": ["Разница между обучением с учителем и без учителя часто путается из-за смешения терминов"]
}}

Каждое значение в weight_adjustments — дельта в диапазоне [-0.05, 0.05] (0, если менять не нужно).
Если ошибки вызваны нехваткой знаний, укажи соответствующие темы в topics_to_research (максимум 3).
Если ошибки вызваны не нехваткой фактов, а тем, что связанные факты не складываются в понятное обобщение
(модель "видит" факты, но не понимает общей идеи) — сформулируй в propose_concepts (максимум 2) короткое
обобщающее утверждение, которое стоило бы явно сохранить в памяти как концепт.

Темы с ошибками:
{topics}
"""

CONTRADICTION_VERIFY_PROMPT = """Ты — верификатор фактов в системе памяти AI-ассистента. Даны два утверждения, помеченные как противоречащие друг другу.
Верни ТОЛЬКО валидный JSON, без пояснений, без markdown-разметки, без ```.

Примеры:
- A: "Вода кипит при 100°C", B: "Вода кипит при 80°C" -> {{"relation": "true_contradiction", "keep": "B", "reason": "Температура кипения зависит от давления, но при нормальных условиях 100°C, поэтому B неверно."}}
- A: "Эйнштейн родился в 1879", B: "Эйнштейн родился в 1879 году" -> {{"relation": "false_positive", "keep": "both", "reason": "Оба утверждения идентичны."}}
- A: "Кофе полезен", B: "Кофе вреден" -> {{"relation": "both_partially_true", "keep": "both", "reason": "Влияние кофе зависит от дозировки и индивидуальных особенностей."}}

Варианты relation:
- "true_contradiction" — утверждения действительно противоречат друг другу.
- "false_positive" — на самом деле не противоречат (разные объекты, время, или случайное совпадение ключевых слов).
- "both_partially_true" — оба верны в своём контексте, keep="both".

Утверждение A: {text_a}
Утверждение B: {text_b}
"""


def parse_llm_json(raw: str) -> Optional[Dict]:
    """Безопасный парсинг JSON из ответа LLM."""
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
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return None
    return None

# ========== НОВЫЙ МОДУЛЬ: классификация намерений ==========
INTENT_CLASSIFICATION_PROMPT = """Проанализируй сообщение пользователя и определи, относится ли оно к управлению памятью.
Если да, укажи команду и извлеки сущности.

Возможные команды:
- "store" — запомнить факт (пользователь хочет, чтобы ты запомнил информацию). Может быть уточнение "глобально" -> scope="global".
- "forget" — забыть факт (удалить информацию).
- "recall" — вспомнить информацию по теме.
- "none" — обычный вопрос, не связанный с управлением памятью.

Также извлеки "content" — текст, который нужно запомнить/забыть/или тему для поиска.
Если в сообщении есть "глобально" или "global" и команда "store", установи scope="global".

Ответь ТОЛЬКО валидным JSON:
{{"intent": "store|forget|recall|none", "content": "извлечённый текст или пустая строка", "scope": "private|shared|global", "confidence": 0.0-1.0}}

Примеры:
- "Запомни, что мой любимый цвет синий" -> {{"intent": "store", "content": "мой любимый цвет синий", "scope": "private", "confidence": 0.95}}
- "Запомни глобально, что Земля круглая" -> {{"intent": "store", "content": "Земля круглая", "scope": "global", "confidence": 0.95}}
- "Забудь всё о погоде" -> {{"intent": "forget", "content": "погода", "scope": "private", "confidence": 0.9}}
- "Что ты знаешь о Питоне?" -> {{"intent": "recall", "content": "Питон", "scope": "private", "confidence": 0.95}}
- "Как дела?" -> {{"intent": "none", "content": "", "scope": "private", "confidence": 1.0}}

Сообщение: {message}
"""

async def classify_intent(message: str) -> Dict:
    """Определяет намерение пользователя с помощью LLM."""
    if not ENABLE_INTENT_CLASSIFICATION:
        # fallback на старую логику
        return {"intent": "none", "content": "", "confidence": 1.0}
    try:
        prompt = INTENT_CLASSIFICATION_PROMPT.format(message=message)
        raw = await call_llm([{"role": "user", "content": prompt}], temp=0.0, max_tokens=150)
        result = parse_llm_json(raw)
        if result and "intent" in result:
            return result
    except Exception as e:
        logger.debug(f"Intent classification failed: {e}")
    return {"intent": "none", "content": "", "confidence": 0.0}

# =====================================================================
# 3. КОГНИТИВНЫЙ КОНТРОЛЛЕР (изменён)
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

        self.router = GCNMemoryRouter(user_id, MEMORY_BASE_DIR)
        self.router.set_llm_caller(call_llm)

        self.memory = self.router.private_memory

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

        self._consolidation_task = None
        self._planner_task = None
        self._research_task = None
        self._reflection_task = None
        self._idle_task = None
        self._start_background_tasks()

        self.current_working_memory: List[str] = []
        self.current_goals: List[Goal] = []
        self.last_prediction_error = 0.0
        self._last_prepare_meta: Dict = {}

        self.prediction_history: List[Dict] = []
        self.reflection_interval = REFLECTION_INTERVAL
        self._last_reflection_time = time.time()

        self.mcp_manager = MCPToolManager()
        self._mcp_task = asyncio.create_task(self.mcp_manager.initialize())

        # ===== Единый реестр инструментов (internal + внешние MCP) =====
        # Раньше внешние MCP-инструменты (mcp_manager) были единственным источником
        # тулов у браузерного чата, а собственные возможности памяти (recall/remember/
        # add_goal) были доступны только снаружи — через mcp_server_blockcoin.py.
        # Из-за этого чат в браузере был "слепее", чем внешний MCP-клиент, даже для
        # своих же данных. Теперь оба набора инструментов регистрируются в одном
        # ToolRegistry с теми же именами/семантикой, что и в mcp_server_blockcoin.py.
        self.tool_registry = ToolRegistry()
        self._register_internal_tools()
        self.tool_router = ToolRouter(
            registry=self.tool_registry,
            llm_raw_caller=call_llm_raw,
            llm_text_caller=call_llm,
        )
        self._external_tools_registered = False

        # Для отслеживания бездействия
        self._last_activity_time = time.time()
        self._idle_consolidation_done = False

        logger.info(f"CognitiveController (GCN) initialized for {user_id[:16]}")

    def _register_internal_tools(self):
        """
        Регистрирует те же операции над памятью, что уже отдаются наружу через
        mcp_server_blockcoin.py (recall/remember/add_goal), напрямую на self.router —
        без круга через самодельный JSON-формат mcp_call. Схемы аргументов совпадают
        с Pydantic Field-описаниями в MCP-сервере, чтобы поведение не расходилось.
        """

        async def _recall(args: Dict) -> Dict:
            query = args.get("query", "")
            top_k = int(args.get("top_k", 5))
            results = await self.router.retrieve(query, top_k=top_k, include_private=True)
            return {
                "results": [
                    {"text": r.get("text", ""), "confidence": r.get("confidence", 0.0),
                     "scope": r.get("scope", "private")}
                    for r in results[:top_k]
                ]
            }

        async def _remember(args: Dict) -> Dict:
            fact = args.get("fact", "")
            if not fact:
                return {"status": "error", "message": "fact обязателен"}
            scope_str = (args.get("scope") or "private").lower()
            scope_enum = {"private": MemoryScope.PRIVATE, "shared": MemoryScope.SHARED,
                          "global": MemoryScope.GLOBAL}.get(scope_str, MemoryScope.PRIVATE)
            obj_id = self.router.add_knowledge(
                subject=fact, predicate="is_fact", obj="true",
                scope=scope_enum, confidence=0.7, author=self.router.user_id,
                source_type="tool_call",
            )
            await self.memory._schedule_save()
            return {"status": "ok", "id": obj_id, "fact": fact}

        async def _add_goal(args: Dict) -> Dict:
            description = args.get("description", "")
            priority = float(args.get("priority", 0.5))
            if not description:
                return {"status": "error", "message": "description обязателен"}
            gid = await self.memory.add_goal(description, priority)
            return {"status": "ok", "id": gid, "description": description}

        async def _web_search(args: Dict) -> Dict:
            query = args.get("query", "")
            data = await deep_search(query, max_results=5)
            return {"context": data.get("context", ""), "sources": data.get("sources", [])}

        async def _fetch_url(args: Dict) -> Dict:
            url = args.get("url", "")
            if not url:
                return {"status": "error", "message": "url обязателен"}
            return await fetch_url(url)

        async def _forget(args: Dict) -> Dict:
            query = args.get("query", "")
            scope_str = (args.get("scope") or "private").lower()
            scope_map = {"private": self.router.private_memory, "shared": self.router.shared_memory,
                         "global": self.router.global_memory}
            memory = scope_map.get(scope_str)
            if memory is None or not query:
                return {"status": "error", "message": "нужен query и корректный scope"}
            memory.reload_if_stale()
            to_remove = [f.id for f in memory.semantic_facts if query.lower() in f.text.lower()]
            if not to_remove:
                return {"status": "ok", "removed": 0, "scope": scope_str, "message": "Ничего не найдено."}
            removed = memory._remove_facts(set(to_remove))
            await memory._schedule_save()
            return {"status": "ok", "removed": removed, "scope": scope_str}

        async def _semantic_search(args: Dict) -> Dict:
            query = args.get("query", "")
            top_k = int(args.get("top_k", 5))
            memory = self.router.private_memory
            memory.reload_if_stale()
            emb = memory.embed_text(query)
            if emb is None:
                return {"error": "Эмбеддинги недоступны."}
            results = memory.store.semantic_search(emb, top_k=top_k * 2)
            return {"results": [
                {"text": memory.store.get(gcn_id).subject if memory.store.get(gcn_id) else "", "score": score}
                for gcn_id, score in results[:top_k]
            ]}

        async def _get_episodes(args: Dict) -> Dict:
            limit = int(args.get("limit", 5))
            memory = self.router.private_memory
            memory.reload_if_stale()
            episodes = memory.episodic_memory[-limit:] if memory.episodic_memory else []
            return {"episodes": [
                {"user": ep.user_msg, "assistant": ep.assistant_msg, "timestamp": ep.timestamp}
                for ep in reversed(episodes)
            ], "count": len(episodes)}

        async def _get_contradictions(args: Dict) -> Dict:
            limit = int(args.get("limit", 5))
            memory = self.router.private_memory
            memory.reload_if_stale()
            pairs = memory.get_unverified_contradictions(limit=limit)
            return {"contradictions": [
                {"a": {"text": a.text, "confidence": a.confidence, "id": a.id},
                 "b": {"text": b.text, "confidence": b.confidence, "id": b.id}}
                for a, b in pairs
            ], "count": len(pairs)}

        async def _get_goals(args: Dict) -> Dict:
            memory = self.router.private_memory
            memory.reload_if_stale()
            goals = await memory.get_active_goals()
            return {"goals": [
                {"description": g.description, "priority": g.priority,
                 "confidence": g.confidence, "status": g.status}
                for g in goals
            ], "count": len(goals)}

        async def _graph_explore(args: Dict) -> Dict:
            seed_text = args.get("seed_text", "")
            depth = int(args.get("depth", 2))
            memory = self.router.private_memory
            memory.reload_if_stale()
            seed_ids = [f.id for f in memory.semantic_facts if seed_text.lower() in f.text.lower()]
            if not seed_ids:
                return {"error": f"Факты с '{seed_text}' не найдены."}
            activation = await memory.spread_activation(seed_ids[:3], max_depth=min(depth, 3))
            sorted_items = sorted(activation.items(), key=lambda x: x[1], reverse=True)
            return {"nodes": [
                {"id": fid, "text": memory.facts_by_id.get(fid).text[:200] if memory.facts_by_id.get(fid) else "",
                 "activation": act}
                for fid, act in sorted_items[:20] if fid not in seed_ids
            ]}

        async def _explain_fact(args: Dict) -> Dict:
            gcn_id = args.get("gcn_id", "")
            self.router.refresh()
            obj = (self.router.private_memory.store.get(gcn_id) or
                   self.router.shared_memory.store.get(gcn_id) or
                   self.router.global_memory.store.get(gcn_id))
            if not obj:
                return {"error": f"Объект {gcn_id} не найден ни в одном слое памяти."}
            store = (self.router.private_memory.store if self.router.private_memory.store.get(gcn_id) else
                     self.router.shared_memory.store if self.router.shared_memory.store.get(gcn_id) else
                     self.router.global_memory.store)
            contradictions = store._graph.get_neighbors(gcn_id, "CONTRADICTS")
            grounds_in = store._graph.get_neighbors(gcn_id, "GROUNDS_IN")
            abstracts_from = store._graph.get_neighbors(gcn_id, "ABSTRACTS_FROM")
            confirming_authors = [e.split("author:", 1)[1] for e in obj.evidence if e.startswith("author:")]
            return {
                "id": obj.id, "type": obj.type.value, "scope": obj.scope.value, "text": obj.subject,
                "confidence": obj.confidence, "version": obj.version, "author": obj.author,
                "source_type": obj.source_type, "created": obj.created.isoformat(),
                "confirming_authors": confirming_authors,
                "contradicts": [t for _, t in contradictions],
                "grounds_in_global": [t for _, t in grounds_in],
                "abstracted_from": [t for _, t in abstracts_from],
            }

        async def _get_memory_stats(args: Dict) -> Dict:
            memory = self.router.private_memory
            memory.reload_if_stale()
            return memory.get_stats()

        async def _resolve_contradiction(args: Dict) -> Dict:
            fact_id_a = args.get("fact_id_a")
            fact_id_b = args.get("fact_id_b")
            verdict = (args.get("verdict") or "").lower()
            memory = self.router.private_memory
            memory.reload_if_stale()

            def find_fact(fid):
                if fid in memory.facts_by_id:
                    return memory.facts_by_id[fid]
                for f in memory.semantic_facts:
                    if f.gcn_id == fid:
                        return f
                return None

            fa, fb = find_fact(fact_id_a), find_fact(fact_id_b)
            if not fa or not fb:
                return {"status": "error", "message": f"Факты не найдены: A={fact_id_a}, B={fact_id_b}"}
            if verdict == "a":
                memory.gcn_store._graph.remove_relation(fa.gcn_id, "CONTRADICTS", fb.gcn_id)
                memory.gcn_store._graph.remove_relation(fb.gcn_id, "CONTRADICTS", fa.gcn_id)
                fa.contradicts.discard(fb.id)
                fb.contradicts.discard(fa.id)
                memory._remove_facts({fb.id})
                await memory._schedule_save()
                return {"status": "ok", "verdict": "a", "kept": fa.text, "removed": fb.text}
            elif verdict == "b":
                memory.gcn_store._graph.remove_relation(fa.gcn_id, "CONTRADICTS", fb.gcn_id)
                memory.gcn_store._graph.remove_relation(fb.gcn_id, "CONTRADICTS", fa.gcn_id)
                fa.contradicts.discard(fb.id)
                fb.contradicts.discard(fa.id)
                memory._remove_facts({fa.id})
                await memory._schedule_save()
                return {"status": "ok", "verdict": "b", "kept": fb.text, "removed": fa.text}
            elif verdict == "both":
                memory.gcn_store._graph.remove_relation(fa.gcn_id, "CONTRADICTS", fb.gcn_id)
                memory.gcn_store._graph.remove_relation(fb.gcn_id, "CONTRADICTS", fa.gcn_id)
                fa.contradicts.discard(fb.id)
                fb.contradicts.discard(fa.id)
                await memory._schedule_save()
                return {"status": "ok", "verdict": "both", "message": "Противоречие снято, оба сохранены."}
            elif verdict == "neither":
                memory._remove_facts({fa.id, fb.id})
                await memory._schedule_save()
                return {"status": "ok", "verdict": "neither", "message": "Оба удалены."}
            return {"status": "error", "message": f"Неизвестный вердикт: {verdict}"}

        async def _generate_image_tool(args: Dict) -> Dict:
            prompt = args.get("prompt", "")
            if not prompt:
                return {"status": "error", "message": "prompt обязателен"}
            enhance = args.get("enhance_prompt", True)
            final_prompt = await self.enhance_prompt(prompt) if enhance else prompt

            image_b64 = await self.generate_image(
                final_prompt,
                steps=args.get("steps"), width=args.get("width"), height=args.get("height"),
                cfg_scale=args.get("cfg_scale"), seed=args.get("seed"), sampler_name=args.get("sampler_name"),
            )
            if not image_b64:
                return {"status": "error", "message": "Не удалось сгенерировать изображение"}

            # --- СОХРАНЕНИЕ НА ДИСК ---
            from GCN.config_ai import GENERATED_IMAGES_DIR
            import base64, os
            from datetime import datetime

            output_dir = GENERATED_IMAGES_DIR
            output_dir.mkdir(parents=True, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = output_dir / f"image_{timestamp}.png"
            with open(filename, "wb") as f:
                f.write(base64.b64decode(image_b64))
            file_path = str(filename.absolute())

            # ✅ Используем переменную окружения или относительный путь
            BASE_URL = os.getenv("SERVER_BASE_URL", "")
            if not BASE_URL:
                # fallback – если не задано, используем относительный путь
                image_url = f"/generated_images/{filename.name}"
            else:
                image_url = f"{BASE_URL}/generated_images/{filename.name}"

            logger.info(f"Image saved: {file_path}, URL: {image_url}")  # <-- ДОБАВЛЕНО

            return {
                "status": "ok",
                "image_url": image_url,
                "file_path": file_path,
                "original_prompt": prompt,
                "enhanced_prompt": final_prompt,
                "message": f"Изображение сгенерировано: {image_url}"
            }

        async def _research_topic(args: Dict) -> Dict:
            topic = args.get("topic", "")
            if not topic:
                return {"status": "error", "message": "topic обязателен"}
            result = await self.research(topic)
            return {
                "topic": topic,
                "hypotheses": result.get("hypotheses", []),
                "evidence": result.get("evidence", []),
                "answer": result.get("answer", ""),
                "confidence": result.get("confidence", 0.0),
            }

        self.tool_registry.register(
            name="recall",
            description="Поиск в памяти пользователя (личная/общая/глобальная) по теме или вопросу.",
            parameters={"type": "object",
                        "properties": {"query": {"type": "string", "description": "Тема или вопрос для поиска"},
                                       "top_k": {"type": "integer", "description": "Сколько результатов вернуть"}},
                        "required": ["query"]},
            handler=_recall,
        )
        self.tool_registry.register(
            name="remember",
            description="Сохранить факт в память пользователя.",
            parameters={"type": "object",
                        "properties": {"fact": {"type": "string", "description": "Текст факта для сохранения"},
                                       "scope": {"type": "string", "enum": ["private", "shared", "global"],
                                                 "description": "Скоуп памяти"}},
                        "required": ["fact"]},
            handler=_remember,
        )
        self.tool_registry.register(
            name="add_goal",
            description="Добавить новую активную цель пользователя.",
            parameters={"type": "object",
                        "properties": {"description": {"type": "string"},
                                       "priority": {"type": "number", "description": "0..1"}},
                        "required": ["description"]},
            handler=_add_goal,
        )
        self.tool_registry.register(
            name="web_search",
            description="Найти актуальную информацию в интернете (курсы, новости, факты после обучения модели).",
            parameters={"type": "object",
                        "properties": {"query": {"type": "string", "description": "Поисковый запрос"}},
                        "required": ["query"]},
            handler=_web_search,
        )
        self.tool_registry.register(
            name="fetch_url",
            description=("Прочитать содержимое конкретной ссылки целиком (например, файл на GitHub, "
                          "документацию, статью), а не искать её текстом в поисковике. Используй, когда "
                          "пользователь прислал прямой URL и просит прочитать/проанализировать содержимое."),
            parameters={"type": "object",
                        "properties": {"url": {"type": "string", "description": "Прямая ссылка (https://...)"}},
                        "required": ["url"]},
            handler=_fetch_url,
        )
        # ===== Инструменты для паритета с mcp_server_blockcoin.py =====
        # Раньше эти операции были доступны только внешнему MCP-клиенту (через
        # mcp_server_blockcoin.py), а браузерный чат из коробки не мог ни
        # забыть факт по запросу, ни сделать чисто векторный поиск, ни
        # посмотреть противоречия/цели/происхождение факта/статистику памяти —
        # то самое "работает не как в MCP", про которое сообщал пользователь.
        self.tool_registry.register(
            name="forget",
            description="Удалить факты, содержащие заданные ключевые слова, из указанного слоя памяти.",
            parameters={"type": "object",
                        "properties": {"query": {"type": "string", "description": "Ключевые слова для удаления"},
                                       "scope": {"type": "string", "enum": ["private", "shared", "global"]}},
                        "required": ["query"]},
            handler=_forget,
        )
        self.tool_registry.register(
            name="semantic_search",
            description="Векторный поиск по смыслу в личной памяти (по эмбеддингам, а не по ключевым словам).",
            parameters={"type": "object",
                        "properties": {"query": {"type": "string"}, "top_k": {"type": "integer"}},
                        "required": ["query"]},
            handler=_semantic_search,
        )
        self.tool_registry.register(
            name="get_episodes",
            description="Вернуть последние диалоги (эпизоды) из личной памяти.",
            parameters={"type": "object", "properties": {"limit": {"type": "integer"}}},
            handler=_get_episodes,
        )
        self.tool_registry.register(
            name="get_contradictions",
            description="Вернуть неразрешённые противоречия в личной памяти.",
            parameters={"type": "object", "properties": {"limit": {"type": "integer"}}},
            handler=_get_contradictions,
        )
        self.tool_registry.register(
            name="get_goals",
            description="Вернуть активные цели пользователя.",
            parameters={"type": "object", "properties": {}},
            handler=_get_goals,
        )
        self.tool_registry.register(
            name="graph_explore",
            description="Исследовать граф ассоциативной памяти, начиная с фактов, содержащих seed_text.",
            parameters={"type": "object",
                        "properties": {"seed_text": {"type": "string"}, "depth": {"type": "integer"}},
                        "required": ["seed_text"]},
            handler=_graph_explore,
        )
        self.tool_registry.register(
            name="explain_fact",
            description=("Объяснить происхождение и статус факта памяти по его gcn_id: кто и когда создал, "
                         "с чем противоречит — используй после recall/semantic_search, когда нужно "
                         "обосновать уверенность в факте."),
            parameters={"type": "object",
                        "properties": {"gcn_id": {"type": "string"}},
                        "required": ["gcn_id"]},
            handler=_explain_fact,
        )
        self.tool_registry.register(
            name="get_memory_stats",
            description="Вернуть статистику по личной памяти пользователя.",
            parameters={"type": "object", "properties": {}},
            handler=_get_memory_stats,
        )
        self.tool_registry.register(
            name="resolve_contradiction",
            description=("Разрешить противоречие между двумя фактами (см. get_contradictions): "
                          "verdict='a'/'b' оставляет один факт и удаляет другой, 'both' снимает "
                          "противоречие сохранив оба, 'neither' удаляет оба."),
            parameters={"type": "object",
                        "properties": {"fact_id_a": {"type": "string"}, "fact_id_b": {"type": "string"},
                                       "verdict": {"type": "string", "enum": ["a", "b", "both", "neither"]}},
                        "required": ["fact_id_a", "fact_id_b", "verdict"]},
            handler=_resolve_contradiction,
        )
        self.tool_registry.register(
            name="generate_image",
            description="Сгенерировать изображение по текстовому описанию через локальный EasyDiffusion.",
            parameters={"type": "object",
                        "properties": {
                            "prompt": {"type": "string", "description": "Описание изображения"},
                            "enhance_prompt": {"type": "boolean", "description": "Улучшить промпт через LLM"},
                            "steps": {"type": "integer"}, "width": {"type": "integer"}, "height": {"type": "integer"},
                            "cfg_scale": {"type": "number"}, "seed": {"type": "integer"},
                            "sampler_name": {"type": "string"},
                        },
                        "required": ["prompt"]},
            handler=_generate_image_tool,
        )
        self.tool_registry.register(
            name="research_topic",
            description="Глубокое исследование темы с генерацией гипотез и сбором доказательств из поиска.",
            parameters={"type": "object",
                        "properties": {"topic": {"type": "string"}},
                        "required": ["topic"]},
            handler=_research_topic,
        )

    async def _ensure_external_tools_registered(self):
        """Внешние MCP-серверы инициализируются асинхронно (self._mcp_task) — их
        инструменты добавляем в общий реестр, как только они станут доступны.
        Дополнительно на каждый вызов просим менеджер повторить попытку для
        серверов, которые не поднялись при старте (ensure_connected сам
        ограничивает частоту попыток) — раньше сервер, временно недоступный в
        момент старта процесса, оставался недоступен до перезапуска, даже если
        поднимался через минуту после этого. register_mcp_tools просто
        перезаписывает записи по тем же именам, так что повторные вызовы (в т.ч.
        после реконнекта) безопасны и подхватывают новые инструменты."""
        if not self.mcp_manager._initialized:
            return
        await self.mcp_manager.ensure_connected()
        self.tool_registry.register_mcp_tools(self.mcp_manager, self._handle_mcp_call)
        self._external_tools_registered = True

    async def _handle_mcp_call(self, server: str, tool: str, args: Dict) -> str:
        """Выполняет вызов внешнего MCP-инструмента (используется ToolRegistry как handler)."""
        try:
            result = await self.mcp_manager.call_tool(server, tool, args)
            return result
        except Exception as e:
            logger.error(f"MCP call error: {e}", exc_info=True)
            return f"Ошибка вызова MCP: {str(e)}"

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
            self._idle_task = asyncio.create_task(self._idle_consolidation())

    # ----- Фоновые задачи (добавлена консолидация по бездействию) -----
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

    async def _idle_consolidation(self):
        """Запускает лёгкую консолидацию при бездействии."""
        while True:
            await asyncio.sleep(60)  # проверяем каждую минуту
            if time.time() - self._last_activity_time > IDLE_CONSOLIDATION_DELAY:
                if not self._idle_consolidation_done:
                    logger.info("Idle consolidation triggered")
                    try:
                        await self.memory.light_consolidation()
                        self._idle_consolidation_done = True
                    except Exception as e:
                        logger.error(f"Idle consolidation error: {e}")
            else:
                self._idle_consolidation_done = False

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
        history_tail = "\n".join(
            f"{item['role'].capitalize()}: {item['content'][:200]}"
            for item in self.history[-4:]
        ) if self.history else "(диалог только начался)"

        active_goals = [obj for obj in self.memory.store._objects.values()
                        if obj.type == KnowledgeType.HYPOTHESIS and obj.object.get("status") == "active"]
        goals_str = "; ".join(g.subject for g in active_goals[:3]) or "нет"

        prompt = ROUTER_PROMPT.format(history_tail=history_tail, goals=goals_str, message=message)
        try:
            raw = await call_llm([{"role": "user", "content": prompt}], temp=0.0, max_tokens=200)
            result = parse_llm_json(raw)
            if result and "search_query" in result:
                return result
            logger.warning(f"Router: bad/incomplete JSON, falling back to heuristics: {raw[:200]!r}")
        except Exception as e:
            logger.warning(f"Router LLM call failed, falling back to heuristics: {e}")

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
            f"На основе диалогов с пользователем сформулируй 1-3 долгосрочные цели, которые могут быть полезны для ассистента. "
            "Цели должны быть конкретными, измеримыми и достижимыми. Например: 'Изучить тему X', 'Научиться делать Y', 'Собрать информацию о Z'. "
            "Ответь в виде списка целей (каждая с новой строки), без дополнительных пояснений.\n\n"
            f"Диалоги:\n{history_summary}"
        )
        try:
            goals_text = await call_llm([{"role": "user", "content": prompt}], temp=0.7, max_tokens=200)
            goals = [g.strip("-• ").strip() for g in goals_text.split('\n') if g.strip()]
            for g in goals:
                await self.memory.add_goal(g, priority=0.5)
            await self.memory._schedule_save()
            logger.info(f"[Planner] Generated goals: {goals}")
        except Exception as e:
            logger.error(f"Planning error: {e}")

    async def _auto_research(self):
        active_goals = [obj for obj in self.memory.store._objects.values()
                        if obj.type == KnowledgeType.HYPOTHESIS and obj.object.get("status") == "active"]
        for goal_obj in active_goals:
            if goal_obj.confidence < 0.5:
                logger.info(f"Auto-research triggered for goal: {goal_obj.subject}")
                await self.research(goal_obj.subject)
                goal_obj.confidence = min(1.0, goal_obj.confidence + 0.2)
                self.memory.store.update(goal_obj.id, {"confidence": goal_obj.confidence}, self.user_id)
                self.memory._sync_goal_from_gcn(goal_obj.id)
        await self.memory._schedule_save()

    # ===== РЕФЛЕКСИЯ =====
    def _compute_prediction_error(self, predicted: List[str], actual: str) -> float:
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
            raw = await call_llm(
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

        # НОВОЕ: сохраняем предложенные рефлексией концепты как низкоуверенные
        # CONCEPT-узлы личной памяти пользователя (не глобальной — это гипотеза
        # саморефлексии по конкретным ошибкам этого пользователя, её ещё
        # предстоит подтвердить обычной консолидацией/form_concepts).
        proposed_concepts = result.get("propose_concepts", [])
        if isinstance(proposed_concepts, list):
            for concept_text in proposed_concepts[:2]:
                if not (isinstance(concept_text, str) and 10 < len(concept_text.strip()) < 400):
                    continue
                try:
                    concept_id = f"concept_{uuid.uuid4()}"
                    concept_obj = KnowledgeObject(
                        id=concept_id,
                        type=KnowledgeType.CONCEPT,
                        subject=concept_text.strip(),
                        predicate="abstracts",
                        object={"source": "reflection"},
                        author=f"reflection:{self.user_id}",
                        created=datetime.now(timezone.utc),
                        confidence=0.4,  # низкая — это гипотеза саморефлексии, а не подтверждённое обобщение
                        scope=MemoryScope.PRIVATE,
                        source_type="reflection_hypothesis",
                    )
                    self.memory.store.create(concept_obj, actor=f"reflection:{self.user_id}")
                    emb = self.memory.embed_text(concept_text.strip())
                    if emb is not None:
                        self.memory.store.set_embedding(concept_id, emb)
                    logger.info(f"[Reflection] Proposed concept saved: {concept_text[:80]}")
                except Exception as e:
                    logger.debug(f"Failed to save proposed concept: {e}")

        self.prediction_history.clear()
        self._last_reflection_time = time.time()

    async def _quick_correction(self, query: str, predicted: List[str], actual: str):
        logger.info(f"[QuickCorrection] High error detected for: {query[:50]}...")
        await self.research(query)

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

    # ===== НОВЫЙ МЕТОД: автоматическое извлечение фактов из сообщения =====
    async def _auto_extract_facts(self, message: str) -> List[str]:
        """Извлекает факты из сообщения пользователя и сохраняет через роутер (глобально)."""
        if not AUTO_EXTRACT_FACTS:
            return []
        # Пропускаем, если сообщение является командой (чтобы не дублировать)
        if any(message.lower().startswith(cmd) for cmd in MEMORY_CONTROL_COMMANDS.keys()):
            return []
        prompt = (
            "Извлеки из сообщения пользователя объективные факты, которые могут быть полезны для запоминания. "
            "Факты должны быть краткими утверждениями, содержащими конкретную информацию. "
            "Игнорируй мнения, команды, вопросы, приветствия. "
            "Если фактов нет, верни пустой ответ. "
            "Каждый факт с новой строки, без нумерации.\n\n"
            f"Сообщение: {message}"
        )
        try:
            raw = await call_llm([{"role": "user", "content": prompt}], temp=0.2, max_tokens=200)
        except Exception as e:
            logger.debug(f"Auto-extract LLM call failed: {e}")
            return []
        if not raw:
            return []
        lines = [line.strip().strip('-•*').strip() for line in raw.split('\n') if line.strip()]
        facts = []
        for line in lines:
            if not (20 < len(line) < 400):
                continue
            if line[0].lower() in ('я', 'ты', 'мы', 'давайте', 'попробуйте'):
                continue
            if not re.search(r'(является|составляет|равен|находится|имеет|был|стал|\d)', line):
                continue
            facts.append(line[:300])
        # Сохраняем через роутер в глобальную память
        for fact in facts[:3]:
            gcn_id = self.router.add_knowledge(
                subject=fact,
                predicate="is_fact",
                obj="true",
                scope=MemoryScope.GLOBAL,
                confidence=AUTO_EXTRACT_CONFIDENCE,
                author=self.user_id,
                source_type="auto_extraction"
            )
            if gcn_id:
                self.memory.hierarchy.add_to_working(gcn_id)
        if facts:
            await self.router.global_memory._schedule_save()
        return facts

    # ===== ОСНОВНАЯ ЛОГИКА ПОДГОТОВКИ СООБЩЕНИЙ =====
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
            # Если пользователь прислал прямую ссылку, не даём LLM-роутеру
            # переписать её в вольный поисковый запрос (легко теряет URL при
            # перефразировании) — читаем ссылку напрямую, это и есть намерение
            # пользователя ("прочитай эту страницу/файл").
            direct_urls_in_message = re.findall(r'https?://\S+', message)
            if direct_urls_in_message:
                search_query = message
                route = {"is_factual_time_sensitive": False}
            else:
                route = await self._route(message)
                search_query = route.get("search_query") or message
            max_res = 7 if route.get("is_factual_time_sensitive") else MAX_PAGES_TO_FETCH
            search_data = await deep_search(search_query, max_results=max_res)
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

                    GLOBAL_FACT_CONFIDENCE = 0.75

                    for f in facts:
                        self.router.add_knowledge(
                            subject=f,
                            predicate="is_fact",
                            obj="true",
                            scope=MemoryScope.GLOBAL,
                            confidence=GLOBAL_FACT_CONFIDENCE,
                            author=self.user_id,
                            source_type="web_search",
                        )
                    await self.router.global_memory._schedule_save()
                    logger.info(f"Extracted {len(facts)} facts from web search -> global memory")
                except Exception as e:
                    logger.warning(f"Fact extraction error: {e}")

        relevant = await self.router.retrieve(message, top_k=7, include_private=True)
        memory_context = ""
        # ИСПРАВЛЕНИЕ (причина №2 — "путаница" памяти в браузерном чате, которой
        # нет в MCP-режиме): раньше блок "=== КОНТЕКСТ ИЗ ДОЛГОСРОЧНОЙ ПАМЯТИ ==="
        # строился из ВСЕХ до 7 результатов router.retrieve() безусловно — retrieve
        # всегда что-то возвращает (ближайшие соседи по эмбеддингам есть почти
        # всегда), даже когда ни один факт реально не относится к вопросу
        # ("нарисуй кота", "привет" и т.п.). Модель получала слабо релевантные или
        # вовсе случайные факты в блоке, который система же промптом называет
        # надёжным источником ("самый надёжный источник"), и путала их с реальным
        # ответом. В MCP-режиме этого не происходит: там recall — это ЯВНЫЙ
        # инструмент, который внешний клиент вызывает только когда сам решил, что
        # нужна память, а не то, что подмешивается в КАЖДОЕ сообщение. Отсекаем
        # низкорелевантные результаты здесь же, не трогая сам `relevant` — он
        # по-прежнему используется ниже для predict_next/uncertainty/working memory.
        context_facts = [f for f in relevant if f.get("_score", f.get("score", 0.0)) >= MEMORY_CONTEXT_MIN_SCORE]
        if context_facts:
            type_labels = {
                "claim": "факт", "concept": "концепт (обобщение)",
                "episode": "прошлый диалог", "goal": "цель",
            }
            scope_labels = {
                "private": "личный",
                "shared": "общий",
                "global": "глобальный"
            }
            lines = []
            for fact in context_facts:
                text = fact["text"][:300]
                conf = fact.get("confidence", 0.5)
                label = type_labels.get(fact.get("type"), "факт")
                scope = fact.get("scope", "private")  # если поле отсутствует, считаем private
                scope_label = scope_labels.get(scope, scope)
                lines.append(
                    f"- [{scope_label} {label}] {text} (уверенность: {conf:.2f}, важность: {fact.get('importance', 1.0):.2f})"
                )

            # УДАЛЁН ВТОРОЙ ЦИКЛ (который перезаписывал lines без scope)

            # НОВОЕ: явно перечисляем непогашенные противоречия
            contradiction_lines = []
            seen_pairs = set()
            for fact in context_facts:
                gcn_id = fact.get("gcn_id")
                if not gcn_id:
                    continue
                for store in (self.router.private_memory.store, self.router.shared_memory.store,
                              self.router.global_memory.store):
                    obj = store.get(gcn_id)
                    if not obj:
                        continue
                    for relation, other_id in store._graph.get_neighbors(gcn_id, "CONTRADICTS"):
                        pair_key = tuple(sorted((gcn_id, other_id)))
                        if pair_key in seen_pairs:
                            continue
                        other_obj = (self.router.private_memory.store.get(other_id) or
                                     self.router.shared_memory.store.get(other_id) or
                                     self.router.global_memory.store.get(other_id))
                        if other_obj:
                            seen_pairs.add(pair_key)
                            contradiction_lines.append(
                                f"- «{obj.subject[:150]}» ПРОТИВОРЕЧИТ «{other_obj.subject[:150]}»")
                    break

            memory_context = "=== КОНТЕКСТ ИЗ ДОЛГОСРОЧНОЙ ПАМЯТИ ===\n" + "\n".join(lines) + "\n\n"
            if contradiction_lines:
                memory_context += ("=== НЕРАЗРЕШЁННЫЕ ПРОТИВОРЕЧИЯ В ПАМЯТИ ===\n"
                                   + "\n".join(contradiction_lines) + "\n\n")
            self.current_working_memory = [f["text"] for f in relevant[:3]]

        predictions = await self.memory.predict_next(self.current_working_memory) if self.current_working_memory else []

        uncertainty = 0.0
        if relevant:
            avg_conf = sum(f["confidence"] for f in relevant) / len(relevant)
            uncertainty = 1.0 - avg_conf
        if search_context:
            uncertainty *= 0.7

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

        # search_meta изначально не содержало "context" — из-за этого при
        # пересборке сообщений после срабатывания ToolRouter (см. process_input/
        # stream_response, search_meta.get("context", "")) уже найденный веб-контент
        # молча терялся и подменялся пустой строкой. Сохраняем его явно в обоих
        # местах, откуда потом читают.
        search_meta["context"] = search_context
        self._last_prepare_meta = {
            "search_meta": search_meta,
            "search_context": search_context,
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


    # ===== ПРОЦЕССИНГ ВХОДА (изменён: добавлена классификация и автоизвлечение) =====
    async def process_input(self, message: str, web_search: bool = False,
                            image_base64: Optional[str] = None,
                            image_mime: Optional[str] = None,
                            reasoning: bool = False) -> Tuple[str, Dict]:
        # Подтягиваем изменения, сделанные другими процессами (например, MCP)
        self.router.refresh(include_private=True)

        # Обновляем время активности
        self._last_activity_time = time.time()

        # 1. Сначала проверяем команды памяти (старый способ для совместимости)
        cmd_response = await self._handle_memory_command(message)
        if cmd_response:
            return cmd_response[0], cmd_response[1]

        # 2. Новая классификация намерений (использует улучшенный промпт)
        if ENABLE_INTENT_CLASSIFICATION:
            intent_data = await classify_intent(message)
            intent = intent_data.get("intent", "none")
            content = intent_data.get("content", "")
            confidence = intent_data.get("confidence", 0.0)
            scope_str = intent_data.get("scope", "private")
            scope_enum = {"private": MemoryScope.PRIVATE, "shared": MemoryScope.SHARED,
                          "global": MemoryScope.GLOBAL}.get(scope_str, MemoryScope.PRIVATE)

            if intent == "store" and content and confidence > 0.6:
                # Сохраняем через роутер (с инжекцией, дедупликацией)
                gcn_id = self.router.add_knowledge(
                    subject=content,
                    predicate="is_fact",
                    obj="true",
                    scope=scope_enum,
                    confidence=0.9,
                    author=self.user_id,
                    source_type="auto_classified"
                )
                if gcn_id:
                    self.memory.hierarchy.add_to_working(gcn_id)
                await self.router.private_memory._schedule_save() if scope_enum == MemoryScope.PRIVATE else await self.router.global_memory._schedule_save()
                return f"✅ Запомнил ({scope_enum.value}): {content}", {"memory": "auto_stored", "id": gcn_id}

            elif intent == "forget" and content and confidence > 0.6:
                # Удаляем из private памяти (можно расширить на другие слои)
                to_remove = {f.id for f in self.memory.semantic_facts if content.lower() in f.text.lower()}
                if to_remove:
                    removed = self.memory._remove_facts(to_remove)
                    await self.memory._schedule_save()
                    return f"✅ Удалено {removed} фактов по запросу '{content}'", {"memory": "auto_forgot"}
                else:
                    return f"Ничего не найдено для удаления по '{content}'", {"memory": "no_match"}

            elif intent == "recall" and content and confidence > 0.6:
                # Используем router.retrieve() для поиска по всем слоям
                facts = await self.router.retrieve(content, top_k=7, include_private=True)
                if not facts:
                    return "Ничего не найдено.", {"memory": "no_recall"}

                scope_labels = {"private": "личный", "shared": "общий", "global": "глобальный"}
                context_lines = []
                for f in facts[:5]:
                    scope = f.get("scope", "private")
                    scope_label = scope_labels.get(scope, scope)
                    context_lines.append(f"- [{scope_label}] {f['text']}")
                context = "\n".join(context_lines)

                messages = [
                    {"role": "system", "content": "Ты — ассистент. На основе фактов дай связный ответ."},
                    {"role": "user", "content": f"Вопрос: {content}\n\nФакты:\n{context}"}
                ]
                response = await call_llm(messages, temp=0.6, max_tokens=500)
                if not response:
                    response = "Вот что я знаю:\n" + context
                return response, {"memory": "recalled"}

        # 3. Автоматическое извлечение фактов (даже без команды)
        if AUTO_EXTRACT_FACTS and not any(cmd in message.lower() for cmd in MEMORY_CONTROL_COMMANDS.keys()):
            extracted = await self._auto_extract_facts(message)
            if extracted:
                for fact in extracted:
                    gcn_id = self.router.add_knowledge(
                        subject=fact,
                        predicate="is_fact",
                        obj="true",
                        scope=MemoryScope.GLOBAL,
                        confidence=AUTO_EXTRACT_CONFIDENCE,
                        author=self.user_id,
                        source_type="auto_extraction"
                    )
                    if gcn_id:
                        self.memory.hierarchy.add_to_working(gcn_id)
                await self.router.global_memory._schedule_save()
                logger.info(f"Auto-extracted {len(extracted)} facts from message")

        # 4. Обычная обработка (инструменты, если нужны, потом ответ)
        await self._ensure_external_tools_registered()

        messages, search_meta = await self._prepare_messages(
            message, web_search, image_base64, image_mime, reasoning
        )

        # РЕШЕНИЕ (нужен ли инструмент) через ToolRouter
        history_tail = "\n".join(
            f"{m.get('role')}: {str(m.get('content'))[:200]}" for m in self.history[-6:]
        )
        tool_run = await self.tool_router.run(message, messages, history_tail=history_tail)
        tool_trace = tool_run.get("tool_trace", [])

        if tool_trace:
            for t in tool_trace:
                self.history.append({
                    "role": "assistant",
                    "content": f"[инструмент {t['tool']}] {str(t['result'])[:500]}"
                })
            messages = self._build_messages(
                message=message,
                web_search=web_search,
                search_context=search_meta.get("context", ""),
                memory_context=self._memory_context_for_rebuild(
                    tool_trace, self._last_prepare_meta.get("memory_context", "")
                ),
                image_base64=image_base64,
                image_mime=image_mime,
                reasoning=reasoning,
                uncertainty=self._last_prepare_meta.get("uncertainty", 0.5),
                predictions=self._last_prepare_meta.get("predictions", []),
                goal_hint=self._last_prepare_meta.get("goal_hint", "")
            )
            messages.append({
                "role": "user",
                "content": build_tool_trace_context(tool_trace) + "\n\nТеперь дай финальный ответ пользователю."
            })

        response = await call_llm(messages)

        self.history.append({"role": "user", "content": message})
        if response:
            self.history.append({"role": "assistant", "content": response})
        self._save_history()

        if response:
            uncertainty = self._last_prepare_meta.get("uncertainty", 0.5)
            salience = 1.0 - uncertainty
            await self.memory.add_episode(message, response, salience=salience)

            relevant = self._last_prepare_meta.get("relevant", [])
            for fact_dict in relevant[:3]:
                gcn_id = fact_dict.get("gcn_id")
                if gcn_id:
                    self.memory.hierarchy.add_to_working(gcn_id)

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

    # ===== ИЗВЛЕЧЕНИЕ ФАКТОВ (без изменений) =====
    async def _extract_facts_llm(self, text: str) -> List[str]:
        """Извлекает факты из текста и сохраняет их через роутер в глобальную память."""
        prompt = (
            "Извлеки из текста только объективные, проверяемые факты. "
            "Факт должен быть кратким утверждением, содержащим конкретную информацию "
            "(числа, даты, имена, определения). "
            "НЕ включай: мнения, прогнозы, инструкции, общие фразы. "
            "Каждый факт — отдельное предложение. "
            "Верни только факты, каждый с новой строки, без нумерации.\n\n"
            f"ТЕКСТ:\n{text[:4000]}"
        )
        try:
            raw = await call_llm(
                [{"role": "user", "content": prompt}],
                temp=0.2,
                max_tokens=300
            )
        except Exception as e:
            logger.warning(f"LLM fact extraction call failed: {e}")
            return []

        if not raw:
            return []

        facts = []
        for line in raw.split('\n'):
            line = line.strip().strip('-•*').strip()
            if not (20 < len(line) < 400):
                continue
            if line[0].lower() in ('я', 'ты', 'мы', 'давайте', 'попробуйте'):
                continue
            if not re.search(r'(является|составляет|равен|находится|имеет|был|стал|\d)', line):
                continue
            facts.append(line[:300])

        # Сохраняем через роутер
        for fact in facts[:10]:
            gcn_id = self.router.add_knowledge(
                subject=fact,
                predicate="is_fact",
                obj="true",
                scope=MemoryScope.GLOBAL,
                confidence=GLOBAL_FACT_CONFIDENCE_THRESHOLD,  # из config_ai
                author=self.user_id,
                source_type="web_search_extraction"
            )
            if gcn_id:
                self.memory.hierarchy.add_to_working(gcn_id)
        if facts:
            await self.router.global_memory._schedule_save()
        return facts

    def _extract_facts_from_text(self, text: str) -> List[str]:
        sentences = re.split(r'[.!?]', text)
        facts = []
        for s in sentences:
            s = s.strip()
            if len(s) > 30 and re.search(r'\b(?:является|составляет|равен|находится|имеет|будет|был|стал)\b', s):
                facts.append(s[:300])
        return facts[:20]

    # ===== КОМАНДЫ ПАМЯТИ (расширенный список команд) =====
    async def _handle_memory_command(self, message: str) -> Optional[Tuple[str, Dict]]:
        lower_msg = message.lower()
        for cmd, action in MEMORY_CONTROL_COMMANDS.items():
            if lower_msg.startswith(cmd):
                rest = message[len(cmd):].strip()
                if not rest:
                    continue

                if action == "store":
                    # Определяем скоуп (как в MCP)
                    is_global = any(w in rest.lower() for w in ("глобально", "global"))
                    scope = MemoryScope.GLOBAL if is_global else MemoryScope.PRIVATE

                    # Очищаем текст от флагов "глобально"/"global"
                    clean_rest = rest
                    for word in ("глобально", "global"):
                        clean_rest = clean_rest.replace(word, "").strip()
                    clean_rest = " ".join(clean_rest.split())

                    # Если есть возможность, улучшим факт через LLM (как раньше)
                    try:
                        prompt = (
                            "Если факт структурирован и содержит достаточно информации по теме, запомни как есть, полный текст. Иначе извлеки из запроса пользователя объективный факт (утверждение, которое может быть проверено или использовано как знание). "
                            "Игнорируй мнения, временные события, эмоции, инструкции и пожелания. "
                            "Сформулируй факт как краткое предложение в настоящем времени (или прошедшем, если это не теряет актуальности). "
                            "Ответь только фактом, без пояснений. Или полным текстом, если факт структурирован.\n\n"
                            f"Запрос: {clean_rest}"
                        )
                        fact_text = await call_llm([{"role": "user", "content": prompt}], temp=0.3, max_tokens=150)
                        fact_text = fact_text.strip()
                        if len(fact_text) < 5:
                            fact_text = clean_rest
                    except Exception:
                        fact_text = clean_rest

                    # Запись через роутер (с инжекцией, дедупликацией и т.д.)
                    gcn_id = self.router.add_knowledge(
                        subject=fact_text,
                        predicate="is_fact",
                        obj="true",
                        scope=scope,
                        confidence=1.0 if scope == MemoryScope.GLOBAL else 0.9,
                        author=self.user_id,
                        source_type="user_command"
                    )

                    # Добавляем в рабочую память
                    if gcn_id:
                        self.memory.hierarchy.add_to_working(gcn_id)

                    # Сохраняем соответствующий слой
                    if scope == MemoryScope.GLOBAL:
                        await self.router.global_memory._schedule_save()
                    elif scope == MemoryScope.SHARED:
                        await self.router.shared_memory._schedule_save()
                    else:
                        await self.router.private_memory._schedule_save()

                    # Формируем ответ (можно через LLM для красоты)
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
                            "content": f"Запомни: {fact_text} (скоуп: {scope.value})"
                        }
                    ]
                    response = await call_llm(messages, temp=0.5, max_tokens=150)
                    if response:
                        return response, {"memory": "stored", "scope": scope.value, "id": gcn_id}
                    else:
                        return f"Запомнил ({scope.value}): {fact_text}", {"memory": "stored", "scope": scope.value,
                                                                          "id": gcn_id}

                elif action == "forget":
                    # Оставляем существующую логику (удаление из private памяти)
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
                        response = await call_llm(messages, temp=0.5, max_tokens=150)
                        if response:
                            return response, {"memory": "forgot", "count": removed}
                        else:
                            return f"Удалено {removed} фактов о '{rest}'", {"memory": "forgot"}
                    else:
                        return "Ничего не найдено для удаления.", {"memory": "no_match"}

                elif action == "recall":
                    # Используем router.retrieve() для поиска по всем слоям
                    facts = await self.router.retrieve(rest, top_k=7, include_private=True)

                    if not facts:
                        return "Ничего не найдено по вашему запросу.", {"memory": "no_recall"}

                    scope_labels = {"private": "личный", "shared": "общий", "global": "глобальный"}
                    context_lines = []
                    for f in facts[:5]:
                        scope = f.get("scope", "private")
                        scope_label = scope_labels.get(scope, scope)
                        context_lines.append(
                            f"- [{scope_label}] {f['text']} (уверенность: {f.get('confidence', 0.5):.2f})")
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

                    response = await call_llm(messages, temp=0.6, max_tokens=500)
                    if not response:
                        answer = "Вот что я знаю:\n" + "\n".join(
                            f"- {f['text']} (уверенность: {f.get('confidence', 0.5):.2f})" for f in facts[:5]
                        )
                        return answer, {"memory": "recalled_fallback"}
                    return response, {"memory": "recalled"}

        return None

    # ===== ВЕРИФИКАЦИЯ ПРОТИВОРЕЧИЙ (без изменений) =====
    async def _verify_pending_contradictions(self, max_checks: int = 5):
        pairs = self.memory.get_unverified_contradictions(limit=max_checks)
        if not pairs:
            return

        actor = f"reflection:{self.user_id}"
        resolved = 0
        for fact_a, fact_b in pairs:
            prompt = CONTRADICTION_VERIFY_PROMPT.format(text_a=fact_a.text, text_b=fact_b.text)
            try:
                raw = await call_llm([{"role": "user", "content": prompt}], temp=0.0, max_tokens=150)
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
                try:
                    self.memory.store._graph.remove_relation(fact_a.gcn_id, "CONTRADICTS", fact_b.gcn_id)
                    self.memory.store._graph.remove_relation(fact_b.gcn_id, "CONTRADICTS", fact_a.gcn_id)
                    fact_a.confidence = min(1.0, fact_a.confidence + 0.05)
                    fact_b.confidence = min(1.0, fact_b.confidence + 0.05)
                    self.memory.store.update(fact_a.gcn_id, {"confidence": fact_a.confidence}, self.user_id)
                    self.memory.store.update(fact_b.gcn_id, {"confidence": fact_b.confidence}, self.user_id)
                except Exception as e:
                    logger.debug(f"Failed to remove contradiction edges: {e}")
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
        fact.confidence *= 0.5
        if fact.confidence < 0.15 and fact.gcn_id:
            try:
                self.memory.store.retract(fact.gcn_id, actor, reason=f"contradiction: {reason}"[:200])
                logger.info(f"[ContradictionVerify] Retracted fact {fact.id}: {reason}")
            except Exception as e:
                logger.debug(f"Retract failed for {fact.gcn_id}: {e}")

    # ===== ИСТОЧНИК ПАМЯТИ ПРИ ПЕРЕСБОРКЕ ПОСЛЕ ИНСТРУМЕНТОВ =====
    @staticmethod
    def _memory_context_for_rebuild(tool_trace: List[Dict[str, Any]], memory_context: str) -> str:
        """
        Вторая часть исправления "путаницы" памяти: если в этом же ходе уже был
        явный вызов internal__recall / internal__semantic_search, не подмешиваем
        ЕЩЁ и автоматический блок "КОНТЕКСТ ИЗ ДОЛГОСРОЧНОЙ ПАМЯТИ" в тот же
        финальный промпт — иначе модель получает один и тот же (или слегка
        разный по top_k/скорингу) набор фактов сразу в двух разных форматах
        (автоблок текстом + JSON-результат инструмента), что и порождало
        расхождения/путаницу, которых нет в MCP-режиме (там память приходит
        только одним способом — через явный вызов инструмента).
        """
        recall_tools = {"internal__recall", "internal__semantic_search"}
        if any(t.get("tool") in recall_tools for t in tool_trace):
            return ""
        return memory_context

    # ===== ПОСТРОЕНИЕ СООБЩЕНИЙ =====
    def _build_messages(self, message: str, web_search: bool, search_context: str,
                        memory_context: str, image_base64: Optional[str],
                        image_mime: Optional[str], reasoning: bool,
                        uncertainty: float, predictions: List[str], goal_hint: str) -> List[Dict]:
        system_parts = [
            "Ты — когнитивный AI-ассистент с доступом к трём источникам знаний:",
            "1. ЛИЧНАЯ ПАМЯТЬ (факты, которые пользователь просил запомнить или извлечены из диалога) — самый надёжный источник.",
            "2. РЕЗУЛЬТАТЫ ПОИСКА В ИНТЕРНЕТЕ (актуальные данные) — используй, если они есть и релевантны.",
            "3. ГЛОБАЛЬНАЯ ПАМЯТЬ (общие факты, накопленные из разных диалогов) — менее приоритетны.",
            "",
            "ПРАВИЛА ОТВЕТА:",
            "- Всегда отдавай приоритет личной памяти над поиском, если информация совпадает.",
            "- Если информация из разных источников противоречит, укажи это и предложи пользователю уточнить.",
            "- Для фактов из поиска указывай источник (URL или название), если он известен.",
            "- Если ты не уверен в ответе (уверенность < 0.7), честно скажи об этом.",
            "- Ответ должен быть структурирован: краткое вступление, основная часть, вывод (если нужно).",
            "- Если в контексте есть несколько фактов по теме, объедини их в связное объяснение, не перечисляй просто список.",
            "- Не выдумывай фактов, которых нет в предоставленном контексте. Если информации недостаточно, скажи об этом прямо.",
            "- В контексте факты помечены типом: [концепт (обобщение)] — это уже консолидированное коллективное знание, "
            "более общее и надёжное, чем разовый [факт]; [прошлый диалог] — эпизод, а не факт, используй его как контекст беседы.",
            "- Если в блоке 'НЕРАЗРЕШЁННЫЕ ПРОТИВОРЕЧИЯ В ПАМЯТИ' есть пары — это не ошибка поиска, а зафиксированное "
            "системой расхождение источников; обязательно назови обе стороны и не выбирай одну молча.",
        ]

        if uncertainty > 0.6:
            system_parts.append(f"Твоя уверенность в ответе низкая ({uncertainty:.2f}). Если не знаешь – скажи об этом.")
        if predictions:
            system_parts.append(f"Возможное продолжение темы: {', '.join(predictions[:3])}.")
        if goal_hint:
            system_parts.append(f"Учитывай активные цели: {goal_hint}.")
        if reasoning:
            system_parts.append(
                "Перед ответом покажи рассуждения: начни с 💭 РАССУЖДЕНИЕ: и заканчивая ---, затем финальный ответ.")
        if web_search:
            system_parts.append("Ты выполнил поиск в интернете, используй полученные данные как основной источник фактов.")

        # Примечание: раньше здесь был текстовый блок с инструкцией эмитировать
        # {"action": "mcp_call", ...} прямо в финальном ответе. Это убрано —
        # решение "нужен ли инструмент" теперь принимается ОТДЕЛЬНЫМ шагом
        # (GCN.tool_router.ToolRouter, вызывается до _build_messages), поэтому
        # генерация финального ответа больше не должна ничего "решать" и не
        # путается между текстом для пользователя и JSON-командой.

        system_content = "\n".join(system_parts)
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

    # ===== ИССЛЕДОВАНИЕ =====
    async def research(self, goal: str) -> Dict[str, Any]:
        prompt = (
            f"Сформулируй 3 чёткие, проверяемые гипотезы по вопросу: {goal}. "
            "Каждая гипотеза должна быть кратким утверждением (не вопросом), содержащим конкретное предположение. "
            "Ответь в виде маркированного списка, без пояснений."
        )
        hypotheses_text = await call_llm([{"role": "user", "content": prompt}], temp=0.8)
        hypotheses = [h.strip("-• ").strip() for h in hypotheses_text.split('\n') if h.strip()][:3]
        if not hypotheses:
            hypotheses = ["Не удалось сгенерировать гипотезы"]

        all_evidence = []
        queries = [goal] + hypotheses[:2]
        for q in queries:
            try:
                data = await deep_search(q, max_results=3)
                for src in data.get("sources", []):
                    all_evidence.append({"source": src.get("url", ""), "title": src.get("title", ""), "query": q})
            except Exception as e:
                logger.debug(f"Research search error for '{q}': {e}")

        evidence_text = "\n".join([f"- {e['title']}: {e['source']} (запрос: {e['query']})" for e in all_evidence[:6]])

        answer_prompt = (
            f"На основе следующих гипотез и собранных доказательств дай развёрнутый ответ на вопрос: {goal}.\n"
            "Укажи уверенность (0-1) для каждого утверждения и приведи аргументы.\n"
            "Структурируй ответ: вступление, основная часть с аргументацией, заключение.\n\n"
            f"Гипотезы: {', '.join(hypotheses)}\n\n"
            f"Источники:\n{evidence_text}"
        )
        answer = await call_llm([{"role": "user", "content": answer_prompt}], temp=0.6)
        return {"answer": answer, "confidence": 0.7, "hypotheses": hypotheses, "evidence": all_evidence}

    # ===== ПОТОКОВЫЙ ОТВЕТ =====
    async def stream_response(self, message: str, web_search: bool = False,
                              image_base64: str = None, image_mime: str = None,
                              reasoning: bool = False, char_by_char: bool = None):
        # Подтягиваем изменения, сделанные другими процессами (например, MCP)
        self.router.refresh(include_private=True)

        self._last_activity_time = time.time()

        cmd_response = await self._handle_memory_command(message)
        if cmd_response:
            yield f"data: {json.dumps({'token': cmd_response[0]})}\n\n"
            yield "data: [DONE]\n\n"
            return

        await self._ensure_external_tools_registered()

        messages, search_meta = await self._prepare_messages(
            message, web_search, image_base64, image_mime, reasoning
        )

        if search_meta.get("sources"):
            yield f"data: {json.dumps({'sources': search_meta['sources']})}\n\n"

        full_response = ""
        try:
            if LM_STUDIO_USE_STREAM:
                # Решение "нужен ли инструмент" через ToolRouter
                history_tail = "\n".join(
                    f"{m.get('role')}: {str(m.get('content'))[:200]}" for m in self.history[-6:]
                )
                tool_run = await self.tool_router.run(message, messages, history_tail=history_tail)
                tool_trace = tool_run.get("tool_trace", [])
                logger.info(f"Tool trace: {tool_trace}")  # ДОБАВЛЕНО

                # Проверяем, не был ли вызван generate_image
                #
                # БАГ (была причина №1, почему картинка генерировалась, сохранялась
                # на диск, но никогда не появлялась в чате): ToolRouter._execute_tool
                # ВСЕГДА сериализует результат хэндлера в JSON-строку через
                # json.dumps(...) перед тем как положить его в tool_trace (это нужно
                # для текстового промпта финального ответа — build_tool_trace_context
                # и few-shot fallback работают со строками). Из-за этого здесь
                # `result` — это уже str, а не dict, и проверка `isinstance(result, dict)`
                # была ЛОЖНОЙ ВСЕГДА, даже когда генерация прошла успешно и image_url
                # реально был внутри. Именно поэтому в логах стабильно вылезало
                # предупреждение "generate_image result does not contain image_url",
                # хотя сам результат содержал image_url — просто в виде текста JSON.
                for t in tool_trace:
                    if t.get("tool") == "internal__generate_image":
                        result = t.get("result")
                        if isinstance(result, str):
                            try:
                                result = json.loads(result)
                            except (json.JSONDecodeError, TypeError):
                                logger.warning(
                                    f"generate_image: не удалось распарсить результат инструмента как JSON: {result[:200]!r}"
                                )
                        logger.info(f"generate_image result: {result}")  # ДОБАВЛЕНО
                        if isinstance(result, dict) and result.get("image_url"):
                            image_url = result["image_url"]
                            logger.info(f"Sending image_url event: {image_url}")  # ДОБАВЛЕНО
                            yield f"data: {json.dumps({'image_url': image_url})}\n\n"
                            break
                        else:
                            logger.warning("generate_image result does not contain image_url")  # ДОБАВЛЕНО

                if tool_trace:
                    for t in tool_trace:
                        yield f"data: {json.dumps({'tool_call': t['tool'], 'result_preview': str(t['result'])[:200]})}\n\n"
                        self.history.append({
                            "role": "assistant",
                            "content": f"[инструмент {t['tool']}] {str(t['result'])[:500]}"
                        })
                    messages = self._build_messages(
                        message=message,
                        web_search=web_search,
                        search_context=search_meta.get("context", ""),
                        memory_context=self._memory_context_for_rebuild(
                            tool_trace, self._last_prepare_meta.get("memory_context", "")
                        ),
                        image_base64=image_base64,
                        image_mime=image_mime,
                        reasoning=reasoning,
                        uncertainty=self._last_prepare_meta.get("uncertainty", 0.5),
                        predictions=self._last_prepare_meta.get("predictions", []),
                        goal_hint=self._last_prepare_meta.get("goal_hint", "")
                    )
                    messages.append({
                        "role": "user",
                        "content": build_tool_trace_context(tool_trace) + "\n\nТеперь дай финальный ответ пользователю."
                    })

                async for token in call_llm_stream(messages):
                    full_response += token
                    yield f"data: {json.dumps({'token': token})}\n\n"
            else:
                response, _ = await self.process_input(message, web_search, image_base64, image_mime, reasoning)
                full_response = response
                if char_by_char is None:
                    char_by_char = STREAM_CHAR_BY_CHAR
                if char_by_char:
                    for ch in full_response:
                        yield f"data: {json.dumps({'token': ch})}\n\n"
                        await asyncio.sleep(STREAM_CHAR_DELAY)
                else:
                    for word in full_response.split():
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

            relevant = self._last_prepare_meta.get("relevant", [])
            for fact_dict in relevant[:3]:
                gcn_id = fact_dict.get("gcn_id")
                if gcn_id:
                    self.memory.hierarchy.add_to_working(gcn_id)

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

    # ===== ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ =====
    async def generate_image(self, prompt: str, steps: Optional[int] = None,
                              width: Optional[int] = None, height: Optional[int] = None,
                              cfg_scale: Optional[float] = None, seed: Optional[int] = None,
                              sampler_name: Optional[str] = None) -> Optional[str]:
        return await generate_image(prompt, steps=steps, width=width, height=height,
                                     cfg_scale=cfg_scale, seed=seed, sampler_name=sampler_name)

    async def enhance_prompt(self, prompt: str) -> str:
        return await enhance_prompt(prompt)

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

    async def get_response(self, message: str, web_search: bool = False,
                           image_base64: str = None, image_mime: str = None,
                           reasoning: bool = False):
        return await self.process_input(message, web_search, image_base64, image_mime, reasoning)

    async def shutdown(self):
        if self._consolidation_task:
            self._consolidation_task.cancel()
        if self._planner_task:
            self._planner_task.cancel()
        if self._research_task:
            self._research_task.cancel()
        if self._reflection_task:
            self._reflection_task.cancel()
        if self._idle_task:
            self._idle_task.cancel()
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
# FastAPI роутер (без изменений)
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
    result = await deep_search(query, max_results=5)
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

@router.get("/gcn_global_stats")
async def gcn_global_stats(address: str = Depends(require_auth)):
    global_mem = GCNMemoryRouter._get_global_memory(MEMORY_BASE_DIR)
    shared_mem = GCNMemoryRouter._get_shared_memory(MEMORY_BASE_DIR)
    return {"global": global_mem.get_stats(), "shared": shared_mem.get_stats()}

@router.post("/force_merge")
async def force_merge(address: str = Depends(require_auth)):
    global_mem = GCNMemoryRouter._get_global_memory(MEMORY_BASE_DIR)
    shared_mem = GCNMemoryRouter._get_shared_memory(MEMORY_BASE_DIR)
    try:
        await global_mem.light_consolidation()
        await shared_mem.light_consolidation()
        return {"status": "ok", "message": "Global/shared consolidation complete"}
    except Exception as e:
        logger.error(f"force_merge failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/apply_global")
async def apply_global(address: str = Depends(require_auth)):
    global_mem = GCNMemoryRouter._get_global_memory(MEMORY_BASE_DIR)
    shared_mem = GCNMemoryRouter._get_shared_memory(MEMORY_BASE_DIR)
    try:
        await global_mem.deep_consolidation()
        await shared_mem.deep_consolidation()
        return {"status": "ok", "message": "Global/shared deep consolidation complete"}
    except Exception as e:
        logger.error(f"apply_global failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

_global_merge_task: Optional[asyncio.Task] = None

def start_global_merge_task():
    global _global_merge_task
    if _global_merge_task is not None and not _global_merge_task.done():
        logger.info("Global merge task already running")
        return
    loop = asyncio.get_event_loop()
    if not loop.is_running():
        logger.warning("start_global_merge_task called without a running loop; skipped")
        return
    _global_merge_task = asyncio.create_task(_global_merge_loop())
    logger.info("Global merge task started")

async def _global_merge_loop():
    global_mem = GCNMemoryRouter._get_global_memory(MEMORY_BASE_DIR)
    shared_mem = GCNMemoryRouter._get_shared_memory(MEMORY_BASE_DIR)
    # НОВОЕ: лёгкий системный роутер только для того, чтобы дотянуться до
    # form_concepts() — сам он не хранит приватных данных пользователя
    # (user_id="system:consolidation" не привязан ни к одному реальному
    # аккаунту), а global/shared внутри него — те же синглтоны, что и
    # global_mem/shared_mem выше (см. GCNMemoryRouter._get_global_memory).
    concept_router = GCNMemoryRouter("system:consolidation", MEMORY_BASE_DIR)
    concept_router.set_llm_caller(call_llm)
    while True:
        await asyncio.sleep(CONSOLIDATION_INTERVAL)
        try:
            await global_mem.light_consolidation()
            await shared_mem.light_consolidation()
        except Exception as e:
            logger.error(f"Global light consolidation error: {e}")
        await asyncio.sleep(DEEP_CONSOLIDATION_INTERVAL - CONSOLIDATION_INTERVAL)
        try:
            await global_mem.deep_consolidation()
            await shared_mem.deep_consolidation()
        except Exception as e:
            logger.error(f"Global deep consolidation error: {e}")
        # НОВОЕ: формирование концептов — шаг абстрагирования поверх
        # консолидированных фактов (см. GCNMemoryRouter.form_concepts).
        # Именно здесь у системы появляется шанс на эмерджентные обобщения,
        # а не только на дедуп/decay сырых фактов.
        try:
            await concept_router.form_concepts(MemoryScope.GLOBAL)
            await concept_router.form_concepts(MemoryScope.SHARED)
        except Exception as e:
            logger.error(f"Concept formation error: {e}")

@router.on_event("startup")
async def _on_router_startup():
    start_global_merge_task()

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