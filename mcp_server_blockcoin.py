#!/usr/bin/env python3
"""
MCP Сервер для BlockcoinWitres (GCN Cognitive Memory) – Рефакторинг
Использует общие утилиты и методы контроллера.

Добавлены:
- Универсальный инструмент `execute_command` для выполнения любых команд через чат-пайплайн.
- Улучшенные описания и метаданные.
- Поддержка возврата структурированных данных (словарей).
"""

import asyncio
import logging
import sys
import time
from collections import OrderedDict
from pathlib import Path
from typing import Optional, Dict, Any
import random
sys.path.insert(0, str(Path(__file__).resolve().parent))
import os
import base64
from datetime import datetime
from mcp.server.fastmcp import FastMCP
from pydantic import Field
from GCN.llm_client import call_llm
from GCN.memory_graph import GCNMemoryRouter, MemoryScope
from GCN.config_ai import MEMORY_BASE_DIR
from GCN.web_search import deep_search
from routes.ai_assistant import get_assistant
from GCN.image_utils import enhance_prompt, generate_image as gen_image
from GCN.config_ai import GENERATED_IMAGES_DIR


try:
    from GCN.config_ai import (
        EASYDIFFUSION_ENABLED, EASYDIFFUSION_URL, EASYDIFFUSION_ENDPOINT,
        EASYDIFFUSION_TIMEOUT,
        EASYDIFFUSION_DEFAULT_STEPS, EASYDIFFUSION_DEFAULT_WIDTH, EASYDIFFUSION_DEFAULT_HEIGHT
    )
except ImportError:
    EASYDIFFUSION_ENABLED = False
    EASYDIFFUSION_URL = "http://localhost:7860"
    EASYDIFFUSION_ENDPOINT = "/v1/sdapi/v1/txt2img"
    EASYDIFFUSION_TIMEOUT = 120
    EASYDIFFUSION_DEFAULT_STEPS = 20
    EASYDIFFUSION_DEFAULT_WIDTH = 512
    EASYDIFFUSION_DEFAULT_HEIGHT = 512

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("blockcoin-mcp")

DEFAULT_USER = "default_user"

# УЛУЧШЕНИЕ: раньше _routers был обычным dict без выгрузки — на каждый новый
# user_id (адрес кошелька) создавался GCNMemoryRouter (эмбеддинги + граф в
# памяти процесса) и никогда не освобождался. Для долгоживущего MCP-сервера
# с большим числом разных пользователей это неограниченная утечка памяти.
# Теперь роутеры хранятся в OrderedDict как LRU: при каждом обращении
# сдвигаются в конец, неактивные дольше MCP_ROUTER_MAX_IDLE_SECONDS и/или
# роутеры сверх MCP_ROUTER_MAX_COUNT выгружаются — но не молча, а с
# сохранением всех трёх слоёв памяти на диск перед выгрузкой, чтобы не
# потерять несохранённые изменения.
_ROUTER_MAX_IDLE_SECONDS = int(os.getenv("MCP_ROUTER_MAX_IDLE_SECONDS", "1800"))  # 30 минут
_ROUTER_MAX_COUNT = int(os.getenv("MCP_ROUTER_MAX_COUNT", "50"))

_routers: "OrderedDict[str, GCNMemoryRouter]" = OrderedDict()
_router_last_used: Dict[str, float] = {}
_router_lock = asyncio.Lock()


async def _save_router(uid: str, router: GCNMemoryRouter) -> None:
    for mem in (router.private_memory, router.shared_memory, router.global_memory):
        try:
            await mem._save_async()
        except Exception as e:
            logger.error(f"Ошибка сохранения памяти при выгрузке роутера {uid[:16]}: {e}")


async def _evict_stale_routers(exclude_uid: Optional[str] = None) -> None:
    now = time.time()
    stale = [
        uid for uid, ts in _router_last_used.items()
        if uid != exclude_uid and now - ts > _ROUTER_MAX_IDLE_SECONDS
    ]
    for uid in stale:
        router = _routers.pop(uid, None)
        _router_last_used.pop(uid, None)
        if router is not None:
            await _save_router(uid, router)
            logger.info(f"Роутер {uid[:16]} выгружен из памяти (простой > {_ROUTER_MAX_IDLE_SECONDS}с)")

    while len(_routers) > _ROUTER_MAX_COUNT:
        oldest_uid, oldest_router = next(iter(_routers.items()))
        if oldest_uid == exclude_uid:
            # текущий пользователь — самый старый в LRU (например, единственный
            # активный при MAX_COUNT=1): выгружать нечего, выходим.
            break
        _routers.pop(oldest_uid, None)
        _router_last_used.pop(oldest_uid, None)
        await _save_router(oldest_uid, oldest_router)
        logger.info(f"Роутер {oldest_uid[:16]} выгружен по лимиту количества (LRU, max={_ROUTER_MAX_COUNT})")


async def get_router(user_id: Optional[str], for_write: bool = False) -> GCNMemoryRouter:
    """
    УЛУЧШЕНИЕ: раньше отсутствие user_id для ЛЮБОГО инструмента (в т.ч.
    remember/forget/add_goal — то есть операций записи) тихо утекало в
    DEFAULT_USER='default_user' с одним лишь warning в лог. Это и есть
    задокументированный баг рассинхрона памяти между чатом (user_id = адрес
    кошелька) и MCP: агент мог молча записать факт "не в ту" память, и
    единственным следом был необязательный к прочтению лог. Для write-тулов
    (for_write=True) теперь это жёсткий отказ, а не предупреждение — молчаливая
    порча памяти становится невозможной в принципе. Для read-тулов поведение
    прежнее (можно посмотреть общую default-память, это безопасно).

    Теперь также async: перед выдачей/созданием роутера выполняет LRU/TTL-
    выгрузку неактивных роутеров (см. _evict_stale_routers).
    """
    uid = user_id or DEFAULT_USER
    if uid == DEFAULT_USER:
        if for_write:
            raise ValueError(
                "user_id обязателен для операций записи в память (remember/forget/add_goal/"
                "resolve_contradiction) — иначе это молча уйдёт в общий DEFAULT_USER и рассинхронизируется "
                "с приватной памятью реального пользователя чата. Передайте user_id (адрес кошелька) явно."
            )
        logger.warning(
            "MCP-вызов без user_id — используется DEFAULT_USER='default_user', "
            "это НЕ приватная память реального пользователя чата. "
            "Передавайте user_id (адрес кошелька) явно."
        )

    async with _router_lock:
        await _evict_stale_routers(exclude_uid=uid)
        if uid not in _routers:
            _routers[uid] = GCNMemoryRouter(uid, Path(MEMORY_BASE_DIR))
            logger.info(f"Память MCP инициализирована для {uid[:16]}")
        _routers.move_to_end(uid)
        _router_last_used[uid] = time.time()
        return _routers[uid]

mcp = FastMCP("BlockcoinWitres Memory", description="Когнитивная память с веб-поиском и генерацией")

_USER_ID_DESC = "Идентификатор пользователя (тот же адрес кошелька, что использует чат). Если не передан — используется общий default_user, а не личная память конкретного человека."

# УЛУЧШЕНИЕ: TOOL_CALL_TIMEOUT_SECONDS в mcp_client_manager.py защищает НАШ
# чат, когда он сам выступает MCP-клиентом. Но когда этот файл работает как
# MCP-сервер для внешних клиентов (Claude Desktop и т.п.), у его тулов не было
# никакого таймаута — зависший LLM/EasyDiffusion/DDG вызов внутри
# execute_command/web_search/generate_image/research_topic вешал сессию
# клиента навсегда. _with_timeout оборачивает такие тулы явным лимитом.
_MCP_TOOL_TIMEOUT_SECONDS = int(os.getenv("MCP_TOOL_TIMEOUT_SECONDS", "120"))


async def _with_timeout(coro, tool_name: str, timeout: Optional[float] = None) -> Any:
    try:
        return await asyncio.wait_for(coro, timeout=timeout or _MCP_TOOL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.error(f"Тул '{tool_name}' превысил таймаут {timeout or _MCP_TOOL_TIMEOUT_SECONDS}с")
        return {
            "status": "error",
            "error": "timeout",
            "message": f"'{tool_name}' не ответил за {timeout or _MCP_TOOL_TIMEOUT_SECONDS}с — операция прервана.",
        }

# ============================================================
# УНИВЕРСАЛЬНЫЙ ИНСТРУМЕНТ
# ============================================================
@mcp.tool()
async def execute_command(
    command: str = Field(..., description="Любая команда на естественном языке (например, 'запомни, что ...' или 'что ты знаешь о ...')"),
    allow_web_search: bool = Field(
        True,
        description="Разрешить пайплайну использовать веб-поиск, если команда его требует.",
    ),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)
) -> Dict[str, Any]:
    """
    Выполняет любую команду через тот же пайплайн, что и обычный чат.
    Возвращает структурированный ответ с результатом и метаданными.
    """
    # ИСПРАВЛЕНО: раньше web_search был жёстко зашит в False, хотя тул
    # заявлен как универсальный вход "любая команда на естественном языке" —
    # команда вида "найди в интернете ..." просто не могла воспользоваться
    # поиском через этот тул. Теперь управляется явным параметром (по
    # умолчанию включено, как и в обычном чате).
    assistant = await get_assistant(user_id or DEFAULT_USER)

    async def _run():
        return await assistant.process_input(command, web_search=allow_web_search)

    result = await _with_timeout(_run(), "execute_command")
    if isinstance(result, dict) and result.get("error") == "timeout":
        return result
    response, meta = result
    return {
        "result": response,
        "meta": meta,
        "user_id": user_id or DEFAULT_USER,
        "timestamp": asyncio.get_event_loop().time()
    }

# ============================================================
# ОСТАЛЬНЫЕ ИНСТРУМЕНТЫ (с улучшенными описаниями)
# ============================================================
@mcp.tool()
async def recall(
    query: str = Field(..., description="Поисковый запрос (тема, вопрос, ключевые слова)"),
    top_k: int = Field(5, description="Максимальное число результатов", ge=1, le=20),
    scope: Optional[str] = Field(None, description="Фильтр по скоупу: 'private', 'shared', 'global'"),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)
) -> Dict[str, Any]:
    """
    Поиск в памяти с фильтром по скоупу. Возвращает список фактов с уверенностью.
    """
    router = await get_router(user_id)
    results = await router.retrieve(query, top_k=top_k*2, include_private=True)
    if scope:
        scope_lower = scope.lower()
        filtered = []
        for item in results:
            gcn_id = item.get("gcn_id")
            if gcn_id:
                obj = (router.private_memory.store.get(gcn_id) or
                       router.shared_memory.store.get(gcn_id) or
                       router.global_memory.store.get(gcn_id))
                if obj and obj.scope.value == scope_lower:
                    filtered.append(item)
        results = filtered[:top_k]
    else:
        results = results[:top_k]

    return {
        "results": [
            {
                "text": item.get("text", ""),
                "confidence": item.get("confidence", 0.0),
                "importance": item.get("importance", 1.0),
                "scope": item.get("scope", "private"),
                "gcn_id": item.get("gcn_id")
            }
            for item in results
        ],
        "count": len(results)
    }

@mcp.tool()
async def remember(
    fact: str = Field(..., description="Факт для запоминания"),
    scope: Optional[str] = Field(
        None,
        description="Скоуп: 'private', 'shared', 'global'. Если не передан явно — "
                    "определяется автоматически по наличию слов 'глобально'/'global' в тексте факта.",
    ),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)
) -> Dict[str, Any]:
    """
    Сохраняет факт в указанный скоуп. Если scope не указан, определяется автоматически по наличию слов "глобально"/"global".
    """
    router = await get_router(user_id, for_write=True)
    router.refresh()

    # ИСПРАВЛЕНО: раньше default для scope был строкой "private" (не None), из-за
    # чего условие `scope is None or scope.lower() == "private"` было истинным
    # и для дефолта, И для ЛЮБОГО явного scope="private" — то есть явный запрос
    # "сохрани приватно" всё равно прогонялся через эвристику по ключевым словам
    # и мог молча уйти в GLOBAL, если факт случайно содержал подстроку "global"
    # (например "global variable"). Теперь автоопределение срабатывает ТОЛЬКО
    # когда scope не передан вовсе; любой явный scope — единственный источник
    # истины и никогда не переопределяется эвристикой.
    if scope is None:
        if "глобально" in fact.lower() or "global" in fact.lower():
            scope_enum = MemoryScope.GLOBAL
        else:
            scope_enum = MemoryScope.PRIVATE
    else:
        scope_map = {"private": MemoryScope.PRIVATE, "shared": MemoryScope.SHARED, "global": MemoryScope.GLOBAL}
        scope_enum = scope_map.get(scope.lower(), MemoryScope.PRIVATE)

    obj_id = router.add_knowledge(
        subject=fact,
        predicate="is_fact",
        obj="true",
        scope=scope_enum,
        confidence=0.7,
        author=router.user_id,
        source_type="mcp_tool"
    )

    if scope_enum == MemoryScope.GLOBAL:
        await router.global_memory._save_async()
    elif scope_enum == MemoryScope.PRIVATE:
        await router.private_memory._save_async()
    elif scope_enum == MemoryScope.SHARED:
        await router.shared_memory._save_async()

    return {
        "status": "ok",
        "id": obj_id,
        "scope": scope_enum.value,
        "fact": fact
    }

@mcp.tool()
async def forget(
    query: str = Field(..., description="Ключевые слова для удаления фактов"),
    scope: str = Field("private", description="Из какого слоя удалять: 'private', 'shared' или 'global'"),
    dry_run: bool = Field(
        True,
        description="Если True (по умолчанию) — только показывает, какие факты БУДУТ удалены, "
                    "ничего не удаляя. Передайте False, чтобы выполнить удаление по-настоящему.",
    ),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)
) -> Dict[str, Any]:
    """
    Удаляет факты, содержащие заданные ключевые слова, из указанного слоя памяти.
    """
    # ИСПРАВЛЕНО: удаление шло по грубому substring-совпадению без всякого
    # подтверждения — короткий/частый query мог снести кучу не связанных
    # между собой фактов за один вызов. dry_run=True по умолчанию делает
    # первый вызов безопасным просмотром кандидатов; реальное удаление
    # требует явного dry_run=False.
    router = await get_router(user_id, for_write=True)
    scope_map = {
        "private": router.private_memory,
        "shared": router.shared_memory,
        "global": router.global_memory,
    }
    memory = scope_map.get(scope.lower())
    if memory is None:
        return {"status": "error", "message": f"Неизвестный scope: {scope}"}
    memory.reload_if_stale()
    to_remove = [f for f in memory.semantic_facts if query.lower() in f.text.lower()]
    if not to_remove:
        return {"status": "ok", "removed": 0, "scope": scope.lower(), "message": "Ничего не найдено."}

    if dry_run:
        return {
            "status": "dry_run",
            "would_remove": len(to_remove),
            "scope": scope.lower(),
            "candidates": [{"id": f.id, "text": f.text[:200]} for f in to_remove[:20]],
            "message": "Ничего не удалено. Повторите вызов с dry_run=False, чтобы удалить эти факты.",
        }

    removed = memory._remove_facts({f.id for f in to_remove})
    await memory._schedule_save()
    return {"status": "ok", "removed": removed, "scope": scope.lower()}

@mcp.tool()
async def web_search(
    query: str = Field(..., description="Поисковый запрос"),
    max_results: int = Field(5, description="Максимальное число страниц для анализа", ge=1, le=10)
) -> Dict[str, Any]:
    """
    Выполняет поиск в DuckDuckGo и возвращает извлечённый контекст и источники.
    """
    data = await _with_timeout(deep_search(query, max_results=max_results), "web_search")
    if isinstance(data, dict) and data.get("error") == "timeout":
        return data
    return {
        "query": query,
        "search_performed": data["search_performed"],
        "sources": data.get("sources", []),
        "context": data.get("context", ""),
        "chunks_found": data.get("chunks_found", 0)
    }

@mcp.tool()
async def generate_image(
    prompt: str = Field(..., description="Описание изображения"),
    steps: int = Field(20, description="Количество шагов диффузии"),
    width: int = Field(512, description="Ширина изображения"),
    height: int = Field(512, description="Высота изображения"),
    cfg_scale: float = Field(7.0, description="Масштаб CFG (guidance scale)"),
    sampler: str = Field("dpmpp_2m", description="Сэмплер"),
    seed: int = Field(-1, description="Зерно (-1 для случайного)"),
    enhance_prompt: bool = Field(True, description="Улучшить промпт через LLM перед генерацией"),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)
) -> Dict[str, Any]:
    """
    Генерирует изображение. Возвращает ссылку на файл.
    """
    if not EASYDIFFUSION_ENABLED:
        return {"status": "error", "message": "Генерация отключена."}

    assistant = await get_assistant(user_id or DEFAULT_USER)

    async def _run():
        # 1. Улучшение промпта (если включено)
        fp = prompt
        if enhance_prompt:
            fp = await assistant.enhance_prompt(prompt)
            logger.info(f"Original prompt: {prompt}\nEnhanced prompt: {fp}")

        # 2. Генерация изображения (используем общую функцию)
        # Раньше cfg_scale/sampler/seed принимались в схеме, но никуда не шли —
        # теперь image_utils.generate_image реально их принимает и передаёт в EasyDiffusion.
        img = await gen_image(fp, steps=steps, width=width, height=height,
                               cfg_scale=cfg_scale, seed=seed, sampler_name=sampler)
        return fp, img

    # Диффузия при большом steps/размере честно может занимать дольше общего
    # дефолта — отдельный таймаут для этого тула (см. _MCP_TOOL_TIMEOUT_SECONDS).
    image_timeout = float(os.getenv("MCP_IMAGE_TOOL_TIMEOUT_SECONDS", "300"))
    run_result = await _with_timeout(_run(), "generate_image", timeout=image_timeout)
    if isinstance(run_result, dict) and run_result.get("error") == "timeout":
        return run_result
    final_prompt, image_b64 = run_result
    if not image_b64:
        return {"status": "error", "message": "Не удалось сгенерировать изображение"}

    # 3. Сохранение на диск и формирование ссылки
    output_dir = GENERATED_IMAGES_DIR
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"image_{timestamp}.png"
    try:
        with open(filename, "wb") as f:
            f.write(base64.b64decode(image_b64))
        file_path = str(filename.absolute())

        # Базовый URL вашего FastAPI-сервера (можно задать через переменную окружения)
        BASE_URL = os.getenv("SERVER_BASE_URL", "http://localhost:8000")
        image_url = f"{BASE_URL}/generated_images/{filename.name}"
        message = f"✅ Изображение сгенерировано. Откройте по ссылке: {image_url}"
    except Exception as e:
        logger.error(f"Failed to save image: {e}")
        file_path = None
        image_url = None
        message = "⚠️ Изображение сгенерировано, но не удалось сохранить на диск."

    return {
        "status": "ok",
        #"image_base64": image_b64,      # можно оставить или убрать
        "file_path": file_path,
        "url": image_url,
        "message": message,
        "original_prompt": prompt,
        "enhanced_prompt": final_prompt if enhance_prompt else None
    }


@mcp.tool()
async def research_topic(
    topic: str = Field(..., description="Тема для исследования"),
    depth: int = Field(2, description="Глубина (количество итераций поиска)", ge=1, le=3),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)
) -> Dict[str, Any]:
    """
    Глубокое исследование темы с генерацией гипотез и сбором доказательств.
    """
    assistant = await get_assistant(user_id or DEFAULT_USER)
    # research может делать несколько раундов поиска (depth итераций) — даём
    # ему больше времени, чем дефолтному тулу.
    research_timeout = float(os.getenv("MCP_RESEARCH_TOOL_TIMEOUT_SECONDS", "240"))
    result = await _with_timeout(assistant.research(topic), "research_topic", timeout=research_timeout)
    if isinstance(result, dict) and result.get("error") == "timeout":
        return result
    return {
        "topic": topic,
        "hypotheses": result.get("hypotheses", []),
        "evidence": result.get("evidence", []),
        "answer": result.get("answer", ""),
        "confidence": result.get("confidence", 0.0)
    }

@mcp.tool()
async def get_episodes(
    limit: int = Field(5, description="Количество последних эпизодов", ge=1, le=20),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)
) -> Dict[str, Any]:
    """
    Возвращает последние диалоги (эпизоды) из личной памяти.
    """
    memory = (await get_router(user_id)).private_memory
    memory.reload_if_stale()
    episodes = memory.episodic_memory[-limit:] if memory.episodic_memory else []
    return {
        "episodes": [
            {"user": ep.user_msg, "assistant": ep.assistant_msg, "timestamp": ep.timestamp}
            for ep in reversed(episodes)
        ],
        "count": len(episodes)
    }

@mcp.tool()
async def get_contradictions(
    limit: int = Field(5, description="Максимальное число пар противоречий", ge=1, le=10),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)
) -> Dict[str, Any]:
    """
    Возвращает неразрешённые противоречия из личной памяти.
    """
    memory = (await get_router(user_id)).private_memory
    memory.reload_if_stale()
    pairs = memory.get_unverified_contradictions(limit=limit)
    return {
        "contradictions": [
            {
                "a": {"text": a.text, "confidence": a.confidence, "id": a.id},
                "b": {"text": b.text, "confidence": b.confidence, "id": b.id}
            }
            for a, b in pairs
        ],
        "count": len(pairs)
    }

@mcp.tool()
async def resolve_contradiction(
    fact_id_a: str = Field(..., description="ID первого факта (числовой или GCN-идентификатор)"),
    fact_id_b: str = Field(..., description="ID второго факта"),
    verdict: str = Field(..., description="Вердикт: 'a' (оставить A), 'b' (оставить B), 'both' (сохранить оба), 'neither' (удалить оба)"),
    reason: str = Field("", description="Причина разрешения (опционально)"),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)
) -> Dict[str, Any]:
    """
    Ручное разрешение противоречия.
    """
    memory = (await get_router(user_id, for_write=True)).private_memory
    memory.reload_if_stale()

    def find_fact(fid):
        if fid in memory.facts_by_id:
            return memory.facts_by_id[fid]
        for f in memory.semantic_facts:
            if f.gcn_id == fid:
                return f
        return None

    fa = find_fact(fact_id_a)
    fb = find_fact(fact_id_b)
    if not fa or not fb:
        return {"status": "error", "message": f"Факты не найдены: A={fact_id_a}, B={fact_id_b}"}

    v = verdict.lower()
    if v == "a":
        memory._remove_facts({fb.id})
        memory.gcn_store._graph.remove_relation(fa.gcn_id, "CONTRADICTS", fb.gcn_id)
        memory.gcn_store._graph.remove_relation(fb.gcn_id, "CONTRADICTS", fa.gcn_id)
        fa.contradicts.discard(fb.id)
        fb.contradicts.discard(fa.id)
        await memory._schedule_save()
        return {"status": "ok", "verdict": "a", "kept": fa.text, "removed": fb.text}
    elif v == "b":
        memory._remove_facts({fa.id})
        memory.gcn_store._graph.remove_relation(fa.gcn_id, "CONTRADICTS", fb.gcn_id)
        memory.gcn_store._graph.remove_relation(fb.gcn_id, "CONTRADICTS", fa.gcn_id)
        fa.contradicts.discard(fb.id)
        fb.contradicts.discard(fa.id)
        await memory._schedule_save()
        return {"status": "ok", "verdict": "b", "kept": fb.text, "removed": fa.text}
    elif v == "both":
        memory.gcn_store._graph.remove_relation(fa.gcn_id, "CONTRADICTS", fb.gcn_id)
        memory.gcn_store._graph.remove_relation(fb.gcn_id, "CONTRADICTS", fa.gcn_id)
        fa.contradicts.discard(fb.id)
        fb.contradicts.discard(fa.id)
        await memory._schedule_save()
        return {"status": "ok", "verdict": "both", "message": "Противоречие снято, оба сохранены."}
    elif v == "neither":
        memory._remove_facts({fa.id, fb.id})
        await memory._schedule_save()
        return {"status": "ok", "verdict": "neither", "message": "Оба удалены."}
    else:
        return {"status": "error", "message": f"Неизвестный вердикт: {verdict}"}

@mcp.tool()
async def get_goals(
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)
) -> Dict[str, Any]:
    """
    Возвращает активные цели пользователя.
    """
    memory = (await get_router(user_id)).private_memory
    memory.reload_if_stale()
    goals = await memory.get_active_goals()
    return {
        "goals": [
            {"description": g.description, "priority": g.priority, "confidence": g.confidence, "status": g.status}
            for g in goals
        ],
        "count": len(goals)
    }

@mcp.tool()
async def add_goal(
    description: str = Field(..., description="Описание цели"),
    priority: float = Field(0.5, description="Приоритет от 0 до 1", ge=0, le=1),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)
) -> Dict[str, Any]:
    """
    Добавляет новую цель в личную память.
    """
    memory = (await get_router(user_id, for_write=True)).private_memory
    memory.reload_if_stale()
    gid = await memory.add_goal(description, priority)
    return {"status": "ok", "id": gid, "description": description}

@mcp.tool()
async def semantic_search(
    query: str = Field(..., description="Поисковый запрос"),
    top_k: int = Field(5, description="Число результатов", ge=1, le=20),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)
) -> Dict[str, Any]:
    """
    Векторный поиск по смыслу (использует эмбеддинги).
    """
    memory = (await get_router(user_id)).private_memory
    memory.reload_if_stale()
    emb = memory.embed_text(query)
    if emb is None:
        return {"error": "Эмбеддинги недоступны."}
    results = memory.store.semantic_search(emb, top_k=top_k*2)
    return {
        "results": [
            {"text": memory.store.get(gcn_id).subject if memory.store.get(gcn_id) else "", "score": score}
            for gcn_id, score in results[:top_k]
        ]
    }

@mcp.tool()
async def graph_explore(
    seed_text: str = Field(..., description="Текст для поиска стартового узла"),
    depth: int = Field(2, description="Глубина обхода графа", ge=1, le=3),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)
) -> Dict[str, Any]:
    """
    Исследует граф синапсов, начиная с фактов, содержащих seed_text.
    """
    memory = (await get_router(user_id)).private_memory
    memory.reload_if_stale()
    seed_ids = [f.id for f in memory.semantic_facts if seed_text.lower() in f.text.lower()]
    if not seed_ids:
        return {"error": f"Факты с '{seed_text}' не найдены."}
    activation = await memory.spread_activation(seed_ids[:3], max_depth=min(depth, 3))
    sorted_items = sorted(activation.items(), key=lambda x: x[1], reverse=True)
    return {
        "nodes": [
            {"id": fid, "text": memory.facts_by_id.get(fid).text[:200] if memory.facts_by_id.get(fid) else "", "activation": act}
            for fid, act in sorted_items[:20] if fid not in seed_ids
        ]
    }

@mcp.tool()
async def explain_fact(
    gcn_id: str = Field(..., description="Идентификатор объекта памяти (gcn_id), полученный из recall/semantic_search"),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)
) -> Dict[str, Any]:
    """
    НОВОЕ: объясняет происхождение и статус утверждения памяти — кто и когда его
    создал, сколько раз и кем подтверждено (author-теги в evidence), с чем
    противоречит, проходило ли верификацию. Раньше эта информация (уже собранная
    в GCN.AIAdapter.explain()/provenance) нигде не была доступна снаружи — ни
    пользователь, ни агент не могли спросить "почему ты в этом уверен" и
    получить настоящий ответ вместо додумывания моделью.
    """
    router = await get_router(user_id)
    router.refresh()
    obj = (router.private_memory.store.get(gcn_id) or
           router.shared_memory.store.get(gcn_id) or
           router.global_memory.store.get(gcn_id))
    if not obj:
        return {"error": f"Объект {gcn_id} не найден ни в одном слое памяти."}

    store = (router.private_memory.store if router.private_memory.store.get(gcn_id) else
            router.shared_memory.store if router.shared_memory.store.get(gcn_id) else
            router.global_memory.store)

    contradictions = store._graph.get_neighbors(gcn_id, "CONTRADICTS")
    grounds_in = store._graph.get_neighbors(gcn_id, "GROUNDS_IN")
    abstracts_from = store._graph.get_neighbors(gcn_id, "ABSTRACTS_FROM")
    confirming_authors = [e.split("author:", 1)[1] for e in obj.evidence if e.startswith("author:")]

    return {
        "id": obj.id,
        "type": obj.type.value,
        "scope": obj.scope.value,
        "text": obj.subject,
        "confidence": obj.confidence,
        "version": obj.version,
        "author": obj.author,
        "source_type": obj.source_type,
        "created": obj.created.isoformat(),
        "confirming_authors": confirming_authors,
        "contradicts": [target for _, target in contradictions],
        "grounds_in_global": [target for _, target in grounds_in],
        "abstracted_from": [target for _, target in abstracts_from],
    }

@mcp.tool()
async def get_memory_stats(
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)
) -> Dict[str, Any]:
    """
    Возвращает статистику по личной памяти.
    """
    memory = (await get_router(user_id)).private_memory
    memory.reload_if_stale()
    stats = memory.get_stats()
    return stats

# ============================================================
# РЕСУРСЫ (также адаптированы под user_id)
# ============================================================
@mcp.resource("memory://{user_id}/facts")
async def list_facts(user_id: str) -> Dict[str, Any]:
    memory = (await get_router(user_id)).private_memory
    memory.reload_if_stale()
    facts = memory.semantic_facts[:20]
    return {
        "total": len(memory.semantic_facts),
        "facts": [{"id": f.id, "text": f.text[:200], "confidence": f.confidence} for f in facts]
    }

@mcp.resource("memory://{user_id}/fact/{fact_id}")
async def get_fact(user_id: str, fact_id: str) -> Dict[str, Any]:
    memory = (await get_router(user_id)).private_memory
    memory.reload_if_stale()
    obj = memory.store.get(fact_id)
    if not obj:
        for f in memory.semantic_facts:
            if str(f.id) == fact_id:
                obj = memory.store.get(f.gcn_id)
                break
    if not obj:
        return {"error": f"Факт {fact_id} не найден."}
    return {
        "id": obj.id,
        "text": obj.subject,
        "confidence": obj.confidence,
        "author": obj.author,
        "created": obj.created.isoformat(),
        "version": obj.version,
        "evidence_count": len(obj.evidence),
        "scope": obj.scope.value
    }

# ============================================================
# ЗАПУСК
# ============================================================
if __name__ == "__main__":
    logger.info("🚀 Запуск рефакторированного MCP сервера BlockcoinWitres...")
    mcp.run()