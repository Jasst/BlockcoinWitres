#!/usr/bin/env python3
"""
MCP Сервер для BlockcoinWitres (GCN Cognitive Memory) – Рефакторинг
Использует общие утилиты и методы контроллера.

ВАЖНО (исправлено): раньше память для MCP инициализировалась ОДИН РАЗ на
модульном уровне под захардкоженным DEFAULT_USER = "default_user":

    router = GCNMemoryRouter(DEFAULT_USER, Path(MEMORY_BASE_DIR))

Тогда как обычный чат (routes/ai_assistant.py) создаёт контроллер по
реальному user_id — адресу кошелька:

    assistant = await get_assistant(address)

Из-за этого приватная память MCP (recall/remember/forget/...) физически
лежала в другом каталоге (MEMORY_BASE_DIR/default_user/...) и никогда не
пересекалась с приватной памятью настоящего пользователя чата — это не
"устаревание", а два независимых набора данных.

Теперь каждый инструмент принимает user_id (адрес кошелька — тот же, что
использует чат). Если он не передан, используется DEFAULT_USER как явный,
залогированный fallback для обратной совместимости со старыми клиентами,
а не тихая подмена данных.

Второе исправление: global/shared-память — это process-level синглтоны
(GCNMemoryRouter._global_instance / _shared_instance), а MCP-сервер живёт
в ОТДЕЛЬНОМ процессе (см. mcp_servers.json: command "python", args
["mcp_server_blockcoin.py"], поднимается через stdio). Поэтому перед
каждым обращением к памяти теперь вызывается router.refresh() /
memory.reload_if_stale(), которые дешёво (через mtime файла) проверяют,
не записал ли что-то другой процесс, и подтягивают изменения — см.
GCNMemoryRouter.refresh() и CognitiveMemory.reload_if_stale() в
memory_graph.py, и MemoryStore._merge_disk_state() в GCN.py (та же логика
защищает и от потери данных при одновременной записи с двух процессов).
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional, Dict

# Добавляем корень проекта
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP
from pydantic import Field

from GCN.memory_graph import GCNMemoryRouter, MemoryScope
from GCN.config_ai import MEMORY_BASE_DIR
from GCN.web_search import deep_search  # только то, что реально используется
from GCN.llm_client import call_llm
# Для исследовательских функций используем контроллер
from routes.ai_assistant import get_assistant

# Импортируем константы для генерации изображений (если нужны)
try:
    from GCN.config_ai import (
        EASYDIFFUSION_ENABLED, EASYDIFFUSION_URL, EASYDIFFUSION_TIMEOUT,
        EASYDIFFUSION_DEFAULT_STEPS, EASYDIFFUSION_DEFAULT_WIDTH, EASYDIFFUSION_DEFAULT_HEIGHT
    )
except ImportError:
    EASYDIFFUSION_ENABLED = False

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("blockcoin-mcp")

# Fallback для клиентов, которые ещё не передают user_id явно.
DEFAULT_USER = "default_user"

# Кеш роутеров по user_id (аналог _assistants в routes/ai_assistant.py) —
# держим по одному GCNMemoryRouter на пользователя вместо одного глобального.
_routers: Dict[str, GCNMemoryRouter] = {}


def get_router(user_id: Optional[str]) -> GCNMemoryRouter:
    uid = user_id or DEFAULT_USER
    if uid == DEFAULT_USER:
        logger.warning(
            "MCP-вызов без user_id — используется DEFAULT_USER='default_user', "
            "это НЕ приватная память реального пользователя чата. "
            "Передавайте user_id (адрес кошелька) явно."
        )
    if uid not in _routers:
        _routers[uid] = GCNMemoryRouter(uid, Path(MEMORY_BASE_DIR))
        logger.info(f"Память MCP инициализирована для {uid[:16]}")
    return _routers[uid]


mcp = FastMCP("BlockcoinWitres Memory", description="Когнитивная память с веб-поиском и генерацией")

_USER_ID_DESC = "Идентификатор пользователя (тот же адрес кошелька, что использует чат). Если не передан — используется общий default_user, а не личная память конкретного человека."

# ------------------------------------------------------------
# ИНСТРУМЕНТЫ (используют общие утилиты)
# ------------------------------------------------------------

@mcp.tool()
async def recall(query: str, top_k: int = 5, scope: Optional[str] = None,
                  user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)) -> str:
    """Поиск в памяти с фильтром по скоупу."""
    router = get_router(user_id)
    results = await router.retrieve(query, top_k=top_k*2, include_private=True)  # retrieve() сам вызывает refresh()
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
    if not results:
        return "Ничего не найдено."
    out = []
    for i, item in enumerate(results, 1):
        text = item.get("text", "")
        conf = item.get("confidence", 0.5)
        imp = item.get("importance", 1.0)
        out.append(f"{i}. {text} (уверенность: {conf:.2f}, важность: {imp:.2f})")
    return "\n".join(out)

@mcp.tool()
async def remember(fact: str, scope: str = "private",
                    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)) -> str:
    """Сохранить факт в указанный скоуп."""
    router = get_router(user_id)
    router.refresh()
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
    return f"✅ Сохранён (ID: {obj_id}, скоуп: {scope})"

@mcp.tool()
async def forget(query: str,
                  user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)) -> str:
    """Удалить факты из личной памяти по ключевым словам."""
    router = get_router(user_id)
    memory = router.private_memory
    memory.reload_if_stale()
    to_remove = [f.id for f in memory.semantic_facts if query.lower() in f.text.lower()]
    if not to_remove:
        return "Ничего не найдено."
    removed = memory._remove_facts(set(to_remove))
    await memory._schedule_save()
    return f"✅ Удалено {removed} фактов"

@mcp.tool()
async def web_search(query: str, max_results: int = 5) -> str:
    """Поиск в интернете через DuckDuckGo (использует общую утилиту)."""
    data = await deep_search(query, max_results=max_results)
    if not data["search_performed"]:
        return "Поиск не дал результатов."
    out = [f"🔍 Результаты по запросу: {query}"]
    for src in data["sources"]:
        out.append(f"• {src['title']} – {src['url']}")
    out.append("\n--- Извлечённый контекст ---\n" + data["context"])
    return "\n".join(out)

@mcp.tool()
async def generate_image(
    prompt: str,
    steps: int = EASYDIFFUSION_DEFAULT_STEPS,
    width: int = EASYDIFFUSION_DEFAULT_WIDTH,
    height: int = EASYDIFFUSION_DEFAULT_HEIGHT
) -> str:
    """Генерация изображения через EasyDiffusion (если включено)."""
    if not EASYDIFFUSION_ENABLED:
        return "Генерация отключена в конфиге."
    import aiohttp
    try:
        payload = {"prompt": prompt, "steps": steps, "width": width, "height": height}
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{EASYDIFFUSION_URL}/generate", json=payload,
                                    timeout=aiohttp.ClientTimeout(total=EASYDIFFUSION_TIMEOUT)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    image_b64 = data.get("image_base64")
                    if image_b64:
                        return f"![Сгенерированное изображение](data:image/png;base64,{image_b64})"
                    else:
                        return "❌ EasyDiffusion вернул пустой ответ."
                else:
                    return f"❌ Ошибка {resp.status}: {await resp.text()}"
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"

@mcp.tool()
async def research_topic(topic: str, depth: int = 2,
                          user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)) -> str:
    """
    Глубокое исследование темы с использованием общего контроллера.
    Получаем ассистента для user_id (тот же, что использует чат) и вызываем его метод research.
    """
    assistant = await get_assistant(user_id or DEFAULT_USER)
    result = await assistant.research(topic)  # возвращает Dict с answer, hypotheses, evidence
    out = [f"🔬 Исследование по теме: {topic}\n"]
    out.append("Гипотезы:\n- " + "\n- ".join(result.get("hypotheses", [])))
    out.append("\nДоказательства:")
    for ev in result.get("evidence", [])[:6]:
        out.append(f"• {ev['title']} – {ev['source']}")
    out.append("\nВывод:\n" + result.get("answer", "Нет ответа."))
    return "\n".join(out)

@mcp.tool()
async def get_episodes(limit: int = 5,
                        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)) -> str:
    memory = get_router(user_id).private_memory
    memory.reload_if_stale()
    episodes = memory.episodic_memory[-limit:] if memory.episodic_memory else []
    if not episodes:
        return "Нет эпизодов."
    out = [f"📜 Последние {len(episodes)} диалогов:"]
    for ep in reversed(episodes):
        out.append(f"User: {ep.user_msg[:100]}")
        out.append(f"AI: {ep.assistant_msg[:100]}\n---")
    return "\n".join(out)

@mcp.tool()
async def get_contradictions(limit: int = 5,
                              user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)) -> str:
    memory = get_router(user_id).private_memory
    memory.reload_if_stale()
    pairs = memory.get_unverified_contradictions(limit=limit)
    if not pairs:
        return "Нет неразрешённых противоречий."
    out = ["⚠️ Найдены противоречия:"]
    for i, (a, b) in enumerate(pairs, 1):
        out.append(f"{i}. A: {a.text} (conf: {a.confidence:.2f})")
        out.append(f"   B: {b.text} (conf: {b.confidence:.2f})")
    return "\n".join(out)

@mcp.tool()
async def resolve_contradiction(fact_id_a: str, fact_id_b: str, verdict: str, reason: str = "",
                                 user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)) -> str:
    """Ручное разрешение противоречия."""
    memory = get_router(user_id).private_memory
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
        return f"Факты не найдены: A={fact_id_a}, B={fact_id_b}"
    v = verdict.lower()
    if v == "a":
        memory._remove_facts({fb.id})
        memory.gcn_store._graph.remove_relation(fa.gcn_id, "CONTRADICTS", fb.gcn_id)
        memory.gcn_store._graph.remove_relation(fb.gcn_id, "CONTRADICTS", fa.gcn_id)
        fa.contradicts.discard(fb.id)
        fb.contradicts.discard(fa.id)
        await memory._schedule_save()
        return f"✅ Оставлен A: {fa.text}"
    elif v == "b":
        memory._remove_facts({fa.id})
        memory.gcn_store._graph.remove_relation(fa.gcn_id, "CONTRADICTS", fb.gcn_id)
        memory.gcn_store._graph.remove_relation(fb.gcn_id, "CONTRADICTS", fa.gcn_id)
        fa.contradicts.discard(fb.id)
        fb.contradicts.discard(fa.id)
        await memory._schedule_save()
        return f"✅ Оставлен B: {fb.text}"
    elif v == "both":
        memory.gcn_store._graph.remove_relation(fa.gcn_id, "CONTRADICTS", fb.gcn_id)
        memory.gcn_store._graph.remove_relation(fb.gcn_id, "CONTRADICTS", fa.gcn_id)
        fa.contradicts.discard(fb.id)
        fb.contradicts.discard(fa.id)
        await memory._schedule_save()
        return "✅ Противоречие снято, оба сохранены."
    elif v == "neither":
        memory._remove_facts({fa.id, fb.id})
        await memory._schedule_save()
        return "✅ Оба удалены."
    else:
        return f"Неизвестный вердикт: {verdict}"

@mcp.tool()
async def get_goals(user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)) -> str:
    memory = get_router(user_id).private_memory
    memory.reload_if_stale()
    goals = await memory.get_active_goals()
    if not goals:
        return "Нет активных целей."
    out = ["🎯 Активные цели:"]
    for g in goals:
        out.append(f"  • {g.description} (приоритет: {g.priority:.2f})")
    return "\n".join(out)

@mcp.tool()
async def add_goal(description: str, priority: float = 0.5,
                    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)) -> str:
    memory = get_router(user_id).private_memory
    memory.reload_if_stale()
    gid = await memory.add_goal(description, priority)
    return f"✅ Цель добавлена (ID: {gid})"

@mcp.tool()
async def semantic_search(query: str, top_k: int = 5,
                           user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)) -> str:
    memory = get_router(user_id).private_memory
    memory.reload_if_stale()
    emb = memory.embed_text(query)
    if emb is None:
        return "Эмбеддинги недоступны."
    results = memory.store.semantic_search(emb, top_k=top_k*2)
    out = []
    for gcn_id, score in results[:top_k]:
        obj = memory.store.get(gcn_id)
        if obj:
            out.append(f"• {obj.subject} (сходство: {score:.3f})")
    return "\n".join(out) if out else "Ничего не найдено."

@mcp.tool()
async def graph_explore(seed_text: str, depth: int = 2,
                         user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)) -> str:
    memory = get_router(user_id).private_memory
    memory.reload_if_stale()
    seed_ids = [f.id for f in memory.semantic_facts if seed_text.lower() in f.text.lower()]
    if not seed_ids:
        return f"Факты с '{seed_text}' не найдены."
    activation = await memory.spread_activation(seed_ids[:3], max_depth=min(depth, 3))
    sorted_items = sorted(activation.items(), key=lambda x: x[1], reverse=True)
    out = [f"🧠 Обход графа (старт: '{seed_text}')"]
    count = 0
    for fid, act in sorted_items:
        if fid in seed_ids:
            continue
        fact = memory.facts_by_id.get(fid)
        if fact:
            out.append(f"  • {fact.text[:80]}... (активация: {act:.3f})")
            count += 1
            if count >= 10:
                break
    if count == 0:
        out.append("  Нет связанных узлов.")
    return "\n".join(out)

@mcp.tool()
async def get_memory_stats(user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC)) -> str:
    memory = get_router(user_id).private_memory
    memory.reload_if_stale()
    stats = memory.get_stats()
    return (
        f"📊 Статистика:\n"
        f"  Фактов: {stats.get('semantic_facts', 0)}\n"
        f"  Эпизодов: {stats.get('episodes', 0)}\n"
        f"  Синапсов: {stats.get('synapses', 0)}\n"
        f"  Активных целей: {stats.get('active_goals', 0)}"
    )

# ------------------------------------------------------------
# РЕСУРСЫ
# Раньше были "memory://facts" и "memory://fact/{fact_id}" — всегда на
# DEFAULT_USER. Теперь user_id — часть URI, иначе ресурс в принципе не
# может указать, чья это память.
# ------------------------------------------------------------
@mcp.resource("memory://{user_id}/facts")
async def list_facts(user_id: str) -> str:
    memory = get_router(user_id).private_memory
    memory.reload_if_stale()
    facts = memory.semantic_facts
    if not facts:
        return "Нет фактов."
    out = [f"📚 Всего фактов: {len(facts)}"]
    for f in facts[:20]:
        out.append(f"  • {f.text[:100]}... (ID: {f.id})")
    if len(facts) > 20:
        out.append(f"  ... и ещё {len(facts)-20}")
    return "\n".join(out)

@mcp.resource("memory://{user_id}/fact/{fact_id}")
async def get_fact(user_id: str, fact_id: str) -> str:
    memory = get_router(user_id).private_memory
    memory.reload_if_stale()
    obj = memory.store.get(fact_id)
    if not obj:
        for f in memory.semantic_facts:
            if str(f.id) == fact_id:
                obj = memory.store.get(f.gcn_id)
                break
    if not obj:
        return f"Факт {fact_id} не найден."
    meta = obj.object if isinstance(obj.object, dict) else {}
    return (
        f"📄 Факт #{fact_id}\n"
        f"  Текст: {obj.subject}\n"
        f"  Уверенность: {obj.confidence:.3f}\n"
        f"  Автор: {obj.author}\n"
        f"  Создан: {obj.created.isoformat()}\n"
        f"  Версия: {obj.version}\n"
        f"  Свидетельств: {len(obj.evidence)}\n"
        f"  Скоуп: {obj.scope.value}"
    )

# ------------------------------------------------------------
# ЗАПУСК
# ------------------------------------------------------------
if __name__ == "__main__":
    logger.info("🚀 Запуск рефакторированного MCP сервера BlockcoinWitres...")
    mcp.run()