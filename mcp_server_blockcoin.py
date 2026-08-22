#!/usr/bin/env python3
"""
MCP Сервер для BlockcoinWitres (GCN Cognitive Memory) – Рефакторинг
Использует общие утилиты и методы контроллера.
"""
import asyncio
import logging
import sys
from pathlib import Path
from typing import Optional
from datetime import datetime

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

# Инициализация памяти для одного пользователя
DEFAULT_USER = "default_user"
router = GCNMemoryRouter(DEFAULT_USER, Path(MEMORY_BASE_DIR))
logger.info(f"Память загружена для {DEFAULT_USER}")

mcp = FastMCP("BlockcoinWitres Memory", description="Когнитивная память с веб-поиском и генерацией")

# ------------------------------------------------------------
# ИНСТРУМЕНТЫ (используют общие утилиты)
# ------------------------------------------------------------

@mcp.tool()
async def recall(query: str, top_k: int = 5, scope: Optional[str] = None) -> str:
    """Поиск в памяти с фильтром по скоупу."""
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
async def remember(fact: str, scope: str = "private") -> str:
    """Сохранить факт в указанный скоуп."""
    scope_map = {"private": MemoryScope.PRIVATE, "shared": MemoryScope.SHARED, "global": MemoryScope.GLOBAL}
    scope_enum = scope_map.get(scope.lower(), MemoryScope.PRIVATE)
    obj_id = router.add_knowledge(
        subject=fact,
        predicate="is_fact",
        obj="true",
        scope=scope_enum,
        confidence=0.7,
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
async def forget(query: str) -> str:
    """Удалить факты из личной памяти по ключевым словам."""
    memory = router.private_memory
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
async def research_topic(topic: str, depth: int = 2) -> str:
    """
    Глубокое исследование темы с использованием общего контроллера.
    Получаем ассистента для DEFAULT_USER и вызываем его метод research.
    """
    assistant = await get_assistant(DEFAULT_USER)
    result = await assistant.research(topic)  # возвращает Dict с answer, hypotheses, evidence
    out = [f"🔬 Исследование по теме: {topic}\n"]
    out.append("Гипотезы:\n- " + "\n- ".join(result.get("hypotheses", [])))
    out.append("\nДоказательства:")
    for ev in result.get("evidence", [])[:6]:
        out.append(f"• {ev['title']} – {ev['source']}")
    out.append("\nВывод:\n" + result.get("answer", "Нет ответа."))
    return "\n".join(out)

@mcp.tool()
async def get_episodes(limit: int = 5) -> str:
    memory = router.private_memory
    episodes = memory.episodic_memory[-limit:] if memory.episodic_memory else []
    if not episodes:
        return "Нет эпизодов."
    out = [f"📜 Последние {len(episodes)} диалогов:"]
    for ep in reversed(episodes):
        out.append(f"User: {ep.user_msg[:100]}")
        out.append(f"AI: {ep.assistant_msg[:100]}\n---")
    return "\n".join(out)

@mcp.tool()
async def get_contradictions(limit: int = 5) -> str:
    memory = router.private_memory
    pairs = memory.get_unverified_contradictions(limit=limit)
    if not pairs:
        return "Нет неразрешённых противоречий."
    out = ["⚠️ Найдены противоречия:"]
    for i, (a, b) in enumerate(pairs, 1):
        out.append(f"{i}. A: {a.text} (conf: {a.confidence:.2f})")
        out.append(f"   B: {b.text} (conf: {b.confidence:.2f})")
    return "\n".join(out)

@mcp.tool()
async def resolve_contradiction(fact_id_a: str, fact_id_b: str, verdict: str, reason: str = "") -> str:
    """Ручное разрешение противоречия."""
    memory = router.private_memory
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
async def get_goals() -> str:
    goals = await router.private_memory.get_active_goals()
    if not goals:
        return "Нет активных целей."
    out = ["🎯 Активные цели:"]
    for g in goals:
        out.append(f"  • {g.description} (приоритет: {g.priority:.2f})")
    return "\n".join(out)

@mcp.tool()
async def add_goal(description: str, priority: float = 0.5) -> str:
    gid = await router.private_memory.add_goal(description, priority)
    return f"✅ Цель добавлена (ID: {gid})"

@mcp.tool()
async def semantic_search(query: str, top_k: int = 5) -> str:
    memory = router.private_memory
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
async def graph_explore(seed_text: str, depth: int = 2) -> str:
    memory = router.private_memory
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
async def get_memory_stats() -> str:
    stats = router.private_memory.get_stats()
    return (
        f"📊 Статистика:\n"
        f"  Фактов: {stats.get('semantic_facts', 0)}\n"
        f"  Эпизодов: {stats.get('episodes', 0)}\n"
        f"  Синапсов: {stats.get('synapses', 0)}\n"
        f"  Активных целей: {stats.get('active_goals', 0)}"
    )

# ------------------------------------------------------------
# РЕСУРСЫ (остаются без изменений)
# ------------------------------------------------------------
@mcp.resource("memory://facts")
async def list_facts() -> str:
    facts = router.private_memory.semantic_facts
    if not facts:
        return "Нет фактов."
    out = [f"📚 Всего фактов: {len(facts)}"]
    for f in facts[:20]:
        out.append(f"  • {f.text[:100]}... (ID: {f.id})")
    if len(facts) > 20:
        out.append(f"  ... и ещё {len(facts)-20}")
    return "\n".join(out)

@mcp.resource("memory://fact/{fact_id}")
async def get_fact(fact_id: str) -> str:
    obj = router.private_memory.store.get(fact_id)
    if not obj:
        for f in router.private_memory.semantic_facts:
            if str(f.id) == fact_id:
                obj = router.private_memory.store.get(f.gcn_id)
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