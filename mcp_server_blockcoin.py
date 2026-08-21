#!/usr/bin/env python3
"""
MCP Сервер для BlockcoinWitres (GCN Cognitive Memory) – Расширенная версия
Версия: 2.0

Новые инструменты:
- web_search – поиск в интернете через DuckDuckGo
- generate_image – генерация изображений через EasyDiffusion
- research_topic – глубокое исследование темы
- get_episodes – последние диалоги
- get_contradictions – список неразрешённых противоречий
- resolve_contradiction – разрешить противоречие (пометить как ложное/истинное)
- recall (улучшен) – теперь можно фильтровать по scope и автору
"""

import asyncio
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime, timezone
import re
# Добавляем корень проекта в sys.path для импорта GCN
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mcp.server.fastmcp import FastMCP
from pydantic import Field

# Импорт твоей системы
from GCN.memory_graph import GCNMemoryRouter, CognitiveMemory
from GCN.GCN import MemoryScope
from GCN.config_ai import (
    MEMORY_BASE_DIR,
    LM_STUDIO_URL,
    LM_STUDIO_API_KEY,
    LM_STUDIO_TIMEOUT,
    EASYDIFFUSION_ENABLED,
    EASYDIFFUSION_URL,
    EASYDIFFUSION_TIMEOUT,
    EASYDIFFUSION_DEFAULT_STEPS,
    EASYDIFFUSION_DEFAULT_WIDTH,
    EASYDIFFUSION_DEFAULT_HEIGHT,
    MAX_PAGES_TO_FETCH,
    SEARCH_CACHE_TTL,
    SEARCH_CACHE_MAX_SIZE,
    DDG_MIN_INTERVAL,
    DDG_MAX_RETRIES,
)
from GCN.config_ai import CONSOLIDATION_INTERVAL, DEEP_CONSOLIDATION_INTERVAL

# Для веб-поиска нужны дополнительные импорты
import aiohttp
from bs4 import BeautifulSoup
try:
    from ddgs import DDGS
    DDGS_AVAILABLE = True
except ImportError:
    DDGS_AVAILABLE = False

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s"
)
logger = logging.getLogger("blockcoin-mcp")

# ============================================================
# ИНИЦИАЛИЗАЦИЯ ПАМЯТИ И ВСПОМОГАТЕЛЬНЫХ КОМПОНЕНТОВ
# ============================================================
BASE_DIR = Path(MEMORY_BASE_DIR)
BASE_DIR.mkdir(exist_ok=True)

# Для демонстрации используем одного пользователя.
# Для многопользовательского режима создавайте роутер на каждый user_id.
DEFAULT_USER = "default_user"
router = GCNMemoryRouter(DEFAULT_USER, BASE_DIR)

logger.info(f"Память загружена для пользователя {DEFAULT_USER}")

# ============================================================
# ВСПОМОГАТЕЛЬНЫЕ ФУНКЦИИ ДЛЯ ВЕБ-ПОИСКА (скопированы из ai_assistant)
# ============================================================
class SearchCache:
    def __init__(self, ttl=SEARCH_CACHE_TTL, maxsize=SEARCH_CACHE_MAX_SIZE):
        self.ttl = ttl
        self.maxsize = maxsize
        self._cache = {}
        self._lock = asyncio.Lock()

    async def get(self, key):
        async with self._lock:
            if key not in self._cache:
                return None
            value, ts = self._cache[key]
            if time.time() - ts > self.ttl:
                del self._cache[key]
                return None
            return value

    async def set(self, key, value):
        async with self._lock:
            self._cache[key] = (value, time.time())
            if len(self._cache) > self.maxsize:
                oldest = min(self._cache.items(), key=lambda x: x[1][1])[0]
                del self._cache[oldest]

# Инициализируем кэш и сессию для веб-запросов
search_cache = SearchCache()
_session = None

async def get_session():
    global _session
    if _session is None or _session.closed:
        _session = aiohttp.ClientSession(
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9"
            },
            timeout=aiohttp.ClientTimeout(total=15)
        )
    return _session

def _hash_query(q: str) -> str:
    import hashlib
    return hashlib.sha256(q.lower().strip().encode()).hexdigest()[:16]

async def _content_has_currency_numbers(text: str) -> bool:
    import re
    if not text:
        return False
    patterns = [
        r'\b\d{1,3}[.,]\d{2}\b',
        r'\b\d{1,3}\.\d{2}\s*(?:₽|руб|RUB|USD|EUR)\b',
        r'(?:USD|EUR|RUB)\s*/\s*(?:RUB|USD|EUR)\s*[:=]?\s*\d{1,3}[.,]\d{2}',
        r'курс.*\d{1,3}[.,]\d{2}',
    ]
    for pat in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return True
    return False

def _extract_text_from_html(html: str) -> str:
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
        if len(text) > 6000:
            text = text[:6000] + "\n...[truncated]"
        return text
    except Exception:
        return ""

async def _search_ddg(query: str, max_results: int = 5) -> List[Dict]:
    if not DDGS_AVAILABLE:
        return []
    loop = asyncio.get_event_loop()
    ddgs = DDGS()
    try:
        results = await loop.run_in_executor(
            None,
            lambda: list(ddgs.text(query, max_results=max_results))
        )
        return [{"title": r.get("title", ""), "url": r.get("href", ""), "snippet": r.get("body", "")} for r in results]
    except Exception as e:
        logger.warning(f"DDG search error: {e}")
        return []

async def _fetch_page(url: str) -> str:
    try:
        session = await get_session()
        async with session.get(url, allow_redirects=True, max_redirects=5) as resp:
            if resp.status != 200:
                return ""
            content_type = resp.headers.get("Content-Type", "")
            if "text/html" not in content_type and "application/xhtml" not in content_type:
                return ""
            html = await resp.text()
            return _extract_text_from_html(html)
    except Exception:
        return ""

async def _deep_search(query: str, max_results: int = MAX_PAGES_TO_FETCH) -> Dict[str, Any]:
    """Упрощённая версия deep_search из ai_assistant.py для MCP."""
    cache_key = _hash_query(query)
    cached = await search_cache.get(cache_key)
    if cached:
        if await _content_has_currency_numbers(cached.get("context", "")):
            return cached
        return cached

    ddg_results = await _search_ddg(query, max_results=max_results + 2)
    if not ddg_results:
        return {"sources": [], "context": "Поиск не дал результатов.", "search_performed": False}

    urls = [r["url"] for r in ddg_results if r.get("url")]
    # Загружаем до max_results страниц параллельно
    sem = asyncio.Semaphore(5)
    async def fetch_one(url):
        async with sem:
            return await _fetch_page(url)
    tasks = [fetch_one(u) for u in urls[:max_results]]
    page_texts = await asyncio.gather(*tasks, return_exceptions=True)

    sources = []
    context_parts = []
    url_to_title = {r["url"]: r["title"] for r in ddg_results}
    for url, text in zip(urls[:max_results], page_texts):
        if isinstance(text, Exception) or not text:
            # Если нет текста, используем сниппет
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
    await search_cache.set(cache_key, result)
    return result

# ============================================================
# ФУНКЦИЯ ВЫЗОВА LLM (для research_topic)
# ============================================================
async def _call_llm(messages, temp=0.7, max_tokens=2048):
    """Вызов LM Studio (аналогично ai_assistant)."""
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {LM_STUDIO_API_KEY}"}
    payload = {"model": "local-model", "messages": messages, "temperature": temp, "max_tokens": max_tokens}
    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(LM_STUDIO_URL, json=payload, headers=headers, timeout=LM_STUDIO_TIMEOUT) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    return data.get("choices", [{}])[0].get("message", {}).get("content", "")
                else:
                    return ""
    except Exception as e:
        logger.error(f"LLM call error: {e}")
        return ""

# ============================================================
# СОЗДАНИЕ MCP СЕРВЕРА
# ============================================================
mcp = FastMCP(
    "BlockcoinWitres Memory",
    description="Расширенная когнитивная память с веб-поиском, генерацией изображений и исследованием."
)

# ============================================================
# НОВЫЕ И УЛУЧШЕННЫЕ ИНСТРУМЕНТЫ
# ============================================================

@mcp.tool()
async def recall(
    query: str = Field(description="Поисковый запрос"),
    top_k: int = Field(default=5, description="Количество результатов"),
    scope: Optional[str] = Field(default=None, description="Фильтр по скоупу: private, shared, global")
) -> str:
    """
    Найти информацию в памяти. Можно отфильтровать по скоупу.
    """
    try:
        # Получаем все результаты через router.retrieve
        results = await router.retrieve(query, top_k=top_k * 2, include_private=True)
        # Фильтруем по scope, если указан
        if scope:
            scope_lower = scope.lower()
            filtered = []
            for item in results:
                gcn_id = item.get("gcn_id")
                if gcn_id:
                    obj = (
                            router.private_memory.store.get(gcn_id) or
                            router.shared_memory.store.get(gcn_id) or
                            router.global_memory.store.get(gcn_id)
                    )
                    if obj and obj.scope.value == scope_lower:
                        filtered.append(item)
                else:
                    # если нет gcn_id, пропускаем (это может быть общий факт без scope)
                    pass
            results = filtered[:top_k]
        else:
            results = results[:top_k]

        if not results:
            return "Ничего не найдено."

        output = []
        for i, item in enumerate(results, 1):
            text = item.get("text", "")
            conf = item.get("confidence", 0.5)
            importance = item.get("importance", 1.0)
            source = "личная" if item.get("gcn_id") and router.private_memory.store.get(item["gcn_id"]) else "общая"
            output.append(f"{i}. {text} (уверенность: {conf:.2f}, важность: {importance:.2f}, источник: {source})")
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Recall error: {e}", exc_info=True)
        return f"Ошибка: {str(e)}"


@mcp.tool()
async def web_search(
    query: str = Field(description="Поисковый запрос для интернета"),
    max_results: int = Field(default=5, description="Максимум результатов")
) -> str:
    """
    Выполнить поиск в интернете через DuckDuckGo и вернуть извлечённый текст со страниц.
    """
    if not DDGS_AVAILABLE:
        return "⚠️ Библиотека ddgs не установлена. Установите: pip install ddgs"
    try:
        data = await _deep_search(query, max_results=max_results)
        if not data["search_performed"]:
            return "Поиск не дал результатов."
        output = []
        output.append(f"🔍 Результаты поиска по запросу: {query}")
        for src in data["sources"]:
            output.append(f"• {src['title']} – {src['url']}")
        output.append("\n--- Извлечённый контекст ---\n")
        output.append(data["context"])
        return "\n".join(output)
    except Exception as e:
        logger.error(f"Web search error: {e}", exc_info=True)
        return f"Ошибка при поиске: {str(e)}"


@mcp.tool()
async def generate_image(
    prompt: str = Field(description="Описание изображения"),
    steps: int = Field(default=EASYDIFFUSION_DEFAULT_STEPS, description="Количество шагов"),
    width: int = Field(default=EASYDIFFUSION_DEFAULT_WIDTH, description="Ширина"),
    height: int = Field(default=EASYDIFFUSION_DEFAULT_HEIGHT, description="Высота")
) -> str:
    """
    Сгенерировать изображение по текстовому описанию через EasyDiffusion.
    Возвращает base64-кодированное изображение (или сообщение об ошибке).
    """
    if not EASYDIFFUSION_ENABLED:
        return "Генерация изображений отключена в конфиге (EASYDIFFUSION_ENABLED=False)."
    try:
        payload = {
            "prompt": prompt,
            "steps": steps,
            "width": width,
            "height": height,
        }
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{EASYDIFFUSION_URL}/generate", json=payload,
                                    timeout=aiohttp.ClientTimeout(total=EASYDIFFUSION_TIMEOUT)) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    image_b64 = data.get("image_base64")
                    if image_b64:
                        # Возвращаем base64, обёрнутый в маркер для отображения
                        return f"![Сгенерированное изображение](data:image/png;base64,{image_b64})"
                    else:
                        return "❌ Сервер EasyDiffusion вернул пустой ответ."
                else:
                    error_text = await resp.text()
                    return f"❌ Ошибка EasyDiffusion: {resp.status} - {error_text[:200]}"
    except Exception as e:
        logger.error(f"Generate image error: {e}", exc_info=True)
        return f"❌ Ошибка генерации: {str(e)}"


@mcp.tool()
async def research_topic(
    topic: str = Field(description="Тема для исследования"),
    depth: int = Field(default=2, description="Глубина (количество итераций, 1-3)")
) -> str:
    """
    Провести глубокое исследование темы: сформулировать гипотезы, собрать доказательства,
    сделать выводы. Использует LLM и веб-поиск.
    """
    try:
        # Формируем гипотезы через LLM
        prompt = (
            f"Сформулируй 3 чёткие, проверяемые гипотезы по вопросу: {topic}. "
            "Каждая гипотеза должна быть кратким утверждением (не вопросом). "
            "Ответь в виде маркированного списка, без пояснений."
        )
        hypotheses_text = await _call_llm([{"role": "user", "content": prompt}], temp=0.8, max_tokens=300)
        if not hypotheses_text:
            return "Не удалось сформулировать гипотезы (ошибка LLM)."

        hypotheses = [h.strip("-• ").strip() for h in hypotheses_text.split('\n') if h.strip()][:3]
        if not hypotheses:
            hypotheses = ["Не удалось сгенерировать гипотезы"]

        # Собираем доказательства (поиск по теме и по гипотезам)
        all_evidence = []
        queries = [topic] + hypotheses[:2]
        for q in queries:
            data = await _deep_search(q, max_results=3)
            for src in data.get("sources", []):
                all_evidence.append({"source": src.get("url", ""), "title": src.get("title", ""), "query": q})

        evidence_text = "\n".join([f"- {e['title']}: {e['source']} (запрос: {e['query']})" for e in all_evidence[:6]])

        # Формируем итоговый ответ через LLM
        answer_prompt = (
            f"На основе следующих гипотез и собранных доказательств дай развёрнутый ответ на вопрос: {topic}.\n"
            "Укажи уверенность (0-1) для каждого утверждения и приведи аргументы.\n"
            "Структурируй ответ: вступление, основная часть с аргументацией, заключение.\n\n"
            f"Гипотезы: {', '.join(hypotheses)}\n\n"
            f"Источники:\n{evidence_text}"
        )
        answer = await _call_llm([{"role": "user", "content": answer_prompt}], temp=0.6, max_tokens=1500)
        if not answer:
            answer = "Не удалось сформировать финальный ответ (ошибка LLM)."

        return f"🔬 **Исследование по теме: {topic}**\n\n**Гипотезы:**\n- " + "\n- ".join(hypotheses) + "\n\n**Доказательства:**\n" + evidence_text + "\n\n**Вывод:**\n" + answer
    except Exception as e:
        logger.error(f"Research error: {e}", exc_info=True)
        return f"Ошибка исследования: {str(e)}"


@mcp.tool()
async def get_episodes(
    limit: int = Field(default=5, description="Количество последних эпизодов")
) -> str:
    """
    Показать последние диалоги (эпизоды) из личной памяти.
    """
    memory = router.private_memory
    episodes = memory.episodic_memory[-limit:] if memory.episodic_memory else []
    if not episodes:
        return "Нет сохранённых эпизодов."
    output = [f"📜 Последние {len(episodes)} диалогов:"]
    for ep in reversed(episodes):
        output.append(f"User: {ep.user_msg[:100]}")
        output.append(f"AI: {ep.assistant_msg[:100]}")
        output.append("---")
    return "\n".join(output)


@mcp.tool()
async def get_contradictions(
    limit: int = Field(default=5, description="Максимум пар противоречий")
) -> str:
    """
    Показать неразрешённые противоречия в памяти (пары фактов).
    """
    memory = router.private_memory
    pairs = memory.get_unverified_contradictions(limit=limit)
    if not pairs:
        return "Нет неразрешённых противоречий."
    output = ["⚠️ Найдены противоречия:"]
    for idx, (a, b) in enumerate(pairs, 1):
        output.append(f"{idx}. A: {a.text} (уверенность: {a.confidence:.2f})")
        output.append(f"   B: {b.text} (уверенность: {b.confidence:.2f})")
        output.append("")
    return "\n".join(output)


@mcp.tool()
async def resolve_contradiction(
    fact_id_a: str = Field(description="ID первого факта (можно указать только один, если второй не нужен)"),
    fact_id_b: str = Field(description="ID второго факта"),
    verdict: str = Field(description="Решение: 'A' - оставить A, 'B' - оставить B, 'both' - оставить оба, 'neither' - удалить оба"),
    reason: str = Field(default="", description="Причина решения (опционально)")
) -> str:
    """
    Разрешить противоречие между двумя фактами вручную.
    """
    memory = router.private_memory
    # Поиск фактов по ID (может быть gcn_id или локальный id)
    def find_fact_by_id(fid):
        if fid in memory.facts_by_id:
            return memory.facts_by_id[fid]
        for f in memory.semantic_facts:
            if f.gcn_id == fid:
                return f
        return None

    fact_a = find_fact_by_id(fact_id_a)
    fact_b = find_fact_by_id(fact_id_b)
    if not fact_a or not fact_b:
        return f"Один из фактов не найден: A={fact_id_a}, B={fact_id_b}"

    verdict_lower = verdict.lower()
    if verdict_lower == "a":
        # Удалить B
        memory._remove_facts({fact_b.id})
        # Удалить противоречие из графа
        try:
            memory.gcn_store._graph.remove_relation(fact_a.gcn_id, "CONTRADICTS", fact_b.gcn_id)
            memory.gcn_store._graph.remove_relation(fact_b.gcn_id, "CONTRADICTS", fact_a.gcn_id)
        except:
            pass
        fact_a.contradicts.discard(fact_b.id)
        fact_b.contradicts.discard(fact_a.id)
        await memory._schedule_save()
        return f"✅ Противоречие разрешено: оставлен факт A: {fact_a.text}"
    elif verdict_lower == "b":
        memory._remove_facts({fact_a.id})
        try:
            memory.gcn_store._graph.remove_relation(fact_a.gcn_id, "CONTRADICTS", fact_b.gcn_id)
            memory.gcn_store._graph.remove_relation(fact_b.gcn_id, "CONTRADICTS", fact_a.gcn_id)
        except:
            pass
        fact_a.contradicts.discard(fact_b.id)
        fact_b.contradicts.discard(fact_a.id)
        await memory._schedule_save()
        return f"✅ Противоречие разрешено: оставлен факт B: {fact_b.text}"
    elif verdict_lower == "both":
        # Оставляем оба, но убираем противоречие
        try:
            memory.gcn_store._graph.remove_relation(fact_a.gcn_id, "CONTRADICTS", fact_b.gcn_id)
            memory.gcn_store._graph.remove_relation(fact_b.gcn_id, "CONTRADICTS", fact_a.gcn_id)
        except:
            pass
        fact_a.contradicts.discard(fact_b.id)
        fact_b.contradicts.discard(fact_a.id)
        await memory._schedule_save()
        return f"✅ Противоречие снято, оба факта сохранены."
    elif verdict_lower == "neither":
        # Удалить оба
        memory._remove_facts({fact_a.id, fact_b.id})
        await memory._schedule_save()
        return f"✅ Оба факта удалены."
    else:
        return f"Неизвестный вердикт: {verdict}. Допустимые: A, B, both, neither."


# ============================================================
# ОСТАВШИЕСЯ ИНСТРУМЕНТЫ (без изменений, но с улучшенной документацией)
# ============================================================

@mcp.tool()
async def remember(
    fact: str = Field(description="Текст факта для запоминания"),
    scope: str = Field(default="private", description="private | shared | global")
) -> str:
    """
    Сохранить факт в память.
    - private: личная память пользователя.
    - global: публичное знание с дедупликацией.
    """
    # (код остаётся как в предыдущей версии)
    scope_map = {
        "private": MemoryScope.PRIVATE,
        "shared": MemoryScope.SHARED,
        "global": MemoryScope.GLOBAL,
    }
    scope_enum = scope_map.get(scope.lower(), MemoryScope.PRIVATE)
    try:
        obj_id = router.add_knowledge(
            subject=fact,
            predicate="is_fact",
            obj="true",
            scope=scope_enum,
            confidence=0.7,
            source_type="mcp_tool"
        )
        if scope_enum == MemoryScope.GLOBAL:
            await router.global_memory._schedule_save()
        elif scope_enum == MemoryScope.PRIVATE:
            await router.private_memory._schedule_save()
        elif scope_enum == MemoryScope.SHARED:
            await router.shared_memory._schedule_save()
        return f"✅ Сохранён (ID: {obj_id}, скоуп: {scope})"
    except Exception as e:
        logger.error(f"Remember error: {e}", exc_info=True)
        return f"❌ Ошибка: {str(e)}"


@mcp.tool()
async def forget(
    query: str = Field(description="Ключевые слова для удаления")
) -> str:
    """Удалить факты из личной памяти по текстовому совпадению."""
    memory = router.private_memory
    to_remove = [f.id for f in memory.semantic_facts if query.lower() in f.text.lower()]
    if not to_remove:
        return "Ничего не найдено."
    try:
        removed = memory._remove_facts(set(to_remove))
        await memory._schedule_save()
        return f"✅ Удалено {removed} фактов"
    except Exception as e:
        logger.error(f"Forget error: {e}", exc_info=True)
        return f"❌ Ошибка: {str(e)}"


@mcp.tool()
async def get_goals() -> str:
    """Показать активные цели."""
    try:
        goals = await router.private_memory.get_active_goals()
        if not goals:
            return "Нет активных целей."
        output = ["🎯 Активные цели:"]
        for g in goals:
            output.append(f"  • {g.description} (приоритет: {g.priority:.2f})")
        return "\n".join(output)
    except Exception as e:
        return f"Ошибка: {str(e)}"


@mcp.tool()
async def add_goal(
    description: str = Field(description="Описание цели"),
    priority: float = Field(default=0.5, description="Приоритет 0-1")
) -> str:
    """Добавить новую цель."""
    try:
        goal_id = await router.private_memory.add_goal(description, priority)
        return f"✅ Цель добавлена (ID: {goal_id})"
    except Exception as e:
        return f"❌ Ошибка: {str(e)}"


@mcp.tool()
async def semantic_search(
    query: str = Field(description="Запрос по смыслу"),
    top_k: int = Field(default=5, description="Количество результатов")
) -> str:
    """Поиск только по эмбеддингам (без графа)."""
    memory = router.private_memory
    emb = memory.embed_text(query)
    if emb is None:
        return "Эмбеддинги не доступны."
    results = memory.store.semantic_search(emb, top_k=top_k * 2)
    output = []
    for gcn_id, score in results[:top_k]:
        obj = memory.store.get(gcn_id)
        if obj:
            output.append(f"• {obj.subject} (сходство: {score:.3f})")
    return "\n".join(output) if output else "Ничего не найдено."


@mcp.tool()
async def graph_explore(
    seed_text: str = Field(description="Текст для старта обхода"),
    depth: int = Field(default=2, description="Глубина 1-3")
) -> str:
    """Исследовать ассоциативный граф через spreading activation."""
    memory = router.private_memory
    seed_ids = [f.id for f in memory.semantic_facts if seed_text.lower() in f.text.lower()]
    if not seed_ids:
        return f"Не найдено фактов с '{seed_text}'"
    activation = await memory.spread_activation(seed_ids[:3], max_depth=min(depth, 3))
    sorted_items = sorted(activation.items(), key=lambda x: x[1], reverse=True)
    output = [f"🧠 Обход графа (старт: '{seed_text}')"]
    count = 0
    for fid, act in sorted_items:
        if fid in seed_ids:
            continue
        fact = memory.facts_by_id.get(fid)
        if fact:
            output.append(f"  • {fact.text[:80]}... (активация: {act:.3f})")
            count += 1
            if count >= 10:
                break
    if count == 0:
        output.append("  Нет связанных узлов.")
    return "\n".join(output)


@mcp.tool()
async def get_memory_stats() -> str:
    """Статистика памяти."""
    stats = router.private_memory.get_stats()
    return (
        f"📊 Статистика:\n"
        f"  Фактов: {stats.get('semantic_facts', 0)}\n"
        f"  Эпизодов: {stats.get('episodes', 0)}\n"
        f"  Синапсов: {stats.get('synapses', 0)}\n"
        f"  Активных целей: {stats.get('active_goals', 0)}"
    )


# ============================================================
# РЕСУРСЫ (без изменений)
# ============================================================

@mcp.resource("memory://facts")
async def list_all_facts() -> str:
    """Список всех фактов (первые 20)."""
    facts = router.private_memory.semantic_facts
    if not facts:
        return "Нет фактов."
    output = [f"📚 Всего фактов: {len(facts)}"]
    for f in facts[:20]:
        output.append(f"  • {f.text[:100]}... (ID: {f.id})")
    if len(facts) > 20:
        output.append(f"  ... и ещё {len(facts) - 20}")
    return "\n".join(output)


@mcp.resource("memory://fact/{fact_id}")
async def get_fact_by_id(fact_id: str) -> str:
    """Детальная информация о факте по ID."""
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


# ============================================================
# ЗАПУСК
# ============================================================
if __name__ == "__main__":
    logger.info("🚀 Запуск расширенного MCP сервера BlockcoinWitres...")
    mcp.run()