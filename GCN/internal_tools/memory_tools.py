"""
memory_tools.py — регистрация внутренних инструментов памяти.

Вынесено из CognitiveController.__init__ (routes/ai_assistant.py) для
уменьшения размера конструктора и возможности тестировать инструменты
памяти отдельно.
"""
import logging
from typing import Dict

from GCN.tool_router import ToolRegistry

logger = logging.getLogger(__name__)


def register(registry: ToolRegistry, controller) -> None:
    """
    Регистрирует инструменты памяти в переданном реестре.

    Args:
        registry:   экземпляр ToolRegistry, куда регистрируются инструменты
        controller: CognitiveController — источник controller.memory_service
    """

    async def _internal_recall(args: Dict) -> str:
        query = args.get("query", "")
        top_k = args.get("top_k", 5)
        scope = args.get("scope")
        results = await controller.memory_service.recall(query, top_k, scope)
        if not results:
            return "Ничего не найдено."
        lines = []
        for f in results[:top_k]:
            lines.append(f"- {f['text']} (уверенность: {f.get('confidence', 0.5):.2f})")
        return "\n".join(lines)

    async def _internal_remember(args: Dict) -> str:
        fact = args.get("fact", "")
        scope = args.get("scope")  # None = автодетекция
        # ИСПРАВЛЕНИЕ: внутренний инструмент вызывается по решению LLM в ответ
        # на явную просьбу пользователя «запомни ...». Это осознанный запрос,
        # а не автоизвлечение из web-поиска — пропускаем фильтр фактологичности
        # (см. user_explicit в MemoryService.remember). Без этого флага простые
        # пользовательские факты без цифр/факт-глаголов («мой любимый цвет
        # синий», «меня зовут Иван») молча отклонялись бы с reason=not_factual.
        result = await controller.memory_service.remember(
            fact, scope, user_explicit=True)
        if result.get("id"):
            returned_fact = result.get("fact") or fact
            return f"Запомнил: {returned_fact} (скоуп: {result.get('scope', 'unknown')})"
        # Сюда попадаем только если remember() вернул rejected/error —
        # отдаём модели причину вместо «Не удалось запомнить», чтобы она
        # могла переформулировать и попробовать снова.
        reason = result.get("reason", "unknown")
        return f"Не удалось запомнить: {reason}"

    async def _internal_add_goal(args: Dict) -> str:
        description = args.get("description", "")
        priority = args.get("priority", 0.5)
        result = await controller.memory_service.add_goal(description, priority)
        return f"Цель добавлена: {description} (приоритет: {priority})"

    registry.register(
        name="recall",
        description="Поиск в памяти по запросу. Аргументы: query (str), top_k (int, опционально), scope (str, опционально: private/shared/global)",
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "top_k": {"type": "integer", "default": 5},
                "scope": {"type": "string", "enum": ["private", "shared", "global"]}
            },
            "required": ["query"]
        },
        handler=_internal_recall,
        server="internal"
    )
    registry.register(
        name="remember",
        description="Запоминает факт. Аргументы: fact (str), scope (str, опционально: private/shared/global, по умолчанию private)",
        parameters={
            "type": "object",
            "properties": {
                "fact": {"type": "string"},
                "scope": {"type": "string", "enum": ["private", "shared", "global"]}
            },
            "required": ["fact"]
        },
        handler=_internal_remember,
        server="internal"
    )
    registry.register(
        name="add_goal",
        description="Добавляет новую цель. Аргументы: description (str), priority (float, опционально, 0-1)",
        parameters={
            "type": "object",
            "properties": {
                "description": {"type": "string"},
                "priority": {"type": "number", "default": 0.5}
            },
            "required": ["description"]
        },
        handler=_internal_add_goal,
        server="internal"
    )
    logger.debug("[internal_tools.memory] зарегистрированы: recall, remember, add_goal")
