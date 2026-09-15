"""
code_tools.py — регистрация внутренних инструментов самоанализа кода.

Вынесено из CognitiveController.__init__ (routes/ai_assistant.py).

Инструменты:
  - read_code          — чтение файла исходного кода проекта
  - search_code        — поиск паттерна в коде
  - project_structure  — дерево структуры проекта
  - analyze_error      — анализ traceback и предложения по исправлению

Все инструменты используют code_analyzer (GCN.code_analyzer.get_analyzer).
Регистрация происходит ТОЛЬКО если ENABLE_CODE_SELF_REFLECTION=True —
это условие переезжает сюда из CognitiveController.__init__.
"""
import logging
from typing import Dict

from GCN.config_ai import ENABLE_CODE_SELF_REFLECTION
from GCN.tool_router import ToolRegistry

logger = logging.getLogger(__name__)


def register(registry: ToolRegistry, controller) -> None:
    """
    Регистрирует инструменты самоанализа кода в переданном реестре.

    Args:
        registry:   экземпляр ToolRegistry
        controller: CognitiveController — пока не используется, но принимается
                    для единой сигнатуры со всеми internal_tools-модулями
    """
    if not ENABLE_CODE_SELF_REFLECTION:
        logger.info("Самоанализ кода отключён в конфигурации")
        return

    from GCN.code_analyzer import get_analyzer
    code_analyzer = get_analyzer()

    async def _internal_read_code(args: Dict) -> str:
        """Читает файл исходного кода проекта."""
        file_path = args.get("path", "")
        max_lines = args.get("max_lines", 500)
        if not file_path:
            return "Ошибка: не указан путь к файлу (аргумент 'path')."
        return await code_analyzer.read_file(file_path, max_lines=max_lines)

    async def _internal_search_code(args: Dict) -> str:
        """Ищет паттерн в коде проекта."""
        pattern = args.get("pattern", "")
        max_results = args.get("max_results", 20)
        if not pattern:
            return "Ошибка: не указан поисковый запрос (аргумент 'pattern')."
        return await code_analyzer.search_in_code(pattern, max_results=max_results)

    async def _internal_project_structure(args: Dict) -> str:
        """Возвращает структуру проекта."""
        max_depth = args.get("max_depth", 3)
        return await code_analyzer.get_project_structure(max_depth=max_depth)

    async def _internal_analyze_error(args: Dict) -> str:
        """Анализирует ошибку и предлагает исправления."""
        error_message = args.get("error", "")
        traceback_str = args.get("traceback", "")
        if not error_message:
            return "Ошибка: не указано сообщение об ошибке (аргумент 'error')."
        return await code_analyzer.analyze_error_location(error_message, traceback_str)

    registry.register(
        name="read_code",
        description="Читает файл исходного кода проекта. Используй для анализа своей работы. Аргументы: path (str, путь относительно корня, например 'GCN/config_ai.py'), max_lines (int, опционально)",
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Путь к файлу"},
                "max_lines": {"type": "integer", "default": 500}
            },
            "required": ["path"]
        },
        handler=_internal_read_code,
        server="internal"
    )
    registry.register(
        name="search_code",
        description="Ищет текст или паттерн в коде проекта. Аргументы: pattern (str), max_results (int, опционально)",
        parameters={
            "type": "object",
            "properties": {
                "pattern": {"type": "string", "description": "Поисковый запрос"},
                "max_results": {"type": "integer", "default": 20}
            },
            "required": ["pattern"]
        },
        handler=_internal_search_code,
        server="internal"
    )
    registry.register(
        name="project_structure",
        description="Возвращает структуру проекта в виде дерева",
        parameters={
            "type": "object",
            "properties": {
                "max_depth": {"type": "integer", "default": 3}
            }
        },
        handler=_internal_project_structure,
        server="internal"
    )
    registry.register(
        name="analyze_error",
        description="Анализирует ошибку по traceback и предлагает исправления. Аргументы: error (str, сообщение ошибки), traceback (str, трассировка стека)",
        parameters={
            "type": "object",
            "properties": {
                "error": {"type": "string", "description": "Сообщение об ошибке"},
                "traceback": {"type": "string", "description": "Трассировка стека"}
            },
            "required": ["error"]
        },
        handler=_internal_analyze_error,
        server="internal"
    )
    logger.info("Инструменты самоанализа кода зарегистрированы")
