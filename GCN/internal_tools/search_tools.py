"""
search_tools.py — регистрация внутренних инструментов поиска.

Вынесено из CognitiveController.__init__ (routes/ai_assistant.py).

Инструменты:
  - web_search         — поиск в интернете через deep_search
  - list_tools         — список зарегистрированных инструментов
  - fetch_github_file  — чтение файла/папки из публичного GitHub

query_expander передаётся снаружи (сейчас — _search_query_expander из
ai_assistant.py) чтобы избежать импорта GCN → routes.
"""
import asyncio
import logging
from typing import Any, Callable, Dict, List, Optional

from GCN.config_ai import MAX_SUBTASKS
from GCN.tool_router import ToolRegistry
from GCN.web_search import deep_search

logger = logging.getLogger(__name__)


def register(
    registry: ToolRegistry,
    controller,
    query_expander: Optional[Callable[[str], Any]] = None,
) -> None:
    """
    Регистрирует инструменты поиска в переданном реестре.

    Args:
        registry:        экземпляр ToolRegistry
        controller:      CognitiveController — источник _web_search_results_this_turn,
                         tool_registry, enhance_prompt
        query_expander:  async def(query: str) -> List[str] | None — расширитель
                         поискового запроса (см. _search_query_expander в ai_assistant.py)
    """

    async def _internal_web_search(args: Dict) -> str:
        # Единый поиск за ход: если детерминированный поиск уже выполнен
        # (см. _force_search_if_requested), не дёргаем DDG повторно.
        if controller._web_search_results_this_turn:
            acc = controller._web_search_results_this_turn[-1]
            acc_ctx = acc.get("context", "")
            return (f"Поиск уже выполнен в этом ходе. Найдено "
                    f"{len(acc.get('sources', []))} источников.\n{acc_ctx[:2500]}")
        query = (args.get("query") or "").strip()
        queries_arg = args.get("queries")
        if isinstance(queries_arg, list) and queries_arg:
            query_list = [str(q).strip() for q in queries_arg if str(q).strip()]
        elif query:
            query_list = [query]
        else:
            return "Не указан запрос для поиска (нужен query или queries)."
        query_list = query_list[:MAX_SUBTASKS]

        max_results = args.get("max_results", 5)
        results = await asyncio.gather(
            *[deep_search(q, max_results=max_results,
                          query_expander=query_expander) for q in query_list],
            return_exceptions=True
        )

        seen_urls = set()
        merged_sources: List[Dict] = []
        context_parts: List[str] = []
        any_ok = False
        for q, data in zip(query_list, results):
            if isinstance(data, Exception):
                logger.warning(f"internal__web_search: запрос '{q}' упал с ошибкой: {data}")
                continue
            if not data.get("search_performed"):
                continue
            any_ok = True
            for s in data.get("sources", []):
                url = s.get("url")
                if url and url not in seen_urls:
                    seen_urls.add(url)
                    merged_sources.append(s)
            if data.get("context"):
                label = f"[Подзапрос: {q}]\n" if len(query_list) > 1 else ""
                context_parts.append(f"{label}{data['context']}")

        merged_context = "\n\n---\n\n".join(context_parts)

        if any_ok:
            controller._web_search_results_this_turn.append({
                "queries": query_list,
                "sources": merged_sources,
                "context": merged_context,
            })

        if not any_ok:
            return "Поиск не дал результатов."
        if merged_context:
            return (f"Найдено {len(merged_sources)} источников по "
                    f"{len(query_list)} запрос(ам).\n{merged_context[:2500]}")
        return "Ничего не найдено."

    async def _internal_list_tools(args: Dict) -> str:
        lines = []
        for name, spec in controller.tool_registry._tools.items():
            lines.append(f"- {name}: {spec.description[:200]}")
        if not lines:
            return "Инструменты не зарегистрированы."
        return "Доступные инструменты:\n" + "\n".join(lines)

    async def _internal_fetch_github_file(args: Dict) -> str:
        path = args.get("path", "")
        repo = args.get("repo", "Jasst/BlockcoinWitres")
        branch = args.get("branch", "main")
        max_lines = args.get("max_lines", 500)
        if not path:
            return "Ошибка: не указан путь к файлу (path)."

        is_dir = path.endswith("/") or "." not in path.split("/")[-1]

        if is_dir:
            api_url = f"https://api.github.com/repos/{repo}/contents/{path.lstrip('/')}?ref={branch}"
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(api_url, timeout=15) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            files = [item["name"] for item in data if item["type"] == "file"]
                            dirs = [item["name"] for item in data if item["type"] == "dir"]
                            result = f"Содержимое директории `{path}`:\n"
                            if dirs:
                                result += "📁 Папки: " + ", ".join(dirs) + "\n"
                            if files:
                                result += "📄 Файлы: " + ", ".join(files) + "\n"
                            return result
                        else:
                            return f"Ошибка API GitHub: {resp.status}"
            except Exception as e:
                return f"Ошибка: {str(e)}"
        else:
            url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path.lstrip('/')}"
            try:
                import aiohttp
                async with aiohttp.ClientSession() as session:
                    async with session.get(url, timeout=15) as resp:
                        if resp.status == 200:
                            content = await resp.text()
                            lines = content.splitlines()
                            if len(lines) > max_lines:
                                content = "\n".join(
                                    lines[:max_lines]) + f"\n... (обрезано, всего {len(lines)} строк)"
                            return f"Файл {path} из репозитория {repo}:\n\n```\n{content}\n```"
                        else:
                            return f"Ошибка загрузки: {resp.status}"
            except Exception as e:
                return f"Ошибка: {str(e)}"

    registry.register(
        name="web_search",
        description=(
            "Выполняет поиск в интернете (DuckDuckGo + чтение страниц). "
            "Если в запросе пользователя есть прямая ссылка (URL) — передай её как query, "
            "содержимое страницы будет прочитано напрямую. "
            "Для составного вопроса (сравнение, несколько разных фактов) передай "
            "queries — список из нескольких коротких поисковых запросов вместо одного query, "
            "они будут выполнены параллельно за один вызов. "
            "Аргументы: query (str, один запрос) ИЛИ queries (list[str], несколько запросов), "
            "max_results (int, опционально)"
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Один поисковый запрос или URL"},
                "queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Несколько поисковых запросов для составного вопроса"
                },
                "max_results": {"type": "integer", "default": 5}
            },
            "required": []
        },
        handler=_internal_web_search,
        server="internal",
        timeout_seconds=90
    )
    registry.register(
        name="list_tools",
        description="Возвращает список всех доступных инструментов с краткими описаниями",
        parameters={"type": "object", "properties": {}},
        handler=_internal_list_tools,
        server="internal"
    )
    registry.register(
        name="fetch_github_file",
        description=(
            "Загружает содержимое файла из публичного репозитория GitHub. "
            "Используй этот инструмент, когда пользователь просит прочитать файлы с GitHub. "
            "Аргументы: path (str, обязательный, путь к файлу, например 'GCN/config_ai.py'), "
            "repo (str, опционально, owner/repo, по умолчанию 'Jasst/BlockcoinWitres'), "
            "branch (str, опционально, ветка, по умолчанию 'main'), "
            "max_lines (int, опционально, максимум строк для возврата, по умолчанию 500)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Путь к файлу в репозитории"},
                "repo": {"type": "string", "description": "Репозиторий в формате owner/repo"},
                "branch": {"type": "string", "description": "Ветка"},
                "max_lines": {"type": "integer", "default": 500}
            },
            "required": ["path"]
        },
        handler=_internal_fetch_github_file,
        server="internal"
    )
    logger.debug("[internal_tools.search] зарегистрированы: web_search, list_tools, fetch_github_file")
