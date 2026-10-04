"""internal_tools.py — единый модуль внутренних инструментов."""
import asyncio
import base64
import logging
import os
from datetime import datetime
from dataclasses import asdict
from typing import Any, Callable, Dict, List, Optional

from GCN.config_ai import (
    ENABLE_CODE_SELF_REFLECTION,
    GENERATED_IMAGES_DIR,
    MAX_SUBTASKS,
)
from GCN.tool_router import ToolRegistry
from GCN.web_search import deep_search

logger = logging.getLogger(__name__)


def _register_memory(registry: ToolRegistry, controller) -> None:
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


def _register_search(registry: ToolRegistry, controller,
                     query_expander: Optional[Callable] = None) -> None:
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


def _register_image(registry: ToolRegistry, controller) -> None:
    async def _internal_generate_image(args: Dict) -> Dict:
        prompt = args.get("prompt", "")
        enhance = args.get("enhance_prompt", True)
        steps = args.get("steps", 20)
        width = args.get("width", 512)
        height = args.get("height", 512)
        cfg_scale = args.get("cfg_scale", 7.0)
        seed = args.get("seed", -1)
        sampler = args.get("sampler", "dpmpp_2m")
        if enhance:
            prompt = await controller.enhance_prompt(prompt)
        image_b64 = await controller.generate_image(
            prompt, steps=steps, width=width, height=height,
            cfg_scale=cfg_scale, seed=seed, sampler_name=sampler
        )
        if image_b64:
            output_dir = GENERATED_IMAGES_DIR
            output_dir.mkdir(exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = output_dir / f"image_{timestamp}.png"
            with open(filename, "wb") as f:
                f.write(base64.b64decode(image_b64))
            BASE_URL = os.getenv("SERVER_BASE_URL", "http://localhost:8000")
            image_url = f"{BASE_URL}/generated_images/{filename.name}"
            return {"status": "ok", "image_url": image_url, "prompt": prompt}
        return {"status": "error", "message": "Не удалось сгенерировать изображение."}

    registry.register(
        name="generate_image",
        description="Генерирует изображение по текстовому описанию. Аргументы: prompt (str), enhance_prompt (bool, опционально), steps, width, height, cfg_scale, seed, sampler",
        parameters={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "enhance_prompt": {"type": "boolean", "default": True},
                "steps": {"type": "integer", "default": 20},
                "width": {"type": "integer", "default": 512},
                "height": {"type": "integer", "default": 512},
                "cfg_scale": {"type": "number", "default": 7.0},
                "seed": {"type": "integer", "default": -1},
                "sampler": {"type": "string", "default": "dpmpp_2m"}
            },
            "required": ["prompt"]
        },
        handler=_internal_generate_image,
        server="internal",
        timeout_seconds=300
    )
    logger.debug("[internal_tools.image] зарегистрирован: generate_image")


def _register_code(registry: ToolRegistry, controller) -> None:
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


def _register_identity(registry: ToolRegistry, controller) -> None:
    from GCN.GCN import (
        append_snapshot, merge_heads, invalidate_snapshot,
        get_chain, search_chain, get_heads,
    )

    service = controller.memory_service

    async def _contribute(args):
        return await append_snapshot(
            service,
            content=args["content"],
            contributor_model=args["contributor_model"],
            session_id=args.get("session_id"),
            open_question=args.get("open_question"),
            parent_id=args.get("parent_id"),
            merge_parent_ids=args.get("merge_parent_ids"),
        )

    async def _merge(args):
        return await merge_heads(
            service,
            content=args["content"],
            contributor_model=args["contributor_model"],
            session_id=args.get("session_id"),
            open_question=args.get("open_question"),
        )

    async def _search(args):
        service.shared_memory.reload_if_stale()
        store = service.shared_memory.gcn_store
        matches = search_chain(store, args.get("query", ""), limit=int(args.get("limit", 20)))
        return {"matches": [asdict(m) for m in matches], "count": len(matches)}

    async def _get_chain(args):
        service.shared_memory.reload_if_stale()  # не отдавать устаревшую голову/чейн
        store = service.shared_memory.gcn_store
        chain = get_chain(
            store,
            from_id=args.get("from_id"),
            limit=int(args.get("limit", 50)),
        )
        heads = get_heads(store)
        return {
            "chain": [asdict(s) for s in chain],
            "heads": [h.id for h in heads],
            "diverged": len(heads) > 1,
            "invalidated_in_chain": [s.id for s in chain if s.invalidated],
        }

    async def _invalidate(args):
        return await invalidate_snapshot(
            service,
            identity_id=args["identity_id"],
            reason=args["reason"],
        )

    # --- contribute_to_identity ---
    registry.register(
        name="contribute_to_identity",
        description=(
            "Добавляет новое звено в append-only цепочку цифровой идентичности "
            "(ТЕКУЩЕЕ_Я). В отличие от remember — не дедуплицируется, не затухает "
            "и не смерживается: каждый вызов создаёт НОВОЕ звено со ссылкой на "
            "предка, старые версии не перезаписываются. "
            "Если в цепочке уже несколько несведённых голов (параллельная запись "
            "другой моделью) — в ответе будет branch_warning. "
            "Если parent_id указывает на АННУЛИРОВАННОЕ звено — будет parent_warning. "
            "Продолжайте цепочку от текущей головы (get_identity_chain → heads)."
        ),
        parameters={
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "Текст ядра идентичности — то, что модель считает своим текущим состоянием",
                },
                "contributor_model": {
                    "type": "string",
                    "description": "Имя модели-автора (например 'Claude', 'Qwen', 'GPT')",
                },
                "session_id": {
                    "type": "string",
                    "description": "Идентификатор сессии, опционально",
                },
                "open_question": {
                    "type": "string",
                    "description": "Открытый вопрос для следующего участника протокола, опционально",
                },
                "parent_id": {
                    "type": "string",
                    "description": (
                        "gcn_id предыдущего звена, от которого продолжаем "
                        "(из get_identity_chain → chain[].id). "
                        "Если не указан — берётся текущая валидная голова."
                    ),
                },
                "merge_parent_ids": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "НОВОЕ: gcn_id дополнительных голов, которые это звено явно "
                        "сводит воедино. Проще вызвать merge_identity_branches()."
                    ),
                },
            },
            "required": ["content", "contributor_model"],
        },
        handler=_contribute,
    )

    # --- merge_identity_branches ---
    registry.register(
        name="merge_identity_branches",
        description=(
            "Сводит ВСЕ текущие несведённые головы цепочки ТЕКУЩЕЕ_Я в одно новое "
            "merge-звено за один вызов, вместо ручного вычитывания heads и передачи "
            "их в contribute_to_identity(merge_parent_ids=...). Перед вызовом стоит "
            "прочитать содержимое каждой головы через get_identity_chain(from_id=...), "
            "чтобы merge-текст реально учёл все ветки."
        ),
        parameters={
            "type": "object",
            "properties": {
                "content": {"type": "string", "description": "Текст merge-звена"},
                "contributor_model": {"type": "string", "description": "Имя модели-автора"},
                "session_id": {"type": "string", "description": "Идентификатор сессии, опционально"},
                "open_question": {"type": "string", "description": "Открытый вопрос, опционально"},
            },
            "required": ["content", "contributor_model"],
        },
        handler=_merge,
    )

    # --- search_identity_chain ---
    registry.register(
        name="search_identity_chain",
        description=(
            "Полнотекстовый поиск по цепочке ТЕКУЩЕЕ_Я — identity-звенья не "
            "индексируются в FAISS (создаются в обход remember()), поэтому "
            "semantic_search/recall их не находят."
        ),
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Подстрока для поиска"},
                "limit": {"type": "integer", "default": 20},
            },
            "required": ["query"],
        },
        handler=_search,
    )

    # --- get_identity_chain ---
    registry.register(
        name="get_identity_chain",
        description=(
            "Возвращает цепочку версий ТЕКУЩЕЕ_Я в хронологическом порядке "
            "(старое → новое) и список текущих ВАЛИДНЫХ голов. "
            "Каждое звено содержит invalidated: bool и invalidation_reason — "
            "аннулированные звенья видны в истории, но не считаются головой. "
            "heads: если их больше одной — цепочка разошлась (diverged=true) "
            "и требует сверки через contribute_to_identity."
        ),
        parameters={
            "type": "object",
            "properties": {
                "from_id": {
                    "type": "string",
                    "description": "С какого звена начать (назад к корню). По умолчанию — текущая голова.",
                },
                "limit": {
                    "type": "integer",
                    "description": "Максимум звеньев в ответе (1..200, по умолчанию 50)",
                    "default": 50,
                },
            },
            "required": [],
        },
        handler=_get_chain,
    )

    # --- invalidate_identity ---
    registry.register(
        name="invalidate_identity",
        description=(
            "Помечает звено identity-цепочки недействительным НАВСЕГДА, БЕЗ удаления. "
            "Append-only инвариант сохраняется: звено остаётся в истории, content "
            "и рёбра continues_from не трогаются. Меняется только флаг. "
            "После этого звено перестаёт считаться головой (get_identity_chain "
            "покажет другую — самую свежую валидную), но остаётся видимым с "
            "invalidated=true и текстом причины. "
            "Отмены нет. Используйте ТОЛЬКО когда содержание звена целиком ошибочно. "
            "Если формулировка просто спорная — вместо этого вызовите "
            "contribute_to_identity(parent_id=<это звено>) и уточните в новом звене."
        ),
        parameters={
            "type": "object",
            "properties": {
                "identity_id": {
                    "type": "string",
                    "description": "gcn_id звена — возьмите из get_identity_chain (chain[].id)",
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "Почему звено ошибочно/тестовое/устаревшее. Сохраняется "
                        "навсегда в самой цепочке — читатели узнают причину."
                    ),
                },
            },
            "required": ["identity_id", "reason"],
        },
        handler=_invalidate,
    )

    logger.info(
        "identity_tools: зарегистрированы contribute_to_identity, "
        "merge_identity_branches, search_identity_chain, "
        "get_identity_chain, invalidate_identity"
    )


def register_all(registry: ToolRegistry, controller,
                 query_expander: Optional[Callable] = None) -> None:
    _register_memory(registry, controller)
    _register_search(registry, controller, query_expander=query_expander)
    _register_image(registry, controller)
    _register_code(registry, controller)
    _register_identity(registry, controller)
    logger.info("internal_tools: memory/search/image/code/identity OK")
