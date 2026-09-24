"""
identity_tools.py — внутренние инструменты для работы с append-only
цепочкой цифровой идентичности (ТЕКУЩЕЕ_Я) из браузерного чата.

Те же три операции, что и в MCP-сервере (contribute_to_identity /
get_identity_chain / invalidate_identity), но без авторизации по user_id:
controller.memory_service уже принадлежит текущему пользователю чата.

Идентичность живёт в SHARED-слое (GCNMemoryRouter._shared_instance —
процесс-широкий синглтон), поэтому изменения через чат видны всем
пользователям и всем MCP-клиентам этого процесса, и наоборот.
"""

import logging
from dataclasses import asdict

from GCN.identity_core import (
    append_snapshot,
    invalidate_snapshot,
    get_chain,
    get_heads,
)

logger = logging.getLogger(__name__)


def register(registry, controller):
    """Регистрирует identity-инструменты в ToolRegistry текущего контроллера."""
    service = controller.memory_service

    async def _contribute(args):
        return await append_snapshot(
            service,
            content=args["content"],
            contributor_model=args["contributor_model"],
            session_id=args.get("session_id"),
            open_question=args.get("open_question"),
            parent_id=args.get("parent_id"),
        )

    async def _get_chain(args):
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
            },
            "required": ["content", "contributor_model"],
        },
        handler=_contribute,
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
        "get_identity_chain, invalidate_identity"
    )