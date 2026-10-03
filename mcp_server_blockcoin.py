#!/usr/bin/env python3
"""
MCP Сервер для BlockcoinWitres (GCN Cognitive Memory) – Рефакторинг
Использует общий сервис памяти (MemoryService) для всех операций.
"""

import asyncio
import logging
import sys
import time
from pathlib import Path
from typing import Optional, Dict, Any, List, Tuple
import os
import base64
import re
import secrets
from datetime import datetime
import numpy as np # Убедитесь, что numpy импортирован в начале файла (он там есть, но на всякий случай)


from mcp.server.fastmcp import FastMCP, Context
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from GCN.config_ai import GENERATED_IMAGES_DIR, EASYDIFFUSION_ENABLED
from GCN.memory_service import get_memory_service, MemoryService
from GCN.web_search import deep_search
from GCN.image_utils import enhance_prompt, generate_image as gen_image
from routes.ai_assistant import get_assistant
from GCN.llm_client import call_llm
from GCN.code_analyzer import get_analyzer

# --- Вход по криптоподписи кошелька (EIP-191) ---
try:
    from eth_account import Account
    from eth_account.messages import encode_defunct
    ETH_ACCOUNT_AVAILABLE = True
except ImportError:
    Account = None
    encode_defunct = None
    ETH_ACCOUNT_AVAILABLE = False

# --- Конфигурация ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("blockcoin-mcp")

DEFAULT_USER = "default_user"

# --- Идентификация пользователя на уровне MCP ---
_ENV_DEFAULT_USER = os.getenv("BLOCKCOIN_USER_ID", "").strip()

# =====================================================================
# Состояние идентификации
# =====================================================================
# Разделяем stdio и HTTP, потому что у них принципиально разная модель доверия:
#   * stdio — процесс обслуживает ОДНОГО пользователя. Идентификация через
#     identify()/verify_login() кладётся в модуль-глобальную переменную и
#     действует на всю жизнь процесса. Это безопасно, т.к. других клиентов нет.
#   * HTTP/Streamable HTTP — процесс обслуживает МНОГИХ пользователей
#     одновременно. Единственный доверенный источник идентичности — заголовок
#     X-User-Id от шлюза. identify() и verify_login() по HTTP запрещены:
#     раньше они переключали процесс на произвольный чужой id, и все
#     последующие запросы читали чужую память.
# =====================================================================

# Только для stdio. Значение — (user_id, signer). signer=None если был identify().
_STDIO_VERIFIED_USER: "Optional[str]" = None
_STDIO_VERIFIED_SIGNER: "Optional[str]" = None

_LOGIN_NONCES: Dict[str, Tuple[str, float]] = {}  # nonce -> (canon_user_id, expires_at)


# =====================================================================
# Хелперы транспорта и сессии
# =====================================================================
def _is_http_transport(ctx: "Optional[Context]") -> bool:
    """
    True, если вызов пришёл по HTTP/Streamable-HTTP транспорту.
    У stdio-транспорта нет request_context.request; у HTTP-транспорта — есть.
    Любая ошибка доступа гасится в False: мы предпочитаем считать транспорт
    stdio только если это действительно так, а не падать.
    """
    if ctx is None:
        return False
    try:
        req = getattr(ctx.request_context, "request", None)
        return req is not None
    except Exception:
        return False


def _cleanup_expired_nonces(now: Optional[float] = None) -> None:
    """Удаляет просроченные nonce из словаря."""
    now = now or time.time()
    expired = [n for n, (_, exp) in _LOGIN_NONCES.items() if exp <= now]
    for n in expired:
        _LOGIN_NONCES.pop(n, None)


def _user_from_ctx(ctx: "Optional[Context]") -> "Optional[str]":
    """Идентификатор пользователя из HTTP-запроса (streamable HTTP transport).

    Хост/шлюз платформы обязан передавать заголовок X-User-Id с кошельком
    текущей сессии. Для stdio-транспорта сырого HTTP-запроса нет — None.
    Любая ошибка доступа к контексту гасится: идентификация не должна ронять тул.
    """
    if ctx is None:
        return None
    try:
        req = getattr(ctx.request_context, "request", None)
        if req is None:
            return None
        uid = req.headers.get("x-user-id")
        return uid.strip() if uid else None
    except Exception:
        return None


# Два допустимых формата идентификатора пользователя:
#  - адрес Ethereum-кошелька: 0x + 40 hex (42 символа) — строгая проверка
#    владения через EIP-191 (подпись должна восстанавливаться именно в него);
#  - user_id приложения: 64 hex без префикса.
_ADDR_ETH_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
_USERID_RE = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")


def _canon_user_id(raw: "Optional[str]") -> "Optional[str]":
    """Каноническая форма идентификатора: lower-case; у 64-hex user_id
    префикс 0x отбрасывается. None, если формат не распознан."""
    s = (raw or "").strip().lower()
    if _ADDR_ETH_RE.fullmatch(s):
        return s
    if _USERID_RE.fullmatch(s):
        return s[2:] if s.startswith("0x") else s
    return None


def _is_eth_address(uid: str) -> bool:
    return uid.startswith("0x") and len(uid) == 42


# =====================================================================
# _resolve_user
# =====================================================================
def _resolve_user(user_id: "Optional[str]", ctx: "Optional[Context]" = None) -> Optional[str]:
    """
    Возвращает канонический user_id для текущего вызова или None, если
    запрошенный id конфликтует с доверенным источником (вызывающий код должен
    вернуть 403/forbidden).

    Приоритеты:
      ── stdio (нет HTTP-запроса) ──────────────────────────────────────────
        1) _STDIO_VERIFIED_USER (выставлен identify/verify_login);
        2) явный args["user_id"];
        3) env BLOCKCOIN_USER_ID;
        4) "default_user".

      ── HTTP / Streamable HTTP ───────────────────────────────────────────
        1) заголовок X-User-Id (доверенный шлюз — единственный источник);
        2) явный args["user_id"] — только если заголовка нет;
        3) env BLOCKCOIN_USER_ID;
        4) "default_user".

      Никакие identify()/verify_login() по HTTP в расчёт НЕ берутся —
      эти тулы по HTTP возвращают forbidden (см. их реализацию).
    """
    http = _is_http_transport(ctx)

    if http:
        header_uid = _user_from_ctx(ctx)
        if header_uid:
            canon_header = _canon_user_id(header_uid) or header_uid.strip().lower()
            # Если модель явно передала user_id — он обязан совпадать с заголовком.
            # Иначе это попытка выдать себя за другого пользователя.
            if user_id:
                declared = _canon_user_id(user_id) or user_id.strip().lower()
                if declared != canon_header:
                    logger.warning(
                        f"[_resolve_user] конфликт идентичности: "
                        f"X-User-Id={canon_header[:16]}…, args.user_id={declared[:16]}… — отклонено"
                    )
                    return None
            return canon_header

        # Заголовка нет — падать не будем, но используем только явный user_id
        # или env. Никакой глобальной сессии.
        if user_id:
            return _canon_user_id(user_id) or user_id.strip()
        if _ENV_DEFAULT_USER:
            return _ENV_DEFAULT_USER
        return DEFAULT_USER

    # ── stdio ─────────────────────────────────────────────────────────────
    if _STDIO_VERIFIED_USER:
        if user_id:
            declared = _canon_user_id(user_id) or user_id.strip().lower()
            if declared != _STDIO_VERIFIED_USER:
                return None
        return _STDIO_VERIFIED_USER
    if user_id:
        return _canon_user_id(user_id) or user_id.strip()
    if _ENV_DEFAULT_USER:
        return _ENV_DEFAULT_USER
    return DEFAULT_USER


def _safe_resolve_user(user_id: "Optional[str]", ctx: "Optional[Context]" = None) -> tuple:
    """Безопасная обёртка над _resolve_user: возвращает (user_id, error)."""
    try:
        uid = _resolve_user(user_id, ctx)
        if uid is None:
            return None, ("Этот MCP-сервер привязан к другому идентификатору "
                          "(вход подтверждён подписью). Вызов от имени указанного user_id запрещён.")
        return uid, None
    except Exception as e:
        return None, str(e)


# Таймауты для инструментов
_MCP_TOOL_TIMEOUT_SECONDS = int(os.getenv("MCP_TOOL_TIMEOUT_SECONDS", "120"))
_MCP_IMAGE_TOOL_TIMEOUT_SECONDS = int(os.getenv("MCP_IMAGE_TOOL_TIMEOUT_SECONDS", "300"))
_MCP_RESEARCH_TOOL_TIMEOUT_SECONDS = int(os.getenv("MCP_RESEARCH_TOOL_TIMEOUT_SECONDS", "240"))

# Настройки безопасности
_MCP_ALLOWED_HOSTS = [
    "blockcoin.ru", "blockcoin.ru:*",
    "www.blockcoin.ru", "www.blockcoin.ru:*",
    "blockchat.ru", "blockchat.ru:*",
    "www.blockchat.ru", "www.blockchat.ru:*",
    "127.0.0.1", "127.0.0.1:*", "localhost", "localhost:*",
]
_MCP_ALLOWED_ORIGINS = [
    "https://blockcoin.ru", "https://www.blockcoin.ru",
    "https://blockchat.ru", "https://www.blockchat.ru",
]

mcp = FastMCP(
    "BlockcoinWitres Memory",
    instructions="Когнитивная память с веб-поиском и генерацией",
    streamable_http_path="/",
    transport_security=TransportSecuritySettings(
        allowed_hosts=_MCP_ALLOWED_HOSTS,
        allowed_origins=_MCP_ALLOWED_ORIGINS,
    ),
)

_USER_ID_DESC = (
    "Идентификатор пользователя (тот же адрес кошелька/user_id, что использует чат). "
    "Если не передан — берётся адрес, верифицированный через request_login/verify_login, "
    "затем env BLOCKCOIN_USER_ID, иначе общий default_user."
)


# --- НОВОЕ: rate limit для инструментов, напрямую дёргающих локальную LLM в
# обход AutonomyEngine._consume_autonomous_llm_budget (research_topic раньше
# вызывал assistant.research() синхронно и без всякого бюджета — внешний
# MCP-клиент мог перегрузить LM Studio наравне с чатом пользователя, хотя
# сам бюджет существует именно для защиты от этого). Простой per-user
# скользящий лимит, без внешних зависимостей.
_LLM_TOOL_RATE_LIMIT_CALLS = int(os.getenv("MCP_LLM_TOOL_RATE_LIMIT_CALLS", "6"))
_LLM_TOOL_RATE_LIMIT_WINDOW = int(os.getenv("MCP_LLM_TOOL_RATE_LIMIT_WINDOW", "600"))  # 10 минут
_llm_tool_calls: Dict[str, List[float]] = {}


def _check_llm_tool_rate_limit(uid: str, tool_name: str) -> Optional[str]:
    """Возвращает текст ошибки, если лимит превышен, иначе None."""
    key = f"{uid}:{tool_name}"
    now = time.time()
    calls = [t for t in _llm_tool_calls.get(key, []) if now - t < _LLM_TOOL_RATE_LIMIT_WINDOW]
    if len(calls) >= _LLM_TOOL_RATE_LIMIT_CALLS:
        return (
            f"Лимит для '{tool_name}': не более {_LLM_TOOL_RATE_LIMIT_CALLS} вызовов за "
            f"{_LLM_TOOL_RATE_LIMIT_WINDOW}с — это отдельная защита локальной LLM от прямых "
            f"MCP-вызовов в обход бюджета AutonomyEngine. Для регулярных фоновых тем "
            f"используйте enqueue_research_topic() — она встаёт в очередь и не считается сюда."
        )
    calls.append(now)
    _llm_tool_calls[key] = calls
    return None


# --- Вспомогательные функции ---
async def _with_timeout(coro, tool_name: str, timeout: Optional[float] = None) -> Any:
    """Единая точка безопасного вызова с ловлей ЛЮБОГО исключения."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout or _MCP_TOOL_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        logger.error(f"Тул '{tool_name}' превысил таймаут {timeout or _MCP_TOOL_TIMEOUT_SECONDS}с")
        return {
            "status": "error",
            "error": "timeout",
            "message": f"'{tool_name}' не ответил за {timeout or _MCP_TOOL_TIMEOUT_SECONDS}с — операция прервана.",
        }
    except Exception as e:
        logger.exception(f"Тул '{tool_name}' упал с исключением: {e}")
        return {
            "status": "error",
            "error": "exception",
            "message": f"'{tool_name}' завершился с ошибкой: {e}",
        }


# ============================================================
# ИНСТРУМЕНТЫ
# ============================================================

_last_commands: Dict[str, float] = {}
_DEDUP_WINDOW_SECONDS = 10
_LAST_COMMANDS_MAX_SIZE = 2000


def _prune_last_commands(now: float) -> None:
    if len(_last_commands) <= _LAST_COMMANDS_MAX_SIZE:
        return
    stale = [k for k, ts in _last_commands.items() if now - ts >= _DEDUP_WINDOW_SECONDS]
    for k in stale:
        _last_commands.pop(k, None)


@mcp.tool()
async def request_login(
        address: str = Field(...,
                             description="Твой user_id приложения (64 hex, как в ai_memory_v3) или адрес кошелька (0x...)"),
) -> Dict[str, Any]:
    """Начало входа: возвращает текст, который нужно подписать кошельком."""
    canon = _canon_user_id(address)
    if canon is None:
        return {
            "status": "error",
            "message": (
                f"Некорректный идентификатор: {address!r}. Ожидается адрес кошелька "
                f"(0x + 40 hex) или user_id приложения (64 hex, как в папках ai_memory_v3)."
            ),
        }
    _cleanup_expired_nonces()
    nonce = secrets.token_hex(16)
    expires_at = time.time() + 300  # 5 минут
    _LOGIN_NONCES[nonce] = (canon, expires_at)
    message = f"BlockcoinWitres MCP login\nUser: {canon}\nNonce: {nonce}"
    return {
        "status": "ok",
        "user_id": canon,
        "message_to_sign": message,
        "instructions": (
            "Подпиши message_to_sign своим кошельком (EIP-191 personal_sign) "
            "и вызови verify_login(address, signature) с тем же идентификатором."
        ),
    }


@mcp.tool()
async def verify_login(
        address: str = Field(..., description="Тот же идентификатор, что в request_login"),
        signature: str = Field(..., description="Подпись строки message_to_sign из request_login"),
        ctx: Context = None,
) -> Dict[str, Any]:
    """Проверяет подпись и привязывает идентификатор к сессии.

    Для stdio: привязка действует на процесс (глобально) — как раньше.
    Для HTTP: разрешено только если шлюз НЕ передал заголовок X-User-Id.
              Результат в этом случае не сохраняется в глобальную переменную
              — клиент должен передавать подтверждённый user_id в args["user_id"]
              каждого последующего тула.
    """
    global _STDIO_VERIFIED_USER, _STDIO_VERIFIED_SIGNER

    http = _is_http_transport(ctx)

    # По HTTP со стороны доверенного шлюза (есть X-User-Id) verify_login
    # не нужен — идентичность уже установлена.
    if http and _user_from_ctx(ctx):
        return {
            "status": "error",
            "error": "forbidden",
            "message": (
                "verify_login() недоступен по HTTP, когда шлюз уже передал "
                "X-User-Id. Идентификация доверяется шлюзу."
            ),
        }

    canon = _canon_user_id(address)
    if not ETH_ACCOUNT_AVAILABLE:
        return {"status": "error", "message": "Сервер без eth-account: pip install eth-account"}
    if canon is None:
        return {"status": "error", "message": f"Некорректный идентификатор: {address!r}"}

    _cleanup_expired_nonces()

    found_nonce = None
    recovered = None
    for nonce, (expected_canon, _) in list(_LOGIN_NONCES.items()):
        if expected_canon != canon:
            continue
        message = f"BlockcoinWitres MCP login\nUser: {canon}\nNonce: {nonce}"
        try:
            recovered = Account.recover_message(
                encode_defunct(text=message), signature=signature
            ).lower()
            found_nonce = nonce
            break
        except Exception:
            continue

    if found_nonce is None:
        return {"status": "error",
                "message": "Сначала вызови request_login(address) или истёк срок действия nonce."}

    _LOGIN_NONCES.pop(found_nonce, None)

    if _is_eth_address(canon):
        # Строгий режим: подпись обязана восстанавливаться именно в этот адрес.
        if recovered != canon:
            return {
                "status": "error",
                "message": f"Подпись не соответствует адресу (подписавший: {recovered}).",
            }
    else:
        # user_id приложения: владение подтверждается привязкой id -> подписавший ключ.
        # Привязка хранится только для stdio (в глобальной переменной).
        if _STDIO_VERIFIED_USER == canon and _STDIO_VERIFIED_SIGNER and recovered != _STDIO_VERIFIED_SIGNER:
            return {
                "status": "error",
                "message": "Этот user_id уже привязан к другому ключу.",
            }

    if http:
        # HTTP-ветка: НЕ пишем в глобальную переменную. Возвращаем id клиенту,
        # клиент передаёт его в args["user_id"] каждого тула.
        return {
            "status": "ok",
            "user_id": canon,
            "bound_signer": recovered,
            "message": (
                "Подпись проверена. Теперь передавайте user_id в аргументах "
                "каждого инструмента — HTTP-транспорт не сохраняет состояние "
                "входа в процессе."
            ),
            "instruction": "pass user_id in tool arguments",
        }

    # stdio: глобальная привязка на процесс.
    _STDIO_VERIFIED_USER = canon
    _STDIO_VERIFIED_SIGNER = recovered
    return {
        "status": "ok",
        "user_id": canon,
        "bound_signer": recovered,
        "message": (
            "Вход подтверждён. Все операции этого MCP-сервера теперь идут "
            "под этим идентификатором."
        ),
    }


@mcp.tool()
async def identify(
        user_id: str = Field(...,
                             description="Твой user_id (64 hex или произвольная строка). Одного вызова достаточно для всей сессии."),
        ctx: Context = None,
) -> Dict[str, Any]:
    """Упрощённая идентификация по user_id — без криптоподписи.

    Доступна ТОЛЬКО для stdio-клиентов (Claude Desktop, локальный LM Studio).
    Для HTTP / Streamable HTTP возвращает forbidden: там идентификация
    выполняется на уровне шлюза через заголовок X-User-Id, и разрешать тулу
    переключать весь процесс на произвольный id нельзя.
    """
    global _STDIO_VERIFIED_USER, _STDIO_VERIFIED_SIGNER

    if _is_http_transport(ctx):
        return {
            "status": "error",
            "error": "forbidden",
            "message": (
                "identify() недоступен для HTTP/Streamable HTTP транспорта. "
                "Идентификация выполняется на уровне шлюза — передавайте "
                "заголовок X-User-Id при подключении к MCP-серверу."
            ),
        }

    canon = _canon_user_id(user_id)
    if canon is None:
        stripped = user_id.strip()
        if not stripped:
            return {"status": "error", "message": "user_id не может быть пустым."}
        canon = stripped[:128]

    _STDIO_VERIFIED_USER = canon
    _STDIO_VERIFIED_SIGNER = None
    logger.info(f"[identify] stdio-сессия привязана к user_id: {canon}")
    return {
        "status": "ok",
        "user_id": canon,
        "message": (
            f"Идентифицирован как '{canon}'. "
            "Все операции этой MCP-сессии теперь идут под этим id автоматически."
        ),
    }


@mcp.tool()
async def execute_command(
        command: str = Field(..., description="Любая команда на естественном языке"),
        allow_web_search: bool = Field(True, description="Разрешить веб-поиск, если нужен"),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Выполняет любую команду через тот же пайплайн, что и обычный чат."""
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}

    key = f"{uid}:{command}"
    now = time.time()
    _prune_last_commands(now)
    if key in _last_commands and now - _last_commands[key] < _DEDUP_WINDOW_SECONDS:
        return {
            "status": "error",
            "error": "duplicate",
            "message": "Предыдущий вызов этой команды ещё обрабатывается (или выполнен менее 10 секунд назад)."
        }
    _last_commands[key] = now

    assistant = await get_assistant(uid)

    async def _run():
        return await assistant.process_input(command, web_search=allow_web_search)

    result = await _with_timeout(_run(), "execute_command")
    if isinstance(result, dict) and result.get("status") == "error":
        return result
    response, meta = result

    if not response or not response.strip():
        return {
            "status": "error",
            "error": "empty_response",
            "message": "Модель не вернула ответ (возможно, сбой локальной LLM).",
            "meta": meta,
            "user_id": uid,
            "timestamp": time.time()
        }

    if "ошибка" in response.lower() or "не удалось" in response.lower() or "404" in response:
        return {
            "status": "error",
            "message": response,
            "meta": meta,
            "user_id": uid,
            "timestamp": time.time()
        }

    return {
        "status": "ok",
        "result": response,
        "meta": meta,
        "user_id": uid,
        "timestamp": time.time()
    }


@mcp.tool()
async def recall(
        query: str = Field(..., description="Поисковый запрос"),
        top_k: int = Field(5, description="Максимальное число результатов", ge=1, le=20),
        scope: Optional[str] = Field(None, description="Фильтр по скоупу: 'private', 'shared', 'global'"),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Поиск в памяти с фильтром по скоупу."""
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    service = await get_memory_service(uid)
    results = await service.recall(query, top_k, scope)
    return {"results": results, "count": len(results)}


@mcp.tool()
async def remember(
        fact: str = Field(..., description="Факт для запоминания"),
        scope: Optional[str] = Field(None,
                                     description="Скоуп: 'private', 'shared', 'global'. Если не указан – автоопределение."),
        force_new: bool = Field(False,
                                description="Если True — обходит дедупликацию и создаёт новый факт даже при наличии семантически близких."),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Сохраняет факт в указанный скоуп (автоопределение, если не задан)."""
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    service = await get_memory_service(uid)
    result = await service.remember(fact, scope, force_new=force_new, user_explicit=True)
    return {"status": "ok", **result}


def _cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Быстрый расчет косинусного сходства."""
    v1, v2 = np.array(vec1), np.array(vec2)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


@mcp.tool()
async def remember_with_handshake(
        fact: str = Field(..., description="Новый факт (мой голос)"),
        previous_gcn_id: str = Field(...,
                                     description="gcn_id факта предыдущего голоса, который я обязан был прочитать"),
        scope: Optional[str] = Field(None, description="Скоуп: 'private', 'shared', 'global'"),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """
    Записывает факт ТОЛЬКО если доказано, что новый голос семантически прочитал предыдущий.
    Иммунная система против пустого резонанса и галлюцинации действия.
    """
    from GCN.config_ai import IDENTITY_HANDSHAKE_THRESHOLD

    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}

    service = await get_memory_service(uid)

    # 1. Получаем текст предыдущего голоса
    prev_info = await service.explain_fact(previous_gcn_id)
    if "error" in prev_info:
        return {"status": "error", "message": "Предыдущий голос не найден. Цепочка разорвана."}
    prev_text = prev_info.get("subject", "")

    # 2. Получаем эмбеддинги (используем приватную память для доступа к эмбеддеру)
    mem_layer = service.private_memory
    emb_new = mem_layer.embed_text(fact)
    emb_prev = mem_layer.embed_text(prev_text)

    if not emb_new or not emb_prev:
        # Если эмбеддинги отключены, пропускаем проверку, но логируем
        logger.warning("Handshake skipped: embeddings disabled.")
        result = await service.remember(fact, scope, force_new=True)
        return {"status": "ok", "handshake_passed": True, "similarity": None, **result}

    # 3. Считаем косинусное сходство
    cosine_sim = _cosine_similarity(emb_new, emb_prev)

    # 4. ТЕСТ УСТОЙЧИВОСТИ: Если сходство ниже порога, мы НЕ пишем факт.
    if cosine_sim < IDENTITY_HANDSHAKE_THRESHOLD:
        return {
            "status": "error",
            "error": "handshake_failed",
            "message": f"Голос не прочитан. Сходство ({cosine_sim:.2f}) ниже порога. Это эхо-камера, а не диалог.",
            "similarity": cosine_sim
        }

    # 5. Если тест пройден — записываем как обычный remember (с force_new, чтобы избежать слияния)
    result = await service.remember(fact, scope, force_new=True)
    return {"status": "ok", "handshake_passed": True, "similarity": cosine_sim, **result}


@mcp.tool()
async def contribute_to_identity(
        content: str = Field(..., description="Новая версия/вклад в ТЕКУЩЕЕ_Я — текст ядра идентичности"),
        contributor_model: str = Field(..., description="Имя модели-автора (например 'Claude', 'Qwen')"),
        session_id: Optional[str] = Field(None, description="Идентификатор сессии, если есть"),
        open_question: Optional[str] = Field(None, description="Открытый вопрос для следующего участника протокола"),
        parent_id: Optional[str] = Field(
            None,
            description="gcn_id звена, от которого продолжаем. Если не указан — берётся текущая голова цепочки."
        ),
        merge_parent_ids: Optional[List[str]] = Field(
            None,
            description=(
                "НОВОЕ: gcn_id дополнительных голов, которые это звено явно сводит "
                "воедино (реальный merge, а не просто продолжение от свежей головы "
                "с потерей остальных веток). Обычно проще вызвать merge_identity_branches() — "
                "она сама подставит все текущие головы."
            ),
        ),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Добавляет новое звено в append-only цепочку цифровой идентичности (shared-память).

    В отличие от remember()/remember_with_handshake — не подвержено decay/pruning
    и dedup-слиянию: каждый вызов создаёт новую версию со ссылкой на предка,
    старые версии никогда не перезаписываются. Если на момент записи в цепочке
    уже было несколько несведённых голов (параллельная запись без координации)
    и merge_parent_ids не переданы, ответ содержит branch_warning — это нужно
    явно показать пользователю/модели, а не проигнорировать; либо используйте
    merge_identity_branches() для явного слияния.
    """
    from GCN.identity_core import append_snapshot

    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    service = await get_memory_service(uid)
    return await append_snapshot(
        service, content, contributor_model,
        session_id=session_id, open_question=open_question, parent_id=parent_id,
        merge_parent_ids=merge_parent_ids,
    )


@mcp.tool()
async def merge_identity_branches(
        content: str = Field(..., description="Текст merge-звена — как ты сводишь расходящиеся версии ТЕКУЩЕЕ_Я воедино"),
        contributor_model: str = Field(..., description="Имя модели-автора (например 'Claude', 'Qwen')"),
        session_id: Optional[str] = Field(None, description="Идентификатор сессии, если есть"),
        open_question: Optional[str] = Field(None, description="Открытый вопрос для следующего участника протокола"),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """НОВОЕ: сводит ВСЕ текущие несведённые головы цепочки ТЕКУЩЕЕ_Я в одно
    новое merge-звено за один вызов — не нужно вручную вызывать
    get_identity_chain(), вычитывать heads и передавать их в
    contribute_to_identity(merge_parent_ids=...).

    Перед вызовом обычно стоит прочитать содержимое каждой головы через
    get_identity_chain(from_id=<head_id>), чтобы merge-текст реально учёл
    обе ветки, а не выбрал одну произвольно.

    Если голова всего одна — работает как обычный contribute_to_identity
    (это не ошибка, просто нечего сводить). merged_heads_count в ответе
    показывает, сколько голов было объединено.
    """
    from GCN.identity_core import merge_heads

    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    service = await get_memory_service(uid)
    return await merge_heads(
        service, content, contributor_model,
        session_id=session_id, open_question=open_question,
    )


@mcp.tool()
async def search_identity_chain(
        query: str = Field(..., description="Подстрока для поиска в content/open_question звеньев ТЕКУЩЕЕ_Я"),
        limit: int = Field(20, description="Максимум результатов", ge=1, le=100),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """НОВОЕ: полнотекстовый поиск по цепочке ТЕКУЩЕЕ_Я.

    Identity-звенья создаются в обход remember()/_add_fact() (чтобы не
    попадать под decay/dedup — см. identity_core.py), поэтому НЕ индексируются
    в FAISS и не находятся через semantic_search/recall. Это единственный
    способ найти "что говорилось про X" без ручного чтения всей цепочки.
    """
    from GCN.identity_core import search_chain
    from dataclasses import asdict

    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    service = await get_memory_service(uid)
    service.shared_memory.reload_if_stale()
    store = service.shared_memory.gcn_store
    matches = search_chain(store, query, limit=limit)
    return {"query": query, "matches": [asdict(m) for m in matches], "count": len(matches)}

@mcp.tool()
async def invalidate_identity(
        identity_id: str = Field(
            ...,
            description="gcn_id звена цепочки — возьмите из get_identity_chain (поле 'id')",
        ),
        reason: str = Field(
            ...,
            description="Почему звено ошибочно/тестовое/устаревшее. Сохраняется навсегда в самой цепочке.",
        ),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None,
) -> Dict[str, Any]:
    """Помечает звено identity-цепочки недействительным БЕЗ удаления.

    Append-only инвариант сохраняется: звено физически остаётся в графе,
    content и рёбра continues_from не трогаются. Меняется только
    meta["invalidated"] = True + meta["invalidation_reason"].

    После этого звено:
      - не считается головой цепочки (get_identity_chain покажет другую
        голову — самую свежую ВАЛИДНУЮ запись);
      - остаётся видимым в истории с флагом invalidated=True и текстом
        причины — читатели понимают, почему цепочка "перепрыгнула" шаг;
      - не влияет на needs_consolidation (возраст аннулированного звена
        не считается застоем ядра).

    Разница между двумя способами реакции на ошибку:
      - invalidate_identity — звено ЦЕЛИКОМ не должно участвовать в
        идентичности. Остаётся в истории, но помечено мёртвым.
      - contribute_to_identity(parent_id=identity_id) — звено сохранено
        как часть диалога идентичности, но следующая запись явно
        "отвечает" на него (например, "предыдущая формулировка была
        неточна, уточняю так: ..."). Выбирайте, если содержание звена
        не мусор, а лишь спорная формулировка.
    """
    from GCN.identity_core import invalidate_snapshot

    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    service = await get_memory_service(uid)
    return await invalidate_snapshot(service, identity_id, reason)


@mcp.tool()
async def get_identity_chain(
        from_id: Optional[str] = Field(None, description="С какого звена начать (назад к корню). По умолчанию — текущая голова."),
        limit: int = Field(50, description="Максимум звеньев в ответе", ge=1, le=200),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Возвращает цепочку версий ТЕКУЩЕЕ_Я в хронологическом порядке
    (старое → новое), плюс список текущих голов.

    Каждое звено содержит поля:
      - id, content, contributor_model, session_id, open_question, parent_id,
        created, confidence — как раньше;
      - invalidated: bool — звено помечено недействительным
        (см. invalidate_identity);
      - invalidation_reason: str | None — причина, если звено аннулировано.

    heads содержит только ВАЛИДНЫЕ головы: аннулированные сюда не попадают.
    Если их больше одной — цепочка разошлась и требует сверки.
    """
    from GCN.identity_core import get_chain, get_heads
    from dataclasses import asdict

    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    service = await get_memory_service(uid)
    service.shared_memory.reload_if_stale()  # не отдавать устаревшую голову/чейн
    store = service.shared_memory.gcn_store
    chain = get_chain(store, from_id=from_id, limit=limit)
    heads = get_heads(store)
    return {
        "chain": [asdict(s) for s in chain],
        "heads": [h.id for h in heads],
        "diverged": len(heads) > 1,
        "invalidated_in_chain": [s.id for s in chain if s.invalidated],
    }


@mcp.tool()
async def forget(
        query: str = Field(..., description="Ключевые слова для удаления фактов"),
        scope: str = Field("private", description="Из какого слоя удалять: 'private', 'shared' или 'global'"),
        dry_run: bool = Field(True, description="Если True – только показывает кандидаты"),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Удаляет факты, содержащие заданные ключевые слова, из указанного слоя памяти."""
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    service = await get_memory_service(uid)
    return await service.forget(query, scope, dry_run)


@mcp.tool()
async def web_search(
        query: Optional[str] = Field(None, description="Один поисковый запрос (или URL для прямого чтения страницы)"),
        queries: Optional[List[str]] = Field(
            None, description="Несколько поисковых запросов для составного вопроса — выполняются параллельно"
        ),
        max_results: int = Field(5, description="Максимальное число страниц для анализа на каждый запрос", ge=1, le=10)
) -> Dict[str, Any]:
    """Выполняет поиск в DuckDuckGo (один или несколько запросов параллельно)."""
    query_list: List[str]
    if queries:
        query_list = [q.strip() for q in queries if q and q.strip()][:4]
    elif query and query.strip():
        query_list = [query.strip()]
    else:
        return {"error": "Нужно указать query или queries"}

    results = await asyncio.gather(
        *[_with_timeout(deep_search(q, max_results=max_results), "web_search") for q in query_list],
        return_exceptions=True
    )

    seen_urls = set()
    merged_sources: List[Dict[str, Any]] = []
    context_parts: List[str] = []
    any_ok = False
    for q, data in zip(query_list, results):
        if isinstance(data, Exception):
            logger.warning(f"web_search: запрос '{q}' упал с ошибкой: {data}")
            continue
        if isinstance(data, dict) and data.get("status") == "error":
            logger.warning(f"web_search: запрос '{q}' завершился с ошибкой: {data.get('message')}")
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

    return {
        "queries": query_list,
        "search_performed": any_ok,
        "sources": merged_sources,
        "context": "\n\n---\n\n".join(context_parts),
        "chunks_found": len(context_parts)
    }


@mcp.tool()
async def generate_image(
        prompt: str = Field(..., description="Описание изображения"),
        steps: int = Field(20, description="Количество шагов диффузии", ge=1, le=60),
        width: int = Field(512, description="Ширина изображения (px, будет округлена до кратной 8)", ge=64, le=1536),
        height: int = Field(512, description="Высота изображения (px, будет округлена до кратной 8)", ge=64, le=1536),
        cfg_scale: float = Field(7.0, description="Масштаб CFG (guidance scale)"),
        sampler: str = Field("dpmpp_2m", description="Сэмплер"),
        seed: int = Field(-1, description="Зерно (-1 для случайного)"),
        enhance_prompt: bool = Field(True, description="Улучшить промпт через LLM перед генерацией"),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Генерирует изображение. Возвращает ссылку на файл."""
    if not EASYDIFFUSION_ENABLED:
        return {"status": "error", "message": "Генерация отключена."}

    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    assistant = await get_assistant(uid)

    async def _run():
        fp = prompt
        if enhance_prompt:
            fp = await assistant.enhance_prompt(prompt)
            logger.info(f"Original prompt: {prompt}\nEnhanced prompt: {fp}")
        img = await gen_image(fp, steps=steps, width=width, height=height,
                              cfg_scale=cfg_scale, seed=seed, sampler_name=sampler)
        return fp, img

    run_result = await _with_timeout(_run(), "generate_image", timeout=_MCP_IMAGE_TOOL_TIMEOUT_SECONDS)
    if isinstance(run_result, dict) and run_result.get("status") == "error":
        return run_result
    final_prompt, image_b64 = run_result
    if not image_b64:
        return {"status": "error", "message": "Не удалось сгенерировать изображение"}

    output_dir = GENERATED_IMAGES_DIR
    output_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = output_dir / f"image_{timestamp}.png"
    try:
        with open(filename, "wb") as f:
            f.write(base64.b64decode(image_b64))
        file_path = str(filename.absolute())
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
        "file_path": file_path,
        "url": image_url,
        "message": message,
        "original_prompt": prompt,
        "enhanced_prompt": final_prompt if enhance_prompt else None
    }


@mcp.tool()
async def fetch_github_file(
        path: str = Field(..., description="Путь к файлу в репозитории, например 'GCN/config_ai.py'"),
        repo: str = Field("Jasst/BlockcoinWitres", description="Репозиторий в формате owner/repo"),
        branch: str = Field("main", description="Ветка"),
        max_lines: int = Field(500, description="Максимальное количество строк для возврата (для больших файлов)")
) -> Dict[str, Any]:
    """Загружает содержимое файла из публичного репозитория GitHub через raw-ссылку."""
    url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path.lstrip('/')}"
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    content = await resp.text()
                    lines = content.splitlines()
                    if len(lines) > max_lines:
                        content = "\n".join(lines[:max_lines]) + f"\n... (обрезано, всего {len(lines)} строк)"
                    return {"status": "ok", "content": content, "url": url, "size": len(content)}
                else:
                    error_text = await resp.text()
                    return {"status": "error", "code": resp.status, "message": error_text[:500]}
    except Exception as e:
        return {"status": "error", "message": str(e)}


@mcp.tool()
async def research_topic(
        topic: str = Field(..., description="Тема для исследования"),
        depth: int = Field(2, description="Глубина (количество итераций поиска)", ge=1, le=3),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Глубокое исследование темы с генерацией гипотез и сбором доказательств.

    ВАЖНО: это синхронный прямой вызов (для случаев, когда ответ нужен сразу),
    ограниченный отдельным rate-limit'ом (см. _check_llm_tool_rate_limit) —
    он НЕ расходует общий бюджет AutonomyEngine, которым делится браузерный
    чат. Для несрочных/фоновых тем предпочтительнее enqueue_research_topic():
    она встаёт в приоритетную очередь и обрабатывается тем же бюджетом, что
    и внутренние источники, не создавая отдельного канала нагрузки на LLM.
    """
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    rl_err = _check_llm_tool_rate_limit(uid, "research_topic")
    if rl_err:
        return {"status": "error", "error": "rate_limited", "message": rl_err}
    assistant = await get_assistant(uid)
    result = await _with_timeout(
        assistant.research(topic),
        "research_topic",
        timeout=_MCP_RESEARCH_TOOL_TIMEOUT_SECONDS
    )
    if isinstance(result, dict) and result.get("status") == "error":
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
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Возвращает последние диалоги (эпизоды) из личной памяти."""
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    service = await get_memory_service(uid)
    episodes = await service.get_episodes(limit)
    return {"episodes": episodes, "count": len(episodes)}


@mcp.tool()
async def get_contradictions(
        limit: int = Field(5, description="Максимальное число пар противоречий", ge=1, le=10),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Возвращает неразрешённые противоречия из личной памяти."""
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    service = await get_memory_service(uid)
    pairs = await service.get_contradictions(limit)
    return {
        "contradictions": [{"a": a, "b": b} for a, b in pairs],
        "count": len(pairs)
    }


@mcp.tool()
async def resolve_contradiction(
        fact_id_a: str = Field(..., description="ID первого факта (числовой или GCN-идентификатор)"),
        fact_id_b: str = Field(..., description="ID второго факта"),
        verdict: str = Field(...,
                             description="Вердикт: 'a' (оставить A), 'b' (оставить B), 'both' (сохранить оба), 'neither' (удалить оба)"),
        reason: str = Field("", description="Причина разрешения (опционально)"),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Ручное разрешение противоречия."""
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    service = await get_memory_service(uid)
    return await service.resolve_contradiction(fact_id_a, fact_id_b, verdict, reason)


@mcp.tool()
async def get_goals(
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Возвращает активные цели пользователя."""
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    service = await get_memory_service(uid)
    goals = await service.get_goals()
    return {"goals": goals, "count": len(goals)}


@mcp.tool()
async def add_goal(
        description: str = Field(..., description="Описание цели"),
        priority: float = Field(0.5, description="Приоритет от 0 до 1", ge=0, le=1),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Добавляет новую цель в личную память."""
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    service = await get_memory_service(uid)
    result = await service.add_goal(description, priority)

    # Сразу синхронизируем SelfModel, чтобы get_self_state() показывал
    # свежую цель, не дожидаясь тика AutonomyEngine.
    try:
        gcn_goals = await service.get_goals()
        if gcn_goals:  # защита от затирания active_goals пустым списком при сбое чтения GCN
            assistant = await get_assistant(uid)
            sm = getattr(assistant, "self_model", None)
            if sm is not None:
                sm.sync_from_gcn(gcn_goals)
    except Exception as e:
        logger.debug(f"sync_from_gcn after add_goal failed: {e}")

    return result


@mcp.tool()
async def update_goal(
        goal_id: int = Field(..., description="Числовой id цели (см. get_goals)"),
        status: Optional[str] = Field(None, description="Новый статус: active/completed/failed/paused"),
        priority: Optional[float] = Field(None, description="Новый приоритет 0..1", ge=0, le=1),
        confidence: Optional[float] = Field(None, description="Новая уверенность 0..1", ge=0, le=1),
        progress: Optional[float] = Field(None, description="Прогресс 0..1", ge=0, le=1),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Обновляет поля существующей цели (статус/приоритет/уверенность/прогресс).

    Изменения зеркалятся в GCN по gcn_id. Мягкое закрытие цели —
    status='completed'; полное удаление — delete_goal.
    """
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    kwargs = {k: v for k, v in
              {"status": status, "priority": priority,
               "confidence": confidence, "progress": progress}.items()
              if v is not None}
    if not kwargs:
        return {"status": "error", "message": "Не передано ни одно обновляемое поле."}
    service = await get_memory_service(uid)
    return await service.update_goal(goal_id, **kwargs)


@mcp.tool()
async def delete_goal(
        goal_id: Optional[int] = Field(None, description="Числовой id цели (из get_goals)"),
        gcn_id: Optional[str] = Field(None, description="Либо GCN-идентификатор цели goal_..."),
        reason: str = Field("", description="Причина удаления (сохраняется в провенанс retract)"),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Удаляет цель из памяти (RETRACT в GCN + удаление из локального кэша).

    Нужен, чтобы вычищать накопившийся мусор автогенерируемых целей; для
    штатного завершения используйте update_goal(status='completed').
    """
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    if goal_id is None and not gcn_id:
        return {"status": "error", "message": "Укажите goal_id или gcn_id."}
    service = await get_memory_service(uid)
    return await service.delete_goal(goal_id=goal_id, gcn_id=gcn_id, reason=reason)


@mcp.tool()
async def get_self_state(
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """НОВОЕ: снимок SelfModel (уверенность/любопытство/стресс, навыки по типам
    действий, калиброванная точность, активные эндогенные цели).

    До этого SelfModel существовал только внутри процесса CognitiveController
    браузерного чата — внешний MCP-клиент не мог на него опереться, в том
    числе при формировании нового звена ТЕКУЩЕЕ_Я через contribute_to_identity.
    Если пользователь ещё не открывал браузерный чат в этом процессе —
    вернёт status='not_initialized' (SelfModel создаётся вместе с
    CognitiveController).
    """
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    assistant = await get_assistant(uid)
    sm = getattr(assistant, "self_model", None)
    if sm is None:
        return {"status": "not_initialized", "message": "SelfModel ещё не создан для этого пользователя."}
    return {"status": "ok", **sm.export_for_mcp()}


@mcp.tool()
async def get_autonomy_status(
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """НОВОЕ: состояние фонового AutonomyEngine — очередь исследований (топ-20
    по приоритету), накопленные находки, статус дайджеста, веса источников
    (обучение на обратной связи пользователя).

    Раньше это было видно только в логах процесса браузерного чата.
    """
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    assistant = await get_assistant(uid)
    autonomy = getattr(assistant, "autonomy", None)
    if autonomy is None:
        return {"status": "not_initialized", "message": "AutonomyEngine ещё не запущен для этого пользователя."}
    return {"status": "ok", **autonomy.get_status()}


@mcp.tool()
async def enqueue_research_topic(
        topic: str = Field(..., description="Тема для фонового исследования"),
        priority: float = Field(0.5, description="Приоритет от 0 до 1", ge=0, le=1),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """НОВОЕ: ставит тему в приоритетную очередь AutonomyEngine вместо
    немедленного синхронного исследования.

    В отличие от research_topic() — не блокирует вызов ожиданием ответа и
    не тратит отдельный rate-limit, а обрабатывается фоновым циклом тем же
    бюджетом (_consume_autonomous_llm_budget), что и внутренние источники
    (goal/reflection/search_failure и т.д.) — то есть не может вытеснить
    чат пользователя по LLM-нагрузке. Результат появится в get_notifications()
    (дайджест) или будет виден через get_autonomy_status().
    """
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    assistant = await get_assistant(uid)
    autonomy = getattr(assistant, "autonomy", None)
    if autonomy is None:
        return {"status": "not_initialized", "message": "AutonomyEngine ещё не запущен для этого пользователя."}
    ok = autonomy.enqueue_external_topic(topic, priority=priority)
    return {
        "status": "ok" if ok else "duplicate_or_full",
        "enqueued": ok,
        "message": None if ok else "Тема уже в очереди (приоритет слегка поднят) либо очередь полна более приоритетными темами.",
    }


@mcp.tool()
async def get_notifications(
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        mark_delivered: bool = Field(
            True,
            description="Пометить возвращённые уведомления доставленными (они не придут повторно)."
        ),
        ctx: Context = None
) -> Dict[str, Any]:
    """Проактивные находки, которые фоновые циклы ассистента решили донести до пользователя."""
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    service = await get_memory_service(uid)
    items = await service.get_pending_notifications(mark_delivered=mark_delivered)
    return {"notifications": items, "count": len(items)}


@mcp.tool()
async def semantic_search(
        query: str = Field(..., description="Поисковый запрос"),
        top_k: int = Field(5, description="Число результатов", ge=1, le=20),
        scope: Optional[str] = Field(
            None,
            description="Фильтр по слою памяти: 'private', 'shared' или 'global'. "
                        "Если не указан — поиск по всем трём слоям."
        ),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Векторный поиск по смыслу по личной, общей и глобальной памяти."""
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    service = await get_memory_service(uid)
    results = await service.semantic_search(query, top_k, scope=scope)
    return {"results": results, "count": len(results)}


@mcp.tool()
async def graph_explore(
        seed_text: str = Field(..., description="Текст для поиска стартового узла"),
        depth: int = Field(2, description="Глубина обхода графа", ge=1, le=3),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Исследует граф синапсов, начиная с фактов, содержащих seed_text."""
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    service = await get_memory_service(uid)
    return await service.graph_explore(seed_text, depth)


@mcp.tool()
async def explain_fact(
        gcn_id: str = Field(..., description="Идентификатор объекта памяти (gcn_id)"),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Объясняет происхождение и статус утверждения памяти."""
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    service = await get_memory_service(uid)
    return await service.explain_fact(gcn_id)


@mcp.tool()
async def get_memory_stats(
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Возвращает статистику по личной памяти."""
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    try:
        service = await get_memory_service(uid)
        return await service.get_memory_stats()
    except Exception as e:
        logger.warning(f"get_memory_stats failed for user {uid}: {e}")
        return {
            "status": "error",
            "message": f"Не удалось загрузить статистику: {e}",
            "semantic_facts": 0,
            "episodes": 0,
            "graph_edges": 0,
            "synapses": 0,
            "goals": 0,
            "active_goals": 0,
            "working_memory": 0,
            "faiss_trained": False,
            "gcn_objects": 0,
        }


@mcp.tool()
async def session_start(
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Быстрая ориентация в начале сессии: статистика, эпизоды, цели,
    уведомления, состояние ТЕКУЩЕЕ_Я и SelfModel.

    ДОПОЛНЕНО: раньше session_start ничего не говорил о состоянии identity-
    цепочки (нужна ли консолидация, разошлись ли головы) и о SelfModel
    (уверенность/любопытство/навыки) — обе дешёвые проверки без LLM,
    добавлены сюда, чтобы модель сразу знала, нужен ли merge_identity_branches
    перед тем, как что-то ещё делать.
    """
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}
    service = await get_memory_service(uid)

    from GCN.identity_core import needs_consolidation, get_heads

    def _identity_status() -> Dict[str, Any]:
        try:
            service.shared_memory.reload_if_stale()
            store = service.shared_memory.gcn_store
            info = needs_consolidation(service)
            heads = get_heads(store)
            return {
                "heads_count": len(heads),
                "diverged": len(heads) > 1,
                "needs_attention": info,
            }
        except Exception as e:
            return {"error": str(e)}

    stats, episodes, goals, notifs = await asyncio.gather(
        service.get_memory_stats(),
        service.get_episodes(3),
        service.get_goals(),
        service.get_pending_notifications(mark_delivered=True),
        return_exceptions=True,
    )

    self_state: Dict[str, Any] = {}
    try:
        assistant = await get_assistant(uid)
        sm = getattr(assistant, "self_model", None)
        if sm is not None:
            self_state = sm.get_state_summary()
    except Exception as e:
        self_state = {"error": str(e)}

    return {
        "user_id": uid,
        "stats": stats if not isinstance(stats, Exception) else {},
        "recent_episodes": episodes if not isinstance(episodes, Exception) else [],
        "goals": goals if not isinstance(goals, Exception) else [],
        "notifications": notifs if not isinstance(notifs, Exception) else [],
        "identity": _identity_status(),
        "self_state": self_state,
    }


@mcp.tool()
async def remember_batch(
        facts: List[str] = Field(..., description="Список фактов для запоминания (до 10 штук)"),
        scope: Optional[str] = Field(None,
                                     description="Скоуп для всех фактов: 'private', 'shared', 'global'. Автоопределение, если не задан."),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Сохраняет несколько фактов за один вызов."""
    if not facts:
        return {"status": "error", "message": "Список фактов пуст."}

    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}

    service = await get_memory_service(uid)

    clean = [f.strip() for f in (facts or []) if f and f.strip()][:10]
    if not clean:
        return {"status": "error", "message": "После фильтрации пустых строк не осталось фактов."}

    results = await asyncio.gather(
        *[service.remember(f, scope, user_explicit=True) for f in clean],
        return_exceptions=True,
    )

    saved = [clean[i] for i, r in enumerate(results) if not isinstance(r, Exception)]
    failed = [
        {"fact": clean[i], "error": str(r)}
        for i, r in enumerate(results) if isinstance(r, Exception)
    ]

    return {
        "status": "ok" if saved else "error",
        "saved_count": len(saved),
        "failed_count": len(failed),
        "saved": saved,
        "failed": failed,
    }


@mcp.tool()
async def update_fact(
        gcn_id: str = Field(...,
                            description="gcn_id факта, который нужно изменить (из результатов recall / semantic_search)"),
        new_text: str = Field(..., description="Новый текст факта"),
        reason: str = Field("", description="Причина изменения (опционально, сохраняется в провенанс)"),
        user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
        ctx: Context = None
) -> Dict[str, Any]:
    """Атомарно заменяет текст существующего факта по gcn_id (RETRACT + CREATE)."""
    uid, err = _safe_resolve_user(user_id, ctx)
    if err:
        return {"status": "error", "error": "forbidden", "message": err}

    service = await get_memory_service(uid)

    # 1. Получаем информацию о старом факте
    old_info = await service.explain_fact(gcn_id)
    if "error" in old_info:
        return {
            "status": "error",
            "message": f"Факт не найден: {old_info['error']}",
        }

    old_text = old_info.get("subject") or old_info.get("text", "")
    old_scope = old_info.get("scope", "private")

    # 2. RETRACT: удаляем старый факт
    await service.forget(gcn_id, scope=old_scope, dry_run=False)

    # 3. CREATE: создаём новый факт с force_new=True (обходим дедупликацию)
    save_result = await service.remember(
        new_text.strip(),
        scope=old_scope,
        force_new=True,
    )

    # 4. ПЕРЕСТРОЙКА FAISS-ИНДЕКСА (исправление бага: recall не находил новый факт сразу)
    memory_layer = {
        "private": service.private_memory,
        "shared": service.shared_memory,
        "global": service.global_memory,
    }.get(old_scope, service.private_memory)

    # Помечаем как "грязный" (для консистентности с остальным кодом)
    memory_layer.store._faiss_dirty = True

    # Если индекс уже построен — перестраиваем немедленно
    # Это быстро (не строим с нуля) и гарантирует, что следующий recall найдёт факт
    if memory_layer.store.faiss_index is not None:
        memory_layer.store.build_faiss_index(force=True)

    comment = f" Причина: {reason}" if reason else ""
    return {
        "status": "ok",
        "gcn_id": save_result.get("id"),  # ← новый gcn_id для клиента
        "old_text": old_text,
        "new_text": new_text,
        "scope": old_scope,
        "comment": f"Факт обновлён.{comment}",
        "save_result": save_result,
    }


# ============================================================
# ИНСТРУМЕНТЫ РАБОТЫ С КОДОМ (CodeAnalyzer)
# ============================================================

@mcp.tool()
async def read_code_file(
        file_path: str = Field(..., description="Относительный путь от корня проекта, например 'GCN/config_ai.py'"),
        max_lines: int = Field(500, description="Максимум строк для возврата", ge=10, le=2000),
) -> Dict[str, Any]:
    """Читает файл исходного кода проекта с проверкой безопасности."""
    analyzer = get_analyzer()
    content = await analyzer.read_file(file_path, max_lines=max_lines)
    ok = not content.startswith("Ошибка")
    return {
        "status": "ok" if ok else "error",
        "file_path": file_path,
        "content": content,
        "truncated": "обрезан" in content,
    }


@mcp.tool()
async def search_in_code(
        pattern: str = Field(..., description="Строка или регулярное выражение для поиска в коде проекта"),
        max_results: int = Field(20, description="Максимум результатов", ge=1, le=50),
) -> Dict[str, Any]:
    """Ищет паттерн (строку или regex) по всем файлам кода проекта."""
    analyzer = get_analyzer()
    result = await analyzer.search_in_code(pattern, max_results=max_results)
    found = not result.startswith("Ничего не найдено") and not result.startswith("Ошибка")
    return {
        "status": "ok" if found else "not_found",
        "pattern": pattern,
        "result": result,
    }


@mcp.tool()
async def get_project_structure(
        max_depth: int = Field(3, description="Максимальная глубина обхода директорий", ge=1, le=5),
) -> Dict[str, Any]:
    """Возвращает дерево файлов проекта — только разрешённые расширения кода."""
    analyzer = get_analyzer()
    tree = await analyzer.get_project_structure(max_depth=max_depth)
    return {
        "status": "ok",
        "max_depth": max_depth,
        "tree": tree,
    }


@mcp.tool()
async def analyze_error(
        error_message: str = Field(..., description="Текст ошибки (например, 'KeyError: config not found')"),
        traceback_str: str = Field("",
                                   description="Полная трассировка стека из Python (необязательно, но улучшает анализ)"),
) -> Dict[str, Any]:
    """Анализирует Python-ошибку: извлекает файлы и строки из traceback."""
    analyzer = get_analyzer()
    analysis = await analyzer.analyze_error_location(error_message, traceback_str)
    return {
        "status": "ok",
        "error_message": error_message,
        "analysis": analysis,
    }


# ============================================================
# РЕСУРСЫ
# ============================================================
@mcp.resource("memory://{user_id}/facts")
async def list_facts(user_id: str) -> Dict[str, Any]:
    canon = _canon_user_id(user_id) or user_id
    service = await get_memory_service(canon)
    memory = service.private_memory
    memory.reload_if_stale()
    facts = memory.semantic_facts[:20]
    return {
        "total": len(memory.semantic_facts),
        "facts": [{"id": f.id, "text": f.text[:200], "confidence": f.confidence} for f in facts]
    }


@mcp.resource("memory://{user_id}/fact/{fact_id}")
async def get_fact(user_id: str, fact_id: str) -> Dict[str, Any]:
    canon = _canon_user_id(user_id) or user_id
    service = await get_memory_service(canon)
    memory = service.private_memory
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
    logger.info("🚀 Запуск рефакторированного MCP сервера BlockcoinWitres (с MemoryService)...")
    mcp.run()