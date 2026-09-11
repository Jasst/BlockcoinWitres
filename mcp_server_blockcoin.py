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
from typing import Optional, Dict, Any, List
import os
import base64
import re
import secrets
from datetime import datetime

from mcp.server.fastmcp import FastMCP, Context
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field

sys.path.insert(0, str(Path(__file__).resolve().parent))

from GCN.config_ai import MEMORY_BASE_DIR, GENERATED_IMAGES_DIR, EASYDIFFUSION_ENABLED
from GCN.memory_service import get_memory_service, MemoryService
from GCN.web_search import deep_search
from GCN.image_utils import enhance_prompt, generate_image as gen_image
from routes.ai_assistant import get_assistant
from GCN.llm_client import call_llm

# --- Вход по криптоподписи кошелька (EIP-191) ---
# pip install eth-account
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

# --- Идентификация пользователя на уровне MCP (см. _resolve_user) ---
# Для локальных stdio-клиентов (Claude Desktop и т.п.) идентификатор
# пользователя можно задать через env BLOCKCOIN_USER_ID в конфиге клиента.
# Для хостед HTTP-клиентов — заголовком X-User-Id (см. ниже).
_ENV_DEFAULT_USER = os.getenv("BLOCKCOIN_USER_ID", "").strip()


# Состояние входа процесса. Для stdio-транспорта процесс обслуживает одного
# пользователя — одной привязки достаточно. (Для streamable HTTP с сессиями
# состояние нужно вести per-session через ctx.session_id — здесь не реализовано,
# т.к. хостед-вариант решается заголовком X-User-Id на шлюзе.)
_VERIFIED_USER: "Optional[str]" = None   # адрес, подтверждённый подписью
_VERIFIED_SIGNER: "Optional[str]" = None  # ключ, которым подписан вход (для 64-hex user_id)
_LOGIN_NONCE: "Optional[str]" = None     # одноразовый вызов подписи


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
#  - user_id приложения: 64 hex без префикса (такой id используется как имя
#    папки памяти ai_memory_v3/<user_id>). К нему владение подтверждается
#    привязкой к подписавшему ключу при первом входе (см. verify_login).
_ADDR_ETH_RE = re.compile(r"^0x[0-9a-fA-F]{40}$")
_USERID_RE = re.compile(r"^(0x)?[0-9a-fA-F]{64}$")


def _canon_user_id(raw: "Optional[str]") -> "Optional[str]":
    """Каноническая форма идентификатора: lower-case; у 64-hex user_id
    префикс 0x отбрасывается — именно в таком виде id идёт в имена папок
    памяти. None, если формат не распознан."""
    s = (raw or "").strip().lower()
    if _ADDR_ETH_RE.fullmatch(s):
        return s
    if _USERID_RE.fullmatch(s):
        return s[2:] if s.startswith("0x") else s
    return None


def _is_eth_address(uid: str) -> bool:
    return uid.startswith("0x") and len(uid) == 42


def _resolve_user(user_id: "Optional[str]", ctx: "Optional[Context]" = None) -> str:
    """Порядок приоритета:
    1) идентификатор, верифицированный через verify_login (высший приоритет
       для этого процесса; явный user_id, ему не соответствующий, отклоняется);
    2) user_id, явно переданный вызывающей моделью (пропускается через
       канонизацию: 0xABC...64hex -> abc...64hex, как в папках памяти);
    3) заголовок X-User-Id из HTTP-запроса (доверенный шлюз платформы);
    4) env BLOCKCOIN_USER_ID (локальный stdio-клиент);
    5) DEFAULT_USER — последний резорт.
    """
    if _VERIFIED_USER:
        if user_id:
            declared = _canon_user_id(user_id) or user_id.strip().lower()
            if declared != _VERIFIED_USER:
                raise PermissionError(
                    f"Этот MCP-сервер привязан к идентификатору {_VERIFIED_USER} (вход "
                    f"подтверждён подписью). Вызов от имени '{user_id}' запрещён."
                )
        return _VERIFIED_USER
    if user_id:
        return _canon_user_id(user_id) or user_id.strip()
    uid = _user_from_ctx(ctx)
    if uid:
        return uid
    if _ENV_DEFAULT_USER:
        return _ENV_DEFAULT_USER
    return DEFAULT_USER


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

# --- Вспомогательные функции ---
async def _with_timeout(coro, tool_name: str, timeout: Optional[float] = None) -> Any:
    """
    ИСПРАВЛЕНИЕ: раньше ловился только asyncio.TimeoutError. Эта обёртка
    используется 4 разными инструментами (execute_command, web_search,
    generate_image, research_topic) как единая точка "безопасного" вызова —
    но любое другое исключение внутри (сетевая ошибка, KeyError, что угодно
    в глубине process_input/deep_search/gen_image) вылетало из тула
    необработанным, вместо структурированного {"status": "error", ...},
    которым отвечают все остальные инструменты в этом файле. Для клиента
    MCP разница ощутимая: сырой traceback/протокольная ошибка вместо
    предсказуемого JSON, который можно показать пользователю или обработать
    программно.
    """
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

# Добавить глобальный словарь для отслеживания последних вызовов
_last_commands: Dict[str, float] = {}
# ИСПРАВЛЕНИЕ: ключ — f"{user_id}:{command}", запись никогда не удалялась —
# при разнообразных командах от разных пользователей словарь рос без
# ограничений в течение всего времени жизни процесса. Записи старше окна
# дедупликации бесполезны, поэтому чистим их по ходу дела (без отдельного
# фонового таска — просто при каждом новом вызове, амортизированно дёшево).
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
    address: str = Field(..., description="Твой user_id приложения (64 hex, как в ai_memory_v3) или адрес кошелька (0x...)"),
) -> Dict[str, Any]:
    """Начало входа: возвращает текст, который нужно подписать кошельком.
    Подпиши его (MetaMask -> Sign message / personal_sign, EIP-191) и передай
    подпись в verify_login. После этого все операции пойдут под твоим id."""
    global _LOGIN_NONCE
    canon = _canon_user_id(address)
    if canon is None:
        return {
            "status": "error",
            "message": (
                f"Некорректный идентификатор: {address!r}. Ожидается адрес кошелька "
                f"(0x + 40 hex) или user_id приложения (64 hex, как в папках ai_memory_v3)."
            ),
        }
    _LOGIN_NONCE = secrets.token_hex(16)
    message = f"BlockcoinWitres MCP login\nUser: {canon}\nNonce: {_LOGIN_NONCE}"
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
) -> Dict[str, Any]:
    """Проверяет подпись и привязывает MCP-процесс к идентификатору.
    Дальше recall/remember/цели/уведомления идут под ним автоматически,
    а попытка передать чужой user_id будет отклонена.
    Для адресов 0x... подпись должна восстанавливаться строго в этот адрес.
    Для 64-hex user_id сервер привязывает id к подписавшему ключу: при повторном
    входе с этим id подписавший должен совпадать (защита от перехвата id)."""
    global _VERIFIED_USER, _VERIFIED_SIGNER, _LOGIN_NONCE
    canon = _canon_user_id(address)
    if not ETH_ACCOUNT_AVAILABLE:
        return {"status": "error", "message": "Сервер без eth-account: pip install eth-account"}
    if canon is None:
        return {"status": "error", "message": f"Некорректный идентификатор: {address!r}"}
    if not _LOGIN_NONCE:
        return {"status": "error", "message": "Сначала вызови request_login(address)."}
    message = f"BlockcoinWitres MCP login\nUser: {canon}\nNonce: {_LOGIN_NONCE}"
    try:
        recovered = Account.recover_message(
            encode_defunct(text=message), signature=signature
        ).lower()
    except Exception as e:
        return {"status": "error", "message": f"Не удалось проверить подпись: {e}"}
    _LOGIN_NONCE = None  # nonce одноразовый

    if _is_eth_address(canon):
        # Строгий режим: подпись обязана восстанавливаться именно в этот адрес.
        if recovered != canon:
            return {
                "status": "error",
                "message": f"Подпись не соответствует адресу (подписавший: {recovered}).",
            }
    else:
        # user_id приложения (64 hex): криптографически не выводится из адреса,
        # поэтому владение подтверждается привязкой id -> подписавший ключ.
        if _VERIFIED_USER == canon and _VERIFIED_SIGNER and recovered != _VERIFIED_SIGNER:
            return {
                "status": "error",
                "message": "Этот user_id уже привязан к другому ключу.",
            }

    _VERIFIED_USER = canon
    _VERIFIED_SIGNER = recovered
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
async def execute_command(
    command: str = Field(..., description="Любая команда на естественном языке"),
    allow_web_search: bool = Field(True, description="Разрешить веб-поиск, если нужен"),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
    ctx: Context = None
) -> Dict[str, Any]:
    """Выполняет любую команду через тот же пайплайн, что и обычный чат."""
    # === ИСПРАВЛЕНИЕ: дедупликация быстрых повторных вызовов ===
    key = f"{_resolve_user(user_id, ctx)}:{command}"
    now = time.time()
    _prune_last_commands(now)
    if key in _last_commands and now - _last_commands[key] < _DEDUP_WINDOW_SECONDS:
        return {
            "status": "error",
            "error": "duplicate",
            "message": "Предыдущий вызов этой команды ещё обрабатывается (или выполнен менее 10 секунд назад)."
        }
    _last_commands[key] = now

    assistant = await get_assistant(_resolve_user(user_id, ctx))
    async def _run():
        return await assistant.process_input(command, web_search=allow_web_search)
    result = await _with_timeout(_run(), "execute_command")
    # ИСПРАВЛЕНИЕ: раньше проверялось только result.get("error") == "timeout" —
    # теперь _with_timeout может вернуть структурированную ошибку и для любого
    # другого исключения (см. _with_timeout), поэтому проверяем общий признак.
    if isinstance(result, dict) and result.get("status") == "error":
        return result
    response, meta = result

    # ИСПРАВЛЕНИЕ: если LLM вернула пустой ответ (сбой call_llm, см.
    # llm_client.call_llm — при ошибке возвращает ""), пустая строка не
    # содержит ни "ошибка", ни "не удалось", ни "404" — раньше это тихо
    # проваливалось в ветку "status": "ok" с пустым result, то есть реальный
    # сбой LLM выглядел для вызывающего MCP-клиента как успешное выполнение.
    if not response or not response.strip():
        return {
            "status": "error",
            "error": "empty_response",
            "message": "Модель не вернула ответ (возможно, сбой локальной LLM).",
            "meta": meta,
            "user_id": _resolve_user(user_id, ctx),
            "timestamp": time.time()
        }

    # === ИСПРАВЛЕНИЕ: если ответ содержит ошибку, возвращаем чёткий статус ===
    if "ошибка" in response.lower() or "не удалось" in response.lower() or "404" in response:
        return {
            "status": "error",
            "message": response,
            "meta": meta,
            "user_id": _resolve_user(user_id, ctx),
            "timestamp": time.time()
        }

    return {
        "status": "ok",
        "result": response,
        "meta": meta,
        "user_id": _resolve_user(user_id, ctx),
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
    service = await get_memory_service(_resolve_user(user_id, ctx))
    results = await service.recall(query, top_k, scope)
    return {
        "results": results,
        "count": len(results)
    }

@mcp.tool()
async def remember(
    fact: str = Field(..., description="Факт для запоминания"),
    scope: Optional[str] = Field(None, description="Скоуп: 'private', 'shared', 'global'. Если не указан – автоопределение."),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
    ctx: Context = None
) -> Dict[str, Any]:
    """Сохраняет факт в указанный скоуп (автоопределение, если не задан)."""
    service = await get_memory_service(_resolve_user(user_id, ctx))
    result = await service.remember(fact, scope)
    return {"status": "ok", **result}

@mcp.tool()
async def forget(
    query: str = Field(..., description="Ключевые слова для удаления фактов"),
    scope: str = Field("private", description="Из какого слоя удалять: 'private', 'shared' или 'global'"),
    dry_run: bool = Field(True, description="Если True – только показывает кандидаты"),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
    ctx: Context = None
) -> Dict[str, Any]:
    """Удаляет факты, содержащие заданные ключевые слова, из указанного слоя памяти."""
    service = await get_memory_service(_resolve_user(user_id, ctx))
    return await service.forget(query, scope, dry_run)

@mcp.tool()
async def web_search(
    query: Optional[str] = Field(None, description="Один поисковый запрос (или URL для прямого чтения страницы)"),
    queries: Optional[List[str]] = Field(
        None, description="Несколько поисковых запросов для составного вопроса — выполняются параллельно"
    ),
    max_results: int = Field(5, description="Максимальное число страниц для анализа на каждый запрос", ge=1, le=10)
) -> Dict[str, Any]:
    """Выполняет поиск в DuckDuckGo (один или несколько запросов параллельно) и
    возвращает извлечённый контекст и источники.

    ИСПРАВЛЕНИЕ (унификация с браузерным чатом): раньше этот инструмент
    поддерживал только один query, тогда как внутренний internal__web_search
    в ai_assistant.py/tool_router.py умел лишь один запрос за раунд ReAct-
    цикла — оба пути были одинаково ограничены. Теперь оба принимают либо
    query, либо queries (список), и ведут себя идентично: одна и та же
    логика deep_search, одна и та же семантика результата, чтобы MCP-клиент
    (например, Claude Desktop) и браузерный чат давали согласованные
    результаты на один и тот же составной вопрос.
    """
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
            # Ошибка (таймаут или исключение) одного из подзапросов не должна
            # обрушивать остальные — возвращаем то, что успело собраться по
            # другим запросам.
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

    assistant = await get_assistant(_resolve_user(user_id, ctx))

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

    # Сохранение на диск
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
    """
    Загружает содержимое файла из публичного репозитория GitHub через raw-ссылку.
    Использует aiohttp, не требует curl.
    """
    url = f"https://raw.githubusercontent.com/{repo}/{branch}/{path.lstrip('/')}"
    try:
        import aiohttp
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=15)) as resp:
                if resp.status == 200:
                    content = await resp.text()
                    # Обрезаем до max_lines, если нужно
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
    """Глубокое исследование темы с генерацией гипотез и сбором доказательств."""
    assistant = await get_assistant(_resolve_user(user_id, ctx))
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
    service = await get_memory_service(_resolve_user(user_id, ctx))
    episodes = await service.get_episodes(limit)
    return {"episodes": episodes, "count": len(episodes)}

@mcp.tool()
async def get_contradictions(
    limit: int = Field(5, description="Максимальное число пар противоречий", ge=1, le=10),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
    ctx: Context = None
) -> Dict[str, Any]:
    """Возвращает неразрешённые противоречия из личной памяти."""
    service = await get_memory_service(_resolve_user(user_id, ctx))
    pairs = await service.get_contradictions(limit)
    return {
        "contradictions": [{"a": a, "b": b} for a, b in pairs],
        "count": len(pairs)
    }

@mcp.tool()
async def resolve_contradiction(
    fact_id_a: str = Field(..., description="ID первого факта (числовой или GCN-идентификатор)"),
    fact_id_b: str = Field(..., description="ID второго факта"),
    verdict: str = Field(..., description="Вердикт: 'a' (оставить A), 'b' (оставить B), 'both' (сохранить оба), 'neither' (удалить оба)"),
    reason: str = Field("", description="Причина разрешения (опционально)"),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
    ctx: Context = None
) -> Dict[str, Any]:
    """Ручное разрешение противоречия."""
    service = await get_memory_service(_resolve_user(user_id, ctx))
    return await service.resolve_contradiction(fact_id_a, fact_id_b, verdict, reason)

@mcp.tool()
async def get_goals(
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
    ctx: Context = None
) -> Dict[str, Any]:
    """Возвращает активные цели пользователя."""
    service = await get_memory_service(_resolve_user(user_id, ctx))
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
    service = await get_memory_service(_resolve_user(user_id, ctx))
    return await service.add_goal(description, priority)

@mcp.tool()
async def get_notifications(
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
    mark_delivered: bool = Field(
        True,
        description="Пометить возвращённые уведомления доставленными (они не придут повторно)."
    ),
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Проактивные находки, которые фоновые циклы ассистента (авто-исследование
    по целям, доисследование тем из рефлексии — как в основном чате, так и
    из фонового цикла этого же MCP-процесса, если он запущен) решили сами
    донести до пользователя, не дожидаясь вопроса.

    MCP-сервер сам ничего не пушит в чат — это pull-инструмент: хост
    (например, BiChat) вызывает его сам, периодически или в начале сессии,
    и показывает пользователю то, что вернулось. Источник данных общий с
    основным чатом (GCN-стор на диске), поэтому находка, сгенерированная
    в процессе браузерного чата, будет видна и здесь, и наоборот — не
    важно, через какой процесс её сгенерировал ассистент.
    """
    service = await get_memory_service(_resolve_user(user_id, ctx))
    items = await service.get_pending_notifications(mark_delivered=mark_delivered)
    return {"notifications": items, "count": len(items)}

@mcp.tool()
async def semantic_search(
    query: str = Field(..., description="Поисковый запрос"),
    top_k: int = Field(5, description="Число результатов", ge=1, le=20),
    scope: Optional[str] = Field(
        None,
        description="Фильтр по слою памяти: 'private', 'shared' или 'global'. "
                    "Если не указан — поиск по всем трём слоям (личная, общая и глобальная память)."
    ),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
    ctx: Context = None
) -> Dict[str, Any]:
    """Векторный поиск по смыслу по личной, общей и глобальной памяти (эмбеддинги)."""
    service = await get_memory_service(_resolve_user(user_id, ctx))
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
    service = await get_memory_service(_resolve_user(user_id, ctx))
    return await service.graph_explore(seed_text, depth)

@mcp.tool()
async def explain_fact(
    gcn_id: str = Field(..., description="Идентификатор объекта памяти (gcn_id)"),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
    ctx: Context = None
) -> Dict[str, Any]:
    """Объясняет происхождение и статус утверждения памяти."""
    service = await get_memory_service(_resolve_user(user_id, ctx))
    return await service.explain_fact(gcn_id)

@mcp.tool()
async def get_memory_stats(
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
    ctx: Context = None
) -> Dict[str, Any]:
    """Возвращает статистику по личной памяти."""
    try:
        service = await get_memory_service(_resolve_user(user_id, ctx))
        return await service.get_memory_stats()
    except Exception as e:
        logger.warning(f"get_memory_stats failed for user {_resolve_user(user_id, ctx)}: {e}")
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
    """Быстрая ориентация в начале сессии: статистика памяти, последние
    эпизоды, активные цели и непрочитанные проактивные уведомления —
    всё одним вызовом вместо четырёх последовательных.

    Рекомендуется вызывать в начале каждого разговора, чтобы сразу иметь
    контекст: что помним, над чем работаем, что нашли фоновые циклы.
    """
    uid = _resolve_user(user_id, ctx)
    service = await get_memory_service(uid)

    stats, episodes, goals, notifs = await asyncio.gather(
        service.get_memory_stats(),
        service.get_episodes(3),
        service.get_goals(),
        service.get_pending_notifications(mark_delivered=True),
        return_exceptions=True,
    )

    return {
        "user_id": uid,
        "stats": stats if not isinstance(stats, Exception) else {},
        "recent_episodes": episodes if not isinstance(episodes, Exception) else [],
        "goals": goals if not isinstance(goals, Exception) else [],
        "notifications": notifs if not isinstance(notifs, Exception) else [],
    }


@mcp.tool()
async def remember_batch(
    facts: List[str] = Field(..., description="Список фактов для запоминания (до 10 штук)"),
    scope: Optional[str] = Field(None, description="Скоуп для всех фактов: 'private', 'shared', 'global'. Автоопределение, если не задан."),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
    ctx: Context = None
) -> Dict[str, Any]:
    """Сохраняет несколько фактов за один вызов вместо последовательных
    remember(). Факты сохраняются параллельно; ошибка одного не блокирует
    остальные.
    """
    if not facts:
        return {"status": "error", "message": "Список фактов пуст."}

    service = await get_memory_service(_resolve_user(user_id, ctx))
    batch = facts[:10]  # hard cap

    results = await asyncio.gather(
        *[service.remember(f.strip(), scope) for f in batch if f and f.strip()],
        return_exceptions=True,
    )

    saved = [batch[i] for i, r in enumerate(results) if not isinstance(r, Exception)]
    failed = [
        {"fact": batch[i], "error": str(r)}
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
    gcn_id: str = Field(..., description="gcn_id факта, который нужно изменить (из результатов recall / semantic_search)"),
    new_text: str = Field(..., description="Новый текст факта"),
    reason: str = Field("", description="Причина изменения (опционально, сохраняется в провенанс)"),
    user_id: Optional[str] = Field(default=None, description=_USER_ID_DESC),
    ctx: Context = None
) -> Dict[str, Any]:
    """Атомарно заменяет текст существующего факта: сначала объясняет
    происхождение старого (explain_fact), затем удаляет его и сохраняет
    новый с тем же скоупом.

    Удобнее пары forget(dry_run=False) + remember() — один вызов, скоуп
    подхватывается автоматически.
    """
    service = await get_memory_service(_resolve_user(user_id, ctx))

    # Получаем старый факт, чтобы знать его скоуп и текст
    old_info = await service.explain_fact(gcn_id)
    if old_info.get("status") == "error" or "not found" in str(old_info.get("error", "")).lower():
        return {
            "status": "error",
            "message": f"Факт с gcn_id='{gcn_id}' не найден: {old_info}",
        }

    old_text = old_info.get("subject") or old_info.get("text", "")
    old_scope = old_info.get("scope", "private")

    # Удаляем старый факт по его тексту (forget ищет по ключевым словам)
    if old_text:
        # Берём первые 6 слов как ключ удаления — достаточно уникально
        keyword = " ".join(old_text.split()[:6])
        await service.forget(keyword, scope=old_scope, dry_run=False)

    # Сохраняем новый факт в том же скоупе
    save_result = await service.remember(new_text.strip(), scope=old_scope)

    comment = f" Причина: {reason}" if reason else ""
    return {
        "status": "ok",
        "old_text": old_text,
        "new_text": new_text,
        "scope": old_scope,
        "comment": f"Факт обновлён.{comment}",
        "save_result": save_result,
    }

# ============================================================
# РЕСУРСЫ
# ============================================================
@mcp.resource("memory://{user_id}/facts")
async def list_facts(user_id: str) -> Dict[str, Any]:
    service = await get_memory_service(_resolve_user(user_id, ctx))
    stats = await service.get_memory_stats()
    # Упрощённо: возвращаем список фактов из service
    # В сервисе нет метода для получения всех фактов, поэтому используем прямой доступ к памяти
    # (можно добавить метод в сервис, но для простоты оставим так)
    memory = service.private_memory
    memory.reload_if_stale()
    facts = memory.semantic_facts[:20]
    return {
        "total": len(memory.semantic_facts),
        "facts": [{"id": f.id, "text": f.text[:200], "confidence": f.confidence} for f in facts]
    }

@mcp.resource("memory://{user_id}/fact/{fact_id}")
async def get_fact(user_id: str, fact_id: str) -> Dict[str, Any]:
    service = await get_memory_service(_resolve_user(user_id, ctx))
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