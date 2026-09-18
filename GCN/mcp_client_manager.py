"""
MCP Client Manager — управление подключениями к MCP-серверам.
Загружает инструменты из конфига и предоставляет единый интерфейс для вызова.

- Конфиг ищется рядом с этим модулем / GCN.config_ai.MEMORY_BASE_DIR (не от
  cwd процесса — важно при запуске под systemd), с переопределением через
  переменную окружения MCP_SERVERS_CONFIG.
- Поддержаны локальные stdio-серверы, удалённые MCP по SSE (cfg с полем
  "url" и опциональным "headers") и удалённые MCP по Streamable HTTP
  (транспорт определяется полем "transport" либо автоматически по URL:
  "…/sse" → SSE, иначе → Streamable HTTP).
- Каждый вызов инструмента ограничен по времени (TOOL_CALL_TIMEOUT_SECONDS,
  с точечными переопределениями per-tool), чтобы зависший внешний сервер не
  вешал весь ответ чата.
- Серверы, которые не удалось поднять при старте, не остаются "мёртвыми" до
  перезапуска процесса: initialize() помнит их конфиг, ensure_connected()
  периодически повторяет попытку.
"""

import json
import logging
import os
import time
import asyncio
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

# ---------------------------------------------------------------------------
# Транспорты. Импортируем отдельно, потому что mcp-версии различаются:
# в одних есть только SSE, в других — только Streamable HTTP, в третьих оба.
# ---------------------------------------------------------------------------
try:
    from mcp.client.sse import sse_client
    SSE_AVAILABLE = True
except ImportError:
    SSE_AVAILABLE = False

try:
    from mcp.client.streamable_http import streamablehttp_client
    STREAMABLE_HTTP_AVAILABLE = True
except ImportError:
    # В mcp < 1.8 модуль назывался по-другому; на всякий случай пробуем
    # известный старый путь — если и его нет, считаем, что транспорта нет.
    try:
        from mcp.client.streamable_http import streamablehttp_client  # noqa: F811
        STREAMABLE_HTTP_AVAILABLE = True
    except ImportError:
        streamablehttp_client = None  # type: ignore
        STREAMABLE_HTTP_AVAILABLE = False

try:
    from GCN.config_ai import (
        TOOL_CALL_TIMEOUT_SECONDS,
        MCP_RECONNECT_INTERVAL,
        MEMORY_BASE_DIR,
        MCP_TOOL_TIMEOUT_OVERRIDES,
    )
except ImportError:
    TOOL_CALL_TIMEOUT_SECONDS = 45
    MCP_RECONNECT_INTERVAL = 120
    MEMORY_BASE_DIR = Path(__file__).resolve().parent
    MCP_TOOL_TIMEOUT_OVERRIDES = {}

logger = logging.getLogger(__name__)


def _resolve_tool_timeout(tool_name: str) -> float:
    """Подбирает тайм-аут под конкретный инструмент (MCP_TOOL_TIMEOUT_OVERRIDES)
    вместо единого TOOL_CALL_TIMEOUT_SECONDS для всех вызовов подряд — иначе
    тяжёлые инструменты (генерация изображения, глубокое исследование)
    обрывались бы клиентом раньше, чем сервер успевает их выполнить."""
    name_lower = (tool_name or "").lower()
    for marker, timeout in MCP_TOOL_TIMEOUT_OVERRIDES.items():
        if marker in name_lower:
            return timeout
    return TOOL_CALL_TIMEOUT_SECONDS


def _default_config_path() -> Path:
    env_path = os.environ.get("MCP_SERVERS_CONFIG")
    if env_path:
        return Path(env_path)
    candidate = MEMORY_BASE_DIR.parent / "mcp_servers.json"
    if candidate.exists():
        return candidate
    return Path(__file__).resolve().parent / "mcp_servers.json"


def _redact_cfg(cfg: Dict) -> Dict:
    """Конфиг сервера для лога — без значений env (могут содержать секреты)."""
    safe = {k: v for k, v in cfg.items() if k != "env"}
    if "env" in cfg:
        safe["env"] = list(cfg["env"].keys())
    return safe


def _detect_transport(url: str, cfg_transport: Optional[str] = None) -> str:
    """
    Возвращает "sse" или "streamable-http".

    Приоритеты:
      1) явное поле "transport" в конфиге сервера (может быть "sse",
         "http", "streamable-http", "streamable_http");
      2) эвристика по URL: заканчивается на "/sse" → SSE, иначе → Streamable HTTP.
    """
    if cfg_transport:
        t = cfg_transport.strip().lower().replace("_", "-")
        if t in ("sse",):
            return "sse"
        if t in ("http", "streamable-http", "streamable", "streamablehttp"):
            return "streamable-http"
    if url.rstrip("/").endswith("/sse"):
        return "sse"
    return "streamable-http"


class MCPToolManager:
    def __init__(self, config_path: Optional[Path] = None, user_id: Optional[str] = None):
        self.config_path = config_path or _default_config_path()
        self.sessions: Dict[str, ClientSession] = {}
        self.tools: Dict[str, List[Dict]] = {}  # server_name -> list of tools
        self._server_configs: Dict[str, Dict] = {}
        self._server_stacks: Dict[str, AsyncExitStack] = {}
        self._failed_servers: Dict[str, float] = {}  # name -> timestamp последней неудачи
        self._fail_counts: Dict[str, int] = {}  # счётчик попыток для экспоненциального backoff
        self._initialized = False
        # ИЗМЕНЕНИЕ: сохраняем user_id для передачи в MCP сервер через заголовок X-User-Id
        self.user_id = user_id.strip().lower() if user_id else None
        # Кэш результатов вызовов инструментов: (server, tool, args_hash) -> (result, timestamp)
        self._tool_cache: Dict[tuple, tuple] = {}
        self._cache_ttl = int(os.getenv("MCP_TOOL_CACHE_TTL", "300"))  # 5 минут по умолчанию
        # Rate limiting: (server, tool) -> [timestamps]
        self._rate_limits: Dict[tuple, List[float]] = {}
        self._rate_limit_calls = int(os.getenv("MCP_RATE_LIMIT_CALLS", "10"))  # вызовов
        self._rate_limit_window = int(os.getenv("MCP_RATE_LIMIT_WINDOW", "60"))  # секунд
        # write-инструменты не кешируются, их вызов инвалидирует кеш
        self._write_tools = {
            "remember", "forget", "add_goal", "resolve_contradiction",
            "update_fact", "record_action",
        }

    async def initialize(self):
        """Подключиться ко всем серверам из конфига."""
        if self._initialized:
            return
        logger.info(f"Загрузка MCP конфига из: {self.config_path}")
        if not self.config_path.exists():
            logger.warning(f"Файл конфигурации MCP не найден: {self.config_path}")
            self._initialized = True
            return

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        servers = config.get("mcpServers", {})
        self._server_configs = servers
        for name, cfg in servers.items():
            await self._connect_one(name, cfg)

        self._initialized = True

    async def _connect_one(self, name: str, cfg: Dict) -> bool:
        logger.info(f"Connecting to MCP server '{name}': {_redact_cfg(cfg)}")
        stack = AsyncExitStack()

        # Таймаут на каждую сетевую операцию подключения. Без него
        # streamablehttp_client/sse_client/stdio_client могут висеть
        # бесконечно, если удалённый сервер недоступен (нет DNS, TCP
        # не отвечает, TLS-хендшейк завис) — и блокировать startup
        # приложения (см. init_global_mcp в routes/ai_assistant.py,
        # который делает await _global_mcp_manager.initialize()).
        # Переопределяется через MCP_CONNECT_TIMEOUT.
        CONNECT_TIMEOUT = float(os.getenv("MCP_CONNECT_TIMEOUT", "10"))

        try:
            if cfg.get("url"):
                url = cfg["url"]

                # Заголовки: сначала собственные заголовки сервера из конфига,
                # затем принудительно — X-User-Id (доверенный идентификатор
                # текущей сессии) и корректный Accept.
                headers: Dict[str, str] = dict(cfg.get("headers") or {})
                uid = cfg.get("user_id") or self.user_id
                if uid:
                    headers["X-User-Id"] = uid.strip().lower()
                    logger.debug(f"'{name}': using user_id={uid[:16]}... for X-User-Id header")

                transport = _detect_transport(url, cfg.get("transport"))

                if transport == "sse":
                    if not SSE_AVAILABLE:
                        raise RuntimeError(
                            "Конфиг сервера '{n}' задаёт SSE ({u}), но установленный "
                            "пакет 'mcp' не предоставляет mcp.client.sse — обновите "
                            "'mcp' или укажите transport=\"streamable-http\"."
                            .format(n=name, u=url)
                        )
                    headers.setdefault("Accept", "text/event-stream")
                    logger.debug(f"'{name}': SSE connect to {url}")
                    read, write = await asyncio.wait_for(
                        stack.enter_async_context(sse_client(url, headers=headers)),
                        timeout=CONNECT_TIMEOUT,
                    )
                else:  # streamable-http
                    if not STREAMABLE_HTTP_AVAILABLE:
                        raise RuntimeError(
                            "Конфиг сервера '{n}' задаёт Streamable HTTP ({u}), но "
                            "установленный пакет 'mcp' не предоставляет "
                            "mcp.client.streamable_http — обновите 'mcp' "
                            "(`pip install -U 'mcp>=1.8'`).".format(n=name, u=url)
                        )
                    # Streamable HTTP принимает оба типа контента; по умолчанию
                    # SDK сам ставит корректный Accept, но подстрахуемся.
                    headers.setdefault("Accept", "application/json, text/event-stream")
                    logger.debug(f"'{name}': Streamable HTTP connect to {url}")
                    # Важно: streamablehttp_client отдаёт ТРИ значения —
                    # read, write и функцию получения mcp-session-id.
                    read, write, _get_session_id = await asyncio.wait_for(
                        stack.enter_async_context(
                            streamablehttp_client(url, headers=headers)
                        ),
                        timeout=CONNECT_TIMEOUT,
                    )
            else:
                command = cfg.get("command")
                args = cfg.get("args", [])
                env = cfg.get("env", {})
                logger.debug(f"'{name}': stdio connect, command={command}, args={args}")
                server_params = StdioServerParameters(command=command, args=args, env=env)
                read, write = await asyncio.wait_for(
                    stack.enter_async_context(stdio_client(server_params)),
                    timeout=CONNECT_TIMEOUT,
                )

            session = await asyncio.wait_for(
                stack.enter_async_context(ClientSession(read, write)),
                timeout=CONNECT_TIMEOUT,
            )
            await asyncio.wait_for(session.initialize(), timeout=CONNECT_TIMEOUT)
            response = await asyncio.wait_for(
                session.list_tools(), timeout=CONNECT_TIMEOUT
            )
            tools = [tool.model_dump() for tool in response.tools]

            self.sessions[name] = session
            self.tools[name] = tools
            self._server_stacks[name] = stack
            self._failed_servers.pop(name, None)
            self._fail_counts.pop(name, None)
            logger.info(f"MCP сервер '{name}' загружен, инструментов: {len(tools)}")
            return True

        except asyncio.TimeoutError:
            logger.error(
                f"Таймаут подключения к MCP серверу '{name}' "
                f"({CONNECT_TIMEOUT}с) — сервер помечен как недоступный, "
                f"повторная попытка через MCP_RECONNECT_INTERVAL."
            )
            self._failed_servers[name] = time.time()
            # Пытаемся закрыть всё, что успело зарегистрироваться в стеке,
            # чтобы не течь сокетами/тасками до повторного подключения.
            try:
                await stack.aclose()
            except Exception as close_err:
                logger.debug(f"Ошибка закрытия stack после таймаута '{name}': {close_err}")
            return False

        except Exception as e:
            logger.exception(f"Ошибка подключения к MCP серверу '{name}': {e}")
            self._failed_servers[name] = time.time()
            try:
                await stack.aclose()
            except Exception:
                pass
            return False

    async def ensure_connected(self):
        """Повторяет попытку подключения к серверам, которые не удалось поднять
        при старте (или отвалились позже) — с экспоненциальным backoff.
        Безопасно вызывать часто — сама решает, нужна ли попытка."""
        if not self._failed_servers:
            return
        now = time.time()
        for name, failed_at in list(self._failed_servers.items()):
            fail_count = self._fail_counts.get(name, 0)
            wait = min(MCP_RECONNECT_INTERVAL * (2 ** fail_count), 3600)
            if now - failed_at < wait:
                continue
            cfg = self._server_configs.get(name)
            if not cfg:
                continue
            logger.info(f"Reconnect attempt #{fail_count + 1} for '{name}' (wait was {wait}s)")
            success = await self._connect_one(name, cfg)
            if success:
                self._fail_counts.pop(name, None)
            else:
                self._fail_counts[name] = fail_count + 1
                self._failed_servers[name] = now  # сбрасываем таймер

    def get_all_tools(self) -> List[Dict]:
        """Возвращает все инструменты со всех серверов с меткой сервера."""
        all_tools = []
        for server_name, tools in self.tools.items():
            for tool in tools:
                tool_with_server = tool.copy()
                tool_with_server["server"] = server_name
                all_tools.append(tool_with_server)
        return all_tools

    def _check_rate_limit(self, server_name: str, tool_name: str) -> bool:
        """Проверяет rate limit для инструмента. Возвращает True, если вызов разрешён."""
        key = (server_name, tool_name)
        now = time.time()

        if key in self._rate_limits:
            self._rate_limits[key] = [
                ts for ts in self._rate_limits[key]
                if now - ts < self._rate_limit_window
            ]
        else:
            self._rate_limits[key] = []

        if len(self._rate_limits[key]) >= self._rate_limit_calls:
            return False

        self._rate_limits[key].append(now)
        return True

    def _get_cache_key(self, server_name: str, tool_name: str, arguments: Dict) -> tuple:
        """Создаёт хэш-ключ для кэширования результата вызова."""
        args_str = json.dumps(arguments, sort_keys=True)
        args_hash = hashlib.sha256(args_str.encode()).hexdigest()[:16]
        return (server_name, tool_name, args_hash)

    def _get_cached_result(self, cache_key: tuple) -> Optional[str]:
        """Возвращает закэшированный результат, если он ещё валиден."""
        if cache_key not in self._tool_cache:
            return None
        result, timestamp = self._tool_cache[cache_key]
        if time.time() - timestamp > self._cache_ttl:
            del self._tool_cache[cache_key]
            return None
        return result

    def _cache_result(self, cache_key: tuple, result: str):
        """Кэширует результат вызова инструмента."""
        self._tool_cache[cache_key] = (result, time.time())
        if len(self._tool_cache) > 1000:
            oldest_keys = sorted(self._tool_cache.keys(),
                                 key=lambda k: self._tool_cache[k][1])[:100]
            for k in oldest_keys:
                del self._tool_cache[k]

    def _is_write_tool(self, tool_name: str) -> bool:
        """Проверяет, является ли инструмент write-инструментом (изменяет состояние)."""
        return any(w in tool_name.lower() for w in self._write_tools)

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict) -> str:
        """Вызвать инструмент на указанном сервере — с ограничением по времени,
        кэшированием результатов и rate limiting, чтобы зависший внешний сервер
        не вешал весь ответ чата навсегда, а частые одинаковые вызовы не
        перегружали сервер.
        """
        session = self.sessions.get(server_name)
        if not session:
            raise ValueError(f"Сервер '{server_name}' не найден")

        if not self._check_rate_limit(server_name, tool_name):
            logger.warning(
                f"Rate limit превышен для {server_name}.{tool_name} "
                f"({self._rate_limit_calls} вызовов за {self._rate_limit_window}с)"
            )
            raise RuntimeError(
                f"Rate limit для инструмента '{server_name}.{tool_name}': "
                f"не более {self._rate_limit_calls} вызовов за {self._rate_limit_window}с"
            )

        if self._is_write_tool(tool_name):
            stale = [k for k in self._tool_cache if k[0] == server_name]
            for k in stale:
                del self._tool_cache[k]
            logger.debug(f"Write-инструмент {tool_name}: инвалидация кеша для {server_name}")
            return await self._execute_tool(server_name, tool_name, arguments)

        cache_key = self._get_cache_key(server_name, tool_name, arguments)
        cached = self._get_cached_result(cache_key)
        if cached is not None:
            logger.debug(f"Кэш-хит для {server_name}.{tool_name}")
            return cached

        result = await self._execute_tool(server_name, tool_name, arguments)

        response_lower = result.lower()
        if "error" not in response_lower and "exception" not in response_lower:
            self._cache_result(cache_key, result)

        return result

    async def _execute_tool(self, server_name: str, tool_name: str, arguments: Dict) -> str:
        """Прямой вызов session.call_tool с per-tool таймаутом."""
        session = self.sessions.get(server_name)
        if not session:
            raise ValueError(f"Сервер '{server_name}' не найден")
        timeout = _resolve_tool_timeout(tool_name)
        try:
            result = await asyncio.wait_for(
                session.call_tool(tool_name, arguments=arguments),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"MCP-инструмент '{server_name}.{tool_name}' не ответил за {timeout}с"
            )
        return result.content[0].text if result.content else str(result)

    async def close(self):
        """Закрыть все соединения."""
        for stack in self._server_stacks.values():
            try:
                await stack.aclose()
            except Exception as e:
                logger.debug(f"Ошибка при закрытии MCP-соединения: {e}")
        self._server_stacks.clear()
        self.sessions.clear()