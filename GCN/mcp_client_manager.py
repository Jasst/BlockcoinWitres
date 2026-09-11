"""
MCP Client Manager — управление подключениями к MCP-серверам.
Загружает инструменты из конфига и предоставляет единый интерфейс для вызова.

- Конфиг ищется рядом с этим модулем / GCN.config_ai.MEMORY_BASE_DIR (не от
  cwd процесса — важно при запуске под systemd), с переопределением через
  переменную окружения MCP_SERVERS_CONFIG.
- Поддержаны локальные stdio-серверы и удалённые MCP по SSE (cfg с полем
  "url" и опциональным "headers").
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
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

try:
    from mcp.client.sse import sse_client
    SSE_AVAILABLE = True
except ImportError:
    SSE_AVAILABLE = False

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


class MCPToolManager:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or _default_config_path()
        self.sessions: Dict[str, ClientSession] = {}
        self.tools: Dict[str, List[Dict]] = {}  # server_name -> list of tools
        self._server_configs: Dict[str, Dict] = {}
        self._server_stacks: Dict[str, AsyncExitStack] = {}
        self._failed_servers: Dict[str, float] = {}  # name -> timestamp последней неудачи
        self._initialized = False

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
        try:
            if cfg.get("url"):
                if not SSE_AVAILABLE:
                    raise RuntimeError(
                        "cfg содержит 'url', но пакет mcp не предоставляет mcp.client.sse "
                        "в этой версии — обновите пакет 'mcp' для поддержки удалённых серверов."
                    )
                url = cfg["url"]
                headers = {**(cfg.get("headers") or {}), "Accept": "text/event-stream"}
                logger.debug(f"'{name}': SSE connect to {url}")
                read, write = await stack.enter_async_context(sse_client(url, headers=headers))
            else:
                command = cfg.get("command")
                args = cfg.get("args", [])
                env = cfg.get("env", {})
                logger.debug(f"'{name}': stdio connect, command={command}, args={args}")
                server_params = StdioServerParameters(command=command, args=args, env=env)
                read, write = await stack.enter_async_context(stdio_client(server_params))

            session = await stack.enter_async_context(ClientSession(read, write))
            await session.initialize()
            response = await session.list_tools()
            tools = [tool.model_dump() for tool in response.tools]

            self.sessions[name] = session
            self.tools[name] = tools
            self._server_stacks[name] = stack
            self._failed_servers.pop(name, None)
            logger.info(f"MCP сервер '{name}' загружен, инструментов: {len(tools)}")
            return True
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
        при старте (или отвалились позже) — не чаще MCP_RECONNECT_INTERVAL на
        сервер. Безопасно вызывать часто — сама решает, нужна ли попытка."""
        if not self._failed_servers:
            return
        now = time.time()
        for name, failed_at in list(self._failed_servers.items()):
            if now - failed_at < MCP_RECONNECT_INTERVAL:
                continue
            cfg = self._server_configs.get(name)
            if not cfg:
                continue
            logger.info(f"Повторная попытка подключения к MCP серверу '{name}'...")
            await self._connect_one(name, cfg)

    def get_all_tools(self) -> List[Dict]:
        """Возвращает все инструменты со всех серверов с меткой сервера."""
        all_tools = []
        for server_name, tools in self.tools.items():
            for tool in tools:
                tool_with_server = tool.copy()
                tool_with_server["server"] = server_name
                all_tools.append(tool_with_server)
        return all_tools

    async def call_tool(self, server_name: str, tool_name: str, arguments: Dict) -> str:
        """Вызвать инструмент на указанном сервере — с ограничением по времени,
        чтобы зависший внешний сервер не вешал весь ответ чата навсегда."""
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
        if result.content:
            return result.content[0].text
        return str(result)

    async def close(self):
        """Закрыть все соединения."""
        for stack in self._server_stacks.values():
            try:
                await stack.aclose()
            except Exception as e:
                logger.debug(f"Ошибка при закрытии MCP-соединения: {e}")
        self._server_stacks.clear()
        self.sessions.clear()