"""
MCP Client Manager — управление подключениями к MCP-серверам.
Загружает инструменты из конфига и предоставляет единый интерфейс для вызова.

Изменения по сравнению с исходной версией:
- config_path больше не резолвится от текущей рабочей директории процесса
  (ломалось при запуске из другого cwd, например под systemd) — теперь по
  умолчанию ищется рядом с этим модулем / GCN.config_ai.MEMORY_BASE_DIR,
  с возможностью переопределить через переменную окружения MCP_SERVERS_CONFIG.
- Поддержаны не только локальные stdio-серверы, но и удалённые MCP по SSE
  (cfg с полем "url" и опциональным "headers") — раньше был захардкожен
  только StdioServerParameters/stdio_client.
- Каждый вызов инструмента ограничен по времени (TOOL_CALL_TIMEOUT_SECONDS) —
  раньше зависший внешний сервер вешал весь ответ чата без возможности выйти.
- Серверы, которые не удалось поднять при старте, не остаются "мёртвыми"
  до перезапуска процесса: initialize() помнит их конфиг и позволяет
  периодически повторять попытку через ensure_connected().
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


def _resolve_tool_timeout(tool_name: str) -> float:
    """
    Подбирает тайм-аут клиента под конкретный инструмент (см.
    MCP_TOOL_TIMEOUT_OVERRIDES в config_ai.py) вместо единого
    TOOL_CALL_TIMEOUT_SECONDS для всех вызовов подряд — иначе клиент
    обрывал бы вызов тяжёлых инструментов (генерация изображения,
    глубокое исследование) раньше, чем сервер сам успевает их выполнить.
    """
    name_lower = (tool_name or "").lower()
    for marker, timeout in MCP_TOOL_TIMEOUT_OVERRIDES.items():
        if marker in name_lower:
            return timeout
    return TOOL_CALL_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)


def _default_config_path() -> Path:
    env_path = os.environ.get("MCP_SERVERS_CONFIG")
    if env_path:
        return Path(env_path)
    # Раньше: Path("mcp_servers.json") — зависело от cwd процесса.
    candidate = MEMORY_BASE_DIR.parent / "mcp_servers.json"
    if candidate.exists():
        return candidate
    # запасной вариант — рядом с текущим файлом
    return Path(__file__).resolve().parent / "mcp_servers.json"


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
        logger.info(f"Загрузка MCP конфига из: {self.config_path}")  # <-- ДОБАВЛЕНО
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
        logger.info(f"Connecting to MCP server '{name}' with config: {cfg}")
        stack = AsyncExitStack()
        try:
            if cfg.get("url"):
                if not SSE_AVAILABLE:
                    raise RuntimeError(
                        "cfg содержит 'url', но пакет mcp не предоставляет mcp.client.sse "
                        "в этой версии — обновите пакет 'mcp' для поддержки удалённых серверов."
                    )
                url = cfg["url"]
                headers = cfg.get("headers") or {}
                headers = {**headers, "Accept": "text/event-stream"}
                logger.info(f"Connecting via SSE to {url}")
                read, write = await stack.enter_async_context(sse_client(url, headers=headers))
            else:
                command = cfg.get("command")
                args = cfg.get("args", [])
                env = cfg.get("env", {})
                logger.info(f"Connecting via stdio: command={command}, args={args}")
                server_params = StdioServerParameters(command=command, args=args, env=env)
                read, write = await stack.enter_async_context(stdio_client(server_params))

            logger.info(f"Creating session for '{name}'")
            session = await stack.enter_async_context(ClientSession(read, write))
            logger.info(f"Initializing session for '{name}'")
            await session.initialize()
            logger.info(f"Listing tools for '{name}'")
            response = await session.list_tools()
            tools = [tool.model_dump() for tool in response.tools]
            logger.info(f"Received {len(tools)} tools for '{name}'")

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
        """
        Повторяет попытку подключения к серверам, которые не удалось поднять
        при старте (или отвалились позже) — не чаще MCP_RECONNECT_INTERVAL на
        сервер. Раньше сервер, недоступный в момент старта процесса, оставался
        недоступен до перезапуска, даже если поднимался через минуту после старта.
        Безопасно вызывать часто — сама решает, нужна ли попытка.
        """
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