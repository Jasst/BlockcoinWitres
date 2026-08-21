"""
MCP Client Manager — управление подключениями к MCP-серверам.
Загружает инструменты из конфига и предоставляет единый интерфейс для вызова.
"""

import json
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any
from contextlib import AsyncExitStack

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

logger = logging.getLogger(__name__)

class MCPToolManager:
    def __init__(self, config_path: Optional[Path] = None):
        self.config_path = config_path or Path("mcp_servers.json")
        self.sessions: Dict[str, ClientSession] = {}
        self.tools: Dict[str, List[Dict]] = {}  # server_name -> list of tools
        self.exit_stack = AsyncExitStack()
        self._initialized = False

    async def initialize(self):
        """Подключиться ко всем серверам из конфига."""
        if self._initialized:
            return
        if not self.config_path.exists():
            logger.warning(f"Файл конфигурации MCP не найден: {self.config_path}")
            self._initialized = True
            return

        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)

        servers = config.get("servers", {})
        for name, cfg in servers.items():
            try:
                command = cfg.get("command")
                args = cfg.get("args", [])
                env = cfg.get("env", {})
                server_params = StdioServerParameters(
                    command=command,
                    args=args,
                    env=env
                )
                # Подключаемся
                read, write = await self.exit_stack.enter_async_context(
                    stdio_client(server_params)
                )
                session = await self.exit_stack.enter_async_context(
                    ClientSession(read, write)
                )
                await session.initialize()
                self.sessions[name] = session
                # Получаем инструменты
                response = await session.list_tools()
                tools = [tool.model_dump() for tool in response.tools]
                self.tools[name] = tools
                logger.info(f"MCP сервер '{name}' загружен, инструментов: {len(tools)}")
            except Exception as e:
                logger.error(f"Ошибка подключения к MCP серверу '{name}': {e}")

        self._initialized = True

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
        """Вызвать инструмент на указанном сервере."""
        session = self.sessions.get(server_name)
        if not session:
            raise ValueError(f"Сервер '{server_name}' не найден")
        result = await session.call_tool(tool_name, arguments=arguments)
        if result.content:
            return result.content[0].text
        return str(result)

    async def close(self):
        """Закрыть все соединения."""
        await self.exit_stack.aclose()