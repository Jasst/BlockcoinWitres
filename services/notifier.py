"""
services/notifier.py — Асинхронный менеджер уведомлений (WebSocket + офлайн-буфер)
"""
import asyncio
import logging
import json
import time
from collections import defaultdict
from typing import Dict, List

logger = logging.getLogger(__name__)


class AsyncMessageNotifier:
    def __init__(self, max_buffer_size: int = 100):
        self.max_buffer_size = max_buffer_size
        # _buffers больше не нужен, всё храним в БД

    async def add_message(self, user_address: str, message: dict):
        """Сохраняет сообщение в БД и отправляет push."""
        from routes.ws import manager
        sent = await manager.send_personal_message(user_address, message)

        if not sent:
            from database import get_db_cursor
            async with get_db_cursor() as conn:
                await conn.execute(
                    "INSERT INTO offline_messages (user_address, payload, created_at) "
                    "VALUES ($1, $2, $3)",
                    user_address, json.dumps(message), time.time()
                )
            logger.debug(f"User {user_address[:16]} offline, saved to DB")

        # 🔥 ИСПРАВЛЕНИЕ: Отправляем push ВЛЮБОМ СЛУЧАЕ
        # Если приложение в фоне на мобильном, WS еще жив, но JS заморожен.
        # Push придет в Service Worker и разбудит телефон.
        await self._send_push_for_message(user_address, message)

    async def _send_push_for_message(self, user_address: str, message: dict):
        """Отправляет push-уведомление для оффлайн-пользователя или если приложение в фоне."""
        try:
            from services.push import send_push

            msg_type = message.get('type', 'message')

            if msg_type == 'incoming_call':
                return

            if msg_type == 'message':
                sender = message.get('sender', '')
                sender_name = message.get('from_name') or (sender[:10] if sender else 'Unknown')

                content = message.get('content', '')
                preview = '💬 New message'
                if content:
                    try:
                        parsed = json.loads(content)
                        preview = '💬 Encrypted message'
                    except (json.JSONDecodeError, TypeError):
                        preview = content[:50] if len(content) > 50 else content

                is_group = bool(message.get('group_id') or message.get('recipient', '').startswith('group:'))

                # ✅ НОВОЕ: Определяем chatId для правильной навигации и дедупликации
                if is_group:
                    group_id = message.get('recipient', '').replace('group:', '')
                    chat_url = f"/chat?group={group_id}"
                    chat_id_for_push = group_id  # нормализованный ID группы
                else:
                    chat_url = f"/chat?address={sender}"
                    chat_id_for_push = sender  # адрес собеседника

                await send_push(
                    user_address=user_address,
                    title=f"💬 {sender_name}",
                    body=preview,
                    url=chat_url,
                    push_type="message",
                    from_address=sender,
                    from_name=sender_name,
                    # ✅ НОВОЕ: передаём chatId чтобы SW мог проверить активный чат
                    chat_id=chat_id_for_push
                )

        except Exception as e:
            logger.error(f"Push for offline message failed: {e}")

    async def add_group_messages(self, group_id: str, members: List[str],
                                 message: dict, exclude_sender: str = None):
        for member in members:
            if member != exclude_sender:
                await self.add_message(member, message)

    async def get_offline_messages(self, user_address: str) -> List[dict]:
        """Возвращает все накопленные сообщения из БД и удаляет их."""
        from database import get_db_cursor
        async with get_db_cursor() as conn:
            rows = await conn.fetch(
                "SELECT payload FROM offline_messages WHERE user_address = $1 ORDER BY created_at ASC",
                user_address
            )
            await conn.execute("DELETE FROM offline_messages WHERE user_address = $1", user_address)
        return [json.loads(row['payload']) for row in rows]

    async def clear_offline_messages(self, user_address: str):
        """Удаляет все накопленные сообщения для пользователя после успешной отправки."""
        from database import get_db_cursor
        async with get_db_cursor() as conn:
            await conn.execute("DELETE FROM offline_messages WHERE user_address = $1", user_address)

    async def get_stats(self) -> dict:
        from database import get_db_cursor
        async with get_db_cursor() as conn:
            count = await conn.fetchval("SELECT COUNT(*) FROM offline_messages")
            return {
                'offline_buffers': count,  # приблизительно
                'total_buffered': count,
                'max_buffer_size': self.max_buffer_size,
            }


message_notifier = AsyncMessageNotifier(max_buffer_size=100)