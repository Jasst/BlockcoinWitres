"""
routes/archive_pins.py — серверное закрепление сообщений.

Архив бесед (hide/unhide) находится в routes/messages.py.

Важно: на сервер передаётся только id закреплённого сообщения.
Текст не хранится: сообщения зашифрованы, сервер не должен видеть
расшифрованное содержимое.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException

from cache import get_user_groups_cached, get_groups_cache_version
from database import get_db_cursor
from dependencies import require_auth
from models import PinMessageRequest

logger = logging.getLogger(__name__)
router = APIRouter(tags=['pins'])


def pin_key(user: str, chat_with: str) -> str:
    """Один закреп на пару участников (общий для обоих) или на группу."""
    if chat_with.startswith('group:'):
        return chat_with
    return '|'.join(sorted([user.lower(), chat_with.lower()]))


async def _message_in_chat(conn, message_id: int, user: str, chat_with: str) -> bool:
    if chat_with.startswith('group:'):
        row = await conn.fetchrow(
            'SELECT id FROM transactions WHERE id = $1 AND recipient = $2',
            message_id, chat_with)
    else:
        row = await conn.fetchrow(
            'SELECT id FROM transactions WHERE id = $1 AND '
            '((sender = $2 AND recipient = $3) OR (sender = $3 AND recipient = $2))',
            message_id, user, chat_with)
    return row is not None


@router.post('/pin_message')
async def pin_message(req: PinMessageRequest, user: str = Depends(require_auth)):
    key = pin_key(user, req.chat_with)
    async with get_db_cursor() as conn:
        if req.message_id is None:
            await conn.execute('DELETE FROM chat_pins WHERE chat_key = $1', key)
            return {'ok': True}

        if req.chat_with.startswith('group:'):
            gid = req.chat_with.split(':', 1)[1]
            groups = await get_user_groups_cached(user, cache_version=await get_groups_cache_version())
            if not any(g['id'] == gid for g in groups):
                raise HTTPException(403, 'No access')

        if not await _message_in_chat(conn, req.message_id, user, req.chat_with):
            raise HTTPException(404, 'Message not found in this chat')

        await conn.execute(
            """
            INSERT INTO chat_pins (chat_key, message_id, content_preview, pinned_by, pinned_at)
            VALUES ($1, $2, NULL, $3, extract(epoch from now()))
            ON CONFLICT (chat_key) DO UPDATE SET
                message_id      = EXCLUDED.message_id,
                content_preview = NULL,
                pinned_by       = EXCLUDED.pinned_by,
                pinned_at       = EXCLUDED.pinned_at
            """,
            key, req.message_id, user,
        )
    return {'ok': True}


@router.get('/get_pin')
async def get_pin(chat_with: str, user: str = Depends(require_auth)):
    key = pin_key(user, chat_with)
    async with get_db_cursor() as conn:
        row = await conn.fetchrow(
            'SELECT message_id FROM chat_pins WHERE chat_key = $1', key)
    if not row:
        return {'message_id': None}
    return {'message_id': row['message_id']}