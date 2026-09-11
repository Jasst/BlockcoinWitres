"""
routes/status.py — Статусы пользователей (онлайн/оффлайн) (асинхронная версия)
"""
import logging
import time

from fastapi import APIRouter, Depends, HTTPException, Request

from database import get_db_cursor
from dependencies import require_auth
from models import HeartbeatRequest, ManyStatusesRequest
from config import ONLINE_TIMEOUT_SECONDS

logger = logging.getLogger(__name__)
router = APIRouter(tags=['status'])

ONLINE_TIMEOUT = ONLINE_TIMEOUT_SECONDS


@router.post('/heartbeat')
async def heartbeat(body: HeartbeatRequest, request: Request, address: str = Depends(require_auth)):
    try:
        async with get_db_cursor() as conn:
            await conn.execute('''
                INSERT INTO user_status (address, last_seen, status, current_chat)
                VALUES ($1, $2, 'online', $3)
                ON CONFLICT(address) DO UPDATE SET
                    last_seen = excluded.last_seen,
                    status = 'online',
                    current_chat = excluded.current_chat
            ''', address, time.time(), body.current_chat)
        # НОВОЕ: оповещаем всех через WebSocket, что этот адрес стал онлайн
        from routes.ws import manager
        await manager.broadcast_status_update(address, 'online')
        return {'status': 'ok'}
    except Exception as e:
        logger.error(f"Heartbeat error: {e}")
        raise HTTPException(500, 'Internal error')

@router.get('/get_status/{address_param}')
async def get_status(address_param: str):
    try:
        async with get_db_cursor() as conn:
            row = await conn.fetchrow(
                'SELECT last_seen, status, current_chat FROM user_status WHERE address = $1',
                address_param
            )
        if not row:
            return {'address': address_param, 'status': 'offline', 'last_seen': None}
        last_seen = row[0]
        is_online = (time.time() - last_seen) < ONLINE_TIMEOUT
        return {
            'address': address_param,
            'status': 'online' if is_online else 'offline',
            'last_seen': last_seen,
            'current_chat': row[2] if is_online else None,
        }
    except Exception as e:
        logger.error(f"Get status error: {e}")
        raise HTTPException(500, 'Internal error')


@router.post('/get_many_statuses')
async def get_many_statuses(body: ManyStatusesRequest, request: Request, address: str = Depends(require_auth)):
    if not body.addresses:
        return {'statuses': {}}
    try:
        placeholders = ','.join(f'${i+1}' for i in range(len(body.addresses)))
        async with get_db_cursor() as conn:
            rows = await conn.fetch(f'''
                SELECT address, last_seen, status, current_chat
                FROM user_status
                WHERE address IN ({placeholders})
            ''', *body.addresses)
        now = time.time()
        result = {}
        for row in rows:
            is_online = (now - row[1]) < ONLINE_TIMEOUT
            result[row[0]] = {
                'status': 'online' if is_online else 'offline',
                'last_seen': row[1],
                'current_chat': row[2] if is_online else None,
            }
        for addr in body.addresses:
            if addr not in result:
                result[addr] = {'status': 'offline', 'last_seen': None, 'current_chat': None}
        return {'statuses': result}
    except Exception as e:
        logger.error(f"Get many statuses error: {e}")
        raise HTTPException(500, 'Internal error')