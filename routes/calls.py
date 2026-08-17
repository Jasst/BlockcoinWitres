"""
routes/calls.py — WebRTC TURN credentials, история звонков
"""
import base64
import hashlib
import hmac
import time
import logging
from fastapi import APIRouter, Depends, HTTPException
from dependencies import require_auth
from config import (
    TURN_ENABLED, TURN_SERVER, STUN_SERVER,
    TURN_STATIC_AUTH_SECRET, TURN_USERNAME, TURN_PASSWORD,
    TURN_REALM, TURN_CREDENTIAL_TTL
)
from models import CallLogEntry, DeleteCallRequest
from database import get_db_cursor

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/calls', tags=['calls'])


def generate_turn_credentials(username: str):
    """Генерирует временные TURN credentials по алгоритму static-auth-secret."""
    if not TURN_STATIC_AUTH_SECRET:
        return {
            "username": TURN_USERNAME,
            "credential": TURN_PASSWORD,
            "ttl": 0
        }
    timestamp = int(time.time()) + TURN_CREDENTIAL_TTL
    turn_username = f"{timestamp}:{username}"
    secret_bytes = TURN_STATIC_AUTH_SECRET.encode('utf-8')
    hm = hmac.new(secret_bytes, turn_username.encode('utf-8'), hashlib.sha1)
    credential = base64.b64encode(hm.digest()).decode()
    return {
        "username": turn_username,
        "credential": credential,
        "ttl": TURN_CREDENTIAL_TTL
    }


@router.get('/turn-credentials')
async def get_turn_credentials(address: str = Depends(require_auth)):
    if not TURN_ENABLED:
        raise HTTPException(503, "WebRTC calls are disabled")

    ice_servers = []
    if STUN_SERVER:
        ice_servers.append({"urls": STUN_SERVER})

    turn_urls = [f"{TURN_SERVER}?transport=udp", f"{TURN_SERVER}?transport=tcp"]
    creds = generate_turn_credentials(address)

    logger.warning(f"TURN credentials for {address[:8]}...: user={creds['username']}, cred={creds['credential']}")

    ice_servers.append({
        "urls": turn_urls,
        "username": creds["username"],
        "credential": creds["credential"],
        "realm": TURN_REALM,
        "ttl": creds["ttl"]
    })

    return {"iceServers": ice_servers}


# ============================================================================
# ЭНДПОИНТЫ ДЛЯ ИСТОРИИ ЗВОНКОВ
# ============================================================================

@router.post('/log')
async def log_call(entry: CallLogEntry, address: str = Depends(require_auth)):
    """Сохраняет запись о завершённом звонке."""
    async with get_db_cursor() as conn:
        await conn.execute("""
            INSERT INTO call_logs (user_address, contact_address, contact_name, direction, status, duration, timestamp)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
        """, address, entry.address, entry.name, entry.direction, entry.status, entry.duration, entry.timestamp)
    return {'ok': True}


@router.get('/history')
async def get_call_history(
    address: str = Depends(require_auth),
    limit: int = 100
):
    """Возвращает историю звонков текущего пользователя."""
    async with get_db_cursor() as conn:
        rows = await conn.fetch("""
            SELECT id, contact_address, contact_name, direction, status, duration, timestamp
            FROM call_logs
            WHERE user_address = $1
            ORDER BY timestamp DESC
            LIMIT $2
        """, address, limit)
    calls = [
        {
            'id': r[0],
            'address': r[1],
            'contact_name': r[2],
            'direction': r[3],
            'status': r[4],
            'duration': r[5],
            'timestamp': r[6]
        }
        for r in rows
    ]
    return {'calls': calls}


@router.post('/delete')
async def delete_call(entry: DeleteCallRequest, address: str = Depends(require_auth)):
    """Удаляет запись о звонке по ID."""
    async with get_db_cursor() as conn:
        # Проверяем, что запись принадлежит пользователю
        row = await conn.fetchrow(
            "SELECT id FROM call_logs WHERE id = $1 AND user_address = $2",
            entry.call_id, address
        )
        if not row:
            raise HTTPException(404, "Call log not found or not yours")
        await conn.execute(
            "DELETE FROM call_logs WHERE id = $1",
            entry.call_id
        )
    return {'ok': True}


@router.post('/clear')
async def clear_call_history(address: str = Depends(require_auth)):
    """Полностью очищает историю звонков пользователя."""
    async with get_db_cursor() as conn:
        await conn.execute(
            "DELETE FROM call_logs WHERE user_address = $1",
            address
        )
    return {'ok': True}