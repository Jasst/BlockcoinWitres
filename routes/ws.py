"""
routes/ws.py — WebSocket менеджер для реального времени
(включая сигнализацию для WebRTC голосовых/видео звонков)
"""
import asyncio
import logging
import time
import traceback
from typing import Dict, Optional
from fastapi import WebSocket, WebSocketDisconnect, APIRouter, Query
from starlette.websockets import WebSocketState

from database import get_db_cursor
from services.notifier import message_notifier
from setup import load_public_key_from_b64
from cache import get_cached_public_key, get_pubkey_cache_version  # <--- добавлен импорт
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature


from services.push import send_push

logger = logging.getLogger(__name__)
router = APIRouter(tags=['websocket'])


class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self._conn_lock = asyncio.Lock()
        self._calls_lock = asyncio.Lock()
        self.calls: Dict[str, Dict] = {}
        self._cleanup_task = None
        asyncio.create_task(self.start_cleanup())

    async def start_cleanup(self):
        if self._cleanup_task is None:
            self._cleanup_task = asyncio.create_task(self._cleanup_old_calls())

    async def connect(self, user_id: str, websocket: WebSocket):
        await websocket.accept()
        async with self._conn_lock:
            self.active_connections[user_id] = websocket
        logger.info(f"User {user_id[:16]} connected via WebSocket")

    async def disconnect(self, user_id: str):
        async with self._conn_lock:
            self.active_connections.pop(user_id, None)
        logger.info(f"User {user_id[:16]} disconnected")

    async def send_personal_message(self, user_id: str, message: dict) -> bool:
        async with self._conn_lock:
            ws = self.active_connections.get(user_id)
        if ws:
            if ws.client_state == WebSocketState.DISCONNECTED:
                await self.disconnect(user_id)
                return False
            try:
                await ws.send_json(message)
                return True
            except Exception as e:
                logger.error(f"Failed to send to {user_id}: {e}")
                await self.disconnect(user_id)
        return False

    async def broadcast(self, message: dict, exclude: str = None):
        async with self._conn_lock:
            connections = list(self.active_connections.items())

        for user_id, ws in connections:
            if user_id == exclude:
                continue
            if ws.client_state == WebSocketState.DISCONNECTED:
                await self.disconnect(user_id)
                continue
            try:
                await ws.send_json(message)
            except Exception as e:
                logger.error(f"Broadcast to {user_id} failed: {e}")
                await self.disconnect(user_id)

    async def broadcast_status_update(self, address: str, status: str):
        await self.broadcast({
            'type': 'status_update',
            'address': address,
            'status': status
        })

    async def get_stats(self):
        async with self._conn_lock:
            return {
                'active_connections': len(self.active_connections),
                'users': list(self.active_connections.keys())[:10]
            }

    async def _cleanup_old_calls(self):
        while True:
            await asyncio.sleep(30)
            now = time.time()
            expired = []
            async with self._calls_lock:
                for call_id, info in self.calls.items():
                    if now - info.get('created_at', 0) > 60:
                        expired.append(call_id)
                for call_id in expired:
                    del self.calls[call_id]
                    logger.info(f"Removed expired call {call_id}")


manager = ConnectionManager()

async def authenticate_websocket(websocket: WebSocket, address: str, signature: str, nonce: str) -> Optional[str]:
    logger.info(f"🔐 WS auth: address={address[:16]}..., nonce={nonce[:16]}..., sig_len={len(signature)}")

    if not address or not signature or not nonce:
        logger.warning("❌ Missing address, signature or nonce")
        return None

    try:
        # Получаем актуальную версию кеша
        cache_version = await get_pubkey_cache_version()
        pubkey, verified = await get_cached_public_key(address, cache_version=cache_version)

        if not pubkey:
            logger.warning(f"❌ Public key not found for {address[:16]}")
            return None
        logger.info(f"✅ Public key found, verified={verified}")

        raw_sig = bytes.fromhex(signature)
        if len(raw_sig) != 64:
            logger.warning(f"❌ Invalid signature length: {len(raw_sig)} (expected 64)")
            return None
        r = int.from_bytes(raw_sig[:32], 'big')
        s = int.from_bytes(raw_sig[32:], 'big')
        der_sig = encode_dss_signature(r, s)

        pubkey_obj = load_public_key_from_b64(pubkey)
        pubkey_obj.verify(der_sig, nonce.encode(), ec.ECDSA(hashes.SHA256()))
        logger.info(f"✅ WebSocket auth SUCCESS for {address[:16]}")
        return address

    except ValueError as e:
        logger.error(f"❌ WebSocket auth ValueError: {e}\n{traceback.format_exc()}")
    except Exception as e:
        logger.error(f"❌ WebSocket auth FAILED: {e}\n{traceback.format_exc()}")
    return None


@router.websocket("/ws")
async def websocket_endpoint(
    websocket: WebSocket,
    address: str = Query(...),
    signature: str = Query(...),
    nonce: str = Query(...)
):
    user_id = await authenticate_websocket(websocket, address, signature, nonce)
    if not user_id:
        await websocket.close(code=1008, reason="Unauthorized")
        return

    await manager.connect(user_id, websocket)

    try:
        # ========== ОТПРАВКА ПРОПУЩЕННЫХ СООБЩЕНИЙ ==========
        try:
            missed = await message_notifier.get_offline_messages(user_id)
            for msg in missed:
                try:
                    await websocket.send_json(msg)
                except Exception as e:
                    logger.error(f"Failed to send missed message: {e}")
            # ИСПРАВЛЕНИЕ: удаляем отправленные сообщения из очереди
            await message_notifier.clear_offline_messages(user_id)
        except Exception as e:
            logger.error(f"Failed to fetch missed messages: {e}")

        # ========== ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ СООБЩЕНИЙ ==========
        while True:
            try:
                data = await websocket.receive_json()
            except WebSocketDisconnect:
                logger.info(f"WebSocketDisconnect for {user_id[:16]}")
                break
            except RuntimeError as e:
                logger.info(f"RuntimeError while receiving: {e}")
                break
            except Exception as e:
                logger.error(f"Unexpected error while receiving: {e}\n{traceback.format_exc()}")
                break

            msg_type = data.get('type')

            if msg_type == 'ping':
                try:
                    await websocket.send_json({'type': 'pong'})
                except Exception as e:
                    logger.error(f"Failed to send pong: {e}")

            elif msg_type == 'mark_read':
                message_id = data.get('message_id')
                if message_id:
                    try:
                        async with get_db_cursor() as conn:
                            await conn.execute("""
                                UPDATE transactions SET status = 'read', read_at = $1
                                WHERE id = $2 AND recipient = $3
                            """, time.time(), message_id, user_id)
                    except Exception as e:
                        logger.error(f"Failed to mark read: {e}")

            # ---------- Обработка звонков (WebRTC) ----------
            elif msg_type == 'call_offer':
                target = data.get('target')
                call_id = data.get('call_id')
                sdp = data.get('sdp')

                if not target or not call_id or not sdp:
                    logger.warning(f"Missing fields in call_offer: target={target}, call_id={call_id}, sdp={sdp is not None}")
                    continue

                from_name = data.get('from_name')
                if not from_name:
                    try:
                        async with get_db_cursor() as conn:
                            row = await conn.fetchrow(
                                "SELECT contact_name FROM contacts WHERE user_address = $1 AND contact_address = $2",
                                target, user_id
                            )
                            if row and row['contact_name']:
                                from_name = row['contact_name']
                    except Exception:
                        pass
                if not from_name:
                    from_name = user_id[:10]

                async with manager._calls_lock:
                    manager.calls[call_id] = {
                        'from': user_id,
                        'to': target,
                        'state': 'offer_sent',
                        'created_at': time.time(),
                        'offer_sdp': sdp,
                        'from_name': from_name,
                        'ice_candidates': []
                    }
                await manager.send_personal_message(target, {
                    'type': 'incoming_call',
                    'call_id': call_id,
                    'from': user_id,
                    'sdp': sdp,
                    'from_name': from_name
                })
                try:
                    await send_push(
                        user_address=target,
                        title="Входящий звонок",
                        body=f"{from_name} звонит вам",
                        push_type="incoming_call",
                        call_id=call_id,
                        from_name=from_name,
                        from_address=user_id
                    )
                    logger.info(f"Call offer {call_id}: push sent to {target[:16]}")
                except Exception as e:
                    logger.error(f"Push failed: {e}")

            elif msg_type == 'get_call':
                call_id = data.get('call_id')
                if not call_id:
                    continue
                async with manager._calls_lock:
                    call_info = manager.calls.get(call_id)
                if call_info and call_info.get('offer_sdp'):
                    try:
                        await websocket.send_json({
                            'type': 'incoming_call',
                            'call_id': call_id,
                            'from': call_info['from'],
                            'sdp': call_info['offer_sdp'],
                            'from_name': call_info.get('from_name', call_info['from'][:10]),
                            'candidates': call_info.get('ice_candidates', [])
                        })
                    except Exception as e:
                        logger.error(f"Failed to send get_call response: {e}")
                else:
                    try:
                        await websocket.send_json({'type': 'call_not_found', 'call_id': call_id})
                    except Exception as e:
                        logger.error(f"Failed to send call_not_found: {e}")

            elif msg_type == 'call_answer':
                target = data.get('target')
                call_id = data.get('call_id')
                sdp = data.get('sdp')
                if not target or not call_id:
                    continue
                async with manager._calls_lock:
                    if call_id in manager.calls:
                        manager.calls[call_id]['state'] = 'answered'
                await manager.send_personal_message(target, {
                    'type': 'call_answer',
                    'call_id': call_id,
                    'from': user_id,
                    'sdp': sdp
                })
                logger.debug(f"Call answer {call_id} from {user_id[:8]} to {target[:8]}")

            elif msg_type == 'call_ice':
                target = data.get('target')
                call_id = data.get('call_id')
                candidate = data.get('candidate')
                if not target or not call_id or not candidate:
                    continue
                async with manager._calls_lock:
                    call_info = manager.calls.get(call_id)
                    if call_info:
                        call_info.setdefault('ice_candidates', []).append(candidate)

                await manager.send_personal_message(target, {
                    'type': 'call_ice',
                    'call_id': call_id,
                    'from': user_id,
                    'candidate': candidate
                })

            elif msg_type == 'call_hangup':
                target = data.get('target')
                call_id = data.get('call_id')
                async with manager._calls_lock:
                    manager.calls.pop(call_id, None)
                if target:
                    await manager.send_personal_message(target, {
                        'type': 'call_hangup',
                        'call_id': call_id,
                        'from': user_id
                    })
                    logger.debug(f"Call hangup {call_id} from {user_id[:8]} to {target[:8]}")

            elif msg_type == 'call_reject':
                target = data.get('target')
                call_id = data.get('call_id')
                async with manager._calls_lock:
                    manager.calls.pop(call_id, None)
                if target:
                    await manager.send_personal_message(target, {
                        'type': 'call_reject',
                        'call_id': call_id,
                        'from': user_id
                    })
                    logger.debug(f"Call reject {call_id} from {user_id[:8]} to {target[:8]}")

    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected for {user_id[:16]}")
    except Exception as e:
        logger.error(f"Unexpected error in websocket loop: {e}\n{traceback.format_exc()}")
    finally:
        await manager.disconnect(user_id)