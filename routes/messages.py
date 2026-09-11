"""
routes/messages.py — Отправка сообщений (Long Polling удалён, используется WebSocket)
"""
import json
import logging
import time
from typing import Optional
import asyncio
from fastapi import APIRouter, Depends, HTTPException, Query, Request

from cache import (
    get_contact_cache_version, get_contact_name_cached,
    get_groups_cache_version, get_user_groups_cached,
)
from config import MESSAGE_FEE, COIN, COIN_NAME, STAKING_FEE_POOL_ADDRESS, ENABLE_STAKING
from dependencies import require_auth, make_rate_limit_dep
from models import SendMessageRequest, MarkReadRequest, MessageStatusesRequest
from services.messaging import get_conversations_list_cached, invalidate_conversations_cache
from services.wallet import staking_manager
from setup import message_limiter
from services.notifier import message_notifier
from services.push import send_push

logger = logging.getLogger(__name__)
router = APIRouter(tags=['messages'])


@router.post('/send_message', status_code=201,
             dependencies=[Depends(make_rate_limit_dep(message_limiter, limit=30))])
async def send_message(body: SendMessageRequest, request: Request, address: str = Depends(require_auth)):
    blockchain = request.app.state.blockchain
    sender = address
    msg_type = body.message_type
    from database import get_db_cursor
    async with get_db_cursor() as conn:

        if MESSAGE_FEE > 0:

            # Атомарное списание комиссии
            row = await conn.fetchrow("""
                UPDATE wallets
                SET balance = balance - $1
                WHERE address = $2
                  AND balance >= $1
                RETURNING balance
            """, MESSAGE_FEE, sender)

            if not row:
                raise HTTPException(
                    402,
                    f'Insufficient balance for fee ({MESSAGE_FEE / COIN:.6f} {COIN_NAME})'
                )

            # Пополнение staking pool
            await conn.execute("""
                INSERT INTO wallets (address, balance)
                VALUES ($1, $2)
                ON CONFLICT(address)
                DO UPDATE
                SET balance = wallets.balance + EXCLUDED.balance
            """,
                               STAKING_FEE_POOL_ADDRESS,
                               MESSAGE_FEE
                               )

            # Лог комиссии
            await conn.execute("""
                INSERT INTO coin_transactions
                (
                    tx_type,
                    sender,
                    recipient,
                    amount,
                    timestamp,
                    note
                )
                VALUES ($1,$2,$3,$4,$5,$6)
            """,
                               'message_fee',
                               sender,
                               STAKING_FEE_POOL_ADDRESS,
                               MESSAGE_FEE,
                               time.time(),
                               'message fee'
                               )

            if ENABLE_STAKING and staking_manager:
                await staking_manager.add_to_fee_pool(
                    MESSAGE_FEE,
                    cursor=conn
                )

        tx_id = None
        group = None
        message_obj = None
        if msg_type == 'group' and body.group_id:
            if not body.encrypted_map:

                raise HTTPException(400, 'Missing encrypted_map')
            groups = await get_user_groups_cached(sender, cache_version=await get_groups_cache_version())
            group = next((g for g in groups if g['id'] == body.group_id), None)
            if not group or sender not in group['members']:

                raise HTTPException(403, 'Access denied')
            tx_id = await blockchain.new_transaction(
                conn, sender, f"group:{body.group_id}",
                json.dumps({'encrypted_map': body.encrypted_map}), None,
                sender_pubkey=body.sender_pubkey,
                metadata={'encryption': 'group-ecdh-v4', 'group_id': body.group_id}
            )
            message_obj = {
                'id': tx_id, 'sender': sender, 'sender_name': None,
                'chatId': f"group:{body.group_id}", 'isGroup': True,
                'recipient': f"group:{body.group_id}",
                'preview': '💬 Новое сообщение в группе',
                'timestamp': time.time(),
                'content': json.dumps({'encrypted_map': body.encrypted_map}),
                'image': None,
                'status': 'sent'
            }



            for member in group['members']:
                await message_notifier.add_message(member, message_obj)

            for member in group['members']:
                if member != sender:
                    asyncio.create_task(
                        send_push(
                            user_address=member,
                            title=f"Группа {group['name']}",
                            body="Новое сообщение в группе",
                            url=f"/chat?start_with=group:{body.group_id}"
                        )
                    )
        else:
            recipient = body.recipient
            if not recipient:

                raise HTTPException(400, 'Missing recipient')
            if sender == recipient:

                raise HTTPException(400, 'Cannot message yourself')
            if not body.payload or not isinstance(body.payload, dict):

                raise HTTPException(400, 'Missing encrypted payload')
            tx_id = await blockchain.new_transaction(
                conn, sender, recipient,
                json.dumps(body.payload), None,
                sender_pubkey=body.sender_pubkey,
                metadata={'encryption': 'hybrid-v2'}
            )
            message_obj = {
                'id': tx_id, 'sender': sender, 'sender_name': None,
                'chatId': recipient, 'isGroup': False,
                'recipient': recipient,
                'preview': '💬 Новое сообщение',
                'timestamp': time.time(),
                'content': json.dumps(body.payload),
                'image': None,
                'status': 'sent'
            }

            await message_notifier.add_message(recipient, message_obj)

            asyncio.create_task(
                send_push(
                    user_address=recipient,
                    title=f"Новое сообщение от {sender[:8]}...",
                    body="Новое сообщение",
                    url=f"/chat?start_with={sender}"
                )
            )

    await invalidate_conversations_cache(sender)
    if msg_type == 'group' and body.group_id and group:
        for member in group['members']:
            await invalidate_conversations_cache(member)
    else:
        await invalidate_conversations_cache(body.recipient)

    return {'message': 'Sent', 'tx_id': tx_id, 'type': msg_type, 'fee': MESSAGE_FEE}

@router.get('/get_conversation')
async def get_conversation(
    request: Request,
    address: str = Depends(require_auth),
    with_:   str = Query(alias='with', default=None),
    last_message_id: Optional[int] = Query(default=None),
    limit:   int = Query(default=30),
    before_id: Optional[int] = Query(default=None),
):
    if not with_:
        raise HTTPException(400, 'Missing "with" parameter')
    chat_with = with_
    limit = min(limit, 50)
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        if chat_with.startswith('group:'):
            group_id = chat_with.split(':', 1)[1]
            groups = await get_user_groups_cached(address, cache_version=await get_groups_cache_version())
            if not any(g['id'] == group_id and address in g['members'] for g in groups):
                raise HTTPException(403, 'No access')
            # Добавлено поле status
            query = ('SELECT id, sender, recipient, content, image, timestamp, metadata, status '
                     'FROM transactions WHERE recipient = $1')
            params = [chat_with]
            if last_message_id:
                row_ts = await conn.fetchrow("SELECT timestamp FROM transactions WHERE id = $1", last_message_id)
                if row_ts:
                    query += ' AND timestamp > $2'
                    params.append(row_ts[0])
            if before_id:
                row_ts = await conn.fetchrow("SELECT timestamp FROM transactions WHERE id = $1", before_id)
                if row_ts:
                    query += ' AND timestamp < $2'
                    params.append(row_ts[0])
            query += ' ORDER BY timestamp ASC LIMIT $2' if len(params) == 1 else f' ORDER BY timestamp ASC LIMIT ${len(params)+1}'
            params.append(limit)
            rows = await conn.fetch(query, *params)
        else:
            # Добавлено поле status
            base_query = """
                SELECT id, sender, recipient, content, image, timestamp, metadata, status
                FROM (
                    SELECT * FROM transactions WHERE sender = $1 AND recipient = $2
                    UNION ALL
                    SELECT * FROM transactions WHERE sender = $3 AND recipient = $4
                ) AS t
            """
            params = [address, chat_with, chat_with, address]
            conditions = []
            if last_message_id:
                conditions.append("id > $5")
                params.append(last_message_id)
            if before_id:
                conditions.append("id < $5")
                params.append(before_id)
            if conditions:
                base_query += " WHERE " + " AND ".join(conditions)
            base_query += " ORDER BY timestamp DESC LIMIT $5" if len(params) == 4 else f" ORDER BY timestamp DESC LIMIT ${len(params)+1}"
            params.append(limit)
            rows = await conn.fetch(base_query, *params)
    messages = []
    for r in rows:
        messages.append({
            'id':           r[0],
            'sender':       r[1],
            'recipient':    r[2],
            'content':      r[3],
            'image':        r[4],
            'timestamp':    r[5],
            'sender_pubkey': None,
            'metadata':     r[6],
            'status':       r[7] if len(r) > 7 else 'sent',   # добавлено
            'sender_name':  (await get_contact_name_cached(address, r[1],
                             cache_version=await get_contact_cache_version()) or r[1][:10] + '...'),
            'recipient_name': (await get_contact_name_cached(address, r[2],
                               cache_version=await get_contact_cache_version()) or r[2][:10] + '...'),
            'is_mine': (r[1] == address),
        })
    if not chat_with.startswith('group:'):
        messages.reverse()
    return {'messages': messages, 'chat_with': chat_with}


@router.get('/get_conversations')
async def get_conversations(address: str = Depends(require_auth)):
    return {'conversations': await get_conversations_list_cached(address)}


@router.post('/mark_conversation_read')
async def mark_conversation_read(body: MarkReadRequest, request: Request, address: str = Depends(require_auth)):
    chat_with = body.chat_with.strip()
    if not chat_with:
        raise HTTPException(400, 'Missing chat_with')
    last_message_id = body.last_message_id
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        if last_message_id is None:
            if chat_with.startswith('group:'):
                row = await conn.fetchrow('SELECT MAX(id) FROM transactions WHERE recipient = $1', chat_with)
            else:
                row = await conn.fetchrow('''
                    SELECT MAX(id) FROM transactions
                    WHERE (sender = $1 AND recipient = $2) OR (sender = $3 AND recipient = $4)
                ''', address, chat_with, chat_with, address)
            last_message_id = row[0] if row and row[0] else 0
        await conn.execute('''
            INSERT INTO read_status (user_address, chat_id, last_read_message_id, read_at)
            VALUES ($1, $2, $3, $4)
            ON CONFLICT(user_address, chat_id) DO UPDATE
            SET last_read_message_id = excluded.last_read_message_id,
                read_at = excluded.read_at
            WHERE excluded.last_read_message_id > read_status.last_read_message_id
        ''', address, chat_with, last_message_id, time.time())
    return {'status': 'ok', 'last_read': last_message_id}


@router.post('/message/{message_id}/delivered')
async def mark_delivered(message_id: int, request: Request, address: str = Depends(require_auth)):
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        await conn.execute('''
            UPDATE transactions SET status = 'delivered'
            WHERE id = $1 AND recipient = $2 AND status = 'sent'
        ''', message_id, address)
    return {'status': 'ok'}


@router.post('/message/{message_id}/read')
async def mark_read(message_id: int, request: Request, address: str = Depends(require_auth)):
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        await conn.execute('''
            UPDATE transactions SET status = 'read', read_at = $1
            WHERE id = $2 AND recipient = $3 AND status IN ('sent', 'delivered')
        ''', time.time(), message_id, address)
    return {'status': 'ok'}


@router.post('/message/statuses')
async def get_message_statuses(body: MessageStatusesRequest, request: Request, address: str = Depends(require_auth)):
    if not body.ids:
        return {}
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        placeholders = ','.join(f'${i+1}' for i in range(len(body.ids)))
        rows = await conn.fetch(f'SELECT id, status FROM transactions WHERE id IN ({placeholders})', *body.ids)
    return {str(row[0]): row[1] for row in rows}


@router.get('/get_public_key/{addr}')
async def get_public_key(addr: str, address: str = Depends(require_auth)):
    from cache import get_cached_public_key, get_pubkey_cache_version, fetch_public_key_from_chain
    pubkey, verified = await get_cached_public_key(addr, cache_version=await get_pubkey_cache_version())
    if not pubkey:
        pubkey, verified = await fetch_public_key_from_chain(addr)
    if pubkey:
        return {'address': addr, 'public_key': pubkey, 'verified': verified}
    raise HTTPException(404, 'Public key not found')


@router.get('/search_messages')
async def search_messages(
    request: Request,
    q:       str = Query(default=''),
    address: str = Depends(require_auth),
):
    if len(q.strip()) < 2:
        raise HTTPException(400, 'Query too short')
    blockchain = request.app.state.blockchain
    results = await blockchain.search_messages(address, q.strip())
    return {'results': results}


@router.post('/clear_conversation')
async def clear_conversation(body: dict, request: Request, address: str = Depends(require_auth)):
    """Удаляет все сообщения в указанном диалоге (для текущего пользователя)."""
    chat_with = body.get('chat_with')
    if not chat_with:
        raise HTTPException(400, 'Missing chat_with')
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        if chat_with.startswith('group:'):
            groups = await get_user_groups_cached(address, cache_version=await get_groups_cache_version())
            group_id = chat_with.split(':', 1)[1]
            if not any(g['id'] == group_id and address in g['members'] for g in groups):
                raise HTTPException(403, 'No access')
            await conn.execute('DELETE FROM transactions WHERE recipient = $1', chat_with)
        else:
            await conn.execute('''
                DELETE FROM transactions
                WHERE (sender = $1 AND recipient = $2) OR (sender = $3 AND recipient = $4)
            ''', address, chat_with, chat_with, address)
        await invalidate_conversations_cache(address)
    return {'status': 'ok'}


@router.post('/delete_message')
async def delete_message(body: dict, request: Request, address: str = Depends(require_auth)):
    """Удаляет одно сообщение, если оно принадлежит текущему пользователю."""
    msg_id = body.get('message_id')
    if not msg_id:
        raise HTTPException(400, 'Missing message_id')
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        row = await conn.fetchrow('SELECT sender, recipient FROM transactions WHERE id = $1', msg_id)
        if not row:
            raise HTTPException(404, 'Message not found')
        sender, recipient = row
        if sender != address:
            raise HTTPException(403, 'You can only delete your own messages')
        await conn.execute('DELETE FROM transactions WHERE id = $1', msg_id)
        await invalidate_conversations_cache(address)
        if recipient.startswith('group:'):
            groups = await get_user_groups_cached(address, cache_version=await get_groups_cache_version())
            group_id = recipient.split(':', 1)[1]
            group = next((g for g in groups if g['id'] == group_id), None)
            if group:
                for member in group['members']:
                    await invalidate_conversations_cache(member)
        else:
            await invalidate_conversations_cache(recipient)
    return {'status': 'ok'}