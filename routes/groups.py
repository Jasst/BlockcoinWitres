"""
routes/groups.py — CRUD-маршруты групповых чатов (асинхронная версия)
"""
import json
import logging
import time
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request

from cache import bump_groups_cache_version, get_groups_cache_version, get_user_groups_cached
from dependencies import require_auth
from models import (CreateGroupRequest, DeleteGroupRequest, RenameGroupRequest,
                    GroupMemberRequest)

logger = logging.getLogger(__name__)
router = APIRouter(tags=['groups'])


@router.get('/get_groups')
async def get_groups(request: Request, address: str = Depends(require_auth)):
    groups = await get_user_groups_cached(address, cache_version=await get_groups_cache_version())
    return {'groups': groups}


@router.post('/create_group', status_code=201)
async def create_group(body: CreateGroupRequest, request: Request, address: str = Depends(require_auth)):
    name = body.name.strip()
    members_set = {m.strip() for m in body.members if m.strip()}
    members_set.add(address)
    members_clean = sorted(members_set)
    invalid = [m for m in members_clean if len(m) != 64]
    if invalid:
        raise HTTPException(400, f'Invalid member addresses: {invalid[:3]}')
    group_id = uuid.uuid4().hex
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        await conn.execute(
            'INSERT INTO groups (id, name, creator, members, created_at) VALUES ($1, $2, $3, $4, $5)',
            group_id, name, address, json.dumps(members_clean), time.time()
        )
    await bump_groups_cache_version()
    from services.messaging import invalidate_conversations_cache
    for member in members_clean:
        await invalidate_conversations_cache(member)
    logger.info(f"Group '{name}' created by {address[:16]}... with {len(members_clean)} members")
    return {
        'message': 'Group created',
        'group_id': group_id,
        'name': name,
        'members': members_clean,
        'member_count': len(members_clean),
    }


@router.post('/delete_group')
async def delete_group(body: DeleteGroupRequest, request: Request, address: str = Depends(require_auth)):
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        row = await conn.fetchrow('SELECT id, name, creator, members FROM groups WHERE id = $1', body.group_id)
        if not row:
            raise HTTPException(404, 'Group not found')
        if row[2] != address:
            raise HTTPException(403, 'Only the group creator can delete this group')
        group_name = row[1]
        members = json.loads(row[3]) if row[3] else []
        await conn.execute('DELETE FROM groups WHERE id = $1', body.group_id)
        # ↓↓↓ ДОБАВИТЬ ЭТУ СТРОКУ ↓↓↓
        await conn.execute('DELETE FROM transactions WHERE recipient = $1', f'group:{body.group_id}')
        # ↑↑↑ КОНЕЦ ВСТАВКИ ↑↑↑
    await bump_groups_cache_version()
    from services.messaging import invalidate_conversations_cache
    for member in members:
        await invalidate_conversations_cache(member)
    logger.info(f"Group '{group_name}' deleted by {address[:16]}...")
    return {'message': 'Group deleted', 'group_id': body.group_id, 'group_name': group_name}

@router.post('/rename_group')
async def rename_group(body: RenameGroupRequest, request: Request, address: str = Depends(require_auth)):
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        row = await conn.fetchrow('SELECT creator, members FROM groups WHERE id = $1', body.group_id)
        if not row:
            raise HTTPException(404, 'Group not found')
        if row[0] != address:
            raise HTTPException(403, 'Only the creator can rename this group')
        members = json.loads(row[1]) if row[1] else []
        await conn.execute('UPDATE groups SET name = $1 WHERE id = $2', body.name, body.group_id)
    await bump_groups_cache_version()
    from services.messaging import invalidate_conversations_cache
    for member in members:
        await invalidate_conversations_cache(member)
    logger.info(f"Group {body.group_id} renamed to '{body.name}' by {address[:16]}...")
    return {'message': 'Group renamed', 'name': body.name}


@router.post('/add_group_member')
async def add_group_member(body: GroupMemberRequest, request: Request, address: str = Depends(require_auth)):
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        row = await conn.fetchrow('SELECT creator, members FROM groups WHERE id = $1', body.group_id)
        if not row:
            raise HTTPException(404, 'Group not found')
        if row[0] != address:
            raise HTTPException(403, 'Only the creator can add members')
        members = json.loads(row[1])
        if body.address in members:
            raise HTTPException(400, 'Address already in group')
        if len(members) >= 50:
            raise HTTPException(400, 'Group member limit (50) reached')
        members.append(body.address)
        members.sort()
        await conn.execute('UPDATE groups SET members = $1 WHERE id = $2',
                           json.dumps(members), body.group_id)
    await bump_groups_cache_version()
    from services.messaging import invalidate_conversations_cache
    await invalidate_conversations_cache(address)
    await invalidate_conversations_cache(body.address)
    logger.info(f"Member {body.address[:16]}... added to group {body.group_id}")
    return {'message': 'Member added', 'members': members}


@router.post('/remove_group_member')
async def remove_group_member(body: GroupMemberRequest, request: Request, address: str = Depends(require_auth)):
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        row = await conn.fetchrow('SELECT creator, members FROM groups WHERE id = $1', body.group_id)
        if not row:
            raise HTTPException(404, 'Group not found')
        creator = row[0]
        members = json.loads(row[1])
        if creator != address:
            raise HTTPException(403, 'Only the creator can remove members')
        if body.address == creator:
            raise HTTPException(400, 'Creator cannot be removed')
        if body.address not in members:
            raise HTTPException(404, 'Address not in group')
        if len(members) <= 2:
            raise HTTPException(400, 'Group must have at least 2 members')
        members.remove(body.address)
        await conn.execute('UPDATE groups SET members = $1 WHERE id = $2',
                           json.dumps(members), body.group_id)
    await bump_groups_cache_version()
    from services.messaging import invalidate_conversations_cache
    await invalidate_conversations_cache(address)
    await invalidate_conversations_cache(body.address)
    logger.info(f"Member {body.address[:16]}... removed from group {body.group_id}")
    return {'message': 'Member removed', 'members': members}