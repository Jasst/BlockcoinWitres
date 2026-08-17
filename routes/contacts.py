"""
routes/contacts.py — CRUD-маршруты для контактов (асинхронная версия)
"""
import hmac
import logging

from fastapi import APIRouter, Depends, HTTPException, Request

from cache import get_cached_public_key, get_pubkey_cache_version
from dependencies import require_auth
from models import AddContactRequest, AddContactFromChatRequest, DeleteContactRequest, EditContactRequest
from services.contacts import add_contact, get_contacts, update_contact_name

logger = logging.getLogger(__name__)
router = APIRouter(tags=['contacts'])


@router.post('/add_contact', status_code=201)
async def add_contact_route(body: AddContactRequest, request: Request, address: str = Depends(require_auth)):
    if await add_contact(address, body.address, body.name):
        from services.messaging import invalidate_conversations_cache
        await invalidate_conversations_cache(address)
        return {'message': 'Contact added'}
    raise HTTPException(500, 'Failed to add contact')


@router.post('/add_contact_from_chat', status_code=201)
async def add_contact_from_chat(body: AddContactFromChatRequest, request: Request, address: str = Depends(require_auth)):
    contact_name = (body.contact_name or '').strip() or body.contact_address[:10] + '...'
    if await add_contact(address, body.contact_address, contact_name):
        from services.messaging import invalidate_conversations_cache
        await invalidate_conversations_cache(address)
        logger.info(f"Contact {body.contact_address[:16]}... added from chat")
        return {'message': 'Contact added'}
    raise HTTPException(500, 'Failed to save to database')


@router.get('/get_contacts')
async def get_contacts_route(address: str = Depends(require_auth)):
    user_contacts = await get_contacts(address)
    for contact in user_contacts:
        if not contact.get('pubkey'):
            try:
                pubkey, verified = await get_cached_public_key(
                    contact['address'], cache_version=await get_pubkey_cache_version())
                contact['pubkey'] = pubkey
                contact['pubkey_verified'] = verified
            except Exception:
                contact['pubkey'] = None
                contact['pubkey_verified'] = False
    return {'contacts': user_contacts}


@router.post('/delete_contact')
async def delete_contact_route(body: DeleteContactRequest, request: Request, address: str = Depends(require_auth)):
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        result = await conn.execute(
            'DELETE FROM contacts WHERE user_address = $1 AND contact_address = $2',
            address, body.address
        )
        deleted = result != "DELETE 0"
    from cache import bump_contact_cache_version
    await bump_contact_cache_version()
    from services.messaging import invalidate_conversations_cache
    await invalidate_conversations_cache(address)
    if deleted:
        logger.info(f"Contact {body.address[:16]}... deleted by {address[:16]}...")
        return {'message': 'Contact deleted'}
    raise HTTPException(404, 'Contact not found')


@router.post('/edit_contact')
async def edit_contact_route(body: EditContactRequest, request: Request, address: str = Depends(require_auth)):
    contact_address = body.address
    new_name = body.name
    if hmac.compare_digest(address, contact_address):
        raise HTTPException(400, 'Cannot edit yourself as a contact')
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        row = await conn.fetchrow(
            'SELECT contact_name FROM contacts '
            'WHERE user_address = $1 AND contact_address = $2',
            address, contact_address
        )
        if not row:
            raise HTTPException(404, 'Contact not found')
        old_name = row[0]
    if old_name == new_name:
        return {'message': 'No changes', 'unchanged': True}
    if await update_contact_name(address, contact_address, new_name):
        from services.messaging import invalidate_conversations_cache
        await invalidate_conversations_cache(address)
        return {'message': 'Contact name updated', 'old_name': old_name, 'new_name': new_name}
    raise HTTPException(500, 'Failed to update contact name')