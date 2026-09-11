"""
services/contacts.py — Бизнес-логика работы с контактами (асинхронная версия)
"""
import logging
import time
from typing import Any, Dict, List

from cache import bump_contact_cache_version
from database import get_db_cursor

logger = logging.getLogger(__name__)


async def add_contact(user_address: str, contact_address: str, contact_name: str) -> bool:
    if not contact_name or not contact_name.strip():
        contact_name = contact_address[:10] + "..."
    try:
        async with get_db_cursor() as conn:
            await conn.execute(
                'INSERT INTO contacts (user_address, contact_address, contact_name, created_at) '
                'VALUES ($1, $2, $3, $4) '
                'ON CONFLICT(user_address, contact_address) DO UPDATE SET '
                'contact_name = EXCLUDED.contact_name, created_at = EXCLUDED.created_at',
                user_address, contact_address, contact_name.strip(), time.time()
            )
        await bump_contact_cache_version()   # <-- добавлен await
        return True
    except Exception as e:
        logger.error(f"Add contact DB error: {e}")
        return False


async def get_contacts(user_address: str) -> List[Dict[str, Any]]:
    async with get_db_cursor() as conn:
        rows = await conn.fetch(
            'SELECT contact_address, contact_name, contact_pubkey, created_at '
            'FROM contacts WHERE user_address = $1 '
            'ORDER BY contact_name COLLATE "C"',
            user_address
        )
        return [
            {'address': row[0], 'name': row[1], 'pubkey': row[2], 'created_at': row[3]}
            for row in rows
        ]


async def update_contact_name(user_address: str, contact_address: str, new_name: str) -> bool:
    if not new_name or not new_name.strip():
        return False
    clean_name = ''.join(c for c in new_name.strip() if ord(c) >= 32 and ord(c) != 127)
    if not clean_name or len(clean_name) > 50:
        return False
    try:
        async with get_db_cursor() as conn:
            result = await conn.execute(
                'UPDATE contacts SET contact_name = $1 '
                'WHERE user_address = $2 AND contact_address = $3',
                clean_name, user_address, contact_address.lower()
            )
            updated = result != "UPDATE 0"
        if updated:
            await bump_contact_cache_version()   # <-- добавлен await
            logger.info(f"Contact name updated: {contact_address[:16]}... → '{clean_name}'")
        return bool(updated)
    except Exception as e:
        logger.error(f"Update contact name DB error: {e}")
        return False