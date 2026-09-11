# services/messaging.py
import logging
import time
from typing import Any, Dict, List

from cache import get_contact_name_cached, get_contact_cache_version
from database import get_db_cursor

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------
# Основная функция получения списка диалогов (с поддержкой статусов)
# ------------------------------------------------------------------
async def get_conversations_list(user_address: str) -> List[Dict[str, Any]]:
    conversations = []
    try:
        async with get_db_cursor() as conn:
            # 1. Личные диалоги
            rows = await conn.fetch("""
                SELECT
                    partner,
                    MAX(id) AS last_msg_id,
                    MAX(timestamp) AS last_ts,
                    (SELECT sender FROM transactions t2
                     WHERE (t2.sender = $1 AND t2.recipient = partner)
                        OR (t2.sender = partner AND t2.recipient = $1)
                     ORDER BY t2.timestamp DESC LIMIT 1) AS last_sender,
                    (SELECT status FROM transactions t2
                     WHERE (t2.sender = $1 AND t2.recipient = partner)
                        OR (t2.sender = partner AND t2.recipient = $1)
                     ORDER BY t2.timestamp DESC LIMIT 1) AS last_status
                FROM (
                    SELECT
                        CASE WHEN sender = $1 THEN recipient ELSE sender END AS partner,
                        id, timestamp, sender
                    FROM transactions
                    WHERE (sender = $1 OR recipient = $1)
                      AND recipient NOT LIKE 'group:%'
                      AND sender NOT LIKE 'group:%'
                ) t
                WHERE partner IS NOT NULL AND partner != $1
                GROUP BY partner
                ORDER BY last_ts DESC
            """, user_address)

            # 2. Групповые диалоги – используем JSONB оператор @>
            group_rows = await conn.fetch("""
                WITH user_groups AS (
                    SELECT id AS group_id, name
                    FROM groups
                    WHERE members::jsonb @> to_jsonb($1::text)
                ),
                last_group_messages AS (
                    SELECT DISTINCT ON (recipient)
                        recipient,
                        id,
                        timestamp,
                        sender
                    FROM transactions
                    WHERE recipient LIKE 'group:%'
                    ORDER BY recipient, timestamp DESC
                )
                SELECT
                    'group:' || ug.group_id AS partner,
                    COALESCE(lgm.id, 0) AS last_msg_id,
                    COALESCE(lgm.timestamp, 0) AS last_ts,
                    lgm.sender AS last_sender,
                    NULL::text AS last_status   -- у групп нет статуса
                FROM user_groups ug
                LEFT JOIN last_group_messages lgm ON lgm.recipient = 'group:' || ug.group_id
                ORDER BY last_ts DESC
            """, user_address)

            # Объединяем
            all_rows = list(rows) + list(group_rows)
            if not all_rows:
                return []

            # Получаем read_status – защита от пустого списка
            chat_ids = [row['partner'] for row in all_rows]
            read_map = {}
            if chat_ids:
                placeholders = ','.join(f'${i+2}' for i in range(len(chat_ids)))
                read_rows = await conn.fetch(f"""
                    SELECT chat_id, last_read_message_id
                    FROM read_status
                    WHERE user_address = $1 AND chat_id IN ({placeholders})
                """, user_address, *chat_ids)
                read_map = {row['chat_id']: row['last_read_message_id'] for row in read_rows}

            # Кэш групп для имён
            from cache import get_user_groups_cached, get_groups_cache_version
            groups = await get_user_groups_cached(user_address, cache_version=await get_groups_cache_version())
            groups_by_id = {g['id']: g for g in groups}

            for row in all_rows:
                partner = row['partner']
                last_msg_id = row['last_msg_id'] or 0
                last_ts = row['last_ts'] or 0.0
                last_sender = row['last_sender']
                last_read_id = read_map.get(partner, 0)

                is_group = partner.startswith('group:')
                if is_group:
                    group_id = partner.split(':', 1)[1]
                    group = groups_by_id.get(group_id)
                    if not group:          # группа удалена – пропускаем
                        continue
                    name = group['name']
                else:
                    name = (await get_contact_name_cached(user_address, partner,
                            cache_version=await get_contact_cache_version())
                            or partner[:10] + "...")

                # Формируем preview
                if last_sender == user_address:
                    status = row.get('last_status', 'sent')
                    if status == 'read':
                        preview = "✓✓ Прочитано"
                    elif status == 'delivered':
                        preview = "✓✓ Доставлено"
                    else:
                        preview = "✓ Отправлено"
                else:
                    if last_read_id >= last_msg_id:
                        preview = "✓ Прочитано"
                    else:
                        preview = "💬 Новое сообщение"

                conversations.append({
                    'address': partner,
                    'name': name,
                    'is_group': is_group,
                    'last_preview': preview,
                    'last_ts': last_ts,
                })

    except Exception as e:
        logger.error(f"Get conversations error: {e}", exc_info=True)

    return sorted(conversations, key=lambda x: x.get('last_ts', 0), reverse=True)
# ------------------------------------------------------------------
# Кэширующая обёртка (TTL 2 секунды)
# ------------------------------------------------------------------
_conversations_cache: dict = {}
_CONV_CACHE_TTL = 2

async def get_conversations_list_cached(user_address: str) -> List[Dict[str, Any]]:
    now = time.time()
    cached = _conversations_cache.get(user_address)
    if cached and now - cached[1] < _CONV_CACHE_TTL:
        return cached[0]
    result = await get_conversations_list(user_address)
    _conversations_cache[user_address] = (result, now)
    return result

async def invalidate_conversations_cache(user_address: str = None) -> None:
    if user_address:
        _conversations_cache.pop(user_address, None)
    else:
        _conversations_cache.clear()