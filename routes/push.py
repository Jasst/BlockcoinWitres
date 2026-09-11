# routes/push.py

import json
import logging
import hashlib

from fastapi import APIRouter, Depends, HTTPException, Request

from dependencies import require_auth
from database import get_db_cursor
from config import VAPID_PUBLIC_KEY

logger = logging.getLogger(__name__)
router = APIRouter(prefix='/push', tags=['push'])


def _endpoint_hash(subscription: dict) -> str:
    """Хеш endpoint'а – стабильный идентификатор подписки (iOS меняет ключи, но endpoint остаётся)."""
    endpoint = subscription.get('endpoint', '')
    return hashlib.sha256(endpoint.encode()).hexdigest()


@router.get('/vapid-public-key')
async def get_vapid_public_key():
    """Возвращает публичный VAPID-ключ для подписки на клиенте."""
    return {'publicKey': VAPID_PUBLIC_KEY}


@router.post('/subscribe')
async def subscribe(request: Request, address: str = Depends(require_auth)):
    """Сохраняет или обновляет push-подписку (UPSERT по user_address + endpoint_hash)."""
    try:
        sub = await request.json()
        if not sub.get('endpoint'):
            raise HTTPException(400, 'Missing endpoint')

        endpoint_hash = _endpoint_hash(sub)
        sub_json = json.dumps(sub)

        async with get_db_cursor() as conn:
            await conn.execute("""
                INSERT INTO push_subscriptions (user_address, subscription, endpoint_hash, created_at)
                VALUES ($1, $2, $3, extract(epoch from now()))
                ON CONFLICT (user_address, endpoint_hash)
                DO UPDATE SET
                    subscription = EXCLUDED.subscription,
                    created_at = extract(epoch from now())
            """, address, sub_json, endpoint_hash)

        logger.info(f"Push subscription saved for {address[:16]}")
        return {'status': 'ok'}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Subscribe error: {e}")
        raise HTTPException(500, 'Failed to save subscription')


@router.post('/unsubscribe')
async def unsubscribe(request: Request, address: str = Depends(require_auth)):
    """Удаляет подписку пользователя."""
    try:
        sub = await request.json()
        endpoint_hash = _endpoint_hash(sub)

        async with get_db_cursor() as conn:
            await conn.execute("""
                DELETE FROM push_subscriptions
                WHERE user_address = $1 AND endpoint_hash = $2
            """, address, endpoint_hash)

        return {'status': 'ok'}

    except Exception as e:
        logger.error(f"Unsubscribe error: {e}")
        raise HTTPException(500, 'Failed')


@router.post('/renew')
async def renew_subscription(request: Request):
    """
    Обновление push-подписки из Service Worker (без require_auth,
    т.к. сессия может быть истекшей). Авторизация через старый endpoint.
    """
    try:
        body = await request.json()
        old_endpoint = body.get('old_endpoint')
        new_subscription = body.get('subscription')

        if not old_endpoint or not new_subscription or not new_subscription.get('endpoint'):
            raise HTTPException(400, 'Missing old_endpoint or subscription')

        old_hash = hashlib.sha256(old_endpoint.encode()).hexdigest()

        async with get_db_cursor() as conn:
            # Находим пользователя по старому endpoint
            row = await conn.fetchrow(
                "SELECT user_address FROM push_subscriptions WHERE endpoint_hash = $1",
                old_hash
            )

            if not row:
                logger.warning(f"Push renew: old endpoint not found")
                raise HTTPException(404, 'Old subscription not found')

            user_address = row['user_address']
            new_hash = _endpoint_hash(new_subscription)
            sub_json = json.dumps(new_subscription)

            # Удаляем старую, вставляем новую
            await conn.execute(
                "DELETE FROM push_subscriptions WHERE endpoint_hash = $1",
                old_hash
            )
            await conn.execute("""
                INSERT INTO push_subscriptions (user_address, subscription, endpoint_hash, created_at)
                VALUES ($1, $2, $3, extract(epoch from now()))
                ON CONFLICT (user_address, endpoint_hash)
                DO UPDATE SET subscription = EXCLUDED.subscription, created_at = EXCLUDED.created_at
            """, user_address, sub_json, new_hash)

        logger.info(f"Push subscription renewed for {user_address[:16]}")
        return {'status': 'ok'}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Renew error: {e}")
        raise HTTPException(500, 'Failed')