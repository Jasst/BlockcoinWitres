import asyncio
import json
import logging
import time
from urllib.parse import urlparse

from pywebpush import webpush, WebPushException

from config import VAPID_PRIVATE_KEY, VAPID_SUBJECT
from database import get_db_cursor

logger = logging.getLogger(__name__)


def _send_push_sync(sub: dict, payload: str, vapid_private_key: str, vapid_claims: dict) -> None:
    endpoint = sub['endpoint']
    parsed = urlparse(endpoint)
    aud = f"{parsed.scheme}://{parsed.netloc}"

    claims = {
        "sub": vapid_claims.get("sub"),
        "aud": aud,
        "exp": int(time.time()) + 43200,  # 12 часов
    }

    webpush(
        subscription_info=sub,
        data=payload,
        vapid_private_key=vapid_private_key,
        vapid_claims=claims,
        timeout=5
    )


async def _send_single_push(sub_id: int, sub: dict, payload: str, user_address: str, retries: int = 1):
    for attempt in range(retries + 1):
        try:
            await asyncio.to_thread(
                _send_push_sync,
                sub,
                payload,
                VAPID_PRIVATE_KEY,
                {"sub": VAPID_SUBJECT}
            )
            logger.debug(f"Push sent → {user_address[:16]} {sub.get('endpoint', '')[:50]}")
            return

        except WebPushException as e:
            status_code = getattr(e.response, "status_code", None) if e.response else None
            response_body = ""
            if e.response:
                try:
                    response_body = e.response.text or ""
                except Exception:
                    pass

            if status_code in (404, 410):
                logger.warning(f"Push dead [{status_code}] for {user_address[:16]}, removing subscription")
                async with get_db_cursor() as conn:
                    await conn.execute("DELETE FROM push_subscriptions WHERE id = $1", sub_id)
                return

            if status_code == 403:
                is_vapid_error = any(phrase in response_body for phrase in [
                    'exp claim', 'aud claim', 'sub claim', 'signature',
                    'unauthorized', 'mismatch', 'invalid key'
                ])
                if is_vapid_error:
                    logger.error(f"Push VAPID error for {user_address[:16]}: {response_body[:200]}")
                    return
                else:
                    logger.warning(f"Push forbidden [{status_code}] for {user_address[:16]}, removing subscription")
                    async with get_db_cursor() as conn:
                        await conn.execute("DELETE FROM push_subscriptions WHERE id = $1", sub_id)
                    return

            if status_code and status_code >= 500 and attempt < retries:
                await asyncio.sleep(0.5 * (attempt + 1))
                continue

            logger.warning(f"Push failed [{status_code}] for {user_address[:16]}: {e}")
            return

        except Exception as e:
            logger.exception(f"Push error for {user_address[:16]}: {e}")
            return


async def send_push(
        user_address: str,
        title: str,
        body: str,
        url: str = "/chat",
        push_type: str = "message",
        call_id: str | None = None,
        from_name: str | None = None,
        from_address: str | None = None,
        chat_id: str | None = None  # ✅ НОВОЕ
):
    if not VAPID_PRIVATE_KEY or not VAPID_SUBJECT:
        logger.warning("Push skipped: VAPID keys not configured")
        return

    async with get_db_cursor() as conn:
        rows = await conn.fetch(
            "SELECT id, subscription FROM push_subscriptions WHERE user_address = $1 ORDER BY created_at DESC",
            user_address
        )

    if not rows:
        logger.debug(f"No push subscriptions for {user_address[:16]}")
        return

    seen_endpoints = set()
    unique_subs = []
    for row in rows:
        try:
            sub = json.loads(row["subscription"])
            endpoint = sub.get("endpoint", "")
            if endpoint and endpoint not in seen_endpoints:
                seen_endpoints.add(endpoint)
                unique_subs.append((row["id"], sub))
        except Exception:
            logger.exception("Invalid subscription JSON, skipping")

    if not unique_subs:
        return

    payload_data = {
        "type": push_type,
        "title": title or ("Incoming call" if push_type == "incoming_call" else "New message"),
        "body": body or "",
        "url": url,
    }

    if push_type == "incoming_call":
        payload_data.update({
            "call_id": call_id,
            "from": from_address or "",
            "from_name": from_name or (from_address[:10] if from_address else "Unknown"),
            "url": f"/chat?call_id={call_id}"
        })

    # ✅ НОВОЕ: Добавляем chat_id для сообщений
    if push_type == "message" and chat_id:
        payload_data["chat_id"] = chat_id

    # ✅ Всегда добавляем from для совместимости со SW
    if from_address:
        payload_data["from"] = from_address

    payload = json.dumps(payload_data, ensure_ascii=False)

    tasks = []
    for sub_id, sub in unique_subs:
        tasks.append(_send_single_push(sub_id, sub, payload, user_address))

    if tasks:
        await asyncio.gather(*tasks, return_exceptions=True)