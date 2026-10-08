"""
routes/auth.py — Регистрация, вход, выход (асинхронная версия)
"""
import base64
import logging
import secrets
import time

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from cache import cache_public_key, clear_all_caches
from config import AIRDROP_AMOUNT, TEMPLATE_FOLDER
from database import get_db_cursor
from models import CreateWalletRequest, LoginRequest
from setup import verify_address_matches_pubkey

logger = logging.getLogger(__name__)
router = APIRouter(tags=['auth'])
templates = Jinja2Templates(directory=TEMPLATE_FOLDER)

# IMPORTANT: do NOT wrap/replace templates.env.globals['url_for'].
# Starlette's built-in url_for is a closure that reads the request from the
# Jinja render context; any wrapper breaks every page with
# "TypeError: url_for() missing 1 required positional argument: 'name'" -> HTTP 500.
# Cache-busting for JS/CSS is handled server-side in main.py via the
# VersionedStaticFiles class (Cache-Control: no-cache headers on /static assets).


def _verify_signature(pubkey_b64: str, signature_hex: str, nonce: str) -> None:
    """Проверяет ECDSA-подпись (secp256r1, SHA-256) над nonce.
    При любой ошибке поднимает HTTPException."""
    try:
        raw_signature = bytes.fromhex(signature_hex.strip())
    except ValueError:
        raise HTTPException(403, 'Invalid signature')
    if len(raw_signature) != 64:
        raise HTTPException(400, 'Invalid signature format (must be 64 bytes raw)')

    r = int.from_bytes(raw_signature[:32], 'big')
    s = int.from_bytes(raw_signature[32:], 'big')
    der_signature = encode_dss_signature(r, s)
    try:
        raw_key = base64.b64decode(pubkey_b64)
        pubkey = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), raw_key)
        pubkey.verify(der_signature, nonce.encode('utf-8'), ec.ECDSA(hashes.SHA256()))
    except Exception as e:
        logger.warning(f"Signature verification failed: {e}")
        raise HTTPException(403, 'Invalid signature')


def _render_protected(request: Request, template: str):
    """Страница, доступная только после входа; иначе редирект на главную."""
    if not request.session.get('address'):
        return RedirectResponse('/')
    return templates.TemplateResponse(request, template, {
        'address': request.session['address'],
    })


@router.get('/', response_class=HTMLResponse)
def index(request: Request):
    if request.session.get('address'):
        return RedirectResponse('/chat')
    return templates.TemplateResponse(request, 'index.html')


@router.get('/chat', response_class=HTMLResponse)
def chat(request: Request):
    return _render_protected(request, 'chat.html')


@router.get('/contacts', response_class=HTMLResponse)
def contacts_page(request: Request):
    return _render_protected(request, 'contacts.html')


@router.get('/groups', response_class=HTMLResponse)
def groups_page(request: Request):
    return _render_protected(request, 'groups.html')


@router.get('/profile', response_class=HTMLResponse)
def profile(request: Request):
    return _render_protected(request, 'profile.html')


@router.get('/wallet', response_class=HTMLResponse)
def wallet_page(request: Request):
    return _render_protected(request, 'wallet.html')


@router.get('/calls', response_class=HTMLResponse)
def calls_page(request: Request):
    return _render_protected(request, 'calls.html')


@router.post('/create_wallet', status_code=201)
async def create_wallet(body: CreateWalletRequest, request: Request):
    address = body.address
    pubkey_b64 = body.public_key
    if not verify_address_matches_pubkey(address, pubkey_b64):
        raise HTTPException(400, 'Public key does not match address')

    try:
        nonce = secrets.token_hex(32)
        async with get_db_cursor() as cursor:
            # ON CONFLICT DO NOTHING: существующий кошелёк не перезаписываем.
            # Раньше здесь обновлялся ws_nonce и повторно начислялся airdrop.
            created = await cursor.fetchrow(
                'INSERT INTO wallets (address, balance, ws_nonce) VALUES ($1, $2, $3) '
                'ON CONFLICT (address) DO NOTHING RETURNING address',
                address, AIRDROP_AMOUNT, nonce
            )
            if created is None:
                raise HTTPException(409, 'Wallet already exists, please log in')

            await cursor.execute(
                'INSERT INTO coin_transactions (tx_type, recipient, amount, timestamp) '
                'VALUES ($1, $2, $3, $4)',
                'airdrop', address, AIRDROP_AMOUNT, time.time()
            )
            await cursor.execute("""
                INSERT INTO pubkey_cache (address, public_key_b64, updated_at, source, verified)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (address) DO UPDATE SET
                    public_key_b64 = EXCLUDED.public_key_b64,
                    updated_at = EXCLUDED.updated_at,
                    source = EXCLUDED.source,
                    verified = EXCLUDED.verified
            """, address, pubkey_b64, time.time(), 'self', 1)

        request.session['address'] = address
        logger.info(f"New wallet registered: {address[:8]}…")
        return {'address': address, 'public_key': pubkey_b64, 'nonce': nonce}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Create wallet error: {e}")
        raise HTTPException(500, 'Wallet creation failed')


@router.post('/login')
async def login(body: LoginRequest, request: Request):
    # 1) Nonce должен быть тем, что сервер выдал в GET /login, и использоваться один раз.
    #    Без этой проверки перехваченный запрос можно было бы повторить (replay).
    expected_nonce = request.session.pop('login_nonce', None)
    if not expected_nonce or body.nonce != expected_nonce:
        raise HTTPException(400, 'Invalid or expired nonce, reload the login page')

    # 2) Ключ должен соответствовать адресу, подпись — этому ключу.
    if not verify_address_matches_pubkey(body.address, body.public_key):
        raise HTTPException(400, 'Public key does not match address')
    _verify_signature(body.public_key, body.signature, body.nonce)

    request.session['address'] = body.address
    async with get_db_cursor() as conn:
        await conn.execute(
            'UPDATE wallets SET ws_nonce = $1 WHERE address = $2',
            body.nonce, body.address
        )
    await cache_public_key(body.address, body.public_key, source='self', verified=True)
    logger.info(f"User logged in: {body.address[:8]}…")
    return {'address': body.address, 'nonce': body.nonce}


@router.get('/nonce')
async def get_nonce(request: Request):
    """Выдаёт одноразовый nonce для входа и сохраняет его в сессии."""
    nonce = secrets.token_hex(32)
    request.session['login_nonce'] = nonce
    return {'nonce': nonce}


@router.get('/login', response_class=HTMLResponse)
def login_page(request: Request):
    nonce = secrets.token_hex(32)
    request.session['login_nonce'] = nonce
    return templates.TemplateResponse(request, 'login.html', {'nonce': nonce})


@router.get('/check_session')
def check_session(request: Request):
    return {
        'authenticated': 'address' in request.session,
        'address': request.session.get('address'),
    }


@router.get('/logout')
async def logout(request: Request):
    # ВНИМАНИЕ: clear_all_caches() очищает кэши для ВСЕХ пользователей.
    # Лучше заменить на очистку кэша только текущего адреса (см. cache.py).
    await clear_all_caches()
    request.session.clear()
    return RedirectResponse('/')