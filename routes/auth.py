"""
routes/auth.py — Регистрация, вход, выход (асинхронная версия)
"""
import logging
import secrets
import time
import base64

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from cache import cache_public_key, clear_all_caches
from config import AIRDROP_AMOUNT, TEMPLATE_FOLDER
from models import CreateWalletRequest, LoginRequest
from setup import verify_address_matches_pubkey
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.asymmetric.utils import encode_dss_signature

logger = logging.getLogger(__name__)
router = APIRouter(tags=['auth'])
templates = Jinja2Templates(directory=TEMPLATE_FOLDER)


@router.get('/', response_class=HTMLResponse)
def index(request: Request):
    if request.session.get('address'):
        return RedirectResponse('/chat')
    return templates.TemplateResponse(request, 'index.html')


@router.get('/chat', response_class=HTMLResponse)
def chat(request: Request):
    if not request.session.get('address'):
        return RedirectResponse('/')
    return templates.TemplateResponse(request, 'chat.html', {
        'address': request.session['address'],
    })


@router.get('/contacts', response_class=HTMLResponse)
def contacts_page(request: Request):
    if not request.session.get('address'):
        return RedirectResponse('/')
    return templates.TemplateResponse(request, 'contacts.html', {
        'address': request.session['address'],
    })


@router.get('/groups', response_class=HTMLResponse)
def groups_page(request: Request):
    if not request.session.get('address'):
        return RedirectResponse('/')
    return templates.TemplateResponse(request, 'groups.html', {
        'address': request.session['address'],
    })


@router.get('/profile', response_class=HTMLResponse)
def profile(request: Request):
    if not request.session.get('address'):
        return RedirectResponse('/')
    return templates.TemplateResponse(request, 'profile.html', {
        'address': request.session['address'],
    })


@router.get('/wallet', response_class=HTMLResponse)
def wallet_page(request: Request):
    if not request.session.get('address'):
        return RedirectResponse('/')
    return templates.TemplateResponse(request, 'wallet.html', {
        'address': request.session['address'],
    })


# ========== НОВЫЙ МАРШРУТ ДЛЯ СТРАНИЦЫ ЗВОНКОВ ==========
@router.get('/calls', response_class=HTMLResponse)
def calls_page(request: Request):
    if not request.session.get('address'):
        return RedirectResponse('/')
    return templates.TemplateResponse(request, 'calls.html', {
        'address': request.session['address'],
    })
# ========================================================


@router.post('/create_wallet', status_code=201)
async def create_wallet(body: CreateWalletRequest, request: Request):
    address = body.address
    pubkey_b64 = body.public_key
    if not verify_address_matches_pubkey(address, pubkey_b64):
        raise HTTPException(400, 'Public key does not match address')
    from database import get_db_cursor
    try:
        nonce = secrets.token_hex(32)
        async with get_db_cursor() as cursor:
            await cursor.execute(
                'INSERT INTO wallets (address, balance, ws_nonce) VALUES ($1, $2, $3) '
                'ON CONFLICT(address) DO UPDATE SET ws_nonce = EXCLUDED.ws_nonce',
                address, AIRDROP_AMOUNT, nonce
            )
            await cursor.execute(
                'INSERT INTO coin_transactions (tx_type, recipient, amount, timestamp) '
                'VALUES ($1, $2, $3, $4)',
                'airdrop', address, AIRDROP_AMOUNT, time.time()
            )
        request.session['address'] = address
        async with get_db_cursor() as conn:
            await conn.execute("""
                INSERT INTO pubkey_cache (address, public_key_b64, updated_at, source, verified)
                VALUES ($1, $2, $3, $4, $5)
                ON CONFLICT (address) DO UPDATE SET
                    public_key_b64 = EXCLUDED.public_key_b64,
                    updated_at = EXCLUDED.updated_at,
                    source = EXCLUDED.source,
                    verified = EXCLUDED.verified
            """, address, pubkey_b64, time.time(), 'self', 1)
        logger.info(f"New wallet registered: {address[:16]}...")
        return {'address': address, 'public_key': pubkey_b64, 'nonce': nonce}
    except Exception as e:
        logger.error(f"Create wallet error: {e}")
        raise HTTPException(500, 'Wallet creation failed')


@router.post('/login')
async def login(body: LoginRequest, request: Request):
    nonce = body.nonce
    if not nonce:
        raise HTTPException(400, 'Nonce missing')

    address = body.address
    pubkey_b64 = body.public_key
    signature_hex = body.signature.strip()

    if not verify_address_matches_pubkey(address, pubkey_b64):
        raise HTTPException(400, 'Public key does not match address')

    try:
        raw_signature = bytes.fromhex(signature_hex)
        if len(raw_signature) != 64:
            raise HTTPException(400, 'Invalid signature format (must be 64 bytes raw)')
        r = int.from_bytes(raw_signature[:32], 'big')
        s = int.from_bytes(raw_signature[32:], 'big')
        der_signature = encode_dss_signature(r, s)

        raw_key = base64.b64decode(pubkey_b64)
        pubkey = ec.EllipticCurvePublicKey.from_encoded_point(ec.SECP256R1(), raw_key)
        pubkey.verify(der_signature, nonce.encode('utf-8'), ec.ECDSA(hashes.SHA256()))
    except HTTPException:
        raise
    except Exception as e:
        logger.warning(f"Signature verification failed: {e}")
        raise HTTPException(403, 'Invalid signature')

    request.session['address'] = address
    from database import get_db_cursor
    async with get_db_cursor() as conn:
        await conn.execute(
            "UPDATE wallets SET ws_nonce = $1 WHERE address = $2",
            nonce, address
        )
    await cache_public_key(address, pubkey_b64, source='self', verified=True)
    logger.info(f"User logged in: {address[:16]}...")
    return {'address': address, 'nonce': nonce}


@router.get('/nonce')
async def get_nonce():
    nonce = secrets.token_hex(32)
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
    await clear_all_caches()
    request.session.clear()
    return RedirectResponse('/')