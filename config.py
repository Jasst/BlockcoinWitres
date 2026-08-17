"""
config.py — Централизованная конфигурация приложения
"""
import os
import secrets
import logging
from pathlib import Path

try:
    from dotenv import load_dotenv
    import pathlib
    load_dotenv(dotenv_path=pathlib.Path(__file__).parent / '.env', override=False)
except ImportError:
    pass

CONFIG = {
    'POW_DIFFICULTY':       int(os.getenv('POW_DIFFICULTY', 6)),
    'POW_MAX_ITERATIONS':   int(os.getenv('POW_MAX_ITERATIONS', 100_000_000)),
    'CACHE_SIZE_GROUPS':    32,
    'CACHE_SIZE_CONTACTS':  64,
    'CACHE_SIZE_PUBKEYS':   256,
    'SESSION_LIFETIME':     int(os.getenv('PERMANENT_SESSION_LIFETIME', 31_536_000)),
    'LOG_MAX_BYTES':        10 * 1024 * 1024,
    'LOG_BACKUP_COUNT':     5,
    'LOG_LEVEL':            os.getenv('LOG_LEVEL', 'INFO').upper(),
    'MAX_UPLOAD_SIZE':      16 * 1024 * 1024,
    'RATE_LIMIT_PER_MINUTE':         int(os.getenv('RATE_LIMIT_PER_MINUTE', 60)),
    'RATE_LIMIT_MESSAGE_PER_MINUTE': int(os.getenv('RATE_LIMIT_MESSAGE_PER_MINUTE', 30)),
    'RATE_LIMIT_API_PER_MINUTE':     int(os.getenv('RATE_LIMIT_API_PER_MINUTE', 120)),
}

BASE_DIR = Path(__file__).resolve().parent

# ---------- PostgreSQL ----------
DATABASE_URL = os.getenv('DATABASE_URL', 'postgresql://user:pass@localhost/bichat')

UPLOAD_FOLDER   = os.getenv('UPLOAD_FOLDER', str(BASE_DIR / 'uploads'))
STATIC_FOLDER   = str(BASE_DIR / 'static')
TEMPLATE_FOLDER = str(BASE_DIR / 'templates')

COIN          = 1000000
COIN_NAME     = "BlockCoin"
TRANSFER_FEE  = int(os.getenv('TRANSFER_FEE', 50000))
MESSAGE_FEE = int(os.getenv('MESSAGE_FEE', 100))
MIN_BALANCE   = 0

AIRDROP_AMOUNT  = int(os.getenv('AIRDROP_AMOUNT', 1000))

MIN_STAKE_AMOUNT  = int(os.getenv('MIN_STAKE_AMOUNT', 10 * COIN))
STAKE_LOCK_BLOCKS = int(os.getenv('STAKE_LOCK_BLOCKS', 100))
BLOCK_REWARD = int(os.getenv('BLOCK_REWARD', 0.1*COIN))
STAKING_FEE_FROM_BLOCK_REWARD = float(os.getenv('STAKING_FEE_FROM_BLOCK_REWARD', 0.1))

ENABLE_MINING  = os.getenv('ENABLE_MINING', '1') == '1'
ENABLE_STAKING = os.getenv('ENABLE_STAKING', '1') == '1'

STAKING_FEE_POOL_ADDRESS = 'staking_fee_pool'

SECRET_KEY = os.getenv('SECRET_KEY')
if not SECRET_KEY or len(SECRET_KEY) < 32:
    SECRET_KEY = secrets.token_hex(32)
    logging.warning("SECRET_KEY NOT FOUND in env! Sessions will reset on restart!")

MAX_CONTENT_LENGTH = CONFIG['MAX_UPLOAD_SIZE']
MAX_SUPPLY = 21_000_000

MAX_MESSAGE_PAYLOAD_SIZE = int(os.getenv('MAX_MESSAGE_PAYLOAD_SIZE', 65536))

ONLINE_TIMEOUT_SECONDS = int(os.getenv('ONLINE_TIMEOUT_SECONDS', 60))
MINING_CHALLENGE_TTL = int(os.getenv('MINING_CHALLENGE_TTL', 60))

# Easy Diffusion (локальная генерация изображений)
EASYDIFFUSION_ENABLED = True
EASYDIFFUSION_URL = os.getenv('EASYDIFFUSION_URL', 'http://localhost:9000')
EASYDIFFUSION_TIMEOUT = int(os.getenv('EASYDIFFUSION_TIMEOUT', 160))
EASYDIFFUSION_DEFAULT_STEPS = int(os.getenv('EASYDIFFUSION_STEPS', 20))
EASYDIFFUSION_DEFAULT_WIDTH = int(os.getenv('EASYDIFFUSION_WIDTH', 512))
EASYDIFFUSION_DEFAULT_HEIGHT = int(os.getenv('EASYDIFFUSION_HEIGHT', 512))

MAX_ENCRYPTED_FILE_SIZE = int(os.getenv('MAX_ENCRYPTED_FILE_SIZE', 5 * 1024 * 1024))
MAX_AUDIO_SIZE = int(os.getenv('MAX_AUDIO_SIZE', 2 * 1024 * 1024))

# VAPID для push-уведомлений
# config.py
VAPID_PRIVATE_KEY = os.getenv('VAPID_PRIVATE_KEY', '').strip()
VAPID_PUBLIC_KEY  = os.getenv('VAPID_PUBLIC_KEY', '').strip()
VAPID_SUBJECT     = os.getenv('VAPID_SUBJECT', 'mailto:default@example.com').strip()

DIFFICULTY_ADJUSTMENT_INTERVAL = 2016
TARGET_BLOCK_TIME = 60
MIN_DIFFICULTY = 5
MAX_DIFFICULTY = 15

STAKING_FEE_INCREASE_INTERVAL = 10000
STAKING_FEE_INCREASE_STEP = 0.01
MAX_STAKING_FEE = 0.9

# ------------------------------
# WebRTC / TURN (голосовые/видео звонки)
# ------------------------------
TURN_ENABLED = os.getenv('TURN_ENABLED', '1') == '1'
STUN_SERVER = os.getenv('STUN_SERVER', 'stun:stun.l.google.com:19302')
TURN_SERVER = os.getenv('TURN_SERVER', 'turn:blockchat.ru:3478')
TURN_STATIC_AUTH_SECRET = os.getenv('TURN_STATIC_AUTH_SECRET', '')
TURN_USERNAME = os.getenv('TURN_USERNAME', '')
TURN_PASSWORD = os.getenv('TURN_PASSWORD', '')
TURN_REALM = os.getenv('TURN_REALM', 'blockchat.ru')
TURN_CREDENTIAL_TTL = int(os.getenv('TURN_CREDENTIAL_TTL', '86400'))

# Валидация критичных конфигов при старте
def _validate_config():
    if VAPID_PRIVATE_KEY and not VAPID_PUBLIC_KEY:
        logging.error("CONFIG ERROR: VAPID_PRIVATE_KEY is set, but VAPID_PUBLIC_KEY is missing! Push will not work.")
    if TURN_ENABLED and not TURN_STATIC_AUTH_SECRET and not (TURN_USERNAME and TURN_PASSWORD):
        logging.warning("CONFIG WARN: TURN_ENABLED=1, but no TURN_STATIC_AUTH_SECRET or TURN_USERNAME/PASSWORD set. Calls may fail.")
    if 'user:pass' in DATABASE_URL:
        logging.error("CONFIG ERROR: DATABASE_URL has default credentials. Change it!")

_validate_config()