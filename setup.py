"""
setup.py — Криптография, логирование, rate limiter (без Flask)
"""
import base64
import hashlib
import hmac
import logging
import logging.handlers
import os
import threading
import time
from collections import defaultdict
from typing import Dict, Tuple

from cryptography.hazmat.primitives.asymmetric import ec
from config import CONFIG

# =============================================================================
# Crypto
# =============================================================================

CURVE = ec.SECP256R1()


def load_public_key_from_bytes(pubkey_bytes: bytes):
    return ec.EllipticCurvePublicKey.from_encoded_point(CURVE, pubkey_bytes)


def load_public_key_from_b64(pubkey_b64: str):
    return load_public_key_from_bytes(base64.b64decode(pubkey_b64))


def verify_address_matches_pubkey(address: str, pubkey_b64: str) -> bool:
    try:
        pubkey_bytes = base64.b64decode(pubkey_b64)
        computed = hashlib.sha256(pubkey_bytes).hexdigest()
        return hmac.compare_digest(computed, address)
    except Exception:
        return False


# =============================================================================
# Logging
# =============================================================================

def setup_logging() -> logging.Logger:
    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'messenger.log')

    file_handler = logging.handlers.RotatingFileHandler(
        log_path,
        maxBytes=CONFIG['LOG_MAX_BYTES'],
        backupCount=CONFIG['LOG_BACKUP_COUNT'],
        encoding='utf-8',
        delay=True,
    )
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s [%(name)s:%(lineno)d] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S',
    ))

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

    is_prod = os.getenv('FLASK_ENV') == 'production'
    log_level = logging.WARNING if is_prod else logging.INFO
    handlers = [file_handler] if is_prod else [file_handler, console_handler]

    logging.basicConfig(level=log_level, handlers=handlers, force=True)
    logging.getLogger('uvicorn').setLevel(logging.WARNING)
    logging.getLogger('sqlite3').setLevel(logging.ERROR)
    logging.getLogger('cryptography').setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.warning(f"=== App started, log path: {log_path} ===")
    return logger


# =============================================================================
# Rate Limiter (thread-safe, framework-agnostic)
# =============================================================================

class RateLimiter:
    def __init__(self, max_requests: int = 60, window_seconds: int = 60):
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._clients: Dict[str, list] = defaultdict(list)
        self._lock = threading.Lock()
        self._cleanup_counter = 0

    def is_allowed(self, client_id: str, limit: int = None) -> Tuple[bool, int]:
        max_req = limit or self.max_requests
        now = time.time()
        with self._lock:
            self._clients[client_id] = [
                t for t in self._clients[client_id]
                if now - t < self.window_seconds
            ]
            if len(self._clients[client_id]) >= max_req:
                return False, 0
            self._clients[client_id].append(now)
            remaining = max_req - len(self._clients[client_id])
            self._cleanup_counter += 1
            if self._cleanup_counter > 100:
                self._cleanup()
                self._cleanup_counter = 0
            return True, remaining

    def _cleanup(self):
        now = time.time()
        expired = [
            cid for cid, ts in self._clients.items()
            if not ts or (now - ts[-1]) > self.window_seconds * 2
        ]
        for cid in expired:
            del self._clients[cid]

    def get_stats(self) -> dict:
        with self._lock:
            return {
                'total_clients': len(self._clients),
                'active_clients': sum(
                    1 for ts in self._clients.values()
                    if ts and time.time() - ts[-1] < self.window_seconds
                ),
                'window_seconds': self.window_seconds,
                'max_requests': self.max_requests,
            }


general_limiter = RateLimiter(max_requests=CONFIG['RATE_LIMIT_PER_MINUTE'])
message_limiter = RateLimiter(max_requests=CONFIG['RATE_LIMIT_MESSAGE_PER_MINUTE'])
api_limiter = RateLimiter(max_requests=CONFIG['RATE_LIMIT_API_PER_MINUTE'])


def get_rate_limit_stats() -> dict:
    return {
        'general': general_limiter.get_stats(),
        'messages': message_limiter.get_stats(),
        'api': api_limiter.get_stats(),
    }