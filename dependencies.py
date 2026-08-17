"""
dependencies.py — FastAPI dependencies: auth, rate limiting
"""
import logging
from typing import Optional

from fastapi import Depends, HTTPException, Request, status

from setup import general_limiter, message_limiter, RateLimiter

logger = logging.getLogger(__name__)


# =============================================================================
# Session helpers
# =============================================================================

def get_session_address(request: Request) -> Optional[str]:
    """Returns the authenticated address from the session, or None."""
    return request.session.get('address')


def require_auth(request: Request) -> str:
    """Dependency: raises 401 if not authenticated, returns address."""
    address = request.session.get('address')
    if not address:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail='Unauthorized',
        )
    return address


# =============================================================================
# Rate-limit dependencies (factory)
# =============================================================================

def make_rate_limit_dep(limiter: RateLimiter, limit: int = None):
    """
    Returns a FastAPI dependency that enforces rate limiting.
    Uses session address if available, otherwise remote IP.
    """
    def _dep(request: Request) -> None:
        client_id = request.session.get('address') or request.client.host or 'unknown'
        allowed, remaining = limiter.is_allowed(client_id, limit)
        if not allowed:
            raise HTTPException(
                status_code=429,
                detail={
                    'error': 'Too many requests',
                    'retry_after': limiter.window_seconds,
                    'limit': limit or limiter.max_requests,
                },
                headers={
                    'Retry-After': str(limiter.window_seconds),
                    'X-RateLimit-Limit': str(limit or limiter.max_requests),
                    'X-RateLimit-Remaining': '0',
                },
            )
    return _dep


# Pre-built dependencies
RateLimitGeneral  = Depends(make_rate_limit_dep(general_limiter))
RateLimitMessages = Depends(make_rate_limit_dep(message_limiter, limit=30))
RateLimitMining   = Depends(make_rate_limit_dep(general_limiter, limit=3))