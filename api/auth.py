"""
COSMEON Authentication — JWT-based auth with role support.

Roles:
  admin   — full access (train, configure scheduler, register users)
  analyst — can analyze regions, view everything
  viewer  — read-only access

Tokens are signed HS256 JWTs, valid 24 hours.
"""
import hashlib
import hmac
import logging
import os
from datetime import datetime, timedelta
from typing import Optional

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

logger = logging.getLogger("cosmeon.auth")

SECRET_KEY = os.getenv("SECRET_KEY", "cosmeon-dev-secret-please-change-in-production")
ALGORITHM = "HS256"
TOKEN_EXPIRE_HOURS = 24

security = HTTPBearer(auto_error=False)


# ── Password hashing ─────────────────────────────────────────────────────────

def hash_password(password: str) -> str:
    """Hash password using PBKDF2-SHA256 with random salt (no extra deps)."""
    salt = os.urandom(16)
    key = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
    return salt.hex() + ":" + key.hex()


def verify_password(password: str, stored_hash: str) -> bool:
    """Verify a plaintext password against a stored PBKDF2 hash."""
    try:
        salt_hex, key_hex = stored_hash.split(":", 1)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(key_hex)
        actual = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100_000)
        return hmac.compare_digest(actual, expected)
    except Exception:
        return False


# ── JWT tokens ────────────────────────────────────────────────────────────────

def _get_jwt():
    """Lazy import PyJWT."""
    try:
        import jwt as _jwt
        return _jwt
    except ImportError:
        return None


def create_token(user_id: int, username: str, role: str) -> str:
    """Create a signed JWT token for a user."""
    import json, base64
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": role,
        "iat": int(datetime.utcnow().timestamp()),
        "exp": int((datetime.utcnow() + timedelta(hours=TOKEN_EXPIRE_HOURS)).timestamp()),
    }
    _jwt = _get_jwt()
    if _jwt:
        return _jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)
    # Fallback: HMAC-signed JSON (no PyJWT) — use for local dev only
    import hmac as _hmac
    header = base64.urlsafe_b64encode(json.dumps({"alg": "HS256"}).encode()).decode().rstrip("=")
    body = base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
    sig_input = f"{header}.{body}".encode()
    sig = _hmac.new(SECRET_KEY.encode(), sig_input, hashlib.sha256).hexdigest()
    return f"{header}.{body}.{sig}"


def decode_token(token: str) -> dict:
    """Decode and validate a JWT token. Raises HTTPException on failure."""
    _jwt = _get_jwt()
    if _jwt:
        try:
            return _jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        except Exception as e:
            raise HTTPException(status_code=401, detail=f"Invalid token: {e}")
    # Fallback: parse and verify our custom token
    import json, base64, hmac as _hmac
    try:
        parts = token.split(".")
        if len(parts) != 3:
            raise ValueError("bad format")
        header, body, sig = parts
        expected = _hmac.new(SECRET_KEY.encode(), f"{header}.{body}".encode(), hashlib.sha256).hexdigest()
        if not _hmac.compare_digest(sig, expected):
            raise ValueError("bad signature")
        payload = json.loads(base64.urlsafe_b64decode(body + "=="))
        if payload.get("exp", 0) < int(datetime.utcnow().timestamp()):
            raise HTTPException(status_code=401, detail="Token expired")
        return payload
    except HTTPException:
        raise
    except Exception:
        raise HTTPException(status_code=401, detail="Invalid token")


# ── FastAPI dependencies ──────────────────────────────────────────────────────

def get_current_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> dict:
    """Require a valid JWT token. Raises 401 if missing or invalid."""
    if not credentials:
        raise HTTPException(status_code=401, detail="Authentication required")
    return decode_token(credentials.credentials)


def get_optional_user(
    credentials: Optional[HTTPAuthorizationCredentials] = Depends(security),
) -> Optional[dict]:
    """Extract user from JWT if present, return None otherwise."""
    if not credentials:
        return None
    try:
        return decode_token(credentials.credentials)
    except HTTPException:
        return None


def require_role(*roles: str):
    """Dependency factory: require the user to have one of the specified roles."""
    def _check(user: dict = Depends(get_current_user)):
        if user.get("role") not in roles:
            raise HTTPException(
                status_code=403,
                detail=f"Access denied — requires role: {list(roles)}",
            )
        return user
    return _check
