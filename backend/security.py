"""Password hashing (bcrypt) and JWT access tokens.

Uses the `bcrypt` library directly — passlib is unmaintained and breaks against
modern bcrypt. Tokens are signed with HS256 using the required SECRET_KEY.

bcrypt note: bcrypt silently TRUNCATES input past 72 bytes (verified on 4.3.0),
so two different long passwords could hash equal. We reject >72 bytes at the API
model boundary (see models.py); the assert here is a defense-in-depth backstop.
"""
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt

from backend.config import settings

ALGORITHM = "HS256"
BCRYPT_MAX_BYTES = 72


def hash_password(password: str) -> str:
    pw = password.encode("utf-8")
    assert len(pw) <= BCRYPT_MAX_BYTES, "password exceeds 72 bytes (guard at model)"
    return bcrypt.hashpw(pw, bcrypt.gensalt()).decode("utf-8")


def verify_password(password: str, hashed: str) -> bool:
    return bcrypt.checkpw(password.encode("utf-8"), hashed.encode("utf-8"))


def create_access_token(subject: str) -> str:
    """Mint a signed JWT whose `sub` is the user id, with an explicit expiry."""
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=settings.access_token_expire_minutes
    )
    payload = {"sub": subject, "exp": expire}
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def decode_access_token(token: str) -> str | None:
    """Return the token's subject (user id) if valid & unexpired, else None."""
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
        return payload.get("sub")
    except jwt.PyJWTError:  # expired, bad signature, malformed — all invalid
        return None
