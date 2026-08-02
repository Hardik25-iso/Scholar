"""Password hashing (bcrypt) and JWT access tokens.

Uses the `bcrypt` library directly — passlib is unmaintained and breaks against
modern bcrypt. Tokens are signed with HS256 using the required SECRET_KEY.

bcrypt note: bcrypt silently TRUNCATES input past 72 bytes (verified on 4.3.0),
so two different long passwords could hash equal. We reject >72 bytes at the API
model boundary (see models.py); the assert here is a defense-in-depth backstop.
"""
import hashlib
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


# Both token kinds are signed with the same key, so the payload must say which
# kind it is or they become interchangeable — and a refresh token is long-lived
# by design. Without this claim, a stolen refresh token would work as an access
# token for its whole lifetime, silently undoing the short access expiry.
ACCESS = "access"
REFRESH = "refresh"


def _create_token(subject: str, kind: str, expires: timedelta) -> str:
    payload = {"sub": subject, "typ": kind, "exp": datetime.now(timezone.utc) + expires}
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def create_access_token(subject: str) -> str:
    """Mint a short-lived JWT whose `sub` is the user id."""
    return _create_token(subject, ACCESS,
                         timedelta(minutes=settings.access_token_expire_minutes))


def create_refresh_token(subject: str) -> str:
    """Mint a long-lived token whose only power is obtaining access tokens."""
    return _create_token(subject, REFRESH,
                         timedelta(days=settings.refresh_token_expire_days))


def _decode(token: str, expected: str) -> str | None:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
    except jwt.PyJWTError:  # expired, bad signature, malformed — all invalid
        return None
    # A token minted before `typ` existed has no claim and is refused rather than
    # assumed to be an access token: those sessions simply need a fresh login,
    # which is the safe direction to fail.
    if payload.get("typ") != expected:
        return None
    return payload.get("sub")


def decode_access_token(token: str) -> str | None:
    """Return the subject of a valid, unexpired ACCESS token, else None."""
    return _decode(token, ACCESS)


def decode_refresh_token(token: str) -> str | None:
    """Return the subject of a valid, unexpired REFRESH token, else None."""
    return _decode(token, REFRESH)


def hash_token(token: str) -> str:
    """Fingerprint for a token we store server-side (password resets).

    Stored hashed for the same reason passwords are: a leaked database must not
    hand over working reset links. SHA-256 rather than bcrypt because the input
    is 32 bytes of `secrets` entropy, not a guessable human password — there is
    no dictionary to slow down, so the work factor would buy nothing.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
