"""Password hashing (bcrypt) and JWT access tokens.

Uses the `bcrypt` library directly — passlib is unmaintained and breaks against
modern bcrypt. Tokens are signed with HS256 using the required SECRET_KEY.

bcrypt note: bcrypt silently TRUNCATES input past 72 bytes (verified on 4.3.0),
so two different long passwords could hash equal. We reject >72 bytes at the API
model boundary (see models.py); the assert here is a defense-in-depth backstop.
"""
import hashlib
import secrets
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


def _create_token(
    subject: str, kind: str, expires: timedelta,
    unique: bool = False, session_id: int | None = None,
) -> str:
    payload = {"sub": subject, "typ": kind, "exp": datetime.now(timezone.utc) + expires}
    if unique:
        payload["jti"] = secrets.token_urlsafe(12)
    if session_id is not None:
        payload["sid"] = str(session_id)
    return jwt.encode(payload, settings.secret_key, algorithm=ALGORITHM)


def create_access_token(subject: str, session_id: int | None = None) -> str:
    """Mint a short-lived JWT whose `sub` is the user id.

    `sid` names the refresh-token record this access token belongs to. It exists
    so logout can end THIS session and no other: the refresh cookie is scoped to
    /auth/refresh and so is never sent to /auth/logout, and widening that scope
    to make logout convenient would mean a long-lived credential travelling on
    requests that have no use for it.
    """
    return _create_token(subject, ACCESS,
                         timedelta(minutes=settings.access_token_expire_minutes),
                         session_id=session_id)


def create_refresh_token(subject: str) -> str:
    """Mint a long-lived token whose only power is obtaining access tokens.

    Carries a random `jti` so every issuance is a DISTINCT string. Without it,
    `exp` has second resolution and two refreshes inside the same second mint
    byte-identical tokens — which would collide on the unique index that makes
    rotation and reuse detection possible, and silently make the new token
    indistinguishable from the one it replaced.
    """
    return _create_token(subject, REFRESH,
                         timedelta(days=settings.refresh_token_expire_days), unique=True)


def _decode(token: str, expected: str) -> dict | None:
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[ALGORITHM])
    except jwt.PyJWTError:  # expired, bad signature, malformed — all invalid
        return None
    # A token minted before `typ` existed has no claim and is refused rather than
    # assumed to be an access token: those sessions simply need a fresh login,
    # which is the safe direction to fail.
    if payload.get("typ") != expected:
        return None
    return payload


def decode_access_token(token: str) -> str | None:
    """Return the subject of a valid, unexpired ACCESS token, else None."""
    payload = _decode(token, ACCESS)
    return payload.get("sub") if payload else None


def access_session_id(token: str) -> int | None:
    """The refresh-token record this access token belongs to, if it names one.

    None for tokens minted before `sid` existed. Those sessions can still be
    logged out of the browser; the server-side record simply outlives them until
    it expires, which is the pre-existing behaviour and not a new gap.
    """
    payload = _decode(token, ACCESS)
    sid = payload.get("sid") if payload else None
    return int(sid) if sid is not None else None


def decode_refresh_token(token: str) -> str | None:
    """Return the subject of a valid, unexpired REFRESH token, else None."""
    payload = _decode(token, REFRESH)
    return payload.get("sub") if payload else None


def hash_token(token: str) -> str:
    """Fingerprint for a token we store server-side (password resets, refresh).

    Stored hashed for the same reason passwords are: a leaked database must not
    hand over working reset links. SHA-256 rather than bcrypt because the input
    is 32 bytes of `secrets` entropy, not a guessable human password — there is
    no dictionary to slow down, so the work factor would buy nothing.
    """
    return hashlib.sha256(token.encode("utf-8")).hexdigest()
