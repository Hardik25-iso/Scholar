"""Auth routes + dependencies: register, login, logout, me.

Token transport = an httpOnly cookie (JS can't read it, so XSS can't steal the
token — unlike localStorage). Because the browser sends cookies automatically,
we add CSRF protection on state-changing routes via the double-submit pattern:

  - On login we set TWO cookies: `access_token` (httpOnly) and `csrf_token`
    (readable by JS).
  - The frontend echoes the csrf value back in an `X-CSRF-Token` header on
    unsafe requests. A cross-site attacker can send the cookie but cannot read
    it to set the matching header (blocked by the same-origin policy), so the
    check fails. Safe methods (GET /me) don't need it.
"""
import logging
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Response, status
from sqlmodel import Session, select

from backend.config import settings
from backend.db import get_session
from backend.db_models import PasswordResetToken, User
from backend.models import (
    ForgotPasswordRequest, LoginRequest, RegisterRequest, ResetPasswordRequest, UserPublic,
)
from backend.security import (
    create_access_token, create_refresh_token, decode_access_token, decode_refresh_token,
    hash_password, hash_token, verify_password,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["auth"])

ACCESS_COOKIE = "access_token"
REFRESH_COOKIE = "refresh_token"
CSRF_COOKIE = "csrf_token"
CSRF_HEADER = "x-csrf-token"

# The refresh cookie is scoped to the one route that consumes it, so it is not
# attached to every ordinary request. A long-lived credential should travel as
# rarely as possible.
REFRESH_PATH = "/auth/refresh"


def _set_access_cookies(response: Response, user_id: int) -> None:
    """Set the short-lived access token and its matching CSRF token."""
    max_age = settings.access_token_expire_minutes * 60
    response.set_cookie(
        ACCESS_COOKIE, create_access_token(str(user_id)), max_age=max_age, httponly=True,
        samesite=settings.cookie_samesite, secure=settings.cookie_secure,
    )
    # The CSRF cookie outlives the access token deliberately: after a silent
    # refresh the page still holds the value it needs to echo back, so an
    # in-flight session never fails a CSRF check for want of a fresh cookie.
    response.set_cookie(
        CSRF_COOKIE, secrets.token_urlsafe(16),
        max_age=settings.refresh_token_expire_days * 86400, httponly=False,
        samesite=settings.cookie_samesite, secure=settings.cookie_secure,
    )


def _set_auth_cookies(response: Response, user_id: int) -> None:
    """Establish a full session: access + CSRF + refresh."""
    _set_access_cookies(response, user_id)
    response.set_cookie(
        REFRESH_COOKIE, create_refresh_token(str(user_id)),
        max_age=settings.refresh_token_expire_days * 86400, httponly=True,
        samesite=settings.cookie_samesite, secure=settings.cookie_secure,
        path=REFRESH_PATH,
    )


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(ACCESS_COOKIE, samesite=settings.cookie_samesite, secure=settings.cookie_secure)
    response.delete_cookie(CSRF_COOKIE, samesite=settings.cookie_samesite, secure=settings.cookie_secure)
    # Must match the path it was set with, or the browser keeps it.
    response.delete_cookie(REFRESH_COOKIE, path=REFRESH_PATH,
                           samesite=settings.cookie_samesite, secure=settings.cookie_secure)


def deliver_reset_token(email: str, token: str) -> None:
    """Get a reset token to its owner.

    NOT IMPLEMENTED — and that is stated rather than faked. There is no mail
    provider configured, so the token is written to the server log, which is
    workable for local development and completely unacceptable in production:
    anyone who can read the logs can take over any account.

    This function is the seam. Wiring a provider means replacing this body and
    nothing else. Until it is replaced, password reset must not be exposed to
    real users — the route works, the delivery does not.
    """
    log.warning(
        "PASSWORD RESET for %s — no mail provider configured, token logged instead: %s",
        email, token,
    )


def get_current_user(
    access_token: str | None = Cookie(default=None),
    session: Session = Depends(get_session),
) -> User:
    """Dependency: resolve the logged-in user from the access-token cookie."""
    if not access_token:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "not authenticated")
    user_id = decode_access_token(access_token)
    if user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired token")
    user = session.get(User, int(user_id))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user no longer exists")
    return user


def require_csrf(
    csrf_token: str | None = Cookie(default=None),
    x_csrf_token: str | None = Header(default=None),
) -> None:
    """Dependency for unsafe routes: header must match the CSRF cookie."""
    if not csrf_token or not x_csrf_token or not secrets.compare_digest(csrf_token, x_csrf_token):
        raise HTTPException(status.HTTP_403_FORBIDDEN, "CSRF check failed")


@router.post("/register", response_model=UserPublic, status_code=status.HTTP_201_CREATED)
def register(body: RegisterRequest, response: Response, session: Session = Depends(get_session)) -> User:
    if session.exec(select(User).where(User.email == body.email)).first():
        raise HTTPException(status.HTTP_409_CONFLICT, "email already registered")
    user = User(email=body.email, hashed_password=hash_password(body.password))
    session.add(user)
    session.commit()
    session.refresh(user)
    _set_auth_cookies(response, user.id)  # log them straight in
    return user


@router.post("/login", response_model=UserPublic)
def login(body: LoginRequest, response: Response, session: Session = Depends(get_session)) -> User:
    user = session.exec(select(User).where(User.email == body.email)).first()
    # Verify even on unknown email path would be ideal; keep simple but generic msg.
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")
    _set_auth_cookies(response, user.id)
    return user


@router.post("/logout", dependencies=[Depends(require_csrf)])
def logout(response: Response, _user: User = Depends(get_current_user)) -> dict[str, str]:
    _clear_auth_cookies(response)
    return {"status": "logged out"}


@router.get("/me", response_model=UserPublic)
def me(user: User = Depends(get_current_user)) -> User:
    return user


@router.post("/refresh", response_model=UserPublic, dependencies=[Depends(require_csrf)])
def refresh(
    response: Response,
    refresh_token: str | None = Cookie(default=None),
    session: Session = Depends(get_session),
) -> User:
    """Exchange a valid refresh token for a fresh access token.

    Deliberately does NOT depend on get_current_user — the whole point is to work
    when the access token has already expired. CSRF still applies: this mints a
    credential, so another site must not be able to trigger it.

    The refresh token is re-issued too, so an active session slides forward
    rather than hitting a hard 30-day wall mid-use.
    """
    user_id = decode_refresh_token(refresh_token) if refresh_token else None
    if user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired refresh token")
    user = session.get(User, int(user_id))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user no longer exists")
    _set_auth_cookies(response, user.id)
    return user


@router.post("/forgot", status_code=status.HTTP_202_ACCEPTED)
def forgot_password(
    body: ForgotPasswordRequest,
    session: Session = Depends(get_session),
) -> dict[str, str]:
    """Begin a password reset. Always 202, whether or not the account exists.

    Answering differently for a known and an unknown address turns this endpoint
    into an account-existence oracle: anyone could test whether a given person
    has an account here. The cost of that uniformity is that a typo'd address
    looks identical to a successful request, which is the right trade.
    """
    user = session.exec(select(User).where(User.email == body.email)).first()
    if user is not None:
        token = secrets.token_urlsafe(32)
        session.add(PasswordResetToken(
            user_id=user.id,
            token_hash=hash_token(token),
            expires_at=datetime.now(timezone.utc)
            + timedelta(minutes=settings.reset_token_expire_minutes),
        ))
        session.commit()
        deliver_reset_token(user.email, token)
    return {"status": "if that account exists, a reset link has been sent"}


@router.post("/reset", response_model=UserPublic)
def reset_password(
    body: ResetPasswordRequest,
    response: Response,
    session: Session = Depends(get_session),
) -> User:
    """Consume a reset token and set a new password.

    Single use and time limited. The token is looked up BY ITS HASH, so the
    plaintext never has to be compared against anything stored — a database leak
    yields hashes that cannot be replayed as links.
    """
    record = session.exec(
        select(PasswordResetToken).where(PasswordResetToken.token_hash == hash_token(body.token))
    ).first()

    now = datetime.now(timezone.utc)
    # `expires_at` comes back from SQLite without a timezone; compare in UTC.
    expired = record is not None and record.expires_at.replace(tzinfo=timezone.utc) < now
    if record is None or record.used_at is not None or expired:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid or expired reset token")

    user = session.get(User, record.user_id)
    if user is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid or expired reset token")

    user.hashed_password = hash_password(body.password)
    record.used_at = now
    # Every other outstanding reset for this account is burned too: if a token
    # was requested because the account may be compromised, leaving the rest
    # live would defeat the point.
    for other in session.exec(
        select(PasswordResetToken).where(
            PasswordResetToken.user_id == user.id, PasswordResetToken.used_at.is_(None)
        )
    ).all():
        other.used_at = now

    session.add(user)
    session.commit()
    session.refresh(user)
    _set_auth_cookies(response, user.id)  # log them straight in
    return user
