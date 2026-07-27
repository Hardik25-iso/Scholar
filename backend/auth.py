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
import secrets

from fastapi import APIRouter, Cookie, Depends, Header, HTTPException, Response, status
from sqlmodel import Session, select

from backend.config import settings
from backend.db import get_session
from backend.db_models import User
from backend.models import LoginRequest, RegisterRequest, UserPublic
from backend.security import create_access_token, decode_access_token, hash_password, verify_password

router = APIRouter(prefix="/auth", tags=["auth"])

ACCESS_COOKIE = "access_token"
CSRF_COOKIE = "csrf_token"
CSRF_HEADER = "x-csrf-token"


def _set_auth_cookies(response: Response, user_id: int) -> None:
    """Set the httpOnly access token + the readable CSRF token together."""
    max_age = settings.access_token_expire_minutes * 60
    token = create_access_token(str(user_id))
    response.set_cookie(
        ACCESS_COOKIE, token, max_age=max_age, httponly=True,
        samesite=settings.cookie_samesite, secure=settings.cookie_secure,
    )
    response.set_cookie(
        CSRF_COOKIE, secrets.token_urlsafe(16), max_age=max_age, httponly=False,
        samesite=settings.cookie_samesite, secure=settings.cookie_secure,
    )


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(ACCESS_COOKIE, samesite=settings.cookie_samesite, secure=settings.cookie_secure)
    response.delete_cookie(CSRF_COOKIE, samesite=settings.cookie_samesite, secure=settings.cookie_secure)


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
