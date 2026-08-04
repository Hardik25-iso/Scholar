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

from backend import mailer
from backend.config import settings
from backend.db import get_session
from backend.db_models import PasswordResetToken, RefreshToken, User
from backend.models import (
    ForgotPasswordRequest, LoginRequest, RegisterRequest, ResetPasswordRequest, UserPublic,
)
from backend.security import (
    access_session_id, create_access_token, create_refresh_token, decode_access_token,
    decode_refresh_token, hash_password, hash_token, verify_password,
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


def _set_access_cookies(response: Response, user_id: int, session_id: int | None = None) -> None:
    """Set the short-lived access token and its matching CSRF token."""
    max_age = settings.access_token_expire_minutes * 60
    response.set_cookie(
        ACCESS_COOKIE, create_access_token(str(user_id), session_id), max_age=max_age,
        httponly=True, samesite=settings.cookie_samesite, secure=settings.cookie_secure,
    )
    # The CSRF cookie outlives the access token deliberately: after a silent
    # refresh the page still holds the value it needs to echo back, so an
    # in-flight session never fails a CSRF check for want of a fresh cookie.
    response.set_cookie(
        CSRF_COOKIE, secrets.token_urlsafe(16),
        max_age=settings.refresh_token_expire_days * 86400, httponly=False,
        samesite=settings.cookie_samesite, secure=settings.cookie_secure,
    )


def _issue_refresh_token(
    session: Session, user_id: int, replaces: RefreshToken | None = None,
) -> tuple[str, int]:
    """Mint a refresh token AND the server-side record that gives it power.

    The record is what makes the token revocable: a JWT is valid until `exp` no
    matter what, so without a row to invalidate there is no such thing as
    ending a session early. Returns the token and its record id, which the
    access token carries as `sid`.
    """
    token = create_refresh_token(str(user_id))
    record = RefreshToken(
        user_id=user_id,
        token_hash=hash_token(token),
        expires_at=datetime.now(timezone.utc)
        + timedelta(days=settings.refresh_token_expire_days),
    )
    session.add(record)
    session.commit()
    session.refresh(record)
    if replaces is not None:
        replaces.replaced_by_id = record.id
        session.add(replaces)
        session.commit()
    return token, record.id


def _set_auth_cookies(
    response: Response, user_id: int, session: Session, replaces: RefreshToken | None = None,
) -> None:
    """Establish a full session: access + CSRF + refresh.

    The refresh token is issued first because the access token names it — that
    link is what lets logout end this session and only this one.
    """
    token, session_id = _issue_refresh_token(session, user_id, replaces)
    _set_access_cookies(response, user_id, session_id)
    response.set_cookie(
        REFRESH_COOKIE, token,
        max_age=settings.refresh_token_expire_days * 86400, httponly=True,
        samesite=settings.cookie_samesite, secure=settings.cookie_secure,
        path=REFRESH_PATH,
    )


def _live_refresh_token(session: Session, token: str) -> RefreshToken | None:
    """The stored record for a presented token, if it is still live.

    Returns None for unknown, revoked or expired — the caller cannot act
    differently on those anyway, except for the one case it checks itself.
    """
    record = session.exec(
        select(RefreshToken).where(RefreshToken.token_hash == hash_token(token))
    ).first()
    if record is None or record.revoked_at is not None:
        return None
    # SQLite returns naive datetimes; compare in UTC.
    if record.expires_at.replace(tzinfo=timezone.utc) < datetime.now(timezone.utc):
        return None
    return record


def _revoke_all_refresh_tokens(session: Session, user_id: int) -> int:
    """End every session this account has. Returns how many were live."""
    now = datetime.now(timezone.utc)
    live = session.exec(
        select(RefreshToken).where(
            RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None)
        )
    ).all()
    for record in live:
        record.revoked_at = now
        session.add(record)
    session.commit()
    return len(live)


def _clear_auth_cookies(response: Response) -> None:
    response.delete_cookie(ACCESS_COOKIE, samesite=settings.cookie_samesite, secure=settings.cookie_secure)
    response.delete_cookie(CSRF_COOKIE, samesite=settings.cookie_samesite, secure=settings.cookie_secure)
    # Must match the path it was set with, or the browser keeps it.
    response.delete_cookie(REFRESH_COOKIE, path=REFRESH_PATH,
                           samesite=settings.cookie_samesite, secure=settings.cookie_secure)


RESET_SUBJECT = "Reset your Scholar password"

RESET_BODY = """\
Someone asked to reset the Scholar password for {email}.

Open this link to choose a new one:

{url}

The link works once and expires in {minutes} minutes. If you did not ask for
this, you can ignore this message — your password has not changed.
"""


def deliver_reset_token(email: str, token: str) -> bool:
    """Get a reset token to its owner. Returns whether mail actually went out.

    With SMTP configured this sends a one-time link. With no provider it falls
    back to logging the token, which is workable on a laptop and an
    account-takeover vector on a server: anyone who can read the logs can take
    over any account. The fallback therefore says so at WARNING level every
    time, so an unconfigured production deploy is noisy rather than silent.
    """
    if mailer.configured():
        url = mailer.link("/reset", token)
        body = RESET_BODY.format(
            email=email, url=url, minutes=settings.reset_token_expire_minutes
        )
        if mailer.send(email, RESET_SUBJECT, body):
            return True
        # Delivery was configured and still failed. Do NOT fall through to
        # logging the token — configuring mail is the operator saying tokens
        # must not appear in logs, and a transient SMTP outage does not revoke
        # that. The user retries; the error is already logged by mailer.send.
        return False

    log.warning(
        "PASSWORD RESET for %s — no mail provider configured, token logged instead: %s",
        email, token,
    )
    return False


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
    # Personal libraries are ordinary workspaces, so every account has one from
    # the start — there is no "user with no workspace" state for routes to
    # handle. Imported here rather than at module scope because workspaces.py
    # imports get_current_user from this module.
    from backend.workspaces import ensure_personal_workspace

    ensure_personal_workspace(session, user)
    session.refresh(user)
    _set_auth_cookies(response, user.id, session)  # log them straight in
    return user


@router.post("/login", response_model=UserPublic)
def login(body: LoginRequest, response: Response, session: Session = Depends(get_session)) -> User:
    user = session.exec(select(User).where(User.email == body.email)).first()
    # Verify even on unknown email path would be ideal; keep simple but generic msg.
    if user is None or not verify_password(body.password, user.hashed_password):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid email or password")
    _set_auth_cookies(response, user.id, session)
    return user


@router.post("/logout", dependencies=[Depends(require_csrf)])
def logout(
    response: Response,
    access_token: str | None = Cookie(default=None),
    _user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> dict[str, str]:
    """End this session — on the server, not only in the browser.

    Clearing the cookie used to be the whole of logging out, which meant a
    refresh token captured beforehand kept working for its full 30 days. Now the
    record is revoked, so the token is dead the moment this returns.

    Identified through the access token's `sid` rather than the refresh cookie,
    which is scoped to /auth/refresh and so never arrives here. Only THIS
    session is revoked: logging out on a laptop should not sign you out on
    your phone.
    """
    session_id = access_session_id(access_token) if access_token else None
    if session_id is not None:
        record = session.get(RefreshToken, session_id)
        if record is not None and record.revoked_at is None:
            record.revoked_at = datetime.now(timezone.utc)
            session.add(record)
            session.commit()
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
    """Exchange a valid refresh token for a fresh access token, and ROTATE it.

    Deliberately does NOT depend on get_current_user — the whole point is to work
    when the access token has already expired. CSRF still applies: this mints a
    credential, so another site must not be able to trigger it.

    Rotation: the presented token is spent and a new one issued, so a refresh
    token is a one-use credential rather than a 30-day bearer key.

    Reuse detection is what rotation buys. A spent token being presented again
    means two parties hold it — the legitimate client and whoever copied it.
    Nothing here can tell which is which, so BOTH lose: every session for the
    account is revoked and someone has to log in again. That is a deliberately
    blunt response, and the right one, because the alternative is leaving a
    thief with a month of access nobody can see.
    """
    user_id = decode_refresh_token(refresh_token) if refresh_token else None
    if user_id is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired refresh token")

    record = _live_refresh_token(session, refresh_token)
    if record is None:
        # The signature was good, so this token WAS issued by us. Whether that
        # is theft depends on WHY it is dead, and `replaced_by_id` is the only
        # thing that distinguishes the two:
        #
        #   rotated (has a successor) -> it was already exchanged once, so a
        #       second presentation means two parties hold it. Theft.
        #   revoked by logout, a reset, or an earlier detection (no successor)
        #       -> just a stale client replaying a token that was cancelled.
        #       A background tab doing this must not sign the user out
        #       everywhere, which would hand anyone holding one dead token a
        #       trivial way to end all of an account's sessions at will.
        spent = session.exec(
            select(RefreshToken).where(RefreshToken.token_hash == hash_token(refresh_token))
        ).first()
        if spent is not None and spent.replaced_by_id is not None:
            revoked = _revoke_all_refresh_tokens(session, spent.user_id)
            log.warning(
                "refresh token reuse for user %s — revoked %d live session(s)",
                spent.user_id, revoked,
            )
        _clear_auth_cookies(response)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid or expired refresh token")

    user = session.get(User, int(user_id))
    if user is None:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "user no longer exists")

    record.revoked_at = datetime.now(timezone.utc)  # spent by this refresh
    session.add(record)
    session.commit()
    _set_auth_cookies(response, user.id, session, replaces=record)
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

    # Every existing session ends too. A reset is usually done BECAUSE the
    # account may be compromised, and until refresh tokens were revocable this
    # was impossible: changing the password left whoever had a stolen refresh
    # token with up to 30 more days of access. Burning the reset links without
    # burning the sessions only ever solved the smaller half of the problem.
    _revoke_all_refresh_tokens(session, user.id)

    _set_auth_cookies(response, user.id, session)  # log them straight in
    return user
