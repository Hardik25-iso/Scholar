"""Session lifetime and account recovery.

Two things that made the product unusable rather than merely unpolished: a
30-minute hard logout with no way back, and a forgotten password meaning
permanent, unrecoverable lockout.
"""
from datetime import datetime, timedelta, timezone

import jwt
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from backend.config import settings
from backend.db import engine
from backend.db_models import PasswordResetToken, RefreshToken, User
from backend.security import ALGORITHM, create_access_token, create_refresh_token, hash_token
from backend.tests.conftest import csrf

GOOD = {"email": "alice@example.com", "password": "validpassword123"}


def force_cookie(client: TestClient, name: str, value: str) -> None:
    """Replace a cookie, removing every existing copy first.

    `client.cookies.set()` alone adds an entry with an empty domain ALONGSIDE
    the server-issued one, and the stale duplicate then shadows anything the
    server sets afterwards — which looks exactly like a broken endpoint.
    """
    for cookie in list(client.cookies.jar):
        if cookie.name == name:
            client.cookies.jar.clear(cookie.domain, cookie.path, cookie.name)
    client.cookies.set(name, value, domain="testserver.local", path="/")


def expired_access_token(subject: str = "1") -> str:
    return jwt.encode(
        {"sub": subject, "typ": "access", "exp": datetime.now(timezone.utc) - timedelta(seconds=1)},
        settings.secret_key, algorithm=ALGORITHM,
    )


# ——— token kinds are not interchangeable ———


def test_a_refresh_token_is_not_accepted_as_an_access_token(client: TestClient):
    """Both are signed with the same key, so without a `typ` claim a long-lived
    refresh token would work as an access token for its entire lifetime —
    silently undoing the short access expiry."""
    client.post("/auth/register", json=GOOD)
    client.cookies.set("access_token", create_refresh_token("1"))
    assert client.get("/auth/me").status_code == 401


def test_an_access_token_is_not_accepted_as_a_refresh_token(alice: TestClient):
    alice.cookies.set("refresh_token", create_access_token("1"))
    assert alice.post("/auth/refresh", headers=csrf(alice)).status_code == 401


def test_a_token_with_no_type_claim_is_refused(client: TestClient):
    """Tokens minted before `typ` existed fail closed — those sessions need a
    fresh login, which is the safe direction."""
    client.post("/auth/register", json=GOOD)
    legacy = jwt.encode(
        {"sub": "1", "exp": datetime.now(timezone.utc) + timedelta(minutes=30)},
        settings.secret_key, algorithm=ALGORITHM,
    )
    client.cookies.set("access_token", legacy)
    assert client.get("/auth/me").status_code == 401


# ——— refresh ———


def test_registering_issues_a_refresh_cookie(client: TestClient):
    client.post("/auth/register", json=GOOD)
    assert client.cookies.get("refresh_token")


def test_refresh_works_after_the_access_token_expires(alice: TestClient):
    """The actual user-facing point: a session should not die mid-sentence."""
    force_cookie(alice, "access_token", expired_access_token())
    assert alice.get("/auth/me").status_code == 401

    assert alice.post("/auth/refresh", headers=csrf(alice)).status_code == 200
    assert alice.get("/auth/me").status_code == 200, "refreshed session still rejected"


def test_refreshing_reissues_the_refresh_cookie(alice: TestClient):
    """An active session should not hit a hard 30-day wall mid-use, so the
    refresh cookie is re-set with a fresh Max-Age on every refresh.

    Asserted on the Set-Cookie header rather than on the token value, which is
    covered separately by the rotation tests below. (Refresh tokens now carry a
    random `jti`, so a new one is a genuinely different string even when issued
    inside the same second as the last.)
    """
    r = alice.post("/auth/refresh", headers=csrf(alice))
    assert r.status_code == 200
    set_cookies = r.headers.get_list("set-cookie")
    refresh_header = next((h for h in set_cookies if h.startswith("refresh_token=")), None)
    assert refresh_header is not None, "refresh cookie was not re-issued"
    assert "Path=/auth/refresh" in refresh_header
    assert "HttpOnly" in refresh_header


def test_refresh_requires_csrf(alice: TestClient):
    """It mints a credential, so another site must not be able to trigger it."""
    assert alice.post("/auth/refresh").status_code == 403


def test_refresh_without_a_token_is_401(client: TestClient):
    client.cookies.set("csrf_token", "v")
    assert client.post("/auth/refresh", headers={"X-CSRF-Token": "v"}).status_code == 401


def test_refresh_rejects_a_token_signed_with_another_key(alice: TestClient):
    forged = jwt.encode({"sub": "1", "typ": "refresh",
                         "exp": datetime.now(timezone.utc) + timedelta(days=1)},
                        "attacker-secret", algorithm=ALGORITHM)
    alice.cookies.set("refresh_token", forged)
    assert alice.post("/auth/refresh", headers=csrf(alice)).status_code == 401


def test_logout_clears_the_refresh_cookie_too(alice: TestClient):
    """Cleared with the same path it was set with, or the browser keeps it."""
    alice.post("/auth/logout", headers=csrf(alice))
    assert not alice.cookies.get("refresh_token")


# ——— rotation and revocation ———
#
# A JWT cannot be revoked: its signature stays valid until `exp`, 30 days out.
# These tests are about the server-side record that takes that power back.


def live_tokens(user_id: int = 1) -> list[RefreshToken]:
    with Session(engine) as session:
        return list(session.exec(
            select(RefreshToken).where(
                RefreshToken.user_id == user_id, RefreshToken.revoked_at.is_(None)
            )
        ).all())


def test_two_refresh_tokens_issued_in_the_same_second_differ(alice: TestClient):
    """`exp` has second resolution, so without a random `jti` two issuances mint
    a byte-identical string — which would collide on the unique index rotation
    depends on, and make the new token indistinguishable from the spent one."""
    first = create_refresh_token("1")
    second = create_refresh_token("1")
    assert first != second


def test_refreshing_spends_the_old_token(alice: TestClient):
    old = alice.cookies.get("refresh_token")
    assert alice.post("/auth/refresh", headers=csrf(alice)).status_code == 200
    new = alice.cookies.get("refresh_token")
    assert new != old, "the token was re-issued unchanged — that is not rotation"

    force_cookie(alice, "refresh_token", old)
    assert alice.post("/auth/refresh", headers=csrf(alice)).status_code == 401


def test_reusing_a_spent_token_revokes_every_session(alice: TestClient, other_client: TestClient):
    """The point of rotation. A spent token presented again means two parties
    hold it; nothing can tell which is the owner, so both lose."""
    stolen = alice.cookies.get("refresh_token")

    # A second device for the same account — its own, separate live token.
    other_client.post("/auth/login", json=GOOD)
    assert len(live_tokens()) == 2

    assert alice.post("/auth/refresh", headers=csrf(alice)).status_code == 200

    force_cookie(alice, "refresh_token", stolen)
    assert alice.post("/auth/refresh", headers=csrf(alice)).status_code == 401
    assert live_tokens() == [], "reuse was detected but the other sessions survived"

    # The untouched second device is now locked out too — deliberately.
    assert other_client.post("/auth/refresh", headers=csrf(other_client)).status_code == 401


def test_a_valid_signature_with_no_stored_record_is_refused(alice: TestClient):
    """The signature alone must not be enough, or revocation means nothing."""
    force_cookie(alice, "refresh_token", create_refresh_token("1"))
    assert alice.post("/auth/refresh", headers=csrf(alice)).status_code == 401


def test_logging_out_kills_the_refresh_token_on_the_server(alice: TestClient):
    """Clearing the cookie used to be the whole of logging out, so a token
    captured beforehand kept working for its full 30 days."""
    captured = alice.cookies.get("refresh_token")
    alice.post("/auth/logout", headers=csrf(alice))

    force_cookie(alice, "refresh_token", captured)
    force_cookie(alice, "csrf_token", "v")
    assert alice.post("/auth/refresh", headers={"X-CSRF-Token": "v"}).status_code == 401


def test_logging_out_on_one_device_leaves_the_others_signed_in(
    alice: TestClient, other_client: TestClient
):
    other_client.post("/auth/login", json=GOOD)
    alice.post("/auth/logout", headers=csrf(alice))
    assert other_client.post("/auth/refresh", headers=csrf(other_client)).status_code == 200


def test_replaying_a_logged_out_token_is_not_treated_as_theft(
    alice: TestClient, other_client: TestClient
):
    """Found by exercising this over real HTTP, not by a unit test.

    Logout-revocation and rotation-revocation both set `revoked_at`, so treating
    any revoked token as stolen meant a stale background tab retrying its
    refresh would sign the user out on every OTHER device — and handed anyone
    holding one dead token a way to do that on demand. Only a token with a
    SUCCESSOR was actually exchanged twice.
    """
    captured = alice.cookies.get("refresh_token")
    other_client.post("/auth/login", json=GOOD)
    alice.post("/auth/logout", headers=csrf(alice))

    force_cookie(alice, "refresh_token", captured)
    force_cookie(alice, "csrf_token", "v")
    assert alice.post("/auth/refresh", headers={"X-CSRF-Token": "v"}).status_code == 401

    assert other_client.post("/auth/refresh", headers=csrf(other_client)).status_code == 200, (
        "a stale tab replaying a logged-out token signed the other device out"
    )


def test_resetting_the_password_ends_every_existing_session(
    alice: TestClient, other_client: TestClient, monkeypatch
):
    """A reset is usually done BECAUSE the account may be compromised. Burning
    the reset links without burning the sessions solved the smaller half."""
    captured = {}
    monkeypatch.setattr("backend.auth.deliver_reset_token",
                        lambda email, token: captured.update(token=token))
    other_client.post("/auth/login", json=GOOD)
    _issue_reset(alice)

    assert alice.post("/auth/reset", json={"token": captured["token"],
                                           "password": "brandnewpassword1"}).status_code == 200
    assert other_client.post("/auth/refresh", headers=csrf(other_client)).status_code == 401


def test_the_reset_survivor_is_the_session_that_did_the_resetting(
    alice: TestClient, monkeypatch
):
    """Revoking everything and then handing back a dead cookie would log the
    user out of the browser they just recovered their account in."""
    captured = {}
    monkeypatch.setattr("backend.auth.deliver_reset_token",
                        lambda email, token: captured.update(token=token))
    _issue_reset(alice)
    alice.post("/auth/reset", json={"token": captured["token"], "password": "brandnewpassword1"})

    assert alice.post("/auth/refresh", headers=csrf(alice)).status_code == 200


def test_only_a_hash_of_the_refresh_token_is_stored(alice: TestClient):
    token = alice.cookies.get("refresh_token")
    (record,) = live_tokens()
    assert token not in record.token_hash
    assert record.token_hash == hash_token(token)


def test_rotation_records_which_token_replaced_which(alice: TestClient):
    """The chain is what makes a reuse attributable to a point in the session's
    history rather than just 'something went wrong'."""
    alice.post("/auth/refresh", headers=csrf(alice))
    with Session(engine) as session:
        records = list(session.exec(select(RefreshToken).order_by(RefreshToken.id)).all())
    assert len(records) == 2
    assert records[0].revoked_at is not None
    assert records[0].replaced_by_id == records[1].id
    assert records[1].revoked_at is None


def test_an_expired_stored_token_is_refused(alice: TestClient):
    """Belt and braces: the JWT `exp` and the row's `expires_at` should agree,
    but the row is the one that governs."""
    token = alice.cookies.get("refresh_token")
    with Session(engine) as session:
        record = session.exec(
            select(RefreshToken).where(RefreshToken.token_hash == hash_token(token))
        ).first()
        record.expires_at = datetime.now(timezone.utc) - timedelta(seconds=1)
        session.add(record)
        session.commit()

    assert alice.post("/auth/refresh", headers=csrf(alice)).status_code == 401


# ——— password reset ———


def _issue_reset(client: TestClient, email: str = GOOD["email"]) -> str:
    """Trigger a reset and read the token back out of the database.

    Reading it from storage rather than from a response is deliberate: the API
    must never return the token, or anyone could reset anyone's password.
    """
    assert client.post("/auth/forgot", json={"email": email}).status_code == 202
    with Session(engine) as session:
        user = session.exec(select(User).where(User.email == email)).first()
        record = session.exec(
            select(PasswordResetToken)
            .where(PasswordResetToken.user_id == user.id)
            .order_by(PasswordResetToken.id.desc())
        ).first()
    return record


def test_forgot_never_reveals_whether_an_account_exists(alice: TestClient):
    """Answering differently would make this an account-existence oracle."""
    known = alice.post("/auth/forgot", json={"email": GOOD["email"]})
    unknown = alice.post("/auth/forgot", json={"email": "nobody@example.com"})
    assert known.status_code == unknown.status_code == 202
    assert known.json() == unknown.json()


def test_the_reset_token_is_never_returned_over_the_wire(alice: TestClient):
    r = alice.post("/auth/forgot", json={"email": GOOD["email"]})
    record = _issue_reset(alice)
    assert record.token_hash not in r.text


def test_only_a_hash_of_the_token_is_stored(alice: TestClient, monkeypatch):
    """A leaked database must not hand over working reset links."""
    captured = {}
    monkeypatch.setattr("backend.auth.deliver_reset_token",
                        lambda email, token: captured.update(token=token))
    record = _issue_reset(alice)
    assert captured["token"] not in record.token_hash
    assert record.token_hash == hash_token(captured["token"])


def test_a_valid_token_sets_a_new_password(alice: TestClient, monkeypatch):
    captured = {}
    monkeypatch.setattr("backend.auth.deliver_reset_token",
                        lambda email, token: captured.update(token=token))
    _issue_reset(alice)

    r = alice.post("/auth/reset", json={"token": captured["token"], "password": "brandnewpassword1"})
    assert r.status_code == 200, r.text

    alice.cookies.clear()
    assert alice.post("/auth/login", json={"email": GOOD["email"],
                                           "password": "brandnewpassword1"}).status_code == 200
    assert alice.post("/auth/login", json=GOOD).status_code == 401, "old password still works"


def test_a_reset_token_works_only_once(alice: TestClient, monkeypatch):
    captured = {}
    monkeypatch.setattr("backend.auth.deliver_reset_token",
                        lambda email, token: captured.update(token=token))
    _issue_reset(alice)

    body = {"token": captured["token"], "password": "brandnewpassword1"}
    assert alice.post("/auth/reset", json=body).status_code == 200
    assert alice.post("/auth/reset", json=body).status_code == 400


def test_using_one_token_burns_every_other_outstanding_one(alice: TestClient, monkeypatch):
    """If a reset was requested because the account may be compromised, leaving
    the other links live would defeat the point."""
    tokens = []
    monkeypatch.setattr("backend.auth.deliver_reset_token",
                        lambda email, token: tokens.append(token))
    _issue_reset(alice)
    _issue_reset(alice)
    assert len(tokens) == 2

    assert alice.post("/auth/reset", json={"token": tokens[1],
                                           "password": "brandnewpassword1"}).status_code == 200
    assert alice.post("/auth/reset", json={"token": tokens[0],
                                           "password": "another1password"}).status_code == 400


def test_an_expired_token_is_refused(alice: TestClient, monkeypatch):
    captured = {}
    monkeypatch.setattr("backend.auth.deliver_reset_token",
                        lambda email, token: captured.update(token=token))
    record = _issue_reset(alice)

    with Session(engine) as session:
        stored = session.get(PasswordResetToken, record.id)
        stored.expires_at = datetime.now(timezone.utc) - timedelta(minutes=1)
        session.add(stored)
        session.commit()

    assert alice.post("/auth/reset", json={"token": captured["token"],
                                           "password": "brandnewpassword1"}).status_code == 400


def test_an_unknown_token_is_refused(alice: TestClient):
    assert alice.post("/auth/reset", json={"token": "not-a-real-token",
                                           "password": "brandnewpassword1"}).status_code == 400


def test_the_new_password_must_meet_the_signup_rules(alice: TestClient, monkeypatch):
    """A reset must not be a way around the password policy."""
    captured = {}
    monkeypatch.setattr("backend.auth.deliver_reset_token",
                        lambda email, token: captured.update(token=token))
    _issue_reset(alice)
    assert alice.post("/auth/reset", json={"token": captured["token"],
                                           "password": "short"}).status_code == 422
    assert alice.post("/auth/reset", json={"token": captured["token"],
                                           "password": "x" * 100}).status_code == 422


def test_a_successful_reset_logs_the_user_in(alice: TestClient, monkeypatch):
    captured = {}
    monkeypatch.setattr("backend.auth.deliver_reset_token",
                        lambda email, token: captured.update(token=token))
    _issue_reset(alice)
    alice.cookies.clear()

    alice.post("/auth/reset", json={"token": captured["token"], "password": "brandnewpassword1"})
    assert alice.get("/auth/me").status_code == 200
