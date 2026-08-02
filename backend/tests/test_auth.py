"""Auth boundary: registration rules, login, session, CSRF, anonymous access.

These are the assertions from the manual end-to-end run, promoted into a suite
that needs no running server, no Ollama and no models.
"""
import pytest
from fastapi.testclient import TestClient

from backend.tests.conftest import csrf

GOOD = {"email": "alice@example.com", "password": "validpassword123"}


# ——— registration validation ———


def test_short_password_rejected(client: TestClient):
    r = client.post("/auth/register", json={"email": "a@example.com", "password": "short"})
    assert r.status_code == 422


def test_malformed_email_rejected(client: TestClient):
    r = client.post("/auth/register", json={"email": "notanemail", "password": "validpassword123"})
    assert r.status_code == 422


def test_password_over_72_bytes_rejected(client: TestClient):
    """bcrypt silently truncates past 72 BYTES, so two long passwords could hash
    equal. The model layer must reject rather than quietly cut."""
    r = client.post("/auth/register", json={"email": "a@example.com", "password": "x" * 100})
    assert r.status_code == 422


def test_multibyte_password_measured_in_bytes_not_chars(client: TestClient):
    """30 four-byte characters is 120 bytes — well past bcrypt's limit, even
    though len() in characters looks harmless."""
    r = client.post("/auth/register", json={"email": "a@example.com", "password": "\U0001f600" * 30})
    assert r.status_code == 422


# ——— registration + login ———


def test_register_succeeds_and_never_echoes_the_password(client: TestClient):
    r = client.post("/auth/register", json=GOOD)
    assert r.status_code == 201
    assert r.json()["email"] == GOOD["email"]
    assert "password" not in r.text.lower()
    assert "hashed" not in r.text.lower()


def test_register_sets_both_auth_cookies(client: TestClient):
    client.post("/auth/register", json=GOOD)
    assert client.cookies.get("access_token")
    assert client.cookies.get("csrf_token")


def test_duplicate_email_conflicts(client: TestClient):
    client.post("/auth/register", json=GOOD)
    r = client.post("/auth/register", json=GOOD)
    assert r.status_code == 409


def test_wrong_password_unauthorized(alice: TestClient):
    r = alice.post("/auth/login", json={"email": GOOD["email"], "password": "wrongpassword"})
    assert r.status_code == 401


def test_unknown_email_gives_the_same_error_as_a_wrong_password(client: TestClient):
    """The message must not reveal whether the account exists."""
    r = client.post("/auth/login", json={"email": "nobody@example.com", "password": "validpassword123"})
    assert r.status_code == 401
    assert r.json()["detail"] == "invalid email or password"


def test_session_restored_from_cookie(alice: TestClient):
    r = alice.get("/auth/me")
    assert r.status_code == 200
    assert r.json()["email"] == GOOD["email"]


# ——— anonymous access is refused everywhere ———


def test_anonymous_cannot_read_the_library(client: TestClient):
    assert client.get("/papers").status_code == 401


@pytest.mark.parametrize(
    "method,path",
    [
        ("post", "/ask"),
        ("post", "/ask/stream"),
        ("post", "/papers"),
        ("delete", "/papers/1"),
        ("post", "/auth/logout"),
    ],
)
def test_anonymous_is_refused_on_every_unsafe_route(client: TestClient, method, path):
    """403, not 401: `dependencies=[Depends(require_csrf)]` is a route-level
    dependency, so the CSRF check runs before the endpoint resolves the user and
    a cookie-less caller trips it first. That is how every unsafe route in the
    app already behaves — the point of this test is that anonymous callers are
    refused, and refused identically."""
    assert getattr(client, method)(path).status_code == 403


@pytest.mark.parametrize("path", ["/ask", "/ask/stream"])
def test_a_valid_csrf_pair_alone_does_not_authenticate(client: TestClient, path):
    """Anyone can mint a matching cookie/header pair — CSRF proves same-origin,
    not identity. Past the CSRF gate the caller must still be 401."""
    client.cookies.set("csrf_token", "self-issued-value")
    r = client.post(path, json={"question": "hi"}, headers={"X-CSRF-Token": "self-issued-value"})
    assert r.status_code == 401


def test_health_needs_no_auth(client: TestClient):
    assert client.get("/health").json() == {"status": "ok"}


# ——— CSRF double-submit ———


def test_logout_without_csrf_header_is_forbidden(alice: TestClient):
    assert alice.post("/auth/logout").status_code == 403


def test_logout_with_mismatched_csrf_header_is_forbidden(alice: TestClient):
    r = alice.post("/auth/logout", headers={"X-CSRF-Token": "not-the-cookie-value"})
    assert r.status_code == 403


def test_logout_with_matching_csrf_header_succeeds(alice: TestClient):
    r = alice.post("/auth/logout", headers=csrf(alice))
    assert r.status_code == 200


def test_invalid_token_is_rejected(client: TestClient):
    client.cookies.set("access_token", "not.a.real.jwt")
    assert client.get("/auth/me").status_code == 401


def test_token_signed_with_a_different_secret_is_rejected(client: TestClient):
    """A forged token must fail the signature check, not just the format check."""
    import jwt

    forged = jwt.encode({"sub": "1"}, "attacker-secret", algorithm="HS256")
    client.cookies.set("access_token", forged)
    assert client.get("/auth/me").status_code == 401
