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
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_health_reports_each_dependency(client: TestClient):
    """/health is a READINESS check: it names what it verified.

    A check that only proves the process is alive reports green while every
    question fails, so it must exercise the things an answer depends on.
    """
    checks = client.get("/health").json()["checks"]
    assert set(checks) == {"database", "embedder", "llm"}
    assert all(v == "ok" for v in checks.values()), checks


def test_health_is_503_when_a_dependency_is_down(client: TestClient, monkeypatch):
    """Down dependency => 503, so a load balancer stops sending traffic here."""
    from backend import generator

    def _dead():
        raise generator.LLMUnavailable("Ollama is not reachable")

    monkeypatch.setattr(generator, "ping_llm", _dead)
    r = client.get("/health")
    assert r.status_code == 503
    assert r.json()["status"] == "degraded"
    assert "not reachable" in r.json()["checks"]["llm"]


def test_security_headers_are_set(client: TestClient):
    """Baseline browser defences ride on every response, including errors."""
    h = client.get("/health").headers
    assert h["X-Content-Type-Options"] == "nosniff"
    assert h["X-Frame-Options"] == "DENY"
    assert "frame-ancestors 'none'" in h["Content-Security-Policy"]


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


# ——— brute-force protection ———


def test_repeated_bad_logins_are_eventually_throttled(client: TestClient):
    """Without a cap, /login is an unlimited password oracle.

    Asserts the wrong password keeps returning 401 up to the budget and then
    flips to 429 — i.e. guessing is bounded, not merely discouraged.
    """
    from backend.ratelimit import auth_limiter

    body = {"email": "nobody@example.com", "password": "wrongpassword123"}
    for _ in range(auth_limiter.limit):
        assert client.post("/auth/login", json=body).status_code == 401

    blocked = client.post("/auth/login", json=body)
    assert blocked.status_code == 429
    # Tell the caller when to come back rather than just refusing.
    assert "Retry-After" in blocked.headers


def test_throttle_counts_attempts_not_just_failures(client: TestClient):
    """The budget is charged before the password check, so a valid password
    cannot be used to reset or dodge the counter."""
    from backend.ratelimit import auth_limiter

    client.post("/auth/register", json={"email": "bob@example.com", "password": "validpassword123"})
    for _ in range(auth_limiter.limit):
        client.post("/auth/login", json={"email": "bob@example.com", "password": "wrongpassword123"})

    good = client.post("/auth/login", json={"email": "bob@example.com", "password": "validpassword123"})
    assert good.status_code == 429


def test_registration_is_throttled(client: TestClient):
    """Stops automated mass-signup and email enumeration from one source."""
    from backend.ratelimit import auth_limiter

    for i in range(auth_limiter.limit):
        client.post("/auth/register", json={"email": f"u{i}@example.com", "password": "validpassword123"})
    r = client.post("/auth/register", json={"email": "toomany@example.com", "password": "validpassword123"})
    assert r.status_code == 429


def test_password_reset_request_is_throttled(client: TestClient):
    """Unthrottled, /forgot is a free email cannon aimed at any address."""
    from backend.ratelimit import auth_limiter

    for _ in range(auth_limiter.limit):
        client.post("/auth/forgot", json={"email": "victim@example.com"})
    assert client.post("/auth/forgot", json={"email": "victim@example.com"}).status_code == 429
