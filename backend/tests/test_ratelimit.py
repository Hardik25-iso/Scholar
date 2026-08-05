"""Per-user budgets on the expensive routes.

These are cost controls, not security boundaries — see the honest limitation in
backend/ratelimit.py about the counter being per-process.
"""
import pytest
from fastapi.testclient import TestClient

from backend.ratelimit import RateLimiter, ask_limiter, upload_limiter
from backend.tests.conftest import csrf


@pytest.fixture(autouse=True)
def _clean_limiters():
    """Counters are process-global, so they leak between tests unless reset."""
    ask_limiter.reset()
    upload_limiter.reset()
    yield
    ask_limiter.reset()
    upload_limiter.reset()


# ——— the limiter itself ———


def test_requests_are_allowed_up_to_the_limit():
    limiter = RateLimiter(3, "things")
    for _ in range(3):
        limiter.check("user-1")
    assert limiter.remaining("user-1") == 0


def test_exceeding_the_limit_raises_429_with_retry_after():
    from fastapi import HTTPException

    limiter = RateLimiter(1, "things")
    limiter.check("user-1")
    with pytest.raises(HTTPException) as exc:
        limiter.check("user-1")
    assert exc.value.status_code == 429
    assert int(exc.value.headers["Retry-After"]) > 0
    assert "things" in exc.value.detail


def test_budgets_are_per_user():
    """One user's loop must not lock everyone else out."""
    limiter = RateLimiter(1, "things")
    limiter.check("user-1")
    limiter.check("user-2")  # must not raise


def test_a_zero_limit_disables_the_check():
    limiter = RateLimiter(0, "things")
    for _ in range(50):
        limiter.check("user-1")


def test_the_window_expires(monkeypatch):
    import backend.ratelimit as module

    clock = {"now": 1000.0}
    monkeypatch.setattr(module.time, "monotonic", lambda: clock["now"])

    limiter = RateLimiter(1, "things")
    limiter.check("user-1")
    clock["now"] += module.WINDOW_SECONDS + 1
    limiter.check("user-1")  # new window, must not raise


def test_the_limiter_is_thread_safe():
    """FastAPI runs sync endpoints in a threadpool, so this is concurrent in
    practice — an unguarded counter would over-admit under load."""
    import threading

    from fastapi import HTTPException

    limiter = RateLimiter(100, "things")
    rejected = []

    def hammer():
        for _ in range(50):
            try:
                limiter.check("shared")
            except HTTPException:
                rejected.append(1)

    threads = [threading.Thread(target=hammer) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(rejected) == 200 - 100, "admitted a different number than the limit allows"


# ——— wired into the routes ———


def test_asking_past_the_limit_returns_429(alice: TestClient, monkeypatch):
    monkeypatch.setattr(ask_limiter, "limit", 2)
    for _ in range(2):
        # 400 (no documents) still consumes budget — the work of finding that
        # out is cheap, but an unauthenticated-looking loop should still be
        # bounded.
        assert alice.post("/ask", json={"question": "hi"}, headers=csrf(alice)).status_code == 400
    r = alice.post("/ask", json={"question": "hi"}, headers=csrf(alice))
    assert r.status_code == 429
    assert "Retry-After" in r.headers


def test_uploading_past_the_limit_returns_429(alice: TestClient, monkeypatch):
    monkeypatch.setattr(upload_limiter, "limit", 1)
    files = {"file": ("a.pdf", b"not a pdf", "application/pdf")}
    # 202: the upload is accepted and the indexing job then fails on content.
    # The budget is charged for the upload, not for the outcome.
    assert alice.post("/papers", headers=csrf(alice), files=files).status_code == 202
    assert alice.post("/papers", headers=csrf(alice), files=files).status_code == 429


def test_the_upload_limit_is_charged_before_the_body_is_read(alice: TestClient, monkeypatch):
    """A rate-limited caller should not get to make the server buffer 20 MB."""
    monkeypatch.setattr(upload_limiter, "limit", 0 - 1)  # negative disables
    monkeypatch.setattr(upload_limiter, "limit", 1)
    files = {"file": ("a.pdf", b"x", "application/pdf")}
    alice.post("/papers", headers=csrf(alice), files=files)

    huge = {"file": ("big.pdf", b"x" * (20 * 1024 * 1024 + 1), "application/pdf")}
    r = alice.post("/papers", headers=csrf(alice), files=huge)
    assert r.status_code == 429, "size check ran before the rate limit"


def test_one_users_budget_does_not_affect_another(
    alice: TestClient, other_client: TestClient, monkeypatch
):
    monkeypatch.setattr(ask_limiter, "limit", 1)
    alice.post("/ask", json={"question": "hi"}, headers=csrf(alice))
    assert alice.post("/ask", json={"question": "hi"}, headers=csrf(alice)).status_code == 429

    other_client.post("/auth/register",
                      json={"email": "mallory@example.com", "password": "validpassword123"})
    r = other_client.post("/ask", json={"question": "hi"}, headers=csrf(other_client))
    assert r.status_code == 400, "second user was blocked by the first user's budget"
