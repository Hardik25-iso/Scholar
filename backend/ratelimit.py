"""Per-user request budgets on the expensive routes.

WHAT THIS PROTECTS. /ask runs retrieval, reranking and a full LLM generation;
upload runs parsing, OCR and embedding. Both are orders of magnitude more
expensive than an ordinary request, and both become a *billing* control the
moment generation moves to a paid API. A single logged-in account with a loop
should not be able to spend an unbounded amount of someone else's money.

HONEST LIMITATION: this counter lives in process memory. With one worker it is
exact. With N workers each holds its own counter, so the real limit is N times
the configured one, and a restart forgets everything. That is a genuine
weakness, not a detail — a shared store (Redis) is the fix, and it is deferred
because it is a new dependency and a deployment decision. Until then the limit
is a guard against runaway loops and honest mistakes, NOT a defence against a
determined attacker. Sized generously for that purpose.

A fixed window rather than a sliding one: a user can burst up to 2x the limit
across a window boundary. For a "stop the runaway loop" guard that is fine, and
it costs one integer per user instead of a timestamp list.
"""
import threading
import time
from dataclasses import dataclass, field

from fastapi import HTTPException, Request, status

from backend.config import settings

WINDOW_SECONDS = 3600


@dataclass
class _Bucket:
    count: int = 0
    window_start: float = field(default_factory=time.monotonic)


class RateLimiter:
    """Counts requests per key per hour. Thread-safe: FastAPI runs sync
    endpoints in a threadpool, so several requests touch this concurrently."""

    def __init__(self, limit: int, name: str) -> None:
        self.limit = limit
        self.name = name
        self._buckets: dict[str, _Bucket] = {}
        self._lock = threading.Lock()

    def check(self, key: str) -> None:
        """Consume one unit for `key`, or raise 429 with a Retry-After."""
        if self.limit <= 0:  # 0 or negative disables the limit entirely
            return

        now = time.monotonic()
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None or now - bucket.window_start >= WINDOW_SECONDS:
                bucket = _Bucket(window_start=now)
                self._buckets[key] = bucket

            if bucket.count >= self.limit:
                retry_after = int(WINDOW_SECONDS - (now - bucket.window_start)) + 1
                raise HTTPException(
                    status.HTTP_429_TOO_MANY_REQUESTS,
                    f"rate limit reached for {self.name} "
                    f"({self.limit} per hour) — try again in {retry_after}s",
                    headers={"Retry-After": str(retry_after)},
                )
            bucket.count += 1

    def reset(self) -> None:
        """Drop all counters. For tests, and for an operator unwedging someone."""
        with self._lock:
            self._buckets.clear()

    def remaining(self, key: str) -> int:
        with self._lock:
            bucket = self._buckets.get(key)
            if bucket is None or time.monotonic() - bucket.window_start >= WINDOW_SECONDS:
                return self.limit
            return max(0, self.limit - bucket.count)


# The shared instances live here rather than in api.py so that papers.py can use
# one without importing api.py, which imports papers.py — a cycle.
ask_limiter = RateLimiter(settings.ask_rate_limit_per_hour, "questions")
upload_limiter = RateLimiter(settings.upload_rate_limit_per_hour, "uploads")

# Brute-force guard on the credential routes. Keyed by client IP, because a
# login attempt has no authenticated user yet — and keying on the SUBMITTED
# email would hand an attacker a lockout weapon against any address they know.
auth_limiter = RateLimiter(settings.auth_rate_limit_per_hour, "sign-in attempts")


def client_key(request: Request) -> str:
    """Rate-limit key for an unauthenticated caller: their IP.

    Behind a reverse proxy every request appears to come from the proxy, so the
    forwarded client IP is preferred when present. NOTE: X-Forwarded-For is
    client-controlled and trivially spoofed unless a trusted proxy overwrites
    it — this is only sound once Scholar is actually deployed behind one.
    """
    forwarded = request.headers.get("x-forwarded-for")
    if forwarded:
        return forwarded.split(",")[0].strip()
    return request.client.host if request.client else "unknown"
