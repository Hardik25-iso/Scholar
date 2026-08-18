"""/healthz reports which commit it is running.

"The app is healthy" and "the app is running the code I just pushed" are two
different claims, and only the first was ever observable. Confirming a deploy
meant opening the hosting dashboard, because nothing the app served could tell
the new build from the old one — which is exactly how a failed build once sat
unnoticed behind a still-healthy previous container.
"""
import os
import subprocess
import sys
from pathlib import Path

from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_healthz_reports_the_build(client: TestClient):
    body = client.get("/healthz").json()
    assert body["status"] == "ok"
    assert body["build"], "a deploy check with no build field cannot confirm a deploy"


def test_the_build_is_reported_when_the_deploy_gate_fails(client: TestClient, monkeypatch):
    """The failing case is when you most need to know WHICH build is broken."""
    from backend import api

    class _DeadEngine:
        def __enter__(self, *a):
            raise RuntimeError("database is gone")

        def __exit__(self, *a):
            return False

    monkeypatch.setattr(api, "Session", lambda *a, **k: _DeadEngine())

    r = client.get("/healthz")
    assert r.status_code == 503
    assert r.json()["status"] == "error"
    assert r.json()["build"], "the broken build must still identify itself"


def _sha_in_env(**env_overrides) -> str:
    """Resolve build_sha in a subprocess with a controlled environment.

    A subprocess because Settings is constructed once at import; monkeypatching
    the environment afterwards would not re-run the resolution being tested.
    """
    env = os.environ.copy()
    for key in ("BUILD_SHA", "RAILWAY_GIT_COMMIT_SHA"):
        env.pop(key, None)
    env.update(env_overrides)
    env["PYTHONPATH"] = str(REPO_ROOT)
    out = subprocess.run(
        [sys.executable, "-c", "from backend.config import settings; print(settings.build_sha)"],
        cwd=REPO_ROOT, env=env, capture_output=True, text=True, timeout=120,
    )
    assert out.returncode == 0, out.stderr[-1500:]
    return out.stdout.strip()


def test_explicit_build_sha_wins():
    assert _sha_in_env(BUILD_SHA="abcdef1234567") == "abcdef1"


def test_falls_back_to_the_platform_variable():
    """Railway injects this for free, so the deployed app self-identifies with
    no build configuration at all."""
    assert _sha_in_env(RAILWAY_GIT_COMMIT_SHA="1234567890abc") == "1234567"


def test_explicit_value_takes_precedence_over_the_platform():
    assert _sha_in_env(BUILD_SHA="aaaaaaa1", RAILWAY_GIT_COMMIT_SHA="bbbbbbb2") == "aaaaaaa"


def test_unset_reports_unknown_not_empty():
    """An empty string renders as a missing field and reads like "this build has
    no commit". The honest meaning is "nobody told me"."""
    assert _sha_in_env() == "unknown"


def test_a_pasted_value_with_whitespace_is_still_usable():
    """Same failure mode as the API key that broke a deploy: dashboards hide
    trailing newlines."""
    assert _sha_in_env(RAILWAY_GIT_COMMIT_SHA="  fedcba9876  \n") == "fedcba9"
