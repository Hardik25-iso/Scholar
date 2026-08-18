"""The modules the Docker build imports must load without a configured deploy.

The image bakes the models in at build time by importing the module that NAMES
each model, so the weights land in a layer instead of being downloaded on every
cold start. A build has no environment and no .env — so any module reachable
from those imports that constructs `Settings()` at import time fails the build
with a missing SECRET_KEY, long before the app ever runs.

This is not hypothetical: adding `from backend.config import settings` to the
top of reranker.py broke the image build, and the whole test suite stayed green
because the repo has a .env. The test therefore runs in a SUBPROCESS from a
different working directory, which is what makes it a real check — in-process,
`backend.config` is already imported and `.env` is one relative path away.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from backend.config import Settings  # noqa: E402 — needs REPO_ROOT resolved first

# Kept in step with the model-baking RUN in the Dockerfile.
BUILD_IMPORTS = "from backend.embedder import MODEL_NAME; from backend.reranker import RERANKER_MODEL"


def _unconfigured_env() -> dict[str, str]:
    """The real environment minus anything that would configure the app.

    Inherited rather than built from scratch on purpose: torch needs PATH and
    SYSTEMROOT to load its DLLs, and an empty environment fails for that reason
    instead of the one under test.
    """
    env = os.environ.copy()
    for key in list(env):
        if key in Settings.model_fields or key.lower() in Settings.model_fields:
            del env[key]
    env["PYTHONPATH"] = str(REPO_ROOT)      # importable...
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    return env


def test_build_time_imports_need_no_environment(tmp_path):
    # ...but cwd is elsewhere, so the repo's .env is NOT found — exactly the
    # build's situation. Without this the test passes for the wrong reason.
    env = _unconfigured_env()
    result = subprocess.run(
        [sys.executable, "-c", f"{BUILD_IMPORTS}; print(MODEL_NAME, RERANKER_MODEL)"],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, (
        "a module the Docker build imports now needs configuration at import "
        "time, which breaks the image build:\n" + result.stderr[-2000:]
    )
    assert "SECRET_KEY" not in result.stderr


def test_the_guard_would_actually_catch_a_regression(tmp_path):
    """Proves the harness above can fail — a green test that cannot go red is
    worse than no test, and this one depends on a subtle cwd/env setup."""
    result = subprocess.run(
        [sys.executable, "-c", "from backend.config import settings"],
        cwd=tmp_path,
        env=_unconfigured_env(),
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode != 0
    assert "secret_key" in result.stderr.lower()
