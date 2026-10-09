"""Modules that name a model must load without a configured deploy — and the
app must never pull torch back in.

Scripts (scripts/fetch_models.py, the eval, one-off tools) import the modules
that NAME each model, and a script has no environment and maybe no .env. Any
module reachable from those imports that constructs `Settings()` at import time
fails there with a missing SECRET_KEY, long before the app ever runs.

This is not hypothetical: adding `from backend.config import settings` to the
top of reranker.py once broke the image build, and the whole test suite stayed
green because the repo has a .env. The test therefore runs in a SUBPROCESS from a
different working directory, which is what makes it a real check — in-process,
`backend.config` is already imported and `.env` is one relative path away.

The second guard exists because the serving image has no torch: the models run
on ONNX Runtime, and torch alone would blow the 512 MB budget. A developer
machine still has torch installed for the eval baseline, so a stray import
would work locally and fail only in the image.
"""
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]

from backend.config import Settings  # noqa: E402 — needs REPO_ROOT resolved first

# The modules that name the models.
BUILD_IMPORTS = "from backend.embedder import MODEL_NAME; from backend.reranker import RERANKER_MODEL"


def _unconfigured_env() -> dict[str, str]:
    """The real environment minus anything that would configure the app.

    Inherited rather than built from scratch on purpose: native libraries
    (onnxruntime, faiss) need PATH and SYSTEMROOT to load their DLLs, and an
    empty environment fails for that reason instead of the one under test.
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


def test_the_app_never_imports_torch_or_transformers(tmp_path):
    """Import the whole app and every model-touching module, then check that
    neither torch nor transformers was loaded along the way — even here, where
    they may well be installed."""
    env = _unconfigured_env()
    env["SECRET_KEY"] = "import-check-only"
    probe = (
        "import sys; import backend.api, backend.indexing, backend.jobs, backend.migrate; "
        "from backend import embedder, reranker, chunker; "
        "bad = sorted(m for m in ('torch', 'transformers', 'sentence_transformers') if m in sys.modules); "
        "print('LOADED:' + ','.join(bad))"
    )
    result = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=300)
    assert result.returncode == 0, result.stderr[-2000:]
    assert result.stdout.strip() == "LOADED:", (
        f"the app imported {result.stdout.strip()} — the serving image has no torch"
    )


def test_requirements_do_not_ship_torch():
    """The deploy requirements must not list torch or anything that drags it in."""
    text = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").lower()
    for banned in ("torch", "sentence-transformers", "transformers"):
        assert banned not in text, f"requirements.txt lists {banned}; the image must stay torch-free"
