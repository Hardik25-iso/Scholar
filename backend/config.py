"""Application configuration, loaded from environment / .env.

All runtime knobs live here so nothing secret is hard-coded. Values come from
(in order) real environment variables, then the .env file (git-ignored).

NOTE: .env may hold unrelated legacy keys from earlier experiments; they are NOT
used by Scholar (generation runs locally via Ollama). `extra="ignore"` below lets
pydantic-settings tolerate such stray keys instead of failing at startup.
"""
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # tolerate unrelated keys in .env (e.g. the legacy one)
    )

    # SQLite lives under data/ (git-ignored) so the DB is never committed.
    database_url: str = "sqlite:///./data/scholar.db"

    # Where per-user libraries (FAISS indexes, lexical indexes, stored files)
    # live. Empty means the repo-local `data/users`, which is right for local
    # development and WRONG for any real deployment: the default sits inside the
    # source tree, so a redeploy that replaces the source silently destroys every
    # user's library. Point this at a mounted volume before serving anyone.
    data_root: str = ""

    # The browser origin allowed to call this API (the Vite dev server).
    frontend_origin: str = "http://localhost:5173"

    # JWT signing secret. REQUIRED — no default on purpose: a fallback like
    # "changeme" would silently "work" in dev and ship forgeable tokens. With no
    # default, a missing SECRET_KEY makes the app fail loudly at startup.
    secret_key: str

    # Access-token lifetime. Short by design: a stolen access token expires
    # quickly. The refresh token is what keeps that from meaning "logged out
    # every 30 minutes" — it is long-lived but can only mint access tokens.
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 30

    # How long a password-reset link stays valid. Short, because the window in
    # which a leaked link is useful should be measured in minutes.
    reset_token_expire_minutes: int = 30

    # Per-user request budgets. /ask runs retrieval, reranking and a full LLM
    # generation, so it is the expensive route and the one worth capping — and
    # it becomes a billing control the moment generation is a paid API.
    ask_rate_limit_per_hour: int = 120
    upload_rate_limit_per_hour: int = 60

    # Cookie flags. In local dev over http://localhost, SameSite=Lax works
    # without Secure (localhost is treated as same-site + a secure context).
    # Set cookie_secure=True (and SameSite=None) once served over HTTPS.
    cookie_secure: bool = False
    cookie_samesite: str = "lax"

    # Whether to OCR scanned PDFs at all. OCR is slow and CPU-hungry, so a small
    # instance may reasonably refuse it — and an explicit switch is the only
    # honest way to express "off". Clearing tessdata_prefix does NOT disable it:
    # Tesseract falls back to its own compiled-in data location, which is how
    # every Linux package install works.
    ocr_enabled: bool = True

    # Where Tesseract's language data lives, for OCR of scanned PDFs. Empty means
    # "discover it" — see parser._tessdata(), which checks TESSDATA_PREFIX and the
    # standard install locations. Set this explicitly when deploying to an image
    # that puts tessdata somewhere non-standard. OCR is optional: with no data
    # directory a scanned PDF is rejected with a clear 422, not a crash.
    tessdata_prefix: str = ""


settings = Settings()
