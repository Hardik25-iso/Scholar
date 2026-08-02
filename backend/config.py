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

    # The browser origin allowed to call this API (the Vite dev server).
    frontend_origin: str = "http://localhost:5173"

    # JWT signing secret. REQUIRED — no default on purpose: a fallback like
    # "changeme" would silently "work" in dev and ship forgeable tokens. With no
    # default, a missing SECRET_KEY makes the app fail loudly at startup.
    secret_key: str

    # Access-token lifetime. Short by design; no refresh tokens yet.
    access_token_expire_minutes: int = 30

    # Cookie flags. In local dev over http://localhost, SameSite=Lax works
    # without Secure (localhost is treated as same-site + a secure context).
    # Set cookie_secure=True (and SameSite=None) once served over HTTPS.
    cookie_secure: bool = False
    cookie_samesite: str = "lax"

    # Where Tesseract's language data lives, for OCR of scanned PDFs. Empty means
    # "discover it" — see parser._tessdata(), which checks TESSDATA_PREFIX and the
    # standard install locations. Set this explicitly when deploying to an image
    # that puts tessdata somewhere non-standard. OCR is optional: with no data
    # directory a scanned PDF is rejected with a clear 422, not a crash.
    tessdata_prefix: str = ""


settings = Settings()
