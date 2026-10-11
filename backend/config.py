"""Application configuration, loaded from environment / .env.

All runtime knobs live here so nothing secret is hard-coded. Values come from
(in order) real environment variables, then the .env file (git-ignored).

`extra="ignore"` below lets pydantic-settings tolerate stray keys in .env instead
of failing at startup — including the legacy ANTHROPIC_API_KEY from an earlier
experiment, which Scholar does not read. (It should still be rotated: it was
exposed in this project's history. Nothing here depends on it.)
"""
import os
from typing import Literal

from pydantic import field_validator
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

    # Where libraries are stored. "local" keeps them under DATA_ROOT (development,
    # tests, and any host with a real disk). "supabase" keeps them in a Supabase
    # Storage bucket, for hosts whose filesystem is wiped on every restart —
    # Render's free tier is one. Either way, which version of a workspace's
    # index is CURRENT is recorded in the database, never inferred from files.
    # A Literal, so a typo ("supabse") fails at startup instead of silently
    # storing everything on a disk that is wiped on the next restart.
    storage_backend: Literal["local", "supabase"] = "local"
    supabase_url: str = ""
    # The service-role key: the server reads and writes a PRIVATE bucket on the
    # user's behalf, after doing its own authorisation. Never sent to a browser.
    supabase_service_key: str = ""
    supabase_bucket: str = "scholar"

    # Local working copies of indexes and documents. Disposable by design: every
    # file here can be re-fetched from storage, so an ephemeral disk is fine.
    # Empty means <system temp>/scholar-cache.
    cache_dir: str = ""

    # ——— models ———
    #
    # The ONNX files from scripts/fetch_models.py. Empty means the repo-local
    # models/ directory; the Docker image sets /opt/models.
    models_dir: str = ""
    # "int8" (default) or "fp32". fp32 mpnet alone is 436 MB of weights, so on a
    # 512 MB instance int8 is the only choice; fp32 is for parity tests and eval.
    model_precision: Literal["int8", "fp32"] = "int8"
    # Texts per embedding call. 1 is fastest AND lightest on a single thread
    # (measured: see embedder.py); raise it only on a machine with cores to
    # spare, and expect activation memory to grow with it.
    embed_batch_size: int = 1

    # ——— reranking cost ———
    #
    # The cross-encoder is the most CPU-hungry step of a question: one full model
    # pass per candidate. On a fraction of a CPU these three are the levers, and
    # each one's effect on retrieval quality is measured, not assumed (see
    # docs/deploy-measurements/).
    #
    # The most candidates one question may rerank. Also a CAP on the client's
    # `candidates` request field — without it any logged-in user could ask for
    # hundreds of model passes in a single request.
    rerank_candidates: int = 20
    # Tokens per (question, passage) pair fed to the cross-encoder; the model's
    # ceiling is 512. Cost grows with length, and most of the relevance signal
    # is near the start of a passage.
    rerank_max_tokens: int = 512
    # Off = keep the fused (dense + lexical) order and skip the cross-encoder.
    rerank_enabled: bool = True

    # ——— limits that keep one document within the instance's budget ———
    #
    # Each bounds a different resource: upload size bounds what is held while
    # receiving, pages bound parse time and memory, chunks bound embedding time
    # — the one that decides whether indexing finishes before an idle server
    # is put to sleep. Measured, not guessed: see docs/deploy-measurements/.
    max_upload_mb: int = 20
    max_pages: int = 0           # 0 = unlimited (local default)
    max_chunks_per_doc: int = 0  # 0 = unlimited (local default)

    # Workspaces whose index is kept loaded in memory. Each holds a FAISS index
    # plus its chunk text, so on a small instance this is a memory setting.
    retriever_cache_size: int = 8

    # With no queue (REDIS_URL empty), index uploads on a background thread
    # instead of inside the request, so the upload returns at once and the
    # browser polls the job. One job at a time either way: a second concurrent
    # indexing run would double peak memory for no gain on one CPU.
    index_in_background: bool = False

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

    # Invitations live longer than resets — people take days to act on them, and
    # the consequence of a stale one is a re-invite rather than a lockout.
    invitation_expire_days: int = 14

    # Per-user request budgets. /ask runs retrieval, reranking and a full LLM
    # generation, so it is the expensive route and the one worth capping — and
    # it becomes a billing control the moment generation is a paid API.
    ask_rate_limit_per_hour: int = 120
    upload_rate_limit_per_hour: int = 60

    # Auth attempts per hour PER CLIENT IP (not per user — a login attempt has
    # no authenticated user yet, and keying on the submitted email would let an
    # attacker lock a victim out by spamming their address). This is the
    # brute-force guard: generous enough that a person fumbling their password
    # never notices, tight enough that credential stuffing is not viable.
    auth_rate_limit_per_hour: int = 20

    # How long to wait on the LLM before giving up. Without this a hung local
    # model pins a threadpool worker forever and the app slowly dies. Generous:
    # a cold model load plus a long grounded answer can legitimately take a
    # while on CPU (measured ~44s cold, ~8s warm to first token).
    llm_timeout_seconds: float = 180.0

    # ——— Which LLM generates answers ———
    #
    # "ollama" (default) runs gemma3:4b on this machine: free, private, and the
    # right choice for local development — but it cannot be deployed, because a
    # hosted box has no Ollama and nobody is going to install a 3.3 GB model to
    # try your site. "hosted" points at any OpenAI-compatible endpoint so
    # Scholar can run somewhere real. Only the provider changes: retrieval,
    # reranking, citations and the grounding contract are identical either way.
    llm_provider: str = "ollama"  # "ollama" | "hosted"

    # Required when llm_provider="hosted"; all three, or startup fails loudly
    # rather than the first question failing. Vendor-neutral on purpose — the
    # OpenAI chat protocol is the de-facto standard, so the same three values
    # select Groq, Google's Gemini/Gemma API, OpenRouter, or a self-hosted
    # server, with no code change. Free-tier examples:
    #
    #   Groq        LLM_BASE_URL=https://api.groq.com/openai/v1
    #               LLM_MODEL=llama-3.3-70b-versatile   (Groq retired Gemma;
    #               Gemma now lives on Google's endpoint below)
    #   Gemini      LLM_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
    #               LLM_MODEL=gemini-2.0-flash
    #   OpenRouter  LLM_BASE_URL=https://openrouter.ai/api/v1
    #               LLM_MODEL=<a model whose id ends in :free>
    #
    # Model ids change as vendors add and retire models — check the provider's
    # current list rather than trusting these examples. A wrong id surfaces as a
    # clear "does not serve a model named ..." error, not a silent fallback.
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    # Ceiling on a generated answer. Grounded answers are short by design (a few
    # paragraphs citing retrieved passages), so this is a runaway guard, not a
    # target — and it caps spend per question on the hosted path.
    llm_max_tokens: int = 2048

    # Cookie flags. In local dev over http://localhost, SameSite=Lax works
    # without Secure (localhost is treated as same-site + a secure context).
    # Set cookie_secure=True (and SameSite=None) once served over HTTPS.
    cookie_secure: bool = False
    cookie_samesite: str = "lax"

    # Query-expansion strategy: "none", "prf" or "hyde".
    #
    # "hyde" is the default because it was measured best: misses 3 -> 1, hit@5
    # 94.1% -> 98.0%, and the `vocabulary` class off 0%. It costs one extra LLM
    # call per question and trades some ranking precision for recall.
    #
    # It is only an honest default because AnswerLog.retrieval_query records the
    # hypothetical it generated. Retrieval is no longer purely deterministic, so
    # the audit trail has to show what was actually searched — otherwise there is
    # no way to tell a changed library from a differently-worded hypothetical.
    # If that column is ever dropped, this must go back to "none".
    #
    # "prf" was built, measured and REJECTED: worse everywhere, and it did not
    # move the class it targeted. Kept selectable so the result stays
    # reproducible — see docs/04-pmf-roadmap.md.
    query_expansion: str = "hyde"

    # Maximal Marginal Relevance: after reranking, prefer passages that are both
    # relevant AND unlike the ones already chosen.
    #
    # Off by default until it is measured on this corpus. The hypothesis is that
    # the chunker's 50-token overlap puts near-duplicate passages in the top k,
    # spending slots on repeated text — see backend/mmr.py. Whether that actually
    # happens here is a question for `python -m backend.eval --mmr`, not for an
    # opinion: PRF also sounded reasonable and was measured and rejected.
    mmr_enabled: bool = False

    # Relevance vs diversity. 1.0 is pure relevance and reduces exactly to the
    # pre-MMR behaviour; lower trades relevance for distinctness. 0.7 is the
    # usual starting point in the literature.
    mmr_lambda: float = 0.7

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

    # ——— Indexing queue ———
    #
    # Where the arq worker and the API meet. Empty means "no queue": uploads are
    # indexed inside the request, which is the old behaviour and races the proxy
    # timeout on a large document. Correct for local development, wrong for a
    # deployment — and either way the IndexJob row records which one happened,
    # so a slow upload is never a mystery.
    #
    # Setting this is only half the job: the worker is a SECOND process,
    #   arq backend.jobs.WorkerSettings
    # and a deploy that starts only the API accepts uploads it will never index.
    redis_url: str = ""

    # ——— Outbound email ———
    #
    # Empty smtp_host means "no mail provider". The app still runs; password
    # resets and invitations fall back to logging their token and SAY SO rather
    # than pretending a message went out. See backend/mailer.py.
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    # STARTTLS on 587 (the common case) vs implicit TLS on 465. Turn starttls
    # off only for a relay on localhost — over the internet it sends the SMTP
    # password in clear text.
    smtp_starttls: bool = True
    smtp_ssl: bool = False
    # A hung mail server must not hold an HTTP worker open indefinitely.
    smtp_timeout: int = 10
    # The From address. Required alongside smtp_host — a message with no sender
    # is rejected by every provider, so treating it as optional would only move
    # the failure to send time.
    mail_from: str = ""

    # Where links in email point. Defaults to frontend_origin, which is right in
    # development and wrong the first time this is deployed behind a real
    # domain — the CORS origin and the public URL are not always the same thing.
    public_app_url: str = ""

    # Which commit this process is running, surfaced on /healthz so a deploy can
    # be verified with one request instead of a trip to the hosting dashboard.
    #
    # Resolved from BUILD_SHA, falling back to the commit the hosting platform
    # injects for free (RAILWAY_GIT_COMMIT_SHA, Render's RENDER_GIT_COMMIT) — the
    # deployed instance therefore reports its commit with no build configuration. A local `docker build` passes
    # --build-arg GIT_SHA instead (see the Dockerfile), so this is not tied to
    # one host.
    #
    # Defaults to "unknown" rather than "", because an empty string renders as a
    # missing field and reads like "this build has no commit" — when the honest
    # meaning is "nobody told me". Truncated to 7 characters: enough to identify
    # a commit by eye, and it is what `git log --oneline` prints.
    build_sha: str = ""

    @field_validator("build_sha", mode="after")
    @classmethod
    def _resolve_build_sha(cls, v: str) -> str:
        injected = os.getenv("RAILWAY_GIT_COMMIT_SHA") or os.getenv("RENDER_GIT_COMMIT") or ""
        return (v or injected).strip()[:7] or "unknown"

    @field_validator("*", mode="before")
    @classmethod
    def _strip_whitespace(cls, v: object) -> object:
        """Trim surrounding whitespace from every string setting.

        Pasting a value into a hosting dashboard is how most of these are set,
        and a trailing newline is invisible there. It is not harmless: a key
        with a trailing "\\n" makes an HTTP header value illegal, so the client
        refuses to send the request at all — and reports it as a *connection*
        error, which sends you hunting for a network fault that does not exist.
        Cost us a deploy; costs one line to prevent.
        """
        return v.strip() if isinstance(v, str) else v


settings = Settings()
