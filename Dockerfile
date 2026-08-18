# Scholar — one container serving the API and the built frontend.
#
# Serving both from a single origin is deliberate: it makes the auth cookie
# same-site in production, so the cross-origin cookie handling that development
# needs disappears entirely once deployed.
#
# Build:  docker build -t scholar .
# Run:    docker run -p 8001:8001 --env-file .env -v scholar-data:/data scholar
#
# NOTE ON GENERATION: this image does NOT contain Ollama. A deployed Scholar
# uses the hosted provider — LLM_PROVIDER=hosted plus LLM_BASE_URL, LLM_MODEL
# and LLM_API_KEY (a free Groq / Gemini / OpenRouter endpoint works; see
# backend/config.py). Left at the "ollama" default the container starts and
# serves the site, but every question fails with a 503: there is no local model
# to reach. /health reports that honestly rather than claiming to be ready.

# ——————————————————————— stage 1: build the frontend ———————————————————————
FROM node:22-slim AS frontend

WORKDIR /app/frontend
# Copy manifests first so `npm ci` is cached until dependencies actually change.
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci

COPY frontend/ ./
# Empty API base => same-origin requests. See frontend/src/api.ts.
ENV VITE_API_BASE=""
RUN npm run build


# ——————————————————————— stage 2: the application ———————————————————————
FROM python:3.12-slim AS app

# PYTHONUNBUFFERED so logs appear in `docker logs` immediately rather than
# sitting in a buffer; PYTHONDONTWRITEBYTECODE to keep the layer clean.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/opt/models

WORKDIR /app

# Tesseract is required to read scanned PDFs (see parser.py); without it those
# uploads fail at ingest rather than at startup, which is a confusing way to
# find out it is missing.
RUN apt-get update \
    && apt-get install --no-install-recommends -y tesseract-ocr curl gosu \
    && rm -rf /var/lib/apt/lists/*

# Dependencies before source, so a code change doesn't reinstall torch.
COPY requirements.txt ./

# CPU-only torch, installed FIRST so the CUDA build is never pulled.
#
# torch arrives transitively via sentence-transformers, and its default wheel
# bundles the CUDA runtime — cuBLAS alone is 423 MB, and the full set adds
# several GB of GPU libraries that a CPU-only container can never use. Naming
# the CPU index here satisfies the dependency before requirements.txt asks for
# it. Embedding and reranking run on CPU either way; this only removes weight.
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install -r requirements.txt

COPY backend/ ./backend/

# Bake the embedding, tokenizer and reranker models into the image.
#
# Without this the first question after every deploy downloads ~500 MB inside
# the request, and a cold container looks broken for minutes. Doing it at build
# time makes image size the cost instead of first-request latency — and means
# the container needs no model downloads at runtime at all.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
from transformers import AutoTokenizer; \
from backend.embedder import MODEL_NAME; \
from backend.reranker import RERANKER_MODEL; \
SentenceTransformer(MODEL_NAME); \
AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True); \
CrossEncoder(RERANKER_MODEL); \
print('models baked into image')"

# The built SPA. api.py serves it only when this directory exists, which is why
# development (where it does not) stays API-only.
COPY --from=frontend /app/frontend/dist ./frontend/dist

# Libraries and the database live on a mounted volume, never in the image: the
# default location is inside the source tree, so a redeploy would wipe every
# user's papers. DATA_ROOT and DATABASE_URL point at /data instead.
ENV DATA_ROOT=/data/workspaces \
    DATABASE_URL=sqlite:////data/scholar.db

# NO `VOLUME /data` HERE, DELIBERATELY. Railway rejects the VOLUME instruction
# outright ("use Railway Volumes") and the build fails before it starts. The
# instruction was never load-bearing: it only declares a default mount point.
# Attaching storage is the operator's job either way —
#   Railway/Fly/Render: attach a volume at /data in the platform's UI
#   plain Docker:       docker run -v scholar-data:/data ...
# and the directory itself is created and chowned below, so an unmounted run
# still works (writing into the container layer, which is lost on restart).

# Run as a non-root user. Anything that escapes the app should not own the box.
#
# UID 1000 specifically: Hugging Face Spaces runs containers as uid 1000 and
# writes fail if the app's user is anything else. Any other host is indifferent
# to the number, so 1000 is the portable choice.
RUN useradd --create-home --uid 1000 scholar \
    && mkdir -p /data \
    && chown -R scholar:scholar /data /app /opt/models

COPY docker-entrypoint.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/docker-entrypoint.sh

# Deliberately NO `USER scholar` here. The chown above applies to the image, but
# a platform that mounts a volume over /data replaces that directory and its
# ownership at runtime — Railway bind-mounts it root-owned, and the app then
# cannot open its own database. Fixing that needs root; running the app as root
# does not. So the entrypoint starts as root purely to chown the mount, then
# execs the app as `scholar`. The app process itself never runs as root.
ENTRYPOINT ["docker-entrypoint.sh"]

# Which commit this image was built from, reported by /healthz so a deploy can
# be confirmed with one request rather than a trip to the hosting dashboard.
#
# DELIBERATELY THE LAST LAYER THAT VARIES. A build arg invalidates its own layer
# and every layer after it, so declaring this near the top would make each new
# commit rebuild the model-baking step — turning a ~1 minute cached deploy back
# into the full ~10 minute download. Everything expensive is already above.
#
# Empty by default. On Railway this stays empty and the app falls back to
# RAILWAY_GIT_COMMIT_SHA, which the platform injects at runtime for free; pass
# it explicitly anywhere else:
#   docker build --build-arg GIT_SHA=$(git rev-parse --short HEAD) .
ARG GIT_SHA=""
ENV BUILD_SHA=$GIT_SHA

EXPOSE 8001

# The readiness check the app already exposes: it verifies the database, the
# embedder and the LLM, so an unhealthy container is one that genuinely cannot
# answer a question — not merely one whose process is alive.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
    CMD curl -fsS "http://127.0.0.1:${PORT:-8001}/healthz" || exit 1

# Shell form, not exec form, so ${PORT} is expanded at runtime.
#
# Railway (and Cloud Run, and Heroku) assign a port and route traffic to it,
# so a hardcoded port means the platform's router never reaches the container
# and the deploy fails health checks while the app is in fact running fine.
# Falls back to 8001 so `docker run -p 8001:8001` still works unchanged.
CMD uvicorn backend.api:app --host 0.0.0.0 --port ${PORT:-8001}
