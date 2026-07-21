"""FastAPI surface for Scholar — the same RAG pipeline, over HTTP.

    POST /ask     {question, k?, candidates?} -> Answer (grounded + citations)
    GET  /health  -> {"status": "ok"}

The retriever and reranker are loaded ONCE at startup (they pull FAISS + two
Transformer models into memory) and reused for every request — see the lifespan
handler. Loading them per request would add ~1-2s of model-loading to every call.

Run:  uvicorn backend.api:app --reload
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from backend.generator import generate
from backend.config import settings
from backend.db import init_db
from backend.models import Answer, AskRequest
from backend.reranker import Reranker
from backend.retriever import Retriever

# Filled in by the lifespan handler at startup; reused across all requests.
_pipeline: dict[str, object] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create DB tables, then load the heavy RAG objects once.
    init_db()
    _pipeline["retriever"] = Retriever()
    _pipeline["reranker"] = Reranker()
    yield
    # Shutdown: nothing to clean up (in-process, no connections).
    _pipeline.clear()


app = FastAPI(title="Scholar RAG API", lifespan=lifespan)

# CORS: the Vite frontend is a browser app on a different origin, so the request
# is blocked without this. Explicit origin from config (never "*") — required now
# and mandatory once cookie auth (allow_credentials) lands in Slice B.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://127.0.0.1:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Plain `def` (not async): generate() calls a blocking LLM. FastAPI runs sync
# endpoints in a threadpool, so a slow answer doesn't block other requests.
@app.post("/ask", response_model=Answer)
def ask(request: AskRequest) -> Answer:
    retriever: Retriever = _pipeline["retriever"]      # type: ignore[assignment]
    reranker: Reranker = _pipeline["reranker"]         # type: ignore[assignment]

    # Stage 1: wide net.  Stage 2: rerank to the top k.  Stage 3: grounded answer.
    candidates = retriever.retrieve(request.question, k=request.candidates)
    citations = reranker.rerank(request.question, candidates, top_k=request.k)
    return generate(request.question, citations)
