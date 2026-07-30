"""FastAPI surface for Scholar — the RAG pipeline + auth + library, over HTTP.

    POST /ask          {question, k?, candidates?} -> Answer  (auth; scoped to
                       the caller's own papers)
    /auth/*            register / login / logout / me         (see auth.py)
    /papers            upload / list / delete                 (see papers.py)
    GET  /health       -> {"status": "ok"}

The reranker model is loaded once and shared (stateless). The retriever is now
PER USER — each account has its own FAISS index — so it is resolved per request
from the library cache rather than a single global.

Run:  uvicorn backend.api:app --reload   (use the .venv interpreter)
"""
import json
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from backend import library
from backend.auth import get_current_user, router as auth_router
from backend.config import settings
from backend.db import init_db
from backend.db_models import User
from backend.embedder import embed
from backend.generator import condense_question, generate, stream_answer, warm_llm
from backend.models import Answer, AskRequest, Citation
from backend.papers import router as papers_router
from backend.reranker import Reranker

# The reranker model is heavy and stateless, so load it once and share it. The
# retriever is now PER USER (each user has their own index), so it is resolved
# per request from the library cache rather than being a single global.
_reranker = Reranker()


def _warm_models() -> None:
    """Preload the three models so the first request is fast, not cold.

    Runs in a background thread (see lifespan) so the server is ready to accept
    requests immediately while the models load. Non-fatal: if Ollama is down we
    log and move on rather than taking the whole app with us.
    """
    try:
        embed(["warm up the embedder"], progress=False)          # mpnet
        _reranker.rerank("warm up", [                            # cross-encoder
            Citation(paper_id="_", page=0, chunk_index=0, score=0.0, text="warm up passage")
        ], top_k=1)
        warm_llm()                                               # Ollama gemma3:4b
        print("[warm] models ready", flush=True)
    except Exception as exc:  # e.g. Ollama not running
        print(f"[warm] skipped (non-fatal): {exc}", flush=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create DB tables, then warm models in the background so startup
    # stays instant while the first-request cold-load happens ahead of time.
    init_db()
    threading.Thread(target=_warm_models, name="warm", daemon=True).start()
    yield


app = FastAPI(title="Scholar RAG API", lifespan=lifespan)

# CORS: the Vite frontend is a browser app on a different origin. allow_credentials
# is required so the browser sends/receives the auth cookie cross-origin — and it
# is INCOMPATIBLE with allow_origins="*", so the origin must stay explicit.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[settings.frontend_origin, "http://127.0.0.1:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth_router)
app.include_router(papers_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Plain `def` (not async): generate() calls a blocking LLM. FastAPI runs sync
# endpoints in a threadpool, so a slow answer doesn't block other requests.
# Auth-protected and scoped: a user can only ever query their OWN papers.
@app.post("/ask", response_model=Answer)
def ask(request: AskRequest, user: User = Depends(get_current_user)) -> Answer:
    query, citations = _resolve_and_retrieve(user.id, request)
    return generate(query, citations)


def _resolve_and_retrieve(user_id: int, request: AskRequest) -> tuple[str, list[Citation]]:
    """Condense a follow-up into a standalone query, then run stage 1+2 on it.

    Returns (query, citations): `query` is what retrieval + generation should use
    (the condensed standalone question for a follow-up, else the original), and
    `citations` are the reranked sources. Raises 400 if the user has no papers.
    """
    retriever = library.get_retriever(user_id)
    if retriever is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "no papers indexed yet — upload a PDF first",
        )
    # For a follow-up, rewrite it to stand alone so retrieval doesn't choke on
    # pronouns ("what about it?"). No-op when there's no history.
    query = condense_question(request.question, request.history)
    candidates = retriever.retrieve(query, k=request.candidates)
    citations = _reranker.rerank(query, candidates, top_k=request.k)
    return query, citations


# Streaming variant of /ask. Retrieval + reranking run up front, so we can send
# the citations immediately, then stream the answer token-by-token as the local
# LLM writes it — instead of the client staring at a blank screen for ~20s.
# Transport is NDJSON (one JSON object per line): a citations line, then token
# lines, then a done line (or an error line if generation fails mid-stream).
# Sync `def` so Starlette iterates the blocking Ollama generator in a threadpool.
@app.post("/ask/stream")
def ask_stream(request: AskRequest, user: User = Depends(get_current_user)) -> StreamingResponse:
    query, citations = _resolve_and_retrieve(user.id, request)

    def ndjson():
        yield json.dumps({"type": "citations",
                          "citations": [c.model_dump() for c in citations]}) + "\n"
        try:
            for delta in stream_answer(query, citations):
                yield json.dumps({"type": "token", "text": delta}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        except Exception as exc:  # surface a mid-stream failure to the client
            yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"

    return StreamingResponse(ndjson(), media_type="application/x-ndjson")
