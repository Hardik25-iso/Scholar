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
import logging
import threading
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from backend import audit, generator, library
from backend.audit_routes import router as audit_router
from backend.auth import get_current_user, require_csrf, router as auth_router
from backend.search import shortlist
from backend.config import settings
from backend.db import get_session, init_db
from backend.db_models import User, Workspace
from backend.embedder import embed
from backend.generator import condense_question, generate, stream_answer, warm_llm
from backend.models import Answer, AskRequest, Citation
from backend.papers import router as papers_router
from backend.ratelimit import ask_limiter
from backend.reranker import Reranker
from backend.workspace_routes import router as workspace_router
from backend.workspaces import get_current_workspace

log = logging.getLogger(__name__)

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
        log.info("models ready")
    except Exception as exc:  # e.g. Ollama not running
        log.warning("model warm-up skipped (non-fatal): %s", exc)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: create DB tables, then warm models in the background so startup
    # stays instant while the first-request cold-load happens ahead of time.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S%z",
    )
    init_db()
    # Loud about the one misconfiguration that destroys data without an error:
    # the default library location lives inside the source tree.
    if not settings.data_root:
        log.warning(
            "DATA_ROOT is unset — user libraries live in the source tree at %s. "
            "A redeploy that replaces the source will DELETE every library. "
            "Set DATA_ROOT to a mounted volume before serving real users.",
            library.DATA_ROOT,
        )
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
app.include_router(audit_router)
app.include_router(workspace_router)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


# Plain `def` (not async): generate() calls a blocking LLM. FastAPI runs sync
# endpoints in a threadpool, so a slow answer doesn't block other requests.
# Auth-protected and scoped: a user can only ever query their OWN papers.
#
# require_csrf, like every other POST: /ask reads no state but it is the most
# EXPENSIVE route in the app (retrieval + reranking + full LLM generation). The
# cookie alone would let any site drive a logged-in user's browser into an
# unbounded compute — and a billed one, once generation moves to a paid API.
@app.post("/ask", response_model=Answer, dependencies=[Depends(require_csrf)])
def ask(
    request: AskRequest,
    user: User = Depends(get_current_user),
    workspace: Workspace = Depends(get_current_workspace),
    session: Session = Depends(get_session),
) -> Answer:
    ask_limiter.check(str(user.id))
    query, citations, retrieval_query = _resolve_and_retrieve(workspace.id, request)
    answer = generate(query, citations)
    _record(session, user.id, workspace.id, request, query, answer, retrieval_query)
    return answer


def _record(session: Session, user_id: int, workspace_id: int, request: AskRequest,
            query: str, answer: Answer, retrieval_query: str | None = None) -> None:
    """Log the served answer with its evidence. Never raises — see audit.record.

    Both ids are kept: who asked, and which library they asked. In a shared
    workspace those are different questions, and an audit trail that could only
    answer one of them would be worth much less.
    """
    audit.record(
        session, user_id, workspace_id, request, query, answer,
        index_dir=library.workspace_index_dir(workspace_id),
        model=generator.MODEL,
        temperature=generator.TEMPERATURE,
        retrieval_query=retrieval_query,
        expansion_mode=settings.query_expansion,
    )


def _resolve_and_retrieve(
    workspace_id: int, request: AskRequest
) -> tuple[str, list[Citation], str]:
    """Condense a follow-up into a standalone query, then run stage 1+2 on it.

    Stage 1 is HYBRID: the dense retriever finds passages that mean the right
    thing, the lexical index finds passages containing the right tokens, and the
    two ranked lists are fused. Neither alone covers what this product is asked —
    "how does the renewal work" needs meaning, "Section 7.2" needs the literal
    string. Stage 2 (the cross-encoder) is unchanged; it just receives a better
    shortlist than either retriever produces on its own.

    Returns (query, citations): `query` is what retrieval + generation should use
    (the condensed standalone question for a follow-up, else the original), and
    `citations` are the reranked sources. Raises 400 if the user has no papers.
    """
    retriever = library.get_retriever(workspace_id)
    if retriever is None:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "no documents indexed yet — upload one first",
        )
    # For a follow-up, rewrite it to stand alone so retrieval doesn't choke on
    # pronouns ("what about it?"). No-op when there's no history.
    query = condense_question(request.question, request.history)
    candidates, retrieval_query = shortlist(
        query,
        library.workspace_index_dir(workspace_id),
        retriever,
        k=request.candidates,
        papers=request.papers,
        expansion_mode=settings.query_expansion,
    )
    # Reranked with the EXPANDED query, not the original. The measured failure
    # was the cross-encoder discarding a correct passage that retrieval had
    # already found, because the question shared no vocabulary with it.
    # Generation still receives the user's own question — `query` — since the
    # expansion terms are a retrieval device, not part of what was asked.
    citations = _reranker.rerank(retrieval_query, candidates, top_k=request.k)
    return query, citations, retrieval_query


# Streaming variant of /ask. Retrieval + reranking run up front, so we can send
# the citations immediately, then stream the answer token-by-token as the local
# LLM writes it — instead of the client staring at a blank screen for ~20s.
# Transport is NDJSON (one JSON object per line): a citations line, then token
# lines, then a done line (or an error line if generation fails mid-stream).
# Sync `def` so Starlette iterates the blocking Ollama generator in a threadpool.
@app.post("/ask/stream", dependencies=[Depends(require_csrf)])
def ask_stream(
    request: AskRequest,
    user: User = Depends(get_current_user),
    workspace: Workspace = Depends(get_current_workspace),
    session: Session = Depends(get_session),
) -> StreamingResponse:
    ask_limiter.check(str(user.id))
    query, citations, retrieval_query = _resolve_and_retrieve(workspace.id, request)

    def ndjson():
        yield json.dumps({"type": "citations",
                          "citations": [c.model_dump() for c in citations]}) + "\n"
        parts: list[str] = []
        try:
            for delta in stream_answer(query, citations):
                parts.append(delta)
                yield json.dumps({"type": "token", "text": delta}) + "\n"
            yield json.dumps({"type": "done"}) + "\n"
        except Exception as exc:  # surface a mid-stream failure to the client
            yield json.dumps({"type": "error", "detail": str(exc)}) + "\n"
            return  # a failed generation is not an answer, so do not log one
        # Logged only after a clean finish, and with the text actually sent —
        # reassembled from the deltas rather than regenerated, so the audit
        # record is what the user saw, not a second guess at it.
        _record(session, user.id, workspace.id, request, query,
                Answer(question=query, answer="".join(parts), citations=citations),
                retrieval_query)

    return StreamingResponse(ndjson(), media_type="application/x-ndjson")
