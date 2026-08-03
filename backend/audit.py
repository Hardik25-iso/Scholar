"""Recording answers with their evidence, and reading them back.

Separated from the route so the /ask path stays a thin orchestration and the
"what exactly do we record" decision lives in one readable place.
"""
import hashlib
import json
import logging
from pathlib import Path

from sqlmodel import Session, select

from backend.db_models import AnswerLog
from backend.models import Answer, AskRequest, Citation

log = logging.getLogger(__name__)

# Fields of a Citation worth keeping forever. `text` is included deliberately:
# the stored passage is the evidence. Re-reading it from the index later would
# defeat the purpose, because the index is exactly what may have changed.
CITATION_FIELDS = (
    "paper_id", "page", "unit", "chunk_index", "faiss_id",
    "score", "rerank_score", "char_start", "char_end", "text",
)


def index_fingerprint(index_dir: str | Path) -> tuple[str, int]:
    """Identify the exact index state an answer was retrieved from.

    Returns (fingerprint, n_chunks). Hashing `chunks.json` rather than the FAISS
    binary is deliberate — the vectors are derived from the chunks, so the chunk
    metadata is the thing that actually defines what could be retrieved, and it
    is stable across a FAISS rebuild that changes nothing semantically (which is
    what `remove_paper` does to every surviving chunk).
    """
    path = Path(index_dir) / "chunks.json"
    if not path.exists():
        return "", 0
    raw = path.read_bytes()
    try:
        n_chunks = len(json.loads(raw))
    except ValueError:
        n_chunks = 0
    return hashlib.sha256(raw).hexdigest()[:16], n_chunks


def _citation_rows(citations: list[Citation]) -> list[dict]:
    return [
        {field: getattr(c, field) for field in CITATION_FIELDS}
        for c in citations
    ]


def record(
    session: Session,
    user_id: int,
    workspace_id: int,
    request: AskRequest,
    query: str,
    answer: Answer,
    index_dir: str | Path,
    model: str,
    temperature: float,
) -> AnswerLog | None:
    """Persist one answered question. Returns None if logging failed.

    Swallows its own errors on purpose: losing an audit row is bad, failing the
    user's question because the audit write failed is worse. The caller has
    already produced a valid answer by this point.
    """
    # The guard covers BUILDING the row as well as writing it. Reading the index
    # fingerprint touches the filesystem and serialising citations can fail on
    # unexpected content; either would otherwise turn a perfectly good answer
    # into a 500 for the sake of bookkeeping.
    try:
        fingerprint, n_chunks = index_fingerprint(index_dir)
        entry = AnswerLog(
            user_id=user_id,
            workspace_id=workspace_id,
            question=request.question,
            query=query,
            answer=answer.answer,
            citations_json=json.dumps(_citation_rows(answer.citations), ensure_ascii=False),
            model=model,
            temperature=temperature,
            k=request.k,
            candidates=request.candidates,
            papers_filter=json.dumps(request.papers) if request.papers else None,
            index_fingerprint=fingerprint,
            n_chunks_indexed=n_chunks,
        )
        session.add(entry)
        session.commit()
        session.refresh(entry)
        return entry
    except Exception as exc:  # noqa: BLE001 — never fail a served answer
        session.rollback()
        # exc_info so a lost audit row leaves a traceback to diagnose, not just
        # a one-line regret.
        log.exception("failed to record answer for user %s: %s", user_id, exc)
        return None


def list_for_workspace(
    session: Session, workspace_id: int, limit: int, offset: int
) -> list[AnswerLog]:
    """Every answer drawn from this library, whoever asked it.

    Workspace-scoped rather than user-scoped: the point of an audit trail in a
    shared library is that a reviewer can see what the TEAM was told, not only
    what they personally asked.
    """
    return list(
        session.exec(
            select(AnswerLog)
            .where(AnswerLog.workspace_id == workspace_id)
            .order_by(AnswerLog.created_at.desc(), AnswerLog.id.desc())
            .offset(offset)
            .limit(limit)
        ).all()
    )


def reproducible_against(entry: AnswerLog, index_dir: str | Path) -> bool:
    """Whether the library is still in the state this answer was drawn from.

    False does not mean the answer was wrong — it means re-running the question
    today would search a different corpus, so any difference in the result is
    explained by the library changing rather than by the system being
    inconsistent. Saying which of those it is, is the whole point.
    """
    current, _ = index_fingerprint(index_dir)
    return bool(entry.index_fingerprint) and current == entry.index_fingerprint
