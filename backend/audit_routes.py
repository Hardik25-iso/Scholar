"""Audit routes: list past answers, inspect one, export its evidence chain.

    GET /audit                  -> my answers, newest first
    GET /audit/{id}             -> one answer with every passage it used
    GET /audit/{id}/export      -> the same as a downloadable JSON or CSV file

Read-only and scoped to the caller, like every library route: a missing entry
and someone else's entry both return 404, so the response never confirms that
another user's answer exists.
"""
import csv
import io
import json

from fastapi import APIRouter, Depends, HTTPException, Query, status
from fastapi.responses import StreamingResponse
from sqlmodel import Session

from backend import audit, library
from backend.auth import get_current_user
from backend.db import get_session
from backend.db_models import AnswerLog, User
from backend.models import AnswerLogDetail, AnswerLogSummary, Citation

router = APIRouter(prefix="/audit", tags=["audit"])


def _citations(entry: AnswerLog) -> list[Citation]:
    return [Citation(**row) for row in json.loads(entry.citations_json)]


def _summary(entry: AnswerLog, reproducible: bool) -> AnswerLogSummary:
    return AnswerLogSummary(
        id=entry.id,
        created_at=entry.created_at,
        question=entry.question,
        n_citations=len(json.loads(entry.citations_json)),
        model=entry.model,
        reproducible=reproducible,
    )


def _owned_entry(entry_id: int, user: User, session: Session) -> AnswerLog:
    entry = session.get(AnswerLog, entry_id)
    if entry is None or entry.user_id != user.id:  # 404, not 403 — hide existence
        raise HTTPException(status.HTTP_404_NOT_FOUND, "audit entry not found")
    return entry


@router.get("", response_model=list[AnswerLogSummary])
def list_answers(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> list[AnswerLogSummary]:
    index_dir = library.user_index_dir(user.id)
    entries = audit.list_for_user(session, user.id, limit, offset)
    return [_summary(e, audit.reproducible_against(e, index_dir)) for e in entries]


@router.get("/{entry_id}", response_model=AnswerLogDetail)
def get_answer(
    entry_id: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> AnswerLogDetail:
    entry = _owned_entry(entry_id, user, session)
    reproducible = audit.reproducible_against(entry, library.user_index_dir(user.id))
    return AnswerLogDetail(
        **_summary(entry, reproducible).model_dump(),
        query=entry.query,
        answer=entry.answer,
        citations=_citations(entry),
        temperature=entry.temperature,
        k=entry.k,
        candidates=entry.candidates,
        papers_filter=json.loads(entry.papers_filter) if entry.papers_filter else None,
        index_fingerprint=entry.index_fingerprint,
        n_chunks_indexed=entry.n_chunks_indexed,
    )


@router.get("/{entry_id}/export")
def export_answer(
    entry_id: int,
    format: str = Query(default="json", pattern="^(json|csv)$"),
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> StreamingResponse:
    """Download one answer with its full evidence chain.

    JSON keeps the whole record including the answer text and retrieval settings.
    CSV is one row per cited passage, for the reviewer who wants the evidence in
    a spreadsheet — it repeats the question on every row rather than needing a
    header block, so the file survives being sorted or filtered.
    """
    entry = _owned_entry(entry_id, user, session)
    citations = _citations(entry)
    reproducible = audit.reproducible_against(entry, library.user_index_dir(user.id))

    if format == "json":
        payload = get_answer(entry_id, user, session).model_dump(mode="json")
        body = json.dumps(payload, indent=2, ensure_ascii=False)
        media_type = "application/json"
    else:
        buffer = io.StringIO()
        writer = csv.writer(buffer, lineterminator="\n")
        writer.writerow([
            "answer_id", "asked_at", "question", "answer", "reproducible_now",
            "source_n", "document", "location", "chunk_index", "faiss_id",
            "char_start", "char_end", "retrieval_score", "rerank_score", "passage",
        ])
        for n, c in enumerate(citations, start=1):
            writer.writerow([
                entry.id, entry.created_at.isoformat(), entry.question, entry.answer,
                reproducible, n, c.paper_id, c.locator, c.chunk_index,
                c.faiss_id if c.faiss_id is not None else "",
                c.char_start, c.char_end, f"{c.score:.6f}",
                "" if c.rerank_score is None else f"{c.rerank_score:.6f}",
                c.text,
            ])
        body = buffer.getvalue()
        media_type = "text/csv; charset=utf-8"

    filename = f"scholar-answer-{entry.id}.{format}"
    return StreamingResponse(
        iter([body]),
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
