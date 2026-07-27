"""Library routes: upload a PDF, list my papers, delete one.

All routes require the logged-in user (get_current_user); the mutating ones also
require the CSRF header (require_csrf), consistent with the auth routes. A user
can only ever see or delete their OWN papers — every query is filtered by
user_id and delete verifies ownership before touching anything.
"""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from sqlmodel import Session, select

from backend import library
from backend.auth import get_current_user, require_csrf
from backend.db import get_session
from backend.db_models import Paper, User
from backend.models import PaperPublic

router = APIRouter(prefix="/papers", tags=["papers"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB cap — sane for a research PDF


def _unique_paper_id(session: Session, user_id: int, stem: str) -> str:
    """A slug unique within this user's library (appends -2, -3, ... on clash)."""
    base = library.slugify(stem)
    taken = {
        p.paper_id
        for p in session.exec(select(Paper).where(Paper.user_id == user_id)).all()
    }
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


@router.post("", response_model=PaperPublic, status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(require_csrf)])
async def upload_paper(
    file: UploadFile,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> Paper:
    # Validate type + size before doing any expensive work.
    name = file.filename or "upload.pdf"
    if not name.lower().endswith(".pdf") and file.content_type != "application/pdf":
        raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "only PDF files are accepted")
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "file exceeds 20 MB limit")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")

    stem = Path(name).stem
    paper_id = _unique_paper_id(session, user.id, stem)

    # Persist the PDF, then index it. If indexing yields nothing (e.g. a scanned
    # PDF with no text layer), roll back the saved file and report it.
    papers_dir = library.user_papers_dir(user.id)
    papers_dir.mkdir(parents=True, exist_ok=True)
    pdf_path = papers_dir / f"{paper_id}.pdf"
    pdf_path.write_bytes(data)

    n_chunks = library.index_pdf(user.id, pdf_path, paper_id)
    if n_chunks == 0:
        pdf_path.unlink(missing_ok=True)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "no extractable text found (is this a scanned PDF?)",
        )

    paper = Paper(
        user_id=user.id, paper_id=paper_id, title=stem,
        filename=name, n_chunks=n_chunks,
    )
    session.add(paper)
    session.commit()
    session.refresh(paper)
    return paper


@router.get("", response_model=list[PaperPublic])
def list_papers(
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> list[Paper]:
    return session.exec(
        select(Paper).where(Paper.user_id == user.id).order_by(Paper.created_at.desc())
    ).all()


@router.delete("/{paper_id}", status_code=status.HTTP_204_NO_CONTENT,
               dependencies=[Depends(require_csrf)])
def delete_paper(
    paper_id: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> None:
    paper = session.get(Paper, paper_id)
    if paper is None or paper.user_id != user.id:  # ownership check
        raise HTTPException(status.HTTP_404_NOT_FOUND, "paper not found")
    library.delete_paper_data(user.id, paper.paper_id)
    session.delete(paper)
    session.commit()
