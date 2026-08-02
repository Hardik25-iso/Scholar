"""Library routes: upload a document, list my papers, delete one.

All routes require the logged-in user (get_current_user); the mutating ones also
require the CSRF header (require_csrf), consistent with the auth routes. A user
can only ever see or delete their OWN papers — every query is filtered by
user_id and delete verifies ownership before touching anything.
"""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from backend import library
from backend.auth import get_current_user, require_csrf
from backend.db import get_session
from backend.db_models import Paper, User
from backend.models import PaperPublic
from backend import parser
from backend.parser import SUPPORTED_EXTENSIONS

router = APIRouter(prefix="/papers", tags=["papers"])

MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # 20 MB cap — sane for a research document

# Served back to the browser on GET /{id}/file. The office formats are
# downloads, not renderable documents, so only PDF and text open in place.
MEDIA_TYPES = {
    ".pdf": "application/pdf",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
    ".pptx": "application/vnd.openxmlformats-officedocument.presentationml.presentation",
    ".xlsx": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    ".txt": "text/plain; charset=utf-8",
    ".md": "text/markdown; charset=utf-8",
}
INLINE_EXTENSIONS = {".pdf", ".txt", ".md"}


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
    # Validate type + size before doing any expensive work. The EXTENSION is the
    # gate, not the browser-supplied content type: content_type is client-chosen
    # and varies by OS for the office formats, while the extension is what
    # actually selects the extractor downstream.
    name = file.filename or "upload.pdf"
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            f"unsupported file type — accepted: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "file exceeds 20 MB limit")
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")

    stem = Path(name).stem
    paper_id = _unique_paper_id(session, user.id, stem)

    # Persist the file, then index it. Indexing has two failure modes and BOTH
    # must roll the saved file back, or the upload leaves an orphan on disk with
    # no matching DB row:
    #   - it raises (corrupt/unreadable file — PyMuPDF raises FileDataError),
    #   - it succeeds but yields nothing (a scan we could not OCR, or an empty file).
    papers_dir = library.user_papers_dir(user.id)
    papers_dir.mkdir(parents=True, exist_ok=True)
    doc_path = papers_dir / f"{paper_id}{suffix}"
    doc_path.write_bytes(data)

    try:
        n_chunks = library.index_document(user.id, doc_path, paper_id)
    except Exception as exc:
        # index_document writes to the FAISS store only as its last step, so a
        # failure here leaves the user's index untouched — only the file needs undoing.
        doc_path.unlink(missing_ok=True)
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "could not read this file (corrupt or not a valid document)",
        ) from exc

    if n_chunks == 0:
        doc_path.unlink(missing_ok=True)
        detail = "no extractable text found"
        if suffix == ".pdf" and not parser.ocr_available():
            # Be specific rather than blaming the file: a scanned PDF is a
            # server capability gap here, not a bad upload.
            detail += " — this looks like a scanned PDF, and OCR is not available on this server"
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, detail)

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


@router.get("/{paper_id}/file")
def get_paper_file(
    paper_id: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> FileResponse:
    """Serve a user's stored document (for the in-app source viewer).

    Ownership-checked like every library route. PDFs and plain text are served
    inline so the browser renders them in an iframe (the viewer appends #page=N
    to jump to a cited page); the office formats have no in-browser renderer, so
    they are served as attachments rather than pretending otherwise. Auth is the
    same-site cookie — no CSRF needed on a GET.
    """
    paper = session.get(Paper, paper_id)
    if paper is None or paper.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "paper not found")
    path = library.stored_path(user.id, paper.paper_id)
    if path is None or not path.exists():
        raise HTTPException(status.HTTP_404_NOT_FOUND, "file no longer on disk")
    suffix = path.suffix.lower()
    return FileResponse(
        path,
        media_type=MEDIA_TYPES.get(suffix, "application/octet-stream"),
        filename=paper.filename,
        content_disposition_type="inline" if suffix in INLINE_EXTENSIONS else "attachment",
    )


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
