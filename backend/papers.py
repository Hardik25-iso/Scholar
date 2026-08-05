"""Library routes: upload a document, list my papers, delete one.

All routes require the logged-in user (get_current_user); the mutating ones also
require the CSRF header (require_csrf), consistent with the auth routes.

Scope is the WORKSPACE, resolved by the get_current_workspace dependency, which
proves membership before the route body runs. A document is visible to everyone
in its workspace and invisible outside it — so the check is "are you a member",
not "did you upload this". Deletion is narrower: see delete_paper.
"""
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import FileResponse
from sqlmodel import Session, select

from backend import library
from backend.auth import get_current_user, require_csrf
from backend.db import get_session
from backend.db_models import (
    JOB_QUEUED, JOB_RUNNING, ROLE_OWNER, IndexJob, Paper, User, Workspace,
)
from backend.indexing import run_index_job
from backend.jobs import enqueue_index_job
from backend.models import IndexJobPublic, PaperPublic
from backend.parser import SUPPORTED_EXTENSIONS
from backend.ratelimit import upload_limiter
from backend.workspaces import get_current_workspace, require_membership

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


def _unique_paper_id(session: Session, workspace_id: int, stem: str) -> str:
    """A slug unique within this WORKSPACE (appends -2, -3, ... on clash).

    Workspace-scoped rather than user-scoped: two members uploading files with
    the same name must not collide on one stored path or one chunk tag.
    """
    base = library.slugify(stem)
    taken = {
        p.paper_id
        for p in session.exec(select(Paper).where(Paper.workspace_id == workspace_id)).all()
    }
    if base not in taken:
        return base
    n = 2
    while f"{base}-{n}" in taken:
        n += 1
    return f"{base}-{n}"


@router.get("/jobs/{job_id}", response_model=IndexJobPublic)
def get_index_job(
    job_id: int,
    workspace: Workspace = Depends(get_current_workspace),
    session: Session = Depends(get_session),
) -> IndexJob:
    """How an upload is getting on.

    Declared before the /{paper_id} routes. Nothing collides today — their
    shapes differ — but FastAPI matches in declaration order, so adding a plain
    `GET /papers/{paper_id}` later would shadow `/papers/jobs` if these came
    after it. Cheap to get right now, confusing to debug then.

    Scoped to the workspace like everything else: a job id must not be a way to
    learn what another workspace is uploading.
    """
    job = session.get(IndexJob, job_id)
    if job is None or job.workspace_id != workspace.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "job not found")
    return job


@router.get("/jobs", response_model=list[IndexJobPublic])
def list_index_jobs(
    workspace: Workspace = Depends(get_current_workspace),
    session: Session = Depends(get_session),
) -> list[IndexJob]:
    """Unfinished uploads in this workspace.

    Exists so a reloaded page can pick work back up. Without it, closing the tab
    loses track of an upload that is still indexing, and the document appears
    later with no explanation of where it came from.
    """
    return session.exec(
        select(IndexJob)
        .where(IndexJob.workspace_id == workspace.id,
               IndexJob.status.in_([JOB_QUEUED, JOB_RUNNING]))
        .order_by(IndexJob.created_at.desc())
    ).all()


@router.post("", response_model=IndexJobPublic, status_code=status.HTTP_202_ACCEPTED,
             dependencies=[Depends(require_csrf)])
async def upload_paper(
    file: UploadFile,
    user: User = Depends(get_current_user),
    workspace: Workspace = Depends(get_current_workspace),
    session: Session = Depends(get_session),
) -> IndexJob:
    # Indexing is the second-most expensive thing this app does (parse, OCR,
    # embed). Charged before reading the body, so a rate-limited caller does not
    # get to make the server buffer 20 MB first.
    upload_limiter.check(str(user.id))

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
    paper_id = _unique_paper_id(session, workspace.id, stem)

    # The file is stored before the job is queued, because the job refers to it
    # by path. A stored file with no job would be an orphan, so the two are
    # written in the order that makes the job the thing that owns cleanup.
    papers_dir = library.workspace_papers_dir(workspace.id)
    papers_dir.mkdir(parents=True, exist_ok=True)
    doc_path = papers_dir / f"{paper_id}{suffix}"
    doc_path.write_bytes(data)

    job = IndexJob(
        workspace_id=workspace.id, user_id=user.id,
        paper_id=paper_id, filename=name, title=stem, suffix=suffix,
    )
    session.add(job)
    session.commit()
    session.refresh(job)

    # Indexing (parse, OCR, embed) is what used to race the proxy timeout, so it
    # leaves the request here. If no queue is reachable we do it inline anyway
    # rather than accepting an upload nothing will ever process — and the job
    # records that it happened that way, because the timeout risk came back with
    # it and a silent difference in behaviour is the kind that bites at 2am.
    if not await enqueue_index_job(job.id):
        job.ran_inline = True
        session.add(job)
        session.commit()
        run_index_job(job.id)
        session.refresh(job)

    return job


@router.get("", response_model=list[PaperPublic])
def list_papers(
    workspace: Workspace = Depends(get_current_workspace),
    session: Session = Depends(get_session),
) -> list[Paper]:
    return session.exec(
        select(Paper)
        .where(Paper.workspace_id == workspace.id)
        .order_by(Paper.created_at.desc())
    ).all()


@router.get("/{paper_id}/file")
def get_paper_file(
    paper_id: int,
    workspace: Workspace = Depends(get_current_workspace),
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
    if paper is None or paper.workspace_id != workspace.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "paper not found")
    path = library.stored_path(workspace.id, paper.paper_id)
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
    workspace: Workspace = Depends(get_current_workspace),
    session: Session = Depends(get_session),
) -> None:
    """Remove a document from the workspace library.

    Deletion is deliberately NARROWER than reading. Everyone in a workspace can
    see every document, but only the person who uploaded it or a workspace owner
    can destroy it — otherwise any member could silently delete a colleague's
    work from a shared library, and there is no undo.
    """
    paper = session.get(Paper, paper_id)
    if paper is None or paper.workspace_id != workspace.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "paper not found")

    membership = require_membership(session, user, workspace.id)
    if paper.user_id != user.id and membership.role != ROLE_OWNER:
        # 403, not 404: the caller can already see this document in the list, so
        # hiding it would turn a permission error into an apparent bug.
        raise HTTPException(
            status.HTTP_403_FORBIDDEN,
            "only the member who uploaded this document, or a workspace owner, can delete it",
        )

    library.delete_paper_data(workspace.id, paper.paper_id)
    session.delete(paper)
    session.commit()
