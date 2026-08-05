"""The indexing work itself, separated from who asks for it.

One function, called from two places: the arq worker (normally) and the upload
request (when no queue is reachable). Written once so the two paths cannot
drift — a fallback that behaves differently from the real path is worse than no
fallback, because it is only exercised when something is already wrong.

It takes a job id rather than arguments, so both callers agree on where the
truth lives: the row. Every exit updates it.
"""
import logging
from datetime import datetime, timezone

from sqlmodel import Session

from backend import library, parser
from backend.db import engine
from backend.db_models import (
    JOB_DONE, JOB_FAILED, JOB_RUNNING, IndexJob, Paper,
)

log = logging.getLogger(__name__)

CORRUPT = "could not read this file (corrupt or not a valid document)"
NO_TEXT = "no extractable text found"
SCANNED = " — this looks like a scanned PDF, and OCR is not available on this server"


def run_index_job(job_id: int) -> None:
    """Index the document a job describes. Never raises.

    Nothing is left to catch the exception: the worker would log a traceback the
    user cannot see, and the inline caller has already answered 202. A failure
    that is not written to the row is a failure nobody can be told about, so
    every outcome — including an unexpected one — ends as a status.
    """
    with Session(engine) as session:
        job = session.get(IndexJob, job_id)
        if job is None:
            log.error("index job %s vanished before it ran", job_id)
            return
        # Refuse to re-run a finished job. arq retries on worker crash, and a
        # second run would index the same document twice into the same store.
        if job.status in (JOB_DONE, JOB_FAILED):
            return

        job.status = JOB_RUNNING
        job.started_at = datetime.now(timezone.utc)
        session.add(job)
        session.commit()

        doc_path = library.workspace_papers_dir(job.workspace_id) / f"{job.paper_id}{job.suffix}"
        try:
            if not doc_path.exists():
                return _fail(session, job, "the uploaded file is no longer on disk")
            n_chunks = library.index_document(job.workspace_id, doc_path, job.paper_id)
        except Exception as exc:                       # noqa: BLE001 — see docstring
            log.exception("index job %s failed", job_id)
            doc_path.unlink(missing_ok=True)
            return _fail(session, job, CORRUPT)

        if n_chunks == 0:
            doc_path.unlink(missing_ok=True)
            detail = NO_TEXT
            if job.suffix == ".pdf" and not parser.ocr_available():
                # Be specific rather than blaming the file: a scanned PDF is a
                # server capability gap here, not a bad upload.
                detail += SCANNED
            return _fail(session, job, detail)

        # The Paper row is created only now. Its existence means "indexed and
        # answerable" — a row written at upload time would put a document in the
        # library that retrieval cannot find.
        paper = Paper(
            workspace_id=job.workspace_id, user_id=job.user_id,
            paper_id=job.paper_id, title=job.title, filename=job.filename,
            n_chunks=n_chunks,
        )
        session.add(paper)
        job.status = JOB_DONE
        job.n_chunks = n_chunks
        job.finished_at = datetime.now(timezone.utc)
        session.add(job)
        session.commit()


def _fail(session: Session, job: IndexJob, reason: str) -> None:
    job.status = JOB_FAILED
    job.error = reason
    job.finished_at = datetime.now(timezone.utc)
    session.add(job)
    session.commit()
