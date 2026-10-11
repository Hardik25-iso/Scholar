"""The indexing work itself, separated from who asks for it.

One function, called from two places: the arq worker (normally) and the upload
request (when no queue is reachable). Written once so the two paths cannot
drift — a fallback that behaves differently from the real path is worse than no
fallback, because it is only exercised when something is already wrong.

It takes a job id rather than arguments, so both callers agree on where the
truth lives: the row. Every exit updates it.

With no queue, the inline path can run on a background thread (`submit`), so
the upload returns at once and the browser polls the job — the same function,
just not inside the request.
"""
import logging
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

from sqlmodel import Session, select

from backend import library, parser
from backend.db import engine
from backend.db_models import (
    JOB_DONE, JOB_FAILED, JOB_QUEUED, JOB_RUNNING, IndexJob, Paper,
)

log = logging.getLogger(__name__)

CORRUPT = "could not read this file (corrupt or not a valid document)"
NO_TEXT = "no extractable text found"
SCANNED = " — this looks like a scanned PDF, and OCR is not available on this server"
GONE = "the uploaded file is no longer stored"
INTERRUPTED = ("the server restarted while this document was waiting or being indexed "
               "— please upload it again")

# ONE worker: indexing is CPU- and memory-bound, and two at once on a small
# instance double peak memory for no gain. Further uploads wait their turn in
# the queue below, visible to the user as "queued".
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="index")


def submit(job_id: int) -> None:
    """Run a job on the background indexer and return immediately."""
    _executor.submit(run_index_job, job_id)


def fail_interrupted_jobs() -> int:
    """Mark jobs a dead process left queued or running as failed.

    Only valid when THIS process is the only indexer (no queue): a background
    thread's jobs die with the process — a redeploy, a crash, or a free-tier
    host putting an idle server to sleep — and without this they would show
    "indexing…" forever. With a queue, a separate worker may own them, so the
    caller must not run this.
    """
    with Session(engine) as session:
        stuck = session.exec(
            select(IndexJob).where(IndexJob.status.in_([JOB_QUEUED, JOB_RUNNING]))
        ).all()
        for job in stuck:
            job.status = JOB_FAILED
            job.error = INTERRUPTED
            job.finished_at = datetime.now(timezone.utc)
            session.add(job)
        session.commit()
        return len(stuck)


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

        try:
            doc_path = library.stored_path(job.workspace_id, job.paper_id)
            if doc_path is None:
                return _fail(session, job, GONE)
            n_chunks = library.index_document(job.workspace_id, doc_path, job.paper_id)
        except parser.DocumentTooLarge as exc:
            # A limit, not a broken file — say which one, so the user knows
            # what to change (split the document) rather than what to blame.
            _discard(job)
            return _fail(session, job, str(exc))
        except Exception:                              # noqa: BLE001 — see docstring
            log.exception("index job %s failed", job_id)
            _discard(job)
            return _fail(session, job, CORRUPT)

        if n_chunks == 0:
            _discard(job)
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
        library.discard_cached_paper(job.workspace_id, job.paper_id, job.suffix)


def _discard(job: IndexJob) -> None:
    """A document that will not be indexed is not kept: no Paper row will ever
    point at it, so it would be an orphan nobody can see or delete."""
    try:
        library.delete_stored_paper(job.workspace_id, job.paper_id)
    except Exception:                                  # noqa: BLE001
        log.warning("could not remove the stored file for job %s", job.id, exc_info=True)


def _fail(session: Session, job: IndexJob, reason: str) -> None:
    job.status = JOB_FAILED
    job.error = reason
    job.finished_at = datetime.now(timezone.utc)
    session.add(job)
    session.commit()
