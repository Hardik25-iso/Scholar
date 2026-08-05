"""The arq queue: enqueueing from the API, and the worker that drains it.

Two processes now, not one:

    uvicorn backend.api:app          # the API
    arq backend.jobs.WorkerSettings  # the indexer

That is the cost of moving indexing out of the request, and it is a real
deployment change — a deploy that starts only the API will accept uploads and
index none of them, which is why `redis_available()` exists and why the API
falls back to indexing inline rather than queueing into a void.

The worker loads the embedding and reranking models, so it wants the same
resources the API does. Run one; scale up only when the queue actually backs up.
"""
import logging

from arq import create_pool
from arq.connections import RedisSettings

from backend.config import settings

log = logging.getLogger(__name__)

INDEX_TASK = "index_document_task"


def redis_settings() -> RedisSettings:
    return RedisSettings.from_dsn(settings.redis_url)


async def index_document_task(ctx: dict, job_id: int) -> None:
    """Worker entry point. Deliberately thin — the work lives in `indexing` so
    the inline fallback runs exactly the same code."""
    from backend.indexing import run_index_job

    # run_index_job is synchronous and CPU-bound (parse, OCR, embed). arq runs
    # one job at a time per worker by default, so blocking the loop here is
    # honest rather than harmful: this worker IS the thing doing the work.
    run_index_job(job_id)


async def enqueue_index_job(job_id: int) -> bool:
    """Hand a job to the queue. Returns whether it was accepted.

    False means Redis was unreachable — not that the job is invalid. The caller
    then runs it inline and records that it did, because an upload that
    disappears into an unreachable queue looks identical to one that worked.
    """
    if not settings.redis_url:
        return False
    try:
        pool = await create_pool(redis_settings())
        try:
            await pool.enqueue_job(INDEX_TASK, job_id)
        finally:
            await pool.aclose()
    except Exception as exc:                            # noqa: BLE001
        log.warning("could not enqueue index job %s (%s) — indexing inline", job_id, exc)
        return False
    return True


class WorkerSettings:
    """`arq backend.jobs.WorkerSettings` picks this up."""

    functions = [index_document_task]
    max_jobs = 1                # indexing is CPU-bound; concurrency buys nothing
    job_timeout = 60 * 30       # OCR on a long scanned document is genuinely slow
    keep_result = 0             # the IndexJob row is the record, not Redis

    @staticmethod
    def redis_settings() -> RedisSettings:
        return RedisSettings.from_dsn(settings.redis_url)
