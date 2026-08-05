"""The indexing queue: what happens between accepting an upload and answering
questions about it.

Indexing used to run inside the upload request, so a large document raced the
proxy timeout. Moving it out buys that back and costs a guarantee: the request
can no longer report the outcome. These tests are about the thing that replaces
it — the job row — and about the two ways the queue can be absent.

Redis is faked. arq's own reliability is not under test; the contract between
this app and the queue is.
"""
import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from backend import jobs
from backend.config import settings
from backend.db import engine
from backend.db_models import JOB_DONE, JOB_FAILED, JOB_QUEUED, IndexJob, Paper
from backend.tests.conftest import csrf, workspace_id

PDF = "application/pdf"


def _upload(client: TestClient, name: str, data: bytes):
    return client.post("/papers", headers=csrf(client), files={"file": (name, data, PDF)})


# ——— the response shape ———


def test_upload_answers_with_a_job_not_a_paper(alice: TestClient):
    r = _upload(alice, "fake.pdf", b"not a pdf")
    assert r.status_code == 202
    body = r.json()
    # The job's own fields, not a Paper's. A Paper only exists once indexing
    # succeeded — one that appeared at upload time would be a document in the
    # library that retrieval cannot find.
    assert set(body) >= {"id", "status", "error", "ran_inline", "paper_id", "filename"}


def test_a_failed_job_reports_the_reason_the_422_used_to_carry(alice: TestClient):
    job = _upload(alice, "fake.pdf", b"not a pdf").json()
    assert job["status"] == JOB_FAILED
    assert "corrupt or not a valid document" in job["error"]
    assert alice.get("/papers").json() == [], "a failed job must not leave a paper behind"


# ——— polling ———


def test_a_job_can_be_polled_by_id(alice: TestClient):
    job = _upload(alice, "fake.pdf", b"not a pdf").json()
    r = alice.get(f"/papers/jobs/{job['id']}")
    assert r.status_code == 200
    assert r.json()["id"] == job["id"]


def test_a_job_from_another_workspace_is_404(alice: TestClient, other_client: TestClient):
    """A job id must not be a way to learn what another workspace is uploading."""
    job = _upload(alice, "fake.pdf", b"not a pdf").json()
    other_client.post("/auth/register",
                      json={"email": "mallory@example.com", "password": "validpassword123"})
    assert other_client.get(f"/papers/jobs/{job['id']}").status_code == 404


def test_unfinished_jobs_are_listed_so_a_reload_can_resume(alice: TestClient, monkeypatch):
    """Closing the tab must not lose track of an upload still being indexed."""
    ws = workspace_id(alice)
    with Session(engine) as session:
        session.add(IndexJob(workspace_id=ws, user_id=1, paper_id="pending",
                             filename="pending.pdf", title="pending", suffix=".pdf",
                             status=JOB_QUEUED))
        session.commit()

    listed = alice.get("/papers/jobs").json()
    assert [j["status"] for j in listed] == [JOB_QUEUED]


def test_finished_jobs_are_not_listed_as_pending(alice: TestClient):
    _upload(alice, "fake.pdf", b"not a pdf")       # completes (as failed) inline
    assert alice.get("/papers/jobs").json() == []


# ——— the queue, and its absence ———


def test_without_redis_the_work_happens_inline_and_says_so(alice: TestClient):
    """A silent fallback would hide that the deployment is missing its worker —
    and with it the proxy-timeout risk this whole feature removes."""
    assert settings.redis_url == "", "tests must not reach a real queue"
    job = _upload(alice, "fake.pdf", b"not a pdf").json()
    assert job["ran_inline"] is True
    assert job["status"] == JOB_FAILED, "inline means finished, not still queued"


@pytest.mark.slow
def test_when_the_queue_accepts_the_request_does_not_index(alice: TestClient, text_pdf: bytes):
    """The point of the queue: the request returns before the work is done."""
    enqueued: list[int] = []

    async def fake_enqueue(job_id: int) -> bool:
        enqueued.append(job_id)
        return True

    import backend.papers as papers_module
    original = papers_module.enqueue_index_job
    papers_module.enqueue_index_job = fake_enqueue
    try:
        job = _upload(alice, "paper.pdf", text_pdf).json()
    finally:
        papers_module.enqueue_index_job = original

    assert enqueued == [job["id"]]
    assert job["status"] == JOB_QUEUED
    assert job["ran_inline"] is False
    assert alice.get("/papers").json() == [], "nothing should be indexed yet"


@pytest.mark.slow
def test_the_worker_finishes_what_the_queue_accepted(alice: TestClient, text_pdf: bytes):
    """Same code the inline path runs — that is why there is only one of it."""
    from backend.indexing import run_index_job

    async def fake_enqueue(job_id: int) -> bool:
        return True

    import backend.papers as papers_module
    original = papers_module.enqueue_index_job
    papers_module.enqueue_index_job = fake_enqueue
    try:
        job = _upload(alice, "paper.pdf", text_pdf).json()
    finally:
        papers_module.enqueue_index_job = original

    run_index_job(job["id"])                       # what the arq worker calls

    done = alice.get(f"/papers/jobs/{job['id']}").json()
    assert done["status"] == JOB_DONE, done
    assert done["n_chunks"] > 0
    assert [p["title"] for p in alice.get("/papers").json()] == ["paper"]


@pytest.mark.slow
def test_running_a_finished_job_again_does_not_index_it_twice(alice: TestClient, text_pdf: bytes):
    """arq retries on worker crash. A second run would index the same document
    into the same store, and every answer would then cite it twice."""
    from backend.indexing import run_index_job

    job = _upload(alice, "paper.pdf", text_pdf).json()   # runs inline, ends done
    assert job["status"] == JOB_DONE

    run_index_job(job["id"])                             # the retry

    with Session(engine) as session:
        papers = session.exec(select(Paper)).all()
    assert len(papers) == 1, "the retry created a second Paper row"


def test_an_unreachable_redis_falls_back_rather_than_raising(monkeypatch):
    """`enqueue` must never propagate a connection error into the request: the
    upload has already been accepted and the file is already on disk."""
    monkeypatch.setattr(settings, "redis_url", "redis://127.0.0.1:6399")  # nothing there
    import asyncio

    assert asyncio.run(jobs.enqueue_index_job(1)) is False


def test_no_redis_url_configured_is_not_an_error(monkeypatch):
    monkeypatch.setattr(settings, "redis_url", "")
    import asyncio

    assert asyncio.run(jobs.enqueue_index_job(1)) is False
