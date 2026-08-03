"""The audit trail: every answer recoverable with the evidence it was built from.

This is the product's central claim, so the tests are about whether the claim is
literally true — that the stored evidence is the evidence actually used, that a
changed library is detectable rather than silent, and that one user can never
read another's answers.
"""
import csv
import io
import json

import pytest
from fastapi.testclient import TestClient

from backend import library
from backend.tests.conftest import csrf, workspace_id

QUESTION = "What does the retriever select?"


@pytest.fixture
def fake_llm(monkeypatch):
    from backend.models import Answer

    def _generate(question, citations):
        return Answer(question=question, answer="Stubbed answer [1].", citations=citations)

    def _stream(question, citations):
        yield "Stubbed "
        yield "answer [1]."

    monkeypatch.setattr("backend.api.generate", _generate)
    monkeypatch.setattr("backend.api.stream_answer", _stream)
    monkeypatch.setattr("backend.api.condense_question", lambda q, h: q)


def _upload(client: TestClient, data: bytes, name: str = "paper.pdf"):
    return client.post("/papers", headers=csrf(client), files={"file": (name, data, "application/pdf")})


def _ask(client: TestClient, question: str = QUESTION, **body):
    return client.post("/ask", json={"question": question, **body}, headers=csrf(client))


# ——— access control ———


def test_audit_requires_auth(client: TestClient):
    assert client.get("/audit").status_code == 401


def test_empty_audit_is_an_empty_list(alice: TestClient):
    assert alice.get("/audit").json() == []


def test_missing_entry_is_404(alice: TestClient):
    assert alice.get("/audit/999").status_code == 404


# ——— recording ———


@pytest.mark.slow
def test_answering_records_an_entry(alice: TestClient, text_pdf: bytes, fake_llm):
    _upload(alice, text_pdf)
    _ask(alice)

    entries = alice.get("/audit").json()
    assert len(entries) == 1
    assert entries[0]["question"] == QUESTION
    assert entries[0]["n_citations"] > 0
    assert entries[0]["model"]


@pytest.mark.slow
def test_the_logged_evidence_is_the_evidence_that_was_served(
    alice: TestClient, text_pdf: bytes, fake_llm
):
    """The claim is that the log shows what produced the answer — so it has to
    match the citations the caller actually received, passage for passage."""
    _upload(alice, text_pdf)
    served = _ask(alice).json()

    entry_id = alice.get("/audit").json()[0]["id"]
    logged = alice.get(f"/audit/{entry_id}").json()

    assert logged["answer"] == served["answer"]
    assert len(logged["citations"]) == len(served["citations"])
    for stored, sent in zip(logged["citations"], served["citations"]):
        assert stored["text"] == sent["text"]
        assert stored["chunk_index"] == sent["chunk_index"]
        assert stored["faiss_id"] == sent["faiss_id"]
        assert stored["rerank_score"] == sent["rerank_score"]


@pytest.mark.slow
def test_retrieval_settings_are_recorded(alice: TestClient, text_pdf: bytes, fake_llm):
    """Without these, a log entry cannot explain why an answer differed."""
    _upload(alice, text_pdf)
    _ask(alice, k=2, candidates=7)

    entry_id = alice.get("/audit").json()[0]["id"]
    logged = alice.get(f"/audit/{entry_id}").json()
    assert logged["k"] == 2
    assert logged["candidates"] == 7
    assert logged["temperature"] == 0.0
    assert logged["papers_filter"] is None


@pytest.mark.slow
def test_a_document_filter_is_recorded(alice: TestClient, text_pdf: bytes, fake_llm):
    _upload(alice, text_pdf)
    _ask(alice, papers=["paper"])
    entry_id = alice.get("/audit").json()[0]["id"]
    assert alice.get(f"/audit/{entry_id}").json()["papers_filter"] == ["paper"]


@pytest.mark.slow
def test_the_condensed_query_is_kept_alongside_the_question(
    alice: TestClient, text_pdf: bytes, monkeypatch
):
    """A follow-up is rewritten before retrieval. Logging only what the user
    typed would hide what was actually searched for."""
    from backend.models import Answer

    monkeypatch.setattr("backend.api.generate",
                        lambda q, c: Answer(question=q, answer="ok [1].", citations=c))
    monkeypatch.setattr("backend.api.condense_question",
                        lambda q, h: "the standalone rewrite of the follow-up")

    _upload(alice, text_pdf)
    alice.post("/ask", json={"question": "what about it?",
                             "history": [{"question": "q", "answer": "a"}]}, headers=csrf(alice))

    logged = alice.get(f"/audit/{alice.get('/audit').json()[0]['id']}").json()
    assert logged["question"] == "what about it?"
    assert logged["query"] == "the standalone rewrite of the follow-up"


@pytest.mark.slow
def test_streaming_logs_the_text_it_actually_sent(alice: TestClient, text_pdf: bytes, fake_llm):
    """Reassembled from the deltas, not regenerated — the record must be what
    the user saw."""
    _upload(alice, text_pdf)
    alice.post("/ask/stream", json={"question": QUESTION}, headers=csrf(alice))

    entries = alice.get("/audit").json()
    assert len(entries) == 1
    logged = alice.get(f"/audit/{entries[0]['id']}").json()
    assert logged["answer"] == "Stubbed answer [1]."


@pytest.mark.slow
def test_a_failed_stream_is_not_logged_as_an_answer(alice: TestClient, text_pdf: bytes, monkeypatch):
    def _explode(question, citations):
        yield "partial "
        raise RuntimeError("ollama went away")

    monkeypatch.setattr("backend.api.stream_answer", _explode)
    monkeypatch.setattr("backend.api.condense_question", lambda q, h: q)

    _upload(alice, text_pdf)
    alice.post("/ask/stream", json={"question": QUESTION}, headers=csrf(alice))
    assert alice.get("/audit").json() == [], "a failed generation is not an answer"


@pytest.mark.slow
def test_a_failed_audit_write_does_not_fail_the_answer(
    alice: TestClient, text_pdf: bytes, fake_llm, monkeypatch
):
    """Losing an audit row is bad; failing the user's question because the audit
    write failed is worse. Breaks the row-BUILDING step, not just the commit,
    because that is the part that was originally outside the guard."""
    def _boom(*args, **kwargs):
        raise RuntimeError("serialisation is on fire")

    _upload(alice, text_pdf)
    monkeypatch.setattr("backend.audit._citation_rows", _boom)

    r = _ask(alice)
    assert r.status_code == 200, "bookkeeping must never break a served answer"
    assert r.json()["citations"], "the answer itself is unaffected"
    assert alice.get("/audit").json() == [], "and nothing was recorded"


# ——— reproducibility fingerprint ———


@pytest.mark.slow
def test_a_fresh_answer_is_marked_reproducible(alice: TestClient, text_pdf: bytes, fake_llm):
    _upload(alice, text_pdf)
    _ask(alice)
    assert alice.get("/audit").json()[0]["reproducible"] is True


@pytest.mark.slow
def test_changing_the_library_makes_an_old_answer_non_reproducible(
    alice: TestClient, text_pdf: bytes, fake_llm
):
    """The honest half of the claim. Retrieval is deterministic, but the corpus
    is not fixed — uploading a document silently changes what a question would
    return. That has to be detectable, not invisible."""
    _upload(alice, text_pdf)
    _ask(alice)
    assert alice.get("/audit").json()[0]["reproducible"] is True

    _upload(alice, text_pdf, name="second.pdf")
    assert alice.get("/audit").json()[0]["reproducible"] is False


@pytest.mark.slow
def test_the_fingerprint_is_recorded_with_the_chunk_count(
    alice: TestClient, text_pdf: bytes, fake_llm
):
    _upload(alice, text_pdf)
    _ask(alice)
    logged = alice.get(f"/audit/{alice.get('/audit').json()[0]['id']}").json()
    assert logged["index_fingerprint"]
    assert logged["n_chunks_indexed"] > 0


def test_fingerprint_of_a_missing_index_is_empty(tmp_path):
    from backend.audit import index_fingerprint

    assert index_fingerprint(tmp_path / "nope") == ("", 0)


# ——— isolation ———


@pytest.mark.slow
def test_one_user_cannot_list_anothers_answers(
    alice: TestClient, other_client: TestClient, text_pdf: bytes, fake_llm
):
    _upload(alice, text_pdf)
    _ask(alice)

    other_client.post("/auth/register",
                      json={"email": "mallory@example.com", "password": "validpassword123"})
    assert other_client.get("/audit").json() == []


@pytest.mark.slow
def test_one_user_cannot_read_anothers_answer_by_id(
    alice: TestClient, other_client: TestClient, text_pdf: bytes, fake_llm
):
    """404 not 403 — a 403 would confirm the entry exists."""
    _upload(alice, text_pdf)
    _ask(alice)
    entry_id = alice.get("/audit").json()[0]["id"]

    other_client.post("/auth/register",
                      json={"email": "mallory@example.com", "password": "validpassword123"})
    assert other_client.get(f"/audit/{entry_id}").status_code == 404
    assert other_client.get(f"/audit/{entry_id}/export").status_code == 404


# ——— export ———


@pytest.mark.slow
def test_json_export_carries_the_whole_evidence_chain(alice: TestClient, text_pdf: bytes, fake_llm):
    _upload(alice, text_pdf)
    served = _ask(alice).json()
    entry_id = alice.get("/audit").json()[0]["id"]

    r = alice.get(f"/audit/{entry_id}/export?format=json")
    assert r.status_code == 200
    assert "attachment" in r.headers["content-disposition"]

    payload = json.loads(r.content)
    assert payload["answer"] == served["answer"]
    assert len(payload["citations"]) == len(served["citations"])
    assert payload["index_fingerprint"]


@pytest.mark.slow
def test_csv_export_is_one_row_per_cited_passage(alice: TestClient, text_pdf: bytes, fake_llm):
    _upload(alice, text_pdf)
    served = _ask(alice, k=3).json()
    entry_id = alice.get("/audit").json()[0]["id"]

    r = alice.get(f"/audit/{entry_id}/export?format=csv")
    assert r.status_code == 200
    rows = list(csv.DictReader(io.StringIO(r.content.decode("utf-8"))))

    assert len(rows) == len(served["citations"])
    assert rows[0]["question"] == QUESTION
    assert rows[0]["passage"] == served["citations"][0]["text"]
    assert rows[0]["location"]  # the honest locator, e.g. "page 1"
    assert rows[0]["faiss_id"] != ""


def test_export_rejects_an_unknown_format(alice: TestClient):
    assert alice.get("/audit/1/export?format=pdf").status_code == 422
