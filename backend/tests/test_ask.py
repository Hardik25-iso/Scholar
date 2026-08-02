"""The /ask and /ask/stream contract: auth, CSRF, scoping, NDJSON framing.

Retrieval and reranking run FOR REAL here — that is the part worth testing at
this level. Generation is stubbed, because an LLM's prose is not a thing an
assertion can pin down, and requiring Ollama would make the suite unrunnable in
CI. Whether the model actually stays grounded and refuses correctly is measured
by the eval harness (`python -m backend.eval`), not here.
"""
import json

import pytest
from fastapi.testclient import TestClient

from backend.tests.conftest import csrf

QUESTION = "What does the retriever select?"


@pytest.fixture
def fake_llm(monkeypatch):
    """Replace generation with a deterministic stub, keeping retrieval real."""
    from backend.models import Answer

    def _generate(question, citations):
        return Answer(question=question, answer="Stubbed answer [1].", citations=citations)

    def _stream(question, citations):
        yield "Stubbed "
        yield "answer "
        yield "[1]."

    monkeypatch.setattr("backend.api.generate", _generate)
    monkeypatch.setattr("backend.api.stream_answer", _stream)
    monkeypatch.setattr("backend.api.condense_question", lambda q, h: q)


def _upload(client: TestClient, data: bytes):
    return client.post("/papers", headers=csrf(client), files={"file": ("paper.pdf", data, "application/pdf")})


def _ask(client: TestClient, **body) -> dict:
    r = client.post("/ask", json={"question": QUESTION, **body}, headers=csrf(client))
    assert r.status_code == 200, r.text
    return r.json()


# ——— auth and CSRF ———


def test_ask_requires_csrf(alice: TestClient):
    """/ask is the most expensive route in the app; the cookie alone must not be
    enough to drive a logged-in user's browser into unbounded compute."""
    assert alice.post("/ask", json={"question": QUESTION}).status_code == 403


def test_ask_stream_requires_csrf(alice: TestClient):
    assert alice.post("/ask/stream", json={"question": QUESTION}).status_code == 403


def test_ask_with_no_papers_is_a_clear_400(alice: TestClient):
    r = alice.post("/ask", json={"question": QUESTION}, headers=csrf(alice))
    assert r.status_code == 400
    assert "upload" in r.json()["detail"].lower()


# ——— the real path ———


@pytest.mark.slow
def test_ask_returns_an_answer_with_scored_citations(alice: TestClient, text_pdf: bytes, fake_llm):
    _upload(alice, text_pdf)
    r = alice.post("/ask", json={"question": QUESTION, "k": 3}, headers=csrf(alice))
    assert r.status_code == 200, r.text

    body = r.json()
    assert body["question"] == QUESTION
    assert body["citations"], "retrieval returned nothing"
    assert len(body["citations"]) <= 3
    for c in body["citations"]:
        assert c["text"].strip()
        assert c["paper_id"] == "paper"
        assert c["page"] >= 0
        assert c["rerank_score"] is not None, "stage-2 score missing"


@pytest.mark.slow
def test_citations_carry_their_audit_trail(alice: TestClient, text_pdf: bytes, fake_llm):
    """faiss_id ties a passage to the exact indexed vector, and the char span
    locates it precisely — both were computed and then dropped before Phase 3."""
    _upload(alice, text_pdf)
    cites = _ask(alice)["citations"]

    for c in cites:
        assert c["faiss_id"] is not None, "faiss_id dropped on the way out"
        assert c["char_end"] > c["char_start"], "empty char span"
        assert len(c["text"]) == c["char_end"] - c["char_start"]


@pytest.mark.slow
@pytest.mark.parametrize("filename,fixture,unit", [
    ("paper.pdf", "text_pdf", "page"),
    ("deck.pptx", "pptx_bytes", "slide"),
    ("fees.xlsx", "xlsx_bytes", "sheet"),
    ("agreement.docx", "docx_bytes", "section"),
])
def test_a_citation_names_its_location_truthfully(
    alice: TestClient, request, fake_llm, filename, fixture, unit
):
    """Calling a slide or a worksheet "page 3" is a small lie, and this product's
    claim is that a citation can be read literally."""
    data = request.getfixturevalue(fixture)
    ctype = "application/pdf" if filename.endswith(".pdf") else "application/octet-stream"
    alice.post("/papers", headers=csrf(alice), files={"file": (filename, data, ctype)})

    cites = _ask(alice)["citations"]
    assert cites, "nothing retrieved"
    assert cites[0]["unit"] == unit
    assert cites[0]["locator"] == f"{unit} {cites[0]['page'] + 1}"


@pytest.mark.slow
def test_citations_come_back_in_rerank_order(alice: TestClient, text_pdf: bytes, fake_llm):
    """The generator numbers sources [1], [2], ... in list order, so that order
    must already be best-first or the numbering is meaningless."""
    _upload(alice, text_pdf)
    cites = alice.post(
        "/ask", json={"question": QUESTION, "k": 5}, headers=csrf(alice)
    ).json()["citations"]
    scores = [c["rerank_score"] for c in cites]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.slow
def test_k_bounds_how_many_sources_reach_the_llm(alice: TestClient, text_pdf: bytes, fake_llm):
    _upload(alice, text_pdf)
    for k in (1, 2):
        cites = alice.post(
            "/ask", json={"question": QUESTION, "k": k, "candidates": 10}, headers=csrf(alice)
        ).json()["citations"]
        assert len(cites) == k


@pytest.mark.slow
def test_a_second_user_retrieves_none_of_the_first_users_passages(
    alice: TestClient, other_client: TestClient, text_pdf: bytes, fake_llm
):
    """The strongest isolation assertion: not a 403, but zero leaked evidence."""
    _upload(alice, text_pdf)
    other_client.post(
        "/auth/register", json={"email": "mallory@example.com", "password": "validpassword123"}
    )
    r = other_client.post("/ask", json={"question": QUESTION}, headers=csrf(other_client))

    assert r.status_code == 400, "an empty library must not fall through to someone else's index"
    if r.status_code == 200:  # defensive: if that ever changes, still assert no leak
        assert r.json()["citations"] == []


# ——— NDJSON streaming protocol ———


@pytest.mark.slow
def test_stream_sends_citations_first_then_tokens_then_done(
    alice: TestClient, text_pdf: bytes, fake_llm
):
    _upload(alice, text_pdf)
    r = alice.post("/ask/stream", json={"question": QUESTION}, headers=csrf(alice))
    assert r.status_code == 200

    frames = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    kinds = [f["type"] for f in frames]

    assert kinds[0] == "citations", "sources must arrive before any prose"
    assert kinds[-1] == "done"
    assert "error" not in kinds
    assert kinds.count("token") >= 3
    assert frames[0]["citations"], "the citations frame was empty"
    assert "".join(f["text"] for f in frames if f["type"] == "token") == "Stubbed answer [1]."


@pytest.mark.slow
def test_stream_reports_a_mid_stream_failure_as_an_error_frame(
    alice: TestClient, text_pdf: bytes, monkeypatch
):
    """Citations are already on the wire when generation dies, so the failure has
    to be delivered in-band rather than as an HTTP status."""
    _upload(alice, text_pdf)

    def _explode(question, citations):
        yield "partial "
        raise RuntimeError("ollama went away")

    monkeypatch.setattr("backend.api.stream_answer", _explode)
    monkeypatch.setattr("backend.api.condense_question", lambda q, h: q)

    r = alice.post("/ask/stream", json={"question": QUESTION}, headers=csrf(alice))
    frames = [json.loads(line) for line in r.text.splitlines() if line.strip()]
    kinds = [f["type"] for f in frames]

    assert kinds[0] == "citations"
    assert kinds[-1] == "error"
    assert "done" not in kinds
    assert "ollama went away" in frames[-1]["detail"]
