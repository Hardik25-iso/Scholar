"""Library routes: upload validation, indexing, ownership, delete.

The upload-validation tests are cheap. The ones that actually index a PDF need
the embedding model and are marked `slow`.
"""
import pytest
from fastapi.testclient import TestClient

from backend import library
from backend.parser import ocr_available
from backend.tests.conftest import csrf, workspace_id

PDF_TYPE = "application/pdf"


def _upload(client: TestClient, name: str, data: bytes, content_type: str = PDF_TYPE):
    return client.post(
        "/papers", headers=csrf(client), files={"file": (name, data, content_type)}
    )


def _index(client: TestClient, name: str, data: bytes, content_type: str = PDF_TYPE) -> dict:
    """Upload and return the FINISHED job.

    Uploading is now asynchronous: the route answers 202 with a job, and the
    outcome that used to be an HTTP status is `status` on that job. No Redis
    runs in the test environment, so indexing happens inline and the job is
    already terminal when the response arrives — the queued path is exercised
    separately in test_jobs.py.
    """
    r = _upload(client, name, data, content_type)
    assert r.status_code == 202, r.text
    return r.json()


def _paper_id(client: TestClient, title: str) -> int:
    """The database id of an indexed paper, looked up by title.

    Not taken from the upload response any more: that returns a JOB, whose id is
    its own. The two happen to collide at 1 in a fresh database, which is
    exactly the kind of coincidence that hides a bug until it doesn't.
    """
    papers = client.get("/papers").json()
    return next(p["id"] for p in papers if p["title"] == title)


# ——— validation, rejected before any expensive work ———


@pytest.mark.parametrize("name", ["photo.png", "archive.zip", "notes.rtf", "data.csv", "noextension"])
def test_unsupported_type_rejected(alice: TestClient, name):
    assert _upload(alice, name, b"hello", "application/octet-stream").status_code == 415


def test_gate_is_the_extension_not_the_client_content_type(alice: TestClient):
    """content_type is client-chosen and varies by OS for the office formats, so
    the extension is what must decide — it is also what selects the extractor."""
    # Right extension, nonsense content type -> gets past the gate, then fails on
    # content when it is actually read.
    assert _index(alice, "report.docx", b"not really a docx",
                  "application/octet-stream")["status"] == "failed"
    # Wrong extension, PDF content type -> still refused, and still up front:
    # the type gate is cheap and stays in the request.
    assert _upload(alice, "photo.png", b"%PDF-1.4", PDF_TYPE).status_code == 415


def test_empty_file_rejected(alice: TestClient):
    assert _upload(alice, "empty.pdf", b"").status_code == 400


def test_oversize_file_rejected(alice: TestClient):
    assert _upload(alice, "big.pdf", b"x" * (20 * 1024 * 1024 + 1)).status_code == 413


def test_upload_requires_csrf(alice: TestClient):
    r = alice.post("/papers", files={"file": ("a.pdf", b"%PDF-1.4", PDF_TYPE)})
    assert r.status_code == 403


# ——— the orphan-file bug ———


def test_corrupt_pdf_fails_the_job_and_never_the_server(alice: TestClient):
    """A file that claims to be a PDF but isn't. PyMuPDF raises FileDataError;
    it escaped as a 500 once, and the fix must survive indexing moving off the
    request — where there is no longer an HTTP status to carry the bad news.

    The upload itself succeeds (the file arrived); the JOB is what fails, with
    the same message the 422 used to carry.
    """
    job = _index(alice, "fake.pdf", b"not a real pdf at all")
    assert job["status"] == "failed"
    assert job["error"] == "could not read this file (corrupt or not a valid document)"


def test_corrupt_pdf_leaves_no_orphan_on_disk(alice: TestClient):
    """The file is written before indexing, so a failure must roll it back —
    otherwise the upload leaves a file with no matching database row."""
    ws = workspace_id(alice)
    _upload(alice, "fake.pdf", b"not a real pdf at all")

    papers_dir = library.workspace_papers_dir(ws)
    leftovers = list(papers_dir.glob("*.pdf")) if papers_dir.exists() else []
    assert leftovers == [], f"orphaned file(s) left behind: {leftovers}"
    assert alice.get("/papers").json() == []


def test_failed_upload_does_not_create_an_index(alice: TestClient):
    ws = workspace_id(alice)
    _upload(alice, "fake.pdf", b"not a real pdf at all")
    assert not (library.workspace_index_dir(ws) / "index.faiss").exists()


# ——— real indexing ———


@pytest.mark.slow
def test_real_pdf_indexes(alice: TestClient, text_pdf: bytes):
    job = _index(alice, "paper.pdf", text_pdf)
    assert job["status"] == "done", job
    assert job["n_chunks"] > 0
    assert job["filename"] == "paper.pdf"
    assert job["title"] == "paper"


@pytest.mark.slow
@pytest.mark.parametrize("filename,fixture", [
    ("agreement.docx", "docx_bytes"),
    ("fees.xlsx", "xlsx_bytes"),
    ("deck.pptx", "pptx_bytes"),
])
def test_office_formats_index_end_to_end(alice: TestClient, request, filename, fixture):
    """The Phase 1 headline: these were a 415 before — the product could not open
    the documents its target user actually has."""
    job = _index(alice, filename, request.getfixturevalue(fixture))
    assert job["status"] == "done", job
    assert job["n_chunks"] > 0


@pytest.mark.slow
@pytest.mark.skipif(not ocr_available(), reason="OCR unavailable here (no Tesseract, or disabled)")
def test_a_scanned_pdf_indexes_and_is_retrievable(alice: TestClient, scanned_pdf: bytes):
    """The single most common professional document — a signed, scanned contract —
    used to be rejected outright with 422."""
    from backend.store import load

    ws = workspace_id(alice)
    job = _index(alice, "signed.pdf", scanned_pdf)
    assert job["status"] == "done", job
    assert job["n_chunks"] > 0

    _, chunks = load(library.workspace_index_dir(ws))
    text = " ".join(" ".join(c.text.split()) for c in chunks)
    assert "Force Majeure" in text


def test_scanned_pdf_without_ocr_explains_the_server_cannot_do_it(
    alice: TestClient, scanned_pdf: bytes, monkeypatch
):
    """Blaming the file would be wrong — the file is fine, the server lacks OCR.

    Turns OCR off via the setting rather than by blanking the tessdata path.
    Those are not the same thing: Tesseract falls back to its own compiled-in
    data location, so on any Linux package install a blanked path still OCRs
    happily. An earlier version of this test patched the path and therefore only
    passed on Windows — it was green locally and red in CI.
    """
    from backend.config import settings

    monkeypatch.setattr(settings, "ocr_enabled", False)
    job = _index(alice, "signed.pdf", scanned_pdf)
    assert job["status"] == "failed", job
    assert "OCR is not available on this server" in job["error"]


@pytest.mark.slow
def test_markdown_indexes_end_to_end(alice: TestClient):
    body = ("Retrieval-augmented generation pairs a parametric model with a "
            "non-parametric memory. ") * 12
    job = _index(alice, "notes.md", f"# Notes\n\n{body}".encode(), "text/markdown")
    assert job["status"] == "done", job
    assert job["n_chunks"] > 0


@pytest.mark.slow
def test_uploaded_file_keeps_its_own_extension_on_disk(alice: TestClient, xlsx_bytes: bytes):
    """Storage used to hardcode `.pdf`; a workbook saved under that name would be
    unreadable by anything that trusted the extension — including the browser."""
    ws = workspace_id(alice)
    _upload(alice, "fees.xlsx", xlsx_bytes)
    assert [p.name for p in library.workspace_papers_dir(ws).glob("*")] == ["fees.xlsx"]


@pytest.mark.slow
def test_office_file_is_served_back_as_a_download(alice: TestClient, xlsx_bytes: bytes):
    """No browser renders a workbook inline — offering it as one would be a lie."""
    _index(alice, "fees.xlsx", xlsx_bytes)
    paper_id = _paper_id(alice, "fees")
    r = alice.get(f"/papers/{paper_id}/file")
    assert r.status_code == 200
    assert r.headers["content-type"].startswith(
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
    )
    assert "attachment" in r.headers["content-disposition"]
    assert r.content == xlsx_bytes


@pytest.mark.slow
def test_deleting_a_non_pdf_removes_its_file(alice: TestClient, xlsx_bytes: bytes):
    ws = workspace_id(alice)
    _index(alice, "fees.xlsx", xlsx_bytes)
    paper_id = _paper_id(alice, "fees")
    assert alice.delete(f"/papers/{paper_id}", headers=csrf(alice)).status_code == 204
    assert list(library.workspace_papers_dir(ws).glob("*")) == []


@pytest.mark.slow
def test_a_table_is_retrievable_with_its_column_labels(alice: TestClient, xlsx_bytes: bytes):
    """Serialising tables is only worth doing if the labels survive into a chunk."""
    from backend.store import load

    ws = workspace_id(alice)
    _upload(alice, "fees.xlsx", xlsx_bytes)
    _, chunks = load(library.workspace_index_dir(ws))
    text = "\n".join(c.text for c in chunks)
    assert "| Tier | Included Seats | Annual Fee |" in text
    assert "| Enterprise | unlimited | 960000 |" in text


@pytest.mark.slow
def test_indexed_paper_appears_in_the_library(alice: TestClient, text_pdf: bytes):
    _upload(alice, "paper.pdf", text_pdf)
    papers = alice.get("/papers").json()
    assert len(papers) == 1
    assert papers[0]["paper_id"] == "paper"


@pytest.mark.slow
def test_duplicate_filename_gets_a_distinct_paper_id(alice: TestClient, text_pdf: bytes):
    _upload(alice, "paper.pdf", text_pdf)
    _upload(alice, "paper.pdf", text_pdf)
    ids = sorted(p["paper_id"] for p in alice.get("/papers").json())
    assert ids == ["paper", "paper-2"]


@pytest.mark.slow
def test_paper_file_is_served_back(alice: TestClient, text_pdf: bytes):
    _index(alice, "paper.pdf", text_pdf)
    paper_id = _paper_id(alice, "paper")
    r = alice.get(f"/papers/{paper_id}/file")
    assert r.status_code == 200
    assert r.headers["content-type"] == PDF_TYPE
    assert r.content.startswith(b"%PDF")


# ——— ownership: a second tenant must see nothing ———


@pytest.mark.slow
def test_a_second_user_sees_an_empty_library(alice: TestClient, other_client: TestClient, text_pdf: bytes):
    _upload(alice, "paper.pdf", text_pdf)
    other_client.post("/auth/register", json={"email": "mallory@example.com", "password": "validpassword123"})
    assert other_client.get("/papers").json() == []


@pytest.mark.slow
def test_cross_user_file_read_is_404_not_403(alice: TestClient, other_client: TestClient, text_pdf: bytes):
    """404 rather than 403 — a 403 would confirm the paper exists."""
    _index(alice, "paper.pdf", text_pdf)
    paper_id = _paper_id(alice, "paper")
    other_client.post("/auth/register", json={"email": "mallory@example.com", "password": "validpassword123"})
    assert other_client.get(f"/papers/{paper_id}/file").status_code == 404


@pytest.mark.slow
def test_cross_user_delete_is_404(alice: TestClient, other_client: TestClient, text_pdf: bytes):
    _index(alice, "paper.pdf", text_pdf)
    paper_id = _paper_id(alice, "paper")
    other_client.post("/auth/register", json={"email": "mallory@example.com", "password": "validpassword123"})
    r = other_client.delete(f"/papers/{paper_id}", headers=csrf(other_client))
    assert r.status_code == 404
    assert len(alice.get("/papers").json()) == 1, "the paper must survive"


# ——— delete ———


@pytest.mark.slow
def test_delete_requires_csrf(alice: TestClient, text_pdf: bytes):
    _index(alice, "paper.pdf", text_pdf)
    paper_id = _paper_id(alice, "paper")
    assert alice.delete(f"/papers/{paper_id}").status_code == 403


@pytest.mark.slow
def test_delete_removes_the_row_the_file_and_the_index(alice: TestClient, text_pdf: bytes):
    ws = workspace_id(alice)
    _index(alice, "paper.pdf", text_pdf)
    paper_id = _paper_id(alice, "paper")

    assert alice.delete(f"/papers/{paper_id}", headers=csrf(alice)).status_code == 204
    assert alice.get("/papers").json() == []
    assert alice.get(f"/papers/{paper_id}/file").status_code == 404
    assert list(library.workspace_papers_dir(ws).glob("*.pdf")) == []
    # Nothing left in the library, so the store files are removed entirely.
    assert not (library.workspace_index_dir(ws) / "index.faiss").exists()


@pytest.mark.slow
def test_deleting_one_of_two_papers_keeps_the_other_indexed(alice: TestClient, text_pdf: bytes):
    """The FAISS index has no native delete — removal rebuilds it. The survivor's
    chunks must come back with contiguous ids, not a corrupt store."""
    from backend.store import load

    ws = workspace_id(alice)
    first = _upload(alice, "first.pdf", text_pdf).json()
    _upload(alice, "second.pdf", text_pdf)

    alice.delete(f"/papers/{first['id']}", headers=csrf(alice))

    index, chunks = load(library.workspace_index_dir(ws))
    assert index.ntotal == len(chunks)
    assert {c.paper_id for c in chunks} == {"second"}
    assert [c.faiss_id for c in chunks] == list(range(len(chunks)))
