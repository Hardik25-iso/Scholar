"""Backup and restore.

The central test writes a library, backs it up, DESTROYS the original, restores
it, and asserts the documents are still there and still searchable. A backup
nobody has restored is a guess, and this is the difference.
"""
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from backend import backup, library
from backend.backup import BackupError
from backend.tests.conftest import csrf, workspace_id


def _upload(client: TestClient, data: bytes, name: str = "paper.pdf"):
    return client.post("/papers", headers=csrf(client),
                       files={"file": (name, data, "application/pdf")})


# ——— archive shape ———


@pytest.mark.slow
def test_backup_contains_both_the_database_and_the_libraries(
    alice: TestClient, text_pdf: bytes, tmp_path: Path
):
    """Both halves must travel together: the database says a paper exists, the
    data tree holds its vectors. One without the other is a broken library."""
    import tarfile

    _upload(alice, text_pdf)
    archive = backup.create(tmp_path)

    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
    assert "meta.json" in names
    assert "scholar.db" in names
    assert any(n.startswith("workspaces/") for n in names)


@pytest.mark.slow
def test_a_fresh_archive_verifies(alice: TestClient, text_pdf: bytes, tmp_path: Path):
    _upload(alice, text_pdf)
    meta = backup.verify(backup.create(tmp_path))
    assert meta["format_version"] == backup.FORMAT_VERSION
    assert meta["has_database"] is True
    assert meta["n_libraries"] >= 1


def test_verify_rejects_something_that_is_not_a_backup(tmp_path: Path):
    import tarfile

    bogus = tmp_path / "bogus.tar.gz"
    payload = tmp_path / "readme.txt"
    payload.write_text("not a backup", encoding="utf-8")
    with tarfile.open(bogus, "w:gz") as tar:
        tar.add(payload, arcname="readme.txt")

    with pytest.raises(BackupError, match="not a readable Scholar backup"):
        backup.verify(bogus)


# ——— the test that matters ———


@pytest.mark.slow
def test_a_destroyed_library_is_fully_restored(alice: TestClient, text_pdf: bytes, tmp_path: Path, fake_llm):
    """Write, back up, destroy, restore, and confirm the documents still answer.

    Deleting the data tree is exactly what a redeploy did before DATA_ROOT was
    configurable, so this reproduces the disaster and then recovers from it.
    """
    import shutil

    _upload(alice, text_pdf, name="contract.pdf")
    before = alice.get("/papers").json()
    assert len(before) == 1

    archive = backup.create(tmp_path)

    # Destroy it the way a redeploy would.
    shutil.rmtree(library.DATA_ROOT, ignore_errors=True)
    library._retrievers.clear()
    assert not library.DATA_ROOT.exists()

    backup.restore(archive, force=True)

    after = alice.get("/papers").json()
    assert [p["paper_id"] for p in after] == [p["paper_id"] for p in before]
    assert after[0]["n_chunks"] == before[0]["n_chunks"]

    # Not just present on disk — actually searchable again.
    r = alice.post("/ask", json={"question": "What does the retriever select?"},
                   headers=csrf(alice))
    assert r.status_code == 200, r.text
    assert r.json()["citations"], "restored library returned no passages"


@pytest.mark.slow
def test_restore_refuses_to_clobber_live_data_without_force(
    alice: TestClient, text_pdf: bytes, tmp_path: Path
):
    """The failure mode to avoid is restoring a stale backup over a live library
    while only meaning to inspect it."""
    _upload(alice, text_pdf)
    archive = backup.create(tmp_path)

    with pytest.raises(BackupError, match="refusing to restore over existing data"):
        backup.restore(archive)


@pytest.mark.slow
def test_restore_replaces_rather_than_merges(alice: TestClient, text_pdf: bytes, tmp_path: Path):
    """A restore reproduces a moment. Merging would resurrect documents that
    were deliberately deleted after the backup was taken."""
    _upload(alice, text_pdf, name="first.pdf")
    archive = backup.create(tmp_path)

    _upload(alice, text_pdf, name="second.pdf")
    assert len(alice.get("/papers").json()) == 2

    backup.restore(archive, force=True)
    restored = [p["paper_id"] for p in alice.get("/papers").json()]
    assert restored == ["first"], f"expected only the backed-up document, got {restored}"


# ——— safety ———


def test_restore_refuses_an_archive_with_traversal_paths(tmp_path: Path, monkeypatch):
    """An archive is untrusted input — the classic tar path-traversal must not
    let it write outside the data directory."""
    import io
    import json
    import tarfile

    malicious = tmp_path / "evil.tar.gz"
    meta = json.dumps({"format_version": backup.FORMAT_VERSION, "has_database": False}).encode()
    with tarfile.open(malicious, "w:gz") as tar:
        info = tarfile.TarInfo("meta.json")
        info.size = len(meta)
        tar.addfile(info, io.BytesIO(meta))
        evil = tarfile.TarInfo("../../escaped.txt")
        evil.size = 3
        tar.addfile(evil, io.BytesIO(b"bad"))

    monkeypatch.setattr(library, "DATA_ROOT", tmp_path / "users")
    with pytest.raises(BackupError, match="unsafe paths"):
        backup.restore(malicious, force=True)


def test_backup_refuses_a_non_sqlite_database(monkeypatch, tmp_path: Path):
    """Better to refuse than to write an archive silently missing half the data."""
    from backend.config import settings

    monkeypatch.setattr(settings, "database_url", "postgresql://localhost/scholar")
    with pytest.raises(BackupError, match="not sqlite"):
        backup.create(tmp_path)


@pytest.fixture
def fake_llm(monkeypatch):
    from backend.models import Answer

    monkeypatch.setattr("backend.api.generate",
                        lambda q, c: Answer(question=q, answer="Stubbed [1].", citations=c))
    monkeypatch.setattr("backend.api.condense_question", lambda q, h: q)
