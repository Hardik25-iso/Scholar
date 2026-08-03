"""The per-user -> per-workspace migration.

The highest-risk code in the project: it moves every document on disk and
rewrites the rows that point at them. Get it wrong and a library is still on
disk but unreachable, which is data loss as far as its owner is concerned.

So these tests are about the properties that make a migration survivable —
idempotency, a dry run that really changes nothing, and never deleting an
original before the copy is verified — not only about the happy path.
"""
import json
from pathlib import Path

import pytest
from sqlmodel import Session, select

from backend import library, migrate
from backend.db import engine
from backend.db_models import Membership, Paper, User, Workspace


def _legacy_library(user_id: int, paper_id: str = "contract") -> Path:
    """Write a library where the pre-workspace code would have put it."""
    root = library.LEGACY_DATA_ROOT / str(user_id)
    (root / "papers").mkdir(parents=True, exist_ok=True)
    (root / "index").mkdir(parents=True, exist_ok=True)
    (root / "papers" / f"{paper_id}.pdf").write_bytes(b"%PDF-1.4 pretend document")
    (root / "index" / "chunks.json").write_text(
        json.dumps([{"paper_id": paper_id, "page": 0, "chunk_index": 0,
                     "text": "a passage", "embed_text": "a passage", "faiss_id": 0}]),
        encoding="utf-8",
    )
    (root / "index" / "index.faiss").write_bytes(b"not a real index, but a real file")
    return root


@pytest.fixture
def legacy_user(client):
    """A user whose library is in the OLD location and has no workspace."""
    with Session(engine) as session:
        user = User(email="legacy@example.com", hashed_password="x")
        session.add(user)
        session.commit()
        session.refresh(user)
        session.add(Paper(workspace_id=0, user_id=user.id, paper_id="contract",
                          title="contract", filename="contract.pdf", n_chunks=1))
        session.commit()
        user_id = user.id
    _legacy_library(user_id)
    return user_id


# ——— the dry run must be inert ———


def test_dry_run_changes_nothing_on_disk(legacy_user):
    source = library.LEGACY_DATA_ROOT / str(legacy_user)
    before = sorted(p.name for p in source.rglob("*"))

    migrate.migrate(dry_run=True)

    assert source.exists(), "dry run moved the original library"
    assert sorted(p.name for p in source.rglob("*")) == before
    assert not (library.DATA_ROOT / "1").exists()


def test_dry_run_changes_nothing_in_the_database(legacy_user):
    migrate.migrate(dry_run=True)
    with Session(engine) as session:
        paper = session.exec(select(Paper)).first()
        assert paper.workspace_id == 0, "dry run re-pointed a paper"
        assert session.exec(select(Workspace)).first() is None


def test_dry_run_describes_what_it_would_do(legacy_user):
    actions = "\n".join(migrate.migrate(dry_run=True))
    assert "legacy@example.com" in actions
    assert "paper(s) would be re-pointed" in actions


# ——— the migration itself ———


def test_migration_creates_a_personal_workspace_and_repoints_papers(legacy_user):
    migrate.migrate()

    with Session(engine) as session:
        membership = session.exec(
            select(Membership).where(Membership.user_id == legacy_user)
        ).first()
        assert membership is not None
        assert membership.role == "owner"

        workspace = session.get(Workspace, membership.workspace_id)
        assert workspace.is_personal is True

        paper = session.exec(select(Paper)).first()
        assert paper.workspace_id == workspace.id

        user = session.get(User, legacy_user)
        assert user.current_workspace_id == workspace.id


def test_migration_moves_the_library_to_its_workspace_directory(legacy_user):
    migrate.migrate()

    with Session(engine) as session:
        workspace_id = session.exec(
            select(Membership).where(Membership.user_id == legacy_user)
        ).first().workspace_id

    target = library.DATA_ROOT / str(workspace_id)
    assert (target / "papers" / "contract.pdf").exists()
    assert (target / "index" / "chunks.json").exists()
    assert (target / "index" / "index.faiss").exists()
    assert not (library.LEGACY_DATA_ROOT / str(legacy_user)).exists(), "original left behind"


def test_file_contents_survive_the_move(legacy_user):
    original = (library.LEGACY_DATA_ROOT / str(legacy_user) / "papers" / "contract.pdf").read_bytes()
    migrate.migrate()

    with Session(engine) as session:
        workspace_id = session.exec(
            select(Membership).where(Membership.user_id == legacy_user)
        ).first().workspace_id
    moved = library.DATA_ROOT / str(workspace_id) / "papers" / "contract.pdf"
    assert moved.read_bytes() == original


# ——— idempotency: a half-finished run must be re-runnable ———


def test_running_twice_is_safe(legacy_user):
    """A migration that dies halfway has to be finishable by running it again,
    not by hand-repairing whatever state it left."""
    migrate.migrate()
    with Session(engine) as session:
        workspaces_after_first = len(session.exec(select(Workspace)).all())
        memberships_after_first = len(session.exec(select(Membership)).all())

    migrate.migrate()

    with Session(engine) as session:
        assert len(session.exec(select(Workspace)).all()) == workspaces_after_first
        assert len(session.exec(select(Membership)).all()) == memberships_after_first
        papers = session.exec(select(Paper)).all()
        assert all(p.workspace_id != 0 for p in papers)


def test_a_second_run_does_not_disturb_an_already_moved_library(legacy_user):
    migrate.migrate()
    with Session(engine) as session:
        workspace_id = session.exec(
            select(Membership).where(Membership.user_id == legacy_user)
        ).first().workspace_id
    target = library.DATA_ROOT / str(workspace_id)
    before = sorted((p.relative_to(target), p.stat().st_size)
                    for p in target.rglob("*") if p.is_file())

    migrate.migrate()

    after = sorted((p.relative_to(target), p.stat().st_size)
                   for p in target.rglob("*") if p.is_file())
    assert after == before


def test_migrating_a_user_who_already_has_a_workspace_is_a_no_op(alice):
    """Accounts created after workspaces existed must pass through untouched."""
    with Session(engine) as session:
        before = len(session.exec(select(Workspace)).all())
    migrate.migrate()
    with Session(engine) as session:
        assert len(session.exec(select(Workspace)).all()) == before


# ——— safety ———


def test_a_failed_copy_never_deletes_the_original(legacy_user, monkeypatch):
    """Copy, verify, THEN remove. An interruption must leave both copies, never
    neither — so the verification failure has to abort before any deletion."""
    def _bad_copy(src, dst, **kwargs):
        Path(dst).mkdir(parents=True, exist_ok=True)
        (Path(dst) / "only-one-file.txt").write_text("incomplete", encoding="utf-8")

    monkeypatch.setattr(migrate.shutil, "copytree", _bad_copy)

    with pytest.raises(RuntimeError, match="copy verification FAILED"):
        migrate.migrate()

    source = library.LEGACY_DATA_ROOT / str(legacy_user)
    assert source.exists(), "the original was deleted after a failed copy"
    assert (source / "papers" / "contract.pdf").exists()


def test_the_legacy_directory_is_left_alone_if_it_still_holds_anything(legacy_user):
    """Never a recursive delete of the legacy root: a directory the migration
    did not account for must survive rather than be silently destroyed."""
    stray = library.LEGACY_DATA_ROOT / "999"
    stray.mkdir(parents=True, exist_ok=True)
    (stray / "unknown.txt").write_text("who put this here", encoding="utf-8")

    actions = "\n".join(migrate.migrate())

    assert stray.exists()
    assert (stray / "unknown.txt").exists()
    assert "left in place" in actions


def test_add_missing_columns_is_idempotent(legacy_user):
    """ALTER TABLE ADD COLUMN fails if the column is already there, so the
    guard has to actually work — this is what makes re-running safe."""
    assert migrate.add_missing_columns(dry_run=False) == []
    assert migrate.add_missing_columns(dry_run=False) == []
