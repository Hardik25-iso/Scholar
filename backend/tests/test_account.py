"""Account export and deletion.

Deletion is the one feature where passing tests are not enough on their own —
"the row is gone" and "the bytes are gone" are different claims, and only the
second is what a privacy policy promises. These tests check the disk.
"""
import io
import json
import tarfile

import pytest
from fastapi.testclient import TestClient
from sqlmodel import Session, select

from backend import library
from backend.db import engine
from backend.db_models import AnswerLog, Membership, Paper, RefreshToken, User, Workspace
from backend.tests.conftest import csrf, workspace_id

GOOD = {"email": "alice@example.com", "password": "validpassword123"}
OTHER = {"email": "mallory@example.com", "password": "validpassword123"}


def _upload(client: TestClient, data: bytes, name: str = "paper.pdf"):
    return client.post("/papers", headers=csrf(client),
                       files={"file": (name, data, "application/pdf")})


def _members(session: Session, ws_id: int) -> list[Membership]:
    return list(session.exec(select(Membership).where(Membership.workspace_id == ws_id)).all())


# ——— export ———


def test_export_is_a_real_archive_naming_the_account(alice: TestClient):
    r = alice.get("/auth/export")
    assert r.status_code == 200
    assert r.headers["content-type"] == "application/gzip"
    assert "attachment" in r.headers["content-disposition"]

    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as archive:
        manifest = json.load(archive.extractfile("account.json"))
    assert manifest["account"]["email"] == GOOD["email"]


def test_export_requires_a_session(client: TestClient):
    assert client.get("/auth/export").status_code == 401


@pytest.mark.slow
def test_export_contains_the_documents_not_just_a_list_of_them(alice: TestClient, text_pdf: bytes):
    """An export that names your files without including them is not an export."""
    _upload(alice, text_pdf, "contract.pdf")

    r = alice.get("/auth/export")
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as archive:
        names = archive.getnames()
        manifest = json.load(archive.extractfile("account.json"))
        stored = archive.extractfile(manifest["documents"][0]["file"]).read()

    assert manifest["documents"][0]["filename"] == "contract.pdf"
    assert any(n.endswith("contract.pdf") for n in names)
    assert stored == text_pdf, "the exported file is not byte-identical to the upload"


@pytest.mark.slow
def test_export_does_not_take_a_colleagues_documents(
    alice: TestClient, other_client: TestClient, text_pdf: bytes
):
    """Shared access is not ownership. "Export my data" must not mean "export
    everything I can currently read"."""
    created = alice.post("/workspaces", json={"name": "Acme Legal"},
                         headers=csrf(alice)).json()
    token = alice.post(f"/workspaces/{created['id']}/invitations",
                       json={"email": OTHER["email"]}, headers=csrf(alice)).json()["token"]
    other_client.post("/auth/register", json=OTHER)
    other_client.post("/workspaces/accept", json={"token": token}, headers=csrf(other_client))

    # Alice uploads into the shared workspace; Mallory can read it but did not
    # upload it.
    alice.post(f"/workspaces/{created['id']}/activate", headers=csrf(alice))
    _upload(alice, text_pdf, "alices_contract.pdf")

    r = other_client.get("/auth/export")
    with tarfile.open(fileobj=io.BytesIO(r.content), mode="r:gz") as archive:
        manifest = json.load(archive.extractfile("account.json"))
        names = archive.getnames()

    assert manifest["documents"] == []
    assert not any("alices_contract" in n for n in names)


# ——— deletion ———


def test_deleting_requires_the_password_again(alice: TestClient):
    """Reachable from a logged-in session and irreversible — a borrowed laptop
    should not be enough."""
    r = alice.post("/auth/delete", json={"password": "wrongpassword"}, headers=csrf(alice))
    assert r.status_code == 401
    with Session(engine) as session:
        assert session.exec(select(User)).all(), "the account was deleted anyway"


def test_deleting_requires_csrf(alice: TestClient):
    assert alice.post("/auth/delete", json={"password": GOOD["password"]}).status_code == 403


def test_deleting_removes_the_account_and_ends_the_session(alice: TestClient):
    r = alice.post("/auth/delete", json={"password": GOOD["password"]}, headers=csrf(alice))
    assert r.status_code == 200, r.text
    assert alice.get("/auth/me").status_code == 401

    with Session(engine) as session:
        assert session.exec(select(User)).all() == []
        assert session.exec(select(RefreshToken)).all() == []


@pytest.mark.slow
def test_deleting_takes_the_files_off_disk(alice: TestClient, text_pdf: bytes):
    """The claim is that the documents are gone, not that the rows are."""
    ws = workspace_id(alice)
    _upload(alice, text_pdf, "contract.pdf")
    papers_dir = library.workspace_papers_dir(ws)
    assert list(papers_dir.glob("*")), "nothing was stored, so this proves nothing"

    alice.post("/auth/delete", json={"password": GOOD["password"]}, headers=csrf(alice))

    assert not list(papers_dir.glob("*")) if papers_dir.exists() else True
    assert not (library.workspace_index_dir(ws) / "index.faiss").exists()
    with Session(engine) as session:
        assert session.exec(select(Paper)).all() == []
        assert session.exec(select(Workspace)).all() == []


def test_deleting_clears_the_audit_log(alice: TestClient):
    with Session(engine) as session:
        user = session.exec(select(User)).first()
        session.add(AnswerLog(
            user_id=user.id, workspace_id=user.current_workspace_id,
            question="q", query="q", answer="a", citations_json="[]",
            model="test", temperature=0.0, k=5, candidates=20,
        ))
        session.commit()

    alice.post("/auth/delete", json={"password": GOOD["password"]}, headers=csrf(alice))
    with Session(engine) as session:
        assert session.exec(select(AnswerLog)).all() == []


# ——— the shared-workspace case, which is where this gets dangerous ———


def _shared_workspace(alice: TestClient, other_client: TestClient) -> int:
    created = alice.post("/workspaces", json={"name": "Acme Legal"},
                         headers=csrf(alice)).json()
    token = alice.post(f"/workspaces/{created['id']}/invitations",
                       json={"email": OTHER["email"]}, headers=csrf(alice)).json()["token"]
    other_client.post("/auth/register", json=OTHER)
    other_client.post("/workspaces/accept", json={"token": token}, headers=csrf(other_client))
    return created["id"]


def test_the_last_owner_of_a_shared_workspace_cannot_just_delete(
    alice: TestClient, other_client: TestClient
):
    """Those documents belong to the workspace, not to whoever created it.
    Silently destroying them — or orphaning them — is worse than refusing."""
    ws_id = _shared_workspace(alice, other_client)

    r = alice.post("/auth/delete", json={"password": GOOD["password"]}, headers=csrf(alice))
    assert r.status_code == 409
    assert "Acme Legal" in r.json()["detail"]
    assert "owner" in r.json()["detail"]

    with Session(engine) as session:
        assert session.get(Workspace, ws_id) is not None


def test_a_member_leaving_does_not_take_the_workspace_with_them(
    alice: TestClient, other_client: TestClient
):
    """Deleting a non-owner's account must leave the team's library intact."""
    ws_id = _shared_workspace(alice, other_client)

    r = other_client.post("/auth/delete", json={"password": OTHER["password"]},
                          headers=csrf(other_client))
    assert r.status_code == 200, r.text

    with Session(engine) as session:
        assert session.get(Workspace, ws_id) is not None, "the workspace was destroyed"
        assert [m.user_id for m in _members(session, ws_id)] == [1]


def test_a_sole_member_workspace_goes_with_the_account(alice: TestClient):
    """Nobody else loses anything, so refusing here would be obstruction."""
    created = alice.post("/workspaces", json={"name": "Solo"}, headers=csrf(alice)).json()

    r = alice.post("/auth/delete", json={"password": GOOD["password"]}, headers=csrf(alice))
    assert r.status_code == 200, r.text
    with Session(engine) as session:
        assert session.get(Workspace, created["id"]) is None


def test_promoting_someone_else_unblocks_deletion(
    alice: TestClient, other_client: TestClient
):
    """The refusal has to be escapable, or it is a trap rather than a guard."""
    ws_id = _shared_workspace(alice, other_client)
    assert alice.post("/auth/delete", json={"password": GOOD["password"]},
                      headers=csrf(alice)).status_code == 409

    # Make the other member an owner, exactly as the error message says to.
    with Session(engine) as session:
        other = session.exec(select(User).where(User.email == OTHER["email"])).first()
        membership = session.exec(
            select(Membership).where(Membership.workspace_id == ws_id,
                                     Membership.user_id == other.id)
        ).first()
        membership.role = "owner"
        session.add(membership)
        session.commit()

    r = alice.post("/auth/delete", json={"password": GOOD["password"]}, headers=csrf(alice))
    assert r.status_code == 200, r.text
    with Session(engine) as session:
        assert session.get(Workspace, ws_id) is not None
