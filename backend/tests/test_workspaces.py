"""Workspaces: membership as the authorisation boundary, and sharing that works.

The isolation tests here are the ones that matter most in the whole suite. A
retrieval bug returns a bad answer; a membership bug shows one company's
contracts to another.
"""
import pytest
from fastapi.testclient import TestClient

from backend.tests.conftest import csrf, workspace_id

OTHER = {"email": "mallory@example.com", "password": "validpassword123"}


def _register(client: TestClient, email: str) -> TestClient:
    r = client.post("/auth/register", json={"email": email, "password": "validpassword123"})
    assert r.status_code == 201, r.text
    return client


def _create_workspace(client: TestClient, name: str = "Acme Legal") -> dict:
    r = client.post("/workspaces", json={"name": name}, headers=csrf(client))
    assert r.status_code == 201, r.text
    return r.json()


def _invite(client: TestClient, ws_id: int, email: str, role: str = "member") -> dict:
    r = client.post(f"/workspaces/{ws_id}/invitations",
                    json={"email": email, "role": role}, headers=csrf(client))
    assert r.status_code == 201, r.text
    return r.json()


def _upload(client: TestClient, data: bytes, name: str = "paper.pdf"):
    return client.post("/papers", headers=csrf(client),
                       files={"file": (name, data, "application/pdf")})


def _uploaded_paper_id(client: TestClient, data: bytes, name: str) -> int:
    """Upload, then look the PAPER's id up from the library.

    Upload answers with a job now, whose id is its own — reading `id` off that
    response gives the wrong number as soon as the two sequences diverge.
    """
    r = _upload(client, data, name=name)
    assert r.status_code == 202, r.text
    papers = client.get("/papers").json()
    return next(p["id"] for p in papers if p["filename"] == name)


# ——— every account gets a personal workspace ———


def test_registering_creates_a_personal_workspace(alice: TestClient):
    workspaces = alice.get("/workspaces").json()
    assert len(workspaces) == 1
    assert workspaces[0]["is_personal"] is True
    assert workspaces[0]["is_current"] is True
    assert workspaces[0]["role"] == "owner"


def test_workspaces_requires_auth(client: TestClient):
    assert client.get("/workspaces").status_code == 401


def test_a_personal_workspace_cannot_be_shared(alice: TestClient):
    """Sharing "my documents" would make the name a lie — create a workspace."""
    personal = workspace_id(alice)
    r = alice.post(f"/workspaces/{personal}/invitations",
                   json={"email": "x@example.com"}, headers=csrf(alice))
    assert r.status_code == 400
    assert "personal" in r.json()["detail"].lower()


# ——— creating and switching ———


def test_creating_a_workspace_makes_you_its_owner(alice: TestClient):
    created = _create_workspace(alice)
    assert created["role"] == "owner"
    assert created["is_personal"] is False
    assert len(alice.get("/workspaces").json()) == 2


def test_creating_does_not_switch_you_into_it(alice: TestClient):
    """Switching should be deliberate — an upload landing in the wrong library
    is a data-placement mistake that is tedious to undo."""
    before = workspace_id(alice)
    _create_workspace(alice)
    assert workspace_id(alice) == before


def test_activating_switches_the_active_workspace(alice: TestClient):
    created = _create_workspace(alice)
    r = alice.post(f"/workspaces/{created['id']}/activate", headers=csrf(alice))
    assert r.status_code == 200
    assert workspace_id(alice) == created["id"]


def test_activating_requires_csrf(alice: TestClient):
    created = _create_workspace(alice)
    assert alice.post(f"/workspaces/{created['id']}/activate").status_code == 403


def test_cannot_activate_a_workspace_you_are_not_in(
    alice: TestClient, other_client: TestClient
):
    """404 not 403 — a 403 would confirm the workspace exists."""
    created = _create_workspace(alice)
    _register(other_client, OTHER["email"])
    r = other_client.post(f"/workspaces/{created['id']}/activate", headers=csrf(other_client))
    assert r.status_code == 404


def test_cannot_list_members_of_a_workspace_you_are_not_in(
    alice: TestClient, other_client: TestClient
):
    created = _create_workspace(alice)
    _register(other_client, OTHER["email"])
    assert other_client.get(f"/workspaces/{created['id']}/members").status_code == 404


@pytest.mark.parametrize("name", ["", "   ", "x" * 81])
def test_workspace_names_are_validated(alice: TestClient, name):
    assert alice.post("/workspaces", json={"name": name}, headers=csrf(alice)).status_code == 422


# ——— invitations ———


def test_inviting_requires_ownership(alice: TestClient, other_client: TestClient):
    """Only an owner decides who else can read the library."""
    created = _create_workspace(alice)
    _register(other_client, OTHER["email"])
    token = _invite(alice, created["id"], OTHER["email"])["token"]
    other_client.post("/workspaces/accept", json={"token": token}, headers=csrf(other_client))

    r = other_client.post(f"/workspaces/{created['id']}/invitations",
                          json={"email": "third@example.com"}, headers=csrf(other_client))
    assert r.status_code == 403


def test_accepting_an_invitation_joins_the_workspace(
    alice: TestClient, other_client: TestClient
):
    created = _create_workspace(alice)
    _register(other_client, OTHER["email"])
    token = _invite(alice, created["id"], OTHER["email"])["token"]

    r = other_client.post("/workspaces/accept", json={"token": token},
                          headers=csrf(other_client))
    assert r.status_code == 200, r.text
    assert r.json()["id"] == created["id"]
    assert created["id"] in [w["id"] for w in other_client.get("/workspaces").json()]


def test_an_invitation_is_not_a_bearer_token(alice: TestClient, other_client: TestClient):
    """The invited EMAIL must match the accepting account. Otherwise a forwarded
    invitation link lets anyone into the workspace, and this is a public join
    link wearing an invitation's clothes."""
    created = _create_workspace(alice)
    token = _invite(alice, created["id"], "intended@example.com")["token"]

    _register(other_client, OTHER["email"])
    r = other_client.post("/workspaces/accept", json={"token": token},
                          headers=csrf(other_client))
    assert r.status_code == 403
    assert "different email" in r.json()["detail"]


def test_an_invitation_works_only_once(alice: TestClient, other_client: TestClient):
    created = _create_workspace(alice)
    _register(other_client, OTHER["email"])
    token = _invite(alice, created["id"], OTHER["email"])["token"]

    assert other_client.post("/workspaces/accept", json={"token": token},
                             headers=csrf(other_client)).status_code == 200
    assert other_client.post("/workspaces/accept", json={"token": token},
                             headers=csrf(other_client)).status_code == 400


def test_a_bogus_invitation_token_is_refused(alice: TestClient):
    r = alice.post("/workspaces/accept", json={"token": "nonsense"}, headers=csrf(alice))
    assert r.status_code == 400


def test_a_revoked_invitation_cannot_be_accepted(alice: TestClient, other_client: TestClient):
    created = _create_workspace(alice)
    invitation = _invite(alice, created["id"], OTHER["email"])
    assert alice.delete(f"/workspaces/{created['id']}/invitations/{invitation['id']}",
                        headers=csrf(alice)).status_code == 204

    _register(other_client, OTHER["email"])
    r = other_client.post("/workspaces/accept", json={"token": invitation["token"]},
                          headers=csrf(other_client))
    assert r.status_code == 400


def test_only_the_hash_of_an_invitation_is_stored(alice: TestClient):
    from sqlmodel import Session, select

    from backend.db import engine
    from backend.db_models import Invitation

    created = _create_workspace(alice)
    token = _invite(alice, created["id"], OTHER["email"])["token"]
    with Session(engine) as session:
        stored = session.exec(select(Invitation)).first()
    assert token not in stored.token_hash


# ——— membership management ———


def test_members_lists_everyone(alice: TestClient, other_client: TestClient):
    created = _create_workspace(alice)
    _register(other_client, OTHER["email"])
    token = _invite(alice, created["id"], OTHER["email"])["token"]
    other_client.post("/workspaces/accept", json={"token": token}, headers=csrf(other_client))

    members = alice.get(f"/workspaces/{created['id']}/members").json()
    assert {m["email"] for m in members} == {"alice@example.com", OTHER["email"]}
    assert {m["role"] for m in members} == {"owner", "member"}


def test_the_last_owner_cannot_be_removed(alice: TestClient):
    """A workspace with no owner has documents nobody can manage and members
    nobody can add — unreachable except by database surgery."""
    created = _create_workspace(alice)
    me = alice.get("/auth/me").json()["id"]
    r = alice.delete(f"/workspaces/{created['id']}/members/{me}", headers=csrf(alice))
    assert r.status_code == 400
    assert "last owner" in r.json()["detail"]


def test_removing_a_member_revokes_their_access(alice: TestClient, other_client: TestClient):
    created = _create_workspace(alice)
    other = _register(other_client, OTHER["email"])
    token = _invite(alice, created["id"], OTHER["email"])["token"]
    other.post("/workspaces/accept", json={"token": token}, headers=csrf(other))
    other_id = other.get("/auth/me").json()["id"]

    assert alice.delete(f"/workspaces/{created['id']}/members/{other_id}",
                        headers=csrf(alice)).status_code == 204
    assert created["id"] not in [w["id"] for w in other.get("/workspaces").json()]
    assert other.post(f"/workspaces/{created['id']}/activate",
                      headers=csrf(other)).status_code == 404


def test_a_removed_member_falls_back_to_their_personal_workspace(
    alice: TestClient, other_client: TestClient
):
    """Otherwise they are left pointed at a library they can no longer read, and
    every subsequent request 404s for no visible reason."""
    created = _create_workspace(alice)
    other = _register(other_client, OTHER["email"])
    token = _invite(alice, created["id"], OTHER["email"])["token"]
    other.post("/workspaces/accept", json={"token": token}, headers=csrf(other))
    other.post(f"/workspaces/{created['id']}/activate", headers=csrf(other))
    other_id = other.get("/auth/me").json()["id"]

    alice.delete(f"/workspaces/{created['id']}/members/{other_id}", headers=csrf(alice))

    assert other.get("/papers").status_code == 200
    current = [w for w in other.get("/workspaces").json() if w["is_current"]]
    assert current and current[0]["is_personal"] is True


# ——— the isolation that matters ———


@pytest.mark.slow
def test_workspace_members_share_one_library(
    alice: TestClient, other_client: TestClient, text_pdf: bytes, fake_llm
):
    """The whole point of the phase: a document uploaded by one member is
    visible, and answerable, to another."""
    created = _create_workspace(alice)
    alice.post(f"/workspaces/{created['id']}/activate", headers=csrf(alice))
    _upload(alice, text_pdf, name="shared_contract.pdf")

    other = _register(other_client, OTHER["email"])
    token = _invite(alice, created["id"], OTHER["email"])["token"]
    other.post("/workspaces/accept", json={"token": token}, headers=csrf(other))
    other.post(f"/workspaces/{created['id']}/activate", headers=csrf(other))

    assert [p["paper_id"] for p in other.get("/papers").json()] == ["shared_contract"]
    r = other.post("/ask", json={"question": "What does the retriever select?"},
                   headers=csrf(other))
    assert r.status_code == 200, r.text
    assert r.json()["citations"], "a member could not retrieve from the shared library"


@pytest.mark.slow
def test_a_non_member_retrieves_none_of_the_workspaces_passages(
    alice: TestClient, other_client: TestClient, text_pdf: bytes, fake_llm
):
    """The strongest isolation assertion: not a 403, but zero leaked evidence."""
    created = _create_workspace(alice)
    alice.post(f"/workspaces/{created['id']}/activate", headers=csrf(alice))
    _upload(alice, text_pdf, name="private_contract.pdf")

    other = _register(other_client, OTHER["email"])
    assert other.get("/papers").json() == []
    r = other.post("/ask", json={"question": "What does the retriever select?"},
                   headers=csrf(other))
    assert r.status_code == 400, "an empty library must not fall through to someone else's"


@pytest.mark.slow
def test_switching_workspaces_switches_the_library(
    alice: TestClient, text_pdf: bytes, fake_llm
):
    """Two libraries owned by the same person must not bleed into each other."""
    personal = workspace_id(alice)
    _upload(alice, text_pdf, name="personal_notes.pdf")

    team = _create_workspace(alice)
    alice.post(f"/workspaces/{team['id']}/activate", headers=csrf(alice))
    assert alice.get("/papers").json() == [], "team workspace started with the personal library"

    _upload(alice, text_pdf, name="team_contract.pdf")
    assert [p["paper_id"] for p in alice.get("/papers").json()] == ["team_contract"]

    alice.post(f"/workspaces/{personal}/activate", headers=csrf(alice))
    assert [p["paper_id"] for p in alice.get("/papers").json()] == ["personal_notes"]


@pytest.mark.slow
def test_deletion_is_narrower_than_reading(
    alice: TestClient, other_client: TestClient, text_pdf: bytes
):
    """Everyone in a workspace can read every document, but a member must not be
    able to silently destroy a colleague's work — there is no undo."""
    created = _create_workspace(alice)
    alice.post(f"/workspaces/{created['id']}/activate", headers=csrf(alice))
    paper_id = _uploaded_paper_id(alice, text_pdf, "alices_contract.pdf")

    other = _register(other_client, OTHER["email"])
    token = _invite(alice, created["id"], OTHER["email"])["token"]
    other.post("/workspaces/accept", json={"token": token}, headers=csrf(other))
    other.post(f"/workspaces/{created['id']}/activate", headers=csrf(other))

    assert other.get(f"/papers/{paper_id}/file").status_code == 200, "member cannot read"
    r = other.delete(f"/papers/{paper_id}", headers=csrf(other))
    assert r.status_code == 403
    assert len(alice.get("/papers").json()) == 1, "the document must survive"

    # The uploader still can.
    assert alice.delete(f"/papers/{paper_id}", headers=csrf(alice)).status_code == 204


@pytest.mark.slow
def test_an_owner_can_delete_any_document_in_the_workspace(
    alice: TestClient, other_client: TestClient, text_pdf: bytes
):
    created = _create_workspace(alice)
    other = _register(other_client, OTHER["email"])
    token = _invite(alice, created["id"], OTHER["email"])["token"]
    other.post("/workspaces/accept", json={"token": token}, headers=csrf(other))
    other.post(f"/workspaces/{created['id']}/activate", headers=csrf(other))
    paper_id = _uploaded_paper_id(other, text_pdf, "members_upload.pdf")

    alice.post(f"/workspaces/{created['id']}/activate", headers=csrf(alice))
    assert alice.delete(f"/papers/{paper_id}", headers=csrf(alice)).status_code == 204


@pytest.mark.slow
def test_the_audit_log_is_shared_by_the_workspace(
    alice: TestClient, other_client: TestClient, text_pdf: bytes, fake_llm
):
    """A reviewer needs to see what the TEAM was told, not only their own asks."""
    created = _create_workspace(alice)
    alice.post(f"/workspaces/{created['id']}/activate", headers=csrf(alice))
    _upload(alice, text_pdf)
    alice.post("/ask", json={"question": "What does the retriever select?"},
               headers=csrf(alice))

    other = _register(other_client, OTHER["email"])
    token = _invite(alice, created["id"], OTHER["email"])["token"]
    other.post("/workspaces/accept", json={"token": token}, headers=csrf(other))
    other.post(f"/workspaces/{created['id']}/activate", headers=csrf(other))

    entries = other.get("/audit").json()
    assert len(entries) == 1, "a member cannot see the workspace's answer log"


@pytest.mark.slow
def test_audit_entries_do_not_leak_across_workspaces(
    alice: TestClient, other_client: TestClient, text_pdf: bytes, fake_llm
):
    _upload(alice, text_pdf)
    alice.post("/ask", json={"question": "What does the retriever select?"},
               headers=csrf(alice))
    entry_id = alice.get("/audit").json()[0]["id"]

    other = _register(other_client, OTHER["email"])
    assert other.get("/audit").json() == []
    assert other.get(f"/audit/{entry_id}").status_code == 404
    assert other.get(f"/audit/{entry_id}/export").status_code == 404


@pytest.fixture
def fake_llm(monkeypatch):
    from backend.models import Answer

    monkeypatch.setattr("backend.api.generate",
                        lambda q, c: Answer(question=q, answer="Stubbed [1].", citations=c))
    monkeypatch.setattr("backend.api.condense_question", lambda q, h: q)
