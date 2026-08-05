"""Account export and deletion — the two things a person is entitled to do with
their own data, and the two that are easiest to implement dishonestly.

Export must contain the actual documents, not a manifest of them. Deletion must
actually remove bytes from disk, not just hide rows. Both are written so that
what the privacy policy claims is what the code does.

The hard case is a SHARED workspace. Deleting an account must not destroy a
team's library — those documents belong to the workspace, not to whoever
uploaded them. So deletion refuses while the account is the last owner of a
workspace that still has other members, and says how to fix it. Refusing is the
honest answer: silently orphaning a team's documents, or silently deleting
them, are both worse than telling someone to hand over ownership first.
"""
import io
import json
import logging
import shutil
import tarfile
from datetime import datetime, timezone

from sqlmodel import Session, select

from backend import library
from backend.db_models import (
    ROLE_OWNER, AnswerLog, IndexJob, Invitation, Membership, Paper,
    PasswordResetToken, RefreshToken, User, Workspace,
)

log = logging.getLogger(__name__)


class DeletionBlocked(Exception):
    """Raised when deleting would take a shared library down with it."""


def _iso(value: datetime | None) -> str | None:
    return value.isoformat() if value else None


def blocking_workspaces(session: Session, user: User) -> list[Workspace]:
    """Shared workspaces this account is the last owner of, and is not alone in.

    Being the sole member is not blocking: nobody else loses anything, so the
    workspace goes with the account.
    """
    blocked: list[Workspace] = []
    for membership in session.exec(
        select(Membership).where(
            Membership.user_id == user.id, Membership.role == ROLE_OWNER
        )
    ).all():
        workspace = session.get(Workspace, membership.workspace_id)
        if workspace is None or workspace.is_personal:
            continue
        members = session.exec(
            select(Membership).where(Membership.workspace_id == workspace.id)
        ).all()
        owners = [m for m in members if m.role == ROLE_OWNER]
        if len(members) > 1 and len(owners) == 1:
            blocked.append(workspace)
    return blocked


def build_export(session: Session, user: User) -> bytes:
    """Everything this account holds, as a .tar.gz.

    Contains the real files, because an export that lists documents without
    including them is not an export. Only documents this account UPLOADED are
    included: the rest belong to workspaces it merely has access to, and taking
    a copy of a colleague's contract is not what "export my data" means.
    """
    memberships = session.exec(
        select(Membership).where(Membership.user_id == user.id)
    ).all()
    workspaces = [session.get(Workspace, m.workspace_id) for m in memberships]
    papers = session.exec(select(Paper).where(Paper.user_id == user.id)).all()
    answers = session.exec(select(AnswerLog).where(AnswerLog.user_id == user.id)).all()

    manifest = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "account": {
            "id": user.id,
            "email": user.email,
            "created_at": _iso(user.created_at),
        },
        "workspaces": [
            {
                "id": w.id, "name": w.name, "is_personal": w.is_personal,
                "role": m.role, "joined_at": _iso(m.created_at),
            }
            for w, m in zip(workspaces, memberships) if w is not None
        ],
        "documents": [
            {
                "paper_id": p.paper_id, "title": p.title, "filename": p.filename,
                "workspace_id": p.workspace_id, "n_chunks": p.n_chunks,
                "uploaded_at": _iso(p.created_at),
                "file": f"documents/{p.workspace_id}/{p.filename}",
            }
            for p in papers
        ],
        # The audit log is the product's central claim, so it belongs in an
        # export: the evidence chain for every answer, not just the answers.
        "answers": [
            {
                "id": a.id, "asked_at": _iso(a.created_at), "question": a.question,
                "query": a.query, "retrieval_query": a.retrieval_query,
                "expansion_mode": a.expansion_mode, "answer": a.answer,
                "model": a.model, "temperature": a.temperature,
                "k": a.k, "candidates": a.candidates,
                "citations": json.loads(a.citations_json or "[]"),
                "index_fingerprint": a.index_fingerprint,
            }
            for a in answers
        ],
        "notes": (
            "documents/ contains only files THIS account uploaded. Files in "
            "shared workspaces uploaded by others are not included, because "
            "they are not this account's data to take."
        ),
    }

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        blob = json.dumps(manifest, indent=2).encode("utf-8")
        info = tarfile.TarInfo("account.json")
        info.size = len(blob)
        archive.addfile(info, io.BytesIO(blob))

        for paper in papers:
            path = library.stored_path(paper.workspace_id, paper.paper_id)
            if path and path.exists():
                archive.add(path, arcname=f"documents/{paper.workspace_id}/{paper.filename}")

    return buffer.getvalue()


def delete_account(session: Session, user: User) -> None:
    """Erase the account and everything that is only its own.

    Order matters: files come off disk BEFORE the rows that say where they are.
    Doing it the other way round loses the paths and leaves the bytes, which is
    the exact failure a deletion feature exists to prevent.
    """
    blocked = blocking_workspaces(session, user)
    if blocked:
        raise DeletionBlocked(", ".join(w.name for w in blocked))

    user_id = user.id
    memberships = session.exec(
        select(Membership).where(Membership.user_id == user_id)
    ).all()

    for membership in memberships:
        workspace = session.get(Workspace, membership.workspace_id)
        if workspace is None:
            continue
        others = [
            m for m in session.exec(
                select(Membership).where(Membership.workspace_id == workspace.id)
            ).all()
            if m.user_id != user_id
        ]
        if others:
            # A shared workspace the account is merely leaving. Its documents
            # stay: they belong to the workspace and the people still in it.
            session.delete(membership)
            continue

        # Nobody else is in it, so the whole library goes — files first.
        for directory in (library.workspace_index_dir(workspace.id),
                          library.workspace_papers_dir(workspace.id)):
            shutil.rmtree(directory, ignore_errors=True)
        library.invalidate(workspace.id)

        for paper in session.exec(
            select(Paper).where(Paper.workspace_id == workspace.id)
        ).all():
            session.delete(paper)
        for job in session.exec(
            select(IndexJob).where(IndexJob.workspace_id == workspace.id)
        ).all():
            session.delete(job)
        for invitation in session.exec(
            select(Invitation).where(Invitation.workspace_id == workspace.id)
        ).all():
            session.delete(invitation)
        session.delete(membership)
        session.delete(workspace)

    # Rows keyed to the person rather than to a workspace.
    for model in (AnswerLog, RefreshToken, PasswordResetToken):
        for row in session.exec(select(model).where(model.user_id == user_id)).all():
            session.delete(row)
    for invitation in session.exec(
        select(Invitation).where(Invitation.invited_by_user_id == user_id)
    ).all():
        session.delete(invitation)

    session.delete(session.get(User, user_id))
    session.commit()
    log.info("account %s deleted", user_id)
