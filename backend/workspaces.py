"""Workspaces: creation, membership, and the dependency every scoped route uses.

THE AUTHORISATION RULE, in one place. A request may act on a workspace if and
only if the caller has a Membership row for it. Every other check in the app is
downstream of that, so it must not be reimplemented per route — the failure mode
of scattered checks is the one route where somebody forgot.

Absence of membership returns 404, never 403. A 403 confirms the workspace
exists, which tells an attacker whether an id is real; a 404 tells them nothing.
The same reasoning already governs papers and audit entries.
"""
import secrets
from datetime import datetime, timedelta, timezone

from fastapi import Depends, HTTPException, status
from sqlmodel import Session, select

from backend.auth import get_current_user
from backend.config import settings
from backend.db import get_session
from backend.db_models import (
    ROLE_MEMBER, ROLE_OWNER, Invitation, Membership, User, Workspace,
)
from backend.security import hash_token


def create_workspace(
    session: Session, owner: User, name: str, is_personal: bool = False
) -> Workspace:
    """Create a workspace and make `owner` its first member."""
    workspace = Workspace(name=name, is_personal=is_personal)
    session.add(workspace)
    session.commit()
    session.refresh(workspace)

    session.add(Membership(user_id=owner.id, workspace_id=workspace.id, role=ROLE_OWNER))
    session.commit()
    return workspace


def ensure_personal_workspace(session: Session, user: User) -> Workspace:
    """The workspace a user's own documents live in, created if absent.

    Called at sign-up and by the migration. Idempotent, so re-running either is
    safe — the migration in particular must be re-runnable after a partial
    failure without duplicating anyone's library.
    """
    existing = session.exec(
        select(Workspace)
        .join(Membership, Membership.workspace_id == Workspace.id)
        .where(Membership.user_id == user.id, Workspace.is_personal == True)  # noqa: E712
    ).first()
    if existing is not None:
        workspace = existing
    else:
        workspace = create_workspace(session, user, name="My Library", is_personal=True)

    if user.current_workspace_id is None:
        user.current_workspace_id = workspace.id
        session.add(user)
        session.commit()
    return workspace


def membership_for(session: Session, user_id: int, workspace_id: int) -> Membership | None:
    return session.exec(
        select(Membership).where(
            Membership.user_id == user_id, Membership.workspace_id == workspace_id
        )
    ).first()


def memberships_of(session: Session, user_id: int) -> list[Membership]:
    return list(
        session.exec(select(Membership).where(Membership.user_id == user_id)).all()
    )


def require_membership(session: Session, user: User, workspace_id: int) -> Membership:
    """The authorisation check. 404 when absent — see the module docstring."""
    membership = membership_for(session, user.id, workspace_id)
    if membership is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    return membership


def require_owner(session: Session, user: User, workspace_id: int) -> Membership:
    """For actions that change WHO ELSE can see the library.

    403 rather than 404 here on purpose: the caller is a member, so the
    workspace's existence is not a secret from them. Hiding it would only make
    a permission error look like a bug.
    """
    membership = require_membership(session, user, workspace_id)
    if membership.role != ROLE_OWNER:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "only an owner can do that")
    return membership


# ——— the dependency scoped routes use ———


def get_current_workspace(
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> Workspace:
    """Resolve the workspace this request acts on, and prove the caller belongs.

    Taken from the user's `current_workspace_id` rather than a request
    parameter, so no route can forget to scope itself and no client can point
    itself at someone else's library by editing an id. Switching is an explicit
    action (POST /workspaces/{id}/activate) that goes through the same
    membership check.
    """
    workspace_id = user.current_workspace_id
    if workspace_id is None:
        # Only reachable for an account created before workspaces existed and
        # not yet migrated. Repair rather than fail: the personal workspace is
        # derivable, and refusing would lock the user out of their own library.
        return ensure_personal_workspace(session, user)

    require_membership(session, user, workspace_id)
    workspace = session.get(Workspace, workspace_id)
    if workspace is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "workspace not found")
    return workspace


# ——— invitations ———


def create_invitation(
    session: Session, workspace: Workspace, inviter: User, email: str, role: str
) -> tuple[Invitation, str]:
    """Mint an invitation. Returns (row, plaintext token) — the token is never
    stored, only its hash, so this is the one moment it exists."""
    token = secrets.token_urlsafe(32)
    invitation = Invitation(
        workspace_id=workspace.id,
        email=email.lower(),
        role=role if role in (ROLE_OWNER, ROLE_MEMBER) else ROLE_MEMBER,
        token_hash=hash_token(token),
        invited_by_user_id=inviter.id,
        expires_at=datetime.now(timezone.utc)
        + timedelta(days=settings.invitation_expire_days),
    )
    session.add(invitation)
    session.commit()
    session.refresh(invitation)
    return invitation, token


def accept_invitation(session: Session, user: User, token: str) -> Workspace:
    """Consume an invitation and join its workspace.

    The invited EMAIL must match the accepting account. Without that check an
    invitation link is a bearer token for the workspace: anyone it was forwarded
    to could join. This is the difference between "an invitation" and "a public
    join link", and the product promised the former.
    """
    invitation = session.exec(
        select(Invitation).where(Invitation.token_hash == hash_token(token))
    ).first()

    now = datetime.now(timezone.utc)
    invalid = (
        invitation is None
        or invitation.accepted_at is not None
        or invitation.revoked_at is not None
        or invitation.expires_at.replace(tzinfo=timezone.utc) < now
    )
    if invalid:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid or expired invitation")
    if invitation.email != user.email.lower():
        raise HTTPException(
            status.HTTP_403_FORBIDDEN, "this invitation was issued to a different email address"
        )

    workspace = session.get(Workspace, invitation.workspace_id)
    if workspace is None:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "invalid or expired invitation")

    # Re-accepting when already a member is a no-op rather than an error: the
    # user's intent is satisfied either way, and a duplicate row would violate
    # the unique constraint.
    if membership_for(session, user.id, workspace.id) is None:
        session.add(Membership(
            user_id=user.id, workspace_id=workspace.id, role=invitation.role,
        ))
    invitation.accepted_at = now
    session.add(invitation)
    session.commit()
    return workspace
