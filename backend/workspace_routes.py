"""Workspace routes: list, create, switch, invite, join, manage members.

    GET    /workspaces                    -> the ones I belong to
    POST   /workspaces                    -> create a team workspace
    POST   /workspaces/{id}/activate      -> make it the one my requests act on
    GET    /workspaces/{id}/members       -> who is in it
    POST   /workspaces/{id}/invitations   -> invite by email (owner only)
    DELETE /workspaces/{id}/members/{uid} -> remove someone (owner only)
    POST   /workspaces/accept             -> join using an invitation token

Every route that names a workspace id proves membership first, and answers 404
when there is none — so an id cannot be probed for existence.
"""
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from sqlmodel import Session, select

from backend import mailer
from backend.auth import get_current_user, require_csrf
from backend.config import settings
from backend.db import get_session
from backend.db_models import (
    ROLE_OWNER, Invitation, Membership, User, Workspace,
)
from backend.models import (
    CreateWorkspaceRequest, InvitationCreated, InviteRequest, MemberPublic,
    WorkspacePublic,
)
from backend import workspaces

router = APIRouter(prefix="/workspaces", tags=["workspaces"])

log = logging.getLogger(__name__)


INVITE_SUBJECT = "You have been invited to {workspace} on Scholar"

INVITE_BODY = """\
{inviter} invited you to the "{workspace}" workspace on Scholar, where you will
be able to read and question its documents.

Open this link to accept:

{url}

The invitation expires in {days} days and can only be accepted by {email}.
If you were not expecting it, you can ignore this message.
"""


def deliver_invitation(email: str, workspace: Workspace, inviter: User, token: str) -> bool:
    """Get an invitation to its recipient. Returns whether mail actually went out.

    The return value is load-bearing: the route hands the token back to the
    INVITER only when this is False. An invitation token is a credential, and a
    credential in an API response is a stopgap for the case where there is no
    other way to deliver it — not a permanent part of the contract.
    """
    if mailer.configured():
        body = INVITE_BODY.format(
            inviter=inviter.email, workspace=workspace.name, email=email,
            url=mailer.link("/join", token), days=settings.invitation_expire_days,
        )
        return mailer.send(email, INVITE_SUBJECT.format(workspace=workspace.name), body)

    log.warning(
        "WORKSPACE INVITE for %s to %r — no mail provider configured, token logged: %s",
        email, workspace.name, token,
    )
    return False


def _public(workspace: Workspace, role: str, is_current: bool) -> WorkspacePublic:
    return WorkspacePublic(
        id=workspace.id, name=workspace.name, is_personal=workspace.is_personal,
        role=role, is_current=is_current, created_at=workspace.created_at,
    )


@router.get("", response_model=list[WorkspacePublic])
def list_workspaces(
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> list[WorkspacePublic]:
    # Repair-on-read: an account migrated mid-flight, or created before
    # workspaces existed, still gets a usable answer instead of an empty list.
    workspaces.ensure_personal_workspace(session, user)
    session.refresh(user)

    result: list[WorkspacePublic] = []
    for membership in workspaces.memberships_of(session, user.id):
        workspace = session.get(Workspace, membership.workspace_id)
        if workspace is not None:
            result.append(_public(workspace, membership.role,
                                  workspace.id == user.current_workspace_id))
    result.sort(key=lambda w: (not w.is_personal, w.name.lower()))
    return result


@router.post("", response_model=WorkspacePublic, status_code=status.HTTP_201_CREATED,
             dependencies=[Depends(require_csrf)])
def create_workspace(
    body: CreateWorkspaceRequest,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> WorkspacePublic:
    """Create a shared workspace. The creator becomes its owner."""
    workspace = workspaces.create_workspace(session, user, name=body.name.strip())
    return _public(workspace, ROLE_OWNER, is_current=False)


@router.post("/accept", response_model=WorkspacePublic, dependencies=[Depends(require_csrf)])
def accept_invitation(
    body: dict,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> WorkspacePublic:
    """Join a workspace using an invitation token.

    Declared BEFORE /{workspace_id}/... routes: FastAPI matches in declaration
    order, and "accept" would otherwise be parsed as a workspace id.
    """
    token = (body or {}).get("token", "")
    if not token:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, "token is required")
    workspace = workspaces.accept_invitation(session, user, token)
    membership = workspaces.membership_for(session, user.id, workspace.id)
    return _public(workspace, membership.role, is_current=False)


@router.post("/{workspace_id}/activate", response_model=WorkspacePublic,
             dependencies=[Depends(require_csrf)])
def activate_workspace(
    workspace_id: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> WorkspacePublic:
    """Point this account's requests at a different workspace.

    Goes through the same membership check as everything else, so switching can
    never be used to reach a library the caller does not belong to.
    """
    membership = workspaces.require_membership(session, user, workspace_id)
    workspace = session.get(Workspace, workspace_id)
    user.current_workspace_id = workspace_id
    session.add(user)
    session.commit()
    return _public(workspace, membership.role, is_current=True)


@router.get("/{workspace_id}/members", response_model=list[MemberPublic])
def list_members(
    workspace_id: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> list[MemberPublic]:
    workspaces.require_membership(session, user, workspace_id)
    members: list[MemberPublic] = []
    for membership in session.exec(
        select(Membership).where(Membership.workspace_id == workspace_id)
    ).all():
        member = session.get(User, membership.user_id)
        if member is not None:
            members.append(MemberPublic(
                user_id=member.id, email=member.email, role=membership.role,
                joined_at=membership.created_at,
            ))
    return members


@router.post("/{workspace_id}/invitations", response_model=InvitationCreated,
             status_code=status.HTTP_201_CREATED, dependencies=[Depends(require_csrf)])
def invite_member(
    workspace_id: int,
    body: InviteRequest,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> InvitationCreated:
    """Invite someone by email. Owner only — this changes who can read the library."""
    workspaces.require_owner(session, user, workspace_id)
    workspace = session.get(Workspace, workspace_id)
    if workspace.is_personal:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "a personal library cannot be shared — create a workspace instead",
        )

    invitation, token = workspaces.create_invitation(
        session, workspace, user, email=body.email, role=body.role
    )
    sent = deliver_invitation(body.email, workspace, user, token)
    # The token goes back to the INVITER only when we could not deliver it
    # ourselves — then it is the only way the invitee can ever receive it. Once
    # mail is configured it stays out of the response entirely, because an
    # invitation token is a credential and a credential should travel to exactly
    # one person by exactly one route.
    return InvitationCreated(
        id=invitation.id, email=invitation.email, role=invitation.role,
        expires_at=invitation.expires_at, delivered=sent,
        token=None if sent else token,
    )


@router.delete("/{workspace_id}/members/{member_user_id}",
               status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_csrf)])
def remove_member(
    workspace_id: int,
    member_user_id: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> None:
    """Remove someone from a workspace. Owner only.

    The last owner cannot be removed: a workspace with no owner has documents
    nobody can manage and members nobody can add or remove — unreachable state,
    recoverable only by direct database surgery.
    """
    workspaces.require_owner(session, user, workspace_id)
    target = workspaces.membership_for(session, member_user_id, workspace_id)
    if target is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "not a member of this workspace")

    if target.role == ROLE_OWNER:
        owners = session.exec(
            select(Membership).where(
                Membership.workspace_id == workspace_id, Membership.role == ROLE_OWNER
            )
        ).all()
        if len(owners) <= 1:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                "cannot remove the last owner — promote someone else first",
            )

    # If this was their active workspace, move them back to their personal one
    # rather than leaving them pointed at a library they can no longer read.
    member = session.get(User, member_user_id)
    if member is not None and member.current_workspace_id == workspace_id:
        member.current_workspace_id = None
        session.add(member)

    session.delete(target)
    session.commit()

    if member is not None:
        workspaces.ensure_personal_workspace(session, member)


@router.delete("/{workspace_id}/invitations/{invitation_id}",
               status_code=status.HTTP_204_NO_CONTENT, dependencies=[Depends(require_csrf)])
def revoke_invitation(
    workspace_id: int,
    invitation_id: int,
    user: User = Depends(get_current_user),
    session: Session = Depends(get_session),
) -> None:
    """Withdraw an invitation that has not been accepted yet."""
    from datetime import datetime, timezone

    workspaces.require_owner(session, user, workspace_id)
    invitation = session.get(Invitation, invitation_id)
    if invitation is None or invitation.workspace_id != workspace_id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "invitation not found")
    invitation.revoked_at = datetime.now(timezone.utc)
    session.add(invitation)
    session.commit()
