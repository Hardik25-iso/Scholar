"""SQLModel database tables.

Kept separate from models.py (the Pydantic API-boundary models) so the two
concerns don't blur: this file is "what's stored", models.py is "what crosses
the wire". A User here never leaves the backend with its hashed_password.
"""
from datetime import datetime, timezone

from sqlalchemy import Column, Text, UniqueConstraint
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


# Workspace roles. Two, not four: there is exactly one privileged action
# (managing who else is in the workspace), so a finer grid would be inventing
# distinctions nothing yet enforces.
ROLE_OWNER = "owner"
ROLE_MEMBER = "member"
ROLES = (ROLE_OWNER, ROLE_MEMBER)


class User(SQLModel, table=True):
    """A registered account. Passwords are stored only as a bcrypt hash."""

    id: int | None = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    hashed_password: str  # bcrypt hash — never the plaintext (set in Slice B)
    created_at: datetime = Field(default_factory=_utcnow)

    # Which workspace this account's requests act on. Every user has a personal
    # workspace created at sign-up, so this is only null in the instant between
    # inserting the user and creating it.
    current_workspace_id: int | None = Field(default=None, foreign_key="workspace.id")


class Workspace(SQLModel, table=True):
    """A shared library. The unit documents, indexes and answers belong to.

    A user's own documents live in a `personal` workspace rather than in a
    special no-workspace case — one code path for storage, retrieval and
    authorisation instead of two that can drift.
    """

    id: int | None = Field(default=None, primary_key=True)
    name: str
    # Personal workspaces cannot be left or renamed away, and are not offered as
    # a place to invite people — flagged rather than inferred from having one
    # member, because a team workspace can legitimately have one member too.
    is_personal: bool = Field(default=False)
    created_at: datetime = Field(default_factory=_utcnow)


class Membership(SQLModel, table=True):
    """Who may act in a workspace, and with what authority.

    This row IS the authorisation check. Every workspace-scoped request resolves
    one, and its absence is indistinguishable from the workspace not existing.
    """

    __table_args__ = (UniqueConstraint("user_id", "workspace_id", name="uq_membership"),)

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="user.id")
    workspace_id: int = Field(index=True, foreign_key="workspace.id")
    role: str = Field(default=ROLE_MEMBER)
    created_at: datetime = Field(default_factory=_utcnow)


class Invitation(SQLModel, table=True):
    """A pending offer to join a workspace.

    Keyed by EMAIL rather than by user id, so someone can be invited before they
    have an account. Only the token's hash is stored, exactly as for password
    resets — a leaked database must not hand over usable invitations.
    """

    id: int | None = Field(default=None, primary_key=True)
    workspace_id: int = Field(index=True, foreign_key="workspace.id")
    email: str = Field(index=True)
    role: str = Field(default=ROLE_MEMBER)
    token_hash: str = Field(index=True, unique=True)
    invited_by_user_id: int = Field(foreign_key="user.id")
    expires_at: datetime
    created_at: datetime = Field(default_factory=_utcnow)
    accepted_at: datetime | None = None
    revoked_at: datetime | None = None


class Paper(SQLModel, table=True):
    """One uploaded document belonging to a WORKSPACE.

    The FAISS chunks live on disk (per-workspace index) tagged with `paper_id`;
    this row is the catalogue entry that lets us list a library and map a
    citation's paper_id back to a human title. `paper_id` is unique WITHIN a
    workspace.

    `user_id` is retained and now means "who uploaded it" — not who may see it,
    which is decided by workspace membership. Deliberately not renamed: a column
    rename buys nothing here and complicates the migration of live data.
    """

    id: int | None = Field(default=None, primary_key=True)
    workspace_id: int = Field(default=0, index=True, foreign_key="workspace.id")
    user_id: int = Field(index=True, foreign_key="user.id")  # uploader
    paper_id: str = Field(index=True)  # slug used to tag chunks + name the file
    title: str                         # human-readable (original filename stem)
    filename: str                      # original upload filename
    n_chunks: int                      # how many chunks this paper contributed
    created_at: datetime = Field(default_factory=_utcnow)


class PasswordResetToken(SQLModel, table=True):
    """A one-time, expiring permission to set a new password.

    Only the token's HASH is stored, for the same reason passwords are hashed: a
    leaked database must not hand over working reset links.

    Single use is enforced by `used_at` rather than by deleting the row, so a
    replayed link is distinguishable from one that never existed — useful when
    someone reports that their reset "didn't work".
    """

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="user.id")
    token_hash: str = Field(index=True, unique=True)
    expires_at: datetime
    created_at: datetime = Field(default_factory=_utcnow)
    used_at: datetime | None = None


# Indexing job states. `failed` is a first-class outcome, not an exception that
# vanished: once indexing leaves the request there is no HTTP status left to
# carry the bad news, so the row has to.
JOB_QUEUED = "queued"
JOB_RUNNING = "running"
JOB_DONE = "done"
JOB_FAILED = "failed"


class IndexJob(SQLModel, table=True):
    """One document waiting to be, or being, indexed.

    Indexing used to run inside the upload request: parse, OCR, embed, write.
    A large document therefore raced the proxy timeout, and the user's only
    feedback was a spinner that either finished or died with no explanation.

    This row is what replaces the HTTP response. It exists BEFORE the work
    starts, so a job that is queued, running, crashed, or lost to a worker
    restart is all visible — the alternative is an upload that silently never
    appears in the library with nothing anywhere saying why.

    `ran_inline` records that no queue was reachable and the work happened in
    the request after all. That is a real difference in behaviour (the request
    blocked, the timeout risk came back) and the log should not pretend
    otherwise.
    """

    id: int | None = Field(default=None, primary_key=True)
    workspace_id: int = Field(index=True, foreign_key="workspace.id")
    user_id: int = Field(index=True, foreign_key="user.id")  # who uploaded it

    paper_id: str          # the slug reserved for it; the stored file is named this
    filename: str          # what the user called it
    title: str
    suffix: str

    status: str = Field(default=JOB_QUEUED, index=True)
    error: str | None = None       # user-facing reason, when status is failed
    n_chunks: int = 0
    ran_inline: bool = Field(default=False)

    created_at: datetime = Field(default_factory=_utcnow, index=True)
    started_at: datetime | None = None
    finished_at: datetime | None = None


class RefreshToken(SQLModel, table=True):
    """One issued refresh token, so a session can be ENDED and not merely expire.

    A JWT alone cannot be revoked — the signature stays valid until `exp`, which
    is 30 days here. That is the whole reason this row exists: the token's power
    now comes from a server-side record, so logging out, or detecting a theft,
    can take it away immediately.

    Rotation: every refresh spends the presented token (`revoked_at`) and issues
    a fresh one, linked through `replaced_by_id`. That chain is what makes theft
    DETECTABLE rather than merely possible. A refresh token is used once, so a
    second use of an already-spent one means two parties hold it — the real
    owner and someone else. Which one is which is unknowable, so the whole family
    is revoked and both must log in again. An unnecessary re-login is a much
    smaller harm than a month of undetected access.

    Only the hash is stored, exactly as for password resets and invitations.
    """

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="user.id")
    token_hash: str = Field(index=True, unique=True)
    expires_at: datetime
    created_at: datetime = Field(default_factory=_utcnow)
    # Set when the token is spent by a refresh, by a logout, or by the mass
    # revocation that follows a detected reuse. One column for all three: what
    # matters at the check is that it is no longer live.
    revoked_at: datetime | None = None
    replaced_by_id: int | None = Field(default=None, foreign_key="refreshtoken.id")


class AnswerLog(SQLModel, table=True):
    """One answered question, with the complete evidence it was built from.

    THIS IS THE PRODUCT'S ACTUAL CLAIM. "Auditable answers over private corpora"
    is a slogan until every answer can be pulled back up months later showing
    which passages produced it, how they were ranked, and what generated the
    prose. A citation the user can click proves the passage exists; this proves
    the answer came FROM it.

    Written after generation, never during — a failed answer should not leave a
    log entry claiming it succeeded. Logging failures are swallowed for the same
    reason the warm-up is non-fatal: losing an audit row is bad, but failing a
    user's question because the audit write failed is worse.

    On reproducibility, precisely: retrieval here is deterministic (fixed
    embeddings, exact FAISS search, deterministic rerank), so the same question
    against the same index returns the same passages. It does NOT follow that
    re-running months later reproduces the answer — the library changes as
    documents are added and removed, and that silently changes retrieval.
    `index_fingerprint` is what makes the difference detectable instead of
    invisible: if it differs, the old answer cannot be reproduced and the log
    says so rather than implying otherwise.
    """

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="user.id")   # who asked
    workspace_id: int = Field(default=0, index=True, foreign_key="workspace.id")
    created_at: datetime = Field(default_factory=_utcnow, index=True)

    question: str                      # exactly what the user typed
    query: str                         # the standalone question after condensing
                                       # a follow-up; what generation received
    answer: str = Field(sa_column=Column(Text))

    # What retrieval and reranking ACTUALLY ran on. With query expansion this is
    # `query` plus a hypothetical answer the LLM invented, and recording it is
    # what keeps the reproducibility claim true: the hypothetical is generated
    # by a model, so without it in the log there is no way to tell whether a
    # different result came from a changed library or a differently-worded
    # hypothetical. Nullable because entries written before expansion existed
    # have no third query, and back-filling one would be a fabrication.
    retrieval_query: str | None = Field(default=None, sa_column=Column(Text, nullable=True))
    expansion_mode: str = Field(default="none")

    # The evidence chain: a JSON array of citations, each with paper_id, page,
    # unit, chunk_index, faiss_id, char span, and both stage scores. Stored as
    # JSON text rather than a child table because it is written once, read whole,
    # and never queried by field — a join would buy nothing and cost a migration.
    citations_json: str = Field(sa_column=Column(Text))

    # What produced the prose, and how retrieval was configured. Without these a
    # log entry cannot explain why an answer differed from one taken yesterday.
    model: str
    temperature: float
    k: int
    candidates: int
    papers_filter: str | None = None    # JSON array, or None for "whole library"

    # Fingerprint of the index state this answer was retrieved from.
    index_fingerprint: str = Field(default="", index=True)
    n_chunks_indexed: int = 0
