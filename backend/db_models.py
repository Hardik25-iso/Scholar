"""SQLModel database tables.

Kept separate from models.py (the Pydantic API-boundary models) so the two
concerns don't blur: this file is "what's stored", models.py is "what crosses
the wire". A User here never leaves the backend with its hashed_password.
"""
from datetime import datetime, timezone

from sqlalchemy import Column, Text
from sqlmodel import Field, SQLModel


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(SQLModel, table=True):
    """A registered account. Passwords are stored only as a bcrypt hash."""

    id: int | None = Field(default=None, primary_key=True)
    email: str = Field(index=True, unique=True)
    hashed_password: str  # bcrypt hash — never the plaintext (set in Slice B)
    created_at: datetime = Field(default_factory=_utcnow)


class Paper(SQLModel, table=True):
    """One uploaded paper belonging to a user.

    The FAISS chunks live on disk (per-user index) tagged with `paper_id`; this
    row is the catalogue entry that lets us list a library and map a citation's
    paper_id back to a human title. `paper_id` is unique WITHIN a user.
    """

    id: int | None = Field(default=None, primary_key=True)
    user_id: int = Field(index=True, foreign_key="user.id")
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
    user_id: int = Field(index=True, foreign_key="user.id")
    created_at: datetime = Field(default_factory=_utcnow, index=True)

    question: str                      # exactly what the user typed
    query: str                         # what retrieval actually ran on — differs
                                       # from `question` for a condensed follow-up
    answer: str = Field(sa_column=Column(Text))

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
