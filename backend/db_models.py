"""SQLModel database tables.

Kept separate from models.py (the Pydantic API-boundary models) so the two
concerns don't blur: this file is "what's stored", models.py is "what crosses
the wire". A User here never leaves the backend with its hashed_password.
"""
from datetime import datetime, timezone

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
    paper_id: str = Field(index=True)  # slug used to tag chunks + name the PDF
    title: str                         # human-readable (original filename stem)
    filename: str                      # original upload filename
    n_chunks: int                      # how many chunks this paper contributed
    created_at: datetime = Field(default_factory=_utcnow)
