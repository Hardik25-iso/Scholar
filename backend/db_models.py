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
