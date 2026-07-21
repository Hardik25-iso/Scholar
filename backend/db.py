"""Database engine + session wiring (SQLite via SQLModel).

init_db() creates tables on startup. get_session() is a FastAPI dependency that
hands each request a session and closes it afterwards.
"""
from collections.abc import Iterator
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from backend.config import settings

# check_same_thread=False: SQLite + FastAPI's threadpool touch the connection
# from different threads; this is the standard, safe setting for that pattern.
engine = create_engine(
    settings.database_url,
    echo=False,
    connect_args={"check_same_thread": False},
)


def init_db() -> None:
    """Create the data/ dir and all tables if they don't exist yet."""
    # Ensure the folder for the SQLite file exists (data/ is git-ignored).
    if settings.database_url.startswith("sqlite"):
        db_path = settings.database_url.split("///")[-1]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    # Import table models so SQLModel.metadata knows about them before create_all.
    import backend.db_models  # noqa: F401

    SQLModel.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
