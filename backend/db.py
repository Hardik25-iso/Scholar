"""Database engine + session wiring (SQLModel; SQLite locally, Postgres deployed).

init_db() creates tables on startup. get_session() is a FastAPI dependency that
hands each request a session and closes it afterwards.
"""
from collections.abc import Iterator
from pathlib import Path

from sqlmodel import Session, SQLModel, create_engine

from backend.config import settings

IS_SQLITE = settings.database_url.startswith("sqlite")


def _engine_options() -> dict:
    if IS_SQLITE:
        # check_same_thread=False: SQLite + FastAPI's threadpool touch the
        # connection from different threads; this is the standard, safe
        # setting for that pattern. SQLite-only — psycopg rejects the option.
        return {"connect_args": {"check_same_thread": False}}
    # A hosted Postgres (Neon) suspends its compute after 5 idle minutes and
    # drops every connection when it does. pre_ping replaces a dead pooled
    # connection instead of failing the request that drew it, and recycling
    # under the suspend window means most of them are never found dead at all.
    # A small pool: one instance, a handful of threads, and a free plan.
    return {"pool_pre_ping": True, "pool_recycle": 240, "pool_size": 3, "max_overflow": 2}


engine = create_engine(settings.database_url, echo=False, **_engine_options())

if not IS_SQLITE:
    from sqlalchemy import event

    @event.listens_for(engine, "connect")
    def _utc_session(dbapi_connection, _record) -> None:
        """Pin every session to UTC.

        Timestamps are timezone-aware in Python and stored in plain TIMESTAMP
        columns, which Postgres fills by converting to the SESSION time zone
        and dropping the offset. auth.py reads them back as naive UTC (an
        expiry check does `.replace(tzinfo=utc)`), so a server whose default
        zone is not UTC would silently shift every expiry by its offset —
        reset links valid for hours too long, or already expired on arrival.
        SQLite stores what it is given, which is why this never showed up.
        """
        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET TIME ZONE 'UTC'")
        dbapi_connection.commit()


def init_db() -> None:
    """Create the data/ dir and all tables if they don't exist yet."""
    # Ensure the folder for the SQLite file exists (data/ is git-ignored).
    if IS_SQLITE:
        db_path = settings.database_url.split("///")[-1]
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    # Import table models so SQLModel.metadata knows about them before create_all.
    import backend.db_models  # noqa: F401

    SQLModel.metadata.create_all(engine)


def get_session() -> Iterator[Session]:
    with Session(engine) as session:
        yield session
