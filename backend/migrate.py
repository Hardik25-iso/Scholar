"""One-way migration: per-user libraries become per-workspace libraries.

    python -m backend.migrate --dry-run     # say exactly what would happen
    python -m backend.migrate               # do it

THIS IS THE HIGHEST-RISK STEP IN THE PROJECT. It moves every user's documents on
disk and rewrites the rows that point at them. Get it wrong and a library is
still on disk but unreachable, which is indistinguishable from data loss to the
person who owned it.

So it is built to the rules that make a migration survivable:

  1. IDEMPOTENT. Every step checks whether it has already been done. A run that
     dies halfway can simply be run again — the second run finishes the job
     rather than duplicating it.
  2. DRY RUN FIRST. `--dry-run` performs no writes at all and prints the exact
     plan, so the risky version is never the first one anybody sees.
  3. COPY, THEN VERIFY, THEN REMOVE. Files are copied to the new location and
     the copy is checked before the old directory is deleted. An interruption
     leaves both copies, never neither.
  4. BACKUP FIRST, and say so. Refuses to run without --i-have-a-backup unless
     there is nothing at risk.

What it does:
  - creates a personal workspace for every user who lacks one, with them as owner
  - points every Paper and AnswerLog at that workspace
  - moves data/users/<user_id>/ to data/workspaces/<workspace_id>/
"""
import argparse
import shutil
import sys
from pathlib import Path

from sqlalchemy import inspect, text
from sqlmodel import Session, SQLModel, select

from backend import library
from backend.db import engine, init_db
from backend.db_models import AnswerLog, Membership, Paper, User, Workspace
from backend.workspaces import ensure_personal_workspace


def _existing_columns(table: str) -> set[str]:
    inspector = inspect(engine)
    if table not in inspector.get_table_names():
        return set()
    return {c["name"] for c in inspector.get_columns(table)}


def add_missing_columns(dry_run: bool) -> list[str]:
    """Add the workspace columns to tables that predate them.

    SQLModel's create_all creates missing TABLES but never alters existing ones,
    so a database written before workspaces existed keeps its old shape and
    every query mentioning workspace_id fails. These are plain ADD COLUMNs,
    which SQLite performs without rewriting the table.
    """
    wanted = {
        "paper": [("workspace_id", "INTEGER NOT NULL DEFAULT 0")],
        "answerlog": [
            ("workspace_id", "INTEGER NOT NULL DEFAULT 0"),
            # Nullable with no default: an entry written before query expansion
            # existed genuinely has no third query, and inventing one would put
            # a fabricated value in an audit trail.
            ("retrieval_query", "TEXT"),
            ("expansion_mode", "VARCHAR NOT NULL DEFAULT 'none'"),
        ],
        "user": [("current_workspace_id", "INTEGER")],
    }
    actions: list[str] = []
    with engine.begin() as conn:
        for table, columns in wanted.items():
            present = _existing_columns(table)
            if not present:
                continue  # table does not exist yet; create_all will make it
            for name, ddl in columns:
                if name in present:
                    continue
                actions.append(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}")
                if not dry_run:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {name} {ddl}"))
    return actions


def _move_library(user_id: int, workspace_id: int, dry_run: bool) -> str | None:
    """Copy one user's library to its workspace location, then remove the old.

    Copy-verify-remove rather than a rename: a rename that fails partway across
    a filesystem boundary can leave a half-moved tree with no way to tell which
    half is authoritative.
    """
    source = library.LEGACY_DATA_ROOT / str(user_id)
    target = library.DATA_ROOT / str(workspace_id)

    if not source.exists():
        return None
    if workspace_id and target.exists() and any(target.iterdir()):
        return f"  library for user {user_id}: already at workspaces/{workspace_id} (skipped)"
    if dry_run:
        # A plan someone reads before doing something irreversible must not
        # print a destination that does not exist. 0 means "workspace not
        # created yet", so say that rather than showing workspaces/0.
        destination = f"workspaces/{workspace_id}" if workspace_id else "workspaces/<new>"
        return f"  library for user {user_id}: users/{user_id} -> {destination}"

    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, target, dirs_exist_ok=True)

    # Verify before deleting anything: same file count, same total bytes.
    src_files = sorted(p.relative_to(source) for p in source.rglob("*") if p.is_file())
    dst_files = sorted(p.relative_to(target) for p in target.rglob("*") if p.is_file())
    if src_files != dst_files:
        raise RuntimeError(
            f"copy verification FAILED for user {user_id}: "
            f"{len(src_files)} source files vs {len(dst_files)} copied. "
            f"Nothing has been deleted; the original is still at {source}."
        )
    for rel in src_files:
        if (source / rel).stat().st_size != (target / rel).stat().st_size:
            raise RuntimeError(
                f"copy verification FAILED for user {user_id}: size mismatch on {rel}. "
                f"Nothing has been deleted; the original is still at {source}."
            )

    shutil.rmtree(source, ignore_errors=True)
    return f"  library for user {user_id}: moved {len(src_files)} files -> workspaces/{workspace_id}"


def _plan(actions: list[str]) -> list[str]:
    """Describe what a real run would do, WITHOUT touching anything.

    Reads through raw SQL rather than the ORM for the same reason the safety
    gate does: a dry run must work on a pre-migration database, and every ORM
    query names columns that database does not have yet. A dry run that only
    works after the migration is worthless — it is the one command someone runs
    when they are nervous.
    """
    with engine.connect() as conn:
        users = list(conn.execute(text("SELECT id, email FROM user ORDER BY id")))
        actions.append(f"{len(users)} user(s) to migrate")

        has_workspaces = "workspace" in inspect(engine).get_table_names()
        for user_id, email in users:
            existing_id = None
            if has_workspaces:
                row = conn.execute(text(
                    "SELECT w.id FROM workspace w JOIN membership m ON m.workspace_id = w.id "
                    "WHERE m.user_id = :uid AND w.is_personal = 1"
                ), {"uid": user_id}).first()
                existing_id = row[0] if row else None

            target = existing_id if existing_id else f"<new for user {user_id}>"
            actions.append(f"  user {user_id} ({email}): personal workspace {target}")

            n_papers = conn.execute(
                text("SELECT COUNT(*) FROM paper WHERE user_id = :uid"), {"uid": user_id}
            ).scalar() or 0
            actions.append(f"    {n_papers} paper(s) would be re-pointed")

            moved = _move_library(user_id, existing_id or 0, dry_run=True)
            if moved:
                actions.append(moved)
    return actions


def migrate(dry_run: bool = False) -> list[str]:
    """Run every step. Safe to run repeatedly."""
    actions: list[str] = []

    schema_actions = add_missing_columns(dry_run)
    actions.extend(schema_actions)

    if dry_run:
        actions.append("CREATE TABLE IF NOT EXISTS workspace, membership, invitation")
        _plan(actions)
        _note_legacy_leftovers(actions, dry_run=True)
        return actions

    init_db()  # creates workspace / membership / invitation tables

    with Session(engine) as session:
        users = list(session.exec(select(User)).all())
        actions.append(f"{len(users)} user(s) to migrate")

        for user in users:
            workspace = ensure_personal_workspace(session, user)
            actions.append(
                f"  user {user.id} ({user.email}) -> workspace {workspace.id} ({workspace.name})"
            )

            # Re-point rows that still carry the pre-workspace default of 0.
            papers = session.exec(
                select(Paper).where(Paper.user_id == user.id, Paper.workspace_id == 0)
            ).all()
            for paper in papers:
                paper.workspace_id = workspace.id
                session.add(paper)

            logs = session.exec(
                select(AnswerLog).where(AnswerLog.user_id == user.id, AnswerLog.workspace_id == 0)
            ).all()
            for entry in logs:
                entry.workspace_id = workspace.id
                session.add(entry)
            session.commit()
            actions.append(f"    {len(papers)} paper(s), {len(logs)} answer log(s) re-pointed")

            moved = _move_library(user.id, workspace.id, dry_run=False)
            if moved:
                actions.append(moved)

    _note_legacy_leftovers(actions, dry_run=False)
    return actions


def _note_legacy_leftovers(actions: list[str], dry_run: bool) -> list[str]:
    """Tidy the legacy root only if it is genuinely empty.

    Never a recursive delete: a directory this migration did not account for
    must survive and be reported, rather than be silently destroyed because it
    happened to sit in the wrong place.
    """
    legacy = library.LEGACY_DATA_ROOT
    if legacy.exists() and not any(legacy.iterdir()):
        if not dry_run:
            legacy.rmdir()
        actions.append(f"removed empty legacy directory {legacy}")
    elif legacy.exists():
        remaining = [p.name for p in legacy.iterdir()]
        actions.append(f"NOTE: {legacy} still contains {remaining} — left in place, inspect manually")
    return actions


def _anything_at_risk() -> bool:
    """Is there real data here that a bad migration could damage?

    Deliberately RAW SQL, not the ORM. The whole point of this check is to run
    against a pre-migration database — one whose `paper` table has no
    `workspace_id` column yet — and an ORM query names every mapped column, so
    `select(Paper)` blows up with "no such column" on exactly the database this
    is meant to protect. The safety gate failing on the unsafe case is the worst
    possible way for it to fail.
    """
    try:
        with engine.connect() as conn:
            if conn.execute(text("SELECT 1 FROM paper LIMIT 1")).first() is not None:
                return True
    except Exception:  # noqa: BLE001 — no table yet is simply "nothing at risk"
        pass
    return library.LEGACY_DATA_ROOT.exists() and any(library.LEGACY_DATA_ROOT.iterdir())


def main() -> int:
    parser = argparse.ArgumentParser(
        prog="python -m backend.migrate",
        description="Migrate per-user libraries to per-workspace libraries.",
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="print the plan and change nothing")
    parser.add_argument("--i-have-a-backup", action="store_true",
                        help="confirm a backup exists (python -m backend.backup create)")
    args = parser.parse_args()

    if not args.dry_run and not args.i_have_a_backup and _anything_at_risk():
        print(
            "Refusing to migrate real data without a backup.\n"
            "  1. python -m backend.backup create backups/\n"
            "  2. python -m backend.migrate --dry-run\n"
            "  3. python -m backend.migrate --i-have-a-backup",
            file=sys.stderr,
        )
        return 1

    print("DRY RUN — nothing will be changed\n" if args.dry_run else "Migrating...\n")
    try:
        for action in migrate(dry_run=args.dry_run):
            print(action)
    except RuntimeError as exc:
        print(f"\nSTOPPED: {exc}", file=sys.stderr)
        return 1
    print("\nDone." if not args.dry_run else "\nDry run complete — nothing was changed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
