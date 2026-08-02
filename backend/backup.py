"""Backup and restore of everything a user would be devastated to lose.

    python -m backend.backup create backups/           # -> scholar-<stamp>.tar.gz
    python -m backend.backup verify backups/scholar-....tar.gz
    python -m backend.backup restore backups/scholar-....tar.gz

WHY THIS IS A MODULE AND NOT A README PARAGRAPH. "Take regular backups" is not a
feature; a backup nobody has ever restored is a guess. This exists so restoring
is a tested code path — `backend/tests/test_backup.py` writes a library, backs
it up, destroys the original and restores it, then asserts the documents are
still retrievable. That test is the actual deliverable.

WHAT IS IN THE ARCHIVE
  meta.json      what version wrote it, when, and what it contains
  scholar.db     the SQLite database (users, papers, answer logs)
  users/         every per-user library: stored files, FAISS index, lexical index

Both parts are needed and they must move together. The database says a paper
exists; the data tree holds its vectors. Restoring one without the other gives a
library that lists documents it cannot search, or an index full of orphans.

NOT IN SCOPE: a non-SQLite database. If `database_url` points at Postgres, that
server has its own backup story and pretending to handle it here would be worse
than saying so — `create` refuses rather than writing an archive that silently
omits half the data.
"""
import argparse
import json
import shutil
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from backend import library
from backend.config import settings

FORMAT_VERSION = 1
META_NAME = "meta.json"
DB_NAME = "scholar.db"
USERS_DIR = "users"


class BackupError(RuntimeError):
    """Raised when an archive cannot be written or is not safe to restore."""


def sqlite_path() -> Path:
    """The database file, or raise if this deployment is not on SQLite."""
    url = settings.database_url
    if not url.startswith("sqlite"):
        raise BackupError(
            f"database_url is {url.split('://')[0]!r}, not sqlite — back that server up with its "
            "own tooling. This tool would otherwise write an archive missing half your data."
        )
    return Path(url.split("///")[-1]).resolve()


def _stamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def create(dest_dir: str | Path) -> Path:
    """Write a compressed archive of the database and every user library."""
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    db = sqlite_path()
    users = Path(library.DATA_ROOT)

    meta = {
        "format_version": FORMAT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "database": db.name,
        "has_database": db.exists(),
        "n_user_libraries": len(list(users.iterdir())) if users.exists() else 0,
    }

    archive = dest_dir / f"scholar-{_stamp()}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        with tempfile.TemporaryDirectory() as tmp:
            meta_file = Path(tmp) / META_NAME
            meta_file.write_text(json.dumps(meta, indent=2), encoding="utf-8")
            tar.add(meta_file, arcname=META_NAME)
        if db.exists():
            tar.add(db, arcname=DB_NAME)
        if users.exists():
            tar.add(users, arcname=USERS_DIR)
    return archive


def read_meta(archive: str | Path) -> dict:
    """The archive's manifest, without extracting anything else."""
    try:
        with tarfile.open(archive, "r:gz") as tar:
            # extractfile RAISES KeyError for a missing member (it returns None
            # only for a non-regular file such as a directory), so both have to
            # be handled or a stray tarball crashes instead of being rejected.
            member = tar.extractfile(META_NAME)
            if member is None:
                raise KeyError(META_NAME)
            return json.loads(member.read().decode("utf-8"))
    except (KeyError, tarfile.TarError, ValueError) as exc:
        raise BackupError(f"{archive} is not a readable Scholar backup: {exc}") from exc


def verify(archive: str | Path) -> dict:
    """Check an archive is readable and self-consistent before trusting it.

    Cheap, and the only thing standing between "we have backups" and "we have
    files that might be backups".
    """
    meta = read_meta(archive)
    if meta.get("format_version") != FORMAT_VERSION:
        raise BackupError(
            f"archive format v{meta.get('format_version')} != v{FORMAT_VERSION} supported here"
        )
    with tarfile.open(archive, "r:gz") as tar:
        names = tar.getnames()
    if meta.get("has_database") and DB_NAME not in names:
        raise BackupError(f"manifest claims a database but {DB_NAME} is missing from the archive")
    return {**meta, "n_entries": len(names)}


def _is_safe(name: str) -> bool:
    """Reject absolute paths and traversal — an archive is untrusted input.

    A malicious or corrupt tarball can otherwise write anywhere the process can
    (the classic tar path-traversal). Python 3.12 has `filter="data"` for this;
    the explicit check keeps the guarantee visible rather than relying on a
    default that changed across versions.
    """
    path = Path(name)
    return not path.is_absolute() and ".." not in path.parts


def restore(archive: str | Path, force: bool = False) -> dict:
    """Replace the database and user libraries with the archive's contents.

    Destructive by nature, so it refuses to run over existing data unless
    `force` is set — the failure mode to avoid is someone restoring a stale
    backup over a live library while trying to inspect it.
    """
    meta = verify(archive)
    db = sqlite_path()
    users = Path(library.DATA_ROOT)

    if not force and (db.exists() or (users.exists() and any(users.iterdir()))):
        raise BackupError(
            "refusing to restore over existing data — pass force=True (or --force) "
            "if you really mean to replace the current database and libraries"
        )

    with tarfile.open(archive, "r:gz") as tar:
        members = [m for m in tar.getmembers() if _is_safe(m.name)]
        if len(members) != len(tar.getmembers()):
            raise BackupError("archive contains unsafe paths — refusing to extract")
        with tempfile.TemporaryDirectory() as tmp:
            # filter="data" is the modern extraction guard (default from 3.14):
            # it refuses absolute paths, traversal, links pointing outside the
            # destination, and device files. Belt and braces with the _is_safe
            # check above — that one states the intent, this one enforces cases
            # a hand-rolled check would miss.
            tar.extractall(tmp, members=members, filter="data")
            staged = Path(tmp)

            staged_db = staged / DB_NAME
            if staged_db.exists():
                db.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(staged_db, db)

            staged_users = staged / USERS_DIR
            if staged_users.exists():
                # Replace wholesale rather than merging: a merge would leave
                # documents that were deleted after the backup was taken, and a
                # restore is supposed to reproduce a moment, not union with one.
                shutil.rmtree(users, ignore_errors=True)
                shutil.copytree(staged_users, users)

    library._retrievers.clear()  # in-process caches now point at replaced files
    return meta


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m backend.backup", description=__doc__.split("\n")[0])
    sub = parser.add_subparsers(dest="command", required=True)

    p_create = sub.add_parser("create", help="write a new backup archive")
    p_create.add_argument("dest", help="directory to write the archive into")

    p_verify = sub.add_parser("verify", help="check an archive is readable and consistent")
    p_verify.add_argument("archive")

    p_restore = sub.add_parser("restore", help="replace current data with an archive")
    p_restore.add_argument("archive")
    p_restore.add_argument("--force", action="store_true",
                           help="allow restoring over existing data (destructive)")

    args = parser.parse_args()
    try:
        if args.command == "create":
            archive = create(args.dest)
            meta = verify(archive)
            size_mb = archive.stat().st_size / 1_048_576
            print(f"wrote {archive} ({size_mb:.1f} MB, {meta['n_user_libraries']} user libraries)")
            print("verified OK")
        elif args.command == "verify":
            meta = verify(args.archive)
            print(json.dumps(meta, indent=2))
        else:
            meta = restore(args.archive, force=args.force)
            print(f"restored from {args.archive} (taken {meta['created_at']})")
    except BackupError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
