"""Per-workspace document library: storage layout, indexing, retrieval scoping.

Each workspace gets an isolated tree in the blob store (blobstore.py):
    {id}/papers/<paper_id>.<ext>        — the original upload, in its own format
    {id}/index/<version>/index.faiss    — that workspace's FAISS index (ALL its documents)
    {id}/index/<version>/chunks.json    — chunk metadata, each chunk tagged by paper_id
    {id}/index/<version>/lexical.db     — the FTS5 mirror of the same chunks

One index PER WORKSPACE, not one shared index filtered by a query parameter.
That makes isolation a property of the storage layout rather than of remembering
a WHERE clause: retrieval physically cannot return another workspace's passages,
because it never opens their index. A filter can be forgotten in one code path;
a directory cannot.

VERSIONS. An index is never modified in place. A change (a new document, a
deletion) produces a complete new version under a fresh key, and the
WorkspaceIndex row is switched to it in one transaction. The blob store may be
remote object storage with no atomic multi-file write and no compare-and-swap,
so this is the only way a reader can never see a FAISS index from one version
beside a chunks.json from another — and the only way a crash halfway through an
upload leaves the library exactly as it was. See db_models.WorkspaceIndex.

Reads work on a LOCAL copy of the current version (CACHE_ROOT), fetched on first
use and reused while it stays current. The cache is disposable: a restart on a
host with an ephemeral disk simply fetches it again.

Personal libraries are ordinary workspaces flagged `is_personal`, so there is
exactly one storage and retrieval path rather than two that drift apart.
"""
import logging
import os
import re
import shutil
import tempfile
import threading
import uuid
from collections import OrderedDict
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlmodel import Session

from backend import audit, lexical
from backend.blobstore import BlobStore, LocalBlobStore, SupabaseBlobStore
from backend.chunker import chunk_pages
from backend.config import settings
from backend.embedder import embed
from backend.parser import DocumentTooLarge, extract_pages, unit_for
from backend.retriever import Retriever
from backend.store import append_to_store, remove_paper

log = logging.getLogger(__name__)

# Where a LOCAL blob store keeps libraries. Configurable, because the fallback
# is INSIDE the source tree: a deployment that replaces the source on redeploy
# would take every user's library with it, with no error to notice. Set
# DATA_ROOT to a mounted volume, or use STORAGE_BACKEND=supabase.
DEFAULT_DATA_ROOT = Path(__file__).parent.parent / "data" / "workspaces"
DATA_ROOT = Path(settings.data_root) if settings.data_root else DEFAULT_DATA_ROOT

# Where libraries lived before workspaces existed. Only the migration reads it.
LEGACY_DATA_ROOT = Path(settings.data_root).parent / "users" if settings.data_root \
    else Path(__file__).parent.parent / "data" / "users"

# Local working copies. Everything here can be re-fetched from the blob store.
CACHE_ROOT = Path(settings.cache_dir) if settings.cache_dir \
    else Path(tempfile.gettempdir()) / "scholar-cache"

INDEX_FILES = ("index.faiss", "chunks.json", "lexical.db")

# How many attempts a write makes when another writer switched the version
# between its read and its commit. Each retry re-applies the change on top of
# the newer version, so nothing is lost — it only costs a re-download.
_PUBLISH_ATTEMPTS = 3

# Advisory-lock namespace (Postgres), so this lock cannot collide with any
# other use of pg_advisory_lock on the same database.
_LOCK_NAMESPACE = 0x5C40


# ——— the blob store ———

_supabase: SupabaseBlobStore | None = None


def blob_store() -> BlobStore:
    """The configured store. Local is rebuilt per call so DATA_ROOT stays
    patchable (the tests redirect it); the remote client is reused."""
    global _supabase
    if settings.storage_backend == "supabase":
        if _supabase is None:
            _supabase = SupabaseBlobStore(settings.supabase_url, settings.supabase_service_key,
                                          settings.supabase_bucket)
        return _supabase
    return LocalBlobStore(DATA_ROOT)


def _index_key(workspace_id: int, version: str, name: str = "") -> str:
    # version "" is the pre-versioning layout, adopted in place.
    base = f"{workspace_id}/index/{version}/" if version else f"{workspace_id}/index/"
    return base + name


def workspace_papers_dir(workspace_id: int) -> Path:
    """Where the LOCAL store keeps a workspace's originals. Only meaningful with
    STORAGE_BACKEND=local; code that must work with any backend goes through
    blob_store() / stored_path() instead."""
    return DATA_ROOT / str(workspace_id) / "papers"


def paper_key(workspace_id: int, paper_id: str, suffix: str) -> str:
    return f"{workspace_id}/papers/{paper_id}{suffix}"


def slugify(name: str) -> str:
    """Filesystem/id-safe slug from a filename stem."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "paper"


# ——— the current-version pointer ———


def _engine():
    # Imported lazily: backend.db builds the engine from settings at import time.
    from backend.db import engine

    return engine


def _current(session: Session, workspace_id: int):
    """The workspace's WorkspaceIndex row, adopting a pre-versioning library.

    A library written before versioning existed has its files directly under
    `{id}/index/` and no row. Rather than demand a migration, the first read
    records that layout as version "" — after which it is an ordinary version
    and the next write replaces it like any other.
    """
    from backend.db_models import WorkspaceIndex

    row = session.get(WorkspaceIndex, workspace_id)
    if row is not None:
        return row
    legacy = blob_store().list_keys(_index_key(workspace_id, ""))
    if not all(_index_key(workspace_id, "", f) in legacy for f in ("index.faiss", "chunks.json")):
        return None
    with tempfile.TemporaryDirectory(dir=_cache_dir()) as tmp:
        if not blob_store().get_to_file(_index_key(workspace_id, "", "chunks.json"),
                                        Path(tmp) / "chunks.json"):
            return None
        fingerprint, n_chunks = audit.index_fingerprint(tmp)
    row = WorkspaceIndex(workspace_id=workspace_id, version="",
                         fingerprint=fingerprint, n_chunks=n_chunks)
    session.add(row)
    try:
        session.commit()
    except IntegrityError:          # another thread adopted it first — same answer
        session.rollback()
        return session.get(WorkspaceIndex, workspace_id)
    session.refresh(row)
    return row


def current_version(workspace_id: int) -> tuple[str, str, int] | None:
    """(version, fingerprint, n_chunks) of the live index, or None if empty."""
    with Session(_engine()) as session:
        row = _current(session, workspace_id)
        return (row.version, row.fingerprint, row.n_chunks) if row else None


def current_fingerprint(workspace_id: int) -> str:
    state = current_version(workspace_id)
    return state[1] if state else ""


# ——— the local cache ———


def _cache_dir() -> Path:
    path = CACHE_ROOT / ".work"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _version_dir(workspace_id: int, version: str) -> Path:
    return CACHE_ROOT / str(workspace_id) / (version or "_legacy")


def _fetch_version(workspace_id: int, version: str, dest: Path) -> None:
    """Download one complete version into `dest` (which must not exist yet)."""
    dest.mkdir(parents=True)
    for name in INDEX_FILES:
        found = blob_store().get_to_file(_index_key(workspace_id, version, name), dest / name)
        # lexical.db may be absent in a library indexed before hybrid search;
        # the FAISS half is not optional.
        if not found and name != "lexical.db":
            raise FileNotFoundError(f"workspace {workspace_id} version {version!r} is missing {name}")


def local_index_dir(workspace_id: int, version: str) -> Path:
    """A complete local copy of one version, fetched if not already cached.

    Fetched into a private temp directory and renamed into place, so a
    half-downloaded version is never visible under its real name — two
    requests racing to fetch the same version both end with a complete copy.
    """
    final = _version_dir(workspace_id, version)
    if final.exists():
        return final
    tmp = _cache_dir() / f"fetch-{uuid.uuid4().hex}"
    try:
        _fetch_version(workspace_id, version, tmp)
        final.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.replace(tmp, final)
        except OSError:
            if not final.exists():   # a real failure, not a lost race
                raise
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return final


def _prune_cache(workspace_id: int, keep: set[str]) -> None:
    """Drop cached versions other than `keep`. The previous version is kept:
    an in-flight request may still be reading its lexical.db."""
    base = CACHE_ROOT / str(workspace_id)
    if not base.exists():
        return
    names = {(v or "_legacy") for v in keep}
    for child in base.iterdir():
        if child.name not in names:
            shutil.rmtree(child, ignore_errors=True)


def current_index_dir(workspace_id: int) -> Path | None:
    """Local copy of the CURRENT index, or None if nothing is indexed."""
    state = current_version(workspace_id)
    return local_index_dir(workspace_id, state[0]) if state else None


# ——— retrieval ———

# One loaded Retriever per workspace, reused across /ask calls (loading the
# FAISS index + chunks.json every request would be wasteful). Each entry
# remembers the version it was loaded from and is replaced as soon as that
# version stops being current, so it never serves a stale library.
#
# BOUNDED, because each entry holds a whole FAISS index plus its chunk text in
# memory — on a small instance this is a memory setting (RETRIEVER_CACHE_SIZE).
# An OrderedDict used as an LRU: least-recently-used is evicted past the cap,
# and an evicted workspace simply reloads on its next question.
_retrievers: OrderedDict[int, tuple[str, Retriever]] = OrderedDict()
_retrievers_lock = threading.Lock()


def invalidate(workspace_id: int) -> None:
    with _retrievers_lock:
        _retrievers.pop(workspace_id, None)


def get_retriever(workspace_id: int) -> Retriever | None:
    """The workspace's retriever, or None if nothing has been indexed yet.

    The returned Retriever's `store_dir` is the version it was loaded from;
    lexical search and the audit fingerprint must use that directory, so an
    answer is always drawn from — and recorded against — ONE version.
    """
    state = current_version(workspace_id)
    if state is None:
        return None
    version = state[0]

    # Locked: FastAPI serves sync endpoints from a threadpool, so several
    # requests can reach this concurrently and would otherwise race on eviction.
    with _retrievers_lock:
        cached = _retrievers.get(workspace_id)
        if cached is not None and cached[0] == version:
            _retrievers.move_to_end(workspace_id)  # mark recently used
            return cached[1]

    # Fetch and load OUTSIDE the lock — reading a large index would otherwise
    # block every other workspace's lookup for the duration.
    retriever = Retriever(local_index_dir(workspace_id, version))

    with _retrievers_lock:
        # Another thread may have loaded it meanwhile; prefer the existing one
        # so callers never hold two different objects for the same version.
        existing = _retrievers.get(workspace_id)
        if existing is not None and existing[0] == version:
            _retrievers.move_to_end(workspace_id)
            return existing[1]
        _retrievers[workspace_id] = (version, retriever)
        _retrievers.move_to_end(workspace_id)
        while len(_retrievers) > max(1, settings.retriever_cache_size):
            _retrievers.popitem(last=False)  # evict least-recently-used
        return retriever


# ——— writing: build a new version, then switch to it ———

_local_locks: dict[int, threading.Lock] = {}
_local_locks_guard = threading.Lock()


@contextmanager
def _workspace_lock(session: Session, workspace_id: int) -> Iterator[None]:
    """Serialise writers to one workspace.

    Postgres: a transaction-scoped advisory lock, so it covers every process
    and every instance, and is released by the commit or rollback that ends
    the transaction. Elsewhere (SQLite, local development) a per-process lock.

    The lock is an optimisation, not the safety net: the version switch is a
    compare-and-swap that detects a concurrent writer regardless, so two
    processes on SQLite (the API plus an arq worker) still cannot lose an
    update — the loser simply retries on top of the winner's version.
    """
    if session.get_bind().dialect.name == "postgresql":
        session.exec(text("SELECT pg_advisory_xact_lock(:ns, :ws)"),
                     params={"ns": _LOCK_NAMESPACE, "ws": workspace_id})
        yield
        return
    with _local_locks_guard:
        lock = _local_locks.setdefault(workspace_id, threading.Lock())
    with lock:
        yield


def _next_version(previous: str | None) -> str:
    serial = int(previous.split("-")[0]) + 1 if previous and previous[0].isdigit() else 1
    # The random suffix means a version key is NEVER reused, even by a writer
    # that crashed after uploading and was retried with the same serial.
    return f"{serial:06d}-{uuid.uuid4().hex[:8]}"


def _upload_version(workspace_id: int, version: str, work: Path) -> None:
    store = blob_store()
    expected = {}
    for name in INDEX_FILES:
        path = work / name
        if path.exists():
            key = _index_key(workspace_id, version, name)
            store.put(key, path)
            expected[key] = path.stat().st_size
    # Trust, but verify: the switch below makes this version live for every
    # reader, so check that what the store holds is what was written.
    listed = store.list_keys(_index_key(workspace_id, version))
    for key, size in expected.items():
        if listed.get(key) != size:
            raise RuntimeError(f"upload verification failed for {key}: "
                               f"stored {listed.get(key)} bytes, wrote {size}")


def _sweep_versions(workspace_id: int, keep: set[str]) -> None:
    """Best-effort: delete stored versions other than `keep`.

    Run AFTER the commit, so a failure here costs storage, never correctness.
    It also removes orphans left by any earlier write that died before its
    switch — those versions were never referenced by anyone.
    """
    try:
        store = blob_store()
        stale = []
        for key in store.list_keys(f"{workspace_id}/index/"):
            rest = key[len(f"{workspace_id}/index/"):]
            version = rest.split("/")[0] if "/" in rest else ""
            if version not in keep:
                stale.append(key)
        if stale:
            store.delete(stale)
    except Exception:                       # noqa: BLE001 — see docstring
        log.warning("could not sweep old index versions for workspace %s", workspace_id, exc_info=True)


def _publish(workspace_id: int, change: Callable[[Path], None]) -> int:
    """Apply `change` to a private copy of the current index and make the
    result current. Returns the new chunk count (0 = the index is now empty).

    `change` must be safe to run more than once: it is re-applied to a newer
    base if another writer got there first.
    """
    from backend.db_models import WorkspaceIndex

    for _ in range(_PUBLISH_ATTEMPTS):
        work = Path(tempfile.mkdtemp(dir=_cache_dir(), prefix="build-"))
        try:
            with Session(_engine()) as session, _workspace_lock(session, workspace_id):
                current = _current(session, workspace_id)
                base_version = current.version if current else None
                if current is not None:
                    shutil.copytree(local_index_dir(workspace_id, current.version), work,
                                    dirs_exist_ok=True)
                change(work)

                if not (work / "index.faiss").exists():
                    # Nothing left to index (the last document was deleted).
                    if current is not None:
                        session.delete(current)
                        session.commit()
                    invalidate(workspace_id)
                    _sweep_versions(workspace_id, keep=set())
                    _prune_cache(workspace_id, keep=set())
                    return 0

                version = _next_version(base_version)
                _upload_version(workspace_id, version, work)
                fingerprint, n_chunks = audit.index_fingerprint(work)

                # The switch. A compare-and-swap on the version this write was
                # built from: if someone else switched it meanwhile, nothing is
                # changed and the whole write is retried on their version.
                now = datetime.now(timezone.utc)
                if current is None:
                    session.add(WorkspaceIndex(workspace_id=workspace_id, version=version,
                                               fingerprint=fingerprint, n_chunks=n_chunks,
                                               updated_at=now))
                    try:
                        session.commit()
                    except IntegrityError:
                        session.rollback()
                        continue
                else:
                    swapped = session.exec(
                        text("UPDATE workspaceindex SET version = :new, fingerprint = :fp, "
                             "n_chunks = :n, updated_at = :now "
                             "WHERE workspace_id = :ws AND version = :old"),
                        params={"new": version, "fp": fingerprint, "n": n_chunks, "now": now,
                                "ws": workspace_id, "old": base_version},
                    ).rowcount
                    if swapped != 1:
                        session.rollback()
                        continue
                    session.commit()

            # Committed: this version is live. Make the build the cached copy
            # (no point downloading what was just written), then tidy up.
            final = _version_dir(workspace_id, version)
            final.parent.mkdir(parents=True, exist_ok=True)
            if not final.exists():
                os.replace(work, final)
            keep = {version} | ({base_version} if base_version is not None else set())
            _prune_cache(workspace_id, keep)
            _sweep_versions(workspace_id, keep)
            invalidate(workspace_id)
            return n_chunks
        finally:
            shutil.rmtree(work, ignore_errors=True)
    raise RuntimeError(
        f"workspace {workspace_id}: the index changed under {_PUBLISH_ATTEMPTS} consecutive "
        "writes; giving up rather than risk losing a change"
    )


# ——— documents ———

# One indexing run at a time per process. Peak memory during indexing is the
# embedding model's activations plus the document; two runs at once double that
# for no throughput gain on a single CPU, and on a 512 MB instance the second
# one is what gets the process killed.
_indexing = threading.Semaphore(1)


def _check_limits(chunks: list) -> None:
    # Pages are checked by the parser, before extraction; this is the bound on
    # embedding work, which is what decides indexing time.
    if settings.max_chunks_per_doc and len(chunks) > settings.max_chunks_per_doc:
        raise DocumentTooLarge(
            f"this document is too long to index here ({len(chunks)} passages; "
            f"the limit is {settings.max_chunks_per_doc})"
        )


def index_document(workspace_id: int, doc_path: Path, paper_id: str) -> int:
    """Parse -> chunk -> embed -> publish a new index version with the document.

    Returns the number of chunks this document contributed, or 0 if it had no
    extractable text — an unOCR-able scan, or a file that is genuinely empty.
    The caller treats 0 as an upload failure. Raises DocumentTooLarge when a
    configured limit is exceeded.
    """
    with _indexing:
        try:
            # The expensive part happens with no lock held and no index
            # touched: a concurrent question, or another workspace's write, is
            # never blocked behind a long document's embedding.
            pages = extract_pages(doc_path, max_pages=settings.max_pages or None)
            chunks = chunk_pages(pages, paper_id, unit=unit_for(doc_path))
            _check_limits(chunks)
            if not chunks:
                return 0
            vectors = embed([c.embed_text for c in chunks], progress=False)

            def add(work: Path) -> None:
                # Both stores are written into the same version together, so
                # they cannot drift apart.
                append_to_store(chunks, vectors, work)
                lexical.index_chunks(chunks, work)

            _publish(workspace_id, add)
            return len(chunks)
        finally:
            _release_memory()


def _release_memory() -> None:
    """Hand freed heap back to the OS after an indexing run (glibc only).

    Parsing and embedding a document allocate and free hundreds of MB of small
    objects. glibc keeps freed memory in its arenas for reuse, so without this
    the process's footprint ratchets up to the largest document it has ever
    indexed and stays there — on a 512 MB instance, the NEXT document is then
    the one that gets the process killed. A no-op anywhere without glibc.
    """
    import gc

    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        pass


def delete_paper_data(workspace_id: int, paper_id: str) -> None:
    """Remove a paper's chunks from both indexes and delete its stored file."""
    if current_version(workspace_id) is not None:
        def drop(work: Path) -> None:
            remove_paper(work, paper_id)
            lexical.remove_paper(work, paper_id)

        _publish(workspace_id, drop)
    delete_stored_paper(workspace_id, paper_id)
    invalidate(workspace_id)


def delete_stored_paper(workspace_id: int, paper_id: str) -> None:
    """Remove a paper's stored original and any local copy of it."""
    store = blob_store()
    keys = list(store.list_keys(f"{workspace_id}/papers/{paper_id}."))
    if keys:
        store.delete(keys)
    for cached in (CACHE_ROOT / "papers" / str(workspace_id)).glob(f"{paper_id}.*"):
        cached.unlink(missing_ok=True)


def delete_workspace_data(workspace_id: int, session: Session) -> None:
    """Every stored byte of a workspace: originals, index versions, cache.

    The pointer row is deleted in the CALLER's session, which commits it along
    with the workspace's other rows. A second session here would deadlock on
    SQLite, whose single writer lock the caller's pending deletes already hold.
    """
    from backend.db_models import WorkspaceIndex

    blob_store().delete_prefix(f"{workspace_id}/")
    row = session.get(WorkspaceIndex, workspace_id)
    if row is not None:
        session.delete(row)
    shutil.rmtree(CACHE_ROOT / str(workspace_id), ignore_errors=True)
    shutil.rmtree(CACHE_ROOT / "papers" / str(workspace_id), ignore_errors=True)
    invalidate(workspace_id)


def store_paper(workspace_id: int, paper_id: str, suffix: str, source: Path) -> None:
    """Keep an uploaded original, and a local copy for the indexer to read."""
    blob_store().put(paper_key(workspace_id, paper_id, suffix), source)
    cached = _paper_cache_path(workspace_id, paper_id, suffix)
    cached.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, cached)


def _paper_cache_path(workspace_id: int, paper_id: str, suffix: str) -> Path:
    return CACHE_ROOT / "papers" / str(workspace_id) / f"{paper_id}{suffix}"


def stored_path(workspace_id: int, paper_id: str) -> Path | None:
    """A local copy of a paper's stored original, whatever its format — or
    None if it's gone. Fetched from the blob store on a cache miss.

    Found by prefix rather than assembled, because the extension varies by
    format and `paper_id` is a slug that only this module mints (see
    `slugify`), so `<paper_id>.` cannot match another paper's file.
    """
    keys = sorted(blob_store().list_keys(f"{workspace_id}/papers/{paper_id}."))
    if not keys:
        return None
    suffix = keys[0][len(f"{workspace_id}/papers/{paper_id}"):]
    cached = _paper_cache_path(workspace_id, paper_id, suffix)
    if cached.exists():
        return cached
    tmp = cached.with_name(f".{uuid.uuid4().hex}{suffix}")
    if not blob_store().get_to_file(keys[0], tmp):
        return None
    os.replace(tmp, cached)
    return cached


def discard_cached_paper(workspace_id: int, paper_id: str, suffix: str) -> None:
    _paper_cache_path(workspace_id, paper_id, suffix).unlink(missing_ok=True)
