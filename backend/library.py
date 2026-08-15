"""Per-workspace document library: storage layout, indexing, retrieval scoping.

Each workspace gets an isolated directory tree under data/workspaces/<id>/:
    papers/<paper_id>.<ext> — the original upload, in its own format
    index/index.faiss       — that workspace's FAISS index (ALL its documents)
    index/chunks.json       — chunk metadata, each chunk tagged by paper_id
    index/lexical.db        — the FTS5 mirror of the same chunks

One index PER WORKSPACE, not one shared index filtered by a query parameter.
That makes isolation a property of the storage layout rather than of remembering
a WHERE clause: retrieval physically cannot return another workspace's passages,
because it never opens their index. A filter can be forgotten in one code path;
a directory cannot.

Personal libraries are ordinary workspaces flagged `is_personal`, so there is
exactly one storage and retrieval path rather than two that drift apart.
"""
import re
import threading
from collections import OrderedDict
from pathlib import Path

from backend import lexical
from backend.chunker import chunk_pages
from backend.config import settings
from backend.embedder import embed
from backend.parser import extract_pages, unit_for
from backend.retriever import Retriever
from backend.store import append_to_store, remove_paper

# Configurable, because the fallback is INSIDE the source tree: a deployment that
# replaces the source on redeploy would take every user's library with it, with
# no error to notice. Set DATA_ROOT to a mounted volume in any real deployment.
DEFAULT_DATA_ROOT = Path(__file__).parent.parent / "data" / "workspaces"
DATA_ROOT = Path(settings.data_root) if settings.data_root else DEFAULT_DATA_ROOT

# Where libraries lived before workspaces existed. Only the migration reads it.
LEGACY_DATA_ROOT = Path(settings.data_root).parent / "users" if settings.data_root \
    else Path(__file__).parent.parent / "data" / "users"

# One loaded Retriever per workspace, reused across /ask calls (loading the
# FAISS index + chunks.json every request would be wasteful). Invalidated
# whenever the workspace's index changes, so it never serves a stale library.
#
# BOUNDED, because each entry holds a whole FAISS index plus its chunk text in
# memory. Unbounded, this grows with the number of workspaces ever queried and
# never gives anything back — fine for one user, a slow leak for many. An
# OrderedDict used as an LRU: least-recently-used is evicted past the cap, and
# an evicted workspace simply reloads from disk on its next question.
_RETRIEVER_CACHE_SIZE = 8
_retrievers: OrderedDict[int, Retriever] = OrderedDict()
_retrievers_lock = threading.Lock()


def workspace_index_dir(workspace_id: int) -> Path:
    return DATA_ROOT / str(workspace_id) / "index"


def workspace_papers_dir(workspace_id: int) -> Path:
    return DATA_ROOT / str(workspace_id) / "papers"


def slugify(name: str) -> str:
    """Filesystem/id-safe slug from a filename stem."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "paper"


def invalidate(workspace_id: int) -> None:
    with _retrievers_lock:
        _retrievers.pop(workspace_id, None)


def get_retriever(workspace_id: int) -> Retriever | None:
    """The workspace's retriever, or None if nothing has been indexed yet."""
    if not (workspace_index_dir(workspace_id) / "index.faiss").exists():
        return None

    # Locked: FastAPI serves sync endpoints from a threadpool, so several
    # requests can reach this concurrently and would otherwise race on eviction.
    with _retrievers_lock:
        cached = _retrievers.get(workspace_id)
        if cached is not None:
            _retrievers.move_to_end(workspace_id)  # mark recently used
            return cached

    # Load OUTSIDE the lock — reading a large index off disk would otherwise
    # block every other workspace's lookup for the duration.
    retriever = Retriever(workspace_index_dir(workspace_id))

    with _retrievers_lock:
        # Another thread may have loaded it meanwhile; prefer the existing one
        # so callers never hold two different objects for the same workspace.
        existing = _retrievers.get(workspace_id)
        if existing is not None:
            _retrievers.move_to_end(workspace_id)
            return existing
        _retrievers[workspace_id] = retriever
        while len(_retrievers) > _RETRIEVER_CACHE_SIZE:
            _retrievers.popitem(last=False)  # evict least-recently-used
        return retriever


def stored_path(workspace_id: int, paper_id: str) -> Path | None:
    """The stored file for a paper, whatever its format — or None if it's gone.

    Globbed rather than assembled, because the extension varies by format and
    `paper_id` is a slug that only this module mints (see `slugify`), so it can
    never contain a glob metacharacter.
    """
    return next(iter(sorted(workspace_papers_dir(workspace_id).glob(f"{paper_id}.*"))), None)


def index_document(workspace_id: int, doc_path: Path, paper_id: str) -> int:
    """Parse -> chunk -> embed -> append a document into the workspace's index.

    Returns the number of chunks this document contributed, or 0 if it had no
    extractable text — an unOCR-able scan, or a file that is genuinely empty.
    The caller treats 0 as an upload failure.
    """
    pages = extract_pages(doc_path)
    chunks = chunk_pages(pages, paper_id, unit=unit_for(doc_path))
    if not chunks:
        return 0
    vectors = embed([c.embed_text for c in chunks], progress=False)
    append_to_store(chunks, vectors, workspace_index_dir(workspace_id))
    # Mirror the same chunks into the workspace's lexical index. Both stores live in
    # the same directory and are written together, so they cannot drift apart.
    lexical.index_chunks(chunks, workspace_index_dir(workspace_id))
    invalidate(workspace_id)
    return len(chunks)


def delete_paper_data(workspace_id: int, paper_id: str) -> None:
    """Remove a paper's chunks from both indexes and delete its stored file."""
    remove_paper(workspace_index_dir(workspace_id), paper_id)
    lexical.remove_paper(workspace_index_dir(workspace_id), paper_id)
    path = stored_path(workspace_id, paper_id)
    if path is not None:
        path.unlink(missing_ok=True)
    invalidate(workspace_id)
