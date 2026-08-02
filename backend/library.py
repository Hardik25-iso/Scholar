"""Per-user document library: storage layout, indexing, retrieval scoping.

Each user gets an isolated directory tree under data/users/<id>/:
    papers/<paper_id>.<ext> — the original upload, in its own format
    index/index.faiss       — that user's FAISS index (ALL their papers)
    index/chunks.json       — chunk metadata, each chunk tagged by paper_id

One index PER USER (not one shared index filtered by user) means a user's data
never mixes with anyone else's, and "delete my paper" is a local rebuild. This
reuses the same ingest pipeline (parser -> chunker -> embedder -> store) that the
CLI uses, so indexing behaves identically here.
"""
import re
from pathlib import Path

from backend import lexical
from backend.chunker import chunk_pages
from backend.embedder import embed
from backend.parser import extract_pages, unit_for
from backend.retriever import Retriever
from backend.store import append_to_store, remove_paper

DATA_ROOT = Path(__file__).parent.parent / "data" / "users"

# One loaded Retriever per user, reused across their /ask calls (loading the
# FAISS index + chunks.json every request would be wasteful). Invalidated
# whenever the user's index changes, so it never serves a stale library.
_retrievers: dict[int, Retriever] = {}


def user_index_dir(user_id: int) -> Path:
    return DATA_ROOT / str(user_id) / "index"


def user_papers_dir(user_id: int) -> Path:
    return DATA_ROOT / str(user_id) / "papers"


def slugify(name: str) -> str:
    """Filesystem/id-safe slug from a filename stem."""
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_") or "paper"


def invalidate(user_id: int) -> None:
    _retrievers.pop(user_id, None)


def get_retriever(user_id: int) -> Retriever | None:
    """The user's retriever, or None if they have not indexed any papers yet."""
    if not (user_index_dir(user_id) / "index.faiss").exists():
        return None
    if user_id not in _retrievers:
        _retrievers[user_id] = Retriever(user_index_dir(user_id))
    return _retrievers[user_id]


def stored_path(user_id: int, paper_id: str) -> Path | None:
    """The stored file for a paper, whatever its format — or None if it's gone.

    Globbed rather than assembled, because the extension varies by format and
    `paper_id` is a slug that only this module mints (see `slugify`), so it can
    never contain a glob metacharacter.
    """
    return next(iter(sorted(user_papers_dir(user_id).glob(f"{paper_id}.*"))), None)


def index_document(user_id: int, doc_path: Path, paper_id: str) -> int:
    """Parse -> chunk -> embed -> append a document into the user's index.

    Returns the number of chunks this document contributed, or 0 if it had no
    extractable text — an unOCR-able scan, or a file that is genuinely empty.
    The caller treats 0 as an upload failure.
    """
    pages = extract_pages(doc_path)
    chunks = chunk_pages(pages, paper_id, unit=unit_for(doc_path))
    if not chunks:
        return 0
    vectors = embed([c.embed_text for c in chunks], progress=False)
    append_to_store(chunks, vectors, user_index_dir(user_id))
    # Mirror the same chunks into the user's lexical index. Both stores live in
    # the same directory and are written together, so they cannot drift apart.
    lexical.index_chunks(chunks, user_index_dir(user_id))
    invalidate(user_id)
    return len(chunks)


def delete_paper_data(user_id: int, paper_id: str) -> None:
    """Remove a paper's chunks from both indexes and delete its stored file."""
    remove_paper(user_index_dir(user_id), paper_id)
    lexical.remove_paper(user_index_dir(user_id), paper_id)
    path = stored_path(user_id, paper_id)
    if path is not None:
        path.unlink(missing_ok=True)
    invalidate(user_id)
