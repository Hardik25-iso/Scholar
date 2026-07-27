"""Per-user paper library: storage layout, indexing, retrieval scoping.

Each user gets an isolated directory tree under data/users/<id>/:
    papers/<paper_id>.pdf   — the original upload (kept for future in-app viewing)
    index/index.faiss       — that user's FAISS index (ALL their papers)
    index/chunks.json       — chunk metadata, each chunk tagged by paper_id

One index PER USER (not one shared index filtered by user) means a user's data
never mixes with anyone else's, and "delete my paper" is a local rebuild. This
reuses the same ingest pipeline (parser -> chunker -> embedder -> store) that the
CLI uses, so indexing behaves identically here.
"""
import re
from pathlib import Path

from backend.chunker import chunk_pages
from backend.embedder import embed
from backend.parser import extract_pages
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


def index_pdf(user_id: int, pdf_path: Path, paper_id: str) -> int:
    """Parse -> chunk -> embed -> append a PDF into the user's index.

    Returns the number of chunks this paper contributed (0 if the PDF had no
    extractable text — e.g. a pure scan, which we treat as an upload failure).
    """
    pages = extract_pages(pdf_path)
    chunks = chunk_pages(pages, paper_id)
    if not chunks:
        return 0
    vectors = embed([c.embed_text for c in chunks], progress=False)
    append_to_store(chunks, vectors, user_index_dir(user_id))
    invalidate(user_id)
    return len(chunks)


def delete_paper_data(user_id: int, paper_id: str) -> None:
    """Remove a paper's chunks from the index and delete its stored PDF."""
    remove_paper(user_index_dir(user_id), paper_id)
    (user_papers_dir(user_id) / f"{paper_id}.pdf").unlink(missing_ok=True)
    invalidate(user_id)
