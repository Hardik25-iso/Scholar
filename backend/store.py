"""FAISS index + JSON metadata store.

Layout on disk:
  <store_dir>/index.faiss   — FAISS IndexFlatIP binary
  <store_dir>/chunks.json   — list of IndexedChunk dicts, keyed by faiss_id

IndexFlatIP with L2-normalised vectors gives true cosine similarity.
Exact search (no approximation) is fine up to ~100 k chunks.
"""
import json
import logging
from pathlib import Path

import faiss
import numpy as np

from backend.models import Chunk, IndexedChunk

EMBEDDING_DIM = 768  # all-mpnet-base-v2


def build(chunks: list[Chunk], vectors: np.ndarray, store_dir: str | Path) -> None:
    """Embed chunks, build FAISS index, persist both artefacts."""
    store_dir = Path(store_dir)
    store_dir.mkdir(parents=True, exist_ok=True)

    assert len(chunks) == vectors.shape[0], "chunk / vector count mismatch"
    assert vectors.shape[1] == EMBEDDING_DIM, f"expected {EMBEDDING_DIM}-dim vectors"

    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(vectors)

    faiss.write_index(index, str(store_dir / "index.faiss"))

    indexed: list[dict] = [
        IndexedChunk(**chunk.model_dump(), faiss_id=i).model_dump()
        for i, chunk in enumerate(chunks)
    ]
    (store_dir / "chunks.json").write_text(
        json.dumps(indexed, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logging.getLogger(__name__).info("stored %d chunks -> %s", len(indexed), store_dir)


def load(store_dir: str | Path) -> tuple[faiss.IndexFlatIP, list[IndexedChunk]]:
    """Load index and chunk metadata from disk."""
    store_dir = Path(store_dir)
    index = faiss.read_index(str(store_dir / "index.faiss"))
    raw = json.loads((store_dir / "chunks.json").read_text(encoding="utf-8"))
    chunks = [IndexedChunk(**item) for item in raw]
    return index, chunks


def _write(index: faiss.IndexFlatIP, chunks: list[IndexedChunk], store_dir: Path) -> None:
    """Persist an index + its chunk metadata to store_dir."""
    store_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(store_dir / "index.faiss"))
    (store_dir / "chunks.json").write_text(
        json.dumps([c.model_dump() for c in chunks], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def append_to_store(chunks: list[Chunk], vectors: np.ndarray, store_dir: str | Path) -> int:
    """Append new chunks+vectors to an existing store (create it if absent).

    Used when a user uploads another paper into their per-user index. Each new
    chunk's faiss_id is the row it lands on in the FAISS index, so ids stay a
    contiguous 0..N-1 mapping into chunks.json. Returns the new total count.
    """
    store_dir = Path(store_dir)
    assert len(chunks) == vectors.shape[0], "chunk / vector count mismatch"
    assert vectors.shape[1] == EMBEDDING_DIM, f"expected {EMBEDDING_DIM}-dim vectors"

    if (store_dir / "index.faiss").exists():
        index, existing = load(store_dir)
    else:
        index, existing = faiss.IndexFlatIP(EMBEDDING_DIM), []

    base = index.ntotal  # next free row id
    index.add(vectors)
    new = [
        IndexedChunk(**c.model_dump(), faiss_id=base + i)
        for i, c in enumerate(chunks)
    ]
    _write(index, existing + new, store_dir)
    return index.ntotal


def remove_paper(store_dir: str | Path, paper_id: str) -> int:
    """Rebuild the store without a given paper's chunks. Returns chunks remaining.

    IndexFlatIP has no native delete, so we reconstruct the vectors we keep
    (they're stored in the flat index — no re-embedding needed), build a fresh
    index, and renumber faiss_ids to stay contiguous. If nothing remains, the
    store files are removed entirely.
    """
    store_dir = Path(store_dir)
    index, chunks = load(store_dir)

    keep = [(i, c) for i, c in enumerate(chunks) if c.paper_id != paper_id]
    if not keep:
        (store_dir / "index.faiss").unlink(missing_ok=True)
        (store_dir / "chunks.json").unlink(missing_ok=True)
        return 0

    vectors = np.vstack([index.reconstruct(i) for i, _ in keep]).astype(np.float32)
    new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    new_index.add(vectors)
    renumbered = [
        IndexedChunk(**{**c.model_dump(exclude={"faiss_id"}), "faiss_id": new_id})
        for new_id, (_, c) in enumerate(keep)
    ]
    _write(new_index, renumbered, store_dir)
    return new_index.ntotal
