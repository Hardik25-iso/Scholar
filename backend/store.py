"""FAISS index + JSON metadata store.

Layout on disk:
  <store_dir>/index.faiss   — FAISS IndexFlatIP binary
  <store_dir>/chunks.json   — list of IndexedChunk dicts, keyed by faiss_id

IndexFlatIP with L2-normalised vectors gives true cosine similarity.
Exact search (no approximation) is fine up to ~100 k chunks.
"""
import json
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
    print(f"Stored {len(indexed)} chunks -> {store_dir}")


def load(store_dir: str | Path) -> tuple[faiss.IndexFlatIP, list[IndexedChunk]]:
    """Load index and chunk metadata from disk."""
    store_dir = Path(store_dir)
    index = faiss.read_index(str(store_dir / "index.faiss"))
    raw = json.loads((store_dir / "chunks.json").read_text(encoding="utf-8"))
    chunks = [IndexedChunk(**item) for item in raw]
    return index, chunks
