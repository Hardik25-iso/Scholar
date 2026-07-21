"""Embedding with sentence-transformers all-mpnet-base-v2.

Embeddings are L2-normalised so that inner product == cosine similarity,
which is what IndexFlatIP in FAISS computes.
"""
import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"

_model: SentenceTransformer | None = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(
    texts: list[str], batch_size: int = 32, progress: bool = True
) -> np.ndarray:
    """Return a float32 array of shape (len(texts), 768), L2-normalised.

    `progress` shows a progress bar — useful when embedding a whole paper at
    ingest time, but noisy for a single query, so the retriever turns it off.
    """
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=progress,
        convert_to_numpy=True,
        normalize_embeddings=True,   # cosine via inner product
    )
    return vectors.astype(np.float32)
