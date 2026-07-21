"""Reranking: stage-2 of retrieval — re-score candidates with a cross-encoder.

Stage 1 (retriever.py) uses a BI-encoder: it embeds the question and each chunk
INDEPENDENTLY into vectors, then compares them. Fast, but it never lets the
question and the chunk "look at" each other, so its ranking is approximate.

A CROSS-encoder instead feeds "question [SEP] chunk" through one Transformer, so
every question token can attend to every chunk token. That joint view judges true
relevance far better — but it costs one full model pass PER candidate, so we only
ever run it on the handful the bi-encoder already shortlisted, not the whole index.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 — small (~80MB), CPU-fast, trained on
MS MARCO query/passage relevance. Ships inside sentence-transformers.
"""
from sentence_transformers import CrossEncoder

from backend.models import Citation

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANKER_MODEL)
    return _model


class Reranker:
    """Re-scores retrieved candidates by true (question, passage) relevance."""

    def rerank(
        self, question: str, candidates: list[Citation], top_k: int = 5
    ) -> list[Citation]:
        """Return the top_k candidates re-ordered by cross-encoder relevance."""
        if not candidates:
            return []

        # Score every (question, passage) pair jointly. Higher = more relevant.
        pairs = [(question, c.text) for c in candidates]
        scores = _get_model().predict(pairs)

        # Attach the stage-2 score to each candidate (keeps the stage-1 cosine too).
        for citation, score in zip(candidates, scores):
            citation.rerank_score = float(score)

        # Best first, then trim to the k we actually feed the LLM.
        ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)
        return ranked[:top_k]
