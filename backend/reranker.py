"""Reranking: stage-2 of retrieval — re-score candidates with a cross-encoder.

Stage 1 (retriever.py) uses a BI-encoder: it embeds the question and each chunk
INDEPENDENTLY into vectors, then compares them. Fast, but it never lets the
question and the chunk "look at" each other, so its ranking is approximate.

A CROSS-encoder instead feeds "question [SEP] chunk" through one Transformer, so
every question token can attend to every chunk token. That joint view judges true
relevance far better — but it costs one full model pass PER candidate, so we only
ever run it on the handful the bi-encoder already shortlisted, not the whole index.

Model: cross-encoder/ms-marco-MiniLM-L-6-v2 — small, CPU-fast, trained on
MS MARCO query/passage relevance. Run on ONNX Runtime from the model authors'
own export; the score is the raw logit, exactly as sentence-transformers
returned it (the model's config sets its activation to Identity, not sigmoid).
"""
from functools import lru_cache

import numpy as np
from tokenizers import Tokenizer

from backend import mmr
from backend.models import Citation
from backend.onnx_runtime import RERANKER_DIR, precision, session, tokenizer

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

# The model's position-embedding limit. A (question, passage) pair longer than
# this has its LONGER side trimmed first ("longest_first"), as before.
MAX_LENGTH = 512

# Pairs scored per model call. Measured in the image on one thread: batch 1 =
# 140 ms/pair at +25 MB peak; batch 8 = 209 ms/pair at +236 MB. One at a time
# is both faster and lighter on a single core — see embedder.DEFAULT_BATCH_SIZE.
BATCH_SIZE = 1


@lru_cache(maxsize=2)
def _encoder(max_length: int = MAX_LENGTH) -> Tokenizer:
    tok = tokenizer(RERANKER_DIR)
    tok.enable_truncation(min(max_length, MAX_LENGTH), strategy="longest_first")
    tok.enable_padding(pad_id=tok.token_to_id("[PAD]"), pad_token="[PAD]")
    return tok


def score_pairs(pairs: list[tuple[str, str]], max_length: int = MAX_LENGTH) -> np.ndarray:
    """Raw relevance logits for (question, passage) pairs. Higher = more relevant.

    `max_length` trims each pair (longer side first) — the main cost lever on a
    slow CPU, since a pass costs roughly in proportion to its length."""
    if not pairs:
        return np.empty(0, dtype=np.float32)
    sess = session(RERANKER_DIR, precision())
    wanted = {i.name for i in sess.get_inputs()}
    tok = _encoder(max_length)

    scores: list[np.ndarray] = []
    for start in range(0, len(pairs), BATCH_SIZE):
        batch = [(str(q).strip(), str(p).strip()) for q, p in pairs[start:start + BATCH_SIZE]]
        encodings = tok.encode_batch(batch)
        feed = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        logits = sess.run(None, {k: v for k, v in feed.items() if k in wanted})[0]
        scores.append(logits[:, 0].astype(np.float32))
    return np.concatenate(scores)


class Reranker:
    """Re-scores retrieved candidates by true (question, passage) relevance."""

    def rerank(
        self,
        question: str,
        candidates: list[Citation],
        top_k: int = 5,
        use_mmr: bool | None = None,
    ) -> list[Citation]:
        """Return the top_k candidates re-ordered by cross-encoder relevance.

        `use_mmr` overrides the configured setting — the eval passes it
        explicitly so both arms of an A/B run against one build.
        """
        if not candidates:
            return []
        # Settings are read HERE, not at import time — see the note below.
        from backend.config import settings

        if not settings.rerank_enabled:
            # Stage 1's fused order IS the ranking. A deliberate, measured
            # trade for a CPU too small to run the cross-encoder in time.
            return candidates[:top_k]

        # Score every (question, passage) pair jointly. Higher = more relevant.
        scores = score_pairs([(question, c.text) for c in candidates],
                             max_length=settings.rerank_max_tokens)

        # Attach the stage-2 score to each candidate (keeps the stage-1 cosine too).
        for citation, score in zip(candidates, scores):
            citation.rerank_score = float(score)

        # Best first, then trim to the k we actually feed the LLM.
        ranked = sorted(candidates, key=lambda c: c.rerank_score, reverse=True)

        # MMR picks from the SCORED list rather than replacing the scoring: it
        # re-orders the cross-encoder's judgement to avoid spending slots on
        # near-duplicate passages. Selecting from `ranked` (not `candidates`)
        # keeps the two paths identical when MMR is off.
        # Settings are read HERE, not at import time: a module that names a
        # model should be importable without a configured deployment — a
        # module-level `settings` would construct Settings() and fail on a
        # missing SECRET_KEY wherever there is no environment (an image build,
        # a one-off script).
        if settings.mmr_enabled if use_mmr is None else use_mmr:
            return mmr.select(ranked, k=top_k, lam=settings.mmr_lambda)
        return ranked[:top_k]
