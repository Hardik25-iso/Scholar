"""Embedding with all-mpnet-base-v2, on ONNX Runtime.

Embeddings are L2-normalised so that inner product == cosine similarity,
which is what IndexFlatIP in FAISS computes.

This reproduces sentence-transformers' pipeline for this model step by step —
strip, tokenize (truncate at 384, pad to the batch's longest), transformer,
mean-pool over real tokens, L2-normalise — so the vectors match the torch
implementation it replaced. That match is a test, not a hope: see
backend/tests/test_onnx_parity.py, which holds this module to embeddings
captured from the torch pipeline.
"""
from functools import lru_cache

import numpy as np
from tokenizers import Tokenizer

from backend.onnx_runtime import EMBEDDER_DIR, session, tokenizer

EMBEDDING_DIM = 768

# The model's fine-tuned max_seq_length (512 is the architectural ceiling; 384
# is what sentence-transformers uses for this model). Anything longer is cut,
# which is why the chunker sizes chunks to fit — see chunker.py.
MAX_SEQ_LENGTH = 384



@lru_cache(maxsize=1)
def _encoder() -> Tokenizer:
    tok = tokenizer(EMBEDDER_DIR)
    tok.enable_truncation(MAX_SEQ_LENGTH, strategy="longest_first")
    # Pad to the longest in each batch (no fixed length), with the model's own
    # pad token. A fixed 384 would multiply the work for a short question.
    tok.enable_padding(pad_id=tok.token_to_id("<pad>"), pad_token="<pad>")
    return tok


def embed(texts: list[str]) -> np.ndarray:
    """Return a float32 array of shape (len(texts), 768), L2-normalised."""
    from backend.config import settings

    out = np.empty((len(texts), EMBEDDING_DIM), dtype=np.float32)
    if not texts:
        return out

    sess = session(EMBEDDER_DIR)
    tok = _encoder()
    # Texts per model call (EMBED_BATCH_SIZE, default 1). Measured in the image
    # on one thread, full-length chunks: batch 1 = 793 ms/chunk at +21 MB peak;
    # batch 8 = 1253 ms at +360 MB. Batching pays off on many cores; on one it
    # only adds padding and cache pressure — slower AND hundreds of MB heavier.
    size = settings.embed_batch_size

    # When batching IS configured, group texts of similar length: padding to the
    # longest member is wasted compute. Results are written back by index, so
    # the caller's order is unaffected.
    order = sorted(range(len(texts)), key=lambda i: len(texts[i]))
    for start in range(0, len(order), size):
        idx = order[start:start + size]
        # .strip(): sentence-transformers strips its inputs, so matching its
        # vectors means doing the same.
        encodings = tok.encode_batch([texts[i].strip() for i in idx])
        ids = np.array([e.ids for e in encodings], dtype=np.int64)
        mask = np.array([e.attention_mask for e in encodings], dtype=np.int64)
        hidden = sess.run(None, {"input_ids": ids, "attention_mask": mask})[0]

        # Mean over REAL tokens only — padding must not dilute the average.
        weights = mask[..., None].astype(np.float32)
        pooled = (hidden * weights).sum(axis=1) / np.clip(weights.sum(axis=1), 1e-9, None)
        norms = np.linalg.norm(pooled, axis=1, keepdims=True)
        out[idx] = pooled / np.clip(norms, 1e-12, None)
    return out
