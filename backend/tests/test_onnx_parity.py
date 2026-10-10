"""The ONNX port is held to the torch pipeline it replaced.

The expected values were captured ONCE from sentence-transformers + torch by
scripts/make_golden_fixtures.py and committed. That matters: a parity test whose
expected values come from the new code would prove only that the new code
agrees with itself.

Three tiers, cheapest first:

  chunker   tokenizer only, every commit. Chunk text, offsets and count must be
            IDENTICAL — a changed chunk boundary changes what can be retrieved.
  int8      the shipped models (`slow`). Quantisation moves the numbers a LOT
            more than one might guess — measured against torch on real corpus
            text: embedding cosine 0.84 worst / 0.91 median, cross-encoder rank
            correlation 0.86 worst / 0.94 mean. Yet the full pipeline lost
            nothing on the eval (hit@5 94.1% -> 94.1%, MRR 0.822 -> 0.835, no
            question flipped; docs/deploy-measurements/eval-int8-batch1.*). So these bars only catch a BROKEN port
            (wrong pooling or padding lands near 0.3); whether drift costs
            retrieval quality is the eval's question, and it answered it.
  fp32      the unquantised export (`slow`, skipped unless fetched with
            --precision fp32). Must match torch to float tolerance — this is
            what proves the pre/post-processing is right, independent of
            quantisation.
"""
import json
from pathlib import Path

import numpy as np
import pytest

from backend import onnx_runtime
from backend.chunker import chunk_pages

FIXTURES = Path(__file__).parent / "fixtures"


def _golden() -> dict:
    return json.loads((FIXTURES / "chunker_golden.json").read_text(encoding="utf-8"))


def _reference() -> tuple[dict, np.lib.npyio.NpzFile]:
    meta = json.loads((FIXTURES / "torch_reference.json").read_text(encoding="utf-8"))
    return meta, np.load(FIXTURES / "torch_reference.npz")


def _ranks(x: np.ndarray) -> np.ndarray:
    return np.argsort(np.argsort(x)).astype(np.float64)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    ra, rb = _ranks(a), _ranks(b)
    ra -= ra.mean()
    rb -= rb.mean()
    return float((ra @ rb) / np.sqrt((ra @ ra) * (rb @ rb)))


# ——— chunker: identical, not close ———


@pytest.mark.parametrize("doc", sorted(_golden()))
def test_chunks_are_identical_to_the_torch_era_chunker(doc):
    entry = _golden()[doc]
    got = [
        {"page": c.page, "chunk_index": c.chunk_index, "text": c.text,
         "char_start": c.char_start, "char_end": c.char_end}
        for c in chunk_pages(entry["pages"], doc)
    ]
    assert len(got) == len(entry["chunks"]), f"{doc}: chunk count changed"
    for mine, theirs in zip(got, entry["chunks"]):
        assert mine == theirs, f"{doc}: chunk {theirs['chunk_index']} differs"


def test_the_shipped_tokenizer_json_really_does_truncate_at_128():
    """Documents WHY onnx_runtime.tokenizer() disables truncation: the file the
    model authors ship caps encodings at 128 tokens. If this ever starts
    failing, the upstream file changed — re-check the assumption, don't delete
    the guard."""
    from tokenizers import Tokenizer

    path = onnx_runtime.models_dir() / onnx_runtime.EMBEDDER_DIR / "tokenizer.json"
    raw = Tokenizer.from_file(str(path))
    assert raw.truncation is not None and raw.truncation["max_length"] == 128
    long_text = "word " * 1000
    assert len(raw.encode(long_text).ids) == 128
    assert len(onnx_runtime.tokenizer(onnx_runtime.EMBEDDER_DIR).encode(long_text).ids) > 1000


# ——— models ———


def _set_precision(monkeypatch, value: str) -> None:
    from backend.config import settings

    monkeypatch.setattr(settings, "model_precision", value)


def _require_files(value: str) -> None:
    for model in (onnx_runtime.EMBEDDER_DIR, onnx_runtime.RERANKER_DIR):
        path = onnx_runtime.models_dir() / model / onnx_runtime._MODEL_FILES[value]
        if not path.exists():
            pytest.skip(f"{value} model not fetched ({path}); run scripts/fetch_models.py --precision {value}")


def _check(monkeypatch, value: str, min_cosine: float, logit_tol: float | None, min_spearman: float,
           min_synthetic_median: float | None = None):
    """Embeddings of REAL corpus text must each clear `min_cosine`.

    The `_synthetic_long` pages (one sentence repeated for 20k characters) are
    held only to a median: they exist to exercise multi-window chunking, and
    dynamic int8 quantisation is genuinely poor on such near-constant input —
    measured worst 0.44 at batch size 1 — which no real document resembles.
    Excluding them from the per-text floor keeps that floor meaningful for the
    text users actually upload, instead of loosening it for everyone.
    """
    _require_files(value)
    _set_precision(monkeypatch, value)
    from backend.embedder import embed
    from backend.reranker import score_pairs

    meta, ref = _reference()

    vectors = embed(meta["texts"], progress=False)
    assert vectors.shape == ref["vectors"].shape
    np.testing.assert_allclose(np.linalg.norm(vectors, axis=1), 1.0, atol=1e-5)
    cosines = (vectors * ref["vectors"]).sum(axis=1)

    synthetic_texts = {c["text"] for c in _golden()["_synthetic_long"]["chunks"]}
    synthetic = np.array([t in synthetic_texts for t in meta["texts"]])
    real = np.where(~synthetic)[0]
    worst = int(real[np.argmin(cosines[real])])
    assert cosines[worst] >= min_cosine, (
        f"{value}: worst embedding cosine to torch {cosines[worst]:.6f} "
        f"(text #{worst}: {meta['texts'][worst][:60]!r})"
    )
    floor = min_cosine if min_synthetic_median is None else min_synthetic_median
    assert np.median(cosines[synthetic]) >= floor, (
        f"{value}: synthetic pages' median cosine {np.median(cosines[synthetic]):.4f}"
    )

    logits = score_pairs(list(zip(meta["pair_questions"], meta["pair_passages"])))
    expected = ref["logits"]
    if logit_tol is not None:
        np.testing.assert_allclose(logits, expected, atol=logit_tol)
    # The reranker's job is an ORDER within one question's candidates — 8
    # pairs per question in the fixture — so that is what is compared.
    for start in range(0, len(expected), 8):
        rho = _spearman(logits[start:start + 8], expected[start:start + 8])
        assert rho >= min_spearman, f"{value}: question #{start // 8} reranked differently (rho={rho:.3f})"


@pytest.mark.slow
def test_fp32_matches_torch_to_float_tolerance(monkeypatch):
    _check(monkeypatch, "fp32", min_cosine=0.9999, logit_tol=1e-3, min_spearman=1.0)


@pytest.mark.slow
def test_int8_stays_close_to_torch(monkeypatch):
    # Floors sit just under the measured values at the SHIPPED batch size of 1
    # (real text: worst 0.842, median 0.906; synthetic median 0.886; reranker
    # rank correlation worst 0.857): loose enough for quantisation, far above
    # what a broken port produces (wrong pooling or padding lands near 0.3).
    _check(monkeypatch, "int8", min_cosine=0.80, logit_tol=None, min_spearman=0.80,
           min_synthetic_median=0.80)
