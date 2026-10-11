"""Freeze the torch pipeline's outputs so the ONNX port can be held to them.

    pip install -r requirements-eval.txt        # torch + sentence-transformers
    python -m backend.eval --build-corpus       # the generated PDFs it reads
    python scripts/make_golden_fixtures.py

The embeddings and logits always come from sentence-transformers + torch, so
re-running this re-captures the true reference. The CHUNKER golden is
different: it records whatever backend.chunker produces at the time, so it was
captured once at the torch-era commit (71dbe23) and re-running it later only
re-baselines it. A parity test whose expected values come from the new code
proves nothing.

What it writes to backend/tests/fixtures/:

  chunker_golden.json   page texts IN, chunks OUT (text, offsets, count). The
                        inputs travel with the outputs, so the test needs no
                        PDFs and no network — CI has neither the fetched paper
                        nor a reason to build one.
  torch_reference.json  the fixed texts and (question, passage) pairs
  torch_reference.npz   mpnet embeddings for those texts and cross-encoder
                        logits for those pairs — numbers only, so loading it
                        never needs allow_pickle.

Only the GENERATED corpus documents and the dataset's own questions are used.
rag_paper.pdf is a third-party paper the repository does not redistribute, and
committing its extracted text would be redistributing it.
"""
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend.eval.__main__ import CORPUS, DATASET  # noqa: E402

OUT = ROOT / "backend" / "tests" / "fixtures"

# Generated from committed .txt sources, so their text is ours to commit.
GENERATED = [p.stem for p in sorted(CORPUS.glob("*.txt"))]


def _long_pages(base: str) -> list[str]:
    """Pages long enough to need several windows, plus awkward characters.

    A single 128-token window would silently pass a test built only from short
    pages — exactly the failure the tokenizer.json default would cause — so the
    golden set must contain pages well past 384 tokens.
    """
    words = base.split()
    long_page = " ".join(words * (1 + 2000 // max(1, len(words))))[:20000]
    awkward = (
        "Section 7.2 — “Force Majeure”. Fees: €1,250.00 / month (excl. VAT). "
        "Contact: avaswani@google.com; ID INV-2024/0042. Ünïcödé façade naïve. "
    ) * 40
    return [long_page, awkward]


def main() -> int:
    # The REFERENCE side must come from torch itself. backend.embedder is the
    # ONNX port now; using it here would make the parity test compare the port
    # with itself and pass by construction.
    from sentence_transformers import CrossEncoder, SentenceTransformer

    from backend.chunker import chunk_pages
    from backend.parser import extract_pages
    from backend.reranker import RERANKER_MODEL

    OUT.mkdir(parents=True, exist_ok=True)

    docs: dict[str, dict] = {}
    for stem in GENERATED:
        pdf = CORPUS / f"{stem}.pdf"
        if not pdf.exists():
            print(f"missing {pdf} — run: python -m backend.eval --build-corpus")
            return 1
        docs[stem] = {"pages": extract_pages(pdf)}
    docs["_synthetic_long"] = {"pages": _long_pages(docs[GENERATED[0]]["pages"][0])}

    all_chunks = []
    for stem, entry in docs.items():
        chunks = chunk_pages(entry["pages"], stem)
        entry["chunks"] = [
            {"page": c.page, "chunk_index": c.chunk_index, "text": c.text,
             "char_start": c.char_start, "char_end": c.char_end}
            for c in chunks
        ]
        all_chunks.extend(chunks)
        print(f"{stem}: {len(entry['pages'])} page(s) -> {len(chunks)} chunk(s)")

    (OUT / "chunker_golden.json").write_text(
        json.dumps(docs, ensure_ascii=False, indent=1), encoding="utf-8"
    )

    questions = [q["question"] for q in json.loads(DATASET.read_text(encoding="utf-8"))["questions"]]
    passages = [c.text for c in all_chunks]
    # Embeddings: every generated passage (they are short documents) plus the
    # questions, so both index-time and query-time inputs are covered.
    texts = passages + questions
    vectors = SentenceTransformer("sentence-transformers/all-mpnet-base-v2").encode(
        texts, batch_size=1, convert_to_numpy=True, normalize_embeddings=True
    )

    # Cross-encoder: each of the first 12 questions against 8 passages, the
    # unit the reranker actually sees — a ranking within one question.
    pair_q, pair_p = [], []
    for i, q in enumerate(questions[:12]):
        for j in range(8):
            pair_q.append(q)
            pair_p.append(passages[(i * 5 + j * 3) % len(passages)])
    logits = CrossEncoder(RERANKER_MODEL).predict(list(zip(pair_q, pair_p)))

    (OUT / "torch_reference.json").write_text(json.dumps(
        {"texts": texts, "pair_questions": pair_q, "pair_passages": pair_p},
        ensure_ascii=False, indent=1,
    ), encoding="utf-8")
    np.savez_compressed(
        OUT / "torch_reference.npz",
        vectors=vectors.astype(np.float32),
        logits=np.asarray(logits, dtype=np.float32),
    )
    print(f"wrote {len(texts)} embeddings and {len(pair_q)} cross-encoder scores to {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
