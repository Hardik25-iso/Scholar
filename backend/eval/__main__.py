"""Retrieval eval: does the pipeline actually put the right passage in front of
the model?

    python -m backend.eval --build-corpus   # generate sample_agreement.pdf
    python -m backend.eval --check          # validate the labels, no models
    python -m backend.eval                  # score retrieval, print a baseline

Why this exists: every phase after this one changes parsing, chunking, retrieval
or the model, and each of those is a plausible-sounding change that can quietly
make answers worse. Without a number, "it feels better" is the only evidence
available, and it is not evidence. No retrieval change should ship without a
before/after from this command.

What it measures — retrieval only, no LLM:

  hit@k   the share of questions where an expected span appears in the top k
          passages. This is the ceiling on answer quality: a passage the model
          never sees cannot be answered from.
  MRR     mean reciprocal rank of the first correct passage. hit@k says the
          evidence arrived; MRR says how near the top, which matters because
          the generator weights early sources more.

Both are reported at stage 1 (bi-encoder, `candidates` deep) and stage 2 (after
the cross-encoder reranks down to k), so the reranker has to justify its cost.

Generation is deliberately out of scope. Whether the model stays grounded in the
passages it was given is a separate question — `--llm` runs the refusal probes
against Ollama for that, and is skipped by default so the retrieval numbers stay
cheap and repeatable.
"""
import argparse
import json
import re
import sys
import tempfile
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from backend.chunker import chunk_pages
from backend.models import Citation

HERE = Path(__file__).parent
CORPUS = HERE / "corpus"
DATASET = HERE / "dataset.json"

DEFAULT_K = 5
DEFAULT_CANDIDATES = 20


# ——— text matching ———


def normalize(text: str) -> str:
    """Fold the differences a PDF introduces but a reader would never notice.

    NFKC turns the ligatures PDF text layers are full of ("ﬁne-tuning") back into
    ASCII, and collapsing whitespace removes the line wrapping that would
    otherwise split an expected span across a newline and score a false miss.
    """
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", text)).strip().lower()


def contains_any(haystack: str, spans: list[str]) -> bool:
    hay = normalize(haystack)
    return any(normalize(span) in hay for span in spans)


# ——— corpus ———


def build_corpus() -> None:
    """Turn every committed .txt into a PDF, so the eval needs no binaries in git.

    A source named `*_2col.txt` is laid out in TWO COLUMNS. That is not cosmetic:
    multi-column reading order is the one thing `get_text(sort=True)` exists to
    fix, and a corpus of single-column documents cannot measure it either way.
    Each `===PAGE===` block becomes one column, paired left/right onto a page.
    """
    import fitz

    for source in sorted(CORPUS.glob("*.txt")):
        blocks = [b.strip() for b in source.read_text(encoding="utf-8").split("===PAGE===")]
        doc = fitz.open()
        if source.stem.endswith("_2col"):
            # blocks[0] is the title banner; the rest pair up as columns.
            title, columns = blocks[0], blocks[1:]
            for i in range(0, len(columns), 2):
                page = doc.new_page()
                page.insert_textbox(fitz.Rect(56, 56, 556, 90), title, fontsize=11)
                page.insert_textbox(fitz.Rect(56, 100, 290, 736), columns[i], fontsize=9)
                if i + 1 < len(columns):
                    page.insert_textbox(fitz.Rect(306, 100, 540, 736), columns[i + 1], fontsize=9)
            n_pages = (len(columns) + 1) // 2
        else:
            for text in blocks:
                doc.new_page().insert_textbox(fitz.Rect(56, 56, 556, 736), text, fontsize=9)
            n_pages = len(blocks)
        out = source.with_suffix(".pdf")
        doc.save(str(out))
        doc.close()
        layout = "2-column" if source.stem.endswith("_2col") else "1-column"
        print(f"built {out.relative_to(HERE.parent.parent)} ({n_pages} pages, {layout})")


def available_docs() -> dict[str, Path]:
    return {p.stem: p for p in sorted(CORPUS.glob("*.pdf"))}


# ——— dataset ———


@dataclass
class Question:
    id: str
    doc: str
    kind: str
    question: str
    expect: list[str]


def load_dataset() -> tuple[list[Question], list[dict]]:
    raw = json.loads(DATASET.read_text(encoding="utf-8"))
    questions = [Question(**{k: v for k, v in q.items()}) for q in raw["questions"]]
    return questions, raw.get("refusals", [])


def check_labels(questions: list[Question]) -> int:
    """Verify every expected span really occurs in its document's extracted text.

    A mislabelled span scores as a permanent miss and quietly drags the baseline
    down forever, which is worse than no eval at all — so this fails loudly.
    """
    from backend.parser import extract_pages

    docs = available_docs()
    bad, skipped = [], []
    for q in questions:
        if q.doc not in docs:
            skipped.append(q.id)
            continue
        text = " ".join(extract_pages(docs[q.doc]))
        for span in q.expect:
            if normalize(span) not in normalize(text):
                bad.append((q.id, span))

    for qid, span in bad:
        print(f"  BAD LABEL {qid}: span not found in the document -> {span!r}")
    if skipped:
        print(f"  skipped {len(skipped)} question(s), document not in the corpus: {', '.join(skipped)}")
    checked = len(questions) - len(skipped)
    print(f"{checked - len({b[0] for b in bad})}/{checked} labels verified against the source text")
    return 1 if bad else 0


# ——— indexing + scoring ———


def build_index(docs: dict[str, Path], store_dir: Path) -> dict[str, int]:
    """Index the corpus exactly the way an upload would, into a throwaway store."""
    from backend import lexical
    from backend.embedder import embed
    from backend.store import append_to_store
    from backend.parser import extract_pages

    counts: dict[str, int] = {}
    for doc_id, path in docs.items():
        chunks = chunk_pages(extract_pages(path), doc_id)
        if not chunks:
            print(f"  WARNING: {doc_id} produced no chunks")
            continue
        append_to_store(chunks, embed([c.embed_text for c in chunks], progress=False), store_dir)
        lexical.index_chunks(chunks, store_dir)
        counts[doc_id] = len(chunks)
        print(f"  indexed {doc_id}: {len(chunks)} chunks")
    return counts


def first_hit_rank(citations: list[Citation], q: Question) -> int | None:
    """1-based rank of the first passage containing an expected span."""
    for rank, c in enumerate(citations, start=1):
        if c.paper_id == q.doc and contains_any(c.text, q.expect):
            return rank
    return None


@dataclass
class Score:
    hits: int = 0
    top1: int = 0
    total: int = 0
    rr: float = 0.0

    def add(self, rank: int | None) -> None:
        self.total += 1
        if rank is not None:
            self.hits += 1
            self.rr += 1 / rank
            if rank == 1:
                self.top1 += 1

    @property
    def hit_rate(self) -> float:
        return self.hits / self.total if self.total else 0.0

    @property
    def hit1(self) -> float:
        return self.top1 / self.total if self.total else 0.0

    @property
    def mrr(self) -> float:
        return self.rr / self.total if self.total else 0.0


def run(k: int, candidates: int, dense_only: bool = False) -> int:
    from backend import lexical
    from backend.reranker import Reranker
    from backend.retriever import Retriever
    from backend.search import shortlist

    questions, refusals = load_dataset()
    docs = available_docs()
    if not docs:
        print("No PDFs in the corpus. Run: python -m backend.eval --build-corpus")
        return 1

    runnable = [q for q in questions if q.doc in docs]
    missing = sorted({q.doc for q in questions} - set(docs))
    if missing:
        print(f"NOTE: skipping {len(questions) - len(runnable)} question(s); "
              f"missing document(s): {', '.join(missing)} (see corpus/README.md)\n")

    with tempfile.TemporaryDirectory(prefix="scholar-eval-") as tmp:
        store = Path(tmp)
        print(f"Indexing {len(docs)} document(s)...")
        per_doc = build_index(docs, store)
        n_chunks = sum(per_doc.values())
        print(f"  {n_chunks} chunks total\n")

        # What actually limits this eval is how hard it is to pick the right
        # chunk out of the whole corpus, so report the size of that haystack.
        # (An earlier version warned per-document that "a doc with <= k chunks
        # cannot fail hit@k". That was simply wrong: retrieval ranks over every
        # chunk in the corpus, not within a document, so a small document's
        # chunks are not guaranteed a place in the top k.)
        print(f"  retrieving {k} of {n_chunks} chunks across {len(per_doc)} documents "
              f"(1 in {n_chunks // k})")
        if n_chunks < 500:
            print("  NOTE: a real professional library is thousands of chunks. These numbers")
            print("        are an optimistic ceiling, not a product metric.")
        print()

        retriever, reranker = Retriever(store), Reranker()
        stage1, stage2 = Score(), Score()
        dense_score, sparse_score = Score(), Score()
        by_kind: dict[str, Score] = {}
        misses: list[Question] = []

        for q in runnable:
            # Deliberately the SAME function /ask uses. Reimplementing the
            # shortlist here would mean the eval measures a lookalike.
            cands = shortlist(q.question, store, retriever, k=candidates, dense_only=dense_only)

            # Only for the per-retriever attribution below, not for scoring.
            dense = retriever.retrieve(q.question, k=candidates)
            sparse = [] if dense_only else lexical.search(q.question, store, k=candidates)

            top = reranker.rerank(q.question, cands, top_k=k)

            # Each retriever scored on its own too, so the fusion has to justify
            # itself against both rather than only against the previous baseline.
            dense_score.add(first_hit_rank(dense, q))
            sparse_score.add(first_hit_rank(sparse, q))

            stage1.add(first_hit_rank(cands, q))
            rank = first_hit_rank(top, q)
            stage2.add(rank)
            by_kind.setdefault(q.kind, Score()).add(rank)
            if rank is None:
                misses.append(q)

    mode = "DENSE ONLY" if dense_only else "HYBRID (dense + lexical, RRF)"
    print("=" * 72)
    print(f"RETRIEVAL — {mode}")
    print(f"k={k}  candidates={candidates}  n={stage2.total} questions over {n_chunks} chunks")
    print("=" * 72)
    if not dense_only:
        print(f"  retriever: dense  only        hit@{candidates} {dense_score.hit_rate:6.1%}"
              f"   MRR {dense_score.mrr:.3f}")
        print(f"  retriever: lexical only       hit@{candidates} {sparse_score.hit_rate:6.1%}"
              f"   MRR {sparse_score.mrr:.3f}")
    print(f"  stage 1 (shortlist,  top {candidates})   hit@{candidates} {stage1.hit_rate:6.1%}"
          f"   MRR {stage1.mrr:.3f}")
    print(f"  stage 2 (reranked,   top {k})    hit@{k}  {stage2.hit_rate:6.1%}"
          f"   MRR {stage2.mrr:.3f}   hit@1 {stage2.hit1:6.1%}")
    print()
    print("  by question kind (after reranking):")
    for kind in sorted(by_kind):
        s = by_kind[kind]
        print(f"    {kind:<10} hit@{k} {s.hit_rate:6.1%}   MRR {s.mrr:.3f}"
              f"   hit@1 {s.hit1:6.1%}   (n={s.total})")

    if misses:
        print(f"\n  {len(misses)} miss(es) — the evidence never reached the model:")
        for q in misses:
            print(f"    [{q.kind}] {q.id}  {q.question}")

    print(f"\n  refusal probes: {len(refusals)} defined, not run "
          f"(needs a live LLM — see the module docstring)")
    print("=" * 68)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(prog="python -m backend.eval", description=__doc__.split("\n")[0])
    p.add_argument("--build-corpus", action="store_true", help="generate PDFs from the committed .txt sources")
    p.add_argument("--check", action="store_true", help="verify labels against the source text; no models loaded")
    p.add_argument("-k", type=int, default=DEFAULT_K, help=f"passages reaching the LLM (default {DEFAULT_K})")
    p.add_argument("--candidates", type=int, default=DEFAULT_CANDIDATES,
                   help=f"stage-1 shortlist depth (default {DEFAULT_CANDIDATES})")
    p.add_argument("--dense-only", action="store_true",
                   help="disable lexical search and fusion — the pre-hybrid baseline")
    args = p.parse_args()

    if args.build_corpus:
        build_corpus()
        return 0
    if args.check:
        return check_labels(load_dataset()[0])
    return run(args.k, args.candidates, dense_only=args.dense_only)


if __name__ == "__main__":
    sys.exit(main())
