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
    kind: str
    question: str
    doc: str | None = None          # None for a question spanning several documents
    expect: list[str] | None = None  # ANY of these -> the evidence arrived
    # MULTI-PART questions. Each inner list is one required piece of evidence
    # (ANY span within it counts); EVERY piece must be present in the top k.
    # "What is the uptime commitment and what happens if it is missed?" is not
    # answerable from one passage, and scoring it as a hit when only half the
    # evidence arrived would measure the wrong thing — the model would have been
    # handed enough to write a confident, incomplete answer.
    all_of: list[list[str]] | None = None
    docs: list[str] | None = None    # documents a multi-document question needs

    def required_groups(self) -> list[list[str]]:
        return self.all_of if self.all_of else [self.expect or []]

    def documents(self) -> list[str]:
        if self.docs:
            return self.docs
        return [self.doc] if self.doc else []


def load_dataset() -> tuple[list[Question], list[dict]]:
    raw = json.loads(DATASET.read_text(encoding="utf-8"))
    questions = [Question(**q) for q in raw["questions"]]
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
        needed = q.documents()
        if not needed or any(d not in docs for d in needed):
            skipped.append(q.id)
            continue
        text = normalize(" ".join(
            page for d in needed for page in extract_pages(docs[d])
        ))
        for group in q.required_groups():
            # ANY span in a group is enough — they are alternative wordings of
            # the same piece of evidence. A group where NONE match is a typo.
            if not any(normalize(span) in text for span in group):
                bad.append((q.id, " | ".join(group)))

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
    """1-based rank at which the question's evidence is COMPLETE.

    For a single-part question that is the rank of the first matching passage.
    For a multi-part one it is the rank at which the LAST missing piece arrives —
    because a half-answered question is not answered, and rewarding partial
    evidence would let the score improve while the model still cannot answer.
    """
    allowed = set(q.documents())
    outstanding = [list(group) for group in q.required_groups()]

    for rank, c in enumerate(citations, start=1):
        if allowed and c.paper_id not in allowed:
            continue
        outstanding = [g for g in outstanding if not contains_any(c.text, g)]
        if not outstanding:
            return rank
    return None


@dataclass
class Score:
    """Retrieval scores, with the structural ceiling multi-part questions impose.

    A question needing evidence from N distinct passages CANNOT complete before
    rank N — one passage cannot contain two documents' clauses. So its best
    possible reciprocal rank is 1/N, and its hit@1 is unattainable for N > 1.

    Comparing a raw MRR of 0.32 on two-part questions against 0.92 on one-part
    questions is therefore meaningless: the ceilings are 0.5 and 1.0. `ceiling`
    tracks the best achievable score for the questions actually scored, so the
    report can show how much of the attainable performance was reached instead
    of implying a failure that is arithmetic.
    """

    hits: int = 0
    top1: int = 0
    total: int = 0
    rr: float = 0.0
    best_rr: float = 0.0  # sum of 1/parts — the perfect score for these questions

    def add(self, rank: int | None, parts: int = 1) -> None:
        self.total += 1
        self.best_rr += 1 / max(1, parts)
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

    @property
    def attained(self) -> float:
        """Share of the ACHIEVABLE ranking quality reached (1.0 == perfect).

        This is the number to compare across question kinds; raw MRR is not.
        """
        return self.rr / self.best_rr if self.best_rr else 0.0

    @property
    def parts(self) -> float:
        """Average pieces of evidence a question in this slice needs."""
        return self.total / self.best_rr if self.best_rr else 1.0


def run(k: int, candidates: int, dense_only: bool = False,
        expansion_mode: str = "none") -> int:
    from backend import lexical
    from backend.reranker import Reranker
    from backend.retriever import Retriever
    from backend.search import shortlist

    questions, refusals = load_dataset()
    docs = available_docs()
    if not docs:
        print("No PDFs in the corpus. Run: python -m backend.eval --build-corpus")
        return 1

    runnable = [q for q in questions if q.documents() and all(d in docs for d in q.documents())]
    missing = sorted({d for q in questions for d in q.documents()} - set(docs))
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
            cands, retrieval_query = shortlist(
                q.question, store, retriever, k=candidates,
                dense_only=dense_only, expansion_mode=expansion_mode,
            )

            # Only for the per-retriever attribution below, not for scoring.
            dense = retriever.retrieve(q.question, k=candidates)
            sparse = [] if dense_only else lexical.search(q.question, store, k=candidates)

            top = reranker.rerank(retrieval_query, cands, top_k=k)

            # Each retriever scored on its own too, so the fusion has to justify
            # itself against both rather than only against the previous baseline.
            parts = len(q.required_groups())
            dense_score.add(first_hit_rank(dense, q), parts)
            sparse_score.add(first_hit_rank(sparse, q), parts)

            stage1.add(first_hit_rank(cands, q), parts)
            rank = first_hit_rank(top, q)
            stage2.add(rank, parts)
            by_kind.setdefault(q.kind, Score()).add(rank, parts)
            if rank is None:
                misses.append(q)

    mode = "DENSE ONLY" if dense_only else "HYBRID (dense + lexical, RRF)"
    mode += f"  [expansion: {expansion_mode}]"
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
    print(f"    {'kind':<12} {'hit@' + str(k):>7} {'MRR':>7} {'of max':>8} {'parts':>6}   n")
    for kind in sorted(by_kind):
        s = by_kind[kind]
        print(f"    {kind:<12} {s.hit_rate:>7.1%} {s.mrr:>7.3f} {s.attained:>8.1%} "
              f"{s.parts:>6.1f}   {s.total}")
    print()
    print("    'of max' is the share of ACHIEVABLE ranking quality reached. A question")
    print("    needing evidence from N passages cannot complete before rank N, so its")
    print("    best possible MRR is 1/N — comparing raw MRR across kinds with different")
    print("    'parts' counts compares different ceilings and means nothing.")
    print("    Above 100% means the labelled parts turned out to share one passage, so")
    print("    the question was not multi-part for retrieval. That is worth knowing:")
    print("    it means the label, not the retriever, was describing the difficulty.")

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
    p.add_argument("--expansion", choices=("none", "prf", "hyde"), default="none",
                   help="query expansion strategy (default none — prf was measured and rejected)")
    args = p.parse_args()

    if args.build_corpus:
        build_corpus()
        return 0
    if args.check:
        return check_labels(load_dataset()[0])
    return run(args.k, args.candidates, dense_only=args.dense_only,
               expansion_mode=args.expansion)


if __name__ == "__main__":
    sys.exit(main())
