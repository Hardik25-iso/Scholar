"""CLI entry point: ask a question against the indexed papers.

Usage:
    python -m backend.query "How does multi-head attention work?" [--k 5]

Flow (the full RAG query path, two-stage retrieval):
    question -> embed -> FAISS top-N candidates -> cross-encoder reranks to top-k
             -> local LLM answers from those chunks
             -> print grounded answer + numbered citations
"""
import argparse
import sys
from pathlib import Path

from backend.generator import generate
from backend.reranker import Reranker
from backend.retriever import Retriever

DEFAULT_STORE = Path(__file__).parent.parent / "data" / "index"


def main() -> None:
    # Papers contain math/Unicode (∈, →, α). Windows' default console codec is
    # cp1252 and can't encode those, so force UTF-8 output regardless of shell.
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    parser = argparse.ArgumentParser(description="Ask the indexed papers a question")
    parser.add_argument("question", type=str, help="Your question")
    parser.add_argument("--k", type=int, default=5,
                        help="How many chunks to keep after reranking (fed to LLM)")
    parser.add_argument("--candidates", type=int, default=20,
                        help="How many chunks to retrieve before reranking")
    parser.add_argument("--store", type=Path, default=DEFAULT_STORE)
    args = parser.parse_args()

    retriever = Retriever(args.store)
    reranker = Reranker()

    # Stage 1: cast a wide net with the fast bi-encoder.
    print(f"Retrieving top-{args.candidates} candidates ...")
    candidates = retriever.retrieve(args.question, k=args.candidates)

    # Stage 2: re-score those candidates with the cross-encoder, keep the best k.
    print(f"Reranking down to top-{args.k} ...")
    citations = reranker.rerank(args.question, candidates, top_k=args.k)

    print("Generating answer with local LLM (gemma3:4b) ...\n")
    answer = generate(args.question, citations)

    print("=" * 70)
    print("ANSWER")
    print("=" * 70)
    print(answer.answer)
    print()
    print("=" * 70)
    print("SOURCES")
    print("=" * 70)
    for i, c in enumerate(answer.citations, start=1):
        snippet = c.text.replace("\n", " ")
        if len(snippet) > 160:
            snippet = snippet[:160] + "…"
        rr = f"{c.rerank_score:+.2f}" if c.rerank_score is not None else "n/a"
        # cosine = stage-1 similarity; rerank = stage-2 relevance the order is by.
        print(f"[{i}] {c.paper_id}, p.{c.page + 1}  (cosine {c.score:.3f}, rerank {rr})")
        print(f"    {snippet}\n")


if __name__ == "__main__":
    main()
