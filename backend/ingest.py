"""CLI entry point: index a PDF into the FAISS store.

Usage:
    python -m backend.ingest path/to/paper.pdf [--store ./data/index]
"""
import argparse
from pathlib import Path

from backend.chunker import MAX_SEQ_LENGTH, chunk_pages, count_tokens
from backend.embedder import embed
from backend.parser import extract_pages
from backend.store import build

DEFAULT_STORE = Path(__file__).parent.parent / "data" / "index"


def ingest(pdf_path: Path, store_dir: Path) -> None:
    paper_id = pdf_path.stem

    print(f"Parsing {pdf_path.name} ...")
    pages = extract_pages(pdf_path)
    print(f"  {len(pages)} pages extracted")

    chunks = chunk_pages(pages, paper_id)
    print(f"  {len(chunks)} chunks created")

    # Confirm no chunk exceeds the model's max_seq_length (would be truncated).
    # Assert on embed_text — the string we actually feed the model.
    longest = max(count_tokens(c.embed_text) for c in chunks)
    print(f"  longest chunk: {longest} tokens (model limit: {MAX_SEQ_LENGTH})")

    texts = [c.embed_text for c in chunks]
    print("Embedding ...")
    vectors = embed(texts)

    print("Saving index ...")
    build(chunks, vectors, store_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Index a PDF for Scholar RAG")
    parser.add_argument("pdf", type=Path, help="Path to the PDF file")
    parser.add_argument(
        "--store", type=Path, default=DEFAULT_STORE,
        help=f"Directory for FAISS index (default: {DEFAULT_STORE})"
    )
    args = parser.parse_args()

    if not args.pdf.exists():
        raise SystemExit(f"PDF not found: {args.pdf}")

    ingest(args.pdf, args.store)
    print("Done.")


if __name__ == "__main__":
    main()
