"""PDF text extraction using PyMuPDF.

Returns one string per page; caller decides how to chunk across them.
"""
from pathlib import Path

import fitz  # PyMuPDF


def extract_pages(pdf_path: str | Path) -> list[str]:
    """Return a list of page texts (one entry per page, 0-indexed)."""
    doc = fitz.open(str(pdf_path))
    pages = [page.get_text() for page in doc]
    doc.close()
    return pages
