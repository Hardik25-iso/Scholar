"""Sliding-window text chunker, measured in REAL model tokens.

We chunk by the embedding model's own word-piece tokens — not whitespace
words — because that is the only count that actually determines whether a
chunk fits the model. all-mpnet-base-v2 truncates anything past its default
max_seq_length of 384 word-pieces (the 512 figure is the architectural max,
but the model was fine-tuned and defaults to 384). Word counts are a proxy
that silently drifts over that limit; counting tokens guarantees a fit.

We use the FAST tokenizer's offset mapping to find, for each token window, the
character span it covers in the original page text — then slice the VERBATIM
original text out of those offsets. This preserves real casing and spacing for
citations (e.g. "Ashish Vaswani", "avaswani@google.com") instead of the
lowercased, re-spaced text a decode() round-trip would produce.

Budget (per chunk, total sequence the model sees must be <= 384):
  CHUNK_TOKENS = 380 content tokens
  + 2 special tokens ([CLS], [SEP]) added at encode time
  = 382  <= 384   OK

Overlap = 50 tokens (~13%) so answers aren't split at chunk boundaries.
"""
from functools import lru_cache

from transformers import AutoTokenizer

from backend.embedder import MODEL_NAME
from backend.models import Chunk

CHUNK_TOKENS = 380   # content word-pieces per chunk (excludes special tokens)
OVERLAP = 50         # word-pieces of overlap between consecutive chunks
SPECIAL_TOKENS = 2   # [CLS] + [SEP] added when the model encodes a chunk
MAX_SEQ_LENGTH = 384  # mpnet default; chunks must stay at or under this


@lru_cache(maxsize=1)
def _tokenizer() -> AutoTokenizer:
    # use_fast=True is required for return_offsets_mapping; assert it so a future
    # environment without the fast tokenizer fails loudly instead of silently.
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    assert tok.is_fast, "offset mapping requires the fast tokenizer"
    return tok


def count_tokens(text: str) -> int:
    """Real sequence length the model sees for `text`, incl. special tokens."""
    return len(_tokenizer().encode(text, add_special_tokens=True))


def _char_span(offsets: list[tuple[int, int]]) -> tuple[int, int] | None:
    """Start/end char offsets covered by a token window.

    Special tokens come back as (0, 0); skip any zero-width offset so we don't
    anchor the slice at char 0. Returns None if the window has no real tokens.
    """
    real = [(s, e) for s, e in offsets if e > s]
    if not real:
        return None
    return real[0][0], real[-1][1]


def chunk_pages(pages: list[str], paper_id: str) -> list[Chunk]:
    """Chunk all pages into overlapping windows of CHUNK_TOKENS content tokens."""
    tok = _tokenizer()
    chunks: list[Chunk] = []

    for page_num, page_text in enumerate(pages):
        enc = tok(
            page_text,
            add_special_tokens=False,         # we measure content tokens only
            return_offsets_mapping=True,
        )
        ids = enc["input_ids"]
        offsets = enc["offset_mapping"]
        if not ids:
            continue

        start = 0
        while start < len(ids):
            end = start + CHUNK_TOKENS
            span = _char_span(offsets[start:end])
            verbatim = page_text[span[0]:span[1]].strip() if span else ""

            # Verbatim text is a fresh string; verify it still fits and trim the
            # rare chunk that re-tokenizes over the limit, so the 384 guarantee
            # holds for what we actually embed.
            while verbatim and count_tokens(verbatim) > MAX_SEQ_LENGTH:
                end -= 8
                span = _char_span(offsets[start:end])
                verbatim = page_text[span[0]:span[1]].strip() if span else ""

            if verbatim:
                chunks.append(
                    Chunk(
                        paper_id=paper_id,
                        page=page_num,
                        chunk_index=len(chunks),
                        text=verbatim,        # verbatim original (display)
                        embed_text=verbatim,  # same span, fed to the embedder
                    )
                )
            if start + CHUNK_TOKENS >= len(ids):
                break
            start += CHUNK_TOKENS - OVERLAP

    return chunks
