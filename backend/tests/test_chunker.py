"""The chunker's token-boundary contract.

Everything downstream assumes two things that nothing else enforces:
  1. no chunk exceeds the embedder's 384-token window, so no text is silently
     truncated at embed time; and
  2. `text` is a VERBATIM slice of the source page, so a citation quotes the
     paper rather than a decode() round-trip of it.

These are cheap (tokenizer only, no embedding model) and run on every commit.
"""
import pytest

from backend.chunker import CHUNK_TOKENS, MAX_SEQ_LENGTH, OVERLAP, chunk_pages, count_tokens

PARAGRAPH = (
    "Retrieval-augmented generation combines a parametric seq2seq model with a "
    "non-parametric memory accessed by a pretrained neural retriever. "
)


def _long_page(repeats: int = 60) -> str:
    return PARAGRAPH * repeats


# ——— the 384-token guarantee ———


def test_no_chunk_exceeds_the_embedder_window():
    chunks = chunk_pages([_long_page()], "paper")
    assert chunks
    over = [(c.chunk_index, count_tokens(c.text)) for c in chunks if count_tokens(c.text) > MAX_SEQ_LENGTH]
    assert over == [], f"chunks over the {MAX_SEQ_LENGTH}-token limit: {over}"


def test_budget_arithmetic_leaves_room_for_the_special_tokens():
    """380 content tokens + [CLS] + [SEP] must fit in 384."""
    assert CHUNK_TOKENS + 2 <= MAX_SEQ_LENGTH


def test_a_long_page_produces_multiple_chunks():
    chunks = chunk_pages([_long_page()], "paper")
    assert len(chunks) > 1, "the page should have been split"


# ——— verbatim slicing ———


def test_chunk_text_is_a_verbatim_slice_of_the_page():
    """Case and spacing must survive — a citation is a quote, not a paraphrase."""
    page = _long_page()
    for chunk in chunk_pages([page], "paper"):
        assert chunk.text in page, "chunk text was not found verbatim in the page"


def test_casing_is_preserved():
    page = "Ashish Vaswani and Noam Shazeer wrote Attention Is All You Need at Google Brain. " * 8
    chunks = chunk_pages([page], "paper")
    assert "Ashish Vaswani" in chunks[0].text


def test_embed_text_matches_the_displayed_text():
    """Today they are the same span. Phase 1 will diverge them (Markdown tables
    in embed_text); until then this pins the current contract."""
    for chunk in chunk_pages([_long_page()], "paper"):
        assert chunk.embed_text == chunk.text


# ——— overlap and ordering ———


def test_consecutive_chunks_overlap():
    """Overlap is what stops an answer being cut in half at a chunk boundary."""
    chunks = chunk_pages([_long_page()], "paper")
    assert len(chunks) >= 2
    a, b = chunks[0], chunks[1]
    tail = a.text[-200:]
    assert any(word in b.text for word in tail.split()[-8:]), "no overlap between adjacent chunks"


def test_overlap_is_smaller_than_the_window():
    """Otherwise the sliding window would never advance."""
    assert 0 < OVERLAP < CHUNK_TOKENS


def test_chunk_indices_are_contiguous_and_ordered():
    chunks = chunk_pages([_long_page(), _long_page()], "paper")
    assert [c.chunk_index for c in chunks] == list(range(len(chunks)))


def test_page_numbers_are_zero_indexed_and_ascending():
    chunks = chunk_pages([_long_page(20), _long_page(20)], "paper")
    pages = [c.page for c in chunks]
    assert min(pages) == 0
    assert set(pages) == {0, 1}
    assert pages == sorted(pages)


# ——— degenerate input ———


@pytest.mark.parametrize("pages", [[], [""], ["", "", ""], ["   \n\t  "]])
def test_empty_input_produces_no_chunks(pages):
    """This is the signal papers.py turns into a 422 for a scanned PDF."""
    assert chunk_pages(pages, "paper") == []


def test_a_page_shorter_than_the_window_is_one_chunk():
    chunks = chunk_pages(["A short page of text.", ""], "paper")
    assert len(chunks) == 1
    assert chunks[0].text == "A short page of text."


# ——— char spans: the step from "page 12" to "these exact characters" ———


def test_the_char_span_slices_back_to_the_chunk_text_exactly():
    """The property the whole citation claim rests on: page[start:end] IS the
    quoted passage. If this drifts, a highlight points at the wrong words."""
    page = _long_page()
    for chunk in chunk_pages([page], "paper"):
        assert page[chunk.char_start:chunk.char_end] == chunk.text


def test_spans_are_recorded_per_page_not_across_the_document():
    """Offsets are into their own page's text, so page 2's first chunk starts
    near 0 again rather than continuing page 1's numbering."""
    page = _long_page(20)
    chunks = chunk_pages([page, page], "paper")
    second_page = [c for c in chunks if c.page == 1]
    assert second_page[0].char_start < len(page)
    for chunk in second_page:
        assert page[chunk.char_start:chunk.char_end] == chunk.text


def test_spans_advance_within_a_page():
    chunks = [c for c in chunk_pages([_long_page()], "paper") if c.page == 0]
    starts = [c.char_start for c in chunks]
    assert starts == sorted(starts)
    assert len(set(starts)) == len(starts), "two chunks share a start offset"


def test_consecutive_spans_overlap_like_the_token_windows():
    chunks = chunk_pages([_long_page()], "paper")
    assert chunks[1].char_start < chunks[0].char_end, "the 50-token overlap is missing"


def test_leading_whitespace_does_not_shift_the_span():
    """`.strip()` shaves the window's edges, so the recorded start must be the
    trimmed text's position, not the raw window's."""
    page = "\n\n\n   " + PARAGRAPH * 30
    chunk = chunk_pages([page], "paper")[0]
    assert page[chunk.char_start:chunk.char_end] == chunk.text
    assert not chunk.text.startswith(" ")


# ——— locator: naming a location truthfully ———


@pytest.mark.parametrize("unit,expected", [
    ("page", "page 1"), ("slide", "slide 1"), ("sheet", "sheet 1"), ("section", "section 1"),
])
def test_a_chunk_carries_the_unit_it_was_chunked_with(unit, expected):
    from backend.models import Citation

    chunk = chunk_pages(["A short page of text."], "paper", unit=unit)[0]
    assert chunk.unit == unit
    citation = Citation(paper_id="p", page=chunk.page, chunk_index=0, score=0.0,
                        text=chunk.text, unit=chunk.unit)
    assert citation.locator == expected


def test_the_default_unit_is_a_page():
    assert chunk_pages(["Some text here."], "paper")[0].unit == "page"


# ——— the downstream reranker window ———


def test_question_plus_chunk_fits_the_cross_encoder():
    """Guard, not a fix. The reranker's ms-marco-MiniLM has its own 512-token
    limit and its own tokenizer, so a chunk sized against mpnet's 384 is not
    automatically safe there — anything over 512 has its tail silently dropped
    before scoring. Measured today: the worst pair is ~399. This test fails if a
    future chunking change (e.g. Phase 1's serialized tables) breaks that."""
    from transformers import AutoTokenizer

    from backend.reranker import RERANKER_MODEL

    tok = AutoTokenizer.from_pretrained(RERANKER_MODEL)
    question = "What is the difference between the RAG-Sequence and RAG-Token models?"

    worst = max(len(tok.encode(question, c.text)) for c in chunk_pages([_long_page()], "paper"))
    assert worst <= tok.model_max_length, (
        f"(question, chunk) pairs reach {worst} tokens, over the reranker's "
        f"{tok.model_max_length}-token limit — passage tails are being dropped"
    )
