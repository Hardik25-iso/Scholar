"""Stage-1 retrieval: lexical search, rank fusion, diversity cap, filtering.

Everything here except the end-to-end shortlist test runs without a model, so
the retrieval logic is covered in the fast tier.
"""
from pathlib import Path

import pytest

from backend import lexical
from backend.fusion import reciprocal_rank_fusion
from backend.models import Chunk, Citation
from backend.search import cap_per_document, shortlist


def cite(paper_id: str, chunk_index: int, text: str = "x") -> Citation:
    return Citation(paper_id=paper_id, page=0, chunk_index=chunk_index, score=0.0, text=text)


def chunk(paper_id: str, chunk_index: int, text: str) -> Chunk:
    return Chunk(paper_id=paper_id, page=0, chunk_index=chunk_index, text=text, embed_text=text)


# ——— FTS5 query escaping ———


@pytest.mark.parametrize("question", [
    "What does Section 7.2 say?",
    "Force Majeure (clause 7.2) — what is it?",
    'Who signed "the agreement"?',
    "AND OR NOT NEAR",
    "column:value",
    "a * b",
])
def test_user_input_never_reaches_fts5_as_syntax(tmp_path: Path, question):
    """FTS5's query language treats `. : " * ( ) -` and bare AND/OR/NOT as
    operators. Unescaped, "Section 7.2" raises `fts5: syntax error near "."` —
    so every token is quoted into a literal."""
    lexical.index_chunks([chunk("doc", 0, "Section 7.2 Force Majeure applies here.")], tmp_path)
    lexical.search(question, tmp_path, k=5)  # must not raise


def test_exact_section_reference_is_found(tmp_path: Path):
    """The query class dense retrieval is worst at, and the reason this exists."""
    lexical.index_chunks([
        chunk("agreement", 0, "7.1 Liability. Each party's aggregate liability is capped."),
        chunk("agreement", 1, "7.2 Force Majeure. Neither party shall be liable for delay."),
        chunk("agreement", 2, "8.1 Termination. Either party may terminate on notice."),
    ], tmp_path)

    top = lexical.search("What does Section 7.2 say?", tmp_path, k=3)
    assert top, "lexical search returned nothing"
    assert "Force Majeure" in top[0].text


def test_a_dotted_reference_stays_one_phrase(tmp_path: Path):
    """`7.2` must not be split into `7` OR `2` — that also matches 7.1, 8.2 and
    every other clause containing a 7, and makes 7.1 and 7.2 tie under fusion."""
    assert lexical.to_match_query("Section 7.2") == '"Section" OR "7.2"'

    lexical.index_chunks([
        chunk("a", 0, "7.1 Liability applies here."),
        chunk("a", 1, "7.2 Force Majeure applies here."),
        chunk("a", 2, "2.7 Renewal applies here."),
    ], tmp_path)
    found = [c.chunk_index for c in lexical.search("7.2", tmp_path, k=5)]
    assert found == [1], f"expected only the 7.2 chunk, got chunks {found}"


def test_search_on_a_missing_index_returns_empty(tmp_path: Path):
    assert lexical.search("anything", tmp_path / "nope", k=5) == []


def test_a_query_of_pure_punctuation_returns_empty(tmp_path: Path):
    lexical.index_chunks([chunk("doc", 0, "some text")], tmp_path)
    assert lexical.search("?!...", tmp_path, k=5) == []


def test_scores_are_higher_is_better(tmp_path: Path):
    """bm25() returns more-negative-is-better; the module negates it so both
    retrievers agree on direction before fusion."""
    lexical.index_chunks([
        chunk("d", 0, "force majeure force majeure force majeure"),
        chunk("d", 1, "an unrelated paragraph about invoicing"),
    ], tmp_path)
    results = lexical.search("force majeure", tmp_path, k=2)
    assert results[0].score >= results[-1].score


def test_removing_a_paper_drops_only_its_rows(tmp_path: Path):
    lexical.index_chunks([chunk("keep", 0, "alpha content"), chunk("drop", 0, "alpha content")], tmp_path)
    lexical.remove_paper(tmp_path, "drop")
    found = {c.paper_id for c in lexical.search("alpha", tmp_path, k=10)}
    assert found == {"keep"}


def test_removing_from_a_missing_index_is_a_no_op(tmp_path: Path):
    lexical.remove_paper(tmp_path / "nope", "whatever")  # must not raise


# ——— rank fusion ———


def test_being_found_by_both_retrievers_beats_topping_only_one():
    """The property that makes fusion worth doing: a passage both retrievers
    surface outranks one that a single retriever puts first and the other never
    returns at all."""
    found_by_both, dense_only_top = cite("d", 1), cite("d", 2)
    fused = reciprocal_rank_fusion([
        [dense_only_top, found_by_both],
        [found_by_both],
    ])
    assert fused[0].chunk_index == found_by_both.chunk_index


def test_rrf_prefers_a_strong_opinion_to_uniform_mediocrity():
    """Documents a real, slightly counter-intuitive consequence of 1/(K+rank).

    Because 1/x is convex, 1/61 + 1/63 > 2/62: ranked 1st-and-3rd narrowly beats
    ranked 2nd-and-2nd. RRF is often described as "consensus wins", which would
    predict the opposite. Pinned so the behaviour is a decision, not a surprise.
    """
    spread, middling = cite("d", 1), cite("d", 2)
    fused = reciprocal_rank_fusion([
        [spread, middling, cite("d", 3)],
        [cite("d", 4), middling, spread],
    ])
    assert fused[0].chunk_index == spread.chunk_index


def test_fusion_deduplicates_across_lists():
    shared = cite("d", 1)
    fused = reciprocal_rank_fusion([[shared], [cite("d", 1), cite("d", 2)]])
    assert [c.chunk_index for c in fused] == [1, 2]


def test_fusion_keeps_a_real_score_not_a_fused_number():
    """`score` is documented as the stage-1 similarity. Writing an RRF value into
    it would make the field a lie; fusion expresses itself through ORDER only."""
    original = cite("d", 1)
    original.score = 0.87
    assert reciprocal_rank_fusion([[original]])[0].score == 0.87


def test_fusion_of_one_list_preserves_its_order():
    ranked = [cite("d", i) for i in range(5)]
    assert reciprocal_rank_fusion([ranked]) == ranked


def test_fusion_handles_an_empty_side():
    """Lexical returns nothing for a punctuation-only query; the dense list must
    still come through unchanged rather than being lost."""
    ranked = [cite("d", i) for i in range(3)]
    assert reciprocal_rank_fusion([ranked, []]) == ranked


def test_fusion_of_nothing_is_empty():
    assert reciprocal_rank_fusion([[], []]) == []


# ——— diversity cap ———


def test_one_document_cannot_monopolise_the_shortlist():
    """"Compare these three contracts" is unanswerable if all five passages come
    from one of them."""
    hogging = [cite("a", i) for i in range(10)] + [cite("b", 0), cite("c", 0)]
    capped = cap_per_document(hogging, limit=5)
    assert len({c.paper_id for c in capped}) == 3
    assert sum(c.paper_id == "a" for c in capped) == 3  # int(5 * 0.6)


def test_over_quota_passages_are_demoted_not_discarded():
    """A document that genuinely holds every answer must still supply them."""
    only_one_doc = [cite("a", i) for i in range(5)]
    capped = cap_per_document(only_one_doc, limit=5)
    assert len(capped) == 5
    assert [c.chunk_index for c in capped] == [0, 1, 2, 3, 4]


def test_cap_preserves_relative_order_within_the_quota():
    ordered = [cite("a", 0), cite("b", 0), cite("a", 1), cite("b", 1)]
    assert [(c.paper_id, c.chunk_index) for c in cap_per_document(ordered, limit=4)] == [
        ("a", 0), ("b", 0), ("a", 1), ("b", 1)
    ]


def test_cap_always_allows_at_least_one_per_document():
    """int(1 * 0.6) == 0 would otherwise return nothing at all."""
    assert len(cap_per_document([cite("a", 0), cite("b", 0)], limit=1)) == 1


# ——— the shortlist, end to end ———


@pytest.mark.slow
def test_shortlist_finds_an_exact_reference_dense_search_misses(tmp_path: Path):
    """The Phase 2 headline, asserted rather than asserted-about."""
    from backend.embedder import embed
    from backend.retriever import Retriever
    from backend.store import append_to_store

    chunks = [
        chunk("agreement", 0, "7.1 Liability. Each party's aggregate liability under this "
                              "agreement is capped at the fees paid in the preceding year."),
        chunk("agreement", 1, "7.2 Force Majeure. Neither party shall be liable for any failure "
                              "or delay resulting from an event beyond its reasonable control."),
        chunk("agreement", 2, "8.1 Termination. Either party may terminate immediately on "
                              "written notice for an unremedied material breach."),
    ]
    append_to_store(chunks, embed([c.embed_text for c in chunks], progress=False), tmp_path)
    lexical.index_chunks(chunks, tmp_path)

    top, _ = shortlist("What does Section 7.2 say?", tmp_path, Retriever(tmp_path), k=3)
    assert "Force Majeure" in top[0].text


@pytest.mark.slow
def test_shortlist_can_be_restricted_to_selected_documents(tmp_path: Path):
    from backend.embedder import embed
    from backend.retriever import Retriever
    from backend.store import append_to_store

    chunks = [
        chunk("contract_a", 0, "7.2 Force Majeure. Neither party shall be liable for delay."),
        chunk("contract_b", 0, "7.2 Assignment. Neither party may assign without consent."),
    ]
    append_to_store(chunks, embed([c.embed_text for c in chunks], progress=False), tmp_path)
    lexical.index_chunks(chunks, tmp_path)

    retriever = Retriever(tmp_path)
    both, _ = shortlist("What does clause 7.2 say?", tmp_path, retriever, k=5)
    assert {c.paper_id for c in both} == {"contract_a", "contract_b"}

    scoped, _ = shortlist("What does clause 7.2 say?", tmp_path, retriever, k=5, papers=["contract_b"])
    assert {c.paper_id for c in scoped} == {"contract_b"}


@pytest.mark.slow
def test_a_lexically_found_passage_still_carries_its_audit_trail(tmp_path: Path):
    """RRF keeps the instance from whichever retriever ranked a passage best. A
    lexical hit knows only (paper_id, chunk_index), so without hydration the
    audit trail would depend on which retriever happened to win."""
    from backend.embedder import embed
    from backend.retriever import Retriever
    from backend.store import append_to_store

    chunks = [
        Chunk(paper_id="agreement", page=0, chunk_index=0,
              text="7.2 Force Majeure. Neither party shall be liable for delay.",
              embed_text="7.2 Force Majeure. Neither party shall be liable for delay.",
              char_start=100, char_end=159, unit="section"),
    ]
    append_to_store(chunks, embed([c.embed_text for c in chunks], progress=False), tmp_path)
    lexical.index_chunks(chunks, tmp_path)

    retriever = Retriever(tmp_path)
    assert lexical.search("7.2", tmp_path, k=5)[0].faiss_id is None, "premise: lexical has none"

    top = shortlist("What does 7.2 say?", tmp_path, retriever, k=5)[0][0]
    assert top.faiss_id == 0
    assert (top.char_start, top.char_end) == (100, 159)
    assert top.locator == "section 1"


@pytest.mark.slow
def test_dense_only_shortlist_reproduces_the_pre_hybrid_path(tmp_path: Path):
    """The eval's A/B depends on this flag genuinely bypassing lexical search."""
    from backend.embedder import embed
    from backend.retriever import Retriever
    from backend.store import append_to_store

    chunks = [chunk("d", i, f"Paragraph number {i} about retrieval augmented generation.")
              for i in range(4)]
    append_to_store(chunks, embed([c.embed_text for c in chunks], progress=False), tmp_path)
    lexical.index_chunks(chunks, tmp_path)

    retriever = Retriever(tmp_path)
    dense_only, _ = shortlist("retrieval", tmp_path, retriever, k=4, dense_only=True)
    assert [c.chunk_index for c in dense_only] == [
        c.chunk_index for c in retriever.retrieve("retrieval", k=4)
    ]
