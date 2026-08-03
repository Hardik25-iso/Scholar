"""Query expansion.

Two strategies are implemented; only one works, and the tests say which. PRF is
kept rather than deleted because "we tried pseudo-relevance feedback" is a
question someone will ask again, and a measured answer is more useful than an
absence.
"""
import pytest

from backend import expansion
from backend.models import Citation


def cite(text: str, chunk_index: int = 0) -> Citation:
    return Citation(paper_id="d", page=0, chunk_index=chunk_index, score=0.0, text=text)


# ——— term mining ———


def test_terms_common_in_the_top_and_rare_below_are_chosen():
    feedback = [cite(f"The availability of the hosted service is measured monthly. {i}", i)
                for i in range(4)]
    background = [cite(f"Invoices are payable within thirty days of receipt. {i}", 10 + i)
                  for i in range(16)]
    terms = expansion.expansion_terms("uptime commitment", feedback + background)
    assert "availability" in terms


def test_a_term_in_only_one_feedback_passage_is_ignored():
    """One passage must not vote its own vocabulary into the query — that is how
    a single off-topic hit turns into query drift."""
    feedback = [cite("singleton vocabulary here", 0)] + [cite("shared wording", i) for i in (1, 2, 3)]
    background = [cite("unrelated filler text", 10 + i) for i in range(16)]
    terms = expansion.expansion_terms("q", feedback + background)
    assert "singleton" not in terms


def test_terms_already_in_the_question_are_not_repeated():
    feedback = [cite("availability availability availability", i) for i in range(4)]
    background = [cite("filler", 10 + i) for i in range(16)]
    assert "availability" not in expansion.expansion_terms("availability target", feedback + background)


def test_stopwords_and_question_words_are_excluded():
    feedback = [cite("the agreement says that it shall be used", i) for i in range(4)]
    background = [cite("filler", 10 + i) for i in range(16)]
    terms = expansion.expansion_terms("q", feedback + background)
    assert not ({"the", "that", "it", "shall", "says", "used"} & set(terms))


def test_a_term_common_everywhere_is_not_chosen():
    """It describes the corpus, not the answer."""
    everywhere = [cite(f"agreement clause number {i}", i) for i in range(20)]
    assert "agreement" not in expansion.expansion_terms("q", everywhere)


def test_expansion_is_skipped_without_a_background_to_contrast_against():
    """With no background every term scores zero, so expansion would be noise."""
    assert expansion.expansion_terms("q", [cite("some text", i) for i in range(3)]) == []


def test_expansion_is_skipped_on_no_candidates():
    assert expansion.expand("q", []) == ("q", [])


def test_the_original_question_stays_at_the_front():
    """The user's wording must still dominate the embedding and the reranker."""
    feedback = [cite(f"availability of the hosted service {i}", i) for i in range(4)]
    background = [cite("filler", 10 + i) for i in range(16)]
    expanded, terms = expansion.expand("uptime commitment", feedback + background)
    assert terms
    assert expanded.startswith("uptime commitment ")


def test_expansion_is_deterministic():
    """PRF's one advantage over HyDE: the same input gives the same query, so
    retrieval stays reproducible and the audit log's claim survives."""
    passages = [cite(f"availability of the hosted service {i}", i) for i in range(4)]
    passages += [cite(f"unrelated filler {i}", 10 + i) for i in range(16)]
    first = expansion.expand("uptime commitment", passages)
    assert first == expansion.expand("uptime commitment", passages)


# ——— the default ———


def test_expansion_is_off_by_default():
    """PRF was measured and REJECTED: it made every category worse and did not
    move the class it targeted. HyDE works but costs an LLM call and makes
    retrieval non-deterministic, so it is opt-in. Neither is a default."""
    from backend.config import settings

    assert settings.query_expansion == "none"


@pytest.mark.slow
def test_shortlist_ignores_expansion_when_mode_is_none(tmp_path):
    from backend import lexical
    from backend.embedder import embed
    from backend.models import Chunk
    from backend.retriever import Retriever
    from backend.search import shortlist
    from backend.store import append_to_store

    chunks = [Chunk(paper_id="d", page=0, chunk_index=i, text=f"Paragraph {i} about retrieval.",
                    embed_text=f"Paragraph {i} about retrieval.") for i in range(4)]
    append_to_store(chunks, embed([c.embed_text for c in chunks], progress=False), tmp_path)
    lexical.index_chunks(chunks, tmp_path)

    _, effective = shortlist("retrieval", tmp_path, Retriever(tmp_path), k=4)
    assert effective == "retrieval", "query was expanded despite mode 'none'"
