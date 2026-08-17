"""MMR selection: does it actually trade relevance for distinctness?

The eval measures whether MMR helps on the corpus. These assert it does what it
claims mechanically — otherwise a flat eval result is ambiguous between "MMR
does not help here" and "MMR is not running".
"""
import pytest

from backend import mmr
from backend.models import Citation


def cite(text: str, score: float, paper: str = "doc", idx: int = 0) -> Citation:
    return Citation(paper_id=paper, page=0, chunk_index=idx, score=0.5,
                    text=text, rerank_score=score)


def test_jaccard_is_zero_for_disjoint_and_one_for_identical():
    a, b = {"alpha", "beta"}, {"gamma", "delta"}
    assert mmr.jaccard(a, b) == 0.0
    assert mmr.jaccard(a, a) == 1.0
    assert mmr.jaccard(a, set()) == 0.0


def test_lambda_one_reduces_to_pure_relevance():
    """The A/B baseline must be exactly the old behaviour, or the comparison is
    measuring two changes at once."""
    cands = [cite("alpha beta gamma", 5.0), cite("alpha beta delta", 4.0),
             cite("wholly different words entirely", 3.0)]
    assert mmr.select(cands, k=2, lam=1.0) == cands[:2]


def test_a_near_duplicate_is_passed_over_for_a_distinct_passage():
    """The failure this feature exists for: the chunker's 50-token overlap puts
    two passages sharing most of their text in the top k, so one slot carries no
    new evidence."""
    original = cite("the transformer uses multi head attention layers", 5.0)
    near_dup = cite("the transformer uses multi head attention heads", 4.9)
    distinct = cite("positional encodings inject order information", 4.0)

    picked = mmr.select([original, near_dup, distinct], k=2, lam=0.5)

    assert original in picked, "the most relevant passage must always survive"
    assert distinct in picked, "a distinct passage should beat a near-duplicate"
    assert near_dup not in picked


def test_the_most_relevant_passage_is_always_first():
    """MMR's first pick has nothing to be redundant against, so it is pure
    relevance. A diversity-aware selector that drops the best answer is broken."""
    best = cite("exactly what was asked", 9.0)
    cands = [best, cite("something else", 1.0), cite("another thing", 0.5)]
    for lam in (0.0, 0.3, 0.7, 1.0):
        assert mmr.select(cands, k=2, lam=lam)[0] is best


def test_relevance_is_normalised_so_lambda_is_meaningful():
    """Cross-encoder scores are logits (~-11..+11); similarity is 0..1. Without
    rescaling, the relevance term dominates and lambda does nothing."""
    cands = [cite("alpha beta gamma delta", 11.0), cite("alpha beta gamma epsilon", 10.5),
             cite("completely unrelated content here", -11.0)]
    # At heavy diversity weighting the near-duplicate must lose despite a large
    # raw-score advantage over the distinct passage.
    picked = mmr.select(cands, k=2, lam=0.2)
    assert picked[1].text == "completely unrelated content here"


def test_selection_reduces_measured_redundancy():
    """The end-to-end property, stated as the eval states it."""
    cands = [cite("alpha beta gamma delta epsilon", 5.0),
             cite("alpha beta gamma delta zeta", 4.9),
             cite("alpha beta gamma delta eta", 4.8),
             cite("entirely separate vocabulary appears here", 4.0)]
    baseline = mmr.select(cands, k=3, lam=1.0)      # pure relevance
    diverse = mmr.select(cands, k=3, lam=0.5)       # MMR
    assert mmr.mean_pairwise_overlap(diverse) < mmr.mean_pairwise_overlap(baseline)


@pytest.mark.parametrize("k", [0, 1, 5, 99])
def test_degenerate_inputs_do_not_raise(k):
    assert mmr.select([], k=k) == []
    one = [cite("only passage", 1.0)]
    assert len(mmr.select(one, k=k)) == min(k, 1)


def test_ties_in_relevance_do_not_crash():
    """Min-max normalisation divides by the spread, which is zero when every
    candidate scored the same."""
    cands = [cite("alpha", 1.0), cite("beta", 1.0), cite("gamma", 1.0)]
    assert len(mmr.select(cands, k=2, lam=0.5)) == 2
