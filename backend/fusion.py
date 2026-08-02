"""Reciprocal Rank Fusion — merge the dense and lexical result lists into one.

THE PROBLEM. Two retrievers, two incompatible score scales: cosine similarity is
roughly 0..1, bm25 is an unbounded log-odds-ish number whose magnitude depends on
corpus statistics. Adding or averaging them means inventing a conversion factor
that has no meaning and that silently shifts as the corpus grows.

THE FIX. Throw the scores away and keep only the RANKS, which are comparable by
construction. A document's fused score is the sum over retrievers of
1 / (K + rank). That is Reciprocal Rank Fusion (Cormack et al., 2009).

Two properties earn it its place here:
  - Being found by BOTH retrievers is rewarded without tuning. A chunk that
    appears in both lists roughly doubles its score, so it beats a chunk ranked
    first by one retriever and missed entirely by the other — exactly the
    behaviour wanted when one retriever is good at meaning and the other at
    exact tokens.
  - The 1/(K+rank) curve is steep at the top and flat in the tail, so the
    difference between rank 1 and 2 matters and the difference between 40 and 41
    does not.

Be precise about the first property: it is "found by both", NOT "ranked evenly
by both". Because 1/x is convex, 1/61 + 1/63 > 2/62 — a chunk ranked 1st and 3rd
narrowly BEATS one ranked 2nd and 2nd. RRF prefers a strong opinion somewhere to
mediocrity everywhere. That is a reasonable behaviour to have, but it is not the
"consensus wins" story RRF is often described with, and a test written on that
assumption fails.

K=60 is the constant from the original paper and the de-facto default. It damps
the head of the curve so a single retriever's top hit cannot dominate on its own.
"""
from backend.models import Citation

RRF_K = 60


def _key(citation: Citation) -> tuple[str, int]:
    """Identity of a chunk across both retrievers.

    (paper_id, chunk_index) rather than the text itself: it is the stable id the
    pipeline already carries, and comparing multi-hundred-character strings to
    deduplicate would be both slower and fragile against whitespace.
    """
    return citation.paper_id, citation.chunk_index


def reciprocal_rank_fusion(
    ranked_lists: list[list[Citation]], k: int = RRF_K, limit: int | None = None
) -> list[Citation]:
    """Fuse ranked lists into one, best first.

    The returned Citation is the instance from whichever list ranked it best, so
    it keeps a real `score` from a real retriever — `score` is documented as the
    stage-1 similarity and would become a lie if a fused number were written into
    it. The fusion result is expressed purely by the ORDER of the list, which is
    all the reranker downstream consumes.
    """
    best: dict[tuple[str, int], Citation] = {}
    best_rank: dict[tuple[str, int], int] = {}
    fused: dict[tuple[str, int], float] = {}

    for ranked in ranked_lists:
        for rank, citation in enumerate(ranked, start=1):
            key = _key(citation)
            fused[key] = fused.get(key, 0.0) + 1 / (k + rank)
            if rank < best_rank.get(key, 10**9):
                best_rank[key] = rank
                best[key] = citation

    order = sorted(fused, key=lambda key: fused[key], reverse=True)
    merged = [best[key] for key in order]
    return merged[:limit] if limit else merged
