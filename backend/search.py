"""Stage-1 retrieval: the hybrid shortlist that feeds the reranker.

One function, called by both the /ask route and the eval harness — which is the
point. When those two build their shortlist differently, the eval stops measuring
the product and starts measuring a lookalike, and every number it prints becomes
a guess about the real thing.

    dense  (embeddings)  -> what the passage MEANS
    sparse (FTS5 bm25)   -> what tokens the passage CONTAINS
    RRF                  -> one ranked list from the two
    expansion            -> a second pass in the answer's own vocabulary
    diversity cap        -> stop one document monopolising the shortlist

Returns the shortlist AND the query that produced it, because expansion changes
the query and the reranker must see the same one — see shortlist().
"""
from pathlib import Path

from backend import expansion, lexical
from backend.fusion import reciprocal_rank_fusion
from backend.models import Citation
from backend.retriever import Retriever

# How many of the shortlist any single document may occupy. Without a cap, a long
# document wins on sheer chunk count: ask "compare these three contracts" and the
# model can be handed five passages from one of them and none from the others,
# which makes the comparison impossible no matter how good the generator is.
# 60% leaves a single relevant document free to dominate when it deserves to,
# while guaranteeing room for at least one competitor.
DIVERSITY_RATIO = 0.6


def cap_per_document(citations: list[Citation], limit: int, ratio: float = DIVERSITY_RATIO) -> list[Citation]:
    """Trim so no document holds more than `ratio` of `limit`, preserving order.

    Over-quota passages are not discarded, only demoted — a document that really
    does hold all the answers still supplies them, just after every other
    document has had its chance at a slot.
    """
    max_per_doc = max(1, int(limit * ratio))
    kept: list[Citation] = []
    demoted: list[Citation] = []
    seen: dict[str, int] = {}

    for citation in citations:
        count = seen.get(citation.paper_id, 0)
        if count < max_per_doc:
            kept.append(citation)
            seen[citation.paper_id] = count + 1
        else:
            demoted.append(citation)

    return (kept + demoted)[:limit]


def _retrieve(
    query: str, store_dir: Path, retriever: Retriever, depth: int, dense_only: bool
) -> list[Citation]:
    """One retrieval pass: dense, plus lexical unless disabled, fused."""
    dense = retriever.retrieve(query, k=depth)
    if dense_only:
        return dense
    # Lexical hits know only (paper_id, chunk_index), and RRF keeps whichever
    # retriever ranked a passage best — so a lexically-found passage would
    # reach the caller missing its faiss_id and char span. Hydrate before
    # fusing, or the audit trail silently depends on which retriever won.
    sparse = retriever.hydrate(lexical.search(query, store_dir, k=depth))
    return reciprocal_rank_fusion([dense, sparse])


def shortlist(
    query: str,
    store_dir: str | Path,
    retriever: Retriever,
    k: int = 20,
    dense_only: bool = False,
    papers: list[str] | None = None,
    expansion_mode: str = "none",
) -> tuple[list[Citation], str]:
    """Return up to k candidate passages for `query`, best first.

    `papers` restricts the search to those paper_ids — the "ask only these
    documents" filter. It is applied AFTER retrieval rather than inside it,
    because both indexes are already scoped to one user; this is a user's own
    narrowing of their own library, not an authorisation boundary. Authorisation
    is the per-user store, which cannot be forgotten the way a filter can.

    `dense_only` reproduces the pre-hybrid behaviour, so the eval can A/B the
    lexical half against the same corpus and code path.
    """
    store_dir = Path(store_dir)

    # Over-fetch when filtering, or a filter that removes most results would
    # leave the reranker with far fewer passages than it was asked for.
    depth = k * 3 if papers else k

    candidates = _retrieve(query, store_dir, retriever, depth, dense_only)
    effective_query = query

    # Expansion runs a SECOND pass and fuses, so a bad expansion can only add
    # candidates, never remove a good one. Both modes are measured, and the
    # default is "none" because PRF was measured and rejected — see the roadmap.
    if expansion_mode == "prf":
        effective_query, terms = expansion.expand(query, candidates)
        expanded = bool(terms)
    elif expansion_mode == "hyde":
        effective_query = f"{query} {expansion.hypothetical_answer(query)}"
        expanded = True
    else:
        expanded = False

    if expanded:
        second = _retrieve(effective_query, store_dir, retriever, depth, dense_only)
        candidates = reciprocal_rank_fusion([candidates, second])

    if papers is not None:
        allowed = set(papers)
        candidates = [c for c in candidates if c.paper_id in allowed]

    # The expanded query is returned so the CALLER can rerank with it. That
    # matters more than it sounds: the measured failure was the cross-encoder
    # discarding a passage already retrieved at rank 2, because the question as
    # asked shared no vocabulary with it. Fixing stage 1 alone would not have
    # fixed that question.
    return cap_per_document(candidates, limit=k), effective_query
