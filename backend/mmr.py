"""Maximal Marginal Relevance — pick k passages that are relevant AND distinct.

THE PROBLEM. The chunker slides a 380-token window with 50 tokens of overlap, so
adjacent chunks share text by construction. When a question is answered in that
shared region, both chunks score well and both land in the top k — spending two
of five slots on nearly the same sentences. The reranker cannot fix this: it
scores each passage against the QUESTION, never against the passages already
chosen, so it has no way to know it is repeating itself.

THE FIX. Choose passages one at a time. At each step take the candidate with the
best combination of "relevant to the question" and "unlike everything picked so
far" (Carbonell & Goldstein, 1998):

    score(c) = λ·relevance(c) − (1−λ)·max_similarity(c, already_selected)

λ=1 is the current behaviour (pure relevance). Lower λ buys diversity with
relevance.

WHY TOKEN OVERLAP AND NOT EMBEDDING COSINE. Textbook MMR measures similarity in
embedding space. That is the wrong instrument here, and the distinction is the
whole point of the feature:

  - Two passages sharing literal text (the overlap artefact) are WASTE. One of
    them adds nothing.
  - Two DIFFERENT passages that independently support the same answer are
    VALUABLE — that is corroboration, and a cited answer is stronger for it.

Cosine similarity scores both cases high and would suppress the second along
with the first. Jaccard overlap on tokens scores only the first, because only
the first actually repeats words. It is also free: no model call, no vector
reconstruction, on a list of 20 candidates.
"""
import math
import re

from backend.models import Citation

# Word characters only, lowercased: "Section 7.2" and "section 7.2" are the same
# text for redundancy purposes, and punctuation differences are not evidence of
# distinct content.
_WORD = re.compile(r"\w+")


def _tokens(text: str) -> set[str]:
    return set(_WORD.findall(text.lower()))


def jaccard(a: set[str], b: set[str]) -> float:
    """Overlap between two token sets, 0.0 (disjoint) to 1.0 (identical)."""
    if not a or not b:
        return 0.0
    intersection = len(a & b)
    if not intersection:
        return 0.0
    return intersection / len(a | b)


def mean_pairwise_overlap(citations: list[Citation]) -> float:
    """Average Jaccard across every pair — how repetitive a result set is.

    Reported by the eval because it is the metric that shows MMR working. If
    hit@k is unchanged but this falls, the same evidence is being delivered in
    fewer duplicated words, which is exactly the intended trade.
    """
    if len(citations) < 2:
        return 0.0
    toks = [_tokens(c.text) for c in citations]
    pairs = [
        jaccard(toks[i], toks[j])
        for i in range(len(toks))
        for j in range(i + 1, len(toks))
    ]
    return sum(pairs) / len(pairs)


def _relevance(citations: list[Citation]) -> dict[int, float]:
    """Map each candidate to a relevance in [0, 1].

    Cross-encoder scores are logits (roughly −11..+11) while similarity is a
    proportion in [0, 1]. Subtracting one from the other without rescaling would
    let relevance dominate entirely and make λ meaningless.

    SIGMOID, NOT MIN-MAX. An ms-marco cross-encoder emits a logit whose sigmoid
    is the calibrated probability that the passage is relevant — so this is what
    the score already means, not a rescaling invented for this function. Min-max
    was tried first and is wrong here for a concrete reason: it is relative to
    the candidate set, so the WORST candidate is always pinned to exactly 0 no
    matter how good it is, and the same passage gets a different relevance
    depending on what else happened to be retrieved. λ would then mean something
    different on every query. Sigmoid is absolute and stable across queries.

    A side effect worth knowing: sigmoid compresses the top end (logits 4 and 5
    both map to ~0.99), so among passages the reranker considers similarly good,
    the diversity term decides — which is exactly the intent.
    """
    return {
        id(c): 1.0 / (1.0 + math.exp(-(c.rerank_score if c.rerank_score is not None else 0.0)))
        for c in citations
    }


def select(citations: list[Citation], k: int, lam: float = 0.7) -> list[Citation]:
    """Choose k passages by Maximal Marginal Relevance, best first.

    `citations` must already carry `rerank_score` — MMR re-orders the
    cross-encoder's judgement, it does not replace it.

    λ=1.0 reduces exactly to "take the top k by relevance", which is the
    pre-MMR behaviour and the A/B baseline.
    """
    if k <= 0 or not citations:
        return []
    if len(citations) <= k:
        return list(citations)

    relevance = _relevance(citations)
    tokens = {id(c): _tokens(c.text) for c in citations}

    remaining = list(citations)
    selected: list[Citation] = []

    while remaining and len(selected) < k:
        best, best_score = None, float("-inf")
        for candidate in remaining:
            # First pick has nothing to be redundant against, so this reduces to
            # pure relevance — MMR always starts from the most relevant passage.
            redundancy = max(
                (jaccard(tokens[id(candidate)], tokens[id(s)]) for s in selected),
                default=0.0,
            )
            score = lam * relevance[id(candidate)] - (1.0 - lam) * redundancy
            if score > best_score:
                best, best_score = candidate, score
        selected.append(best)
        remaining.remove(best)

    return selected
