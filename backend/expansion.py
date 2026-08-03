"""Query expansion for the vocabulary gap.

THE PROBLEM THIS SOLVES. Some questions share no rare term with the passage that
answers them. The agreement says *availability*; the user asks about *uptime*.
The paper says *Adam*; the user asks what *optimizer* was used. Neither retriever
can bridge that: lexical search only helps when the rare token is in the QUERY,
and the embedder puts "uptime" and "availability" near each other but not near
enough to beat twenty passages that discuss neither.

Measured before building this: those questions scored 0% while every other
category scored 100%.

WHY PSEUDO-RELEVANCE FEEDBACK AND NOT HyDE. The obvious alternative is to ask an
LLM to rewrite the question or hallucinate an answer to embed. It works, and it
is the wrong trade HERE, for a reason specific to this product: the audit log
promises that a logged answer can be reproduced against the same index. An LLM
in the retrieval path makes retrieval non-deterministic, so two runs of the same
question could retrieve different passages and the central claim quietly stops
being true. PRF is deterministic, adds no API call, no latency and no
dependency — and it is aimed squarely at vocabulary mismatch, which is the
measured failure.

HOW IT WORKS. Run retrieval once. Treat the top few passages as probably
relevant, and the rest of the shortlist as background. A term that is common in
the top passages and rare in the background is a term the answer's own
vocabulary uses — exactly what the question was missing. Add those terms and run
again.

The failure mode of PRF is query drift: if the first pass is wrong, expansion
confidently makes it wronger. Two guards, both measured rather than assumed:
terms must appear in at least two of the top passages (one passage cannot vote
itself into the query), and the original results are fused with the expanded
ones rather than replaced, so expansion can only add candidates.
"""
import re
from collections import Counter

from backend.models import Citation

# Passages treated as "probably relevant" for mining terms. Small on purpose:
# the wider the feedback set the more background vocabulary leaks in.
FEEDBACK_DEPTH = 4
# Terms added to the query. Enough to bridge a synonym gap, few enough that the
# original question still dominates the embedding.
MAX_TERMS = 6
# A term must appear in at least this many feedback passages. One passage cannot
# vote its own vocabulary into the query.
MIN_PASSAGES = 2

_TERM = re.compile(r"[a-z][a-z0-9\-']{2,}", re.IGNORECASE)

# Closed-class words carry no topic and would crowd out real signal. Kept as a
# literal set rather than a dependency — this list is stable, and adding nltk
# for it would be a large dependency for ~120 words.
STOPWORDS = frozenset("""
a about above after again against all also am an and any are aren't as at be because been
before being below between both but by can cannot could couldn't did didn't do does doesn't
doing don't down during each few for from further had hadn't has hasn't have haven't having
he her here hers herself him himself his how i if in into is isn't it its itself let's me
more most mustn't my myself no nor not of off on once only or other ought our ours ourselves
out over own same shan't she should shouldn't so some such than that the their theirs them
themselves then there these they this those through to too under until up very was wasn't we
were weren't what when where which while who whom why with won't would wouldn't you your
yours yourself yourselves shall may might must upon within without whether any each per
""".split())

# Words that describe the SHAPE of a question rather than its subject. They are
# frequent in retrieved prose and would be picked as "relevant" every time.
QUESTION_WORDS = frozenset("""
say says said state states stated mean means use used uses using make makes made
give gives given take takes taken apply applies applied happen happens
""".split())


def _terms(text: str) -> set[str]:
    return {
        t.lower() for t in _TERM.findall(text)
        if t.lower() not in STOPWORDS and t.lower() not in QUESTION_WORDS
    }


def expansion_terms(
    query: str,
    citations: list[Citation],
    depth: int = FEEDBACK_DEPTH,
    max_terms: int = MAX_TERMS,
) -> list[str]:
    """Terms the answer's own vocabulary uses that the question did not.

    Scored as (share of feedback passages containing the term) minus (share of
    background passages containing it). A term common everywhere in the
    shortlist describes the corpus, not the answer, and scores near zero.
    """
    if len(citations) <= depth:
        # Nothing to contrast against — every passage would be "relevant", so
        # every term would score 0 and expansion would be noise.
        return []

    asked = _terms(query)
    feedback = [_terms(c.text) for c in citations[:depth]]
    background = [_terms(c.text) for c in citations[depth:]]

    in_feedback = Counter(t for terms in feedback for t in terms)
    in_background = Counter(t for terms in background for t in terms)

    scored: list[tuple[float, str]] = []
    for term, count in in_feedback.items():
        if term in asked or count < MIN_PASSAGES:
            continue
        score = count / len(feedback) - in_background[term] / len(background)
        if score > 0:
            scored.append((score, term))

    scored.sort(key=lambda pair: (-pair[0], pair[1]))  # ties broken alphabetically
    return [term for _, term in scored[:max_terms]]


HYDE_PROMPT = """Write one short sentence that could plausibly appear in a \
document as the answer to the question. Use the formal vocabulary such a \
document would use, not the wording of the question. Do not hedge, do not \
explain, do not say you are unsure — invent specifics if you must. Reply with \
the sentence only."""


def hypothetical_answer(query: str) -> str:
    """Ask the LLM what the answer would LOOK like, and search for that instead.

    This is HyDE. It exists because pseudo-relevance feedback provably cannot fix
    vocabulary mismatch: PRF mines terms from the first pass, and the first pass
    is wrong precisely BECAUSE of the mismatch — measured, see the roadmap. An
    LLM has the outside knowledge PRF lacks: that a service level is stated as
    *availability*, that an optimiser is named *Adam*.

    THE COST IS REAL AND IT IS NOT LATENCY. It makes retrieval non-deterministic,
    and this product's audit log promises a logged answer can be reproduced
    against the same index. Anything shipping this must also record the
    hypothetical it used, or that promise quietly stops being true. Temperature
    is 0 to make that as close to reproducible as a local LLM allows.
    """
    import ollama

    from backend.generator import MODEL

    response = ollama.chat(
        model=MODEL,
        messages=[{"role": "system", "content": HYDE_PROMPT},
                  {"role": "user", "content": query}],
        options={"temperature": 0.0, "num_predict": 60},
    )
    return response["message"]["content"].strip()


def expand(query: str, citations: list[Citation], **kwargs) -> tuple[str, list[str]]:
    """Return (expanded query, terms added). Unchanged query if nothing qualifies.

    The original question is kept verbatim at the front: it is what the user
    actually asked, and both the embedder and the cross-encoder should still
    weight it most heavily. Expansion terms are appended, not substituted.
    """
    terms = expansion_terms(query, citations, **kwargs)
    if not terms:
        return query, []
    return f"{query} {' '.join(terms)}", terms
