"""Generation: turn retrieved passages into a grounded, cited answer.

This is the "G" in RAG. We do NOT ask the model what it knows — we hand it the
retrieved passages as numbered sources and instruct it to answer using ONLY
those. That constraint is the whole point: it turns a general LLM into a
system that can only speak from the papers, and cite where each claim came from.

WHICH model writes the words is a config choice, not a code one — see llm.py.
Locally that is Gemma 3 4B via Ollama (free, private); a deployed instance uses
a hosted model instead. The prompts, grounding contract, and citation format in
this file are identical either way, which is what makes the swap safe.
"""
from collections.abc import Iterator

from backend.config import settings
from backend.llm import LLMUnavailable, get_provider  # noqa: F401 (re-exported)
from backend.models import Answer, ChatTurn, Citation

# The system prompt encodes the grounding contract. Kept strict on purpose:
# use only the sources, cite by number, and refuse when the answer isn't there.
SYSTEM_PROMPT = """You are a precise research assistant. Answer the user's \
question USING ONLY the numbered sources provided below.

Rules:
- Use only information found in the sources. Do not use any outside knowledge.
- After each claim, cite the source number(s) in square brackets, e.g. [1] or [2][3].
- If the sources do not contain the answer, reply with exactly:
  "The provided papers do not contain the answer to this question."
- Be concise and precise. Do not invent citations."""


def active_model() -> str:
    """The model that will answer right now — recorded in the audit log.

    "Which model produced this answer" is the single most useful field for
    explaining why two answers to the same question differ, so it is read from
    the live provider rather than hard-coded.
    """
    return get_provider().MODEL


def active_temperature() -> float:
    """The sampling temperature actually in effect, for the audit log.

    Both providers pin this to 0 for reproducibility, which is what the audit
    log's "same question, same answer" promise rests on. Read from the provider
    rather than hard-coded so the log stays true if that ever stops holding.
    """
    return get_provider().TEMPERATURE


def _format_sources(citations: list[Citation]) -> str:
    """Render citations as the numbered [n] blocks the prompt refers to."""
    blocks = []
    for i, c in enumerate(citations, start=1):
        # page + 1 because pages are stored 0-indexed but humans count from 1.
        header = f"[{i}] (paper: {c.paper_id}, page {c.page + 1})"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)


def _user_message(question: str, citations: list[Citation]) -> str:
    """The user turn shared by the batch and streaming paths."""
    return f"Sources:\n\n{_format_sources(citations)}\n\nQuestion: {question}"


def generate(question: str, citations: list[Citation]) -> Answer:
    """Ask the configured LLM to answer `question` grounded in `citations`."""
    answer_text = get_provider().complete(
        system=SYSTEM_PROMPT,
        user=_user_message(question, citations),
        max_tokens=settings.llm_max_tokens,
    )
    return Answer(question=question, answer=answer_text, citations=citations)


def stream_answer(question: str, citations: list[Citation]) -> Iterator[str]:
    """Yield the grounded answer as incremental text deltas, as the LLM writes.

    Same prompt and grounding contract as generate(); only the transport differs
    (token-by-token instead of one blocking blob). The caller already holds the
    citations, so it can show sources immediately and stream the prose on top.
    """
    yield from get_provider().stream(
        system=SYSTEM_PROMPT,
        user=_user_message(question, citations),
        max_tokens=settings.llm_max_tokens,
    )


CONDENSE_PROMPT = """Given the conversation so far and a follow-up question, \
rewrite the follow-up as a standalone question that can be understood on its own, \
without the conversation. Resolve pronouns and references ("it", "that", "the \
second one") to what they refer to. Preserve the user's intent. Do NOT answer it. \
Return ONLY the rewritten question, nothing else."""

# Keep the condense prompt bounded: only the most recent turns matter for
# resolving a follow-up, and a short prompt keeps this extra call fast.
_CONDENSE_HISTORY_TURNS = 4

# A standalone question is one sentence; this caps the extra call's cost.
_CONDENSE_MAX_TOKENS = 80


def condense_question(question: str, history: list[ChatTurn]) -> str:
    """Rewrite a follow-up into a standalone question using recent history.

    Returns the original question unchanged if there is no history or the model
    returns nothing usable — so this can never make retrieval worse than before.
    """
    if not history:
        return question
    recent = history[-_CONDENSE_HISTORY_TURNS:]
    convo = "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in recent)
    rewritten = get_provider().complete(
        system=CONDENSE_PROMPT,
        user=f"Conversation:\n{convo}\n\nFollow-up: {question}\n\nStandalone question:",
        max_tokens=_CONDENSE_MAX_TOKENS,
    )
    return rewritten.strip() or question


def ping_llm() -> None:
    """Cheap liveness probe used by /health.

    Must stay fast: it confirms the provider answers and the configured model
    exists, without generating a token — a full generation would make /health as
    slow and expensive as /ask.
    """
    get_provider().ping()


def warm_llm() -> None:
    """Preload the model so the first real answer doesn't pay the cold start.

    A no-op on hosted providers, which have nothing local to load.
    """
    get_provider().warm()
