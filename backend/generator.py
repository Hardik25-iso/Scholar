"""Generation: turn retrieved passages into a grounded, cited answer.

This is the "G" in RAG. We do NOT ask the model what it knows — we hand it the
retrieved passages as numbered sources and instruct it to answer using ONLY
those. That constraint is the whole point: it turns a general LLM into a
system that can only speak from the papers, and cite where each claim came from.

The LLM is Gemma 3 4B, running locally via Ollama (no API cost, on-device).
Swapping models later means changing MODEL below — nothing else in the app.
"""
from collections.abc import Iterator

import httpx
import ollama

from backend.config import settings
from backend.models import Answer, ChatTurn, Citation

MODEL = "gemma3:4b"


class LLMUnavailable(RuntimeError):
    """The model did not answer in time, or Ollama is not reachable.

    Raised instead of letting httpx's error surface, so callers (the /ask routes)
    can turn it into a clean 503 rather than a 500 with a stack trace.
    """


# One shared client with an explicit timeout. The module-level `ollama.chat`
# helper has NO timeout: if the model hangs, the call never returns and the
# threadpool worker serving that request is pinned forever. Enough of those and
# the app stops answering entirely while still reporting itself healthy.
_client = ollama.Client(timeout=settings.llm_timeout_seconds)


def _chat(**kwargs):
    """ollama.chat with a timeout, and connection/timeout errors normalised."""
    try:
        return _client.chat(**kwargs)
    except (httpx.TimeoutException, httpx.ConnectError, ConnectionError) as exc:
        raise LLMUnavailable(
            f"the local model did not respond within "
            f"{settings.llm_timeout_seconds:.0f}s (is Ollama running?)"
        ) from exc

# temperature 0 => deterministic, faithful answers (no creative drift). Named
# rather than inlined because the audit log records it: "which model, at what
# temperature" is part of explaining why two answers differed.
TEMPERATURE = 0.0

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


def _format_sources(citations: list[Citation]) -> str:
    """Render citations as the numbered [n] blocks the prompt refers to."""
    blocks = []
    for i, c in enumerate(citations, start=1):
        # page + 1 because pages are stored 0-indexed but humans count from 1.
        header = f"[{i}] (paper: {c.paper_id}, page {c.page + 1})"
        blocks.append(f"{header}\n{c.text}")
    return "\n\n".join(blocks)


def generate(question: str, citations: list[Citation]) -> Answer:
    """Ask the local LLM to answer `question` grounded in `citations`."""
    response = _chat(
        model=MODEL,
        messages=_messages(question, citations),
        options={"temperature": TEMPERATURE},
    )

    answer_text = response["message"]["content"].strip()
    return Answer(question=question, answer=answer_text, citations=citations)


def _messages(question: str, citations: list[Citation]) -> list[dict[str, str]]:
    """The system + user turns shared by the batch and streaming paths."""
    user_message = f"Sources:\n\n{_format_sources(citations)}\n\nQuestion: {question}"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]


CONDENSE_PROMPT = """Given the conversation so far and a follow-up question, \
rewrite the follow-up as a standalone question that can be understood on its own, \
without the conversation. Resolve pronouns and references ("it", "that", "the \
second one") to what they refer to. Preserve the user's intent. Do NOT answer it. \
Return ONLY the rewritten question, nothing else."""

# Keep the condense prompt bounded: only the most recent turns matter for
# resolving a follow-up, and a short prompt keeps this extra call fast.
_CONDENSE_HISTORY_TURNS = 4


def condense_question(question: str, history: list[ChatTurn]) -> str:
    """Rewrite a follow-up into a standalone question using recent history.

    Returns the original question unchanged if there is no history or the model
    returns nothing usable — so this can never make retrieval worse than before.
    """
    if not history:
        return question
    recent = history[-_CONDENSE_HISTORY_TURNS:]
    convo = "\n".join(f"Q: {t.question}\nA: {t.answer}" for t in recent)
    response = _chat(
        model=MODEL,
        messages=[
            {"role": "system", "content": CONDENSE_PROMPT},
            {"role": "user",
             "content": f"Conversation:\n{convo}\n\nFollow-up: {question}\n\nStandalone question:"},
        ],
        options={"temperature": 0.0, "num_predict": 80},
    )
    return response["message"]["content"].strip() or question


def ping_llm() -> None:
    """Cheap liveness probe for the LLM: does the Ollama daemon answer, and is
    our model present? Used by /health, so it must stay fast — listing models
    costs nothing, whereas generating a token would make /health as slow as /ask.
    """
    try:
        result = _client.list()
    except (httpx.TimeoutException, httpx.ConnectError, ConnectionError) as exc:
        raise LLMUnavailable("Ollama is not reachable") from exc
    names = {m.get("model") or m.get("name") for m in result.get("models", [])}
    if MODEL not in names:
        raise LLMUnavailable(f"model {MODEL} is not pulled (run: ollama pull {MODEL})")


def warm_llm() -> None:
    """Trigger Ollama to load the model into memory now, so the first real
    answer doesn't pay the ~30s+ cold-load. A 1-token generation is enough."""
    _chat(
        model=MODEL,
        messages=[{"role": "user", "content": "hi"}],
        options={"num_predict": 1},
    )


def stream_answer(question: str, citations: list[Citation]) -> Iterator[str]:
    """Yield the grounded answer as incremental text deltas, as the LLM writes.

    Same prompt and grounding contract as generate(); only the transport differs
    (token-by-token instead of one blocking blob). The caller already holds the
    citations, so it can show sources immediately and stream the prose on top.
    """
    stream = _chat(
        model=MODEL,
        messages=_messages(question, citations),
        options={"temperature": TEMPERATURE},
        stream=True,
    )
    # The timeout also covers the gaps BETWEEN chunks, so a model that stalls
    # mid-answer raises here rather than leaving the stream open indefinitely.
    try:
        for part in stream:
            delta = part["message"]["content"]
            if delta:
                yield delta
    except (httpx.TimeoutException, httpx.ConnectError, ConnectionError) as exc:
        raise LLMUnavailable("the local model stalled mid-answer") from exc
