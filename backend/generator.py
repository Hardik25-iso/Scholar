"""Generation: turn retrieved passages into a grounded, cited answer.

This is the "G" in RAG. We do NOT ask the model what it knows — we hand it the
retrieved passages as numbered sources and instruct it to answer using ONLY
those. That constraint is the whole point: it turns a general LLM into a
system that can only speak from the papers, and cite where each claim came from.

The LLM is Gemma 3 4B, running locally via Ollama (no API cost, on-device).
Swapping models later means changing MODEL below — nothing else in the app.
"""
from collections.abc import Iterator

import ollama

from backend.models import Answer, Citation

MODEL = "gemma3:4b"

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
    response = ollama.chat(
        model=MODEL,
        messages=_messages(question, citations),
        # temperature 0 => deterministic, faithful answers (no creative drift).
        options={"temperature": 0.0},
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


def stream_answer(question: str, citations: list[Citation]) -> Iterator[str]:
    """Yield the grounded answer as incremental text deltas, as the LLM writes.

    Same prompt and grounding contract as generate(); only the transport differs
    (token-by-token instead of one blocking blob). The caller already holds the
    citations, so it can show sources immediately and stream the prose on top.
    """
    stream = ollama.chat(
        model=MODEL,
        messages=_messages(question, citations),
        options={"temperature": 0.0},
        stream=True,
    )
    for part in stream:
        delta = part["message"]["content"]
        if delta:
            yield delta
