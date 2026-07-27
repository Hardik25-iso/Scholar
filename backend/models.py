from datetime import datetime

from pydantic import BaseModel, EmailStr, field_validator


class Chunk(BaseModel):
    """A text chunk extracted from a PDF before embedding.

    `text` and `embed_text` describe the SAME span of the paper:
      - text:       verbatim original-text slice — what we show as the citation.
      - embed_text: the exact string passed to the embedder — what produced the
                    vector. Kept as its own field so the embedded input is
                    auditable and the <=384-token guarantee is asserted on it.
    """
    paper_id: str        # stem of the source filename, e.g. "attention_is_all_you_need"
    page: int            # 0-indexed page number in the source PDF
    chunk_index: int     # position of this chunk within the paper
    text: str            # verbatim original text (display / citations)
    embed_text: str      # exact text fed to the embedder


class IndexedChunk(Chunk):
    """A Chunk that has been embedded and stored in FAISS."""
    faiss_id: int        # row index in the FAISS index


class Citation(BaseModel):
    """A retrieved source passage that an answer is allowed to draw on.

    One Citation == one chunk the retriever returned for a question. The
    generator numbers these ([1], [2], ...) in the order they appear here, so
    the LLM's inline citations map straight back to a paper + page.
    """
    paper_id: str        # which paper the passage came from
    page: int            # 0-indexed page in the source PDF (show page + 1 to humans)
    chunk_index: int     # position of the chunk within that paper
    score: float         # stage-1 cosine similarity to the question (0..1)
    text: str            # verbatim passage — what we display as the citation
    rerank_score: float | None = None  # stage-2 cross-encoder relevance (logit;
                                       # None until a reranker scores it)


class Answer(BaseModel):
    """A grounded answer plus the exact passages it was built from."""
    question: str
    answer: str                    # LLM text, with inline [n] citation markers
    citations: list[Citation]      # the numbered sources, in [1..n] order


class AskRequest(BaseModel):
    """Body of a POST /ask call.

    candidates = how many chunks the bi-encoder retrieves (stage 1);
    k          = how many survive reranking and reach the LLM (stage 2).
    """
    question: str
    k: int = 5
    candidates: int = 20


# ——— Auth boundary models ———

class RegisterRequest(BaseModel):
    """Sign-up payload. EmailStr validates format; password rules enforced below."""
    email: EmailStr
    password: str

    @field_validator("password")
    @classmethod
    def password_rules(cls, v: str) -> str:
        # Min length 8 (NIST SP 800-63B) — no forced composition rules.
        if len(v) < 8:
            raise ValueError("password must be at least 8 characters")
        # bcrypt truncates silently past 72 BYTES (not chars) — reject loudly so
        # long/multibyte passwords can never be quietly cut. Same failure
        # philosophy as the chunker's 384-token guard.
        if len(v.encode("utf-8")) > 72:
            raise ValueError("password must be at most 72 bytes")
        return v


class LoginRequest(BaseModel):
    email: EmailStr
    password: str


class UserPublic(BaseModel):
    """A user as exposed over the wire — never includes the password hash."""
    id: int
    email: EmailStr


# ——— Library boundary models ———

class PaperPublic(BaseModel):
    """A paper as exposed over the wire (the library catalogue entry)."""
    id: int
    paper_id: str
    title: str
    filename: str
    n_chunks: int
    created_at: datetime
