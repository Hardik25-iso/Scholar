from datetime import datetime

from pydantic import BaseModel, EmailStr, computed_field, field_validator


# What one entry in a document's page list actually IS, per format. Only PDF has
# real pages; labelling a slide or a worksheet "page 3" is a small lie, and this
# product's whole claim is that a citation can be trusted literally.
UNIT_PAGE = "page"
UNIT_SLIDE = "slide"
UNIT_SHEET = "sheet"
UNIT_SECTION = "section"


class Chunk(BaseModel):
    """A text chunk extracted from a document before embedding.

    `text` and `embed_text` describe the SAME span of the document:
      - text:       verbatim original-text slice — what we show as the citation.
      - embed_text: the exact string passed to the embedder — what produced the
                    vector. Kept as its own field so the embedded input is
                    auditable and the <=384-token guarantee is asserted on it.

    `char_start`/`char_end` locate that slice inside its page's extracted text.
    They are what turns "somewhere on page 12" into "these exact characters",
    which is the difference between a pointer and a citation.
    """
    paper_id: str        # stem of the source filename, e.g. "attention_is_all_you_need"
    page: int            # 0-indexed index into the document's unit list
    chunk_index: int     # position of this chunk within the paper
    text: str            # verbatim original text (display / citations)
    embed_text: str      # exact text fed to the embedder

    # Defaulted, not required: stores written before these existed must keep
    # loading. A 0/0 span reads as "unknown", never as "the start of the page".
    char_start: int = 0
    char_end: int = 0
    unit: str = UNIT_PAGE  # what `page` counts — see UNIT_* above


class IndexedChunk(Chunk):
    """A Chunk that has been embedded and stored in FAISS."""
    faiss_id: int        # row index in the FAISS index


class Citation(BaseModel):
    """A retrieved source passage that an answer is allowed to draw on.

    One Citation == one chunk the retriever returned for a question. The
    generator numbers these ([1], [2], ...) in the order they appear here, so
    the LLM's inline citations map straight back to a document and a location.
    """
    paper_id: str        # which paper the passage came from
    page: int            # 0-indexed unit in the source document (show page + 1)
    chunk_index: int     # position of the chunk within that paper
    score: float         # stage-1 similarity to the question
    text: str            # verbatim passage — what we display as the citation
    rerank_score: float | None = None  # stage-2 cross-encoder relevance (logit;
                                       # None until a reranker scores it)

    # Audit trail. faiss_id identifies the exact indexed vector this passage came
    # from, so a logged answer can be tied back to a specific row of a specific
    # index rather than to text that merely looks the same.
    faiss_id: int | None = None
    char_start: int = 0
    char_end: int = 0
    unit: str = UNIT_PAGE

    @computed_field
    @property
    def locator(self) -> str:
        """Human-readable location, honest about what the number counts.

        Serialised, so the UI does not have to reimplement the mapping and drift
        from it — "slide 3" and "sheet 2" are decided in exactly one place.
        """
        return f"{self.unit} {self.page + 1}"


class Answer(BaseModel):
    """A grounded answer plus the exact passages it was built from."""
    question: str
    answer: str                    # LLM text, with inline [n] citation markers
    citations: list[Citation]      # the numbered sources, in [1..n] order


class ChatTurn(BaseModel):
    """One prior exchange in a conversation, sent back with a follow-up so the
    question can be condensed into a standalone one before retrieval."""
    question: str
    answer: str


class AskRequest(BaseModel):
    """Body of a POST /ask call.

    candidates = how many chunks stage 1 shortlists (dense + lexical, fused);
    k          = how many survive reranking and reach the LLM (stage 2);
    history    = prior turns (oldest first). When non-empty, the question is a
                 follow-up and gets condensed into a standalone query first.
    papers     = restrict the search to these paper_ids. None (the default) means
                 the caller's whole library. This is the user narrowing their own
                 search, NOT an authorisation boundary — that is enforced by the
                 per-user index, which cannot be forgotten the way a filter can.
    """
    question: str
    k: int = 5
    candidates: int = 20
    history: list[ChatTurn] = []
    papers: list[str] | None = None


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


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    """A reset token plus the new password, held to the same rules as sign-up."""
    token: str
    password: str

    _check_password = field_validator("password")(RegisterRequest.password_rules.__func__)


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


# ——— Audit boundary models ———

class AnswerLogSummary(BaseModel):
    """One row of the audit list — enough to find an answer, not the whole thing."""
    id: int
    created_at: datetime
    question: str
    n_citations: int
    model: str
    reproducible: bool   # is the library still in the state this was drawn from?


class AnswerLogDetail(AnswerLogSummary):
    """A logged answer with its complete evidence chain."""
    query: str                  # what retrieval ran on (differs for a follow-up)
    answer: str
    citations: list[Citation]
    temperature: float
    k: int
    candidates: int
    papers_filter: list[str] | None
    index_fingerprint: str
    n_chunks_indexed: int
