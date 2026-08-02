"""Retrieval: embed a question and pull the most similar chunks from FAISS.

This is the "R" in RAG. The question is embedded with the SAME model that
embedded the chunks at index time (all-mpnet-base-v2) — both land in the same
768-dim space, so "nearest vector" means "closest in meaning".

Because the stored vectors are L2-normalised and the index is IndexFlatIP,
each returned score is a cosine similarity in [-1, 1] (in practice ~0..1).
"""
from pathlib import Path

from backend.embedder import embed
from backend.models import Citation
from backend.store import load

DEFAULT_STORE = Path(__file__).parent.parent / "data" / "index"


class Retriever:
    """Loads a FAISS store once and answers top-k similarity queries against it."""

    def __init__(self, store_dir: str | Path = DEFAULT_STORE) -> None:
        # Load the index + chunk metadata a single time; reused for every query.
        self.index, self.chunks = load(store_dir)
        # Chunk metadata addressed the way every other component identifies a
        # passage. chunks.json is the single source of truth for the audit
        # fields, so anything that finds a passage some other way (lexical
        # search) can be completed from here rather than storing its own copy.
        self._by_location = {(c.paper_id, c.chunk_index): c for c in self.chunks}

    def hydrate(self, citations: list[Citation]) -> list[Citation]:
        """Fill in audit fields on citations that came from another retriever.

        Lexical search identifies a passage by (paper_id, chunk_index) and knows
        nothing about faiss_id, char offsets or the document's unit. Copying
        those into the FTS5 table would duplicate them into a second store that
        can silently drift from chunks.json; looking them up keeps one owner.
        """
        for citation in citations:
            if citation.faiss_id is not None:
                continue
            chunk = self._by_location.get((citation.paper_id, citation.chunk_index))
            if chunk is None:
                continue
            citation.faiss_id = chunk.faiss_id
            citation.char_start = chunk.char_start
            citation.char_end = chunk.char_end
            citation.unit = chunk.unit
        return citations

    def retrieve(self, question: str, k: int = 5) -> list[Citation]:
        """Return the k chunks most similar to `question`, best first."""
        # (1, 768) query vector; progress bar off for a single sentence.
        q_vec = embed([question], progress=False)

        # FAISS returns scores and row-ids for the k nearest vectors.
        scores, ids = self.index.search(q_vec, k)

        citations: list[Citation] = []
        for score, faiss_id in zip(scores[0], ids[0]):
            if faiss_id == -1:        # FAISS pads with -1 if fewer than k exist
                continue
            chunk = self.chunks[faiss_id]
            citations.append(
                Citation(
                    paper_id=chunk.paper_id,
                    page=chunk.page,
                    chunk_index=chunk.chunk_index,
                    score=float(score),
                    text=chunk.text,
                    # Carried through, not dropped: faiss_id ties a citation to
                    # the exact indexed vector, and the char span turns "page 12"
                    # into the precise slice. Both are what the audit log needs
                    # to make an answer reproducible rather than merely plausible.
                    faiss_id=chunk.faiss_id,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    unit=chunk.unit,
                )
            )
        return citations
