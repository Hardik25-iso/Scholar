"""Lexical (keyword) search over a user's chunks, via SQLite FTS5.

WHY THIS EXISTS. Dense retrieval matches meaning, which is exactly wrong for the
queries this product's users actually type. "Section 7.2", "Force Majeure", a
party name, a date, an invoice number — these carry almost no semantic signal;
their value is that they are *those exact tokens*. An embedding of "Section 7.2"
sits near every other section reference in the document.

WHY FTS5 AND NOT `rank_bm25`. SQLite ships FTS5 with a native `bm25()` ranking
function in the stdlib build, so this costs zero new dependencies and the index
lives on disk next to everything else rather than being rebuilt in memory per
process.

STORAGE. One FTS5 database per user, beside their FAISS index, so a lexical
search is scoped to that user's library by construction — the same isolation
property the per-user vector index already has, rather than a WHERE clause that
someone can forget.
"""
import re
import sqlite3
from pathlib import Path

from backend.models import Chunk, Citation

DB_NAME = "lexical.db"

# FTS5's own query language treats a pile of characters as operators. User input
# is never a query expression here — it is a bag of terms to match — so every
# term is wrapped in double quotes, which makes it a literal.
#
# The inner `[.\-/]` is what makes "Section 7.2" work. FTS5's tokenizer splits on
# punctuation, so a term must keep its internal dots to stay one unit: quoted,
# "7.2" becomes the PHRASE [7, 2] and matches only 7.2, while splitting it first
# would produce `"7" OR "2"`, which also matches 7.1 and 2.7 and every other
# clause containing a 7. That distinction is the entire point of lexical search
# here, and losing it makes 7.1 and 7.2 tie.
_TERM = re.compile(r"\w+(?:[.\-/]\w+)*", re.UNICODE)


def _connect(store_dir: Path) -> sqlite3.Connection:
    store_dir.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(store_dir / DB_NAME)
    conn.execute(
        # `chunks` is the searchable text; the rest are UNINDEXED columns riding
        # along so a hit can be turned into a Citation without a second table.
        "CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5("
        "  text, paper_id UNINDEXED, page UNINDEXED, chunk_index UNINDEXED"
        ")"
    )
    return conn


def to_match_query(question: str) -> str:
    """Turn free text into a safe FTS5 MATCH expression.

    Each term is double-quoted, which makes it a literal rather than a fragment
    of FTS5 syntax: `Section 7.2` matches the phrase instead of raising
    `fts5: syntax error near "."`, bare AND/OR/NOT in a question are matched as
    words rather than parsed as operators, and a stray parenthesis or asterisk
    cannot change the shape of the query.

    Terms are OR'd rather than AND'd because a natural-language question carries
    filler ("what does ... say?") that no passage contains; requiring every term
    would match nothing. bm25 then does the real work, weighting a rare term
    like `7.2` far above a common one like `what`.
    """
    return " OR ".join(f'"{term}"' for term in _TERM.findall(question))


def index_chunks(chunks: list[Chunk], store_dir: str | Path) -> None:
    """Mirror chunks into the user's FTS5 index. Appends, like the vector store."""
    if not chunks:
        return
    conn = _connect(Path(store_dir))
    try:
        conn.executemany(
            "INSERT INTO chunks (text, paper_id, page, chunk_index) VALUES (?, ?, ?, ?)",
            [(c.text, c.paper_id, c.page, c.chunk_index) for c in chunks],
        )
        conn.commit()
    finally:
        conn.close()


def remove_paper(store_dir: str | Path, paper_id: str) -> None:
    """Drop one paper's rows. Unlike FAISS, FTS5 deletes in place — no rebuild."""
    path = Path(store_dir) / DB_NAME
    if not path.exists():
        return
    conn = sqlite3.connect(path)
    try:
        conn.execute("DELETE FROM chunks WHERE paper_id = ?", (paper_id,))
        conn.commit()
    finally:
        conn.close()


def search(question: str, store_dir: str | Path, k: int = 20) -> list[Citation]:
    """Return the k best lexical matches, best first.

    `score` carries the negated bm25 value. SQLite's bm25() returns a number
    that is more negative the better the match; negating it makes "higher is
    better" true here as it is for the cosine scores, so the fusion step
    downstream does not need to know which retriever a result came from.
    """
    path = Path(store_dir) / DB_NAME
    match = to_match_query(question)
    if not path.exists() or not match:
        return []

    conn = sqlite3.connect(path)
    try:
        rows = conn.execute(
            "SELECT text, paper_id, page, chunk_index, bm25(chunks) AS score "
            "FROM chunks WHERE chunks MATCH ? ORDER BY score LIMIT ?",
            (match, k),
        ).fetchall()
    except sqlite3.OperationalError:
        # A malformed MATCH should degrade to "no lexical results", leaving the
        # dense side to answer, rather than failing the user's question.
        return []
    finally:
        conn.close()

    return [
        Citation(text=text, paper_id=paper_id, page=page, chunk_index=chunk_index, score=-score)
        for text, paper_id, page, chunk_index, score in rows
    ]
