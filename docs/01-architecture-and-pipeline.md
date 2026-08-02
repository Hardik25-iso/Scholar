# Scholar — Architecture & Pipeline

> How the whole system fits together: the stack, the two phases (indexing and
> querying), and the exact path a request takes from a typed question to a
> streamed, cited answer.

Scholar is a **RAG** (Retrieval-Augmented Generation) research assistant — the
same core idea as NotebookLM. You upload PDFs; you ask questions; you get answers
that are built **only** from passages retrieved out of *your* papers, with each
claim cited back to the exact page. It is deliberately *not* a chatbot that
answers from the model's memory — it is a **retrieval system with a chat surface**.

---

## 1. The big picture

There are two phases. They are completely separate in time and in code.

```
┌──────────────────────── PHASE 1: INDEXING (when you upload a PDF) ───────────────────────┐
│                                                                                          │
│   PDF ──▶ parse (PyMuPDF) ──▶ chunk (token-accurate) ──▶ embed (mpnet) ──▶ FAISS index   │
│          one string/page      ~380-token windows        768-dim vectors    (per user)    │
│                                                                                          │
└──────────────────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────── PHASE 2: QUERYING (every question you ask) ──────────────────────┐
│                                                                                          │
│   question ──▶ (condense follow-up) ──▶ embed ──▶ FAISS top-20 ──▶ rerank top-5 ──▶ LLM   │
│                 resolve "it"/"that"     768-dim    bi-encoder      cross-encoder    Gemma │
│                                          vector     recall          precision      (local)│
│                                                                             │            │
│                                                     answer + citations ◀────┘            │
│                                                     (streamed token by token)            │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

**Why split it this way?** Indexing is expensive (parsing + embedding a whole
paper) and only has to happen **once per document**. Querying must be fast and
happens on **every question**. Precomputing the vectors at indexing time is what
makes retrieval at query time near-instant.

---

## 2. The stack — what each piece does and why

| Layer | Tool | Job | Why this one |
|-------|------|-----|--------------|
| API | **FastAPI** (Python, port 8001) | HTTP endpoints, request validation | async-friendly, Pydantic-native, auto docs |
| Typing | **Pydantic v2** | validate every API request/response | one schema = validation + serialization |
| PDF parsing | **PyMuPDF (`fitz`)** | extract text per page | fast, keeps page numbers (needed for citations) |
| Chunking | **HuggingFace tokenizer** | split text into model-sized windows | counts *real* word-piece tokens, not words |
| Embeddings | **sentence-transformers `all-mpnet-base-v2`** | text → 768-dim vector | strong general-purpose semantic encoder |
| Vector store | **FAISS `IndexFlatIP`** | nearest-neighbour search | exact cosine search, no server, fast to ~100k chunks |
| Reranking | **cross-encoder `ms-marco-MiniLM-L-6-v2`** | re-score top candidates | precision the bi-encoder can't reach |
| Generation | **Ollama running `gemma3:4b`** | write the grounded answer | free, local, on-device (no API cost) |
| Persistence | **SQLite + SQLModel** | users, paper metadata | one file, zero server; SQLModel = tables that are Pydantic models |
| Auth | **bcrypt + JWT (httpOnly cookie) + CSRF** | accounts & sessions | standard, secure, cross-origin-safe |
| Frontend | **React + Vite + TypeScript + Tailwind + react-router** | the workspace UI | fast dev, typed, component-based |

The embedder, reranker, and LLM are all **pretrained** and used off the shelf —
nothing is trained or fine-tuned. That is the whole point of RAG (see doc 02).

---

## 3. Phase 1 — the indexing pipeline (module by module)

Triggered by `POST /papers` (an upload). Lives mostly in `backend/library.py`,
which chains the same four modules the original CLI used.

### 3.1 `parser.py` — PDF → text
```python
extract_pages(pdf_path) -> list[str]   # one string per page, 0-indexed
```
Uses PyMuPDF to pull the text layer out of each page. Returning **per page** (not
one giant string) is deliberate: the page index travels with the text all the way
to the citation, so an answer can say "p.5".

### 3.2 `chunker.py` — text → model-sized windows
An LLM/embedder can only read a bounded number of tokens at once. `all-mpnet-base-v2`
truncates anything past **384 word-piece tokens**. So we cut each page into
overlapping windows:
- `CHUNK_TOKENS = 380` content tokens (+2 special tokens = 382 ≤ 384 — fits)
- `OVERLAP = 50` tokens (~13%) so an answer isn't split across a chunk boundary

The clever bit: we tokenize with the **fast tokenizer's offset mapping**, which
tells us the *character span* each token covers. We slice the **verbatim original
text** out of those offsets — so the stored chunk keeps real casing/spacing
("Ashish Vaswani", "avaswani@google.com") instead of the lowercased mush a
decode round-trip would give. That verbatim text is what you later see as a
citation. (Detail in doc 02 §3.)

Output: a list of `Chunk` objects (`paper_id`, `page`, `chunk_index`, `text`,
`embed_text`).

### 3.3 `embedder.py` — text → vectors
```python
embed(texts) -> np.ndarray  # shape (N, 768), float32, L2-normalised
```
Runs each chunk through mpnet to get a **768-dimensional vector** that captures
its meaning. Vectors are **L2-normalised** so that an inner product equals cosine
similarity (doc 02 §4).

### 3.4 `store.py` + per-user layout — vectors → FAISS
`append_to_store()` adds the new vectors to the user's FAISS index (creating it on
first upload) and appends the chunk metadata to `chunks.json`. Each chunk's
`faiss_id` is the exact row it occupies in the index, so the vector and its
metadata stay in lockstep.

Per-user disk layout (`backend/library.py`):
```
data/users/<user_id>/
   papers/<paper_id>.pdf     # the original upload (for the in-app viewer)
   index/index.faiss         # this user's FAISS index — ALL their papers
   index/chunks.json         # chunk metadata, each tagged with its paper_id
```
**One index per user** (not one shared index filtered by user) means data never
mixes between accounts, and "delete my paper" is a clean local rebuild.

### 3.5 Upload validation & failure handling (`papers.py`)
Before any expensive work: reject non-PDFs (415), files > 20 MB (413), empty
files (400). After indexing, if a PDF yielded **zero** chunks (e.g. a scanned
image with no text layer), the saved file is rolled back and a 422 is returned.
Indexing is **synchronous** — the HTTP request embeds the paper before it returns
(a few to ~30 seconds), which is why the UI shows an "indexing…" progress state.

---

## 4. Phase 2 — the querying pipeline (a request's life)

Triggered by `POST /ask/stream` (streaming) or `POST /ask` (blocking). The core
lives in `backend/api.py::_resolve_and_retrieve` + `backend/generator.py`.

Step by step, for the question **"What about the decoder?"** asked *after* a
question about attention:

**① Auth & scoping.** `get_current_user` reads the JWT from the httpOnly cookie
and resolves your account. `library.get_retriever(user.id)` loads (once, then
cached) *your* FAISS index. If you have no papers → 400 "upload a PDF first". You
can only ever query your own library.

**② Condense the follow-up** (`generator.condense_question`). "What about the
decoder?" is meaningless to a search engine on its own. Using recent chat history,
the LLM rewrites it into a standalone query like *"How does the decoder work in
the Transformer?"*. With no history this is a no-op, so it can never make things
worse. (Doc 02 §9.)

**③ Embed the query.** The standalone question is embedded with the **same mpnet
model** used at indexing time — both live in the same 768-dim space, so "nearest
vector" means "closest in meaning".

**④ Stage-1 retrieval — bi-encoder recall** (`retriever.py`). FAISS compares the
query vector against every chunk vector and returns the **top 20** by cosine
similarity. Fast, wide net, approximate.

**⑤ Stage-2 reranking — cross-encoder precision** (`reranker.py`). A cross-encoder
re-reads each `(question, chunk)` pair *together* and re-scores true relevance,
keeping the **top 5**. This fixes cases where the bi-encoder ranked a vaguely-similar
chunk too high. (Doc 02 §6 — this is the retrieve-then-rerank pattern.)

**⑥ Grounded generation** (`generator.py`). The 5 passages are formatted as
numbered sources and handed to Gemma with a strict system prompt: *use only these
sources, cite claims by [n], and if the answer isn't here, say so.* `temperature=0`
makes it faithful, not creative.

**⑦ Stream + cite.** For `/ask/stream`, the server sends the **citations first**
(so the Sources panel fills instantly), then streams the answer **token by token**
as Gemma writes it (doc 03 §3). The `[n]` markers in the text map 1:1 to the
citation list; the frontend turns them into clickable superscripts.

```
"What about the decoder?"
      │  ① auth → your index
      │  ② condense → "How does the decoder work in the Transformer?"
      │  ③ embed (mpnet, 768-dim)
      ▼
  FAISS top-20  ──④──▶  cross-encoder rerank  ──⑤──▶  top-5 passages
                                                          │
                                        ⑥ Gemma (temp 0, grounded prompt)
                                                          │
                          ⑦ stream: citations first, then answer tokens ▶ browser
```

---

## 5. Module map (where everything lives)

```
backend/
  parser.py      PDF → per-page text (PyMuPDF)
  chunker.py     text → token-accurate overlapping chunks
  embedder.py    text → 768-dim L2-normalised vectors (mpnet)
  store.py       FAISS index + chunks.json (build / append / remove_paper)
  retriever.py   stage-1: embed query → FAISS top-k (bi-encoder)
  reranker.py    stage-2: cross-encoder re-scores candidates
  generator.py   grounding prompt, generate() + stream_answer() + condense_question()
  library.py     per-user index: index_pdf / get_retriever / delete_paper_data
  papers.py      routes: upload / list / delete / serve-file
  api.py         app wiring, /ask + /ask/stream, _resolve_and_retrieve
  auth.py        register/login/logout/me, cookies, CSRF, get_current_user
  security.py    bcrypt hashing + JWT encode/decode
  db.py          SQLite engine + session dependency
  db_models.py   SQLModel tables: User, Paper
  models.py      Pydantic API models: Chunk, Citation, Answer, AskRequest, ...
  config.py      env-loaded settings (pydantic-settings)

frontend/src/
  pages/Workspace.tsx   the three-pane app (library · chat · sources)
  pages/Landing.tsx     public landing page
  pages/AuthForm.tsx    login / signup
  components/            Library, SourcePanel, AnswerView, QueryBar, PdfViewer, icons
  auth/                  AuthContext + ProtectedRoute
  api.ts                typed client (mirrors backend models); streaming reader
```

**Next:** `02-ml-dl-nlp-concepts.md` explains the *why* behind embeddings,
transformers, vector search, reranking, and grounded generation.
`03-engineering-auth-frontend.md` covers FastAPI, streaming, auth, and the UI.
