# Scholar — Engineering, Auth & Frontend

> The non-ML machinery that makes Scholar a real multi-user product: the API
> layer, streaming, persistence, the security model, and how the React frontend
> consumes it all. Doc 01 = pipeline, doc 02 = the ML; this = everything else.

---

## 1. FastAPI — the API layer

FastAPI is the web framework. Three ideas make it a good fit for Scholar:

### 1.1 Pydantic models are the API contract
Every request body and response is a typed **Pydantic** model (`backend/models.py`).
FastAPI uses them to **validate incoming JSON automatically** (a missing field → a
422 error before your code runs) and to **serialize responses** (including nested
objects) back to JSON. You write zero parsing code.

```
AskRequest   { question, k=5, candidates=20, history[] }   ← validated on the way in
Answer       { question, answer, citations[] }             ← serialized on the way out
Citation     { paper_id, page, chunk_index, score, text, rerank_score }
```

The same object that crosses the wire is the same shape the frontend's `api.ts`
declares in TypeScript — so a Python → JSON → TypeScript round-trip stays
type-safe end to end.

### 1.2 Dependency injection
Endpoints declare what they need as parameters and FastAPI supplies them:
- `user: User = Depends(get_current_user)` → auth + the logged-in user
- `session: Session = Depends(get_session)` → a DB session, opened and closed per request
- `dependencies=[Depends(require_csrf)]` → CSRF enforcement on mutating routes

This keeps cross-cutting concerns (auth, DB, CSRF) out of the endpoint body and
makes them reusable and testable.

### 1.3 The "warm model" pattern (load once, serve many)
Loading FAISS + the two Transformer models costs seconds. Doing that **per request**
would make every question pay it. So the reranker is instantiated **once** at import,
retrievers are **cached per user** (`library._retrievers`), and there's an explicit
model-warming call so the first real answer doesn't eat the LLM's cold-start.

> **The general principle of ML serving:** split *loading* (slow, once, at startup)
> from *inference* (fast, every request). Servers pay the load cost upfront so each
> call stays cheap.

Endpoints that call the blocking LLM are plain `def` (not `async def`) — FastAPI
runs sync endpoints in a **threadpool**, so one slow generation doesn't freeze the
whole server.

---

## 2. Persistence — SQLite + SQLModel

**SQLite** is a zero-configuration, single-file database (`data/scholar.db`).
**SQLModel** lets one class be **both** a database table **and** a Pydantic model —
so "type everything at the boundary" extends all the way down to storage, with no
hand-written SQL.

Two tables (`backend/db_models.py`):

```
User   id · email (unique) · hashed_password · created_at
Paper  id · user_id (FK) · paper_id (slug) · title · filename · n_chunks · created_at
```

The **vectors themselves are NOT in SQLite** — they live in per-user FAISS files.
SQLite holds only lightweight metadata and the catalogue of who owns what. This
separation keeps the relational store small and the vector store specialized.

---

## 3. Streaming answers (NDJSON)

Generation takes ~10–25s locally. Making the user stare at a blank screen that whole
time is bad UX, so `/ask/stream` **streams**.

**Transport: NDJSON** (newline-delimited JSON) — one JSON object per line:
```
{"type":"citations","citations":[...]}   ← sent FIRST (sources panel fills instantly)
{"type":"token","text":"Multi"}          ← then many token lines as Gemma writes
{"type":"token","text":"-head"}
...
{"type":"done"}                           ← or {"type":"error","message":...}
```

Server side (`generator.stream_answer`) sets `stream=True` on the Ollama call and
**yields each token delta** as it arrives; FastAPI wraps that generator in a
`StreamingResponse`. Because retrieval + reranking finish *before* generation, the
citations can be sent up front and the prose streamed on top of them.

Frontend side (`api.ts`): reads the response body as a stream, splits on newlines,
parses each line, and dispatches callbacks — `onCitations` (render the Sources
panel), `onToken` (append text + a live caret), `onDone`/`onError`. React state
updates on each token, so the answer visibly types itself out.

> **Why not Server-Sent Events or WebSockets?** NDJSON over a plain POST is the
> simplest thing that works: one request, one response, incremental lines. No extra
> protocol, no persistent connection to manage. (Don't over-engineer.)

---

## 4. The security model (auth)

Scholar is multi-user, so it needs real accounts. Four building blocks:

### 4.1 Password hashing — bcrypt
Passwords are **never stored in plaintext**. `security.py` hashes them with **bcrypt**
(`$2b$12$...`) and verifies by re-hashing the input. bcrypt is deliberately **slow**
(a work factor of 12) so brute-forcing stolen hashes is expensive.

**The 72-byte trap:** bcrypt **silently truncates** input past 72 bytes — two
different long passwords could hash equal. Scholar rejects `>72 bytes` (and `<8
chars`) at the **Pydantic boundary** with a clear 422, so it fails loudly instead of
silently. *(Exact same "fail at the model's limit, don't approximate" discipline as
the chunker's 384-token guard in doc 02 §3.)* Password rules are min-8, **no forced
symbols** — aligned with NIST SP 800-63B, which found composition rules hurt more
than help.

### 4.2 Sessions — JWT in an httpOnly cookie
On login, the server mints a **JWT** (JSON Web Token) — a signed token whose payload
says "user id = X, expires at T". It's signed with `HS256` using a secret
(`SECRET_KEY`, required from env, **no default** so a missing key fails loudly rather
than shipping forgeable tokens). Because it's signed, the server can trust it without
a session table.

The token is delivered as an **httpOnly cookie**: JavaScript **cannot read it**, so a
cross-site-scripting (XSS) bug can't steal it — unlike a token kept in
`localStorage`. The browser sends it automatically on every request. Tokens are
short-lived (`exp`, 30 min), no refresh-token machinery.

### 4.3 CSRF — double-submit cookie
Cookies are sent automatically by the browser — including on requests a malicious
site tricks you into making (Cross-Site Request Forgery). Defense: on login the
server also sets a **second, readable** `csrf_token` cookie. The frontend reads it and
echoes it in an `X-CSRF-Token` **header** on every state-changing request (POST/DELETE).
A cross-site attacker can *send* the cookie but **cannot read** it to set the matching
header (the same-origin policy blocks that), so the check fails. Safe methods (GET)
don't need it.

```
login  → sets  access_token (httpOnly)  +  csrf_token (readable)
mutate → must send  X-CSRF-Token: <value of csrf_token cookie>   else 403
```

### 4.4 Cross-origin, done right
The frontend (`localhost:5173`) and API (`localhost:8001`) are **different origins**,
so:
- fetches use `credentials: "include"` (send/receive the cookie cross-origin)
- CORS uses `allow_credentials=True` with the **explicit** frontend origin (credentials
  are **incompatible** with `allow_origins="*"`, so the origin must be named)
- `SameSite=Lax` works because `localhost:5173` and `localhost:8001` are the *same
  site* (ports don't change the site); in production over HTTPS you'd switch to
  `SameSite=None; Secure`.

---

## 5. Per-user isolation (multi-tenancy)

Every account's data is walled off:
- **Storage:** one FAISS index per user under `data/users/<id>/` (doc 01 §3.4).
- **Retrieval:** `/ask` loads only the caller's index; no user's chunks are ever in
  another user's search space.
- **Ownership checks:** every `/papers/{id}` route verifies `paper.user_id == user.id`
  before reading, serving, or deleting — a 404 otherwise (don't even reveal existence).

This is defense in depth: isolation by storage layout *and* by explicit per-request
checks.

---

## 6. Upload & delete internals

### 6.1 Upload (`papers.py` → `library.index_pdf`)
Validate (type/size/non-empty) → save the PDF → run the indexing pipeline (parse →
chunk → embed → `append_to_store`) → write a `Paper` row. Indexing is **synchronous**
inside the request; if it yields zero chunks (scanned PDF, no text layer), the file is
rolled back and a 422 explains why. `append_to_store` adds the new vectors to the
existing index and keeps `faiss_id`s a contiguous `0..N-1` map into `chunks.json`.

### 6.2 Delete = rebuild (`store.remove_paper`)
`IndexFlatIP` has **no native delete**. So removing a paper **rebuilds** the user's
index from the chunks that remain:
1. `index.reconstruct(i)` pulls each kept vector back out of the flat index — **no
   re-embedding needed** (a flat index stores the raw vectors, so we already have them).
2. A fresh index is built from those vectors, `faiss_id`s renumbered to stay contiguous.
3. If nothing remains, the index files are deleted outright.

> **Interview angle:** knowing *why* delete is a rebuild (flat indexes are
> append-only) and *why* it's cheap (vectors are reconstructable, so no re-embed)
> shows you understand the datastore, not just the API.

---

## 7. The frontend

React + Vite + TypeScript + Tailwind, organized around the RAG workflow.

### 7.1 Structure & routing
- `main.tsx` wraps the app in `BrowserRouter` + `AuthProvider`.
- `App.tsx` = routes: `/` (public landing), `/login`, `/signup`, and a **protected**
  `/app` (the workspace) gated by `ProtectedRoute`.
- `auth/AuthContext.tsx` holds the current user; on load it calls `/auth/me` to
  restore a session from the cookie. `ProtectedRoute` redirects to `/login` until that
  resolves.

### 7.2 The workspace — three panes
`pages/Workspace.tsx` composes:
- **Library** (left) — upload dropzone + paper cards + delete; shows an "indexing…"
  progress card during the synchronous upload.
- **Chat** (center) — the question/answer thread + composer. Marks follow-up turns and
  shows the streaming caret.
- **Sources** (right) — the retrieved passages with rerank meters; a card highlights
  when you hover/click its citation.

### 7.3 The feature that ties it to the RAG core: citation ↔ source linking
The LLM emits `[n]` as plain text (it has no idea a UI exists). `AnswerView` splits the
answer on the regex `\[\d+\]`, and rehydrates each marker into a clickable superscript
whose number indexes into the **same `citations` array** the reranker produced. Hover a
marker → its Source card lights up and scrolls into view; click a source → the
**PdfViewer** opens the stored PDF to that page (`#page=N`). That claim → passage →
page chain is the product expression of the whole grounded-RAG idea.

### 7.4 Design system
An editorial "digital manuscript" look: warm paper background, ink text, a single
oxide-red accent, Newsreader (serif) + Atkinson Hyperlegible (body) + JetBrains Mono
(labels), a soft 3-step elevation/shadow scale, and motion that respects
`prefers-reduced-motion`. Accessibility (WCAG-AA contrast, visible focus rings, 44px
touch targets) is treated as a requirement, not polish.

---

## 8. How to run it (recap)

Three processes, all at once:
1. **Ollama** with `gemma3:4b` pulled (local LLM).
2. **Backend:** `.venv\Scripts\python.exe -m uvicorn backend.api:app --port 8001`
   (use the venv Python; no `--reload` — it can silently re-exec to global Python).
3. **Frontend:** `cd frontend && npm run dev` → open `http://localhost:5173`.

Sign up → upload a PDF → ask → follow up → click a citation. That exercises every
layer in these three docs: auth, upload/indexing, retrieval, reranking, grounded
streaming generation, and citation linking.
