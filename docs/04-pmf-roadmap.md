# Scholar — Roadmap to Product-Market Fit

> Docs 01–03 describe what Scholar *is* (pipeline, ML, engineering). This doc is about what it
> has to become: the gap between a system that works correctly and a product a team will depend
> on for real work. Target user: **professionals and teams querying reports, contracts, and
> internal documents.** Wedge: **auditable answers over private corpora.**

---

## 1. Where we actually stand

Scholar works. A full end-to-end run against an isolated sandbox instance passed **37 of 38**
assertions: auth, upload, indexing (60 chunks from a real PDF), two-stage retrieval, grounded
answers with page-anchored citations, correct refusal when the corpus lacks the answer, NDJSON
streaming, and verified cross-tenant isolation.

"Works" is not "product-market fit". Measured against the target user, six structural gaps:

| # | Gap | Evidence |
|---|---|---|
| 1 | **Cannot read most of their documents** | Only text-layer PDFs accepted. `.docx`/`.pptx`/`.xlsx` → 415. A scanned/signed contract → 0 chunks → 422. |
| 2 | **Tables are garbled** | `parser.py` is 15 lines calling bare `page.get_text()`. A results table retrieved as `"-Token 43.5 54.8 46.5 51.9 ... RAG-Sequence 44.0 55.8..."` — unlabeled number soup. |
| 3 | **Search misses exact terms** | Semantic-only retrieval. "Section 7.2", "Force Majeure", party names, dates, dollar amounts are where dense embeddings are weakest. |
| 4 | **"Auditable" is not yet true** | Citations resolve to a *page*, not a span. Nothing is logged. The promise is a claim, not a feature. |
| 5 | **Built for one person** | `db_models.py` has only `User` and `Paper(user_id)`. No workspace, membership, or role. Storage hard-coded to `data/users/<id>/`. |
| 6 | **Cannot be depended on** | Zero tests, no eval, no CI, in-process state, FAISS on ephemeral disk with no backups, no password reset, no rate limiting. |

**Intended outcome:** a product a team can adopt for real work — it reads their actual documents,
finds the exact clause, proves where every claim came from, survives a redeploy, and does not lie
to them on the landing page.

---

## 2. Architecture: where WAT applies (and where it must not)

**WAT = Workflows, Agents, Tools.** Applied honestly, not decoratively — `claude.md` mandates
*"No abstractions until we need them twice."*

| Layer | Contents | Status |
|---|---|---|
| **Tools** | Typed, independently testable capabilities: `search_semantic`, `search_lexical`, `rerank`, `extract_text`, `ocr_page`, `extract_tables`, `fetch_span`, `list_documents` | Introduce in Phase 2 |
| **Workflows** | Deterministic orchestration, no LLM in the control loop: `answer_question` (today's path), `ingest_document` | Exists — keep |
| **Agents** | LLM-driven control flow, iterative: `research_agent` for multi-hop and comparative queries | Phase 6 only |

**Three deliberate decisions:**

1. **Do not refactor the working pipeline into WAT as a first step.** Pure risk, no user-visible
   benefit, and no tests to protect it. The `tools/` package earns its existence in Phase 2, when
   lexical search arrives and there are genuinely *two* search implementations to unify — that is
   the "needed twice" trigger. The eval harness also needs to call retrieval without generation.
2. **The default query path stays a Workflow forever.** Single-shot questions must not pay agent
   latency or non-determinism. Auditability depends on reproducible retrieval.
3. **Agents are last and narrow.** Only the comparative/multi-hop class ("which of these contracts
   auto-renew?") justifies an agent loop, and only after the deterministic path is measurably good.
   The buyer's objection is *"can I trust this answer"*, not *"can it plan"* — and an agent loop
   makes answers nondeterministic, latency unbounded, and the audit trail branch-dependent. That
   works directly against what we're selling. Build one only on **measured** failure: if the
   `tags=["comparative"]` eval cluster still scores below ~40% after Phase 2.

**Where the boundary actually sits.** Not "is an LLM involved" — that would misclassify
`condense_question`, which calls an LLM but whose control flow is fixed regardless of the reply.
The test is: **does control flow branch on model output?**

| | Branches on model output? | Layer |
|---|---|---|
| `condense_question` | No — one call, result feeds a fixed pipeline | Workflow |
| retrieve → fuse → rerank → generate | No | Workflow |
| "which tool next, and how many times?" | Yes | Agent |

This is a testable predicate, and it means adding a second deterministic LLM call later doesn't
accidentally promote the pipeline to an agent.

**Workflows earn a directory, not a base class.** No `Workflow` ABC, no step list, no DAG — that
would be ceremony around 18 readable lines. What is earned is extracting
`api._resolve_and_retrieve` into `backend/workflows/ask.py::answer_question(ctx, req) -> Answer`:
the route drops to three lines, the workflow becomes testable without a FastAPI client, and the
Phase 3 audit-log write happens in one place instead of being duplicated across `/ask` and
`/ask/stream`.

---

## 3. Phases

### Phase 0 — Make change safe and measurable *(1 week)* — **DONE**

Nothing downstream is trustworthy without this. Every later phase changes parsing, retrieval, or
the model — each needs a before/after number or it is guesswork.

- **Integration suite** — DONE. `backend/tests/`, **66 tests**, promoted from the sandbox script and
  converted to `TestClient` so they need no running server, no Ollama and no network. Split by a
  `slow` marker: 50 run in ~13s, 16 load the real embedding/reranking models (~35s).
  `conftest.py` redirects `DATABASE_URL` and `library.DATA_ROOT` to a temp tree **before** importing
  `backend.*`, then asserts the redirection held — so a future change that breaks isolation aborts
  the run instead of writing into the real library.
- **Eval harness** — DONE. `python -m backend.eval`. 30 hand-labelled questions over two documents,
  tagged `semantic` / `exact` / `table`, plus 5 refusal probes. `--check` verifies every expected
  span really occurs in the extracted text (all 30 verified), because a mislabelled span scores as a
  permanent miss and quietly drags the baseline down forever. Corpus is generated or downloaded, not
  committed.
- **Unit tests** for `chunker.chunk_pages` — DONE. The 380/50/384 budget, verbatim slicing, overlap,
  page/index ordering, degenerate input.
- **CI** — DONE. `.github/workflows/ci.yml`: fast tests, model-backed tests (Hugging Face cache),
  eval-label validation, and `npm run build` (`tsc -b`) for the frontend.
- **Bug fixes — two confirmed and fixed, one disproved:**
  - `papers.py` — corrupt PDF returned **500** and orphaned the file. `fitz.open` raises
    `FileDataError`; the rollback `pdf_path.unlink()` only ran inside the `if n_chunks == 0` branch,
    so it never executed. **Fixed**: indexing is wrapped, the file is rolled back on either failure
    mode. Verified 422 with an empty library through a real server.
  - `api.py` — `/ask` and `/ask/stream` had no `require_csrf`, and `api.ts::askStream` omitted the
    header. **Fixed** on both sides. Note the resulting status for an anonymous caller is **403, not
    401**: `dependencies=[...]` resolves before the endpoint's own `get_current_user`. That is
    already how every other unsafe route in the app behaves, so this is consistent — but it means an
    expired session gets a 403 the frontend does not route to the login page. Worth revisiting in
    Phase 4 alongside refresh tokens.
  - ~~`reranker.py` — `CrossEncoder` overflow~~ — **DISPROVED, no change made.** The `666 > 512`
    warning does not come from the reranker. It comes from `chunker.py` tokenizing a *whole page*
    (measured: up to 1223 tokens) before windowing it — intentional and harmless, since the page is
    never fed to a model. Measured on the real corpus, the worst `(question, chunk)` pair reaches
    **399** tokens against the cross-encoder's 512 limit, so nothing is being truncated. Instead of a
    speculative fix, `test_chunker.py` now asserts that ceiling, so Phase 1's longer `embed_text`
    (serialized tables) cannot silently breach it. The false-alarm warning is silenced at its real
    source so a genuine one is not lost in the noise.

**Acceptance — met.** 66/66 pytest green; 18/18 verification through a real uvicorn server
(including live generation and NDJSON streaming); corrupt upload returns 422 and leaves no file on
disk; eval prints the baseline below.

#### Recorded baseline — 2026-08-02

`k=5`, `candidates=20`, n=30 over 65 chunks. **No retrieval change ships without moving these.**

| Slice | hit@5 | MRR | hit@1 |
|---|---|---|---|
| stage 1 (bi-encoder, top 20) | 96.7% *(hit@20)* | 0.711 | — |
| **stage 2 (after rerank, top 5)** | **93.3%** | **0.801** | **70.0%** |
| — `semantic` (n=11) | 81.8% | 0.639 | 54.5% |
| — `exact` (n=15) | 100.0% | 0.900 | 80.0% |
| — `table` (n=4) | 100.0% | 0.875 | 75.0% |

The reranker earns its cost: it raises MRR from 0.711 to 0.801 while cutting 20 candidates to 5.

**Read this baseline with two caveats, both printed by the tool itself:**

1. **65 chunks is a toy corpus.** Retrieving 5 of 65 is a 1-in-13 task; a real professional library
   is thousands of chunks. These numbers are an optimistic ceiling, not a product metric.
2. **`exact` scoring 100% is not evidence that hybrid retrieval is unnecessary.** `sample_agreement`
   produces only 5 chunks, so with `k=5` it *cannot* fail hit@5 — the harness now prints a warning
   saying exactly that. The honest signal for those questions is hit@1 (80%) and MRR. Phase 2 must
   grow the corpus with distractor documents **before** claiming a hybrid-retrieval win.

The two misses are both `semantic` and both real: *"which components are fine-tuned and which are
kept frozen"* and *"what optimizer is used"*. The second is the textbook case for lexical search —
the answer is the single word `Adam` inside a chunk about training cost — which is direct support
for Phase 2 that did not exist before.

**Still outstanding from this phase:** refusal-correctness is defined (5 probes) but not scored, as
it needs a live LLM. The eval reports it as not-run rather than pretending otherwise.

---

### Phase 1 — Read the documents professionals actually have *(2 weeks)* — **DONE**

The largest user-visible jump. Before this, the product could not open a `.docx` at all.

- **Format expansion — DONE.** `.docx`, `.pptx`, `.xlsx`, `.txt`/`.md`, each an extractor behind the
  same `list[str]` contract `extract_pages` already returned, so `chunk_pages` and everything
  downstream is untouched. The 415 gate in `papers.py` now checks the **extension**, not the
  browser-supplied `content_type` — the latter is client-chosen and varies by OS for the office
  formats, while the extension is what actually selects the extractor. Storage keeps each file's own
  extension (it used to hardcode `.pdf`, which would have made a stored workbook unreadable);
  `library.stored_path` resolves it. Office formats are served back as **attachments**, since no
  browser renders them inline.
- **Table serialization — DONE.** Tables render as Markdown so each number keeps its row and column
  label. This is the whole point: flattened, a fee table reads as `Standard 25 180,000 Professional
  100 480,000` — the numbers are present but attached to nothing, and no retrieval quality makes
  that answerable. `.docx` extraction walks the document body in order, so a table stays between the
  paragraphs that introduce and follow it (python-docx's `.paragraphs` / `.tables` are separate flat
  lists that lose the interleaving).
- **`get_text(sort=True)` — MEASURED AND REJECTED. It does the opposite of what this roadmap
  claimed.** It was billed as the "smallest change, largest jump" for multi-column reading order.
  `sort` orders text blocks by vertical position first, so on a two-column page it reads *across*
  both columns and interleaves them line by line:

  | | extracted text |
  |---|---|
  | default | `...vacancy fell to 3.1% in the third quarter, the lowest level...` |
  | `sort=True` | `...vacancy fell to` **`Rail freight volumes tell a`** `3.1% in the third` **`different story from the`**`...` |

  PyMuPDF's default follows the content stream, which for a well-formed PDF already runs column by
  column. Quantified on a purpose-built two-column brief added to the eval corpus: **the default
  preserves 5 of 5 labelled spans; `sort=True` preserves 0 of 5.** On the full eval it moved 3 of 30
  questions (2 worse, 1 better) — noise — while visibly destroying two-column text. It would only
  help on a PDF whose content stream is already scrambled, which is not the common case. **Not
  shipped.** The rejection is recorded in `parser.py` so it is not re-attempted.
- **`find_tables()` — MEASURED AND REJECTED** (earlier finding, unchanged). On this repo's own PDFs:
  `"lines"` (default) finds **0 tables** on a page that visibly contains one; `"text"` returns **1
  "table" of 64 rows × 6 cols** swallowing the whole page including prose. PyMuPDF itself suggests a
  separate layout package on every call. For this audience the native `.xlsx`/`.docx` path covers
  most business tables anyway — those arrive already structured, with no detection step.
- **OCR fallback — DONE and verified end to end.** `parser._ocr_page` routes an imaged PDF page
  through PyMuPDF's `get_textpage_ocr`, per page and only where there is no text layer — running it
  on a page that already has text would be slower and worse than the text the PDF already carries
  (asserted by `test_ocr_is_skipped_on_pages_that_already_have_text`).

  **Route chosen: Tesseract**, because Claude vision is billed per token — ~$0.047/page on Opus 5,
  ~$0.006 on Haiku 4.5, i.e. $9 or $1.20 for a 200-page contract. Tesseract is $0 and local, at the
  cost of a system binary in the deployment.

  Two things this needed beyond installing it:
  1. **PyMuPDF does not search for Tesseract's language data** — it must be told. `parser._tessdata()`
     resolves it from `settings.tessdata_prefix`, then `TESSDATA_PREFIX`, then the standard install
     paths, so it works out of the box on a normal install and can still be pinned for a container.
  2. **Absence must degrade, not crash.** With no language data a scanned page yields `""` — the same
     "no extractable text" 422 as before — and the message now says OCR is unavailable *on the
     server* rather than blaming a perfectly good file.

  Verified with a fixture that is genuinely scanned: text rendered, rasterised at 200 dpi, text layer
  discarded. A separate test asserts the fixture really has no text layer, so the OCR test cannot
  pass without OCR actually running. CI installs `tesseract-ocr` so this stays covered.
- **Word-level spans for Phase 3 — NOT DONE.** Building page text by joining `page.get_text("words")`
  with per-word char spans and bboxes is the foundation span highlighting needs. Deferred: it is
  Phase 3 work, and doing it here would have meant changing the extraction contract twice.

**Acceptance — met.** A scanned PDF, a `.docx`, an `.xlsx`, a `.pptx` and a `.md` all index
end-to-end and answer with citations; a table's column labels survive into the retrieved chunk
(`test_a_table_is_retrievable_with_its_column_labels`); a scanned contract's text is recovered by OCR
(`test_a_scanned_pdf_indexes_and_is_retrievable`). 106 tests green.

#### What "page" means now — an honest caveat

Only PDF has real pages. The other formats emit the closest natural unit they actually have:
`.pptx` → a slide and `.xlsx` → a worksheet (both exact); `.docx` and `.txt`/`.md` → a **section**,
split on the author's own explicit page breaks where present and otherwise on a 3000-character
budget. So a `.docx` citation reading "page 7" means "the 7th section as this parser divided it",
**not** what Word shows as page 7 — Word paginates at render time from fonts, margins and printer
metrics that are not in the file. The division is deterministic and reproducible, which is what a
citation requires; it just is not Word's. Fixing the *label* means giving `Citation` a format-aware
locator instead of a bare `page`, which is a schema change and belongs with the span-citation work
in Phase 3.

#### Eval after Phase 1 — 2026-08-02

`k=5`, `candidates=20`, **n=35 over 66 chunks** (corpus grew by the two-column brief, so this is
**not** directly comparable to the Phase 0 aggregate).

| Slice | hit@5 | MRR | hit@1 |
|---|---|---|---|
| stage 2 (after rerank, top 5) | 91.4% | 0.824 | 74.3% |
| — `exact` (n=15) | 100.0% | 0.900 | 80.0% |
| — `layout` (n=5, new) | 100.0% | 1.000 | 100.0% |
| — `semantic` (n=11) | 72.7% | 0.621 | 54.5% |
| — `table` (n=4) | 100.0% | 0.875 | 75.0% |

The `layout` row scores perfectly only because that document is a single chunk. The meaningful
measurement for layout is the 5/5-vs-0/5 span-survival test above, not this row.

---

### Phase 2 — Retrieval they can trust *(1.5 weeks)* — **DONE**

**First, the corpus had to be able to measure anything.** The Phase 0/1 eval was two topically
unrelated documents — an ML paper and a contract — so *any* contract question landed in the contract
regardless of retrieval quality. Added three distractors before touching retrieval: a second
services agreement (**with its own Section 7.2, about Assignment rather than Force Majeure**), an
information security policy, and a second two-column brief. 76 chunks over 6 documents, 43 questions.
Still small, but "what does Section 7.2 say?" is now a real disambiguation rather than a giveaway.

- **Hybrid retrieval via SQLite FTS5 — DONE, zero new dependencies.** `backend/lexical.py` mirrors
  each chunk into a per-user FTS5 table beside the FAISS index, written in the same call so the two
  cannot drift. Scoping is by storage layout, not a `WHERE` clause someone can forget — the same
  property the vector index already had. `bm25()` scores are negated on the way out so both
  retrievers agree that higher is better.
- **Reciprocal Rank Fusion — DONE.** `backend/fusion.py`. The two score scales are incomparable
  (cosine ≈ 0..1; bm25 unbounded and corpus-dependent), so ranks are fused instead of scores. The
  fused result is expressed purely as ORDER — no RRF value is written into `Citation.score`, which
  is documented as the stage-1 similarity and would become a lie.
- **Per-document diversity cap — DONE.** No document may hold more than 60% of the shortlist.
  Over-quota passages are demoted, not dropped, so a document that genuinely holds every answer
  still supplies them.
- **Document filter — DONE.** `AskRequest.papers` restricts a query to selected documents. Applied
  after retrieval and explicitly **not** an authorisation boundary — that remains the per-user store.
- **`backend/search.py` instead of `backend/tools/`.** The plan called for a Tools package with
  `search_semantic` / `search_lexical` / `rerank` behind interfaces. Built as one module with one
  function, because there turned out to be exactly two callers (`/ask` and the eval) and no
  polymorphism to serve — `claude.md` says no abstractions until needed twice, and an interface with
  one implementation each is not that. The real win was **making the eval call the same function
  `/ask` calls**: when they build the shortlist separately, the eval measures a lookalike and every
  number it prints is a guess about the product.

**Acceptance — met, with numbers.** `--dense-only` reproduces the old path against the same corpus,
so this is a true A/B rather than a comparison across two different eval sets:

| | dense only | **hybrid** | delta |
|---|---|---|---|
| stage 1 recall (hit@20) | 97.7% | **100.0%** | +2.3 |
| stage 1 MRR | 0.714 | **0.822** | +0.108 |
| stage 2 hit@5 | 93.0% | **95.3%** | +2.3 |
| stage 2 MRR | 0.845 | **0.868** | +0.023 |
| stage 2 hit@1 | 76.7% | **79.1%** | +2.4 |
| `semantic` hit@5 | 76.9% | **84.6%** | +7.7 |
| `exact` / `layout` / `table` | — | unchanged | no regression |

**Stage-1 recall is now 100%: every question's evidence reaches the shortlist.** Retrieval has
stopped being the ceiling.

#### Four things worth knowing, three of them surprises

1. **Lexical alone outranks dense alone on this corpus** — MRR 0.856 vs 0.714. On professional
   documents full of clause numbers, party names, dates and amounts, BM25 is not the junior partner
   the "semantic search" framing implies. Do not drop it later as redundant.
2. **The reranker is now the bottleneck, not retrieval.** Stage 1 finds the evidence for 100% of
   questions; stage 2 delivers 95.3%. The cross-encoder is *discarding* correct passages retrieval
   already found. That is where the next retrieval-quality work is, not in stage 1.
3. **A tokenizer bug found by a failing test.** The first implementation split `7.2` into `"7" OR
   "2"` before quoting, which matches 7.1, 2.7 and every other clause containing a 7 — and produced
   an exact RRF tie between 7.1 and 7.2. Terms now keep internal `.`/`-`/`/`, so `"7.2"` is an FTS5
   *phrase*. Pinned by `test_a_dotted_reference_stays_one_phrase`.
4. **RRF does not mean "consensus wins", and this roadmap's earlier phrasing was wrong.** Because
   `1/x` is convex, `1/61 + 1/63 > 2/62`: a chunk ranked 1st-and-3rd narrowly beats one ranked
   2nd-and-2nd. What RRF actually rewards is being *found by both* retrievers at all, which is still
   the property that matters here. Both behaviours are now pinned by tests so neither is a surprise.

**Predicted and wrong:** I expected BM25 to fix `rag-04` ("what optimizer is used?") because the
answer contains the rare token `Adam`. It did not, and the reason is instructive — lexical search
helps when the rare token is in the *query*, not when it is in the *answer*. That question needs
query expansion, which is not in this plan.

---

### Phase 3 — Auditability: the actual wedge *(1.5 weeks)* — **MOSTLY DONE**

The differentiator. It must be literally true, not a marketing line — which is why the honest
qualifications below are part of the deliverable, not footnotes to it.

- **Audit log — DONE.** `AnswerLog` records the question, the *condensed* query retrieval actually
  ran on, the answer, the complete evidence chain (every passage with its `faiss_id`, char span,
  locator, and both stage scores), the model and temperature, and the retrieval settings. Written
  after generation, never during: a failed stream leaves no entry claiming success. `GET /audit`,
  `GET /audit/{id}`, both scoped to the caller and 404-not-403 on someone else's entry.
- **Export — DONE.** `GET /audit/{id}/export?format=json|csv`. JSON keeps the whole record; CSV is
  one row per cited passage and repeats the question on every row, so the file survives being
  sorted or filtered in a spreadsheet.
- **Char spans — DONE.** The chunker already *computed* the window's char span and threw it away.
  It is now carried on `Chunk` → `Citation`, with the invariant asserted directly:
  `page[char_start:char_end] == chunk.text`. `faiss_id` is likewise passed through instead of
  dropped, so a logged passage ties to a specific row of a specific index rather than to text that
  merely looks the same.
- **Format-aware locator — DONE** (deferred from Phase 1). `Citation.locator` is a computed field
  serialised to the client: "page 12", "slide 3", "sheet 2", "section 7". Calling a worksheet
  "page 1" is a small lie, and this product's claim is that a citation reads literally.

#### On "reproducible", precisely

The original wording here was *"retrieval is deterministic, so a logged answer is reproducible"*.
The first half is true; the second does not follow. Retrieval is deterministic given a **fixed
index** — but the library changes as documents are added and removed, and that silently changes
what any question returns. A log that ignored this would imply a guarantee it cannot make.

So every entry stores an `index_fingerprint` (a hash of `chunks.json`, which is what actually
defines the retrievable set) and the chunk count. `GET /audit` reports `reproducible` per entry by
comparing against the library *now*. A `false` does not mean the answer was wrong — it means
re-running today would search a different corpus, so any difference is explained by the library
having changed rather than by the system being inconsistent. **Saying which of those it is, is the
entire value.** Verified end to end: an answer reads `reproducible: true`, and flips to `false` the
moment another document is uploaded.

#### A bug this phase surfaced in Phase 2's work

RRF keeps the `Citation` instance from whichever retriever ranked a passage best. Lexical hits are
built from the FTS5 table, which knows only `(paper_id, chunk_index)` — so **a passage that the
lexical retriever ranked higher reached the caller with no `faiss_id` and no char span**, and the
audit trail silently depended on which retriever happened to win. Caught by
`test_citations_carry_their_audit_trail`.

Fixed by hydrating lexical results from `chunks.json` via `Retriever.hydrate`, rather than copying
the audit fields into the FTS5 schema — two stores holding the same fields is exactly the kind of
duplication that drifts apart later. One owner, looked up on the way through.

#### Not done, and why

- **In-PDF highlight rectangles.** The span data now exists, but drawing a rectangle needs
  `pdfjs-dist` replacing the native `<iframe>` — a real frontend dependency and a UI rewrite, which
  is more than this phase should absorb without asking. What shipped instead: the citation card
  shows the honest locator with the exact char range on hover, and the source drawer shows the
  verbatim passage.
- **A related honesty fix that could not wait.** Phase 1 added formats no browser renders inline,
  and the drawer would have pointed an `<iframe>` at a `.docx` — which makes the browser *download*
  it. Non-PDF sources now show the cited passage verbatim plus a download link, and say why.
- **Claude `citations: {enabled: true}`.** Still the right move when generation moves to the
  managed API (it returns `cited_text` + `char_location` for free, and is incompatible with
  `output_config.format`). Not applicable while generation is local Ollama.

**Acceptance — met except the highlight.** Every answer is retrievable from the audit log with its
evidence; the evidence is byte-identical to what was served (including via streaming, reassembled
from the deltas rather than regenerated); a changed library is detectable. 169 tests green; 31/31
through a real server.

---

### Phase 4 — Durable and operable *(2 weeks)* — **MOSTLY DONE**

Prerequisite for any external user, and for Phase 5 — team data multiplies the cost of data loss.

- **The silent data-loss bug — FIXED.** `library.DATA_ROOT` was resolved relative to its own source
  file, so every user library lived *inside the source tree*. A redeploy that replaces the source
  destroyed all of it, with no error to notice. Now `settings.data_root`, and startup logs a loud
  warning when it is unset, naming the path that is about to be at risk.
- **Backup and restore — DONE, and tested by actually restoring.** `python -m backend.backup
  create|verify|restore`. The archive carries the database *and* the data tree together, because
  either alone is a broken library — the database says a paper exists, the data tree holds its
  vectors. `restore` refuses to run over existing data without `--force` (the failure mode is
  someone restoring a stale backup while only meaning to inspect it) and replaces rather than
  merges (a restore reproduces a moment; a merge would resurrect deleted documents). Archives are
  untrusted input, so extraction is guarded by both an explicit traversal check and
  `filter="data"`. Refuses outright on a non-SQLite `database_url` rather than writing an archive
  silently missing half the data.
- **Refresh tokens — DONE.** A 30-minute hard logout mid-session is gone. The critical detail:
  both token kinds are signed with the same key, so the payload carries a `typ` claim — without it
  a long-lived refresh token would work as an access token for its entire lifetime, silently
  undoing the short access expiry. Tokens minted before `typ` existed fail closed. The refresh
  cookie is scoped to `/auth/refresh` so a long-lived credential is not attached to every request,
  and the CSRF cookie deliberately outlives the access token so a silently-refreshed session never
  fails a CSRF check for want of a fresh cookie. The frontend spends one refresh on a 401 and
  replays the request, sharing a single in-flight refresh across concurrent callers.
- **Password reset — DONE server-side.** Permanent lockout is gone. Only the token *hash* is
  stored; `/auth/forgot` always returns 202 so it cannot be used as an account-existence oracle;
  tokens are single-use, expiring, and using one burns every other outstanding token for that
  account.
- **Rate limiting — DONE, with a stated limitation.** Per-user hourly budgets on `/ask` and upload
  — the two expensive routes, and a *billing* control once generation is a paid API. The counter
  is in process memory: exact with one worker, N times the limit with N workers, forgotten on
  restart. That makes it a guard against runaway loops and honest mistakes, **not** a defence
  against a determined attacker. A shared store (Redis) is the fix and is a deployment decision.
- **Structured logging — DONE.** `print` replaced with the `logging` module throughout the request
  path, so output has levels, timestamps and logger names, and a lost audit row leaves a traceback.

#### Not done, and why

- **Job queue for indexing.** `papers.py` still indexes synchronously inside the upload request, so
  a large document can exceed a proxy timeout. This genuinely needs a new dependency (`arq`/RQ plus
  Redis, or equivalent) and a deployment decision, so it is not something to slip in unannounced.
  **This is the top remaining item in this phase.**
- **Refresh-token rotation and revocation.** Refreshing re-issues the cookie so the expiry slides,
  but the previous refresh token stays valid until it expires — a stolen one cannot be revoked.
  Real rotation needs server-side token state (a table like `PasswordResetToken`), which is a
  contained piece of work but was not in scope here.
- **Reset-link delivery.** `auth.deliver_reset_token` logs the token instead of emailing it,
  because no mail provider is configured. It is written as a seam — wiring a provider means
  replacing that function body and nothing else — and it says plainly in its own docstring that
  password reset must not be exposed to real users until it is replaced. No reset UI was built, on
  purpose: a form promising an email that never arrives is worse than no form.
- **`cookie_secure=True` / `SameSite=None`** remain settings, correct for the eventual HTTPS
  deployment and wrong for local development. Flipping them is a deploy-time change, not a code one.

**Acceptance — met for durability.** A library was built, backed up, **deleted the way a redeploy
would delete it**, restored, and then still answered questions through the API with citations.
207 tests green; 23/23 through a real server across a stop/destroy/restore/restart cycle.

- **Persistent storage + tested restore.** FAISS indexes and documents move off ephemeral local
  disk. A redeploy currently deletes every user's library silently.
- **Auth completeness** — password reset (today a forgotten password is permanent lockout),
  refresh tokens (30-min JWT logs users out mid-session), `cookie_secure=True` + `SameSite=None`.
- **Rate limiting and per-workspace cost caps** — now a billing control, not just performance.
- **Observability** — structured logs (replacing `print` in `api.py`), error tracking, per-request
  token/cost attribution.
- **Job queue for indexing** — `papers.py` indexes synchronously inside the upload request; a large
  document will exceed the proxy timeout. Also unlocks the progress UI the frontend already implies.

**Acceptance:** a redeploy preserves all libraries; restore from backup verified; a 200-page
document indexes without timing out.

---

### Phase 5 — Teams *(2 weeks)* — **DONE**

The highest-risk step in this roadmap, gated behind Phase 0's tests and Phase 4's backups
specifically so it could be attempted safely.

- **Schema — DONE.** `Workspace`, `Membership(user, workspace, role)`, `Invitation`. `Paper` and
  `AnswerLog` gain `workspace_id`; `User` gains `current_workspace_id`. A personal library is an
  ordinary workspace flagged `is_personal`, so there is exactly **one** storage, retrieval and
  authorisation path rather than two that drift apart.
- **Storage migration — DONE.** `data/users/<user_id>/` → `data/workspaces/<workspace_id>/`.
- **Authorisation — DONE.** Ownership checks became membership checks, resolved by a single
  `get_current_workspace` dependency. Scope comes from the user's active workspace, **not** a
  request parameter, so no route can forget to scope itself and no client can point itself at
  another library by editing an id. Absence of membership is 404, never 403.
- **Invitations and roles — DONE.** Two roles, `owner` and `member`, because there is exactly one
  privileged action (managing who else is in the workspace); a finer grid would invent distinctions
  nothing enforces.

#### Three deliberate authorisation choices

1. **Deletion is narrower than reading.** Everyone in a workspace sees every document, but only the
   uploader or an owner can destroy one. Otherwise any member could silently delete a colleague's
   work from a shared library, and there is no undo.
2. **An invitation is not a bearer token.** The invited email must match the accepting account —
   otherwise a forwarded link is a public join link wearing an invitation's clothes.
3. **The last owner cannot be removed.** A workspace with no owner has documents nobody can manage
   and members nobody can add, recoverable only by database surgery.

#### The migration, and what rehearsing it found

Built to the rules that make a migration survivable: idempotent, `--dry-run` first, copy-verify-then-
remove (an interruption leaves both copies, never neither), and refusal to run on real data without
`--i-have-a-backup`. Never a recursive delete of the legacy root — an unaccounted-for directory is
reported and left alone.

**Rehearsing it against a copy of the real database found three bugs, all in the safety machinery
itself.** None would have appeared in unit tests, because all three only occur on a *pre-migration*
database:

| Bug | Why it mattered |
|---|---|
| `_anything_at_risk()` queried `Paper` through the ORM | The backup gate crashed on exactly the database it exists to protect — every ORM query names `workspace_id`, which does not exist yet. Now raw SQL. |
| `backup create` reported **"0 libraries"** | It only looked at the new workspace path, so the backup you are *told to take before migrating* contained none of the data at risk. Now captures both roots. |
| `--dry-run` crashed | The one command someone runs when they are nervous was the one that could not run. Now reads through raw SQL. |

Verified end to end on a copy of the real data: backup → dry run → migrate → serve. Six accounts
migrated, libraries moved under their workspace ids, files byte-identical, and the migrated library
still served through the API.

#### A test isolation breach, found and fixed

`conftest.py` redirected `library.DATA_ROOT` but not the newly-added `LEGACY_DATA_ROOT`, so the
migration tests wrote fixture directories into the developer's **real** `data/users`. Four stray
files, no damage to existing data (verified by content and by an unchanged database hash), removed.
The guard now asserts over *every* writable root, so the next one added fails loudly instead of
being noticed later as stray files.

**Acceptance — met.** Two users in one workspace share a library and both can retrieve from it; a
non-member gets 404 and **zero leaked passages**; the existing isolation tests still pass, extended
to workspaces. 251 tests green; 27/27 through a real server against migrated data.

#### Not done

- **No workspace UI.** The API client is typed and complete (`listWorkspaces`, `createWorkspace`,
  `activateWorkspace`, `inviteMember`, …), but there is no switcher in the interface yet, so teams
  are currently reachable only through the API.
- **Invitation delivery** shares the password-reset gap: no mail provider, so the token is logged
  and also returned to the *inviter* to pass on by hand. That return value is a stated stopgap and
  must be removed the moment mail is wired up — a token in an API response is a credential in a
  place credentials do not belong.

---

### Phase 5 — original plan *(superseded by the section above)*

- **Schema** — `Workspace`, `Membership(user, workspace, role)`; `Paper` gains `workspace_id`.
- **Storage migration** — `data/users/<id>/` → `data/workspaces/<id>/`, with a migration for
  existing single-user libraries.
- **Authorization** — the ownership checks in `papers.py` become membership checks. Keep the
  current 404-not-403 behavior; it hides existence, and it works today.
- **Invitations** and role management.

**Acceptance:** two users in one workspace share a library; a non-member gets 404; existing
isolation tests still pass, extended to workspaces.

---

### Phase 6 — Agentic research layer — **GATED: NOT BUILT, on the evidence**

This phase was always conditional: *"only for the query class the deterministic workflow
demonstrably fails."* That condition has now been measured, and **it is not met.** The agent is not
built, and this section records why — so the decision is revisitable rather than forgotten.

#### What was measured

The eval gained a multi-part scoring mode. A question needing evidence from several passages only
counts as a hit when **every** piece is present in the top k — scoring it on the first fragment
would reward handing the model enough to write a confident, incomplete answer. Nine multi-hop and
comparative questions were added over the existing corpus, including cross-document comparisons
between the two agreements.

`k=5`, `candidates=20`, n=51 over 76 chunks:

| kind | hit@5 | of achievable | n |
|---|---|---|---|
| layout | 100% | 100% | 7 |
| exact | 100% | 92% | 19 |
| table | 100% | 88% | 4 |
| semantic | 100% | 85% | 11 |
| **multihop** (within a document) | **100%** | — | 3 |
| **comparative** (across documents) | **80%** | **63%** | 5 |
| **vocabulary** | **0%** | **0%** | 2 |

*"of achievable" matters:* a question needing N passages cannot complete before rank N, so its best
possible MRR is 1/N. Comparing a comparative question's raw MRR of 0.317 against an exact question's
0.921 compares a 0.5 ceiling with a 1.0 one and means nothing. The eval now reports attainment
against the ceiling for exactly this reason.

#### Why the agent is not justified

1. **Within-document multi-hop already works — 100%.** The deterministic path retrieves both halves
   of "what is the commitment, and what happens if it is missed" without help.
2. **Cross-document comparison is the weakest class but is not failing.** 80% at k=5, and **100% at
   k=12**. The evidence is retrievable; it is simply spread deeper in the list because a comparison
   inherently needs one passage per document. That is an argument for an adaptive `k`, not for an
   LLM planning loop.
3. **The one genuinely broken class is not one an agent fixes.** `vocabulary` scores **0%** — and
   decomposing the question into sub-questions would decompose the same unmatched words.

#### The real finding: a distinct failure mode, previously miscounted

Two questions fail because the query and the passage **share no rare term**: the document says
*availability*, the question says *uptime*; the answer contains *Adam*, the question says *optimizer*.
Lexical search cannot help — it only helps when the rare token is in the **query**, not the answer —
and the embedder does not bridge the gap either. These were sitting inside the `semantic` bucket,
dragging it down and hiding the fact that everything else in it now scores 100%.

**The next retrieval work is query expansion, not an agent.** It addresses the only class at 0%.
That work was then done — see below.

---

### Query expansion — PRF built and REJECTED, HyDE works but is opt-in

Two strategies were implemented and measured against the same corpus. `--expansion none|prf|hyde`
makes this reproducible.

| | none | prf | **hyde** |
|---|---|---|---|
| **hit@5** | 94.1% | 92.2% | **98.0%** |
| MRR | **0.822** | 0.771 | 0.805 |
| hit@1 | **72.5%** | 66.7% | 70.6% |
| misses | 3 | 4 | **1** |
| — `vocabulary` (the target) | 0% | **0%** | **50%** |
| — `comparative` | 80% | 60% | **100%** |
| — `table` MRR | **0.875** | 0.667 | 0.667 |

#### Why pseudo-relevance feedback failed — and it is not a tuning problem

PRF was the *right-looking* choice: deterministic, no dependency, no LLM call, and aimed squarely at
vocabulary mismatch. It made every category worse and moved the target class **not at all**.

The diagnosis is structural. PRF mines expansion terms from the top few passages of the first
retrieval pass, assuming that pass is roughly right and merely needs richer vocabulary. Printing the
feedback set for the failing questions shows the assumption is false here:

```
Q: What is the uptime commitment and what happens if it is missed?
   feedback 1: sample_agreement#4  '8. TERMINATION...'
   feedback 2: sample_agreement#2  '3. FEES AND PAYMENT...'
   feedback 3: market_brief_2col#0 'QUARTERLY MARKET BRIEF...'
   TERMS: ['law', 'next', 'thirty', 'governed', 'governing', 'signed']
```

The passage that answers the question is not in the feedback set — **because of the vocabulary gap
PRF was brought in to fix.** The first pass is wrong precisely for the reason expansion was needed,
so expansion confidently makes it wronger. No amount of tuning depth or term count escapes that
circularity: PRF can enrich a nearly-right query, and cannot rescue a wrong one.

Kept in the tree at `expansion_mode="prf"` rather than deleted, because "did you try PRF?" is a
question that will be asked again, and a measured answer beats an absence.

#### Why HyDE works, and what it costs

HyDE asks the LLM what the answer would *look* like and searches for that. It has the outside
knowledge PRF lacks — that a service level is written as *availability*, that an optimiser is named
*Adam*:

```
Q: What optimizer is used to train the model?
   -> "AdamW was employed for the optimization process."
```

It cuts misses from 3 to 1, takes `comparative` to 100%, and halves the `vocabulary` failures.

**It is off by default anyway**, for a reason specific to this product rather than a general
preference: the audit log promises a logged answer can be reproduced against the same index. An LLM
in the retrieval path makes retrieval non-deterministic, so the central claim quietly weakens.
Temperature is 0, which makes it *nearly* reproducible, and "nearly" is not what the log says.

The precision cost is also real and should not be waved past: MRR 0.822 → 0.805, hit@1 72.5% →
70.6%, and `table` MRR 0.875 → 0.667. HyDE trades ranking precision for recall. Recall is the more
valuable of the two here — a passage the model never sees cannot be used — but it is a trade.

#### HyDE is now the default — with the column that makes it honest

`AnswerLog` gained `retrieval_query` and `expansion_mode`, and `QUERY_EXPANSION` now defaults to
`hyde`. The column is not bookkeeping; it is the precondition. Retrieval is no longer purely
deterministic, so without the generated hypothetical in the log there is no way to tell a changed
library from a differently-worded hypothetical — and `reproducible` would be claiming more than it
can support. `test_the_default_requires_the_audit_column_that_justifies_it` is the tripwire: if the
column is ever dropped, that test forces the default back to `none`.

Three things this needed beyond flipping a setting:

1. **A migration.** `backend/migrate.py` adds both columns to existing databases. `retrieval_query`
   is nullable with no default — an entry written before expansion existed genuinely has no third
   query, and back-filling one would put a fabricated value in an audit trail. Verified on a
   database that already had `answerlog` without the columns.
2. **A fallback.** If the LLM is unreachable, `hypothetical_answer` logs and returns `""`, and
   retrieval proceeds unexpanded. Expansion improves recall; it is not required for a correct
   answer, and a retrieval-time optimisation must never be why a user gets an error.
3. **Three queries, kept separate.** `question` (what was typed) → `query` (standalone, after
   condensing a follow-up; **this is what generation receives**) → `retrieval_query` (plus the
   hypothetical; what retrieval and reranking ran on).

**The separation is load-bearing, and it is observable.** Verified end to end on the question that
scored 0% before: the hypothetical invented *"ninety-nine point nine percent (99.9%) uptime
guarantee"*, and the answer correctly says **99.5%** — the real figure from the retrieved clause.
The hypothetical steers retrieval and never reaches the generator, so it cannot contaminate the
answer. That is the grounding contract doing its job under a change that could easily have broken it.

The eval now defaults to the *configured* strategy rather than to `none`, so `python -m backend.eval`
measures what actually ships instead of a variant nobody runs.

#### Two mistakes this investigation corrected

- **"The reranker is the bottleneck" was wrong.** That came from reading an aggregate (stage 1 100%,
  stage 2 95.3%). Per question, the reranker **promotes 9, demotes 4, and loses 2** — strongly net
  positive. Both losses are the vocabulary cases, where the gold passage genuinely scores low
  (−8.4 and −11.1) because it does not look like an answer to the question as asked.
- **One "multi-hop" question was not multi-hop.** Both of `hop-01`'s required spans live in the
  **same chunk**, so it measured a synonym gap while wearing a multi-hop label. Removed as a
  duplicate of `msa-06`. The eval now flags attainment above 100% precisely because that means the
  labelled parts shared a passage — the label, not the retriever, was describing the difficulty.

#### When to revisit

Build the agent when a measured class actually fails: comparative dropping below ~70% at a realistic
`k` on a corpus of thousands of chunks, or a new class (multi-step reasoning over retrieved numbers)
scoring near zero. The questions are in the eval now, so that check is one command.

---

### Phase 6 — original plan *(retained for reference; superseded by the gate above)*

Only now, and only for the query class the deterministic workflow demonstrably fails.

- **`research_agent`** — decompose a comparative/multi-hop question, run retrieval per sub-question
  via the Phase 2 Tools, iterate if evidence is thin, synthesize with citations preserved.
- **Routing** — classify query complexity; simple questions keep the fast deterministic Workflow.
- Every agent-retrieved passage still flows into the Phase 3 audit log.

**Acceptance:** "which of these contracts auto-renew, and on what notice period?" produces a
correct per-document answer with citations; simple queries show no added latency.

---

### Phase 7 — Launch *(1 week)*

- **Honest copy.** `frontend/src/pages/Landing.tsx` line 82 ("Runs a local model — your papers
  never leave your machine") and line 112 ("Generation runs locally — nothing is sent to a
  third-party model") become **false** the moment generation moves to the managed API. Rewrite to
  what will be true. This is a correctness fix, not marketing.
- **Data-handling posture** — a plain statement of what leaves the customer's environment; ToS,
  privacy policy, account deletion, data export.
- Decide whether a self-hosted generation option belongs on the enterprise roadmap.

---

## 4. Smallest change, largest jump

**Superseded by measurement — kept as a record of a wrong call.** This section originally said
*"Phase 1's OCR fallback plus `get_text(sort=True)`"*. Both halves were wrong about `sort=True`:
it does not fix multi-column reading order, it **breaks** it (5/5 → 0/5 labelled spans; see Phase 1).

What actually turned out to be the largest jump was the least glamorous item on the list: **native
`.docx` / `.xlsx` / `.pptx` ingest**. It converts an entire rejected document class into a working
one with no detection heuristics at all, because those formats arrive already structured. OCR
remains valuable and is built, but it is gated on a system binary; format expansion was gated on
nothing and shipped.

The general lesson, which is why Phase 0 came first: **two of the three extraction "wins" in the
original plan (`find_tables()`, `sort=True`) were rejected by measurement, and neither would have
been caught without the eval.**

---

## 5. New dependencies — approval needed

`claude.md` says *"Ask before adding a dependency."* Verified against what is actually installed:

| Phase | Dependency | Why | State |
|---|---|---|---|
| 0 | `pytest` | No test runner exists | **installed** — `requirements-dev.txt`, kept out of the runtime requirements so a deployment does not ship a test runner. `httpx` (for `TestClient`) was already present via FastAPI, so this is the only addition Phase 0 made. |
| 1 | `python-docx`, `python-pptx`, `openpyxl` | `.docx` / `.pptx` / `.xlsx` ingest | **installed**, pinned in `requirements.txt` |
| 1 | Tesseract **system binary** — OCR route **chosen** over Claude vision, which is billed per token (~$0.047/page on Opus 5, ~$0.006 on Haiku 4.5) | Scanned-document OCR | **still not installed** — `winget install --id UB-Mannheim.TesseractOCR -e`. Code path is written and degrades safely without it. |
| 1 | `anthropic` | Implied by the decided managed-API swap | not yet added |
| 2 | *(none)* | Hybrid search uses **built-in SQLite FTS5** | **available, 3.45.3** |
| 4 | queue + storage libs (e.g. `arq`/RQ, S3 client) | Async indexing, durable storage | to be chosen |

Phase 2 needing nothing is the notable result — the obvious choice (`rank_bm25`) is avoidable.

---

## 6. Verification

The suite is self-isolating (see Phase 0) and safe to run anywhere. Anything that needs a *server*
runs against the **sandboxed instance**, never the live one — an isolated copy with its own DB, data
tree, and port, reusable at `:8002`.

```bash
pytest -m "not slow"          # ~13s — logic, validation, auth, CSRF, framing
pytest                        # ~46s — adds real embedding, FAISS and reranking
python -m backend.eval --build-corpus   # once
python -m backend.eval --check          # labels still valid after editing them
python -m backend.eval                  # the number that gates retrieval changes
```

1. `pytest` — integration + unit suite green (66 tests as of Phase 0).
2. `python -m backend.eval` — hit@5 / MRR / hit@1 against the recorded baseline.
   **No retrieval change ships without a number.**
3. Manual end-to-end through the UI: sign up → upload a scanned contract, a `.docx`, and an
   `.xlsx` → ask an exact-term question ("what does Section 7.2 say?") → confirm the cited span
   highlights → confirm the answer appears in the audit log and re-runs identically.
4. Isolation re-check: a second account cannot read or retrieve the first's documents (extend the
   existing Mallory test to workspaces in Phase 5).
5. Confirm real data untouched after sandbox runs by comparing `data/scholar.db` checksum and file
   count.

---

## 7. Risks

- **Phase 5 storage migration** is the highest-risk step; it must not start before Phase 0's tests
  and Phase 4's backups exist.
- **OCR is slow and variable** — it belongs in the Phase 4 job queue, not the request path.
- **Scope discipline.** Glean and Hebbia have teams and enterprise connectors. Scholar should not
  chase connector breadth; the defensible position is depth of verifiability on documents the user
  uploads directly.
- **Eval set quality caps everything.** 30 sloppy labels produce confident, wrong conclusions.
  `--check` now guards against *mislabelled* spans, but it cannot guard against an *unrepresentative*
  corpus — and at 65 chunks the current one is unrepresentative. Growing it with distractor documents
  is the first task of Phase 2, before any hybrid-retrieval claim is measured.
