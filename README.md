# Scholar

**Auditable answers over private documents.** Upload contracts, reports and papers; ask questions in
plain language; get answers assembled *only* from passages retrieved out of your own files, each one
cited to the exact page and character span it came from.

Every answer is logged with the evidence that produced it, so it can be pulled back up months later
and checked — or shown to be no longer reproducible, and told so plainly.

> Built as a study of retrieval-augmented generation done properly: not "wire an LLM to a vector
> store", but the parts that decide whether anyone can trust the output — layout-aware parsing, hybrid
> retrieval, reranking, span-level citation, and an audit trail. Every retrieval change in this repo
> shipped with a before/after number from the eval harness, and several plausible ideas were measured
> and **rejected**.

---

## Contents

- [What it does](#what-it-does)
- [Quickstart](#quickstart)
- [How it works](#how-it-works)
- [Results](#results)
- [What was measured and rejected](#what-was-measured-and-rejected)
- [Testing](#testing)
- [Deployment](#deployment)
- [Known limits](#known-limits)

---

## What it does

| | |
|---|---|
| **Reads real documents** | PDF (including **scanned**, via OCR), `.docx`, `.pptx`, `.xlsx`, `.txt`, `.md`. Tables are serialised to Markdown so the model sees labelled columns instead of loose numbers. |
| **Finds exact terms** | Hybrid retrieval — dense embeddings **and** SQLite FTS5 BM25, fused with Reciprocal Rank Fusion. "Section 7.2" and "Force Majeure" are exactly where dense-only search is weakest. |
| **Cites precisely** | Every citation carries the document, page/slide/sheet, character span, FAISS vector id, and both ranking scores. |
| **Proves itself** | Every answer is written to an audit log with its full evidence chain, model, parameters and an index fingerprint — and is exportable as JSON/CSV. |
| **Works for teams** | Workspaces with membership, roles and email invitations. A non-member gets a 404, not a 403. |
| **Refuses honestly** | When the corpus does not contain the answer, it says so instead of inventing one. |
| **Yours to take back** | Export everything — documents, answers, evidence — as a `.tar.gz`. Delete the account and the files come off disk, not just out of the database. |

---

## Quickstart

Requires Python 3.11+, Node 18+, and [Ollama](https://ollama.com) for generation.

```bash
git clone https://github.com/Hardik25-iso/Scholar.git && cd Scholar
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Set `SECRET_KEY` in `.env` — it is the one value with no default, on purpose:

```bash
python -c "import secrets; print(secrets.token_urlsafe(48))"
```

Pull the generation model (Gemma 3 4B — set in `backend/generator.py`), then start the API and the
frontend:

```bash
ollama pull gemma3:4b
```

```bash
uvicorn backend.api:app --port 8001
```

```bash
cd frontend && npm install && npm run dev
```

Open http://localhost:5173, create an account, drop in a document, and ask it something.

**Optional — the indexing worker.** Without it, uploads are indexed inside the request, which is fine
locally and races the proxy timeout on a large document in production. With `REDIS_URL` set, run:

```bash
arq backend.jobs.WorkerSettings
```

---

## How it works

Two phases. **Indexing** happens once per document; **querying** happens per question.

```
INDEX   file → extract (layout-aware, OCR fallback) → chunk (token-exact)
              → embed → FAISS + FTS5, per workspace

QUERY   question → condense follow-ups → expand (HyDE)
              → dense search ─┐
                              ├─ Reciprocal Rank Fusion → rerank → generate
              → BM25 search ──┘        (cross-encoder)     (cited)
```

| Stage | Choice | Why |
|---|---|---|
| Chunking | 380 content tokens + 50 overlap, via tokenizer offset mapping | The embedder's limit is 384 including special tokens. Counting characters and hoping is how tails get silently dropped. |
| Embedding | `all-mpnet-base-v2` (768-d), cosine via normalised `IndexFlatIP` | — |
| Lexical | **SQLite FTS5** with native `bm25()` | Zero new dependencies. Terms are quoted so `7.2` survives tokenisation rather than splitting into `7 OR 2`. |
| Fusion | RRF, K=60 | Rewards passages *both* retrievers found — it prefers one strong opinion to uniform mediocrity. |
| Reranking | `ms-marco-MiniLM-L-6-v2` cross-encoder, 20 → 5 | Measured: promotes 9, demotes 4, loses 2. Net strongly positive. |
| Expansion | HyDE (hypothetical answer, temperature 0) | Measured below. Recorded in the audit log, because it makes retrieval non-deterministic. |

```
backend/
  api.py         routes + the answer pipeline      indexing.py   the indexing job
  parser.py      multi-format extraction, OCR      jobs.py       arq queue + worker
  chunker.py     token-exact splitting             lexical.py    FTS5 index
  retriever.py   dense search                      fusion.py     RRF
  reranker.py    cross-encoder                     expansion.py  HyDE / PRF
  workspaces.py  the authorisation rule            audit.py      the evidence log
  eval/          the measurement harness
frontend/src/    React + Vite + Tailwind
docs/            architecture, ML notes, and the full measured roadmap
```

---

## Results

All numbers come from `python -m backend.eval` — **51 hand-labelled questions over 7 documents**
(contracts, a financial brief, a two-column market report, an academic paper), tagged by failure
mode: `exact` 19, `semantic` 11, `layout` 7, `comparative` 5, `table` 4, `multihop` 3,
`vocabulary` 2. Every labelled span is verified to occur in the extracted text (`--check`), because a
mislabelled span scores as a permanent miss and quietly drags the baseline down forever.

### Where each retriever gets you

Printed by a single `python -m backend.eval` run, so all three rows are the same questions against
the same index — not three experiments stitched together.

| Retriever | recall (hit@20) | MRR |
|---|---|---|
| dense only | 98.0% | 0.672 |
| lexical only (BM25) | 96.1% | 0.803 |
| **hybrid (RRF)** | **100.0%** | 0.793 |

**Stage-1 recall reaching 100% is the headline**: the evidence for every question reaches the
shortlist, which moves the ceiling from retrieval onto ranking. After reranking to the top 5:
**hit@5 98.0%, MRR 0.805, hit@1 70.6%.**

Two things here are worth stating plainly rather than smoothing over:

- **Lexical alone outranks dense alone** (MRR 0.803 vs 0.672). On professional documents — clause
  numbers, party names, dollar amounts — BM25 is not the weak leg of a hybrid system, and a
  vectors-only design would be leaving the better ranker out.
- **Fusion's MRR (0.793) sits just below lexical's (0.803)** while its recall is higher. That is the
  trade RRF makes, and it is the right one here: a passage that never reaches the shortlist cannot be
  recovered by the reranker, whereas a slightly worse rank still lands in front of it.

### Query expansion

| | none | prf | **hyde** |
|---|---|---|---|
| **hit@5** | 94.1% | 92.2% | **98.0%** |
| MRR | **0.822** | 0.771 | 0.805 |
| hit@1 | **72.5%** | 66.7% | 70.6% |
| misses | 3 | 4 | **1** |
| `vocabulary` (the target class) | 0% | 0% | **50%** |
| `comparative` | 80% | 60% | **100%** |

HyDE is the default. **The cost is real and not waved past**: it trades ranking precision for recall
(MRR 0.822 → 0.805, hit@1 72.5% → 70.6%), and costs one extra LLM call per question. Recall was the
more valuable side here — a passage that never reaches the shortlist cannot be recovered by
reranking, while a slightly worse rank still lands on screen.

It is only an *honest* default because the audit log records the generated hypothetical. Retrieval is
no longer purely deterministic, so without that column there would be no way to tell a changed
library from a differently-worded hypothetical.

### End to end

The naive configuration against the shipped one, both measured on the same 51 questions:

| | dense only, no expansion | **shipped (hybrid + HyDE)** |
|---|---|---|
| questions whose evidence never reached the model | **4** | **1** |
| `semantic` hit@5 | 90.9% | **100%** |
| `comparative` hit@5 | 80% | **100%** |
| `vocabulary` hit@5 | 0% | **50%** |

The one remaining miss is `msa-06` — *"What is the uptime commitment and what happens if it is
missed?"* against a contract that expresses it as a service-credit table. It is a vocabulary gap, it
is known, and it is in the eval so that any future change to retrieval has to face it.

### Test suite

**310 tests**, no network and no running server required. The isolation guard asserts that storage
has been redirected to a temp tree *before* importing the app — if a future change breaks it, the
suite refuses to run rather than writing into a real library.

---

## What was measured and rejected

The eval harness repeatedly overturned plausible-sounding plans. These are kept because a negative
result you can reproduce is worth more than a good idea you never tested.

| Idea | Verdict |
|---|---|
| **PRF / RM3 expansion** | **Rejected.** Worse on every metric, and it did not move the class it targeted. It re-weights terms from the top results — when retrieval already missed, it amplifies the miss. Structurally circular, not a tuning problem. |
| **`page.get_text(sort=True)`** for reading order | **Rejected on measurement.** It *destroyed* two-column reading order on the test corpus: 5/5 labelled spans preserved without it, 0/5 with it. |
| **PyMuPDF `find_tables()`** | Unusable on the real documents tested; a hand-rolled column serialiser did better. |
| **"The reranker is the bottleneck"** | **Wrong.** It came from reading an aggregate. Per question the reranker promotes 9, demotes 4, loses 2 — both losses are vocabulary cases where the gold passage genuinely looks nothing like the question. |
| **An agentic research layer** | **Built nothing.** Measurement gated it: within-document multi-hop already scored 100%, and the only 0% class (`vocabulary`) is not one an agent fixes. Query expansion closed that gap instead. |

One labelled question turned out not to be multi-hop at all — both required spans lived in the same
chunk, so it was measuring a synonym gap while wearing a multi-hop label. The harness now flags
attainment above 100% precisely because that means the labelled parts shared a passage.

---

## Testing

```bash
python -m pytest backend/tests -q
```

The eval corpus is **generated from committed `.txt` sources** rather than checked in as binaries, so
build it once first:

```bash
python -m backend.eval --build-corpus
```

```bash
python -m backend.eval
```

```bash
cd frontend && npm run build
```

`--check` validates every label against the extracted text without loading a model;
`--expansion none|prf|hyde` and `--dense-only` reproduce every comparison in this README.

---

## Deployment

Three things separate a working local install from something you can put in front of users:

1. **`DATA_ROOT` must point at a mounted volume.** The default lives inside the source tree, so a
   redeploy that replaces the source destroys every library. Backup and restore are implemented and
   tested (`backend/backup.py`).
2. **`SMTP_HOST` must be set.** Without it, password-reset and invitation tokens are written to the
   log — workable on a laptop, an account-takeover vector on a server. The app says so at WARNING
   level every time.
3. **`REDIS_URL` plus the arq worker.** Without them uploads index inside the request. The app falls
   back rather than dropping work, and records `ran_inline` on the job so a deployment missing its
   worker is visible rather than mysterious.

Sessions use short-lived access tokens with rotating, revocable refresh tokens: presenting a spent
refresh token is treated as theft and revokes every session for that account.

See [`.env.example`](.env.example) for every setting.

### Before you serve strangers

`/privacy` and `/terms` ship with the app and are written from the code — but they describe *what the
software does*, and are **not a lawyer-reviewed policy**. Running this as a public or paid service
also needs your operating entity, a contact address, governing law and jurisdiction, your
sub-processors, and a lawful basis if you serve users in the EU or UK. Both pages carry a visible
note saying exactly this.

---

## Known limits

- **Generation runs locally through Ollama** (Gemma 3 4B). Answer quality is bounded by the local
  model; retrieval quality — every number above — is measured independently of it.
- **The eval set is 51 questions over 7 documents.** Large enough to catch regressions, small enough
  that a single question moves a percentage point. Treat deltas, not absolutes, as the signal.
- **OCR needs the Tesseract binary.** Without it a scanned PDF fails with a message naming the
  server's missing capability rather than blaming the file.
- **Cross-document comparison is the weakest class** — 80% at k=5 without expansion. It is measured,
  not ignored.
- **The legal pages are a starting point, not advice** — see "Before you serve strangers" above.
- The AI assists; it never replaces reading the source. That is what the citations are for.

---

## Documentation

| | |
|---|---|
| [`docs/01-architecture-and-pipeline.md`](docs/01-architecture-and-pipeline.md) | How the pipeline fits together |
| [`docs/02-ml-dl-nlp-concepts.md`](docs/02-ml-dl-nlp-concepts.md) | The ML/NLP concepts, from the ground up |
| [`docs/03-engineering-auth-frontend.md`](docs/03-engineering-auth-frontend.md) | Auth, API and frontend engineering |
| [`docs/04-pmf-roadmap.md`](docs/04-pmf-roadmap.md) | The full roadmap, with every measurement and every rejected idea |
