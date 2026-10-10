# Deploy measurements: Scholar on a 512 MB / 0.1 CPU instance

Every limit in [`render.yaml`](../../render.yaml) is set from a number in this directory. Raw
outputs sit beside this file: `eval-*.json|txt` from `python -m backend.eval --json-out`, and
`probe-*.json` from [`scripts/rss_probe.py`](../../scripts/rss_probe.py).

## How it was measured

- **Quality:** `python -m backend.eval --expansion none --no-mmr`. That's 51 questions over the
  corpus, with HyDE off because it is not deterministic.
  [`scripts/eval_diff.py`](../../scripts/eval_diff.py) compares two runs question by question.
  The bar is a hit@5 drop of at most 1 point, and with this many questions that means **no
  question may go from hit to miss**.
- **RAM and latency:** the real image, run as
  `docker run --memory=512m --memory-swap=512m --cpus=0.1`. That's Render's free tier with no
  swap, and the CPU quota as the worst case.
  - The probe drives the container over HTTP and answers generation with a fake LLM, so `/ask`
    timings are our own CPU.
  - Memory comes from `docker stats`, which is cgroup accounting and the number the OOM killer
    acts on, plus the uvicorn process's `VmHWM`.
  - Host: Ryzen 7 6800H, Docker Desktop (WSL2).

## Retrieval quality

| Arm | hit@5 | MRR | hit@1 | Questions lost vs A |
|---|---|---|---|---|
| A: torch (sentence-transformers 3.0.1) | 94.1% | 0.822 | 72.5% | — |
| B: ONNX fp32 | 94.1% | 0.822 | 72.5% | none (identical, question by question) |
| C: ONNX int8, batch 8 | 94.1% | 0.825 | 72.5% | none |
| **C: ONNX int8, batch 1 (ships)** | **94.1%** | **0.835** | **74.5%** | **none** |
| int8, rerank 10 candidates | 92.2% | 0.843 | 76.5% | `cmp-01` |
| int8, rerank input ≤ 256 tokens | 88.2% | 0.743 | 64.7% | `msa-02`, `hop-02`, `cmp-01` |
| int8, both of the above | 90.2% | 0.757 | 64.7% | `hop-02`, `cmp-01` |
| int8, 12 candidates × 256 tokens | 88.2% | 0.749 | 64.7% | `msa-02`, `hop-02`, `cmp-01` |
| **int8, no reranker (free tier)** | **92.2%** | **0.772** | **66.7%** | `rag-03`, `rag-08`, `cmp-01` (gains `msa-06`, `cmp-02`) |

int8 drifts noticeably at the vector level against torch. On real corpus text the embedding cosine
is 0.84 at worst and 0.91 at the median, and the cross-encoder's rank correlation is 0.86 at worst.
None of that reaches the ranking: arm C loses nothing. Dynamic quantisation computes its scales per
call, padding included, which is why batch 1 scores slightly *better* than batch 8.

## Memory and speed under the caps

| | torch baseline (2 GB, 1 CPU) | int8, 0.1 CPU, as first built | **int8, free-tier config** |
|---|---|---|---|
| survives 512 MB | no: peak 1,395 MiB | **OOM-killed at warm-up** | yes |
| cgroup peak | 1,395 MiB | > 512 MiB | **386 MiB** |
| process peak RSS | — | — | 451 MiB (anon ~383; the rest is shared libraries) |
| `/ask` p50 / p95 | 4.0 / 4.2 s | 31.8 s (reranker on) | **0.45 / 0.58 s** |
| index the Attention paper (15 pp, 36 chunks) | 70 s | 323 s | **183 s** |
| index 30 dense pages (30 chunks) | — | — | 174 s |
| a 31-page PDF (cap 30) | — | — | refused in 0.1 s, before parsing |
| boot until models ready | 13 s | 142 s | 41 s |
| first question after a cold restart | — | — | 8.6 s, then 0.8 s |

Each change from "as first built" to the free-tier configuration, and the measurement behind it:

1. **No weight pre-packing** (`session.disable_prepacking`). ONNX Runtime keeps a second,
   re-laid-out copy of every quantised matrix. The mpnet session drops from +218 MB to +122 MB at
   about 10% inference speed. Without this, the process was OOM-killed during warm-up.
2. **Batch size 1 for both models.** Per call, measured on one thread with full-length inputs:

   | Model | Batch 1 | Batch 8 |
   |---|---|---|
   | embedder | 793 ms at +21 MB | 1,253 ms at +360 MB |
   | reranker | 140 ms/pair at +25 MB | 209 ms/pair at +236 MB |

   On one core, batching only adds padding and cache pressure.
3. **Models built once, under a lock.** `lru_cache` doesn't lock, so the start-up warm-up and the
   first request could each build a 120 MB session at the same time. Fixing that took peak process
   RSS from 513 to 451 MiB and boot from 62 s to 41 s.
4. **`malloc_trim` after each indexing run.** Without it, glibc keeps a document's freed
   allocations and the *next* document is the one that runs out (anon 276 → 353 MB during
   embedding, back to 241 MB after a trim).
5. **No reranker on the free tier.** One full-length 384-token inference has an inherent +67 MB
   transient, and that's fine. The reranker is the real problem: about 1.4 s of CPU per question,
   which 0.1 CPU turns into about 30 s. Every cheaper way of running it lost more quality than
   switching it off (see the quality table). It is one setting, `RERANK_ENABLED`, so any instance
   with ≥ 1 GB and a whole CPU gets the full 94.1% / 0.835 pipeline back.

## How the caps were derived

- **About 5 s of CPU per passage at 0.1 CPU** (183 s for 36 passages, with parsing).
- The hard constraint is Render's 15-minute idle window. Polling keeps the server awake only while
  the tab is open, so a job must also finish if the user closes it. `MAX_CHUNKS_PER_DOC=60` is
  about 5 minutes; `MAX_PAGES=30` refuses large PDFs cheaply, before parsing.
- The original 5-minute target holds for any document within the caps.
- `MAX_UPLOAD_MB=10` bounds what is received; the page and passage caps are what actually bound
  the work.

## Not measured here

- Render's real CPU. Render may let a free instance burst above 0.1 CPU; nothing promises it, so
  every number here assumes it doesn't.
- Supabase and Neon latency from Render's region. The probe ran storage locally against a local
  Postgres.
- A cold start that fetches the index from Supabase.
