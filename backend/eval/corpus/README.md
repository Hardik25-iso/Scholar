# Eval corpus

Two documents. Neither is committed as a PDF — one is generated, one is fetched —
because binaries do not belong in the repo and the paper is not ours to
redistribute.

## `sample_agreement.pdf` — generated, no network needed

Built from `sample_agreement.txt` by `python -m backend.eval --build-corpus`.

A synthetic services agreement, written for this eval. It is **not** a real
contract and nothing in it is legal drafting worth copying. It exists because
the target user searches documents like this one, and because it contains the
query shapes dense embeddings are worst at: numbered sections (`Section 7.2`),
named clauses (`Force Majeure`), party names, dates, money, and a small fee
table. It is the measuring stick for Phase 2's hybrid retrieval.

## `rag_paper.pdf` — fetch it yourself

The RAG paper (Lewis et al., 2020), arXiv:2005.11401. Download it to this
directory as `rag_paper.pdf`:

```bash
curl -L -o backend/eval/corpus/rag_paper.pdf https://arxiv.org/pdf/2005.11401
```

The eval skips any question whose document is missing and says so in the report,
so the harness still runs with only the generated agreement.

## Adding documents

1. Put the source in this directory.
2. Add questions to `../dataset.json` with `"doc"` set to the file stem.
3. Run `python -m backend.eval --check` — it verifies every expected span really
   occurs in the extracted text, so a typo in a label fails loudly instead of
   silently scoring as a miss forever.
