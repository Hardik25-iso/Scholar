# Scholar — The ML / DL / NLP / AI Concepts

> Every machine-learning idea Scholar uses, explained from the ground up, with
> the *why* — so you can defend each design decision in an interview.

Scholar is a **RAG** system. To understand it you need six ideas: embeddings,
transformers, tokenization/chunking, vector search, the bi-encoder/cross-encoder
split, and grounded LLM generation. This doc builds them up in order, then ties
them together.

---

## 1. The core idea: RAG (Retrieval-Augmented Generation)

An LLM has two possible sources of knowledge:

| Knowledge type | Where it lives | Example |
|----------------|----------------|---------|
| **Parametric** | baked into the model's weights during training | "Paris is the capital of France" |
| **Non-parametric** | supplied at query time, in the prompt | the PDF you uploaded 2 minutes ago |

A plain chatbot answers from **parametric** knowledge — which is frozen at
training time, can't cite anything, and hallucinates confidently when it doesn't
know. **RAG** flips this: at query time you **retrieve** relevant passages from an
external store and put them in the prompt, and the LLM answers from **those**.

> **The key insight:** RAG injects fresh knowledge at *inference* time instead of
> baking it into weights via *training*. That's why Scholar can answer about a
> paper it has never seen — and why we never train or fine-tune anything.

**Why not just fine-tune the model on the papers?** Because you'd have to retrain
every time a user uploads a new PDF — absurd. Retrieval solves *"what does this
document say"*; the pretrained LLM supplies the general skill of *"read English,
reason, write a grounded answer"*. Two different problems, two different tools.

The **"R"** (retrieval) is docs 1 §4 steps ③–⑤. The **"G"** (generation) is step ⑥.

---

## 2. Embeddings — turning meaning into geometry

An **embedding** is a fixed-length list of numbers (a **vector**) that represents
the *meaning* of a piece of text. Scholar uses `all-mpnet-base-v2`, which produces
a **768-dimensional** vector for any sentence/passage.

The magic property: **texts with similar meaning get vectors that are close
together**, even with zero shared words.
- "How does attention work?" and "the mechanism computes weighted sums of values"
  land near each other.
- "How does attention work?" and "the price of tea in China" land far apart.

So meaning becomes **geometry**: similarity = distance in a 768-dimensional space.
This is what makes *semantic* search possible — you can find the right passage by
meaning, not by keyword matching.

**How does the model learn this?** `all-mpnet-base-v2` was trained with
**contrastive learning** on ~1 billion sentence pairs: pull paraphrases together,
push unrelated sentences apart. After training, the geometry encodes semantics.

**Interview line:** *"An embedding maps text into a vector space where cosine
distance approximates semantic similarity; retrieval is then just nearest-neighbour
search in that space."*

---

## 3. Transformers & tokenization (what's under the embedder AND the LLM)

Both the embedder (mpnet) and the generator (Gemma) are **Transformers** — the
architecture from *Attention Is All You Need* (the very paper Scholar was first
tested on).

### 3.1 Attention, in one paragraph
A Transformer reads a sequence of tokens and, at every layer, lets each token
**attend to** every other token — computing how much each other token matters to
it, then mixing their representations accordingly. Stacked over many layers, this
builds a rich, context-aware representation of the text. "Multi-head" attention
just does this several times in parallel, each head learning to focus on a
different kind of relationship.

- **mpnet** uses the Transformer as an **encoder**: read text → one summary vector.
- **Gemma** uses it as a **decoder**: read text → generate the next token, over
  and over (see §7).

### 3.2 Tokenization — models don't read characters, they read tokens
Text is first split into **tokens** (word-pieces) by a tokenizer with a fixed
vocabulary. "tokenization" might become `token` + `##ization`. Models operate on
these sub-word units.

### 3.3 The chunking problem (a real, subtle bug class)
mpnet can only process **384 tokens** at once; anything longer is silently
**truncated** — you'd embed only the first part of a long passage and never know.
So Scholar's `chunker.py`:
1. Cuts each page into windows of **380 content tokens** (+2 special = 382 ≤ 384).
2. Overlaps windows by **50 tokens** so a fact spanning a boundary appears whole in
   at least one chunk.
3. Counts **real word-piece tokens**, not words — because words are a lying proxy
   (one "word" can be several tokens). Counting the model's own tokens *guarantees*
   the fit.
4. Uses the tokenizer's **offset mapping** to slice the **verbatim** original text
   for each window (preserving exact casing/spacing for citations).

> **Interview angle:** this is the same failure class as *any* silent truncation
> past a model limit — you must measure in the model's own units and fail loudly,
> never approximate. (The auth layer does the identical thing with bcrypt's
> 72-byte limit — see doc 03.)

**Chunk size is a tradeoff:** too small → context gets fragmented and answers lose
coherence; too large → each chunk dilutes to an average meaning and retrieval gets
fuzzy (and you hit the token limit). ~380 tokens with 13% overlap is a sensible
middle for research prose.

---

## 4. Vector search — cosine similarity & FAISS

Once every chunk is a 768-dim vector, "find the relevant passage" becomes "find
the nearest vectors to the query vector". Two pieces:

### 4.1 Cosine similarity via normalized inner product
**Cosine similarity** measures the *angle* between two vectors, ignoring their
length — perfect for meaning, where direction matters and magnitude doesn't. Its
value runs from -1 (opposite) to 1 (identical direction).

Trick used in `embedder.py`: if you **L2-normalise** every vector (scale it to
length 1), then the **inner (dot) product** of two vectors *equals* their cosine
similarity. So you get cosine for free from the cheaper dot-product operation.

### 4.2 FAISS `IndexFlatIP`
**FAISS** is a library for fast similarity search over vectors. `IndexFlatIP`:
- **Flat** = exact, brute-force — it really does compare against every vector (no
  approximation). Fine up to ~100k chunks; a research library is far below that.
- **IP** = inner product — which, with our normalized vectors, is cosine similarity.

`index.search(query_vec, k)` returns the `k` nearest chunks and their scores. This
is **k-nearest-neighbours (kNN)** search — a classic, *non-learned* algorithm.
FAISS is math and bookkeeping, **not** a model.

> **Interview line:** *"Retrieval is exact kNN by cosine similarity; I normalize
> vectors so FAISS's inner-product index gives cosine directly."*

---

## 5. Why one stage isn't enough — the bi-encoder's blind spot

The retriever is a **bi-encoder**: it embeds the query and each chunk
**independently**, then compares the two finished vectors. This is *fast* (chunk
vectors are precomputed; only the query is embedded at query time) but
**approximate** — the two encodings never actually "look at" each other, so the
model can rank a superficially-similar chunk above the truly relevant one.

In practice you'll see the genuinely-best passage land at rank 4 while a chunk of
boilerplate ranks 1. Good enough to get the answer *into the candidate set*, not
good enough to be the final ranking.

---

## 6. The fix: cross-encoders & retrieve-then-rerank

A **cross-encoder** (`ms-marco-MiniLM-L-6-v2`) scores a `(question, chunk)` pair by
feeding them **together** through one Transformer: `[question] [SEP] [chunk]`. Now
every word of the question can attend to every word of the chunk, so it judges
*true* relevance — it can notice "the question asks *why* single-head is worse and
this chunk literally says *single-head averaging inhibits this*".

The catch: a cross-encoder must run **once per candidate** (you can't precompute
its scores, because they depend on the specific question). That's far too slow to
run over the whole index — but perfectly fine over a shortlist.

Hence the **two-stage / retrieve-then-rerank** pattern:

| Stage | Model | Role | Speed | Runs over |
|-------|-------|------|-------|-----------|
| 1 | bi-encoder (mpnet) | **recall** — cast a wide net | fast | the whole index → top 20 |
| 2 | cross-encoder | **precision** — sharpen the ranking | slow | just the 20 → top 5 |

> **Interview line:** *"Bi-encoder for recall, cross-encoder for precision, run the
> expensive model only on the shortlist. Classic retrieve-then-rerank."*

The cross-encoder's scores are unbounded logits (roughly -4…+8 in practice);
Scholar surfaces them in the Sources panel as relevance meters. Positive ≈ genuinely
relevant, negative ≈ not — a signal the raw cosine can't give you.

**k = 5** (chunks fed to the LLM) and **20 candidates** are tunable knobs: more
candidates = better chance of catching the right chunk but slower reranking; larger
k = more context for the LLM but more prompt noise and cost.

---

## 7. Generation — the local LLM writes the answer

The generator is **Gemma 3 4B** run locally via **Ollama**. Concepts:

### 7.1 Autoregressive generation
An LLM generates text **one token at a time**: given everything so far, predict the
next token, append it, repeat. This is why answers can be **streamed** — each token
is available the instant it's produced (doc 03 §3).

### 7.2 "4B parameters" and quantization
Gemma 3 4B has **4 billion** learned weights. At full precision that's ~16 GB —
too big for a laptop GPU. **Quantization** stores each weight in ~4 bits instead of
16, shrinking the model to ~3 GB (fits a 4 GB GPU) at a small, usually-unnoticeable
accuracy cost. Like saving a photo at lower quality: much smaller, still clearly the
same picture.

### 7.3 Temperature = 0
**Temperature** controls randomness in token selection. High temperature = more
creative/varied; **0 = always pick the most likely token** = deterministic and
faithful. For grounded QA you want faithfulness to the sources, not creativity, so
Scholar uses `temperature=0`.

### 7.4 Pretrained, remote-free, swappable
Gemma is used **as-is** (no fine-tuning), runs **entirely on your machine** (no API,
no cost, private), and is isolated behind one `MODEL` constant — swapping to another
local model, or a hosted one, is a one-line change that touches nothing else.

---

## 8. Grounding & hallucination control

A raw LLM will happily invent a plausible answer. Scholar prevents this with a
**grounding contract** encoded in the system prompt (`generator.py`):

1. Answer **using only** the numbered sources provided.
2. **Cite** every claim by source number `[n]`.
3. If the sources don't contain the answer, reply *exactly* "The provided papers do
   not contain the answer to this question." — an explicit **refusal**, not a guess.
4. Don't invent citations.

This is what converts a general LLM into a system that can only speak from your
papers. Combined with `temperature=0`, it keeps answers faithful and traceable.

**Why citations matter beyond looking nice:** grounding is only *trustworthy* if
it's *verifiable*. "[2]" as text is meaningless unless the user can click it and
read exactly what [2] says, on which page. The citation → source → PDF-page chain is
what separates a retrieval system from a chatbot.

**Honest limitation:** grounding constrains the LLM but can't force perfection — a
4B model can still occasionally paraphrase loosely or mis-cite. Retrieval quality
also bounds everything: if the right passage never makes the top-5, no prompt can
save the answer. That's why the two-stage retrieval (recall + precision) matters so
much — it's the real ceiling on answer quality.

---

## 9. Conversation memory — contextual query rewriting

Follow-ups like "What about the decoder?" or "why is it better?" are **meaningless
to a retriever** — "it"/"the decoder" only make sense given the prior conversation.
Embedding them directly retrieves garbage.

Scholar's fix (`generator.condense_question`): before retrieval, use the LLM to
**rewrite the follow-up into a standalone question** using the last few turns of
history:
- history: *"How does multi-head attention work?"* → *(answer)*
- follow-up: *"why is it better than a single head?"*
- **condensed:** *"Why is using multiple attention heads better than a single head?"*

Then that standalone query goes through the normal embed → retrieve → rerank path.
Two design safeguards:
- **No history → no-op** (returns the question unchanged), so it can never make a
  first question worse.
- Only the **last few turns** are used, keeping the extra LLM call fast.

This is the standard RAG technique variously called **query condensation /
contextualization / history-aware retrieval**. It's why Scholar feels like a
conversation rather than a series of unrelated searches.

---

## 10. Concepts NOT built (and what they'd be)

Being honest about the boundary:

| Idea | What it is | Status |
|------|-----------|--------|
| **Hybrid / BM25 search** | combine keyword (lexical) + vector (semantic) retrieval | not built — pure vector today |
| **Query expansion / HyDE** | generate a hypothetical answer, embed *that* to retrieve | not built |
| **RAGAS evaluation** | metrics for faithfulness / answer-relevance / context-precision | not built |
| **Fine-tuning the embedder** | adapt mpnet to a narrow domain if retrieval is weak | not needed at this scale |

None of these change the core: **retrieve → rerank → grounded generate**. They're
quality/eval refinements on top.

---

## Cheat-sheet (say these out loud)

- *"RAG injects knowledge at inference via retrieval instead of baking it into
  weights via training — that's why I never fine-tune."*
- *"Embeddings turn meaning into geometry; retrieval is kNN by cosine similarity,
  which I get from a normalized inner-product FAISS index."*
- *"Bi-encoder for recall, cross-encoder for precision — retrieve-then-rerank."*
- *"Grounding = a strict prompt contract (only these sources, cite by number,
  refuse otherwise) plus temperature 0; verifiable via page-level citations."*
- *"Follow-ups are condensed into standalone queries before retrieval, so context
  survives without confusing the vector search."*
