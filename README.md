# StreamFlix RAG — Retrieval-Augmented Support with an Evaluation Harness

A question-answering system over a customer-support knowledge base, built
around the part that usually gets skipped: **measuring whether it actually
works.**

Assembling a RAG pipeline is a weekend of gluing libraries together. The
hard question is the one that follows — *is this good enough to put in
front of customers, and how would I know?* This project treats that
question as the deliverable. The retrieval and generation code exists to
give the evaluation harness something to measure.

> **Status: in progress.** Phases 1–2 of 7 are complete and tested.
> Phases 3–7 are not built yet. The roadmap below marks exactly where the
> line is. Nothing in this README describes results that don't exist.

---

## The central design decision

Every metric in this repo is scored against a **golden set of 120
questions with hand-labelled ground-truth sources**, written before any
retrieval code existed.

That ordering matters. A golden set written after you can see what your
system retrieves is a description of your system, not a test of it. The
questions here were written against the corpus alone, including 25
questions the corpus deliberately **cannot** answer — because a support
bot that confidently answers an unanswerable question is worse than one
that says "I don't know."

| Question type | n | What it tests |
|---|---:|---|
| `single_hop` | 60 | Direct factual lookup — one article answers it |
| `multi_hop` | 20 | Synthesis across 2–4 articles |
| `ambiguous` | 15 | Underspecified queries with multiple valid readings |
| `out_of_scope` | 25 | **Refusal behaviour** — the corpus is silent, the system must say so |

---

## What's built

### Phase 1 — Corpus & golden set ✅

A synthetic StreamFlix help centre: **45 articles** across 6 categories
(billing 10, devices 9, account 8, streaming 8, content 6, trial 4).

Synthetic on purpose. A real support corpus is either proprietary or so
clean it makes retrieval look easy. Writing it means the ground truth is
exact, and it means **known flaws can be planted deliberately**:

| Flaw | What it is |
|---|---|
| `contradiction-refund-window` | Two articles state different refund windows (30 vs 14 days) |
| `outdated-basic-plus` | A stale article describing a discontinued plan |
| `near-duplicate-device-setup` | Three device-setup articles with near-identical structure |
| `near-duplicate-cancellation` | **Discovered, not planted** — surfaced by similarity analysis |

That fourth one is worth a note. A test asserting the planted device
cluster was the most confusable pair in the corpus **failed**: an
unplanted pair (`bill-003` / `trial-004`, 32.1% Jaccard) beat it at 30.5%.
Rather than loosen the threshold to make the test pass, the emergent
duplicate was documented as a real corpus property and the original test
corrected to assert something true. Also catalogued: **8 coverage gaps**
— topics the corpus intentionally doesn't cover, which is what the
out-of-scope questions probe.

**Phase 1 established the floor.** Before building anything dense, BM25
(Okapi, k1=1.5, b=0.75, dependency-free implementation) was run against
the golden set:

| Category | BM25 recall@5 |
|---|---:|
| `single_hop` | 93.3% |
| `multi_hop` | 67.9% |
| `ambiguous` | 48.9% |
| **overall (in-scope)** | **81.0%** |

93% on single-hop from pure term matching reframed the whole project. The
interesting work is not "can dense retrieval beat random" — it's whether
it can beat a strong lexical baseline on the questions that are actually
hard. Everything downstream reports **per category**, because the
aggregate is dominated by the 60 single-hop questions and hides exactly
the cases that matter.

### Phase 2 — Chunking, embeddings, retrieval baseline ✅

Five chunking strategies, an embedding layer with two interchangeable
backends, and an exact-search vector store.

**Chunking is a real experiment here, not a default.** Phase 1 confirmed
every article fits inside a 512-token embedding window, so splitting is
never forced — which makes `whole_article` a legitimate arm rather than a
strawman.

| Strategy | Chunks | Median tokens |
|---|---:|---:|
| `whole_article` | 45 | 175 |
| `markdown_section` | 202 | 41 |
| `fixed_token_128` | 90 | 127 |
| ~~`fixed_token_256`~~ | ~~45~~ | — *dropped, see below* |
| `sentence_window` | 994 | 11 |

`fixed_token_256` **collapsed into `whole_article`** — byte-identical
output. The longest article is 215 tokens, so a 256-token window never
has anything to split. The chunker ran, did nothing, and would have
appeared in the Phase 3 bake-off as a distinct configuration with
mysteriously identical numbers. The notebook detects this dynamically and
drops the arm.

**Two embedding backends** behind one interface: BGE-small via
sentence-transformers, and a scikit-learn TF-IDF + SVD fallback. The
fallback isn't a stub — latent semantic analysis is a real dense-retrieval
method, and keeping it first-class means CI exercises the entire pipeline
in seconds with no model download, and a fresh clone works before anyone
installs torch. It also gives the Phase 3 bake-off a genuine progression:
**lexical → classic dense → modern dense.**

**The vector store is numpy, deliberately.** At 50–400 chunks, exact
cosine search is a single matrix multiply well under a millisecond. An
ANN index cannot beat that — it can only approximate it while adding a
dependency and contaminating retrieval measurements with index error. ANN
indexes earn their complexity somewhere north of ~100k vectors; the
rationale is written up in `src/retrieval/vectorstore.py` rather than left
implicit.

---

## Findings so far

**1. The untuned dense baseline loses to BM25.**

| Category | BM25 | Dense (LSA) | Δ |
|---|---:|---:|---:|
| `single_hop` | 93.3% | 86.7% | −6.7pp |
| `multi_hop` | 67.9% | 67.9% | −0.0pp |
| `ambiguous` | 48.9% | 43.3% | −5.6pp |
| **overall** | **81.0%** | **75.9%** | **−5.1pp** |

Reported rather than buried. This corpus uses consistent product
vocabulary and the questions reuse it, which is precisely the regime where
exact term matching is strong — a documented result in the IR literature,
not an anomaly. It is also *one untuned configuration* on the fallback
backend. The conclusion isn't "dense retrieval is bad"; it's that a
lexical baseline has to be **beaten, not assumed away**, and that shipping
the dense system on intuition would have made the product worse.

**2. One question defeats both retrievers — and it will look like a
hallucination.**

`mh-011` *("Charged after I cancelled — what do I do?")* returns 0%
recall@5 under dense retrieval and 50% under BM25. Neither surfaces both
sides of the planted refund contradiction. The query's dominant terms —
"charged", "cancelled" — saturate the free-trial articles, which crowd out
the refund policy that actually answers it.

Because the failure is **shared** across lexical and dense, it isn't a
"wrong retriever" problem, which rules out the easy fix. And downstream it
becomes dangerous: the generator will receive five plausible, on-topic,
*wrong* chunks and answer fluently from them. In evaluation this presents
as a **faithfulness failure** — the model said something unsupported —
when the fault is entirely upstream in retrieval. Attributing it to the
generator would aim the fix at the wrong component.

This is the clearest argument for the harness scoring retrieval and
generation **separately**. Context precision/recall isolate it; an
end-to-end answer-quality score would average it away.

---

## Roadmap

| Phase | Status |
|---|---|
| 1. Corpus + golden set | ✅ Complete |
| 2. Chunking, embeddings, retrieval baseline | ✅ Complete |
| 3. Retrieval bake-off — strategy × backend × top-k, BM25 as a first-class arm | Not started |
| 4. Generation layer — prompting, grounding, refusal behaviour | Not started |
| 5. Evaluation harness — faithfulness, answer relevancy, context precision/recall | Not started |
| 6. LLM-as-judge + failure analysis | Not started |
| 7. Decision memo + deployment recommendation | Not started |

---

## Running it

Phases 1–2 need **no API key and no model download**. This is enforced by
CI, which installs only the scientific stack and fails if a credential is
present.

```bash
pip install numpy pandas scikit-learn matplotlib pytest

python -c "from src.corpus.build import write_corpus, write_golden_set; write_corpus(); write_golden_set()"

python notebooks/01_corpus_construction.py    # corpus audit + BM25 floor
python notebooks/02_chunking_embedding.py     # chunking + retrieval baseline

pytest tests/ -q                              # 134 tests
```

For the transformer embedding arm and the later LLM phases, see
[`SETUP.md`](SETUP.md) — including how to configure a project-scoped key
with a spend cap.

Notebooks exist as paired `.py` and `.ipynb`. The `.py` is the source of
truth and what CI runs; the `.ipynb` is generated from it and committed
with outputs stripped.

---

## Security posture

This repo talks to a paid API, so credential handling is part of the
engineering rather than an afterthought.

- The API key is read from `os.environ` **at call time** in exactly one
  module (`src/llm/provider.py`). It is never stored on an object, passed
  as an argument, or written to disk.
- **16 dedicated security tests** assert the key cannot escape through the
  paths that leak credentials in practice: object `repr`, string
  formatting, pickled state, and cache filenames.
- One of those tests initially passed in a clean environment but failed
  locally, because `load_dotenv()` repopulated a deliberately-deleted
  variable. The test was non-hermetic — it was fixed, and a regression
  guard added so the suite's result no longer depends on whether a local
  `.env` exists.
- `.env` is gitignored; `.env.example` carries placeholders only.
- CI runs `detect-secrets` against a committed baseline and fails if
  `.env` ever becomes tracked.
- Cost controls are in code, not discipline: a hard per-run call ceiling
  (`MAX_LLM_CALLS_PER_RUN`) that raises rather than warns, and a response
  cache keyed on model + prompt + params — never on the key.

---

## Layout

```
src/
  corpus/      seed articles, golden set, corpus build + BM25 difficulty analysis
  retrieval/   chunking strategies, embedding backends, vector store
  llm/         provider abstraction, credential masking, call budget, response cache
  generation/  (Phase 4)
  evaluation/  (Phase 5)
notebooks/     01 corpus construction · 02 chunking + embedding
tests/         134 tests — corpus, difficulty, chunking, retrieval, provider security
data/          generated corpus + golden set (regenerable)
reports/       figures
```

---

## Related projects

Part of a three-project series sharing the StreamFlix universe:

- **[A/B Test Analysis](https://github.com/janeruxi1/ab-testing-project)** — experiment design, sequential testing, CUPED, sensitivity analysis
- **[Churn & Retention](https://github.com/janeruxi1/StreamFlix-churn-retention)** — churn modelling, causal uplift, cost-aware targeting policy
