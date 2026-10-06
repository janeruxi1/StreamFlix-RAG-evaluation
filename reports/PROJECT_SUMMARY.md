# Project Summary — StreamFlix RAG Evaluation

**What was built + what was found.** Single-page catalog of the project's
components and headline results. Every number here is verified against
live notebook output by `notebooks/07_decision_memo.py`, which CI runs
and which fails the build on drift.

---

## 60-second overview

A question-answering system over a customer-support knowledge base, built
around the part that usually gets skipped: **measuring whether it
actually works.**

Assembling a RAG pipeline is a weekend of gluing libraries together. The
hard question is the one that follows — *is this good enough to put in
front of customers, and how would I know?* This project treats that
question as the deliverable; the retrieval and generation code exists to
give the evaluation harness something to measure.

**Headline outcome.** Ship the retrieval layer (**0.882** context recall
at ~**591 context tokens**/query). Hold the generation layer — it is
built, tested, and **unmeasured**, because no LLM arm has run. The split
is the recommendation, not a hedge.

---

## The design decision everything rests on

A **golden set of 120 questions with hand-labelled ground-truth sources**,
written *before* any retrieval code existed. A golden set written
afterwards describes your system rather than testing it.

| Question type | n | What it tests |
|---|---:|---|
| `single_hop` | 60 | Direct factual lookup |
| `multi_hop` | 20 | Synthesis across 2–4 articles |
| `ambiguous` | 15 | Underspecified queries |
| `out_of_scope` | 25 | **Refusal** — the corpus is silent, the system must say so |

Those 25 unanswerable questions are the deployment risk. A support bot
that confidently answers one is worse than one that says "I don't know."

---

## Headline numbers

| Metric | Value | Read against |
|---|---:|---|
| context recall | **0.882** | — |
| context precision | **0.211** | ceiling **0.441** → 48% of what's structurally possible |
| context cost | **~591 tokens** | 4.5× cheaper than the top-of-table config |
| extractive baseline, in-scope answer rate | **62.1%** | a floor for non-LLM methods |
| extractive baseline, out-of-scope refusal | **56.0%** | — |
| judge accuracy on the validation gate | **50%** | 80% required — **fails** |
| judge length bias | **−0.521** | material on 3/3 probes |

---

## What each phase produced

| Phase | Built | Key finding |
|---|---|---|
| 1 | 45-article corpus + 120-question golden set | BM25 gets **93.3%** recall@5 on single-hop — the floor is high, so per-category reporting is mandatory |
| 2 | 5 chunking strategies, 2 embedding backends, vector store | `fixed_token_256` was a silent no-op: the same words as `whole_article`, one chunk per article |
| 3 | 48-configuration bake-off | recall@k is monotone in k, so **no quality metric can select depth** — it's a cost decision |
| 4 | 5 prompt variants + extractive baseline + refusal detection | safety and helpfulness move against each other; every variant scored as a *pair* |
| 5 | Evaluation harness with a validated judge | context metrics are **computed, not judged** — ground truth exists, so using an LLM would be strictly worse |
| 6 | Judge bias audit | the audited judge **penalises** length by −0.521, the opposite of the documented LLM-judge bias |
| 7 | Decision memo + verification | 14 memo claims recomputed; CI fails on drift |

---

## Five findings worth the reader's time

**1. A lexical baseline had to be beaten, not assumed away.** BM25 reaches
93.3% recall@5 on single-hop questions. Dense retrieval's advantage over
it is +0.009 recall (95% CI [−0.023, +0.042], p=0.655) — **not
statistically distinguishable**, on only 8 of 95 questions.

**2. Ranking a depth sweep by recall is arithmetic, not evidence.**
recall@k cannot decrease as k grows, and MRR and nDCG turned out monotone
too. There is no quality-only metric that selects retrieval depth, which
makes measuring context cost mandatory rather than a refinement.

**3. Context precision must be read against its ceiling.** Raw 0.211 looks
broken until the ceiling turns out to be 0.441. Per category this
inverts: `single_hop` looks worst and is *nearest* its limit; the real
weak spot is `ambiguous` at **22% of a 0.800 ceiling** — most relevant
material available, least of it found. That's query understanding, not
ranking.

**4. Two of four RAGAS metrics need no LLM.** Ground-truth labels exist,
so context precision and recall are *computed*. Using a model to
approximate something you can calculate is noisier, priced per question,
irreproducible across model versions, and injects judge error into a
number that had none.

**5. The judge is validated before it's believed — and it fails.** The
keyless judge scores 50% on 9 cases with known verdicts. It rates
contradictions as *fully faithful*, because a contradicting sentence
reuses nearly every term of the context it contradicts. That failure is
the argument for a semantic judge, made by measurement rather than
assertion.

---

## Corrections made along the way

Left visible rather than edited away, because the progression is part of
the work.

- **Phase 2 → 3:** "dense loses to BM25" didn't survive a fair chunk-level
  comparison with paired inference.
- **Phase 2 → 3:** mh-011 was called unfixable; 10 of 48 configurations
  retrieve both articles, and an oracle showed it was a *ranking* failure
  rather than a candidate failure — opposite fixes.
- **Phase 3 review:** the equivalence claim was affirming the null.
  Re-stated against a declared margin, the cheap configuration is
  INCONCLUSIVE, not equivalent.
- **Phase 5 review:** the judge was being shown different context than the
  generator, which penalised exactly the prompt variants that cite.
- **First credentialed run:** chunk boundaries depended on whether
  `tiktoken` was installed — 202 chunks in CI, 209 on a machine that had
  run `pip install -r requirements.txt`. Every downstream number moved
  with an optional package while the memo's verification passed in CI.
  Token counts that decide anything are now a deterministic estimate.
- **First credentialed run:** with a key, Phase 5 pointed the LLM judge at
  the extractive baseline's answers and never at an LLM's. The memo
  described the generation layer as one judge run from being measured;
  that run would have graded a copier.
- **First credentialed run:** the judge-validation gate marked a correct
  verdict wrong. A two-claim answer with one fabricated claim scores
  exactly 0.5, and "low" was a strict `< 0.5`, so a flawless judge would
  have scored 83% — one slip from failing a gate it should pass.
- **First credentialed run:** the fabricated-citation count used
  `precision or 1.0`, which turned a precision of 0.0 — every citation
  invented — into a perfect score.
- **First credentialed run:** the notebook sync gate could not pass in CI.
  Cell ids were random locally, absent in CI, and rewritten by the
  commit hook, so three tools disagreed about every notebook.
- **Phase 6 review:** Cohen's kappa returned 1.0 where it is mathematically
  undefined.

---

## What is NOT measured

**No LLM arm has ever run.** Faithfulness, relevancy and correctness have
a complete harness and zero readings. The generation layer is built,
tested, and unvalidated — and the memo says so on every line rather than
implying otherwise.

Closing that gap costs **about $3** in judge calls plus cents of
generation, in one run of `python scripts/run_llm_eval.py`.

---

## Repository

```
src/
  corpus/      seed articles, golden set, BM25 difficulty analysis
  retrieval/   chunking, embeddings, vector store, retrieval arms
  generation/  prompt variants, RAG pipeline, refusal detection, baseline
  evaluation/  retrieval metrics, RAG metrics, LLM judge, bias audit
  llm/         provider abstraction, credential masking, call budget
notebooks/     01 corpus · 02 chunking · 03 bake-off · 04 generation
               05 evaluation · 06 judge audit · 07 decision memo
reports/       decision_memo.md, PROJECT_SUMMARY.md, figures
scripts/       build_notebooks.py, check_repo.py
tests/         356 tests
```

Everything runs with **no API key**: CI installs no LLM client and fails
if a credential is present, proving the pipeline degrades rather than
stops.
