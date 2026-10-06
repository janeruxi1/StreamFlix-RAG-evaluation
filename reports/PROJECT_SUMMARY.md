# Project Summary — StreamFlix RAG Evaluation

**What was built + what was found.** Single-page catalog of the project's
components and headline results. The decision memo's numbers are checked
by `notebooks/07_decision_memo.py`, which CI runs and which fails the
build on drift.

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

**Headline outcome.** Ship the retrieval layer. Pilot the `cited`
generation layer with a support agent in the loop. Do not ship the
`naive` prompt. The cited arm refused **25 of 25** unanswerable questions
with **0** fabricated citations and **0.984** faithfulness — and 25
questions can only bound its true refusal rate at **86.7%**, which is why
the recommendation is a pilot.

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

**Retrieval** (BM25, `markdown_section`, depth 15 — recomputed in CI)

| Metric | Value | Read against |
|---|---:|---|
| context recall | **0.882** | 0.967 at the top of the table, for 4.4x the context |
| context precision | **0.211** | ceiling **0.441** → 48% of what's structurally possible |
| context cost | **~571 estimated tokens** | under a 600-token budget |

**Generation** (`gpt-4o-mini`, judged by `gpt-4o` — from the measured records)

| Arm | Refuses out-of-scope | Answers in-scope | Faithfulness (answered) | Correctness |
|---|---:|---:|---:|---:|
| extractive baseline | 56.0% | 62.1% | 0.810 | 0.268 |
| `naive` | 0.0% | 100.0% | 0.764 | 0.795 |
| `cited` | 100.0% | 80.0% | 0.984 | 0.733 |

**The judge**

| Check | Result |
|---|---|
| validation gate (9 known-verdict cases) | `gpt-4o` **100%**, lexical judge **50%** |
| length probes | longer answer scored lower on **2 of 3**, for stated reasons that name added claims |
| per-word penalty, tested on real answers | predicted −0.171, observed −0.015 [−0.040, +0.010] |
| self-consistency | identical on repeat |

---

## What each phase produced

| Phase | Built | Key finding |
|---|---|---|
| 1 | 45-article corpus + 120-question golden set | BM25 gets **93.3%** recall@5 on single-hop — the floor is high, so per-category reporting is mandatory |
| 2 | 5 chunking strategies, 2 embedding backends, vector store | `fixed_token_256` was a silent no-op: the same words as `whole_article`, one chunk per article |
| 3 | 48-configuration bake-off (64 with the transformer arm) | recall@k is monotone in k, so **no quality metric can select depth** — it's a cost decision |
| 4 | 5 prompt variants + extractive baseline + refusal detection | "use only the context" produced **zero** refusals; explicit permission to refuse produced 25 of 25 |
| 5 | Evaluation harness with a validated judge | the faithfulness gap between arms was **composition**: 0.984 vs 0.764 overall, 0.984 vs 0.970 on the same questions |
| 6 | Judge audit | the "length bias" was the probes: the judge named the added claims, and real answers 28 words apart show no penalty |
| 7 | Decision memo + verification | 53 memo claims checked: 11 recomputed live, 42 against committed records |

---

## Five findings worth the reader's time

**1. A safe-looking prompt that never refuses.** `grounded` tells the
model to answer only from the context. It answered all 25 unanswerable
questions. Restricting the *source* of an answer is not the same as
permitting the model to decline, and only the second changed behaviour.

**2. The headline faithfulness comparison was comparing different
questions.** `cited` 0.984 against `naive` 0.764 looks like a careful
writer against a careless one. Each arm is averaged over the questions it
chose to answer. On the 76 both answered it is 0.984 against 0.970,
−0.015 [−0.040, +0.010]. `cited` is not more careful; it knows when to
stop.

**3. A single correctness number hid two opposite effects.** `naive` is
more correct on answerable questions by +0.160 [+0.102, +0.219] and less
correct on unanswerable ones by −0.308 [−0.496, −0.116]. They cancel to
+0.062 [−0.007, +0.133], which reads as "no difference".

**4. Twenty-five out of twenty-five is a bound, not a guarantee.** A
system failing one time in ten still produces a clean sweep on 25
questions about 7% of the time. The Wilson lower bound is 86.7%, and
that number, not the 100%, sets the recommendation.

**5. The instrument has a blind spot, and it is the most expensive
failure.** Two help articles state different refund windows. `naive`
asserts one, `cited` names neither, and the judge scored both 0.80
against a reference that names the conflict. A judge that checks the
claims an answer makes does not penalise the caveat it leaves out.

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
- **Phase 6 review:** Cohen's kappa returned 1.0 where it is mathematically
  undefined.
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
- **After the credentialed run:** the length-bias probes were written for
  a judge that counts words. A judge that counts claims found real added
  claims in two of them, so the measured "bias" was partly the probes.
- **After the credentialed run:** the judge audit used prompt length as a
  proxy for answer length. Measured, the shortest instruction (`naive`)
  writes the longest answers.
- **After the credentialed run:** the memo quoted ~591 context tokens and
  the bake-off 571 for the same configuration, from two different
  multipliers. There is now one definition.

---

## What the evidence cannot bear

- **The safety number is 25 questions**, written by the same person who
  wrote the corpus.
- **The judge is validated on 9 cases and audited on 5 probes.** No
  second validated rater exists, and the judge is lenient on omission.
- **Generation was only measured on the 571-token context.** The
  0.967-recall configuration might reduce over-refusal or dilute
  faithfulness; nothing here says which.
- **One model, one run, one synthetic corpus.**

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
               metrics/   measured records from the credentialed run
               llm_run/   that run's output and environment manifest
scripts/       build_notebooks.py, check_repo.py, run_llm_eval.py
tests/         362 tests
```

Everything runs with **no API key**: CI installs no LLM client and fails
if a credential is present, proving the pipeline degrades rather than
stops. The credentialed run is one command, `python scripts/run_llm_eval.py`,
and costs about $3.
