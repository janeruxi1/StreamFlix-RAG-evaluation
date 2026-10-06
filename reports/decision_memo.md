# StreamFlix Support RAG — Deployment Decision Memo

**From:** Xi Ru, Data Science
**To:** Support Platform, Customer Experience
**Re:** Whether to deploy retrieval-augmented answering on the help centre

---

## TL;DR

**Recommendation: ship the retrieval layer. Hold the generation layer.**

The split is the recommendation, not a hedge.

- **Retrieval is measured and ready.** 48 configurations compared with
  paired inference, multiplicity correction and held-out selection. The
  shipping configuration reaches **0.882** context recall at ~**591
  context tokens** per query.
- **Generation is built, tested, and unmeasured.** The full harness
  exists and points at it. No LLM arm has ever run, so **no claim about
  answer quality in this project is backed by evidence.**

Shipping both on the strength of the first would borrow credibility from
the measured half to cover the unmeasured half. That is the specific
mistake this project was built to avoid, and it would be strange to
finish by making it.

**Cost to close the gap: about $3 in judge calls plus cents of
generation, in one run of `python scripts/run_llm_eval.py`.** That is the
entire distance between a conditional recommendation and an evidenced
one.

---

## What ships

| | |
|---|---|
| **Chunking** | `markdown_section` — split on the author's own headings |
| **Retrieval** | BM25 (Okapi, k1=1.5, b=0.75) at depth 15 |
| **Context cost** | ~591 context tokens per query |
| **Corpus** | 45 articles, evaluated on 120 questions |

**Why this configuration and not the top of the recall table.** It sits on
the Pareto frontier under a 600-token context budget. The highest-recall
configuration costs **4.5× more context** for a recall difference that
fails a paired significance test.

The honest form of that claim matters: the cheap configuration is
**INCONCLUSIVE, not equivalent**. Against a declared 0.05 margin of
practical equivalence its confidence interval reaches 0.084, so it spills
past the margin. What can be said is that the 4.5× saving is certain and
the recall cost is bounded above by 0.084 at 95% confidence. Ninety-five
questions cannot resolve it more finely than that.

---

## Evidence — retrieval

| Metric | Value | Read against |
|---|---:|---|
| context recall | **0.882** | — |
| context precision | **0.211** | a ceiling of **0.441**, so 48% of what is structurally possible |

**Precision must be read against its ceiling.** Articles split into ~4.5
chunks, so a single-article question retrieved at depth 15 can fill at
most ~4.5 of 15 slots with relevant material. The rest *must* be
irrelevant. Reported against an implicit ceiling of 1.0 — the usual way —
the same system looks broken.

Per category, which inverts the raw reading:

| Category | recall | precision | ceiling | % of max |
|---|---:|---:|---:|---:|
| `single_hop` | 0.983 | 0.186 | 0.292 | 63% |
| `multi_hop` | 0.817 | 0.313 | 0.620 | 51% |
| `ambiguous` | 0.561 | 0.178 | 0.800 | **22%** |

`single_hop` looks worst on raw precision and is *nearest* its limit. The
real weak spot is `ambiguous`, which has the most relevant material
available and finds the least of it — invisible in the raw column where
it and `single_hop` differ by 0.008.

---

## What is NOT measured

Ordered by how much each should hold up a deployment.

**1. Answer quality — entirely unmeasured.** No LLM arm has run.
Faithfulness, relevancy and correctness have a harness and no readings.
The extractive baseline's **62.1%** in-scope answer rate and **56.0%**
out-of-scope refusal rate are a floor for a *non-LLM* method, not a
forecast for the LLM.

**2. The judge — fails its own gate.** The only judge exercised end to end
scores **50%** on the Phase 5 validation suite, below the 80% trust
threshold. Its faithfulness numbers are harness output, not evidence. It
is in the repository because its failures are informative: it rates
contradictions as fully faithful, because a contradicting sentence reuses
nearly every term of the context it contradicts.

**3. Judge bias at deployment scale.** The audited judge penalises answer
length by **-0.521** on average, material on 3 of 3 probes — the opposite
of the documented LLM-judge tendency to reward verbosity. Whether the
intended production judge shows the opposite bias is unknown, and it
determines whether prompt-variant comparisons mean anything at all.

---

## Known failure modes, ranked by cost to a customer

Ranked by expense, not frequency. A frequency-ordered list inverts the
priority.

**1. Answering an unanswerable question — 11 of 25 out-of-scope.**
A confident wrong answer about billing is the most expensive output this
system can produce. It's also the failure the corpus was built to
provoke: the hardest out-of-scope questions have a topically adjacent
article that is silent on the actual question, so retrieval returns
something plausible and the generator answers from it.

**2. Ambiguous questions under-retrieve — 22% of ceiling.**
Not a ranking failure. The system retrieves the wrong articles because it
cannot tell which reading of the question was meant. That is query
understanding, and neither retrieval tuning nor prompting addresses it.

**3. The planted contradiction (mh-011) — unresolved.**
Two articles state different refund windows. Both retrieve at depth 15,
so the evidence is present. Whether any answer *flags* the conflict
rather than silently picking one is unmeasured and needs the LLM arms.

**4. Over-refusal — 34 in-scope questions.**
A silent UX failure: the user gets nothing when the evidence was
available. Cheaper than a wrong answer, and invisible to any metric that
only counts hallucinations.

---

## Deployment conditions

If the retrieval layer ships on its own — as a "related articles" surface
rather than an answering bot — it needs no further validation. That is the
recommendation.

If the generation layer is to ship, these gate it:

1. **The judge passes its validation gate** (≥80% on the 9-case suite).
   Until then no faithfulness number means anything.
2. **The prompt ladder is compared on a judge without material length
   bias**, or on length-normalised answers. The rungs differ
   systematically in verbosity, so a biased judge would rank them on
   length.
3. **Out-of-scope refusal is measured on the LLM arms.** The 25
   unanswerable questions are the deployment risk, and the baseline's
   56.0% tells us nothing about what the LLM will do.
4. **mh-011 is resolved** — does any variant flag the contradiction?

---

## What would change this recommendation

- **The LLM arms run and refusal on out-of-scope is high.** Then the
  generation layer becomes shippable and this memo is superseded.
- **The transformer retrieval arm beats BM25 materially.** It has never
  run; `sentence-transformers` was not installed. If it wins by more than
  the equivalence margin, the shipping configuration changes.
- **The corpus grows.** Every ceiling in this memo is a function of 45
  articles at depth 15. More articles move the precision ceiling and may
  change which chunking strategy sits on the frontier.

---

## Verification

Every number in this memo is recomputed by
`notebooks/07_decision_memo.py` and compared against live output. CI runs
it and fails the build on drift.

Numbers in a markdown file rot silently, and a stale memo is worse than
no memo because it carries the authority of having been checked once.
Every figure here is reproducible from this repository at the commit it
was written against.
