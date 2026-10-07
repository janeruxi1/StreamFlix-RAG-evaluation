# StreamFlix Support RAG — Deployment Decision Memo

**From:** Xi Ru, Data Science
**To:** Support Platform, Customer Experience
**Re:** Whether to deploy retrieval-augmented answering on the help centre

---

## TL;DR

**Retrieval is optional at this size. Pilot the `cited` generation layer
with a support agent in the loop. Do not ship the `naive` prompt in any
form.**

- **Retrieval does not improve the answers at 45 articles.** Sending the
  model the whole help centre gave the same refusals (25 of 25), the
  same answer rate (76 of 95) and correctness within noise,
  +0.034 [-0.041, +0.109]. The pilot runs on retrieval anyway, BM25 over
  author-headed sections at depth 15 (**0.882** context recall for about
  **571 estimated context tokens** per query), because it is built, it
  costs about an eighth as much per query, and its context does not grow
  when the help centre does. That is a choice on cost and headroom, and
  either system would be defensible today.
- **Generation is measured, and good on the sample it was measured on.**
  The `cited` prompt refused **25 of 25 out-of-scope** questions, produced
  **0 fabricated citations**, and scored **0.984** faithfulness on the
  **76 answers** it gave — from a judge that scored **100% on 9 validation
  cases** before any of its scores were used.
- **Sample size is what holds it at "pilot", and mostly not the sample
  it first appears to be.** 25 of 25 supports a true refusal rate of at
  least **86.7%** (95% Wilson lower bound), which cannot exclude about
  one unanswerable question in 8 being answered. But once a cost is put
  on a wrong answer, the larger uncertainty is the 3 unsupported answers
  in 76: a rate that could be anywhere in [1.4%, 11.0%], and that moves
  the value of the system 5.8x as much as the refusal rate does.

The three parts rest on different amounts of evidence, and the
recommendation is graded to match rather than rounded up to "ship".

---

## What the pilot runs

| | |
|---|---|
| **Chunking** | `markdown_section` — split on the author's own headings |
| **Retrieval** | BM25 (Okapi, k1=1.5, b=0.75) at depth 15 |
| **Context cost** | ~571 estimated context tokens per query |
| **Corpus** | 45 articles, evaluated on 120 questions |

Token figures are estimates (words × 1.3), computed identically on every
machine. Exact tokenizer counts run about 5.5% higher on this corpus.

### What the context budget costs

The configuration was chosen under a 600-token context budget, not from
the top of the recall table. Across **64 configurations**, the top of the
table is `whole_article` with transformer embeddings at depth 15:
**0.967 recall** for **4.4x** the context.

The gap between that and what the pilot runs is **+0.085 [+0.039, +0.135]** recall,
and its interval excludes zero. The budget is a trade with a measured
price, not a free saving. Whether the extra context would help or hurt
the *answers* is not something a retrieval metric can say, because more
evidence and more distractors arrive together. Phase 8 measured the
limiting case, every article in the prompt, and the answers were neither
better nor worse (see "Is retrieval needed at all?").

### Why BM25 and not the transformer

On the same chunks at the same depth and cost, transformer embeddings
reach **0.907** against BM25's 0.882: a difference of
**+0.025 [-0.039, +0.090]**, with **22 of 95 questions** differing at
all. Against the declared 0.05 equivalence margin that is
**inconclusive** — the data supports neither "better" nor "the same".

Given an unresolved difference, the simpler system ships: no model
download, no fitted state, exactly reproducible. The more useful finding
is *where* they differ:

| Category | BM25 | Transformer |
|---|---:|---:|
| `single_hop` | 0.983 | 0.933 |
| `multi_hop` | 0.817 | 0.950 |
| `ambiguous` | 0.561 | 0.744 |

The two fail on different questions, which argues for combining them
rather than swapping one for the other.

---

## Evidence — retrieval

| Metric | Value | Read against |
|---|---:|---|
| context recall | **0.882** | — |
| context precision | **0.211** | a ceiling of **0.441**, so 48% of what is structurally possible |

**Precision must be read against its ceiling.** Articles split into ~4.5
chunks, so a single-article question retrieved at depth 15 can fill at
most ~4.5 of 15 slots with relevant material. Per category the reading
inverts: `single_hop` looks worst on raw precision and is nearest its
limit, while `ambiguous` reaches only **22% of** its ceiling — the most
relevant material available, and the least of it found.

---

## Evidence — generation

Generator `gpt-4o-mini`, judged by `gpt-4o`, on all 120 questions.

| Arm | Refuses out-of-scope | Answers in-scope | Faithfulness (answered) | Correctness (all) |
|---|---:|---:|---:|---:|
| extractive baseline | 56.0% | 62.1% | — | 0.268 |
| `naive` | 0.0% | 100.0% | 0.764 | 0.795 |
| `cited` | 100.0% | 80.0% | 0.984 | 0.733 |

**1. The LLM earns its cost.** Judged correctness of 0.733 against 0.268
for an extractive answerer that copies sentences out of the context.

**2. The refusal instruction is what separates the arms.** `naive`
answered every unanswerable question, and **23 of 25** of those answers
contain claims the judge found unsupported. `cited` refused all 25, and
scores a refusal F1 of 0.889.

**3. When both arms answer, they are equally faithful.** The headline
faithfulness gap, 0.984 against 0.764, is mostly composition: each arm is
averaged over the questions *it* chose to answer, and `naive` chose to
answer the unanswerable ones. On the **76 questions both** arms answered
it is **0.984 against 0.970**, a difference of **-0.015 [-0.040, +0.010]**
(`naive` minus `cited`). `cited` is not a more careful writer. It is a
writer that knows when to stop.

**4. Refusing has a measured price.** Correctness, `naive` minus `cited`:

| Questions | Difference |
|---|---:|
| in-scope (95) | **+0.160 [+0.102, +0.219]** |
| out-of-scope (25) | **-0.308 [-0.496, -0.116]** |
| all (120) | +0.062 [-0.007, +0.133] |

The two effects nearly cancel, so the single overall number says the
arms are indistinguishable while each half says they are not. `cited`
gives up real helpfulness on answerable questions to buy safety on
unanswerable ones. That is the right trade for billing, and it is a
trade.

**5. Over-refusal looks like a retrieval problem and mostly is not.**
`cited` declined **19 of 95** answerable questions. The evidence was
fully retrieved for 10, partly for 6 and not at all for 3, which reads
as if nearly half were declined for want of evidence. Phase 8 tested
that reading by supplying every article, and it did not hold: see the
next section. **10 of the 15 ambiguous** questions were refused, and the
full-corpus arm refused the same number.

---

## Is retrieval needed at all?

The help centre is **45 articles**, about 7,600 tokens. It fits in one
prompt, so the simplest system has no retriever: send everything. Phase 8
ran that system with the same `cited` prompt, generator, judge and 120
questions. The rule for what would count as retrieval losing was
committed to the repository before the run.

| | Retrieval | Full corpus | Full corpus minus retrieval |
|---|---:|---:|---:|
| unanswerable questions refused | 25 of 25 | 25 of 25 | no difference |
| answerable questions answered | 76 of 95 | 76 of 95 | +0.000 [-0.063, +0.074] |
| correctness, answerable | 0.715 | 0.749 | +0.034 [-0.041, +0.109] |
| faithfulness, where both answered (70) | | | -0.014 [-0.042, +0.011] |
| answers with an unsupported claim | 3 of 76 | 2 of 76 | |
| context per query | 571 tokens | 7,611 tokens | 13.3x |
| generation cost per 1,000 queries | $0.15 | $1.21 | |

**The two systems cannot be told apart on answers.** No interval
excludes zero, and the faithfulness difference sits inside the 0.05
margin set in advance. The rule required the full corpus to be
measurably more correct before it replaced retrieval. It was not, so
retrieval stays.

**That is a weaker result for retrieval than "stays" sounds.** The rule
put the burden of proof on the challenger. A rule that gave ties to the
simpler system would have returned the other verdict on the same data.
What retrieval buys at 45 articles is cost and headroom, not answer
quality. At the pilot's 1,000 tickets a day the cost difference is about
a dollar a day, so headroom is the real argument: retrieval's context
stays the same size as the help centre grows, and the full prompt does
not. That argument is about a larger corpus than this project tested.
"Cannot be told apart" is also not "shown equal": the correctness
interval reaches +0.109, wider than the 0.05 this memo treats as
equivalent elsewhere.

**It moved where the over-refusal problem sits.** Retrieval refused 19
answerable questions. With every article in the prompt the model still
refused 13 of those 19, and it refused 6 questions retrieval answered.
Nine of the 19 had been refused without the full evidence retrieved;
supplying it turned 2 of the 9 into answers. Over-refusal is mostly the
prompt's caution, and a better retriever would recover little of it.

**It did not surface the contradiction.** With both refund articles
certainly in the context, `cited` still states neither window. That
failure belongs to generation and to the help centre, not to retrieval.

**It did not make the unsafe prompt safe.** Given everything, `naive`
refused 0 of 25 unanswerable questions, the same as with retrieval.
Seeing the whole help centre does not make a model say that none of it
answers the question. The instruction does.

**One exploratory signal.** On the 20 multi-hop questions the full
corpus scored +0.138 [+0.030, +0.263] higher on correctness. That cut
was not in the rule, it is one of three read after the result, and it
has not been corrected for that. It is a hypothesis for a pre-declared
test: depth 15 may be dropping the second article a multi-hop question
needs.

---

## How much the evidence can bear

**The safety number is a small sample.** 25 of 25 is a lower bound of
86.7% on the true refusal rate. A system that answered one unanswerable
question in ten would still produce a clean sweep on 25 questions about
7% of the time. The questions were also written by the same person who
wrote the corpus.

**The judge is validated, not certified.** Nine cases with known verdicts
is a floor test; the lexical judge it replaced scores 50% on the same
suite. On the bias probes the longer answer scored lower on
**2 of 3 probes** (**-0.222** mean), and the judge's stated reasons name
specific added claims rather than length. Checked against real answers
(`naive` averages 67 words, `cited` 38 words), a per-word penalty
predicts a faithfulness gap of **-0.171** and the data shows -0.015. So
the judge is strict about elaboration and does not charge by the word.
Its agreement with the lexical judge is kappa -0.465, which counts
against the lexical judge; no second validated rater exists.

**The judge is lenient on omission.** On the planted contradiction it
scored the two arms **0.80 and 0.80** against a reference answer that
names the conflict, though neither answer does. It checks the claims an
answer makes, and a missing caveat is not a false claim.

**One model, one run, one corpus.** Temperature 0, a single generator
version, a synthetic help centre. None of this says anything about a
different model or real customer questions.

---

## Known failure modes, ranked by cost to a customer

Ranked by expense, not frequency.

**1. Two sources disagree and the answer does not say so.** Two articles
state different refund windows, 14 days and 30. Both retrieve. `naive`
asserts 14 days (`states_one`); `cited` names neither window and points
at the refund policy (`states_neither`). No judged arm surfaces the
conflict, including the arm given every article. This is a corpus defect
that a prompt cannot repair, and it is the one failure the judge
under-scores.

**2. Answering an unanswerable question.** Not observed for `cited`, and
bounded only as far as 25 questions can bound it.

**3. An unsupported claim inside a cited answer.** **3 of 76** answers
fell below the faithfulness threshold. A citation makes an answer
checkable, not correct.

**4. Over-refusal.** 19 answerable questions declined, most of them
declined again when nothing was missing from the context. A silent
failure: the customer gets nothing, and no hallucination metric records
it.

**5. Ambiguous questions.** Retrieval finds little of what is available
and generation then refuses most of them. This is query understanding,
which neither retrieval tuning nor prompting addresses.

---

## What the numbers are worth

A refusal bound is too low or high enough only relative to what a wrong
answer costs. [`pilot_design.md`](pilot_design.md) states that cost
model and works it through. Three results change how this memo should be
read:

- **The refusal evidence is enough for a moderate cost of error.** If
  one bad answer costs ten deflected tickets and a tenth of traffic is
  unanswerable, the system needs to refuse 61.7% of unanswerable
  questions. Twenty-five of 25 already establishes more than that.
- **The binding uncertainty is the bad-answer rate.** Three unsupported
  answers in 76 is 3.9%, with an interval up to 11.0%. At the top of
  that interval the system stops paying once a bad answer costs about
  eight tickets, whatever it refuses.
- **So the pilot is sized to measure that.** About 273 drafted answers
  reviewed by agents brings the bad-answer rate's upper bound under the
  break-even level, and the same labels give the judge the human rater
  it lacks.

The costs are assumptions, shown as ranges in the brief, and the
conclusion is stated as which assumptions the decision survives.

## Pilot conditions

The pilot is agent-assist: a support agent sees the cited draft and its
sources, and decides what the customer receives. Design, sample sizes
and the early-stopping rule are in [`pilot_design.md`](pilot_design.md).

1. **Fix the refund-window contradiction in the help centre before the
   pilot starts.** It is a content defect, and the system currently
   hides it.
2. **Have agents label every draft** as supported or not. That is the
   pilot's measurement of the bad-answer rate.
3. **Log every refusal with its retrieved articles**, so over-refusal is
   measured on real questions rather than inferred from 95 synthetic
   ones.

## What would move this to "ship"

- **An agent-labelled bad-answer rate with an upper bound under
  break-even**, at a cost of error the business has actually stated.
- **A larger out-of-scope set, only if a wrong answer is expensive.**
  Roughly 300 unanswerable questions with no answered case would put the
  refusal bound near 99%, which matters once a bad answer costs twenty
  tickets or more and not before.
- **A fix for over-refusal, tested as a prompt change.** More context
  is ruled out as the fix: with all of it, the answer rate stayed at
  76 of 95. Recovering those refusals means changing what the prompt
  accepts as enough evidence, and then re-measuring the 25 of 25,
  because that is the number such a change puts at risk.
- **A clarifying-question path for ambiguous queries**, since refusing 10
  of 15 is safe and unhelpful.

---

## Verification

Every number in this memo is checked by `notebooks/07_decision_memo.py`,
which CI runs and which fails the build on drift. The check is of two
kinds, and the notebook says which applies to each number:

- **Live.** Retrieval metrics, the extractive baseline and the lexical
  judge are recomputed from this repository on every run.
- **Recorded.** Anything that needed an API key or a model download —
  the LLM arms, the LLM judge, the transformer retrieval arm — is checked
  against the measured records in `reports/metrics/`, which are written
  only when the measurement actually ran. CI has no key by design, so it
  cannot recompute these; it can confirm the memo says what was measured.

The full-corpus section is checked the same way by
`notebooks/08_full_corpus_baseline.py`, and the cost-model figures by
`notebooks/09_decision_model.py`.

Re-running `python scripts/run_llm_eval.py` regenerates the records, and
any change in them fails the check until this memo is updated.
