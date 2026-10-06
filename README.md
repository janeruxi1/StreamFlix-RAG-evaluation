# StreamFlix RAG — Retrieval-Augmented Support with an Evaluation Harness

A question-answering system over a customer-support knowledge base, built
around the part that usually gets skipped: **measuring whether it actually
works.**

Assembling a RAG pipeline is a weekend of gluing libraries together. The
hard question is the one that follows — *is this good enough to put in
front of customers, and how would I know?* This project treats that
question as the deliverable. The retrieval and generation code exists to
give the evaluation harness something to measure.

The work was framed by [`reports/scenario_brief.md`](reports/scenario_brief.md)
— a stakeholder asking not for an assistant, but for a defensible answer to
whether one is safe to deploy.

> **Status: all 7 phases complete, and measured end to end.** The
> recommendation is graded to the evidence: **ship** the retrieval layer,
> **pilot** the cited generation layer with a person in the loop, and
> **do not ship** the naive prompt. The cited arm refused 25 of 25
> unanswerable questions with zero fabricated citations and 0.984
> faithfulness — and 25 questions can only bound its true refusal rate
> at 86.7%, which is why it is a pilot and not a launch.

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

`fixed_token_256` **collapsed into `whole_article`** — one chunk per
article with the same words, only the whitespace changed. The longest
article is ~215 estimated tokens, so a 256-token window never has
anything to split. The chunker ran, did nothing, and would have
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

### Phase 3 — Retrieval bake-off ✅

48 configurations: 4 chunking strategies × 3 retrieval arms (BM25,
LSA-dense, RRF hybrid) × 4 depths, all indexing the same text and scored
by the same code.

That last point is a correction. Phase 1 ran BM25 over whole *articles*
while Phase 2 ran dense retrieval over *chunks* — the unit of retrieval
differed, so any gap between them could have been the unit rather than
the method. Those numbers were never comparable, and Phase 3 rebuilds the
comparison properly.

Most of the phase is about not being fooled by a 48-row table. See the
findings below.

With `sentence-transformers` installed the sweep adds a BGE-small arm and
becomes 64 configurations. CI does not download the model, so the 48-row
sweep is what CI reproduces; the transformer comparison is checked
against a committed record, [`reports/metrics/03_retrieval_arms.json`](reports/metrics/03_retrieval_arms.json).
It changes the top of the table and not what ships — see finding 7.

### Phase 4 — Generation, grounding & refusal ✅

Five prompt variants forming a ladder — `naive` → `grounded` →
`grounded_refusal` → `cited` → `strict` — each adding exactly one
mechanism so its contribution is attributable.

Everything measured in this phase is **judge-free**: every metric is a
regex, a set operation, or a count. Cheap deterministic checks are
exhausted before paying for an LLM judge, and they already catch the
failures that decide deployability — answering when it should refuse,
refusing when it should answer, and citing sources it was never shown.
Answer *correctness* needs a judge and is deferred to Phase 5.

**An extractive baseline the LLM has to beat.** A non-LLM answerer picks
sentences from retrieved context by IDF-weighted term overlap and refuses
below a threshold (tuned on the dev split only). It exists so "the LLM
answers well" is a measurable claim rather than an assumption — and it
keeps the whole phase runnable in CI with no API key. It reaches 62.1%
answer rate in-scope and 56.0% refusal on out-of-scope, F1 0.589, at
~7ms and zero cost per question. Its weaknesses are the point: it cannot
synthesise across articles, paraphrase, or detect contradictions, which
is exactly the list of things an LLM is being paid to add.

**The central tension.** Every instruction that makes a model more
willing to refuse also makes it refuse questions it could have answered.
There is no prompt that wins both ends, so every variant is scored on a
*pair* of rates — refusal on the 25 out-of-scope questions and answer
rate on the 95 in-scope ones — and neither is reported alone. The
baseline's threshold sweep shows the same trade-off through a single
knob: raising it from 0.15 to 0.50 lifts out-of-scope refusal from 33% to
80% while dropping in-scope answering from 100% to 37%.

**Partial refusals count as answers**, because unsupported claims were
made either way. They come in two shapes and both must be caught:
*refuse-then-answer* ("I don't have enough information, but generally…")
and *answer-then-refuse* ("Premium costs $19.99 [bill-001]. I don't have
enough information about student discounts.").

The second is easy to miss, and missing it fails in the dangerous
direction: a model that answers an out-of-scope question and appends a
hedge gets counted as having correctly refused, so the safety metric
reports the opposite of what happened. The detector was originally
inspecting only text *after* the refusal phrase and had exactly this
hole — latent, because the extractive baseline never produces that shape,
but it would have corrupted every LLM arm the moment a key was added.

**Blame attribution is built in.** The pipeline keeps retrieved chunks
alongside every answer, so a wrong answer with zero retrieval recall is
identifiable as an upstream failure rather than a hallucination — the
distinction Phase 3 flagged and Phase 6 depends on.

**The ladder, measured** on `gpt-4o-mini` over all 120 questions:

| Rung | Refuses out-of-scope | Answers in-scope | Refusal F1 | Fabricated citations | Mean answer |
|---|---:|---:|---:|---:|---:|
| extractive baseline | 56.0% | 62.1% | 0.589 | 0 | 25 words |
| `naive` | 0.0% | 100.0% | 0.000 | not applicable | 67 words |
| `grounded` | 0.0% | 100.0% | 0.000 | not applicable | 36 words |
| `grounded_refusal` | 100.0% | 74.7% | 0.855 | not applicable | 25 words |
| `cited` | 100.0% | 80.0% | 0.889 | 0 | 38 words |
| `strict` | 100.0% | 73.7% | 0.848 | 0 | 43 words |

Three things in that table are worth more than the ranking. Telling the
model to use only the context (`grounded`) produced **no refusals at
all**; it took explicit permission to refuse. Asking for citations did
not cost answers: `cited` answered 80.0% against 74.7% for the plain
refusal rung, a five-question gap on 95 and so not a ranking. And the
most elaborate prompt (`strict`) is no safer than `cited`, both refusing
25 of 25, while answering six fewer questions. Each rung adds one
mechanism, so each of those is attributable to it.

### Phase 5 — The evaluation harness ✅

Four metrics in the RAGAS tradition — context precision, context recall,
faithfulness, answer relevancy — with one deliberate departure and one
addition.

**The departure: two of the four need no LLM.** RAGAS computes all four
with a judge because it assumes no ground-truth labels. This project has
exact labels — every golden question names its source articles — so
context precision and recall are *computed*, not estimated. Using a model
to approximate a quantity you can calculate is worse on every axis:
noisier, priced per question, not reproducible across model versions, and
it injects the judge's error into a number that had none. It also means a
credential outage degrades the harness instead of stopping it.

**The addition: the judge is validated before it is believed.** The
standard practice is to report faithfulness to three decimals from a model
whose agreement with ground truth was never measured — an unknown error
rate hiding inside the headline number. Here the judge first faces 9 cases
whose correct verdict follows from how they were written (supported,
fabricated, contradicted, refusal, off-topic), and scores below 80%
disqualify its numbers.

Two metric definitions decide what the numbers mean, and both were
corrected during review. The judge is shown context in *exactly* the form
the generator saw it, article-id tags included — stripping them turns
every citation into an unverifiable claim and penalises precisely the
prompt variants that follow the citation instruction. And faithfulness
covers *answered* questions only: a refusal asserts nothing, so a correct
judge scores it vacuously faithful at 1.0, meaning a system that refuses
everything would report a perfect score.

The keyless `LexicalJudge` scores **50%** and fails the gate — which is
exactly why it's in the repo. It rates contradictions as *fully faithful*,
because a contradicting sentence reuses nearly every term of the context
it contradicts, and it rates refusals as maximally unfaithful because they
share no vocabulary. Neither is fixable by tuning a threshold; they're
what "semantic" means. That's the argument for paying for an LLM judge
made by measurement rather than assertion.

**With a key, the judge is `gpt-4o` and it passes at 100%** on the same
9 cases. Only then are the LLM arms judged — if the judge fails its gate
the notebook stops before spending on scores it could not trust.

| Arm | Faithfulness (answered) | Correctness (all 120) | Answered |
|---|---:|---:|---:|
| extractive baseline | 0.810 | 0.268 | 70 |
| `naive` | 0.764 | 0.795 | 120 |
| `cited` | 0.984 | 0.733 | 76 |

That table invites a wrong reading, and the notebook is built to stop
it. `cited` looks far more faithful than `naive`, but each arm is
averaged over the questions *it* chose to answer. On the 76 questions
both answered, faithfulness is 0.984 against 0.970 — a difference of
−0.015 [−0.040, +0.010]. The gap in the table is `naive` answering 25
unanswerable questions. See finding 8.

**Context precision must be read against its ceiling.** The raw 0.211
looks alarming until you notice the ceiling is 0.441, not 1.0 — articles
split into ~4.5 chunks, so a single-article question retrieved at depth 15
can only ever fill ~4.5 of 15 slots with relevant material. The rest
*must* be irrelevant. Per category:

| Category | recall | precision | ceiling | % of max |
|---|---:|---:|---:|---:|
| `single_hop` | 0.983 | 0.186 | 0.292 | 63% |
| `multi_hop` | 0.817 | 0.313 | 0.620 | 51% |
| `ambiguous` | 0.561 | 0.178 | 0.800 | **22%** |

Reading the columns together inverts the raw story. `single_hop` looks
worst on precision but is *nearest* its limit. The real weak spot is
`ambiguous`, which has the most relevant material available and finds the
least of it — invisible in the raw numbers, where it and `single_hop` look
almost identical. Combined with its weak recall, the pattern says the
system isn't failing to *rank* the right articles; it's failing to work
out which articles an underspecified question is about. That's query
understanding, and no amount of retrieval or prompt tuning fixes it.

### Phase 6 — Auditing the judge ✅

Phase 5 asked whether the judge gets known cases *right*. That's
necessary and not sufficient — a judge can score 100% on unambiguous
cases and still be useless for comparing prompt variants, because it may
respond to properties unrelated to answer quality.

**Paired probes, not correlations.** Correlating score against answer
length across the golden set does *not* measure length bias: longer
answers may genuinely be more complete, so the correlation confounds bias
with quality — in the direction that makes bias look real when it isn't.
Each probe here is instead a **pair identical in every respect except one
manipulated variable** — same claims, same context, same citations, only
phrasing length differs. Any gap is attributable because nothing else
moved. Three dimensions: length, context ordering, and self-consistency.

**The keyless judge has a severe, opposite bias.** `LexicalJudge`
*penalises* length by −0.521 on average, materially on 3 of 3 probes. The
mechanism is transparent: faithfulness is the fraction of *answer* terms
found in context, so padding words are absent from the context and drag
the ratio down. That's the **opposite** of the documented LLM-judge
tendency to reward verbosity. Two judges with opposite biases are both
wrong, and neither is fixable by choosing a threshold — the direction is
a property of the scoring mechanism.

**Converted into a statement about a real conclusion.** At −0.014 score
per answer word, two prompt variants differing by ~30 words would differ
by ~0.43 on faithfulness from length alone — far above the 0.15 material
threshold. So this judge cannot be used for the Phase 4 ladder
comparison. Stated precisely: it does *not* invalidate Phase 4's
judge-free results (refusal rates, citation integrity, blame attribution
are all counts and regexes), only faithfulness comparisons across
variants.

**Inter-judge agreement, and why 1.000 isn't reassuring.** Cohen's kappa
rather than raw agreement, because raw agreement is inflated whenever one
verdict dominates. The two keyless raters diverge by up to 0.320 on the
probes yet agree on all 40 golden-set answers — including which 15 to
reject, so this is genuine agreement with variance, not the degenerate
case where kappa is undefined.

The reason they agree is the point: the probes separate them on
**padding**, and the extractive baseline never pads — it copies sentences
verbatim. The one thing these formulations disagree about doesn't occur
in this data. So agreement here is evidence the *data* lacks the
separating feature, not evidence either rater is right. Both are still
lexical, and both still score contradictions as faithful.

**Audited with a key, `gpt-4o` behaves differently, and the probes turn
out to be part of the result.** The longer answer scored lower on 2 of 3
length probes (−0.222 on average). Read alone that is a length bias. The
judge's stated reasons say otherwise: it marked specific added sentences
as unsupported — "up to four devices in your household can watch
different titles", "acting promptly is advisable" — and it is right that
the context states neither. The probes assumed their padding asserted
nothing, which held for a judge that counts words and not for one that
counts claims.

So the notebook tests the length reading against real answers. On the
same 76 questions, `naive` writes 28 more words than `cited`. A per-word
penalty of the size the probes imply predicts a faithfulness gap of
−0.171; the data shows −0.015 [−0.040, +0.010]. The judge is strict about
elaboration and does not charge by the word.

Agreement between `gpt-4o` and the lexical judge is kappa −0.465, worse
than chance. That counts against the lexical judge, which failed its
gate: 15 of the 26 disagreements are refusals it scores as unfaithful.
It does not certify `gpt-4o`, which would need a second rater that also
passes — and this project has none.

### Phase 7 — Decision memo, and keeping it honest ✅

The deliverable is [`reports/decision_memo.md`](reports/decision_memo.md).
**Recommendation: ship the retrieval layer, pilot the `cited` generation
layer with a support agent in the loop, and do not ship the `naive`
prompt.**

The three parts rest on different amounts of evidence and are graded to
match. Retrieval was measured across the full sweep with paired
inference, multiplicity correction and held-out selection. `cited`
generation is measured and good — and held at "pilot" by sample size, not
by its scores: 25 of 25 refusals supports a true rate of at least 86.7%
and cannot exclude about one unanswerable question in eight being
answered.

**Failure modes are ranked by cost to a customer, not frequency.** The
one ranked first is a single question. Two help articles state different
refund windows; `naive` asserts one of them, `cited` names neither, and
no judged arm says the sources disagree. The judge scored both answers
0.80 against a reference that names the conflict, so it is also the one
failure the instrument under-reports.

**Every memo number is verified by `notebooks/07_decision_memo.py`**,
which checks 53 claims and exits non-zero on drift. CI runs it. The check
is of two kinds and the notebook says which applies to each number: 11
are recomputed live, and 42 that needed an API key or a model download
are checked against the measured records in `reports/metrics/`. CI has no
key by design, so it cannot re-measure those; it can confirm the memo
says what was measured.

---

## Findings so far

Findings 1 and 2 are Phase 2 results. Phase 3 revised both — the
revisions are findings 5 and 6, and the progression is left visible
rather than edited away.

**1. The untuned dense baseline loses to BM25.** *(Phase 2 — later
qualified)*

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

> **Phase 3 qualification:** with a fair chunk-level comparison and a
> *paired* bootstrap, the BM25–dense gap at depth 10 is +0.009 recall
> (95% CI [−0.023, +0.042], p=0.655) — **not statistically
> distinguishable**, and the two arms differ on only 8 of 95 questions.
> The Phase 2 point estimate was real; the conclusion drawn from it was
> stronger than the evidence supported.

**2. One question defeats both retrievers — and it will look like a
hallucination.** *(Phase 2 — later overturned, see finding 6)*

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

**3. Ranking a depth sweep by recall is arithmetic, not evidence.**

recall@k cannot decrease as k grows, so sweeping depth and sorting by
recall crowns the deepest configuration by construction — it would do so
even for a retriever returning documents at random. The obvious escape is
a rank-sensitive metric, but MRR and nDCG turned out to be monotone in
depth too, for a structural reason: retrieving deeper only ever *appends*
to the ranked list, so the first correct document never moves and each
extra item adds non-negative discounted gain.

There is therefore no quality-only metric that can select retrieval
depth. Depth is a **cost** decision, which makes measuring context tokens
mandatory rather than a refinement.

**4. The cost saving is certain; the recall cost is not resolved.**

| | recall | tokens/query |
|---|---:|---:|
| Top of the recall table (`whole_article` + LSA, depth 15) | 0.918 | 2,543 |
| Cheapest serious contender (`markdown_section` + LSA, depth 15) | 0.878 | 565 |

The cheap configuration is **4.5× cheaper**, and the paired difference is
+0.040 recall, 95% CI [−0.003, +0.084], p=0.075.

The tempting conclusion — "not significant, so they're equivalent, ship
the cheap one" — is **affirming the null**, and this project doesn't get
to make it. A non-significant result on 95 questions is largely a
statement about sample size. Making a positive equivalence claim requires
a **declared margin of practical equivalence** (0.05 recall here, about
five questions, fixed before looking at results), and then asking whether
the whole interval falls inside it. It doesn't — the upper bound of 0.084
spills past 0.05, so the verdict is **INCONCLUSIVE**, not equivalent.

What can honestly be said: the 4.5× saving is certain, and the recall
cost is **bounded above by 0.084 at 95% confidence**. That's a real
engineering decision with a quantified worst case, rather than a false
claim of equivalence. Resolving it needs a larger golden set, not more
analysis.

Against the leader, the 47 challengers resolve as 1 equivalent, 44
different, 2 inconclusive — and the single equivalent one is no cheaper,
so it buys nothing.

This still supports what chunking is *for*. The Pareto frontier shows
chunking buying large, certain cost reductions for small, uncertain
recall costs. The saving compounds in Phase 5, since fewer distractor
tokens is exactly the condition under which faithfulness improves.

**Recommended configuration:** `markdown_section` + BM25 at depth 15 —
recall 0.882 at 571 tokens/query, chosen on the Pareto frontier under a
600-token budget with the quality question explicitly left open.

**5. The bake-off finds a group, not a winner.**

At fixed depth 10, 7 of 11 challengers look significantly worse — but
only **4 survive Holm-Bonferroni correction**. The other 3 stars were
multiplicity artifacts, which is exactly what a 12-way comparison at
α=0.05 predicts. Ninety-five questions can
separate a real tail — `sentence_window`, the most elaborate strategy
producing the most chunks, is beaten across every arm — but cannot rank
the top 5 against each other. Inside that group, cost and simplicity
decide; between group and tail, the measurement decides.

Comparisons are **paired** bootstraps, because every configuration is
scored on the same questions. Marginal CIs would be dominated by question
difficulty, which is shared across arms and therefore irrelevant to which
arm is better. p-values are Holm-corrected across each family of tests.

**6. mh-011 is resolved — and Phase 2's conclusion was wrong.**

Phase 2 found the refund-contradiction question failed under both
retrievers and concluded it was "a property of the question" that
"changing retriever will not fix." That was too strong.

10 of 48 configurations retrieve both articles, and an oracle over the
pooled candidates found both for **every** chunking strategy at depth 30.
So it was never a candidate-generation failure — the retriever always
surfaced them and simply ranked them below the free-trial articles.

Ranking failures and candidate failures need opposite fixes (retrieve
deeper / rerank versus query rewriting or a corpus change). Phase 2 had
no oracle, couldn't tell them apart, and guessed the harder one. What it
got right is the consequence: at depth 5 the generator still receives
five confident, on-topic, wrong chunks.

Findings 3 to 6 quote the 48-configuration sweep that runs without a
model download, which is what CI reproduces. Findings 7 to 9 come from
the credentialed run and are checked against `reports/metrics/`.

**7. The transformer arm tops the table and does not change what ships.**

With BGE-small added, the best configuration is `whole_article` at depth
15: 0.967 recall, against 0.882 for the shipping configuration at 4.4x
less context. That gap is +0.085 [+0.039, +0.135], and it is real. The
context budget is a trade with a measured price.

On the *same* chunks, depth and cost, the transformer scores 0.907
against BM25's 0.882: +0.025 [−0.039, +0.090], 22 of 95 questions
differing. Against the declared 0.05 margin that is inconclusive —
neither better nor the same. Given an unresolved difference, the arm
with no model dependency ships.

| Category | BM25 | Transformer |
|---|---:|---:|
| `single_hop` | 0.983 | 0.933 |
| `multi_hop` | 0.817 | 0.950 |
| `ambiguous` | 0.561 | 0.744 |

The aggregate hides that the two fail on different questions, which is
an argument for combining them and not for swapping one for the other.

**8. A faithfulness gap that was mostly composition.**

`cited` scores 0.984 faithfulness and `naive` 0.764. On the 76 questions
both answered, it is 0.984 against 0.970. Almost the whole gap is `naive`
answering the 25 unanswerable questions, where 23 of its answers contain
claims the judge found unsupported.

The same split explains correctness. `naive` is more correct on
answerable questions by +0.160 [+0.102, +0.219] and less correct on
unanswerable ones by −0.308 [−0.496, −0.116]. Overall the two cancel to
+0.062 [−0.007, +0.133], so a single number reports "no difference"
between two arms that differ sharply in opposite directions.

**9. Twenty-five out of twenty-five is a bound, not a guarantee.**

A system that answered one unanswerable question in ten would still
score a clean sweep on 25 questions about 7% of the time. The Wilson
lower bound on `cited`'s refusal rate is 86.7%. The memo recommends a
pilot on that number and says roughly 300 clean out-of-scope questions
would be needed to put the bound near 99%.

---

## Roadmap

| Phase | Status |
|---|---|
| 1. Corpus + golden set | ✅ Complete |
| 2. Chunking, embeddings, retrieval baseline | ✅ Complete |
| 3. Retrieval bake-off — strategy × backend × depth, BM25 as a first-class arm | ✅ Complete |
| 4. Generation layer — prompting, grounding, refusal behaviour | ✅ Complete |
| 5. Evaluation harness — faithfulness, answer relevancy, context precision/recall | ✅ Complete |
| 6. Judge audit — bias probes, inter-judge agreement | ✅ Complete |
| 7. Decision memo + deployment recommendation | ✅ Complete |

---

## Try it

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Most RAG demos show you an answer. This one shows the answer **and the
evidence for whether to believe it** — retrieval, refusal detection,
citation integrity, and the ground-truth evaluation side by side. There's
a third tab that audits the judge, because a score from an uncalibrated
instrument is a number rather than a measurement.

Runs with no API key. See [`app/README.md`](app/README.md) for what's
worth trying.

## Running it

Phases 1–2 need **no API key and no model download**. This is enforced by
CI, which installs only the scientific stack and fails if a credential is
present.

```bash
pip install numpy pandas scikit-learn matplotlib pytest

python -c "from src.corpus.build import write_corpus, write_golden_set; write_corpus(); write_golden_set()"

python notebooks/01_corpus_construction.py    # corpus audit + BM25 floor
python notebooks/02_chunking_embedding.py     # chunking + retrieval baseline
python notebooks/03_retrieval_bakeoff.py      # 48-config sweep (~90s)
python notebooks/04_generation.py             # grounding + refusal (no key needed)
python notebooks/05_evaluation.py             # eval harness (no key needed)
python notebooks/06_judge_analysis.py         # judge audit (no key needed)
python notebooks/07_decision_memo.py          # verifies every memo number

pytest tests/ -q                              # 362 tests
```

Token counts throughout are **estimates** (words × 1.3), computed the
same way on every machine. `tiktoken` is optional and used only to
display an exact count beside the estimate; installing it changes no
result. That separation is deliberate, and
[`tests/test_token_determinism.py`](tests/test_token_determinism.py)
enforces it — an earlier version let the tokenizer decide chunk
boundaries, so the same code produced 202 chunks in CI and 209 on a
machine that had installed `requirements.txt`.

### With a key: the credentialed run

```bash
python scripts/run_llm_eval.py --dry-run      # the plan and its cost; spends nothing
python scripts/run_llm_eval.py                # asks before spending, then runs 01-07
```

One command runs the generation arms and the judge, stops at the first
failure, and keeps the evidence: each notebook's output in
`reports/llm_run/` (scrubbed of credentials, with a manifest of what was
installed) and the measured numbers in `reports/metrics/`. Roughly $3 at
list prices with the default two judged arms; `JUDGE_ARMS=all` judges
every prompt variant. If the judge fails its own validation gate, the LLM
arms are not judged and that money is not spent.

For the transformer embedding arm and the later LLM phases, see
[`SETUP.md`](SETUP.md) — including how to configure a project-scoped key
with a spend cap.

Notebooks exist as paired `.py` and `.ipynb`. The `.py` is the source of
truth and what CI runs; the `.ipynb` is generated from it and committed
with outputs stripped. Either file works from any working directory — the
notebook locates the project root by searching up *and* down for a marker
file, so it resolves whether Jupyter was started inside `notebooks/`, at
the project root, or in a parent folder holding several projects.

`python scripts/build_notebooks.py` regenerates the `.ipynb` files from
their `.py` sources, and `python scripts/check_repo.py` enforces the
repo-wide invariants (stripped outputs, `.py`/`.ipynb` parity, encoding,
no credential shapes) across every file rather than just the current
diff. CI runs both before the tests, and fails if the notebooks have
drifted from their sources.

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
  retrieval/   chunking strategies, embedding backends, vector store, retrieval arms
  llm/         provider abstraction, credential masking, call budget, response cache
  generation/  prompt variants, RAG pipeline, refusal detection,
               extractive non-LLM baseline
  evaluation/  retrieval metrics, paired bootstrap, Holm correction,
               equivalence testing, Pareto frontier, LLM judge,
               judge validation suite, RAG metrics
notebooks/     01 corpus · 02 chunking + embedding · 03 retrieval bake-off
               04 generation, grounding + refusal · 05 evaluation harness
               06 judge audit · 07 decision memo + verification
tests/         362 tests — corpus, difficulty, chunking, retrieval, metrics,
               generation, refusal, provider security
data/          generated corpus + golden set (regenerable)
reports/       decision_memo.md, PROJECT_SUMMARY.md, scenario_brief.md,
               figures
app/           Streamlit demo
```

---

## Related projects

Part of a three-project series sharing the StreamFlix universe:

- **[A/B Test Analysis](https://github.com/janeruxi1/ab-testing-project)** — experiment design, sequential testing, CUPED, sensitivity analysis
- **[Churn & Retention](https://github.com/janeruxi1/StreamFlix-churn-retention)** — churn modelling, causal uplift, cost-aware targeting policy
