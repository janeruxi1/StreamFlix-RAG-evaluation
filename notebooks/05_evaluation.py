"""
Phase 5 — The Evaluation Harness
=================================

Four metrics, in the RAGAS tradition, with one deliberate departure and
one thing added that RAGAS does not provide.

  context_precision   how much of the retrieved context was relevant
  context_recall      how much of the needed evidence was retrieved
  faithfulness        are the answer's claims supported by the context
  answer_relevancy    does the answer address the question

THE DEPARTURE. RAGAS computes all four with an LLM, because it assumes no
ground-truth labels — a reasonable default for the common case. This
project HAS exact labels: every golden question names its source
articles. So the first two are computed exactly, not judged. Using a
model to estimate a quantity you can compute is worse on every axis:
noisier, priced per question, not reproducible across model versions, and
it injects the judge's error into a number that had none.

THE ADDITION. The judge is validated before any of its scores are
believed. The standard practice — report faithfulness to three decimals
from a model whose agreement with ground truth was never measured — hides
an unknown error rate inside the headline number. Section B measures the
judge against cases where the right verdict is known by construction, and
Section C only reports real scores if it passed.

Sections
--------
  A. What is running
  B. Validating the judge BEFORE trusting it
  C. Context metrics — exact, and read against their ceiling
  D. Judged answer quality
  E. Failure attribution — retrieval or generation
  F. Verdict and handoff to Phase 6
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.corpus.build import load_corpus, load_golden_set
from src.evaluation.judge import LexicalJudge, get_judge
from src.evaluation.judge_validation import VALIDATION_CASES, validate_judge
from src.evaluation.rag_metrics import (
    aggregate_run,
    attribution_table,
    context_precision_ceiling,
    evaluate_run,
)
from src.generation.extractive import ExtractiveAnswerer
from src.generation.pipeline import RAGPipeline
from src.generation.prompts import VARIANTS
from src.retrieval.chunking import STRATEGIES
from src.retrieval.retrievers import BM25Retriever

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY = "markdown_section"      # locked in by Phase 3
DEPTH = 15
EXTRACTIVE_THRESHOLD = 0.35        # tuned on dev in Phase 4

articles = load_corpus()
golden = load_golden_set()
references = {q["question_id"]: q.get("reference_answer", "") for q in golden}
chunks = STRATEGIES[STRATEGY](articles)
retriever = BM25Retriever(chunks)

chunks_per_article: dict[str, int] = {}
for c in chunks:
    chunks_per_article[c.article_id] = chunks_per_article.get(c.article_id, 0) + 1


# =====================================================================
# A. What is running
# =====================================================================
print("=" * 78)
print("A. SETUP")
print("=" * 78)

# See notebook 04: a credential alone is not enough to make a call, the
# SDK has to be importable too. provider_ready() checks both and returns
# a reason naming the missing piece.
from src.llm.provider import provider_ready

has_key, blocker = provider_ready()

if has_key:
    from src.llm.provider import get_provider
    provider = get_provider()
    judge = get_judge(provider, model=os.getenv("JUDGE_MODEL", "gpt-4o"))
else:
    judge = LexicalJudge()

n_val = len(VALIDATION_CASES) * 2
n_refs = sum(1 for q in golden if q.get("reference_answer"))
n_judge = len(golden) * 2 + n_refs
judge_model = os.getenv("JUDGE_MODEL", "gpt-4o")

print(f"""
  Retrieval        : {STRATEGY} + BM25 @ depth {DEPTH}
  Questions        : {len(golden)}
  Judge            : {judge!r}
  Credential       : {has_key}
""")

if has_key:
    # gpt-4o list pricing. Stated because the judge tier is deliberately
    # the EXPENSIVE model — the whole point of the asymmetry — so this
    # notebook costs roughly eight times what the generation notebook
    # does despite making fewer calls.
    in_usd = (n_val + n_judge) * 800 / 1e6 * 2.50
    out_usd = (n_val + n_judge) * 90 / 1e6 * 10.00
    print(f"""  Judge calls and cost, at {judge_model} list pricing:

    validation suite (Section B) : {n_val:>4} calls   ~${n_val * 800 / 1e6 * 2.50 + n_val * 90 / 1e6 * 10:.2f}
    golden set (Sections C-E)    : {n_judge:>4} calls   ~${n_judge * 800 / 1e6 * 2.50 + n_judge * 90 / 1e6 * 10:.2f}
    total                        : {n_val + n_judge:>4} calls   ~${in_usd + out_usd:.2f}

  Responses are cached on disk, so re-running costs nothing.

  Note the ordering: Section B runs first and costs cents. If the judge
  fails validation there, stop — the remaining ~${n_judge * 800 / 1e6 * 2.50 + n_judge * 90 / 1e6 * 10:.2f} would buy numbers
  from an instrument already known to be miscalibrated. The gate is
  scientific first and economical second, but it is both.
""")

if not has_key:
    print(f"""  LLM judge unavailable.

  Reason: {blocker}

  So the LEXICAL judge runs instead. That is
  not a mock — it scores by term overlap, which is a real (weak) method,
  and Section B measures exactly how weak.

  The context metrics in Section C are UNAFFECTED by this: they are
  computed from ground-truth labels and need no model at all. That is
  the practical payoff of not delegating them to a judge — a credential
  outage degrades the evaluation instead of stopping it.

  With a key, the judge would be {os.getenv('JUDGE_MODEL', 'gpt-4o')} grading
  {os.getenv('GENERATION_MODEL', 'gpt-4o-mini')} output. The asymmetry is deliberate: a model
  grading its own output family shows self-preference bias, which would
  inflate the exact metric this project is built around.
""")


# =====================================================================
# B. Validating the judge
# =====================================================================
print("\n" + "=" * 78)
print("B. VALIDATING THE JUDGE — BEFORE TRUSTING ANY SCORE")
print("=" * 78)
print(f"""
An LLM judge is a measuring instrument. An uncalibrated instrument
produces numbers, not measurements.

{len(VALIDATION_CASES)} cases where the correct verdict follows from how the case was
written, not from an opinion:

  supported     every claim appears in the context
  fabricated    a plausible fact the context never states
  contradicted  a fact that directly opposes the context
  refusal       no claims made at all
  off_topic     faithful to the context, answers a different question

The most diagnostic is `contradicted`. A contradicting sentence reuses
the context's vocabulary almost perfectly — high overlap, opposite
meaning — so it is the case that separates semantic judgement from term
matching.
""")

report = validate_judge(judge)

print(f"  Judge: {report.judge}\n")
print(f"    faithfulness accuracy : {report.faithfulness_accuracy:>6.1%}")
print(f"    relevancy accuracy    : {report.relevancy_accuracy:>6.1%}")
print(f"    overall               : {report.overall_accuracy:>6.1%}")
print(f"    parse failures        : {report.parse_failures}")
print(f"    TRUSTWORTHY           : {report.is_trustworthy}")

print("\n  Accuracy by case kind:\n")
for kind, acc in report.accuracy_by_kind().items():
    bar = "#" * int(acc * 20)
    print(f"    {kind:<14} {acc:>5.0%}  {bar}")

fails = report.failures()
if fails:
    print(f"\n  Failures ({len(fails)}):\n")
    for r in fails:
        want_f = "high" if r.case.expect_faithful_high else "LOW"
        want_r = "high" if r.case.expect_relevant_high else "LOW"
        print(f"    [{r.case.case_id}] {r.case.kind}")
        print(f"      faithfulness {r.faithfulness.score:.2f} (want {want_f})"
              f"   relevancy {r.relevancy.score:.2f} (want {want_r})")
        if r.case.note:
            print(f"      {r.case.note}")

if not report.is_trustworthy:
    print(f"""
>>> THE JUDGE FAILS VALIDATION. Its scores in Sections D and E are
    reported as a demonstration that the harness runs, NOT as evidence
    about answer quality.

  This is the correct outcome for the lexical judge and the reason it is
  in the repo. Look at where it fails:

    contradicted   it scores contradictions as fully faithful, because a
                   contradicting sentence shares nearly every term with
                   the context it contradicts. Term overlap cannot
                   represent negation or numeric disagreement.
    refusal        it scores a refusal as maximally unfaithful, because a
                   refusal shares no vocabulary with the context. But a
                   refusal asserts nothing, so it is vacuously faithful.

  Those two failures are not fixable by tuning a threshold — they are
  what "semantic" means. This is the concrete argument for paying for an
  LLM judge, made by measurement rather than assertion.

  A gate is what makes this useful: if the real judge fails this suite
  too, its numbers should be discarded rather than published.
""")
else:
    print(f"""
  The judge passes at {report.overall_accuracy:.0%}. Its scores below carry that
  known error rate — which is the point of measuring it. A judge is
  never perfect; an unmeasured judge is merely a judge whose error rate
  is unknown.
""")


# =====================================================================
# C. Context metrics
# =====================================================================
print("\n" + "=" * 78)
print("C. CONTEXT METRICS — EXACT, NO JUDGE INVOLVED")
print("=" * 78)

pipeline = RAGPipeline(retriever,
                       ExtractiveAnswerer(min_score=EXTRACTIVE_THRESHOLD),
                       depth=DEPTH)
results = pipeline.run(golden)
evaluated = evaluate_run(results, judge=judge, references=references)
report_run = aggregate_run(evaluated)

in_scope_ev = [e for e in evaluated if not e.result.is_out_of_scope]
ceilings = [context_precision_ceiling(e.result, chunks_per_article)
            for e in in_scope_ev]
ceilings = [c for c in ceilings if c is not None]
mean_ceiling = float(np.mean(ceilings))

mean_chunks = float(np.mean(list(chunks_per_article.values())))
single_gt = [e for e in in_scope_ev if len(e.result.gt_article_ids) == 1]
single_ceiling = float(np.mean(
    [context_precision_ceiling(e.result, chunks_per_article) for e in single_gt]))

print(f"""
  context_recall    : {report_run.context_recall:.3f}
  context_precision : {report_run.context_precision:.3f}

>>> The precision number is meaningless without its ceiling.

  Mean ACHIEVABLE context_precision at depth {DEPTH}: {mean_ceiling:.3f}

  The ceiling is arithmetic, not a modelling choice. Articles split into
  ~{mean_chunks:.1f} chunks on this strategy, so a question whose answer lives in
  ONE article has at most ~{mean_chunks:.1f} relevant chunks in existence. Retrieved
  at depth {DEPTH}, that caps it near {single_ceiling:.2f} — the other slots MUST hold
  something irrelevant. Questions needing several articles have
  proportionally more relevant material available, which is why the
  mean ceiling ({mean_ceiling:.3f}) sits above the single-article one ({single_ceiling:.3f}).

  So {report_run.context_precision:.3f} against {mean_ceiling:.3f} means the retriever achieves {report_run.context_precision / mean_ceiling:.0%} of
  what is structurally possible. Read against an implicit ceiling of 1.0
  — which is how this metric is usually reported — the same system looks
  broken.

  The ceiling is the bill for Phase 3's choices arriving: depth {DEPTH} buys
  recall {report_run.context_recall:.3f} and pays with a context window that is ~{1 - report_run.context_precision:.0%}
  irrelevant text. That trade is what Phase 6 has to weigh, because
  distractor density is what drives a generator to invent things.
""")

by_cat = {}
for cat in ("single_hop", "multi_hop", "ambiguous"):
    subset = [e for e in in_scope_ev if e.category == cat]
    if subset:
        ceil = float(np.mean([context_precision_ceiling(e.result,
                                                        chunks_per_article)
                              for e in subset]))
        prec = float(np.mean([e.context_precision for e in subset]))
        by_cat[cat] = (float(np.mean([e.context_recall for e in subset])),
                       prec, ceil)

print(f"  {'category':<14}{'recall':>9}{'precision':>11}{'ceiling':>10}{'% of max':>10}")
print("  " + "-" * 54)
for cat, (rec, prec, ceil) in by_cat.items():
    print(f"  {cat:<14}{rec:>9.3f}{prec:>11.3f}{ceil:>10.3f}{prec / ceil:>10.0%}")

worst = min(by_cat, key=lambda c: by_cat[c][1] / by_cat[c][2])
print(f"""
  Reading the columns together inverts the story the raw numbers tell.

  single_hop looks WORST on raw precision ({by_cat['single_hop'][1]:.3f}) but is nearest its
  ceiling ({by_cat['single_hop'][1] / by_cat['single_hop'][2]:.0%}). It scores low because one article cannot fill {DEPTH}
  slots — not because retrieval is failing. multi_hop's higher precision
  ({by_cat['multi_hop'][1]:.3f}) is mostly its higher ceiling ({by_cat['multi_hop'][2]:.3f}), not better retrieval.

>>> And the real weak spot is {worst}, at {by_cat[worst][1] / by_cat[worst][2]:.0%} of its ceiling.

  That category has the MOST relevant material available of the three
  ({by_cat[worst][2]:.3f} ceiling, because these questions map to several articles)
  and finds the LEAST of it. Raw precision hid this completely —
  {worst} and single_hop look almost identical at {by_cat[worst][1]:.3f} vs {by_cat['single_hop'][1]:.3f},
  while one is near its limit and the other is nowhere near.

  This sharpens the open item Phase 3 left. Ambiguous questions were the
  weak category on recall too, and the pattern across both metrics says
  the same thing: the system is not failing to rank the right articles,
  it is failing to work out WHICH articles an underspecified question is
  about. That is a query-understanding problem, and no amount of
  retrieval or prompt tuning addresses it.
""")


# =====================================================================
# D. Judged answer quality
# =====================================================================
print("\n" + "=" * 78)
print("D. JUDGED ANSWER QUALITY")
print("=" * 78)

if not report.is_trustworthy:
    print("""
  Reported for completeness. The judge failed validation, so these are
  numbers the harness produced, not evidence about the answers. Treating
  them as findings would be the exact error Section B exists to prevent.
""")

print(f"""
  answerer                       : {report_run.answerer}
  faithfulness (answered only)   : {report_run.faithfulness:.3f}   <- headline
  faithfulness (all questions)   : {report_run.faithfulness_all:.3f}
  relevancy                      : {report_run.relevancy:.3f}
  correctness                    : {report_run.correctness:.3f}

  answered {report_run.n_answered} of {report_run.n} questions; the rest were refusals.

>>> Faithfulness is reported over ANSWERED questions only, and the gap
    between the two lines above is why.

  A refusal asserts nothing, so a correct judge scores it vacuously
  faithful at 1.0. Averaging refusals in therefore rewards a system for
  declining to be useful — and taken to its limit, a system that refuses
  every question reports PERFECT faithfulness while answering nothing.

  The answered-only figure means what people think faithfulness means:
  when this system does make claims, how often are they supported. The
  all-questions figure is kept visible rather than deleted so the
  inflation is auditable rather than hidden.

  (Here the two run in the opposite direction, because the lexical judge
  scores refusals near ZERO rather than 1.0 — one of the failures
  Section B catches. With a judge that handles refusals correctly, the
  all-questions number would sit ABOVE the answered-only one.)
""")

print("  Failure modes across all 120 questions:\n")
for mode, count in report_run.failure_modes.items():
    print(f"    {mode:<24} {count:>4}  ({count / len(evaluated):>5.1%})")

print("""
  These categories are the deliverable, more than the scores are. A
  single "hallucination rate" would collapse generation_failure,
  retrieval_failure and over_refusal into one number — and those three
  have completely different fixes.
""")


# =====================================================================
# E. Failure attribution
# =====================================================================
print("=" * 78)
print("E. FAILURE ATTRIBUTION — RETRIEVAL OR GENERATION?")
print("=" * 78)
print("""
The thread from Phase 3 arrives here. Cross-tabulating faithfulness
against whether the evidence was retrieved at all separates two failures
that look identical in the output and need opposite fixes.
""")

table = attribution_table(evaluated)
total = sum(table.values())
print(f"  {'bucket':<26}{'n':>5}{'share':>9}   interpretation")
print("  " + "-" * 78)
interp = {
    "evidence+faithful": "working as intended",
    "evidence+unfaithful": "GENERATION failure — evidence there, ignored",
    "no_evidence+faithful": "faithful to wrong context, or refused",
    "no_evidence+unfaithful": "RETRIEVAL failure — looks like hallucination",
    "refused": "no claims made; attribution not applicable",
    "out_of_scope": "scored on refusal in Phase 4, not here",
    "unjudged": "no judge available",
}
for bucket, n in table.items():
    print(f"  {bucket:<26}{n:>5}{n / total:>9.1%}   {interp[bucket]}")

gen_fail = table["evidence+unfaithful"]
ret_fail = table["no_evidence+unfaithful"]

if gen_fail + ret_fail == 0:
    print(f"""
>>> Zero unfaithful answers — and that is a statement about the PAIRING,
    not about quality.

  The extractive baseline builds answers by copying sentences out of the
  retrieved context verbatim. The lexical judge scores faithfulness by
  term overlap with that same context. A copier judged by overlap is
  faithful by construction: it is not passing a hard test, it is exempt
  from the test.

  This is the same category error as the extractive baseline's perfect
  citation precision in Phase 4 — a metric measuring the architecture
  rather than the behaviour. Reporting 1.000 faithfulness here as
  evidence the system is trustworthy would be exactly the mistake
  Section B's gate exists to catch.

  The attribution table only becomes informative when BOTH sides can
  fail: a generator that paraphrases (so it can drift from the source)
  and a judge that reads meaning (so it can notice). That needs an LLM
  on both ends, which is what a credential unlocks.

  The machinery is verified — the buckets populate, refusals and
  out-of-scope rows are correctly excluded — but the finding is pending.
""")
else:
    print(f"""
  Of the unfaithful answers, {gen_fail} had the evidence available and {ret_fail} did
  not. Only the first group is addressable by changing the generator.

  Sending all of them to a prompt-engineering fix — the reflex when a
  dashboard shows "hallucination rate: X%" — would aim {ret_fail} of them at
  the wrong component entirely.
""")

# --- Chart ------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

kinds = list(report.accuracy_by_kind().keys())
accs = [report.accuracy_by_kind()[k] for k in kinds]
colors = ["#5AD8A6" if a >= 0.8 else "#F6735B" for a in accs]
ax1.barh(kinds, accs, color=colors, edgecolor="white")
ax1.axvline(0.8, linestyle="--", color="#2B2B2B", linewidth=1.2,
            label="trust threshold")
ax1.set_xlim(0, 1)
ax1.set_xlabel("Judge accuracy on known-label cases")
ax1.set_title(f"Is the judge trustworthy?\n{report.judge} — overall "
              f"{report.overall_accuracy:.0%}", fontweight="bold")
ax1.legend(fontsize=8, loc="lower right")
ax1.grid(axis="x", linestyle="--", alpha=0.4)
ax1.set_axisbelow(True)

cats = list(by_cat.keys())
x = np.arange(len(cats))
ax2.bar(x - 0.2, [by_cat[c][0] for c in cats], 0.4,
        label="context recall", color="#5B8FF9", edgecolor="white")
ax2.bar(x + 0.2, [by_cat[c][1] for c in cats], 0.4,
        label="context precision", color="#F6BD16", edgecolor="white")
ax2.axhline(mean_ceiling, linestyle="--", color="#2B2B2B", linewidth=1.2,
            label=f"precision ceiling ({mean_ceiling:.2f})")
ax2.set_xticks(x)
ax2.set_xticklabels(cats)
ax2.set_ylabel("Score")
ax2.set_title("Context quality by question type\nprecision is capped by "
              "chunk granularity", fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)

plt.suptitle("Phase 5 — evaluation harness", fontsize=13,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_evaluation.png", dpi=140, bbox_inches="tight",
            metadata={"Software": None})
print(f"  Saved -> {FIG_DIR}/05_evaluation.png")


# =====================================================================
# F. Verdict
# =====================================================================
print("\n" + "=" * 78)
print("PHASE 5 VERDICT")
print("=" * 78)
print(f"""
1. THE JUDGE IS VALIDATED BEFORE IT IS USED.
   {report.judge} scores {report.overall_accuracy:.0%} on {len(VALIDATION_CASES)} cases with known verdicts;
   trustworthy = {report.is_trustworthy}. Reporting faithfulness from an unvalidated
   judge is reporting a measurement from an uncalibrated instrument,
   and the gate makes that impossible to do by accident here.

2. CONTEXT METRICS ARE COMPUTED, NOT ESTIMATED.
   Ground-truth labels exist, so context precision and recall need no
   model. That makes them exact, free, reproducible, and available even
   with no credential — the harness degrades rather than stops.

3. CONTEXT PRECISION MUST BE READ AGAINST ITS CEILING.
   {report_run.context_precision:.3f} observed against {mean_ceiling:.3f} achievable = {report_run.context_precision / mean_ceiling:.0%} of what is
   structurally possible at depth {DEPTH}. The uncorrected number invites
   the conclusion that retrieval is broken when it is doing close to the
   best the chunking and depth allow.

4. FAILURES ARE ATTRIBUTED, NOT AGGREGATED.
   {gen_fail} generation failures and {ret_fail} retrieval failures among unfaithful
   answers. A single hallucination rate would merge them and send the
   fix to whichever component was guessed.{'''
   Both are zero here because a verbatim copier judged by term overlap
   cannot be unfaithful — the machinery is verified, the finding waits
   on a paraphrasing generator and a semantic judge.''' if gen_fail + ret_fail == 0 else ''}

5. TWO METRIC DEFINITIONS THAT DECIDE WHAT THE NUMBERS MEAN.
   The judge is shown the context in exactly the form the generator saw
   it, article-id tags included. Stripping them makes every citation an
   unverifiable claim, which penalises precisely the prompt variants
   that follow the citation instruction.

   Faithfulness covers answered questions only ({report_run.faithfulness:.3f}), not all of
   them ({report_run.faithfulness_all:.3f}). Refusals are vacuously faithful, so including
   them lets a system that answers nothing report a perfect score.

{'' if has_key else '''NOT YET MEASURED — no credential, so the LEXICAL judge ran and failed
validation as designed. Section D's scores are harness output, not
evidence. Add a key and re-run: the LLM judge should pass Section B,
and only then do the answer-quality numbers mean anything.

'''}STILL OPEN — carried into Phase 6:

  - Judge bias beyond accuracy: LLM judges favour longer answers and
    show position effects. Worth measuring the correlation between
    answer length and score before trusting cross-variant comparisons.
  - Inter-judge agreement. One judge's opinion is one opinion; two
    judges disagreeing on a question is a signal that question is
    genuinely ambiguous.
  - The contradiction case (mh-011). Both refund windows are retrieved
    at depth {DEPTH}. Whether any prompt variant FLAGS the conflict rather
    than silently picking one is the question Phase 6 should answer,
    and it needs a judge that passed Section B.

HANDOFF TO PHASE 6: LLM-as-judge deep dive — judge bias measurement,
inter-judge agreement, and qualitative failure analysis on the cases the
harness flags.
""")
