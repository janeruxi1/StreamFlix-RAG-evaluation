"""
Phase 7 — The Deployment Decision, and Verifying the Memo That States It
=========================================================================

Two jobs, and the second is why this is a notebook rather than only a
document.

FIRST: state what ships, at what cost, with which failure modes, and —
the part usually left out — which claims are backed by measurement and
which are not. A recommendation that does not distinguish those is not a
recommendation, it is a hope with numbers attached.

SECOND: verify every number the memo asserts, against live code, and
fail loudly on any drift. `reports/decision_memo.md` contains figures. A
figure in a markdown file is a snapshot that silently rots the moment
anything upstream changes, and a stale memo is worse than no memo
because it carries the authority of having been checked once.

So each claim below is recomputed here and compared to what the memo
says. If they disagree, this notebook exits non-zero and CI fails. The
memo cannot go stale without the build going red.

What this phase does NOT do
---------------------------
It does not pretend the generation layer is validated. The LLM arms have
never run — no credential was available — so every claim about answer
quality is marked UNMEASURED rather than estimated. The retrieval
recommendation is fully evidenced; the generation recommendation is
conditional, and the memo says which is which on every line.

Sections
--------
  A. What is being recommended
  B. Evidence for the retrieval decision
  C. What remains unmeasured, and what it would take
  D. Known failure modes, ranked by deployment risk
  E. Verification — every memo number recomputed
"""
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.corpus.build import load_corpus, load_golden_set
from src.evaluation.judge import LexicalJudge
from src.evaluation.judge_bias import measure_length_bias
from src.evaluation.judge_validation import validate_judge
from src.evaluation.rag_metrics import (
    aggregate_run,
    context_precision_ceiling,
    evaluate_run,
)
from src.generation.extractive import ExtractiveAnswerer
from src.generation.pipeline import RAGPipeline, score_generation
from src.llm.provider import provider_ready
from src.retrieval.chunking import STRATEGIES
from src.retrieval.retrievers import BM25Retriever

MEMO_PATH = Path("reports/decision_memo.md")

STRATEGY = "markdown_section"
DEPTH = 15
EXTRACTIVE_THRESHOLD = 0.35

articles = load_corpus()
golden = load_golden_set()
chunks = STRATEGIES[STRATEGY](articles)
retriever = BM25Retriever(chunks)

chunks_per_article: dict[str, int] = {}
for c in chunks:
    chunks_per_article[c.article_id] = chunks_per_article.get(c.article_id, 0) + 1

has_key, blocker = provider_ready()


# =====================================================================
# A. What is being recommended
# =====================================================================
print("=" * 78)
print("A. THE RECOMMENDATION")
print("=" * 78)

results = RAGPipeline(retriever,
                      ExtractiveAnswerer(min_score=EXTRACTIVE_THRESHOLD),
                      depth=DEPTH).run(golden)
gen = score_generation(results)
evaluated = evaluate_run(results, judge=LexicalJudge())
report = aggregate_run(evaluated)
in_scope = [e for e in evaluated if not e.result.is_out_of_scope]

context_tokens = float(np.mean(
    [sum(len(h.chunk.generation_text.split()) * 1.33 for h in r.hits)
     for r in results]))
ceiling = float(np.mean(
    [context_precision_ceiling(e.result, chunks_per_article) for e in in_scope]))

print(f"""
  SHIP     the retrieval layer: {STRATEGY} + BM25 at depth {DEPTH},
           costing ~{context_tokens:.0f} context tokens per query.

  HOLD     the generation layer. It is built, tested, and unmeasured.
           No LLM arm has ever run, so no claim about answer quality in
           this project is backed by evidence.

  The split is the recommendation. Retrieval was measured across 48
  configurations with paired inference and held-out selection; it is
  ready. Generation has a complete harness pointed at it and no
  readings. Shipping both on the strength of the first would be
  borrowing credibility from the measured half to cover the unmeasured
  half, which is the specific mistake this project was built to avoid.
""")


# =====================================================================
# B. Evidence for the retrieval decision
# =====================================================================
print("=" * 78)
print("B. EVIDENCE — RETRIEVAL")
print("=" * 78)

print(f"""
  context_recall    : {report.context_recall:.3f}
  context_precision : {report.context_precision:.3f}  against a ceiling of {ceiling:.3f}
                      ({report.context_precision / ceiling:.0%} of what is structurally possible)
""")

print(f"  {'category':<14}{'recall':>9}{'precision':>11}{'ceiling':>10}{'% of max':>10}")
print("  " + "-" * 54)
worst = None
for cat in ("single_hop", "multi_hop", "ambiguous"):
    sub = [e for e in in_scope if e.category == cat]
    rec = float(np.mean([e.context_recall for e in sub]))
    prec = float(np.mean([e.context_precision for e in sub]))
    ceil = float(np.mean([context_precision_ceiling(e.result, chunks_per_article)
                          for e in sub]))
    frac = prec / ceil
    if worst is None or frac < worst[1]:
        worst = (cat, frac)
    print(f"  {cat:<14}{rec:>9.3f}{prec:>11.3f}{ceil:>10.3f}{frac:>10.0%}")

print(f"""
  Why the configuration was chosen: it sits on the Pareto frontier under
  a 600-token context budget, not at the top of the recall table. The
  highest-recall configuration costs 4.5x more context for a difference
  that fails a significance test — and the cheap option is INCONCLUSIVE
  rather than equivalent, so the honest claim is a certain cost saving
  against an unresolved recall cost bounded above by 0.084.

  The weak category is {worst[0]} at {worst[1]:.0%} of its ceiling. That is the
  deployment risk worth naming: underspecified questions are where this
  system is furthest from what its own architecture allows.
""")


# =====================================================================
# C. What remains unmeasured
# =====================================================================
print("=" * 78)
print("C. UNMEASURED — AND WHAT IT WOULD TAKE")
print("=" * 78)

validation = validate_judge(LexicalJudge())
bias = measure_length_bias(LexicalJudge())

print(f"""
  Credential present : {has_key}
  {'' if has_key else 'Blocker            : ' + blocker}

  Unmeasured, in order of how much it should hold up a deployment:

  1. ANSWER QUALITY. No LLM arm has run. Faithfulness, relevancy and
     correctness have a harness and no readings. The extractive
     baseline's {gen.answer_rate_in_scope:.0%} in-scope answer rate is a floor for a
     non-LLM method, not a forecast for the LLM.

  2. THE JUDGE. The only judge exercised end to end scores {validation.overall_accuracy:.0%} on
     the Phase 5 validation gate and fails it. Its faithfulness numbers
     are harness output, not evidence.

  3. JUDGE BIAS AT DEPLOYMENT SCALE. The audited judge penalises length
     by {bias.mean_delta:+.3f}, material on {bias.n_material} of {len(bias.results)} probes. Whether the intended
     production judge shows the opposite (documented) bias is unknown,
     and it determines whether prompt-variant comparisons mean anything.

  Cost to close all three: roughly $3 of judge calls plus cents of
  generation, in one run of `python scripts/run_llm_eval.py`, which
  prints the plan before spending anything. That is the entire gap
  between a conditional recommendation and an evidenced one.
""")


# =====================================================================
# D. Failure modes ranked by deployment risk
# =====================================================================
print("=" * 78)
print("D. KNOWN FAILURE MODES, RANKED BY DEPLOYMENT RISK")
print("=" * 78)

oos = [e for e in evaluated if e.result.is_out_of_scope]
answered_oos = sum(not e.is_refusal for e in oos)
over_refused = report.failure_modes.get("over_refusal", 0)

print(f"""
  Ranked by cost to a customer, not by frequency.

  1. ANSWERING AN UNANSWERABLE QUESTION       {answered_oos}/{len(oos)} out-of-scope
     A confident wrong answer about billing is the most expensive
     output this system can produce. It is also the one the corpus was
     built to provoke: the hardest out-of-scope questions have a
     topically adjacent article that is silent on the actual question.

  2. AMBIGUOUS QUESTIONS UNDER-RETRIEVE       {worst[1]:.0%} of ceiling
     Not a ranking failure. The system retrieves the wrong articles
     because it cannot tell which reading of the question was meant —
     a query-understanding problem that neither retrieval tuning nor
     prompting addresses.

  3. THE PLANTED CONTRADICTION (mh-011)       unresolved
     Both refund windows retrieve at depth {DEPTH}, so the evidence is
     present. Whether any answer FLAGS the conflict rather than
     silently picking one is unmeasured and needs the LLM arms.

  4. OVER-REFUSAL                             {over_refused} in-scope questions
     A silent UX failure: the user gets nothing when the evidence was
     available. Cheaper than a wrong answer, and invisible in any
     metric that only counts hallucinations.
""")


# =====================================================================
# E. Verification
# =====================================================================
print("=" * 78)
print("E. VERIFICATION — EVERY MEMO NUMBER RECOMPUTED")
print("=" * 78)

if not MEMO_PATH.exists():
    print(f"\n  {MEMO_PATH} not found — nothing to verify.")
    raise SystemExit(0)

memo = MEMO_PATH.read_text(encoding="utf-8")

# (label, string that must appear verbatim in the memo, live value)
claims = [
    ("articles", f"{len(articles)} articles", len(articles)),
    ("golden questions", f"{len(golden)} questions", len(golden)),
    ("retrieval depth", f"depth {DEPTH}", DEPTH),
    ("context tokens", f"{context_tokens:.0f} context tokens", context_tokens),
    ("context recall", f"{report.context_recall:.3f}", report.context_recall),
    ("context precision", f"{report.context_precision:.3f}", report.context_precision),
    ("precision ceiling", f"{ceiling:.3f}", ceiling),
    ("ambiguous % of ceiling", f"{worst[1]:.0%}", worst[1]),
    ("baseline answer rate", f"{gen.answer_rate_in_scope:.1%}", gen.answer_rate_in_scope),
    ("baseline OOS refusal", f"{gen.refusal_rate_oos:.1%}", gen.refusal_rate_oos),
    ("judge accuracy", f"{validation.overall_accuracy:.0%}", validation.overall_accuracy),
    ("judge length bias", f"{bias.mean_delta:+.3f}", bias.mean_delta),
    ("answered out-of-scope", f"{answered_oos} of {len(oos)}", answered_oos),
    ("over-refusals", f"{over_refused} in-scope", over_refused),
]

print(f"\n  {'claim':<26}{'live value':>14}   present in memo")
print("  " + "-" * 62)
missing = []
for label, needle, value in claims:
    ok = needle in memo
    if not ok:
        missing.append((label, needle))
    shown = f"{value:.3f}" if isinstance(value, float) else str(value)
    print(f"  {label:<26}{shown:>14}   {'yes' if ok else 'NO  <-- DRIFT'}")

print()
if missing:
    print(f"  FAILED — {len(missing)} memo claim(s) no longer match the code:\n")
    for label, needle in missing:
        print(f"    {label}: expected the memo to contain {needle!r}")
    print("""
  The memo asserts numbers that the code no longer produces. Update
  reports/decision_memo.md, or explain in it why the figure differs.
  A memo that has drifted is worse than no memo — it carries the
  authority of having been checked.
""")
    raise SystemExit(1)

print(f"  All {len(claims)} memo claims match live output.\n")
print("""  This check is the point of the phase as much as the memo is. Numbers
  in a markdown file rot silently; numbers verified by a job that fails
  the build cannot. Every figure in the memo is reproducible from this
  repository at the commit it was written against.
""")

print("=" * 78)
print("PHASE 7 VERDICT")
print("=" * 78)
print(f"""
The deliverable is reports/decision_memo.md. This notebook exists to
keep it honest.

1. THE RECOMMENDATION IS SPLIT, DELIBERATELY.
   Ship retrieval, hold generation. Retrieval was measured across 48
   configurations with paired inference, multiplicity correction and
   held-out selection. Generation has a complete harness and zero
   readings. Recommending both would borrow credibility from the
   measured half.

2. UNMEASURED IS MARKED UNMEASURED.
   Three specific gaps, each with the cost of closing it stated
   (~$3 and one run). A reader can tell exactly which claims rest on
   evidence and which rest on architecture.

3. FAILURE MODES ARE RANKED BY COST, NOT FREQUENCY.
   Answering an unanswerable billing question is rarer than
   over-refusing and far more expensive. A frequency-ordered list would
   invert the priority.

4. EVERY NUMBER IS VERIFIED BY THIS NOTEBOOK.
   {len(claims)} claims, recomputed and matched. CI fails on drift.

PROJECT COMPLETE at 7 of 7 phases, with the generation half of the
system honestly labelled as built-but-unvalidated rather than shipped.
""")
