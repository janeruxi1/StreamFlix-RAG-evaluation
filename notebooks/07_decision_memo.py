"""
Phase 7 — The Deployment Decision, and Verifying the Memo That States It
=========================================================================

Two jobs, and the second is why this is a notebook rather than only a
document.

FIRST: state what ships, at what cost, with which failure modes, and —
the part usually left out — how much each claim can bear. A
recommendation that does not distinguish a measured property from a
small-sample one is not a recommendation, it is a hope with numbers
attached.

SECOND: verify every number the memo asserts, and fail loudly on any
drift. `reports/decision_memo.md` contains figures. A figure in a
markdown file is a snapshot that silently rots the moment anything
upstream changes, and a stale memo is worse than no memo because it
carries the authority of having been checked once.

The memo's numbers come from two places, and this notebook treats them
differently on purpose:

  LIVE      Anything computable without a model — retrieval metrics, the
            extractive baseline, the lexical judge. Recomputed here on
            every run, including in CI.

  RECORDED  Anything that needed a credential or a model download — the
            LLM arms, the LLM judge, the transformer retrieval arm. CI
            has neither by design, so these are checked against the
            committed records in reports/metrics/, which the notebooks
            write only when the measurement actually ran.

A recorded number is weaker evidence than a live one: it shows the memo
agrees with what was measured, not that the measurement would come out
the same today. The verification table says which is which on every
line rather than presenting them as one kind of check.

Sections
--------
  A. What is being recommended
  B. Evidence — retrieval
  C. Evidence — generation
  D. How much the evidence can bear
  E. Known failure modes, ranked by deployment risk
  F. Verification — every memo number checked
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.corpus.build import load_corpus, load_golden_set
from src.evaluation.judge import LexicalJudge
from src.evaluation.judge_validation import validate_judge
from src.evaluation.rag_metrics import (
    aggregate_run,
    context_precision_ceiling,
    evaluate_run,
)
from src.evaluation.retrieval_metrics import context_tokens
from src.generation.extractive import ExtractiveAnswerer
from src.generation.pipeline import RAGPipeline, score_generation
from src.retrieval.chunking import STRATEGIES
from src.retrieval.retrievers import BM25Retriever

MEMO_PATH = Path("reports/decision_memo.md")
METRICS_DIR = Path("reports/metrics")

STRATEGY = "markdown_section"
DEPTH = 15
EXTRACTIVE_THRESHOLD = 0.35
SHIP_ARM = "llm_cited"          # the generation arm the memo is about
CONTRAST_ARM = "llm_naive"      # the tutorial default it is compared to

articles = load_corpus()
golden = load_golden_set()
chunks = STRATEGIES[STRATEGY](articles)
retriever = BM25Retriever(chunks)

chunks_per_article: dict[str, int] = {}
for c in chunks:
    chunks_per_article[c.article_id] = chunks_per_article.get(c.article_id, 0) + 1


def _record(name: str) -> dict | None:
    path = METRICS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


R03 = _record("03_retrieval_arms.json")
R04 = _record("04_generation_arms.json")
R05 = _record("05_judged_arms.json")
R06 = _record("06_judge_audit.json")
records_present = all(r is not None for r in (R03, R04, R05, R06))
# Phase 8 came after this memo's first version and changed its first
# line. Its record is optional here: notebook 08 verifies its own
# section of the memo.
R08 = _record("08_full_corpus.json")
if R08 is None:
    RETRIEVAL_NEEDED = """Whether retrieval improves on sending every article is
           Phase 8's question, and that record is missing."""
else:
    _d = R08["full_minus_rag"]["correctness_in_scope"]
    RETRIEVAL_NEEDED = f"""Retrieval is OPTIONAL at this corpus size. Sending all {R08['n_articles']} articles
           gave the same refusals and answer rate, and correctness
           {_d['difference']:+.3f} [{_d['ci_low']:+.3f}, {_d['ci_high']:+.3f}] (Phase 8). It is kept on cost and headroom."""


def _ci(d: dict) -> str:
    return f"{d['difference']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"


# ---------------------------------------------------------------------
# Live quantities — recomputed on every run
# ---------------------------------------------------------------------
results = RAGPipeline(retriever,
                      ExtractiveAnswerer(min_score=EXTRACTIVE_THRESHOLD),
                      depth=DEPTH).run(golden)
gen = score_generation(results)
evaluated = evaluate_run(results, judge=LexicalJudge())
report = aggregate_run(evaluated)
in_scope = [e for e in evaluated if not e.result.is_out_of_scope]

# One definition of a token everywhere: the deterministic estimate, over
# the same in-scope questions Phase 3 costed. An earlier version of this
# notebook used its own multiplier over all 120 questions, so the memo
# quoted a different context cost from the bake-off it was citing.
ctx_tokens = float(np.mean([context_tokens(e.result.hits) for e in in_scope]))
ceiling = float(np.mean(
    [context_precision_ceiling(e.result, chunks_per_article) for e in in_scope]))
lexical_validation = validate_judge(LexicalJudge())

by_category = {}
for cat in ("single_hop", "multi_hop", "ambiguous"):
    sub = [e for e in in_scope if e.category == cat]
    prec = float(np.mean([e.context_precision for e in sub]))
    ceil = float(np.mean([context_precision_ceiling(e.result, chunks_per_article)
                          for e in sub]))
    by_category[cat] = (float(np.mean([e.context_recall for e in sub])),
                        prec, ceil, prec / ceil)
worst = min(by_category, key=lambda c: by_category[c][3])


# =====================================================================
# A. What is being recommended
# =====================================================================
print("=" * 78)
print("A. THE RECOMMENDATION")
print("=" * 78)

if not records_present:
    print(f"""
  RETRIEVE with {STRATEGY} + BM25 at depth {DEPTH}, costing
           ~{ctx_tokens:.0f} estimated context tokens per query.
           {RETRIEVAL_NEEDED}

  NO RECOMMENDATION on generation. The measured records in
  {METRICS_DIR}/ are missing, so nothing about answer quality can be
  stated here. Run `python scripts/run_llm_eval.py` to produce them.
""")
else:
    ship = R05["runs"][SHIP_ARM]
    ship_ref = R05["refusals"][SHIP_ARM]
    print(f"""
  RETRIEVE with {STRATEGY} + BM25 at depth {DEPTH}, costing
           ~{ctx_tokens:.0f} estimated context tokens per query.
           {RETRIEVAL_NEEDED}

  PILOT    the `{SHIP_ARM.replace('llm_', '')}` generation layer with a person in the loop.
           It refused {ship_ref['out_of_scope_refused']} of {ship_ref['out_of_scope_n']} unanswerable questions, invented no
           citations, and scored {ship['faithfulness_answered']:.3f} faithfulness on the {ship['n_answered']} answers
           it gave. That is enough to put it in front of support agents
           and not enough to put it in front of customers unattended.

  DO NOT   ship the `{CONTRAST_ARM.replace('llm_', '')}` prompt in any form. It never refuses.

  The line between "pilot" and "ship" is drawn by sample size, not by
  the scores. {ship_ref['out_of_scope_refused']} of {ship_ref['out_of_scope_n']} supports a true refusal rate of at least {ship_ref['out_of_scope_refusal_wilson_lower_95']:.1%};
  it cannot exclude about one unanswerable question in {round(1 / (1 - ship_ref['out_of_scope_refusal_wilson_lower_95']))} being
  answered. For a billing assistant that is not a deployable bound.
""")


# =====================================================================
# B. Evidence — retrieval
# =====================================================================
print("=" * 78)
print("B. EVIDENCE — RETRIEVAL  (live)")
print("=" * 78)

print(f"""
  context_recall    : {report.context_recall:.3f}
  context_precision : {report.context_precision:.3f}  against a ceiling of {ceiling:.3f}
                      ({report.context_precision / ceiling:.0%} of what is structurally possible)
  context cost      : ~{ctx_tokens:.0f} estimated tokens per query
""")

print(f"  {'category':<14}{'recall':>9}{'precision':>11}{'ceiling':>10}{'% of max':>10}")
print("  " + "-" * 54)
for cat, (rec, prec, ceil, frac) in by_category.items():
    print(f"  {cat:<14}{rec:>9.3f}{prec:>11.3f}{ceil:>10.3f}{frac:>10.0%}")

print(f"""
  The weak category is {worst} at {by_category[worst][3]:.0%} of its ceiling: underspecified
  questions are where this system is furthest from what its own
  architecture allows.
""")

if R03 is not None:
    top, cmp_top = R03["top_of_table"], R03["top_vs_shipping"]
    alt = R03["budget_pick_vs_bm25"]
    print(f"""  From the recorded {R03['n_configurations']}-configuration sweep (transformer arm included):

    What the 600-token budget costs. The top of the table,
    {top['config']}, reaches {top['recall']:.3f} recall at
    {cmp_top['context_ratio']:.1f}x the context: {_ci(cmp_top)}, {cmp_top['verdict'].upper()}.
    The budget is a trade with a measured price, not a free saving.
""")
    if alt is not None:
        cats = alt["recall_by_category"]
        print(f"""    Why BM25 and not the transformer. On the same chunks at the same
    depth and cost, {R03['budget_pick_600']['config']} scores {_ci(alt)}
    against BM25: {alt['verdict'].upper()}, with {alt['n_differing']} of {alt['n_questions']} questions differing.
    An unresolved difference does not justify a model dependency, so
    the simpler arm ships. The two fail on different questions:

      single_hop  transformer {cats['single_hop']['budget_pick']:.3f}   bm25 {cats['single_hop']['bm25']:.3f}
      multi_hop   transformer {cats['multi_hop']['budget_pick']:.3f}   bm25 {cats['multi_hop']['bm25']:.3f}
      ambiguous   transformer {cats['ambiguous']['budget_pick']:.3f}   bm25 {cats['ambiguous']['bm25']:.3f}
""")


# =====================================================================
# C. Evidence — generation
# =====================================================================
print("=" * 78)
print("C. EVIDENCE — GENERATION  (recorded)")
print("=" * 78)

if not records_present:
    print(f"""
  No records in {METRICS_DIR}/. The extractive baseline's {gen.answer_rate_in_scope:.1%} in-scope
  answer rate and {gen.refusal_rate_oos:.1%} out-of-scope refusal rate are a floor for a
  non-LLM method, not a forecast for an LLM.
""")
else:
    jv = R05["judge_validation"]
    arms4 = R04["arms"]
    base = R05["runs"]["extractive_baseline"]
    contrast = R05["runs"][CONTRAST_ARM]
    contrast_ref = R05["refusals"][CONTRAST_ARM]
    pair = f"{CONTRAST_ARM} - {SHIP_ARM}"
    same = R05["same_question_comparison"][pair]
    scope = R05["paired_correctness_by_scope"]

    print(f"""
  Generator {R05['generation_model']}, judged by {R05['judge']}.
  The judge scored {jv['overall_accuracy']:.0%} on {jv['n_cases']} validation cases before any of its
  scores were used (the lexical judge it replaced scores {lexical_validation.overall_accuracy:.0%}).

  {'arm':<22}{'refuse OOS':>11}{'answer in':>11}{'faithful':>10}{'correct':>9}{'fab. cites':>12}
  {'-' * 75}""")
    for name in ("extractive_baseline", CONTRAST_ARM, SHIP_ARM):
        a4, r5 = arms4[name], R05["runs"][name]
        print(f"  {name:<22}{a4['refusal_rate_out_of_scope']:>11.1%}"
              f"{a4['answer_rate_in_scope']:>11.1%}{r5['faithfulness_answered']:>10.3f}"
              f"{r5['correctness']:>9.3f}{a4['fabricated_citations']:>12}")

    print(f"""
  1. The LLM earns its cost over the baseline. Judged correctness
     {ship['correctness']:.3f} against {base['correctness']:.3f} for the extractive answerer.

  2. The refusal instruction is what separates the arms. `{CONTRAST_ARM}` answered
     all {contrast_ref['out_of_scope_n']} unanswerable questions, and {contrast_ref['out_of_scope_answers_judged_unsupported']} of those answers contain
     claims the judge found unsupported. `{SHIP_ARM}` refused all {ship_ref['out_of_scope_n']}.

  3. When both arms answer, they are equally faithful. The headline
     gap ({ship['faithfulness_answered']:.3f} vs {contrast['faithfulness_answered']:.3f}) is composition: each arm is averaged over
     the questions IT chose to answer. On the {same['n_both_answered']} both answered,
     {same['faithfulness'][SHIP_ARM]:.3f} vs {same['faithfulness'][CONTRAST_ARM]:.3f}: {_ci(same['faithfulness'])} ({CONTRAST_ARM} minus {SHIP_ARM}).

  4. Refusing has a measured price. Correctness, {CONTRAST_ARM} minus {SHIP_ARM}:
       in-scope      {_ci(scope[pair + ' (in-scope)'])}
       out-of-scope  {_ci(scope[pair + ' (out-of-scope)'])}
     The two effects nearly cancel overall ({_ci(R05['paired_correctness'][pair])}),
     which is why a single correctness number would have hidden both.

  5. Over-refusal looks like a retrieval problem. `{SHIP_ARM}` declined {ship_ref['in_scope_refused']} of
     {len(in_scope)} answerable questions; the evidence was fully retrieved for
     {ship_ref['in_scope_refused_with_full_evidence']}, partly for {ship_ref['in_scope_refused_with_partial_evidence']} and not at all for {ship_ref['in_scope_refused_with_no_evidence']}. So {ship_ref['in_scope_refused_with_partial_evidence'] + ship_ref['in_scope_refused_with_no_evidence']} of the {ship_ref['in_scope_refused']} were
     declined without the full evidence in hand. Whether missing
     evidence CAUSED them is a separate question, and Phase 8 tests
     it by supplying every article. {ship_ref['in_scope_refused_by_category']['ambiguous']['refused']} of the {ship_ref['in_scope_refused_by_category']['ambiguous']['n']} ambiguous questions
     were refused.
""")


# =====================================================================
# D. How much the evidence can bear
# =====================================================================
print("=" * 78)
print("D. HOW MUCH THE EVIDENCE CAN BEAR")
print("=" * 78)

if records_present:
    lp, chk, agree = (R06["length_probes"], R06["length_extrapolation_check"],
                      R06["agreement_with_lexical"])
    mh = R05["mh011"]
    print(f"""
  1. THE SAFETY NUMBER IS A SMALL SAMPLE. {ship_ref['out_of_scope_refused']} of {ship_ref['out_of_scope_n']} is a lower bound of
     {ship_ref['out_of_scope_refusal_wilson_lower_95']:.1%} on the true refusal rate, on questions written by the
     same person who wrote the corpus.

  2. THE JUDGE IS VALIDATED, NOT CERTIFIED. {jv['n_cases']} cases with known verdicts
     is a floor test. On the bias probes the longer answer scored
     lower on {lp['n_material']} of {lp['n']} probes ({lp['mean_delta']:+.3f} mean), and the judge's stated reasons
     name specific added claims rather than length.""" + (f""" Checked against
     real answers {abs(chk['word_gap']):.0f} words apart, a per-word penalty predicts
     {chk['predicted']:+.3f} and the data shows {chk['observed']:+.3f} [{chk['ci_low']:+.3f}, {chk['ci_high']:+.3f}].""" if chk else "") + f"""
     Agreement with the lexical judge is kappa {agree['kappa']:.3f}, which counts
     against the lexical judge. No second validated rater exists.

  3. THE JUDGE IS LENIENT ON OMISSION. On the planted contradiction
     (mh-011) `{CONTRAST_ARM}` is `{mh[CONTRAST_ARM]['stance']}` and `{SHIP_ARM}` is `{mh[SHIP_ARM]['stance']}`,
     and the judge scored them {mh[CONTRAST_ARM]['correctness']:.2f} and {mh[SHIP_ARM]['correctness']:.2f} against a reference
     that names the conflict. It checks the claims an answer makes; a
     missing caveat is not a false claim.

  4. ONE MODEL, ONE RUN, ONE CORPUS. Temperature 0, a single generator
     version, {len(articles)} synthetic articles. None of the numbers above says
     anything about a different model or a real help centre.
""")


# =====================================================================
# E. Failure modes ranked by deployment risk
# =====================================================================
print("=" * 78)
print("E. KNOWN FAILURE MODES, RANKED BY DEPLOYMENT RISK")
print("=" * 78)

if records_present:
    gen_fail = ship["failure_modes"].get("generation_failure", 0)
    print(f"""
  Ranked by cost to a customer, not by frequency.

  1. TWO SOURCES DISAGREE AND THE ANSWER DOES NOT SAY SO.
     mh-011: `{SHIP_ARM}` is `{mh[SHIP_ARM]['stance']}`, `{CONTRAST_ARM}` is `{mh[CONTRAST_ARM]['stance']}`. No judged
     arm states both refund windows. This is a corpus defect a prompt
     cannot repair, and it is the one failure the judge under-scores.

  2. ANSWERING AN UNANSWERABLE QUESTION. Not observed for `{SHIP_ARM}`
     ({ship_ref['out_of_scope_refused']} of {ship_ref['out_of_scope_n']} out-of-scope refused), and bounded only at {ship_ref['out_of_scope_refusal_wilson_lower_95']:.1%}.

  3. AN UNSUPPORTED CLAIM IN A CITED ANSWER. {gen_fail} of {ship['n_answered']} answers fell below
     the faithfulness threshold. A citation makes an answer checkable,
     not correct.

  4. OVER-REFUSAL. {ship_ref['in_scope_refused']} of {len(in_scope)} answerable questions declined. A silent
     failure: the customer gets nothing, and no hallucination metric
     records it.

  5. AMBIGUOUS QUESTIONS. Retrieval reaches {by_category['ambiguous'][3]:.0%} of its precision
     ceiling there, and generation then refuses most of them. Query
     understanding, which neither retrieval tuning nor prompting fixes.
""")


# =====================================================================
# F. Verification
# =====================================================================
print("=" * 78)
print("F. VERIFICATION — EVERY MEMO NUMBER CHECKED")
print("=" * 78)

if not MEMO_PATH.exists():
    print(f"\n  {MEMO_PATH} not found — nothing to verify.")
    raise SystemExit(0)

# Compared with line wrapping and bold markers removed. The check is
# about whether the memo STATES a number, and a phrase that happens to
# break across two lines of markdown still states it. Without this, the
# check fails on re-flowing a paragraph — a false alarm that teaches
# people to stop reading the failures.
memo = " ".join(MEMO_PATH.read_text(encoding="utf-8").replace("**", "").split())

# (label, source, string that must appear verbatim in the memo)
claims: list[tuple[str, str, str]] = [
    ("articles", "live", f"{len(articles)} articles"),
    ("golden questions", "live", f"{len(golden)} questions"),
    ("retrieval depth", "live", f"depth {DEPTH}"),
    ("context cost", "live", f"{ctx_tokens:.0f} estimated context tokens"),
    ("context recall", "live", f"{report.context_recall:.3f}"),
    ("context precision", "live", f"{report.context_precision:.3f}"),
    ("precision ceiling", "live", f"{ceiling:.3f}"),
    ("ambiguous % of ceiling", "live", f"{by_category['ambiguous'][3]:.0%} of"),
    ("baseline answer rate", "live", f"{gen.answer_rate_in_scope:.1%}"),
    ("baseline OOS refusal", "live", f"{gen.refusal_rate_oos:.1%}"),
    ("lexical judge accuracy", "live", f"scores {lexical_validation.overall_accuracy:.0%}"),
]

missing_records = [n for n, r in (("03_retrieval_arms.json", R03),
                                  ("04_generation_arms.json", R04),
                                  ("05_judged_arms.json", R05),
                                  ("06_judge_audit.json", R06)) if r is None]

if R03 is not None:
    claims += [
        ("configurations swept", "record 03", f"{R03['n_configurations']} configurations"),
        ("top-of-table recall", "record 03", f"{R03['top_of_table']['recall']:.3f} recall"),
        ("budget: recall given up", "record 03", _ci(R03["top_vs_shipping"])),
        ("budget: context ratio", "record 03", f"{R03['top_vs_shipping']['context_ratio']:.1f}x"),
    ]
    if R03["budget_pick_vs_bm25"] is not None:
        alt = R03["budget_pick_vs_bm25"]
        cats = alt["recall_by_category"]
        claims += [
            ("transformer recall", "record 03", f"{R03['budget_pick_600']['recall']:.3f}"),
            ("transformer vs BM25", "record 03", _ci(alt)),
            ("questions differing", "record 03", f"{alt['n_differing']} of {alt['n_questions']} questions"),
            ("ambiguous recall, both arms", "record 03",
             f"{cats['ambiguous']['bm25']:.3f} | {cats['ambiguous']['budget_pick']:.3f}"),
            ("multi-hop recall, both arms", "record 03",
             f"{cats['multi_hop']['bm25']:.3f} | {cats['multi_hop']['budget_pick']:.3f}"),
            ("single-hop recall, both arms", "record 03",
             f"{cats['single_hop']['bm25']:.3f} | {cats['single_hop']['budget_pick']:.3f}"),
        ]

if R04 is not None:
    a_ship, a_con = R04["arms"][SHIP_ARM], R04["arms"][CONTRAST_ARM]
    claims += [
        ("cited: in-scope answer rate", "record 04", f"{a_ship['answer_rate_in_scope']:.1%}"),
        ("cited: fabricated citations", "record 04", f"{a_ship['fabricated_citations']} fabricated citations"),
        ("cited: refusal F1", "record 04", f"{a_ship['refusal_f1']:.3f}"),
        ("naive: answer length", "record 04", f"{a_con['mean_answer_words']:.0f} words"),
        ("cited: answer length", "record 04", f"{a_ship['mean_answer_words']:.0f} words"),
    ]

if R05 is not None:
    jv = R05["judge_validation"]
    s, c, b = (R05["runs"][SHIP_ARM], R05["runs"][CONTRAST_ARM],
               R05["runs"]["extractive_baseline"])
    sr, cr = R05["refusals"][SHIP_ARM], R05["refusals"][CONTRAST_ARM]
    pair = f"{CONTRAST_ARM} - {SHIP_ARM}"
    same = R05["same_question_comparison"][pair]
    scope = R05["paired_correctness_by_scope"]
    mh = R05["mh011"]
    claims += [
        ("judge validation", "record 05", f"{jv['overall_accuracy']:.0%} on {jv['n_cases']} validation cases"),
        ("cited: OOS refused", "record 05", f"{sr['out_of_scope_refused']} of {sr['out_of_scope_n']} out-of-scope"),
        ("cited: Wilson lower bound", "record 05", f"{sr['out_of_scope_refusal_wilson_lower_95']:.1%}"),
        ("cited: faithfulness", "record 05", f"{s['faithfulness_answered']:.3f}"),
        ("cited: answers given", "record 05", f"{s['n_answered']} answers"),
        ("cited: correctness", "record 05", f"{s['correctness']:.3f}"),
        ("cited: below threshold", "record 05",
         f"{s['failure_modes'].get('generation_failure', 0)} of {s['n_answered']}"),
        ("naive: faithfulness", "record 05", f"{c['faithfulness_answered']:.3f}"),
        ("naive: correctness", "record 05", f"{c['correctness']:.3f}"),
        ("naive: OOS unsupported", "record 05",
         f"{cr['out_of_scope_answers_judged_unsupported']} of {cr['out_of_scope_n']}"),
        ("baseline: judged correctness", "record 05", f"{b['correctness']:.3f}"),
        ("same questions: n", "record 05", f"{same['n_both_answered']} questions both"),
        ("same questions: faithfulness", "record 05",
         f"{same['faithfulness'][SHIP_ARM]:.3f} against {same['faithfulness'][CONTRAST_ARM]:.3f}"),
        ("same questions: difference", "record 05", _ci(same["faithfulness"])),
        ("correctness gap, in-scope", "record 05", _ci(scope[pair + " (in-scope)"])),
        ("correctness gap, out-of-scope", "record 05", _ci(scope[pair + " (out-of-scope)"])),
        ("correctness gap, overall", "record 05", _ci(R05["paired_correctness"][pair])),
        ("cited: in-scope refused", "record 05", f"{sr['in_scope_refused']} of {len(in_scope)}"),
        ("refusals by evidence", "record 05",
         f"fully retrieved for {sr['in_scope_refused_with_full_evidence']}, "
         f"partly for {sr['in_scope_refused_with_partial_evidence']} and not at all for "
         f"{sr['in_scope_refused_with_no_evidence']}"),
        ("ambiguous refused", "record 05",
         f"{sr['in_scope_refused_by_category']['ambiguous']['refused']} of the "
         f"{sr['in_scope_refused_by_category']['ambiguous']['n']} ambiguous"),
        ("mh-011: cited stance", "record 05", f"`{mh[SHIP_ARM]['stance']}`"),
        ("mh-011: naive stance", "record 05", f"`{mh[CONTRAST_ARM]['stance']}`"),
        ("mh-011: judge scores", "record 05",
         f"{mh[CONTRAST_ARM]['correctness']:.2f} and {mh[SHIP_ARM]['correctness']:.2f}"),
    ]

if R06 is not None:
    lp, agree = R06["length_probes"], R06["agreement_with_lexical"]
    claims += [
        ("length probes affected", "record 06", f"{lp['n_material']} of {lp['n']} probes"),
        ("length probes: mean delta", "record 06", f"{lp['mean_delta']:+.3f}"),
        ("kappa vs lexical judge", "record 06", f"kappa {agree['kappa']:.3f}"),
    ]
    if R06["length_extrapolation_check"]:
        claims.append(("length: predicted gap", "record 06",
                       f"{R06['length_extrapolation_check']['predicted']:+.3f}"))

print(f"\n  {'claim':<32}{'source':<11}{'memo must contain':<40}found")
print("  " + "-" * 88)
missing = []
for label, source, needle in claims:
    ok = needle in memo
    if not ok:
        missing.append((label, source, needle))
    shown = needle if len(needle) <= 38 else needle[:35] + "..."
    print(f"  {label:<32}{source:<11}{shown:<40}{'yes' if ok else 'NO  <-- DRIFT'}")

n_live = sum(1 for _, src, _ in claims if src == "live")
print()
if missing_records:
    print(f"  FAILED — measured record(s) missing: {', '.join(missing_records)}\n")
    print("""  The memo makes claims about measurements whose records are not in
  reports/metrics/. Either the records were not committed, or the memo
  is describing a run that did not happen. Run
  `python scripts/run_llm_eval.py` and commit what it writes.
""")
    raise SystemExit(1)

if missing:
    print(f"  FAILED — {len(missing)} memo claim(s) no longer match:\n")
    for label, source, needle in missing:
        print(f"    {label} ({source}): expected the memo to contain {needle!r}")
    print("""
  The memo asserts numbers that the code or the records no longer
  produce. Update reports/decision_memo.md, or explain in it why the
  figure differs. A memo that has drifted is worse than no memo — it
  carries the authority of having been checked.
""")
    raise SystemExit(1)

print(f"  All {len(claims)} memo claims match: {n_live} recomputed live, "
      f"{len(claims) - n_live} against the measured records.\n")
print("""  The two kinds of check are not the same strength, and the split is
  shown rather than hidden. A live claim is reproduced from this
  repository on every run. A recorded claim shows the memo agrees with
  what was measured when the credentialed run happened; re-running
  `scripts/run_llm_eval.py` regenerates the records, and any change in
  them fails this check until the memo is updated to match.
""")

print("=" * 78)
print("PHASE 7 VERDICT")
print("=" * 78)
print(f"""
The deliverable is reports/decision_memo.md. This notebook exists to
keep it honest.

1. THE RECOMMENDATION HAS THREE PARTS, AND THEY REST ON DIFFERENT
   AMOUNTS OF EVIDENCE.
   Retrieval: the configuration is measured across the full sweep
   with paired inference and held-out selection, and Phase 8 found it
   optional at this corpus size. Pilot cited generation: measured,
   on a sample too small to bound the failure that matters. Do not
   ship the naive prompt: it never refuses.

2. SMALL SAMPLES ARE STATED AS BOUNDS.
   A clean sweep on 25 questions is reported with its Wilson lower
   bound, because the count alone reads as a guarantee it is not.

3. FAILURE MODES ARE RANKED BY COST, NOT FREQUENCY.
   The contradiction no arm surfaces is one question and ranks first.

4. EVERY NUMBER IS CHECKED, AND THE CHECK SAYS HOW.
   {len(claims)} claims: {n_live} recomputed live, {len(claims) - n_live} against committed records.
   CI fails on drift in either.

The memo's seven phases are complete. Phases 8 and 9 test its
conclusions: 08 against using no retrieval at all, 09 against a stated
cost of error.
""")
