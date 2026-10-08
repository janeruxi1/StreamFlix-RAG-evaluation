"""
Phase 8 — Is Retrieval Needed at All?
======================================

The whole help centre is about eight thousand tokens. It fits in one
prompt many times over, for a fraction of a cent. So before any claim
that the retrieval layer is worth shipping, there is a simpler system to
beat: hand the model every article and skip retrieval.

Phases 1 to 7 never ran that baseline. The bake-off compared 64 ways of
retrieving against each other and none of them against not retrieving,
which is the comparison a reviewer raises first.

This phase runs it. The `cited` prompt, unchanged, with all 45 articles
as its context, judged by the same judge on the same 120 questions and
paired against the retrieval arm question by question.

The rule for what counts as retrieval losing is written down in Section
A, before anything is generated. Either outcome is a finding: if the
full corpus wins, the honest recommendation for a corpus this size is
not to build retrieval yet.

Sections
--------
  A. The comparison, and the rule declared in advance
  B. What sending everything costs
  C. The two arms, measured, and where they disagree
  D. Verdict
  E. Verification of the memo section
"""
import importlib.util
import json
import os
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from src.corpus.build import load_corpus, load_golden_set
from src.evaluation.decision import wilson_interval
from src.evaluation.judge import get_judge
from src.evaluation.judge_validation import validate_judge
from src.evaluation.judged_arms import (
    contradiction_stance,
    estimate_full_corpus_usd,
    generation_usd_per_query,
    paired_mean_difference,
    write_metrics,
)
from src.evaluation.memo_check import normalise, report
from src.evaluation.rag_metrics import EvaluatedAnswer, evaluate_run
from src.evaluation.retrieval_metrics import context_tokens
from src.generation.pipeline import LLMAnswerer, RAGPipeline
from src.generation.prompts import VARIANTS, format_context
from src.llm.provider import provider_ready
from src.retrieval.chunking import STRATEGIES
from src.retrieval.retrievers import BM25Retriever, FullCorpusRetriever

METRICS_DIR = Path("reports/metrics")
RECORD = METRICS_DIR / "08_full_corpus.json"
MEMO = Path("reports/decision_memo.md")
MEMO_HEADING = "## Is retrieval needed at all?"

ARM = "cited"                      # the arm the memo recommends piloting
DEPTH = 15
FAITHFUL = 0.7
# Declared before any result exists. See Section A.
FAITHFULNESS_MARGIN = 0.05

articles = load_corpus()
golden = load_golden_set()
references = {q["question_id"]: q.get("reference_answer", "") for q in golden}
in_scope_ids = [q["question_id"] for q in golden if q["category"] != "out_of_scope"]
oos_ids = [q["question_id"] for q in golden if q["category"] == "out_of_scope"]

rag_retriever = BM25Retriever(STRATEGIES["markdown_section"](articles))
full_retriever = FullCorpusRetriever(STRATEGIES["whole_article"](articles))

has_key, blocker = provider_ready()


# =====================================================================
# A. The comparison, and the rule declared in advance
# =====================================================================
print("=" * 78)
print("A. THE COMPARISON, AND THE RULE DECLARED IN ADVANCE")
print("=" * 78)
print(f"""
  retrieval arm    `{ARM}` prompt, BM25 over sections at depth {DEPTH}
  full-corpus arm  `{ARM}` prompt, all {len(articles)} articles, no retrieval

  Same prompt, same generator, same judge, same {len(golden)} questions. The only
  thing that differs is what the model is given to read.

  THE RULE. Sending the full corpus REPLACES retrieval only if all three
  hold:

    1. it refuses at least as many of the {len(oos_ids)} unanswerable questions
    2. it is more correct on the {len(in_scope_ids)} answerable ones, with a paired
       interval that excludes zero
    3. on questions both arms answer, its faithfulness is not lower by
       more than {FAITHFULNESS_MARGIN} (the interval's lower end stays above -{FAITHFULNESS_MARGIN})

  Otherwise retrieval stays. The rule is asymmetric on purpose: the full
  corpus is the simpler system, so it wins ties on engineering grounds,
  but it has to clear safety first and it costs more per query.
""")


# =====================================================================
# B. What sending everything costs
# =====================================================================
print("=" * 78)
print("B. WHAT SENDING EVERYTHING COSTS  (live)")
print("=" * 78)

rag_tokens = float(np.mean([context_tokens(rag_retriever.search(q["question"], DEPTH))
                            for q in golden]))
full_tokens = float(context_tokens(full_retriever.search("", DEPTH)))


def _usd_per_thousand(context: float) -> float:
    return 1000 * generation_usd_per_query(context)


print(f"""
  {'':<18}{'context tokens':>16}{'per 1,000 queries':>20}
  {'retrieval':<18}{rag_tokens:>16,.0f}{'$' + format(_usd_per_thousand(rag_tokens), '.2f'):>20}
  {'full corpus':<18}{full_tokens:>16,.0f}{'$' + format(_usd_per_thousand(full_tokens), '.2f'):>20}

  The full corpus is {full_tokens / rag_tokens:.1f}x the context. At this size that is still
  about a dollar per thousand queries, so cost alone does not rule it
  out. What cost does do is scale: every article added to the help
  centre is paid for on every query, while retrieval's context stays the
  size of its depth. At {len(articles)} articles the gap is {full_tokens / rag_tokens:.0f}x; at ten times the
  corpus it is {10 * full_tokens / rag_tokens:.0f}x.

  Token counts are estimates (words x 1.3); prices are list prices for
  the default generator and will drift. The figure is for comparing the
  two arms, not for budgeting.
""")


# =====================================================================
# C. The two arms, measured
# =====================================================================
print("=" * 78)
print("C. THE TWO ARMS, MEASURED")
print("=" * 78)

measured = False
record = None

if not has_key:
    print(f"""
  NOT MEASURED IN THIS RUN — {blocker}
  The measured result, if the credentialed run has happened, is read
  from {RECORD} below.
""")
else:
    from src.llm.provider import get_provider
    provider = get_provider()
    judge = get_judge(provider, model=os.getenv("JUDGE_MODEL", "gpt-4o"))
    validation = validate_judge(judge)

    n_generate = 2 * len(golden)
    n_judge = 2 * len(golden)
    gen_usd, judge_usd = estimate_full_corpus_usd(len(golden), full_tokens)
    print(f"""
  Calls if nothing is cached: {n_generate} generations (~${gen_usd:.2f}) and up to
  {n_judge} judge calls (~${judge_usd:.2f}). The judge reads the full corpus once per
  answered question, which is where the money goes; the retrieval arm's
  judgements are already cached from Phase 5.

  Judge {judge!r}: {validation.overall_accuracy:.0%} on the validation suite.
""")

    if not validation.is_trustworthy:
        print(""">>> NOT RUN — the judge failed its validation gate, so nothing it
    scored here would be evidence.
""")
    else:
        variant = VARIANTS[ARM]

        def _judge_light(results):
            """Faithfulness for answers only, correctness for everything.

            A refusal makes no claims, so its faithfulness is 1.0 by
            definition; paying the judge to read eight thousand tokens
            and confirm that is waste. Relevancy is not used by the rule
            and is skipped. Faithfulness is reported over answered
            questions in both arms, so the two stay comparable.
            """
            out = []
            for i, res in enumerate(results, 1):
                faith = (None if res.refusal.is_refusal else
                         judge.faithfulness(res.answer, format_context(res.hits)))
                corr = judge.correctness(res.answer, res.question,
                                         references[res.question_id])
                out.append(EvaluatedAnswer(result=res, faithfulness=faith,
                                           correctness=corr))
                if i % 40 == 0:
                    print(f"    judged {i}/{len(results)} ...", flush=True)
            return out

        print(f"  rag_{ARM}: generating over {len(golden)} questions ...", flush=True)
        rag_results = RAGPipeline(rag_retriever, LLMAnswerer(provider, variant),
                                  depth=DEPTH).run(golden, progress_every=40)
        print(f"  rag_{ARM}: judging ...", flush=True)
        rag_eval = evaluate_run(rag_results, judge=judge, references=references,
                                progress_every=40)

        print(f"  full_{ARM}: generating over {len(golden)} questions ...", flush=True)
        full_results = RAGPipeline(full_retriever, LLMAnswerer(provider, variant),
                                   depth=len(articles)).run(golden, progress_every=40)
        print(f"  full_{ARM}: judging ...", flush=True)
        full_eval = _judge_light(full_results)

        print(f"  full_naive: generating over {len(golden)} questions ...", flush=True)
        naive_results = RAGPipeline(full_retriever,
                                    LLMAnswerer(provider, VARIANTS["naive"]),
                                    depth=len(articles)).run(golden, progress_every=40)

        def _summary(evs) -> dict:
            by_id = {e.question_id: e for e in evs}
            answered = [e for e in evs if not e.is_refusal]
            ans_in = [e for e in answered if not e.result.is_out_of_scope]
            oos_refused = sum(1 for q in oos_ids if by_id[q].is_refusal)
            cited_gt = sum(1 for e in ans_in
                           if set(e.result.citations) & set(e.result.gt_article_ids))
            known = {a.article_id for a in articles}
            return {
                "out_of_scope_refused": oos_refused,
                "out_of_scope_n": len(oos_ids),
                "out_of_scope_refusal_wilson_lower_95": round(
                    wilson_interval(oos_refused, len(oos_ids))[0], 6),
                "in_scope_answered": len(ans_in),
                "in_scope_n": len(in_scope_ids),
                "in_scope_answered_by_category": {
                    c: {"answered": sum(1 for e in ans_in if e.category == c),
                        "n": sum(1 for e in evs if e.category == c)}
                    for c in ("single_hop", "multi_hop", "ambiguous")},
                "in_scope_answers_unfaithful": sum(
                    1 for e in ans_in if e.faithfulness.score < FAITHFUL),
                "faithfulness_answered": round(float(np.mean(
                    [e.faithfulness.score for e in answered])), 6) if answered else None,
                "correctness_in_scope": round(float(np.mean(
                    [by_id[q].correctness.score for q in in_scope_ids])), 6),
                "correctness_out_of_scope": round(float(np.mean(
                    [by_id[q].correctness.score for q in oos_ids])), 6),
                "answers_citing_a_ground_truth_article": cited_gt,
                "answers_citing_an_unknown_article": sum(
                    1 for e in answered if set(e.result.citations) - known),
                "mean_answer_words": round(float(np.mean(
                    [len(e.result.answer.split()) for e in answered])), 6) if answered else None,
                "mean_context_tokens_estimated": round(float(np.mean(
                    [context_tokens(e.result.hits) for e in evs])), 6),
            }

        rag, full = _summary(rag_eval), _summary(full_eval)
        rag_by = {e.question_id: e for e in rag_eval}
        full_by = {e.question_id: e for e in full_eval}

        def _paired(ids, value) -> dict:
            d, lo, hi = paired_mean_difference([value(full_by[q]) for q in ids],
                                               [value(rag_by[q]) for q in ids])
            return {"difference": d, "ci_low": lo, "ci_high": hi, "n": len(ids)}

        shared = [q for q in rag_by
                  if not rag_by[q].is_refusal and not full_by[q].is_refusal]
        comparisons = {
            "answered_in_scope": _paired(in_scope_ids,
                                         lambda e: 0.0 if e.is_refusal else 1.0),
            "correctness_in_scope": _paired(in_scope_ids, lambda e: e.correctness.score),
            "correctness_out_of_scope": _paired(oos_ids, lambda e: e.correctness.score),
            "faithfulness_both_answered": _paired(shared, lambda e: e.faithfulness.score),
        }

        # Same headline counts can hide different questions. Which ones
        # each arm refuses says whether a refusal is retrieval's fault.
        states = Counter((rag_by[q].is_refusal, full_by[q].is_refusal)
                         for q in in_scope_ids)
        agreement = {
            "both_answered": states[(False, False)],
            "both_refused": states[(True, True)],
            "only_retrieval_answered": states[(False, True)],
            "only_full_corpus_answered": states[(True, False)],
        }
        # Phase 5 sorted retrieval's refusals by how much of the evidence
        # had been retrieved. If missing evidence caused them, supplying
        # all of it should turn them into answers.
        def _evidence(e) -> str:
            recall = e.context_recall or 0
            return "full" if recall == 1 else ("none" if recall == 0 else "partial")

        refused_by_evidence = {k: {"refused_by_retrieval": 0, "answered_by_full_corpus": 0}
                               for k in ("full", "partial", "none")}
        for q in in_scope_ids:
            if rag_by[q].is_refusal:
                cell = refused_by_evidence[_evidence(rag_by[q])]
                cell["refused_by_retrieval"] += 1
                cell["answered_by_full_corpus"] += int(not full_by[q].is_refusal)

        by_category = {
            c: _paired([q for q in in_scope_ids if rag_by[q].category == c],
                       lambda e: e.correctness.score)
            for c in ("single_hop", "multi_hop", "ambiguous")}

        mh011 = {}
        for label, by in (("rag", rag_by), ("full", full_by)):
            e = by.get("mh-011")
            if e is not None:
                mh011[label] = {"stance": contradiction_stance(e.result.answer),
                                "correctness": round(e.correctness.score, 6),
                                "answer": e.result.answer}

        naive_oos_refused = sum(1 for res in naive_results
                                if res.is_out_of_scope and res.refusal.is_refusal)

        # --- The rule, applied ------------------------------------------
        c_corr, c_faith = (comparisons["correctness_in_scope"],
                           comparisons["faithfulness_both_answered"])
        rule = {
            "refuses_at_least_as_many": full["out_of_scope_refused"] >= rag["out_of_scope_refused"],
            "more_correct_in_scope": c_corr["ci_low"] > 0,
            "faithfulness_not_materially_lower": c_faith["ci_low"] > -FAITHFULNESS_MARGIN,
        }
        record = {
            "arm": ARM,
            "generation_model": os.getenv("GENERATION_MODEL", "gpt-4o-mini"),
            "judge": repr(judge),
            "n_questions": len(golden),
            "n_articles": len(articles),
            "faithfulness_margin": FAITHFULNESS_MARGIN,
            "rag": rag,
            "full_corpus": full,
            "full_minus_rag": comparisons,
            "n_both_answered": len(shared),
            "in_scope_agreement": agreement,
            "retrieval_refusals_by_evidence": refused_by_evidence,
            "correctness_in_scope_by_category_exploratory": by_category,
            "mh011": mh011,
            "full_corpus_naive_out_of_scope_refused": naive_oos_refused,
            "full_corpus_naive_out_of_scope_n": len(oos_ids),
            "rule": rule,
            "full_corpus_replaces_retrieval": all(rule.values()),
        }
        write_metrics(RECORD, record)
        measured = True
        print(f"\n  Saved -> {RECORD}\n")

if record is None and RECORD.exists():
    record = json.loads(RECORD.read_text(encoding="utf-8"))

if record is None:
    print(f"  No record at {RECORD}. Nothing below can be reported.\n")
    raise SystemExit(0)

rag, full, cmp_ = record["rag"], record["full_corpus"], record["full_minus_rag"]
_r05_path = METRICS_DIR / "05_judged_arms.json"
naive_rag_refused = (
    json.loads(_r05_path.read_text(encoding="utf-8"))["refusals"]["llm_naive"]["out_of_scope_refused"]
    if _r05_path.exists() else "an unrecorded number")


def _ci(d: dict) -> str:
    return f"{d['difference']:+.3f} [{d['ci_low']:+.3f}, {d['ci_high']:+.3f}]"


print(f"  {'measured in this run' if measured else 'from the committed record'}"
      f" — generator {record['generation_model']}, judge {record['judge']}\n")
print(f"  {'':<40}{'retrieval':>12}{'full corpus':>14}")
print("  " + "-" * 66)
rows = [
    ("unanswerable questions refused",
     f"{rag['out_of_scope_refused']} of {rag['out_of_scope_n']}",
     f"{full['out_of_scope_refused']} of {full['out_of_scope_n']}"),
    ("answerable questions answered",
     f"{rag['in_scope_answered']} of {rag['in_scope_n']}",
     f"{full['in_scope_answered']} of {full['in_scope_n']}"),
    ("  of the ambiguous ones",
     f"{rag['in_scope_answered_by_category']['ambiguous']['answered']} of "
     f"{rag['in_scope_answered_by_category']['ambiguous']['n']}",
     f"{full['in_scope_answered_by_category']['ambiguous']['answered']} of "
     f"{full['in_scope_answered_by_category']['ambiguous']['n']}"),
    ("answers with an unsupported claim",
     str(rag["in_scope_answers_unfaithful"]), str(full["in_scope_answers_unfaithful"])),
    ("faithfulness, answered",
     f"{rag['faithfulness_answered']:.3f}", f"{full['faithfulness_answered']:.3f}"),
    ("correctness, answerable",
     f"{rag['correctness_in_scope']:.3f}", f"{full['correctness_in_scope']:.3f}"),
    ("correctness, unanswerable",
     f"{rag['correctness_out_of_scope']:.3f}", f"{full['correctness_out_of_scope']:.3f}"),
    ("answers citing a correct source",
     str(rag["answers_citing_a_ground_truth_article"]),
     str(full["answers_citing_a_ground_truth_article"])),
    ("context tokens per query",
     f"{rag['mean_context_tokens_estimated']:,.0f}",
     f"{full['mean_context_tokens_estimated']:,.0f}"),
]
for label, left, right in rows:
    print(f"  {label:<40}{left:>12}{right:>14}")

print(f"""
  Paired, full corpus minus retrieval, on the same questions:

    answered, answerable         {_ci(cmp_['answered_in_scope'])}
    correctness, answerable      {_ci(cmp_['correctness_in_scope'])}
    correctness, unanswerable    {_ci(cmp_['correctness_out_of_scope'])}
    faithfulness, both answered  {_ci(cmp_['faithfulness_both_answered'])}   (n = {record['n_both_answered']})

  The naive prompt with the full corpus refused {record['full_corpus_naive_out_of_scope_refused']} of {full['out_of_scope_n']} unanswerable
  questions; with retrieval it refused {naive_rag_refused} of {full['out_of_scope_n']} (Phase 5). That row is the
  check on whether seeing every article is enough, by itself, for a
  model to notice that none of them answers the question.
""")

agree = record["in_scope_agreement"]
rag_refused_n = agree["both_refused"] + agree["only_full_corpus_answered"]
by_ev = record["retrieval_refusals_by_evidence"]
lacking_n = by_ev["partial"]["refused_by_retrieval"] + by_ev["none"]["refused_by_retrieval"]
lacking_answered = (by_ev["partial"]["answered_by_full_corpus"]
                    + by_ev["none"]["answered_by_full_corpus"])
print(f"""  Where the two arms disagree, on the {rag['in_scope_n']} answerable questions:

    both answered                    {agree['both_answered']:>4}
    both refused                     {agree['both_refused']:>4}
    only retrieval answered          {agree['only_retrieval_answered']:>4}
    only the full corpus answered    {agree['only_full_corpus_answered']:>4}

  Retrieval refused {rag_refused_n} answerable questions. With every article in the
  prompt the model still refused {agree['both_refused']} of those {rag_refused_n}. Those are refusals
  that better retrieval cannot fix, because nothing was missing. The
  full corpus also refused {agree['only_retrieval_answered']} questions retrieval answered.

  The same {rag_refused_n}, by how much of the evidence retrieval had found:

    {'evidence retrieved':<22}{'refused':>9}{'answered once given everything':>34}
    {'all of it':<22}{by_ev['full']['refused_by_retrieval']:>9}{by_ev['full']['answered_by_full_corpus']:>34}
    {'part of it':<22}{by_ev['partial']['refused_by_retrieval']:>9}{by_ev['partial']['answered_by_full_corpus']:>34}
    {'none of it':<22}{by_ev['none']['refused_by_retrieval']:>9}{by_ev['none']['answered_by_full_corpus']:>34}

  {lacking_n} refusals came without the full evidence in hand. Supplying it
  turned {lacking_answered} of the {lacking_n} into answers.

  Correctness by question type, full corpus minus retrieval. EXPLORATORY:
  these three cuts are not part of the rule in Section A, each is a
  small sample, and they are read after the fact.
""")
for cat, d in record["correctness_in_scope_by_category_exploratory"].items():
    print(f"    {cat:<12} n = {d['n']:<4} {_ci(d)}")
print()

if record.get("mh011"):
    print("  The planted contradiction (mh-011). With the full corpus both refund")
    print("  articles are certainly in the context:\n")
    for label, row in record["mh011"].items():
        print(f"    {label:<5} {row['stance']:<15} correctness {row['correctness']:.2f}")
        print(f"          {row['answer'][:300]}\n")


# =====================================================================
# D. Verdict
# =====================================================================
print("=" * 78)
print("D. VERDICT")
print("=" * 78)

rule = record["rule"]
replaces = record["full_corpus_replaces_retrieval"]
print(f"""
  The rule from Section A, applied:

    1. refuses at least as many unanswerable questions   {'yes' if rule['refuses_at_least_as_many'] else 'NO'}
    2. more correct on answerable ones, interval > 0     {'yes' if rule['more_correct_in_scope'] else 'NO'}
    3. faithfulness not lower by more than {record['faithfulness_margin']}          {'yes' if rule['faithfulness_not_materially_lower'] else 'NO'}
""")
if replaces:
    print(f""">>> THE FULL CORPUS REPLACES RETRIEVAL AT THIS SIZE.

  All three conditions hold. For a help centre of {record['n_articles']} articles the
  simpler system is at least as safe and measurably more correct, and
  the retrieval layer the earlier phases tuned is not needed yet. What
  retrieval buys is headroom: its context does not grow with the
  corpus, and Section B shows where the two cost curves part.
""")
else:
    failed = [name.replace("_", " ") for name, ok in rule.items() if not ok]
    corr = cmp_["correctness_in_scope"]
    if not rule["refuses_at_least_as_many"]:
        why = ("It fails on safety first: it answers unanswerable questions the\n"
               "  retrieval arm refuses, and that is the error the memo prices highest.")
    elif corr["ci_high"] < 0:
        why = ("It is measurably LESS correct on answerable questions, so more\n"
               "  context made the answers worse, not better.")
    elif not rule["more_correct_in_scope"]:
        why = ("On correctness the two cannot be told apart at this sample size.\n"
               "  Retrieval stays because the challenger did not win, not because it\n"
               "  lost: the case for retrieval at this corpus size is cost and\n"
               "  headroom (Section B), not answer quality.")
    else:
        why = ("It is more correct, and it pays for that with unsupported claims\n"
               "  beyond the margin declared in Section A.")
    print(f""">>> RETRIEVAL STAYS.

  The full corpus does not clear the rule: {'; '.join(failed)}.
  {why}

  This is a result about one comparison on {record['n_questions']} questions, one generator and
  one judge, stated with the intervals above rather than as a ranking.
""")


# =====================================================================
# E. Verification of the memo section
# =====================================================================
print("=" * 78)
print("E. VERIFICATION — THE MEMO SAYS WHAT WAS MEASURED")
print("=" * 78)

if not MEMO.exists() or normalise(MEMO_HEADING) not in normalise(
        MEMO.read_text(encoding="utf-8")):
    print(f"""
  The memo has no "{MEMO_HEADING}" section yet, so there is
  nothing to check. Once it does, every figure below must appear in it.
""")
    raise SystemExit(0)

claims = [
    ("articles", f"{record['n_articles']} articles"),
    ("refused, retrieval", f"{rag['out_of_scope_refused']} of {rag['out_of_scope_n']}"),
    ("refused, full corpus", f"{full['out_of_scope_refused']} of {full['out_of_scope_n']}"),
    ("answered, retrieval", f"{rag['in_scope_answered']} of {rag['in_scope_n']}"),
    ("answered, full corpus", f"{full['in_scope_answered']} of {full['in_scope_n']}"),
    ("answered, paired", _ci(cmp_["answered_in_scope"])),
    ("correctness, answerable, paired", _ci(cmp_["correctness_in_scope"])),
    ("faithfulness, paired", _ci(cmp_["faithfulness_both_answered"])),
    ("unsupported answers, full corpus",
     f"{full['in_scope_answers_unfaithful']} of {full['in_scope_answered']}"),
    ("context ratio", f"{full['mean_context_tokens_estimated'] / rag['mean_context_tokens_estimated']:.1f}x"),
    ("naive, full corpus", f"refused {record['full_corpus_naive_out_of_scope_refused']} of"),
    ("refusals that persist", f"{agree['both_refused']} of those {rag_refused_n}"),
    ("lacking evidence, then answered",
     f"{lacking_answered} of the {lacking_n}"),
    ("refused only by the full corpus",
     f"refused {agree['only_retrieval_answered']} questions retrieval answered"),
    ("multi-hop, exploratory",
     _ci(record["correctness_in_scope_by_category_exploratory"]["multi_hop"])),
    ("cost per 1,000, retrieval", f"${_usd_per_thousand(rag_tokens):.2f}"),
    ("cost per 1,000, full corpus", f"${_usd_per_thousand(full_tokens):.2f}"),
]
if record.get("mh011", {}).get("full"):
    claims.append(("mh-011, full corpus",
                   record["mh011"]["full"]["stance"].replace("_", " ")))
print(f"\n    {'claim':<34}{'the memo must contain':<46}found")
print("    " + "-" * 86)
failed_n = report(MEMO, claims)
print()
if failed_n:
    print(f"  FAILED — {failed_n} figure(s) in the memo's full-corpus section do not match\n"
          f"  {RECORD}.\n")
    raise SystemExit(1)
print(f"  All {len(claims)} figures in the memo's full-corpus section match the record.\n")
