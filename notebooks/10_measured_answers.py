"""
Phase 10 — The Record Behind the Demo
======================================

The memo's numbers are aggregates. A reader who wants to see what
"refused 25 of 25" or "3 of 76 unsupported" looks like needs the answers
themselves, and a public demo cannot call a paid model to produce them.

This phase writes every measured answer, with the judge's score and
stated reason, to one record. The demo replays it. No key, no cost, and
no second code path that could show something the memo does not report.

That last property is checked, not assumed. Section C recomputes the
headline numbers from the per-question rows and compares them with the
records Phases 5 and 8 wrote. CI runs this file, so a demo record that
has drifted from the memo fails the build.

Sections
--------
  A. What is in the record
  B. Writing it (credentialed run only; replays from the response cache)
  C. Consistency with the Phase 5 and Phase 8 records
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.corpus.build import load_corpus, load_golden_set
from src.evaluation.demo_record import (
    ARMS,
    RECORD_PATH,
    answer_entry,
    check_against_records,
    load_record,
)
from src.evaluation.judge import get_judge
from src.evaluation.judge_validation import validate_judge
from src.evaluation.judged_arms import write_metrics
from src.evaluation.rag_metrics import evaluate_run
from src.generation.pipeline import LLMAnswerer, RAGPipeline
from src.generation.prompts import VARIANTS, format_context
from src.llm.provider import provider_ready
from src.retrieval.chunking import STRATEGIES
from src.retrieval.retrievers import BM25Retriever, FullCorpusRetriever

METRICS_DIR = Path("reports/metrics")
DEPTH = 15

articles = load_corpus()
golden = load_golden_set()
references = {q["question_id"]: q.get("reference_answer", "") for q in golden}
known_ids = {a.article_id for a in articles}
has_key, blocker = provider_ready()


# =====================================================================
# A. What is in the record
# =====================================================================
print("=" * 78)
print("A. WHAT IS IN THE RECORD")
print("=" * 78)
print(f"""
  One row per question ({len(golden)}), one entry per arm:
""")
for arm, meta in ARMS.items():
    print(f"    {arm:<12} {meta['title']:<28} {meta['context']}")
print("""
  Each entry holds the answer as generated, whether the harness counted
  it as a refusal, the articles it cited, and the judge's faithfulness
  and correctness scores with the reason the judge gave. Question text,
  reference answers and source labels are not duplicated here; the demo
  reads those from the golden set.
""")


# =====================================================================
# B. Writing it
# =====================================================================
print("=" * 78)
print("B. WRITING THE RECORD")
print("=" * 78)

written = False
if not has_key:
    print(f"""
  NOT WRITTEN IN THIS RUN — {blocker}
  The committed record, if there is one, is checked in Section C.
""")
else:
    from src.llm.provider import get_provider
    provider = get_provider()
    judge = get_judge(provider, model=os.getenv("JUDGE_MODEL", "gpt-4o"))
    validation = validate_judge(judge)
    print(f"""
  Every call below repeats one made in Phase 5 or Phase 8, so on a
  machine that ran those phases this replays from the response cache and
  spends nothing.

  Judge {judge!r}: {validation.overall_accuracy:.0%} on the validation suite.
""")
    if not validation.is_trustworthy:
        print(">>> NOT WRITTEN — the judge failed its validation gate.\n")
    else:
        rag = BM25Retriever(STRATEGIES["markdown_section"](articles))
        full = FullCorpusRetriever(STRATEGIES["whole_article"](articles))
        answers: dict[str, dict] = {q["question_id"]: {} for q in golden}

        for arm, variant in (("rag_naive", "naive"), ("rag_cited", "cited")):
            print(f"  {arm}: generating over {len(golden)} questions ...", flush=True)
            results = RAGPipeline(rag, LLMAnswerer(provider, VARIANTS[variant]),
                                  depth=DEPTH).run(golden, progress_every=40)
            print(f"  {arm}: judging ...", flush=True)
            for ev in evaluate_run(results, judge=judge, references=references,
                                   progress_every=40):
                answers[ev.question_id][arm] = answer_entry(
                    ev.result, ev.faithfulness, ev.correctness, ev.relevancy,
                    known_ids)

        print(f"  full_cited: generating over {len(golden)} questions ...", flush=True)
        results = RAGPipeline(full, LLMAnswerer(provider, VARIANTS["cited"]),
                              depth=len(articles)).run(golden, progress_every=40)
        print("  full_cited: judging ...", flush=True)
        for i, res in enumerate(results, 1):
            # Same calls as Phase 8: a refusal makes no claims, so it is
            # not sent to the judge for faithfulness.
            faith = (None if res.refusal.is_refusal else
                     judge.faithfulness(res.answer, format_context(res.hits)))
            corr = judge.correctness(res.answer, res.question,
                                     references[res.question_id])
            answers[res.question_id]["full_cited"] = answer_entry(
                res, faith, corr, None, known_ids)
            if i % 40 == 0:
                print(f"    judged {i}/{len(results)} ...", flush=True)

        print(f"  full_naive: generating over {len(golden)} questions ...", flush=True)
        results = RAGPipeline(full, LLMAnswerer(provider, VARIANTS["naive"]),
                              depth=len(articles)).run(golden, progress_every=40)
        for res in results:
            answers[res.question_id]["full_naive"] = answer_entry(
                res, known_ids=known_ids)

        write_metrics(RECORD_PATH, {
            "generation_model": os.getenv("GENERATION_MODEL", "gpt-4o-mini"),
            "judge": repr(judge),
            "n_questions": len(golden),
            "retrieval": {"chunking": "markdown_section", "retriever": "bm25",
                          "depth": DEPTH},
            "arms": list(ARMS),
            "answers": answers,
        })
        written = True
        print(f"\n  Saved -> {RECORD_PATH}  ({RECORD_PATH.stat().st_size / 1024:.0f} KB)\n")


# =====================================================================
# C. Consistency with the Phase 5 and Phase 8 records
# =====================================================================
print("=" * 78)
print("C. THE DEMO RECORD IS THE RUN THE MEMO REPORTS")
print("=" * 78)

record = load_record(RECORD_PATH)
r05_path, r08_path = METRICS_DIR / "05_judged_arms.json", METRICS_DIR / "08_full_corpus.json"
if record is None or not r05_path.exists() or not r08_path.exists():
    print(f"""
  Nothing to check: {RECORD_PATH}, {r05_path.name} or {r08_path.name}
  is missing. Run `python scripts/run_llm_eval.py` to produce them.
""")
    raise SystemExit(0)

r05 = json.loads(r05_path.read_text(encoding="utf-8"))
r08 = json.loads(r08_path.read_text(encoding="utf-8"))
checks = check_against_records(record, golden, r05, r08)

print(f"""
  {'written in this run' if written else 'from the committed record'} — generator {record['generation_model']}, judge {record['judge']}

    {'recomputed from the per-question rows':<46}{'records 05/08':>14}{'rows':>12}  match
    {'-' * 80}""")
for label, expected, actual, ok in checks:
    print(f"    {label:<46}{str(expected):>14}{str(actual):>12}  {'yes' if ok else 'NO'}")

failed = [label for label, _, _, ok in checks if not ok]
print()
if failed:
    print(f"  FAILED — {len(failed)} aggregate(s) recomputed from the demo record do not\n"
          f"  match what Phases 5 and 8 measured: {'; '.join(failed)}.\n"
          f"  The demo would show a different run from the one the memo reports.\n")
    raise SystemExit(1)
print(f"  All {len(checks)} aggregates recomputed from the demo record match the\n"
      f"  Phase 5 and Phase 8 records. The demo replays the run the memo reports.\n")
