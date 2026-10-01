"""Run the real-LLM evaluation and save results to reports/results/.

This is the one step that needs a credential and costs money. Everything it
produces is written as plain JSON so the results can be committed and
re-read (notebook 06, the README table, the decision memo) by anyone
without a key.

Order of operations is the point:

  1. Validate the judge on the labelled suite. If it fails the gate, stop —
     nothing downstream would be evidence.
  2. Run each prompt variant over the golden set (cached on disk).
  3. Score with the judge and the exact context metrics.
  4. Write reports/results/llm_eval.json.

Usage:
    python scripts/run_llm_eval.py --dry-run     # cost estimate, no calls
    python scripts/run_llm_eval.py               # real run
    python scripts/run_llm_eval.py --variants cited strict
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.corpus.build import load_corpus, load_golden_set  # noqa: E402
from src.evaluation.failure_analysis import classify_run, summarise  # noqa: E402
from src.evaluation.judge import get_judge  # noqa: E402
from src.evaluation.judge_validation import VALIDATION_CASES, validate_judge  # noqa: E402
from src.evaluation.rag_metrics import aggregate_run, evaluate_run  # noqa: E402
from src.generation.pipeline import LLMAnswerer, RAGPipeline, score_generation  # noqa: E402
from src.generation.prompts import VARIANTS  # noqa: E402
from src.llm.provider import get_provider, provider_ready  # noqa: E402
from src.retrieval.chunking import STRATEGIES  # noqa: E402
from src.retrieval.retrievers import BM25Retriever  # noqa: E402

STRATEGY, DEPTH = "markdown_section", 15
OUT = ROOT / "reports" / "results" / "llm_eval.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", nargs="+", default=list(VARIANTS))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    golden = load_golden_set()
    n_gen = len(args.variants) * len(golden)
    n_judge = len(args.variants) * (2 * len(golden)) + 2 * len(VALIDATION_CASES)
    print(f"{n_gen} generation calls + ~{n_judge} judge calls "
          f"(responses are cached; re-runs are free)")
    if args.dry_run:
        return 0

    ok, why = provider_ready()
    if not ok:
        print(f"Cannot run: {why}\nSee SETUP.md.")
        return 2

    # The default ceiling (2000) is below a full five-variant run.
    os.environ.setdefault("MAX_LLM_CALLS_PER_RUN", str(n_gen + n_judge + 100))
    provider = get_provider()
    judge = get_judge(provider, model=os.getenv("JUDGE_MODEL", "gpt-4o"))

    validation = validate_judge(judge)
    lo, hi = validation.overall_ci
    print(f"Judge validation: {validation.overall_accuracy:.1%} "
          f"(95% CI {lo:.1%}-{hi:.1%}), trustworthy={validation.is_trustworthy}")
    if not validation.is_trustworthy:
        print("Judge failed validation — stopping before spending the rest.")
        return 3

    articles = load_corpus()
    references = {q["question_id"]: q.get("reference_answer", "") for q in golden}
    retriever = BM25Retriever(STRATEGIES[STRATEGY](articles))

    out = {"config": {"strategy": STRATEGY, "depth": DEPTH,
                      "generation_model": provider.default_model,
                      "judge_model": os.getenv("JUDGE_MODEL", "gpt-4o")},
           "judge_validation": {"accuracy": validation.overall_accuracy,
                                "ci95": [lo, hi],
                                "by_kind": validation.accuracy_by_kind()},
           "variants": {}}

    for name in args.variants:
        print(f"== {name}")
        pipe = RAGPipeline(retriever, LLMAnswerer(provider, VARIANTS[name]),
                           depth=DEPTH)
        results = pipe.run(golden, progress_every=30)
        ev = evaluate_run(results, judge=judge, references=references)
        out["variants"][name] = {
            "generation": asdict(score_generation(results, name)),
            "evaluation": asdict(aggregate_run(ev, name)),
            "failures": {k: dict(v) if hasattr(v, "items") else v
                         for k, v in {"by_cause": summarise(classify_run(ev)).by_cause}.items()},
            "answers": [{"question_id": r.question_id, "answer": r.answer}
                        for r in results],
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, default=float), encoding="utf-8")
    print(f"Wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
