"""
Phase 6 — Failure Analysis
===========================

Phase 5 produced scores. This phase asks what to DO about them: for every
failing question, which component owns the fix?

  retrieval   evidence never reached the generator (or only part did)
  generation  evidence was there and the answer was still wrong
  corpus      the knowledge base itself is the root cause (tagged, not
              exclusive — a question can fail via retrieval AND touch a
              contradictory article)

Everything here runs without an API key, on the extractive baseline. If
`reports/results/llm_eval.json` exists (written by
`scripts/run_llm_eval.py`), the LLM variants are analysed the same way and
shown alongside — so the section structure does not change when a key is
added, only the number of rows.

Sections
--------
  A. Failure taxonomy for the keyless baseline
  B. Where failures concentrate — by category and by corpus flaw
  C. LLM variants, if results exist
  D. What to fix first
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from src.corpus.build import load_corpus, load_golden_set
from src.evaluation.failure_analysis import (
    OWNER,
    classify_run,
    render_markdown,
    summarise,
)
from src.evaluation.rag_metrics import evaluate_run
from src.generation.extractive import ExtractiveAnswerer
from src.generation.pipeline import RAGPipeline
from src.retrieval.chunking import STRATEGIES
from src.retrieval.retrievers import BM25Retriever

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
RESULTS = Path("reports/results/llm_eval.json")

STRATEGY, DEPTH, THRESHOLD = "markdown_section", 15, 0.35

articles = load_corpus()
golden = load_golden_set()
retriever = BM25Retriever(STRATEGIES[STRATEGY](articles))

# =====================================================================
# A. Failure taxonomy
# =====================================================================
print("=" * 78)
print("A. FAILURE TAXONOMY — extractive baseline, markdown_section + BM25 @ 15")
print("=" * 78)

pipe = RAGPipeline(retriever, ExtractiveAnswerer(min_score=THRESHOLD), depth=DEPTH)
evaluated = evaluate_run(pipe.run(golden), judge=None)
records = classify_run(evaluated)
summary = summarise(records)

print(f"\n  {summary.n} questions, {summary.n_failures} failures "
      f"({summary.n_failures / summary.n:.0%})\n")
print(f"  {'cause':<20}{'owner':<13}{'n':>4}")
print("  " + "-" * 37)
for cause, n in summary.by_cause.most_common():
    print(f"  {cause:<20}{OWNER[cause]:<13}{n:>4}")

# =====================================================================
# B. Where failures concentrate
# =====================================================================
print("\n" + "=" * 78)
print("B. WHERE FAILURES CONCENTRATE")
print("=" * 78)

df = pd.DataFrame([{"category": r.category, "cause": r.cause} for r in records])
ct = pd.crosstab(df["category"], df["cause"])
print("\n" + ct.to_string())

if summary.flaw_exposure:
    print("\n  Questions whose ground truth includes a documented corpus flaw:\n")
    for flaw, c in sorted(summary.flaw_exposure.items()):
        tot = c.get("failed", 0) + c.get("passed", 0)
        print(f"    {flaw:<32} {c.get('failed', 0)}/{tot} failed")

fig, ax = plt.subplots(figsize=(8, 4.5))
ct.plot.barh(stacked=True, ax=ax, colormap="tab20")
ax.set_xlabel("questions")
ax.set_title("Failure causes by question category (extractive baseline)")
ax.legend(fontsize=7, loc="lower right")
fig.tight_layout()
fig.savefig(FIG_DIR / "06_failure_analysis.png", dpi=130)
plt.close(fig)

# =====================================================================
# C. LLM variants, if results exist
# =====================================================================
print("\n" + "=" * 78)
print("C. LLM VARIANTS")
print("=" * 78)

if RESULTS.exists():
    data = json.loads(RESULTS.read_text(encoding="utf-8"))
    print(f"\n  Generation {data['config']['generation_model']}, "
          f"judge {data['config']['judge_model']}\n")
    print(f"  {'variant':<18}{'refuse_oos':>11}{'answer_in':>11}{'faith':>8}{'rel':>7}")
    for name, v in data["variants"].items():
        g, e = v["generation"], v["evaluation"]
        print(f"  {name:<18}{g['refusal_rate_oos']:>11.1%}"
              f"{g['answer_rate_in_scope']:>11.1%}"
              f"{e['faithfulness']:>8.3f}{e['relevancy']:>7.3f}")
else:
    print("""
  No LLM results found (reports/results/llm_eval.json).

  This is expected without a credential. To produce them:

      python scripts/run_llm_eval.py --dry-run     # cost estimate
      python scripts/run_llm_eval.py               # real run

  Until then, no claim in this repo is made about LLM answer quality.
""")

# =====================================================================
# D. What to fix first
# =====================================================================
print("=" * 78)
print("D. WHAT TO FIX FIRST")
print("=" * 78)
owners = summary.by_owner
print(f"\n  failures by owner: retrieval={owners.get('retrieval', 0)}, "
      f"generation={owners.get('generation', 0)}\n")

out = Path("reports/failure_analysis.md")
out.write_text(render_markdown(summary, "Failure analysis: extractive baseline"),
               encoding="utf-8")
print(f"  wrote {out}")
