"""Draw the README's summary figure from the recorded results (no API key, no data download).

    python scripts/make_readme_figure.py      # writes reports/figures/00_readme_summary.png
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "figures" / "00_readme_summary.png"


def main() -> Path:
    r = json.loads((ROOT / "reports" / "metrics" / "05_judged_arms.json").read_text(encoding="utf-8"))["refusals"]
    arms = [("llm_cited", "cited prompt"), ("llm_naive", "naive prompt")]
    good, bad, grey = "#2E7D5B", "#D1495B", "#BFBFBF"
    fig, axes = plt.subplots(1, 2, figsize=(11, 2.9))
    panels = [
        ("25 questions the help centre cannot answer", "out_of_scope_n",
         [("declined", lambda a: a["out_of_scope_refused"], good),
          ("answered anyway", lambda a: a["out_of_scope_n"] - a["out_of_scope_refused"], bad)]),
        ("95 questions it can answer", None,
         [("answered", lambda a: a["in_scope_answered"], good),
          ("declined", lambda a: a["in_scope_refused"], grey)]),
    ]
    for ax, (title, _, parts) in zip(axes, panels):
        for y, (key, label) in enumerate(arms):
            left = 0
            for name, value, colour in parts:
                v = value(r[key])
                ax.barh(y, v, left=left, color=colour, edgecolor="white", height=0.6,
                        label=name if y == 0 else None)
                if v:
                    ax.text(left + v / 2, y, str(v), ha="center", va="center", color="white", fontsize=10,
                            fontweight="bold")
                left += v
        ax.set_yticks(range(len(arms)), [label for _, label in arms])
        ax.invert_yaxis()
        ax.set_xticks([])
        ax.set_title(title, fontsize=11, loc="left")
        ax.legend(frameon=False, fontsize=8, loc="lower right", bbox_to_anchor=(1, -0.32), ncol=2)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
    fig.suptitle("What the assistant did with 120 test questions written before it was built",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=150, bbox_inches="tight")
    return OUT


if __name__ == "__main__":
    print("wrote", main().relative_to(ROOT))
