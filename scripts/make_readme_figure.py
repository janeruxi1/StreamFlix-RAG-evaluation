"""Draw the README's summary figure from the recorded results (no API key, no data download).

    python scripts/make_readme_figure.py   ->  reports/figures/00_readme_summary.png
"""
import json
from pathlib import Path

# --- shared README figure style (same block in every repo) ---------------------------------
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#a3a29d"
plt.rcParams.update({
    "font.size": 11, "axes.titlesize": 12.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.titlepad": 12, "axes.edgecolor": GRID, "axes.labelcolor": MUTED, "xtick.color": MUTED,
    "ytick.color": MUTED, "text.color": INK, "axes.spines.top": False, "axes.spines.right": False,
    "axes.spines.left": False, "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
    "axes.axisbelow": True, "figure.facecolor": "white", "savefig.facecolor": "white",
    "xtick.major.size": 0, "ytick.major.size": 0,
})


def save(fig, path):
    fig.tight_layout(w_pad=3)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print("wrote", path.name)
# ---------------------------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parents[1]
M = ROOT / "reports" / "metrics"


def rec(name):
    return json.loads((M / f"{name}.json").read_text(encoding="utf-8"))


ref, full = rec("05_judged_arms")["refusals"], rec("08_full_corpus")
arms = [("llm_cited", "cited prompt"), ("llm_naive", "naive prompt")]
fig, axes = plt.subplots(1, 3, figsize=(16, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.1]})


def stacked(ax, parts, total, title):
    for y, (key, label) in enumerate(arms):
        left = 0
        for name, fn, colour in parts:
            v = fn(ref[key])
            ax.barh(y, v, left=left, color=colour, height=0.55, edgecolor="white", linewidth=2,
                    label=name if y == 0 else None)
            if v >= 5:
                ax.text(left + v / 2, y, str(v), ha="center", va="center", color="white", fontsize=11,
                        fontweight="bold")
            left += v
    ax.set_yticks(range(len(arms)), [lab for _, lab in arms])
    ax.invert_yaxis()
    ax.set_xlim(0, total)
    ax.set_xticks([])
    ax.grid(False)
    ax.legend(frameon=False, fontsize=10, loc="upper center", bbox_to_anchor=(0.5, -0.04), ncol=3)
    ax.set_title(title)


stacked(axes[0], [("declined (right)", lambda a: a["out_of_scope_refused"], BLUE),
                  ("answered anyway", lambda a: a["out_of_scope_n"] - a["out_of_scope_refused"], ORANGE)],
        ref["llm_cited"]["out_of_scope_n"],
        f"1 · {ref['llm_cited']['out_of_scope_n']} questions the help centre can't answer:\n"
        "the cited prompt declines them all")
stacked(axes[1], [("answered, supported", lambda a: a["in_scope_answered"] - a["in_scope_answers_unfaithful"], BLUE),
                  ("unsupported claim", lambda a: a["in_scope_answers_unfaithful"], ORANGE),
                  ("declined", lambda a: a["in_scope_refused"], GREY)],
        ref["llm_cited"]["in_scope_answered"] + ref["llm_cited"]["in_scope_refused"],
        f"2 · {ref['llm_cited']['in_scope_answered'] + ref['llm_cited']['in_scope_refused']} answerable questions: "
        "the price\nof caution is declining some")

ax = axes[2]
d = full["full_minus_rag"]
rows = [("Questions answered", d["answered_in_scope"]), ("Correctness", d["correctness_in_scope"])]
for i, (label, e) in enumerate(rows):
    ax.plot([e["ci_low"], e["ci_high"]], [i, i], color=GREY, lw=2.5, solid_capstyle="round")
    ax.plot(e["difference"], i, "o", ms=9, color=GREY, mec="white", mew=2)
    ax.text(e["ci_high"] + 0.01, i, f"{e['difference']:+.3f}", va="center", fontsize=10.5)
ax.axvline(0, color=MUTED, lw=1)
ax.set_yticks(range(len(rows)), [r[0] for r in rows])
ax.set_ylim(-0.6, 1.6)
ax.set_xlim(-0.15, 0.2)
ax.set_xlabel("whole help centre in the prompt minus retrieval (share, 95% interval)")
ax.grid(axis="y", visible=False)
ratio = full["full_corpus"]["mean_context_tokens_estimated"] / full["rag"]["mean_context_tokens_estimated"]
ax.set_title(f"3 · Retrieval vs sending every article: a tie,\nat {ratio:.1f}x less context")

save(fig, ROOT / "reports" / "figures" / "00_readme_summary.png")


# ---- hero: safety against helpfulness, every prompt -------------------------------------------
gen = rec("04_generation_arms")["arms"]
names = {"llm_naive": "naive", "llm_grounded": "grounded", "llm_cited": "cited (recommended)",
         "llm_strict": "strict", "llm_grounded_refusal": "grounded + refusal rule",
         "extractive_baseline": "extractive, no LLM"}
offs = {"llm_naive": (10, 0, "left"), "llm_grounded": (-10, 0, "right"), "llm_cited": (-10, 10, "right"),
        "llm_strict": (-10, -12, "right"), "llm_grounded_refusal": (-10, 6, "right"),
        "extractive_baseline": (10, 0, "left")}
fig, ax = plt.subplots(figsize=(11, 4.6))
ax.axvspan(95, 105, color="#eaf2fc", lw=0)
ax.text(95.6, 57, "declines every\nunanswerable\nquestion", color=MUTED, fontsize=10, va="bottom")
for k, lab in names.items():
    a = gen[k]
    x, y = 100 * a["refusal_rate_out_of_scope"], 100 * a["answer_rate_in_scope"]
    c = BLUE if k == "llm_cited" else (ORANGE if k == "llm_naive" else GREY)
    ax.plot(x, y, "o", ms=11 if k == "llm_cited" else 9, color=c, mec="white", mew=2, zorder=3)
    dx, dy, ha = offs[k]
    ax.annotate(f"{lab}  ({y:.0f}% answered)", (x, y), (dx, dy), textcoords="offset points", ha=ha, va="center",
                fontsize=10.5, fontweight="bold" if k == "llm_cited" else "normal")
ax.set_xlim(30, 105)
ax.set_ylim(55, 104)
ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:.0f}%"))
ax.set_xlabel(f"share of the {ref['llm_cited']['out_of_scope_n']} unanswerable questions declined (safety)")
ax.set_ylabel("share of answerable\nquestions answered")
ax.set_title("No prompt is both perfectly safe and perfectly helpful: `cited` declines every question the help\n"
             "centre can't answer and still answers 80%; `naive` answers everything, including 60% it shouldn't")
save(fig, ROOT / "reports" / "figures" / "00_readme_hero.png")
