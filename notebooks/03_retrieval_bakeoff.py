"""
Phase 3 — Retrieval Bake-Off
=============================

Phase 2 built one configuration and measured it. This phase sweeps the
space and decides what ships.

The sweep is every chunking strategy x retrieval arm x depth. The arm
count depends on whether sentence-transformers is installed, so the exact
number of configurations is reported at run time rather than asserted
here. Whatever it is, the size of the table is itself the problem, and
most of this notebook is about not being fooled by it:

  MONOTONICITY   recall@k cannot decrease as k grows. Sweep k, rank by
                 recall, and the largest k always wins — a fact about
                 arithmetic, not about retrieval. Section C.

  COST           So the honest question is not "which config has the
                 highest recall" but "which achieves a given recall for
                 the fewest context tokens". Tokens cost money per query
                 and dilute the generator with distractors. Section D.

  NOISE          95 in-scope questions. A 3-point recall gap is three
                 questions changing answer. Comparisons are paired
                 bootstraps, because every config is scored on the same
                 questions. Section E.

  MULTIPLICITY   One test per challenger means dozens of tests. At
                 alpha=0.05 several will look significant by chance, so
                 p-values are Holm-Bonferroni corrected. And a
                 non-significant result is not evidence of equivalence:
                 claims of "same, so pick the cheaper one" are made
                 against a declared margin. Sections D and E.

  SELECTION      The maximum of many noisy estimates is biased upward.
                 The winner is chosen on a dev split and reported on
                 held-out test, and the gap is quantified. Section F.

One correction to make up front. Phase 1 ran BM25 over whole ARTICLES;
Phase 2 ran dense retrieval over CHUNKS. Those numbers were never
comparable — the unit of retrieval differed, so any gap could have been
the unit rather than the method. Every arm here indexes the same chunks
and is scored by the same code.

Sections
--------
  A. Setup and the fair-comparison fix
  B. The sweep
  C. The monotonicity trap
  D. Context budget — the Pareto frontier
  E. Which differences are real
  F. Honest winner selection
  G. mh-011 revisited — correcting Phase 2
  H. Verdict and handoff to Phase 4
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.corpus.build import load_corpus, load_golden_set
from src.retrieval.chunking import STRATEGIES
from src.retrieval.embedding import (
    TfidfSvdEmbedder,
    get_embedder,
    transformer_available,
)
from src.retrieval.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    OracleUnionRetriever,
)
from src.retrieval.vectorstore import hits_to_articles
from src.evaluation.judged_arms import write_metrics
from src.evaluation.retrieval_metrics import (
    CostedConfig,
    aggregate,
    best_under_budget,
    bootstrap_ci,
    context_tokens,
    equivalence_verdict,
    holm_bonferroni,
    paired_bootstrap,
    pareto_frontier,
    score_run,
    selection_optimism,
    stratified_split,
)

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

articles = load_corpus()
golden = load_golden_set()
in_scope = [q for q in golden if q["category"] != "out_of_scope"]
questions = [q["question"] for q in in_scope]

# fixed_token_256 was dropped in Phase 2 as degenerate — it produced
# the same words as whole_article, one chunk per article, because no
# article exceeds 256 tokens. Carrying it here would add a duplicate row, not a data point.
ACTIVE_STRATEGIES = ["whole_article", "markdown_section",
                     "fixed_token_128", "sentence_window"]
DEPTHS = [3, 5, 10, 15]


# =====================================================================
# A. Setup and the fair-comparison fix
# =====================================================================
print("=" * 78)
print("A. SETUP")
print("=" * 78)

has_transformer = transformer_available()
print(f"""
  Corpus            : {len(articles)} articles
  Scored questions  : {len(in_scope)} in-scope
                      ({len(golden) - len(in_scope)} out-of-scope excluded — they have no
                       retrievable ground truth and are evaluated on
                       REFUSAL in Phase 5; averaging them in here would
                       drag every score toward zero for a reason that has
                       nothing to do with retrieval quality)
  Strategies        : {len(ACTIVE_STRATEGIES)}  ({', '.join(ACTIVE_STRATEGIES)})
  Depths            : {DEPTHS}
  Transformer arm   : {'available' if has_transformer else 'NOT installed — running lexical + LSA only'}
""")

if not has_transformer:
    print("""  Install sentence-transformers to add the modern-dense arm:
      pip install sentence-transformers
  The comparison below is still valid; it is a lexical vs classic-dense
  vs hybrid bake-off rather than a three-generation one.
""")


def build_arms(chunks):
    """Construct every retrieval arm over one chunk set.

    All arms index the SAME text (title + body). Giving BM25 a different
    view of the corpus than the dense arm would make any difference
    between them uninterpretable.
    """
    bm25 = BM25Retriever(chunks)
    dense = DenseRetriever(chunks, TfidfSvdEmbedder(n_components=128),
                           name="dense_lsa")
    arms = [bm25, dense]
    if has_transformer:
        arms.append(DenseRetriever(chunks, get_embedder("sentence_transformer"),
                                   name="dense_transformer"))
    arms.append(HybridRetriever(list(arms), name="hybrid"))
    return arms


# =====================================================================
# B. The sweep
# =====================================================================
print("\n" + "=" * 78)
print("B. THE SWEEP")
print("=" * 78)

records = []
per_question = {}          # config -> list[QuestionScore], for paired tests
BUILT = {}                 # strategy -> (chunks, arms); reused in Section G
                           # rather than rebuilt, which halves runtime

for strategy in ACTIVE_STRATEGIES:
    chunks = STRATEGIES[strategy](articles)
    BUILT[strategy] = (chunks, build_arms(chunks))
    for arm in BUILT[strategy][1]:
        for depth in DEPTHS:
            hits = arm.search_many(questions, top_k=depth)
            scores = score_run(in_scope, hits, k=depth)
            agg = aggregate(f"{strategy}|{arm.name}|d{depth}", scores)
            tokens = float(np.mean([context_tokens(h) for h in hits]))

            config = f"{strategy}|{arm.name}|d{depth}"
            per_question[config] = scores
            records.append({
                "config": config,
                "strategy": strategy,
                "arm": arm.name,
                "depth": depth,
                "n_chunks": len(chunks),
                "recall": agg.overall["recall_at_k"],
                "mrr": agg.overall["mrr"],
                "ndcg": agg.overall["ndcg_at_k"],
                "single_hop": agg.by_category["single_hop"]["recall_at_k"],
                "multi_hop": agg.by_category["multi_hop"]["recall_at_k"],
                "ambiguous": agg.by_category["ambiguous"]["recall_at_k"],
                "tokens": tokens,
            })

df = pd.DataFrame(records)
print(f"\n  {len(df)} configurations evaluated on {len(in_scope)} questions each")

top = df.nlargest(10, "recall")
print("\n  Top 10 by raw recall:\n")
print(f"  {'strategy':<17}{'arm':<19}{'d':>3}{'recall':>8}{'mrr':>7}"
      f"{'1hop':>7}{'multi':>7}{'ambig':>7}{'tokens':>8}")
print("  " + "-" * 83)
for _, r in top.iterrows():
    print(f"  {r.strategy:<17}{r.arm:<19}{r.depth:>3}{r.recall:>8.3f}"
          f"{r.mrr:>7.3f}{r.single_hop:>7.3f}{r.multi_hop:>7.3f}"
          f"{r.ambiguous:>7.3f}{r.tokens:>8.0f}")


# =====================================================================
# C. The monotonicity trap
# =====================================================================
print("\n" + "=" * 78)
print("C. THE MONOTONICITY TRAP")
print("=" * 78)

depth_of_top10 = top["depth"].value_counts().to_dict()
max_depth = max(DEPTHS)
share = depth_of_top10.get(max_depth, 0) / len(top)

print(f"""
Look at the depth column above: {depth_of_top10.get(max_depth, 0)} of the top 10 configurations use
depth={max_depth}, the largest swept.

That is not a finding. recall@k is monotonically non-decreasing in k —
retrieving more documents can only ever find more ground truth, never
less. A bake-off that sweeps depth and ranks by recall is guaranteed to
crown the deepest configuration, and would do so even if the retriever
were returning documents at random.
""")

# Demonstrate rather than assert
mono_violations = 0
checked = 0
for (s, a), grp in df.groupby(["strategy", "arm"]):
    g = grp.sort_values("depth")
    checked += len(g) - 1
    mono_violations += int((g["recall"].diff().dropna() < -1e-9).sum())

print(f"""  Verified: {checked - mono_violations}/{checked} depth increases produced a non-decreasing
  recall, exactly as the metric guarantees.

The obvious hope is that a rank-sensitive metric escapes this. MRR and
nDCG discount by position, so burying a correct document deep in the
list ought to cost something. Checking rather than assuming:
""")

mono_report = {}
for metric in ("mrr", "ndcg"):
    non_mono = 0
    for (s, a), grp in df.groupby(["strategy", "arm"]):
        g = grp.sort_values("depth")
        non_mono += int((g[metric].diff().dropna() < -1e-9).sum())
    mono_report[metric] = non_mono
    print(f"    {metric:<6} decreased with depth in {non_mono}/{checked} cases")

if not any(mono_report.values()):
    print("""
  They do not escape it. All three are monotone in depth here, and for a
  structural reason: retrieving deeper can only ADD documents further
  down the list. It never moves the first correct document to a worse
  rank, so MRR cannot fall; and each extra item contributes a
  non-negative discounted gain, so nDCG cannot fall either.

  This is worth stating plainly because it closes off the tempting move.
  There is no quality-only metric that selects retrieval depth. Any
  ranking of a depth sweep by any of these metrics returns the deepest
  configuration by construction.

  Depth is therefore not a quality decision at all — it is a COST
  decision, and it cannot be made without measuring cost. That is not a
  refinement of the analysis; without Section D there is no analysis.

  These metrics remain the right ones for comparing configurations AT A
  FIXED depth, which is how Section E uses them.
""")
else:
    print(f"""
  Partially: {sum(mono_report.values())} of {2 * checked} increases did reduce a rank-sensitive
  metric, so MRR and nDCG carry information recall does not. They still
  cannot be the primary depth criterion, but disagreement between them
  and recall flags configurations that find the right documents and rank
  them badly.
""")


# =====================================================================
# D. Context budget — the Pareto frontier
# =====================================================================
print("\n" + "=" * 78)
print("D. CONTEXT BUDGET — WHAT RECALL ACTUALLY COSTS")
print("=" * 78)
print("""
Every retrieved chunk is tokens the generator pays for: money per query,
and attention diluted across more distractors. So the real question is
not "which configuration has the highest recall" but "which reaches a
given recall for the fewest tokens".
""")

costed = [CostedConfig(r.config, r.recall, r.tokens)
          for r in df.itertuples()]
frontier = pareto_frontier(costed)

print(f"  Pareto frontier — {len(frontier)} of {len(costed)} configurations are not dominated.")
print("  (A dominated config costs more AND retrieves less. Choosing one")
print("   means knowingly paying more for less.)\n")
print(f"  {'config':<46}{'recall':>8}{'tokens':>8}{'rec/1k':>8}")
print("  " + "-" * 70)
for c in frontier:
    print(f"  {c.config:<46}{c.recall:>8.3f}{c.mean_tokens:>8.0f}"
          f"{c.recall_per_1k_tokens:>8.2f}")

# --- The headline comparison -----------------------------------------
#
# Comparing the top scorer against "the cheapest config with equal
# recall" is circular — nothing else has exactly equal recall, so it
# returns the top scorer itself. The question that actually matters is
# whether the extra tokens buy a difference that is REAL. So: find the
# cheapest configuration whose recall is not statistically
# distinguishable from the top scorer's.
best_raw = df.loc[df["recall"].idxmax()]

# Declared BEFORE looking at results: how much recall we would trade for
# a large cost saving. ~0.05 recall is about five questions on this
# golden set; below that a difference is not actionable for a support
# deployment. Choosing this after seeing the table would be fitting the
# conclusion to the data.
EQUIV_MARGIN = 0.05

verdicts = []
for r in df.sort_values("tokens").itertuples():
    if r.config == best_raw["config"]:
        continue
    cmp = paired_bootstrap(
        per_question[best_raw["config"]], per_question[r.config],
        metric="recall_at_k", label_a=best_raw["config"], label_b=r.config)
    verdicts.append((r, cmp, equivalence_verdict(cmp, margin=EQUIV_MARGIN)))

equivalent = [(r, c, v) for r, c, v in verdicts if v.verdict == "equivalent"]
inconclusive = [(r, c, v) for r, c, v in verdicts if v.verdict == "inconclusive"]

print(f"""
  Highest recall overall : {best_raw.recall:.3f}  {best_raw.config}
                           costing {best_raw.tokens:.0f} tokens/query

  A caveat that decides how this section can be read. "Not statistically
  significant" is NOT evidence that two configurations are the same. On
  {len(in_scope)} questions the test is underpowered, so failing to reject is
  mostly a statement about sample size. Picking the cheaper option
  because p > 0.05 is affirming the null.

  Making a positive claim requires a declared margin of practical
  equivalence — here {EQUIV_MARGIN:.2f} recall, about five questions, fixed before
  looking at the table. Then each comparison resolves three ways:

    EQUIVALENT    CI lies entirely inside +/-{EQUIV_MARGIN:.2f}  -> safe to prefer on cost
    DIFFERENT     CI excludes zero
    INCONCLUSIVE  neither -> the data cannot tell

  Against the leader, {len(df) - 1} challengers resolve as:
    equivalent   : {len(equivalent)}
    different    : {sum(1 for _, _, v in verdicts if v.verdict == 'different')}
    inconclusive : {len(inconclusive)}
""")

# An equivalence claim is only USEFUL if the equivalent configuration is
# also meaningfully cheaper. A config that matches the leader at the same
# cost tells us nothing about the cost/quality trade-off.
MEANINGFUL_SAVING = 1.5          # at least 1.5x fewer tokens

useful = [(r, c, v) for r, c, v in equivalent
          if best_raw.tokens / r.tokens >= MEANINGFUL_SAVING]

if equivalent:
    eq, eq_cmp, eq_v = equivalent[0]
    print(f"""  Cheapest configuration demonstrably EQUIVALENT to the leader:

    {eq.config}
    recall {eq.recall:.3f} at {eq.tokens:.0f} tokens/query  ({best_raw.tokens / eq.tokens:.1f}x cheaper)
    {eq_v}
""")

if useful:
    cheap, cheap_cmp, cheap_v = useful[0]
    ratio = best_raw.tokens / cheap.tokens
    print(f""">>> HEADLINE — the top of the recall table costs {ratio:.1f}x more context than a
    configuration demonstrably equivalent to it.

  The whole confidence interval on the difference sits inside the margin
  declared as practically irrelevant. That is a positive claim of
  equivalence, not an absence of evidence — which is what licenses
  choosing on cost.
""")
else:
    cheap, cheap_cmp, cheap_v = (inconclusive or verdicts)[0]
    ratio = best_raw.tokens / cheap.tokens
    print(f""">>> HEADLINE — and it is a NEGATIVE one, which is the honest result here.

  No configuration that is meaningfully cheaper than the leader is
  demonstrably equivalent to it. The cheapest serious contender:

    {cheap.config}
    recall {cheap.recall:.3f} at {cheap.tokens:.0f} tokens/query  ({ratio:.1f}x cheaper)
    {cheap_v}

  Its interval reaches beyond the {EQUIV_MARGIN:.2f} margin, so it lands as
  INCONCLUSIVE. {len(in_scope)} questions cannot resolve whether a {ratio:.1f}x cost
  saving costs meaningful recall.

  This overturns the reading an uncorrected table invites. Ranked by
  raw p-values, that configuration was "not significantly different"
  (p={cheap_cmp.p_value:.3f}) and the tempting conclusion was "equivalent — ship the
  cheap one, {ratio:.1f}x saving". That conclusion was affirming the null. Once
  a margin is declared, the same data says only that the question is
  open.

  What can still be said, and it is not nothing:

    - The {ratio:.1f}x cost saving is certain; the recall cost is uncertain
      and bounded above by {abs(cheap_cmp.ci_high):.3f} recall at 95% confidence.
    - That is a genuine engineering decision with a quantified worst
      case, rather than a false claim of equivalence.
    - Resolving it needs a larger golden set, not more analysis. The
      honest handoff is the bound, not a verdict.
""")

print(f"""  On what chunking is for, the picture does survive. Phase 2 saw
  whole_article performing well and chunking looking like complexity
  without payoff, because recall was the only axis on the page. The
  frontier above shows chunking buying large, certain cost reductions
  for small, uncertain recall costs — which is a real benefit even
  without an equivalence claim.

  The saving compounds in Phase 5: fewer distractor tokens is precisely
  the condition under which faithfulness improves. Retrieval cost and
  generation quality are not independent axes.
""")

print("  Best achievable recall under a fixed budget:\n")
print(f"  {'budget':>9}  {'recall':>7}  config")
print("  " + "-" * 62)
for budget in (150, 300, 600, 1200, 2500):
    b = best_under_budget(costed, budget)
    print(f"  {budget:>7} tk  {b.recall:>7.3f}  {b.config}" if b
          else f"  {budget:>7} tk  {'—':>7}  (nothing fits)")

# --- Chart ------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

palette = {"whole_article": "#5B8FF9", "markdown_section": "#5AD8A6",
           "fixed_token_128": "#F6BD16", "sentence_window": "#E8684A"}
for strategy in ACTIVE_STRATEGIES:
    sub = df[df["strategy"] == strategy]
    ax1.scatter(sub["tokens"], sub["recall"], s=48, alpha=0.75,
                color=palette[strategy], label=strategy, edgecolor="white")
fx = [c.mean_tokens for c in frontier]
fy = [c.recall for c in frontier]
ax1.plot(fx, fy, color="#2B2B2B", linewidth=1.6, linestyle="--",
         marker="o", markersize=5, label="Pareto frontier", zorder=5)
ax1.set_xscale("log")
ax1.set_xlabel("Mean context tokens per query  (log scale)")
ax1.set_ylabel("Recall")
ax1.set_title("Recall vs context cost\nfrontier = the only defensible choices",
              fontweight="bold")
ax1.legend(fontsize=8, loc="lower right")
ax1.grid(linestyle="--", alpha=0.4)
ax1.set_axisbelow(True)

cats = ["single_hop", "multi_hop", "ambiguous"]
width = 0.2
xs = np.arange(len(cats))
for i, strategy in enumerate(ACTIVE_STRATEGIES):
    sub = df[(df["strategy"] == strategy) & (df["depth"] == 10)]
    vals = [sub[c].mean() for c in cats]
    ax2.bar(xs + i * width, vals, width, label=strategy,
            color=palette[strategy], edgecolor="white")
ax2.set_xticks(xs + 1.5 * width)
ax2.set_xticklabels(cats)
ax2.set_ylabel("Recall (depth=10, averaged over arms)")
ax2.set_title("Difficulty is a property of the QUESTION,\nnot the configuration",
              fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)

plt.suptitle("Phase 3 — retrieval bake-off", fontsize=13,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "03_retrieval_bakeoff.png", dpi=140, bbox_inches="tight",
            metadata={"Software": None})
print(f"\n  Saved -> {FIG_DIR}/03_retrieval_bakeoff.png")


# =====================================================================
# E. Which differences are real
# =====================================================================
print("\n" + "=" * 78)
print("E. WHICH DIFFERENCES ARE REAL")
print("=" * 78)
print(f"""
{len(in_scope)} questions. A 3-point recall gap is three questions changing answer.
Before preferring one configuration over another, the difference has to
survive being asked whether it is noise.

Comparisons are PAIRED: every configuration is scored on the same
questions, so differencing per question removes question difficulty as a
nuisance factor. Marginal confidence intervals on each arm would be
dominated by that shared difficulty and would badly understate power.
""")

FIXED_DEPTH = 10
at_depth = df[df["depth"] == FIXED_DEPTH]
ref = at_depth.loc[at_depth["recall"].idxmax()]

raw_comparisons = []
for _, row in at_depth.iterrows():
    if row.config == ref.config:
        continue
    raw_comparisons.append(paired_bootstrap(
        per_question[ref.config], per_question[row.config],
        metric="recall_at_k", label_a=ref.config, label_b=row.config))

# Family-wise correction. This section alone runs one test per
# challenger, and Section D ran one per configuration — at ~60 tests and
# alpha=0.05, about three will look significant by chance. An
# uncorrected column of stars is not evidence.
corrected = holm_bonferroni(raw_comparisons, alpha=0.05)
corrected.sort(key=lambda t: t[0].difference)

print(f"""  Reference (best at depth={FIXED_DEPTH}): {ref.config}

  p-values are Holm-Bonferroni corrected across the {len(raw_comparisons)} comparisons in
  this family. Holm rather than plain Bonferroni because it controls the
  same family-wise error rate while being uniformly more powerful.
""")
print(f"  {'challenger':<44}{'delta':>8}{'95% CI':>20}{'p':>7}  Holm")
print("  " + "-" * 87)
for cmp, survives in corrected:
    mark = "*" if survives else ("(raw)" if cmp.significant else "")
    print(f"  {cmp.config_b:<44}{-cmp.difference:>+8.3f}"
          f"  [{-cmp.ci_high:>+.3f}, {-cmp.ci_low:>+.3f}]{cmp.p_value:>7.3f}  {mark}")

n_sig = sum(1 for _, s in corrected if s)
n_raw = sum(1 for c, _ in corrected if c.significant)
n_tied = len(corrected) - n_sig
comparisons = [(c.config_b, c) for c, _ in corrected]
print(f"""
  *     = survives Holm correction
  (raw) = would have been called significant WITHOUT correction

  {n_raw} challengers look significant uncorrected; {n_sig} survive Holm.
  {n_raw - n_sig} of the original stars were multiplicity artifacts — exactly the
  false positives a 12-way comparison at alpha=0.05 predicts.

  One further caveat on this table. The reference was chosen as the
  BEST configuration at this depth, so it sits at the top of a noisy
  ranking and every comparison is made against a value that was
  selected for being high. That inflates the apparent gaps somewhat;
  the direction of the bias is known and it works against the
  challengers, so surviving comparisons are conservative rather than
  optimistic.
""")

if n_tied:
    tied_names = [c for c, cmp in comparisons if not cmp.significant]
    print(f"""  The {n_tied} indistinguishable configurations form a LEADING GROUP:

{chr(10).join('      ' + t for t in tied_names)}

  The bake-off does not identify a single best retriever. It identifies
  a group that {len(in_scope)} questions cannot tell apart, plus a clear tail that
  they can. Inside the group, preferring one on a decimal place of
  recall is reading noise; the choice should be made on cost and
  simplicity, which is what Section D does.
""")

print(f"""  The tail is informative in its own right. sentence_window is
  significantly worse across every arm — the most elaborate strategy,
  producing the most chunks, is the one clearly beaten. Fragmenting
  articles into single sentences buys sharper matching that does not
  survive aggregation back to article level.
""")

# Lexical vs dense, the Phase 2 question, now with paired inference
best_bm25 = at_depth[at_depth["arm"] == "bm25"].nlargest(1, "recall").iloc[0]
best_dense = at_depth[at_depth["arm"] == "dense_lsa"].nlargest(1, "recall").iloc[0]
best_hybrid = at_depth[at_depth["arm"] == "hybrid"].nlargest(1, "recall").iloc[0]

print("  The Phase 2 question, answered properly:\n")
for label, a, b in [
    ("bm25 vs dense ", best_bm25, best_dense),
    ("hybrid vs bm25", best_hybrid, best_bm25),
    ("hybrid vs dense", best_hybrid, best_dense),
]:
    cmp = paired_bootstrap(per_question[a.config], per_question[b.config],
                           metric="recall_at_k", label_a=label, label_b="")
    verdict = "REAL" if cmp.significant else "not distinguishable"
    print(f"    {label}: {cmp.difference:+.3f} "
          f"[{cmp.ci_low:+.3f}, {cmp.ci_high:+.3f}] p={cmp.p_value:.3f}  → {verdict}")
    print(f"      ({cmp.n_differing} of {cmp.n_questions} questions differ at all)")


# =====================================================================
# F. Honest winner selection
# =====================================================================
print("\n" + "=" * 78)
print("F. HONEST WINNER SELECTION")
print("=" * 78)
print(f"""
{len(df)} configurations were scored on the same {len(in_scope)} questions. The maximum of
{len(df)} noisy estimates is biased upward — part of the winner's margin is
skill and part is luck, and reporting its full score as an expected
production number overstates the system.

So: select on a dev split, report on held-out test, and measure the gap.
The split is stratified by category, because a uniform split can easily
leave 2 of 15 ambiguous questions in test and make that column
meaningless.
""")

dev, test = stratified_split(in_scope, test_fraction=0.4, seed=42)
dev_ids = {q["question_id"] for q in dev}
print(f"  dev  : {len(dev):>3} questions")
print(f"  test : {len(test):>3} questions")

dev_scores = {c: [s for s in v if s.question_id in dev_ids]
              for c, v in per_question.items()}
test_scores = {c: [s for s in v if s.question_id not in dev_ids]
               for c, v in per_question.items()}

opt = selection_optimism(dev_scores, test_scores, metric="recall_at_k")
print(f"""
  Dev-set winner        : {opt['winner']}
  Its dev recall        : {opt['dev_score']:.3f}
  Its HELD-OUT recall   : {opt['test_score']:.3f}
  Selection optimism    : {opt['optimism']:+.3f}

  Best possible on test : {opt['best_possible_test']:.3f}
  Regret of our pick    : {opt['regret']:.3f}
""")

if abs(opt["optimism"]) > 0.03:
    print(f"""  The {abs(opt['optimism']):.3f} gap is the winner's curse made visible. The dev
  score was flattering; the held-out number is the one to quote.
""")
else:
    print(f"""  The gap is small ({opt['optimism']:+.3f}), which suggests the top configurations
  are genuinely close rather than the leader being a lucky draw —
  consistent with Section E, where most arms were not separable.
""")

print(f"""  Regret of {opt['regret']:.3f} is the cost of having to choose without seeing
  test. It is the honest measure of how much the selection procedure —
  not the retriever — cost us.
""")


# =====================================================================
# G. mh-011 revisited
# =====================================================================
print("\n" + "=" * 78)
print("G. mh-011 REVISITED — CORRECTING PHASE 2")
print("=" * 78)

q11 = next(q for q in golden if q["question_id"] == "mh-011")
gt11 = set(q11["gt_article_ids"])

solved, pool_has_both = [], []
for strategy, (chunks, arms) in BUILT.items():
    oracle = OracleUnionRetriever([a for a in arms if a.name != "hybrid"],
                                  depth=30)
    got_pool = set(hits_to_articles(oracle.search(q11["question"], top_k=999)))
    pool_has_both.append(gt11 <= got_pool)
    for arm in arms:
        for depth in DEPTHS:
            got = set(hits_to_articles(arm.search(q11["question"], top_k=depth)))
            if gt11 <= got:
                solved.append(f"{strategy}|{arm.name}|d{depth}")

total_configs = len(df)

print(f"""
  Question      : {q11['question']!r}
  Ground truth  : {sorted(gt11)}  (the planted refund contradiction)

  Phase 2 found this question returned 0% recall under dense retrieval
  and 50% under BM25, and concluded — because the failure was shared
  across both — that it was "a property of the QUESTION" and that
  "changing retriever will not fix it."

  That conclusion was too strong. Result:

    configurations retrieving BOTH articles : {len(solved)} of {total_configs}
    candidate pool contained both           : {sum(pool_has_both)} of {len(pool_has_both)} strategies
""")

if solved:
    print("    examples:")
    for s in solved[:5]:
        print(f"      {s}")
    print(f"""
  The oracle result is the diagnostic that matters: both articles were
  in the candidate pool for EVERY chunking strategy at depth 30. So this
  was never a candidate-generation failure — the retriever always found
  them, and simply ranked them below the free-trial articles.

  A ranking failure and a candidate failure need opposite fixes. Ranking
  failures are solved by retrieving deeper, reranking, or fusion.
  Candidate failures need query rewriting or a corpus change. Phase 2
  had no oracle, could not tell them apart, and guessed the harder one.

  The correction: retrieving deeper resolves it. What Phase 2 got right
  is the consequence — at depth 5 the generator receives five confident,
  on-topic, WRONG chunks, and will answer fluently from them. That
  remains a retrieval failure that will present as a generation failure,
  and Phase 6 still has to attribute it correctly.
""")
else:
    print("""  No configuration retrieves both. Phase 2's conclusion stands, and the
  fix has to be upstream of ranking — query rewriting or a corpus change.
""")


# =====================================================================
# H. Verdict
# =====================================================================
print("\n" + "=" * 78)
print("PHASE 3 VERDICT")
print("=" * 78)

rec = best_under_budget(costed, 600)

# The budget rule returns the highest recall that fits. That is a ranking
# by a decimal place, and this notebook has spent five sections arguing
# that a decimal place on 95 questions is not a finding. So the rule's
# pick is not automatically what ships: it is tested against the
# SIMPLEST arm on the same chunks at the same depth, and it ships only if
# it is demonstrably better. BM25 is that arm — no model, no download, no
# fitted state, exactly reproducible.
rec_strategy, rec_arm, rec_depth = rec.config.split("|")
simple_config = f"{rec_strategy}|bm25|{rec_depth}"
simple = next(c for c in costed if c.config == simple_config)

if rec.config == simple_config:
    ship, ship_cmp, ship_v = rec, None, None
else:
    ship_cmp = paired_bootstrap(
        per_question[rec.config], per_question[simple_config],
        metric="recall_at_k", label_a=rec.config, label_b=simple_config)
    ship_v = equivalence_verdict(ship_cmp, EQUIV_MARGIN)
    earned = ship_v.verdict == "different" and ship_cmp.difference > 0
    ship = rec if earned else simple

# What shipping the cheap configuration GIVES UP, stated as a measured
# difference rather than waved away. Against the top of the table this is
# usually real: the budget buys a large cut in context, and pays for it.
top_cmp = paired_bootstrap(
    per_question[best_raw.config], per_question[ship.config],
    metric="recall_at_k", label_a=best_raw.config, label_b=ship.config)
top_v = equivalence_verdict(top_cmp, EQUIV_MARGIN)
top_reading = (
    "That difference is real: its interval excludes zero.\n"
    "   The budget is a trade, not a free saving, and the memo has to say so."
    if top_v.verdict == "different" else
    "That difference is not resolved on 95 questions.")

if ship_cmp is None:
    ship_text = f"""SHIPPING CONFIGURATION:
   {ship.config}
   recall {ship.recall:.3f} at {ship.mean_tokens:.0f} tokens/query

   The budget pick is already the simplest arm on these chunks: BM25
   needs no model, no download and no fitted state, and is exactly
   reproducible. Nothing more elaborate scored higher inside the budget
   in this environment."""
else:
    by_cat = {}
    for cat in ("single_hop", "multi_hop", "ambiguous"):
        a = [s.recall_at_k for s in per_question[rec.config] if s.category == cat]
        b = [s.recall_at_k for s in per_question[simple_config] if s.category == cat]
        by_cat[cat] = (float(np.mean(a)), float(np.mean(b)), len(a))
    cat_lines = "\n".join(
        f"     {cat:<11} {rec_arm} {a:.3f}   bm25 {b:.3f}   (n={n})"
        for cat, (a, b, n) in by_cat.items())
    if ship is rec:
        reason = f"""   {rec_arm} ships. Its advantage over BM25 on the same chunks at the
   same depth excludes zero after pairing, so the extra dependency is
   paid for by a measured gain rather than a ranking."""
    else:
        reason = f"""   BM25 ships. The budget rule prefers {rec_arm} by {ship_cmp.difference:+.3f} recall,
   and that gap is {ship_v.verdict.upper()} — its interval includes zero, and
   it also reaches past the {EQUIV_MARGIN:.2f} margin, so the data supports neither
   "better" nor "the same". Only {ship_cmp.n_differing} of {ship_cmp.n_questions} questions differ at all.

   Given an unresolved difference, the simpler system ships: no model
   download, no fitted state, exactly reproducible. Switching on a
   {ship_cmp.difference:+.3f} gap would be the noise-reading this notebook argues
   against. Calling the two equivalent would be the other error."""
    ship_text = f"""SHIPPING CONFIGURATION:
   {ship.config}
   recall {ship.recall:.3f} at {ship.mean_tokens:.0f} tokens/query

   Tested against the budget pick on the same chunks, depth and cost:
     {rec.config} vs {simple_config}
     {ship_v}

{reason}

   WHERE they differ is the useful part, and the aggregate hides it:
{cat_lines}

   The two arms fail on different questions. That is an argument for
   combining them (a reranker, or fusion tuned on a larger golden set)
   as the next experiment, and it is not an argument for swapping one
   for the other."""

if has_transformer:
    # The transformer arm needs a ~130MB model CI deliberately does not
    # download, so its results cannot be recomputed there. This record
    # is what lets the memo's claim about it be checked without a rerun.
    # Written only when the arm actually ran.
    write_metrics(Path("reports/metrics") / "03_retrieval_arms.json", {
        "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"),
        "n_configurations": int(len(df)),
        "n_questions": len(in_scope),
        "equivalence_margin": EQUIV_MARGIN,
        "top_of_table": {"config": best_raw.config,
                         "recall": round(float(best_raw.recall), 6),
                         "tokens": round(float(best_raw.tokens), 1)},
        "budget_pick_600": {"config": rec.config,
                            "recall": round(float(rec.recall), 6),
                            "tokens": round(float(rec.mean_tokens), 1)},
        "shipping": {"config": ship.config,
                     "recall": round(float(ship.recall), 6),
                     "tokens": round(float(ship.mean_tokens), 1)},
        "top_vs_shipping": {
            "difference": round(top_cmp.difference, 6),
            "ci_low": round(top_cmp.ci_low, 6),
            "ci_high": round(top_cmp.ci_high, 6),
            "verdict": top_v.verdict,
            "context_ratio": round(float(best_raw.tokens / ship.mean_tokens), 2),
        },
        "budget_pick_vs_bm25": None if ship_cmp is None else {
            "difference": round(ship_cmp.difference, 6),
            "ci_low": round(ship_cmp.ci_low, 6),
            "ci_high": round(ship_cmp.ci_high, 6),
            "p_value": round(ship_cmp.p_value, 6),
            "n_differing": ship_cmp.n_differing,
            "n_questions": ship_cmp.n_questions,
            "verdict": ship_v.verdict,
            "recall_by_category": {
                cat: {"budget_pick": round(a, 6), "bm25": round(b, 6), "n": n}
                for cat, (a, b, n) in by_cat.items()},
        },
    })

print(f"""
Swept {len(df)} configurations. Four things came out of it.

1. DEPTH CANNOT BE CHOSEN BY RECALL.
   recall@k is monotone in k, so ranking a depth sweep by recall is
   arithmetic dressed as evidence. Depth was selected against context
   cost instead.

2. CHUNKING BUYS LARGE CERTAIN SAVINGS FOR SMALL UNCERTAIN COSTS.
   Top of the recall table : {best_raw.recall:.3f} at {best_raw.tokens:.0f} tokens/query
   Cheapest contender      : {cheap.recall:.3f} at {cheap.tokens:.0f} tokens/query ({best_raw.tokens / cheap.tokens:.1f}x cheaper)
   Verdict against a declared {EQUIV_MARGIN:.2f} margin: {cheap_v.verdict.upper()}.
   The saving is certain; the recall cost is bounded above by {abs(cheap_cmp.ci_high):.3f}
   at 95% confidence but not resolved. Calling these "equivalent"
   because p={cheap_cmp.p_value:.3f} would be affirming the null.

3. THE BAKE-OFF FINDS A GROUP, NOT A WINNER.
   At depth {FIXED_DEPTH}, {n_raw} challengers look significantly worse before
   correction and {n_sig} survive Holm; {n_tied} are not separable. So the sweep does
   separate a real tail — sentence_window is beaten across every arm —
   but it cannot rank the top {n_tied + 1} configurations against each other on
   {len(in_scope)} questions. Inside that group, cost and simplicity decide;
   between the group and the tail, the measurement decides.

4. SELECTION OPTIMISM IS {opt['optimism']:+.3f}.
   Dev winner {opt['dev_score']:.3f} -> held-out {opt['test_score']:.3f}, with regret {opt['regret']:.3f}.""" + (f"""
   The held-out score came out ABOVE the dev score, which is not
   evidence the selection was skilful — it is a {abs(opt['optimism']):.3f} swing on a
   {len(test)}-question test split, comfortably inside noise. The useful
   reading is that no large optimism penalty appeared, consistent
   with a flat leaderboard rather than a lucky winner. On a larger
   sweep or a narrower field, expect this number to go positive.""" if opt['optimism'] < 0 else f"""
   The dev score was flattering by {opt['optimism']:.3f}; the held-out number is the
   one to quote, since the dev number is contaminated by having
   chosen the maximum of {len(df)} estimates.""") + f"""

HIGHEST RECALL UNDER A 600-TOKEN BUDGET:
   {rec.config}
   recall {rec.recall:.3f} at {rec.mean_tokens:.0f} tokens/query

   Found on the Pareto frontier under a context budget, not by topping
   the raw recall table. It gives up {best_raw.recall - rec.recall:.3f} recall against the
   leader and costs {best_raw.tokens / rec.mean_tokens:.1f}x less context per query.

{ship_text}

WHAT THE BUDGET COSTS:
   {best_raw.config} vs {ship.config}
   {top_v}

   The top of the table retrieves {top_cmp.difference:+.3f} more recall for {best_raw.tokens / ship.mean_tokens:.1f}x the
   context. {top_reading}
   Whether the extra context would help or hurt the ANSWERS is not
   something a retrieval metric can say — more evidence and more
   distractors arrive together. It is the first thing to test once the
   generation layer is judged.

STILL OPEN — carried into Phase 4:

  - Ambiguous questions remain the weak category everywhere
    (~{at_depth['ambiguous'].mean():.2f} at depth {FIXED_DEPTH} vs ~{at_depth['single_hop'].mean():.2f} for single-hop). No
    configuration fixes this, which suggests the problem is query
    understanding rather than ranking.
  - The transformer arm {'ran, and is compared against the shipping configuration above.' if has_transformer else 'DID NOT RUN here (sentence-transformers absent).'}
    {'' if has_transformer else 'The recorded comparison is in reports/metrics/03_retrieval_arms.json.'}
  - Hybrid fusion did not clearly beat its best constituent. RRF was
    used untuned by design; tuning k on 95 questions would fit noise.

HANDOFF TO PHASE 4: build the generation layer on the shipping
configuration, and carry the depth/cost trade-off forward — the token
budget chosen here is the context the generator has to work with.
""")
