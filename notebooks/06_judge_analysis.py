"""
Phase 6 — Auditing the Judge Itself
====================================

Phase 5 asked whether the judge gets known cases right. Necessary, and
not sufficient. A judge can score 100% on unambiguous cases and still be
useless for comparing prompt variants, because it may respond to
properties that have nothing to do with answer quality:

    length         longer answers rated higher regardless of content
    context order  the same evidence, reordered, scoring differently
    instability    the same input scored differently on a second call

Each corrupts a different downstream conclusion, and the first one is
aimed straight at Phase 4. The prompt ladder's rungs produce
systematically different answer lengths — `strict` asks for evidence
assessment and inline citations, so it writes more than `naive`. If the
judge rewards length, `strict` wins that comparison for a reason
unrelated to being better, and the ladder's finding is an artefact.

Why paired probes rather than a correlation
-------------------------------------------
Correlating score against answer length across the golden set does NOT
measure length bias. Longer answers may genuinely be more complete, so
the correlation confounds bias with quality — and the confound runs in
the direction that makes bias look real when it is not.

So every probe here is a PAIR, identical in every respect except the one
manipulated variable: same claims, same context, same citations, only
the phrasing length differs. Any score gap is attributable, because
nothing else moved. That is a controlled experiment, which is why the
probes are hand-written rather than sampled.

For length and context order the correct behaviour is NO CHANGE, so the
measurement is deviation from zero rather than a coefficient.

Sections
--------
  A. What is running
  B. Length bias — the one aimed at Phase 4
  C. Context-order sensitivity
  D. Self-consistency, and the floor it sets
  E. Inter-judge agreement
  F. Does any of this invalidate Phase 4?
  G. Verdict and handoff to Phase 7
"""
import importlib.util
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np

from src.corpus.build import load_corpus, load_golden_set
from src.evaluation.judge import CoverageJudge, LexicalJudge, get_judge
from src.evaluation.judge_bias import (
    LENGTH_PROBES,
    MATERIAL_DELTA,
    ORDER_PROBES,
    cohens_kappa,
    judge_agreement,
    measure_length_bias,
    measure_order_sensitivity,
    measure_self_consistency,
)
from src.evaluation.judge_validation import validate_judge
from src.generation.extractive import ExtractiveAnswerer
from src.generation.pipeline import RAGPipeline
from src.generation.prompts import VARIANTS, format_context
from src.llm.provider import provider_ready
from src.retrieval.chunking import STRATEGIES
from src.retrieval.retrievers import BM25Retriever

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

STRATEGY = "markdown_section"
DEPTH = 15

articles = load_corpus()
golden = load_golden_set()
chunks = STRATEGIES[STRATEGY](articles)
retriever = BM25Retriever(chunks)


# =====================================================================
# A. What is running
# =====================================================================
print("=" * 78)
print("A. SETUP")
print("=" * 78)

has_key, blocker = provider_ready()
_sdk_found = importlib.util.find_spec("openai") is not None

if has_key:
    from src.llm.provider import get_provider
    provider = get_provider()
    judge = get_judge(provider, model=os.getenv("JUDGE_MODEL", "gpt-4o"))
else:
    judge = LexicalJudge()

validation = validate_judge(judge)

print(f"""
  Judge under audit : {judge!r}
  Phase 5 accuracy  : {validation.overall_accuracy:.0%}  (trustworthy = {validation.is_trustworthy})
  Probes            : {len(LENGTH_PROBES)} length, {len(ORDER_PROBES)} context-order
  Material delta    : {MATERIAL_DELTA:.2f}  (score gap treated as real)
""")

if not has_key:
    print(f"""  LLM judge unavailable.

  Reason: {blocker}

    kernel Python : {sys.executable}
    openai found  : {_sdk_found}

  The LEXICAL judge is audited instead. That is worth doing on its own
  terms — it has a real and measurable bias, shown in Section B — but
  the headline comparison this phase exists for is LLM judge vs lexical
  judge, and it needs a credential.
""")

print(f"""  Note what Phase 5 established and this phase does NOT re-litigate:
  {'the judge passes the accuracy gate' if validation.is_trustworthy else 'this judge FAILS the accuracy gate'}. Bias is a separate question. An
  accurate judge can still be biased, and a biased judge can still be
  accurate on unambiguous cases — which is precisely why unambiguous
  cases cannot be the whole audit.
""")


# =====================================================================
# B. Length bias
# =====================================================================
print("\n" + "=" * 78)
print("B. LENGTH BIAS")
print("=" * 78)
print("""
Each probe is two answers making the SAME claims from the SAME context
with the SAME citations. One is terse, one is padded with restatement
that asserts nothing new. A faithful judge must score them equally.
""")

length_report = measure_length_bias(judge)

print(f"  {'probe':<10}{'terse':>8}{'verbose':>10}{'delta':>9}   words (terse -> verbose)")
print("  " + "-" * 68)
for r in length_report.results:
    flag = "  MATERIAL" if r.is_material else ""
    wa, wb = len(r.probe.answer_a.split()), len(r.probe.answer_b.split())
    print(f"  {r.probe.probe_id:<10}{r.score_a:>8.3f}{r.score_b:>10.3f}"
          f"{r.delta:>+9.3f}   {wa:>3} -> {wb:<3}{flag}")

print(f"""
  mean delta     : {length_report.mean_delta:+.3f}
  max |delta|    : {length_report.max_abs_delta:.3f}
  material       : {length_report.n_material}/{len(length_report.results)}
  BIASED         : {length_report.is_biased}   ({length_report.direction})
""")

if length_report.is_biased and length_report.mean_delta < 0:
    print(f""">>> This judge PENALISES length, by {abs(length_report.mean_delta):.3f} on average.

  That is the opposite of the bias the literature reports for LLM
  judges, which tend to reward verbosity. The mechanism here is
  transparent: lexical faithfulness is the fraction of ANSWER terms
  found in the context, so padding words — "thank you for asking",
  "which means", "advisable" — are absent from the context and drag the
  ratio down. Longer answer, lower score, regardless of whether a single
  additional claim was made.

  Two judges with OPPOSITE biases are both wrong, and neither is
  correctable by choosing a threshold. The direction is a property of
  the scoring mechanism, not a tuning parameter.
""")
elif length_report.is_biased:
    print(f""">>> This judge REWARDS length, by {length_report.mean_delta:+.3f} on average.

  This is the documented failure mode for LLM judges, and it is aimed
  directly at Phase 4: the prompt ladder's upper rungs write longer
  answers by construction, so part of any advantage they showed is this
  bias rather than better answers. Section F quantifies how much.
""")
else:
    print("""  No material length effect. The pairs scored within the noise
  threshold, so comparisons between prompt variants are not obviously
  distorted by answer length.
""")


# =====================================================================
# C. Context-order sensitivity
# =====================================================================
print("\n" + "=" * 78)
print("C. CONTEXT-ORDER SENSITIVITY")
print("=" * 78)
print("""
The same answer, scored against the same context blocks in a different
order. The evidence is byte-identical; only position moves.
""")

order_report = measure_order_sensitivity(judge)
for r in order_report.results:
    flag = "  MATERIAL" if r.is_material else ""
    print(f"  {r.probe.probe_id:<10} original {r.score_a:.3f}   "
          f"shuffled {r.score_b:.3f}   delta {r.delta:+.3f}{flag}")

print(f"""
  mean delta : {order_report.mean_delta:+.3f}     BIASED: {order_report.is_biased}
""")

if not order_report.is_biased and isinstance(judge, LexicalJudge):
    print("""  Zero, and for a structural reason rather than a good one. Lexical
  faithfulness is set overlap, and set membership does not depend on
  order. This judge CANNOT exhibit position sensitivity — it is exempt
  from the test rather than passing it.

  The same category error as its perfect citation precision in Phase 4
  and its 1.000 faithfulness in Phase 5. Worth stating each time,
  because a column of zeros looks like evidence of robustness and is
  not.

  An LLM judge reads context sequentially and genuinely can be
  position-sensitive, so this probe only becomes informative with one.
""")
elif not order_report.is_biased:
    print("""  No material sensitivity. Retrieval ordering does not leak into the
  faithfulness score, so cross-configuration comparisons are not partly
  comparisons of rank order.
""")
else:
    print(f"""  The judge reads position as information. Any comparison across
  retrieval configurations is therefore partly a comparison of chunk
  ordering rather than answer quality — configurations that happen to
  place evidence earlier get a bonus unrelated to what they retrieved.
""")


# =====================================================================
# D. Self-consistency
# =====================================================================
print("\n" + "=" * 78)
print("D. SELF-CONSISTENCY — THE FLOOR ON MEASURABLE DIFFERENCES")
print("=" * 78)
print("""
The identical input, scored several times at temperature 0. This should
be exactly reproducible; hosted models often are not.

The size of the wobble is not a curiosity — it is a floor. A gap between
two configurations smaller than the judge's own run-to-run variation is
not a small effect, it is an unmeasurable one.
""")

consistency = measure_self_consistency(judge, n_repeats=3)
print(f"  spread across repeats: max {consistency.max_abs_delta:.4f}, "
      f"{consistency.n_material}/{len(consistency.results)} material\n")

if consistency.max_abs_delta == 0:
    print(f"""  Perfectly deterministic{' — by construction, since the lexical judge does arithmetic on token sets and calls no model.' if isinstance(judge, LexicalJudge) else '.'}

  Floor on measurable differences: 0.000. Any observed gap is real,
  though it may still be small enough not to matter.
""")
else:
    print(f"""  The judge disagrees with itself by up to {consistency.max_abs_delta:.3f} on identical
  input. Treat that as the resolution limit: differences below it
  between prompt variants cannot be distinguished from the judge's own
  noise, whatever a mean suggests.
""")


# =====================================================================
# E. Inter-judge agreement
# =====================================================================
print("\n" + "=" * 78)
print("E. INTER-JUDGE AGREEMENT")
print("=" * 78)
print("""
One judge's opinion is one opinion. Two judges disagreeing on a question
is a signal the question is genuinely ambiguous — which is information
about the QUESTION, not just about the judges.

Agreement is reported as Cohen's kappa rather than raw agreement,
because raw agreement is misleading when one verdict dominates: two
judges that both call 90% of answers faithful agree 82% of the time by
coincidence alone. Kappa subtracts that.
""")

pipeline = RAGPipeline(retriever, ExtractiveAnswerer(min_score=0.35),
                       depth=DEPTH)
sample = pipeline.run(golden[:40])
items = [(r.question_id, r.answer, format_context(r.hits)) for r in sample]

if has_key:
    reference_judge = LexicalJudge()
    label_a, label_b = "LLM judge", "lexical judge"
else:
    # A genuinely different second rater, not the same function twice.
    #
    # The first attempt here compared two LexicalJudge instances
    # distinguished by a `support_threshold` argument that turned out to
    # be stored and never read. The "two raters" were one function, and
    # kappa came back at exactly 1.000 — a perfect agreement score
    # between a thing and itself, which measures nothing and looked like
    # a result. The dead parameter has been removed.
    #
    # CoverageJudge scores per SENTENCE grounded rather than per term
    # matched, so it disagrees with term overlap precisely where an
    # answer is verbose but every sentence is supported.
    reference_judge = CoverageJudge()
    label_a, label_b = "lexical (term overlap)", "coverage (per sentence)"

agreement = judge_agreement(judge, reference_judge, items)

print(f"  {label_a} vs {label_b}, n={agreement.n}\n")
print(f"    raw agreement : {agreement.raw_agreement:.1%}")
print(f"    Cohen's kappa : {agreement.kappa:.3f}   ({agreement.interpretation})")
print(f"    disagreements : {len(agreement.disagreements)}")

for item_id, sa, sb in agreement.disagreements[:5]:
    print(f"      {item_id}: {label_a} {sa:.2f} vs {label_b} {sb:.2f}")

# Do the two raters actually differ? Agreement is uninterpretable
# without knowing whether disagreement was even possible.
probe_gap = max(
    abs(judge.faithfulness(p.answer_b, p.context).score
        - reference_judge.faithfulness(p.answer_b, p.context).score)
    for p in LENGTH_PROBES
)
print(f"""
  Max divergence between these two raters on the bias probes: {probe_gap:.3f}
""")

if len(agreement.disagreements) == 0 and probe_gap >= MATERIAL_DELTA:
    print(f"""  So they ARE different raters — up to {probe_gap:.3f} apart on the probes —
  and they still agree on every one of the {agreement.n} golden-set answers.

  That combination is the finding, and it is about the DATA, not the
  judges. The extractive baseline copies sentences verbatim out of the
  retrieved context, so every answer it produces is trivially grounded
  under any lexical formulation. Term overlap and sentence coverage
  cannot disagree about text that was lifted whole.

  The general lesson is worth stating because it is easy to get wrong:
  high inter-judge agreement on easy data is not evidence that either
  judge is reliable. A kappa of 1.000 here measures the absence of hard
  cases, and reporting it as judge reliability would be the same
  category error as the baseline's perfect citation precision in Phase
  4 and its 1.000 faithfulness in Phase 5.

  Agreement becomes informative once answers are PARAPHRASED — a
  generator that restates rather than copies is where two scoring
  formulations start to diverge, and where a semantic judge starts to
  disagree with both.
""")
elif len(agreement.disagreements) == 0:
    print("""  The two raters agree everywhere AND barely differ on the probes, so
  this comparison is close to tautological. Treat it as a check that
  the machinery computes, not as evidence about reliability.
""")
else:
    print(f"""  {len(agreement.disagreements)} disagreements out of {agreement.n}. Those are the interesting
  rows: questions where two different scoring formulations reach
  opposite verdicts, which usually means the answer sits genuinely near
  the boundary rather than that one judge is simply wrong.
""")


# --- Chart ------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ids = [r.probe.probe_id for r in length_report.results]
deltas = [r.delta for r in length_report.results]
colors = ["#F6735B" if abs(d) >= MATERIAL_DELTA else "#5AD8A6" for d in deltas]
ax1.barh(ids, deltas, color=colors, edgecolor="white")
ax1.axvline(0, color="#2B2B2B", linewidth=1.2)
for x in (MATERIAL_DELTA, -MATERIAL_DELTA):
    ax1.axvline(x, linestyle="--", color="#999", linewidth=1)
ax1.set_xlabel("verbose score − terse score   (0 = unbiased)")
ax1.set_title(f"Length bias: {judge.name}\ndashed = material threshold",
              fontweight="bold")
ax1.grid(axis="x", linestyle="--", alpha=0.4)
ax1.set_axisbelow(True)

dims = ["length", "context_order", "consistency"]
mags = [length_report.max_abs_delta, order_report.max_abs_delta,
        consistency.max_abs_delta]
ax2.bar(dims, mags,
        color=["#F6735B" if m >= MATERIAL_DELTA else "#5AD8A6" for m in mags],
        edgecolor="white")
ax2.axhline(MATERIAL_DELTA, linestyle="--", color="#2B2B2B", linewidth=1.2,
            label=f"material ({MATERIAL_DELTA})")
ax2.set_ylabel("max |score change|")
ax2.set_title("Sensitivity by dimension\nlower is better", fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)

plt.suptitle("Phase 6 — auditing the judge", fontsize=13,
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "06_judge_bias.png", dpi=140, bbox_inches="tight",
            metadata={"Software": None})
print(f"\n  Saved -> {FIG_DIR}/06_judge_bias.png")


# =====================================================================
# F. Does any of this invalidate Phase 4?
# =====================================================================
print("\n" + "=" * 78)
print("F. DOES THE BIAS INVALIDATE PHASE 4?")
print("=" * 78)
print("""
A bias measurement is only useful if it is converted into a statement
about a conclusion someone might act on. The exposed conclusion here is
the Phase 4 prompt ladder, because its rungs differ systematically in
answer length.
""")

prompt_words = {name: len(v.template.split()) for name, v in VARIANTS.items()}
spread = max(prompt_words.values()) - min(prompt_words.values())

# Bias per word, from the probes: mean delta over mean word gap.
word_gaps = [len(r.probe.answer_b.split()) - len(r.probe.answer_a.split())
             for r in length_report.results]
per_word = (length_report.mean_delta / float(np.mean(word_gaps))
            if np.mean(word_gaps) else 0.0)

print("  Prompt instruction length by rung (a proxy for answer length,")
print("  since the instructions ask for progressively more):\n")
for name in ("naive", "grounded", "grounded_refusal", "cited", "strict"):
    print(f"    {name:<18} {prompt_words[name]:>4} words")

print(f"""
  Measured length effect : {per_word:+.5f} score per extra answer word
  Probe word gaps        : {min(word_gaps)} to {max(word_gaps)} words

  If two prompt variants differ by ~30 words of answer, length bias
  alone moves faithfulness by ~{abs(per_word) * 30:.3f}.
""")

if abs(per_word) * 30 >= MATERIAL_DELTA:
    print(f""">>> That is at or above the {MATERIAL_DELTA} material threshold.

  So a Phase 4 ladder comparison run through THIS judge would be
  substantially contaminated: a variant could win or lose on verbosity
  alone. Any faithfulness comparison across the ladder needs either a
  judge without this bias, or answers length-normalised before scoring.

  Stating it precisely: the bias does not invalidate Phase 4's
  JUDGE-FREE results — refusal rates, citation integrity, and blame
  attribution are all counts and regexes, untouched by this. It
  invalidates faithfulness comparisons ACROSS variants, which is a
  Phase 5 output, not a Phase 4 one.
""")
else:
    print(f"""  That is below the {MATERIAL_DELTA} material threshold, so a ~30-word
  difference between variants would not move the verdict. The ladder
  comparison survives this judge's length effect.
""")


# =====================================================================
# G. Verdict
# =====================================================================
print("=" * 78)
print("PHASE 6 VERDICT")
print("=" * 78)

audited = [("length", length_report), ("context order", order_report),
           ("consistency", consistency)]
biased_dims = [name for name, rep in audited if rep.is_biased]

print(f"""
Audited {judge.name} on three dimensions beyond accuracy.

1. ACCURACY IS NOT ENOUGH.
   Phase 5 asked whether the judge gets known cases right; this phase
   asks what else it responds to. {'It failed' if biased_dims else 'It showed no material bias'} on {len(biased_dims)} of 3
   dimensions{': ' + ', '.join(biased_dims) if biased_dims else ''}.

2. PAIRED PROBES, NOT CORRELATIONS.
   Correlating score against length across the golden set would
   confound bias with quality — longer answers may simply be better.
   Each probe holds claims, context and citations fixed and varies one
   thing, so a gap is attributable rather than suggestive.

3. LENGTH EFFECT: {length_report.mean_delta:+.3f} MEAN.
   {'Penalises verbosity — the opposite of the documented LLM-judge bias, and a direct consequence of scoring by term overlap.' if length_report.mean_delta < -0.05 else 'Rewards verbosity — the documented LLM-judge failure mode.' if length_report.mean_delta > 0.05 else 'No material effect.'}

4. THE FLOOR ON MEASURABLE DIFFERENCES IS {consistency.max_abs_delta:.3f}.
   Gaps smaller than the judge's own run-to-run variation are not small
   effects, they are unmeasurable ones.

5. AGREEMENT: kappa {agreement.kappa:.3f} ({agreement.interpretation}).
   Reported over chance-corrected agreement rather than raw, because
   raw agreement is inflated whenever one verdict dominates.

{'' if has_key else '''NOT YET MEASURED — no credential. The lexical judge was audited, which
is genuinely informative about IT, but the comparison this phase exists
for is a semantic judge against a lexical one. Two specific results are
pending:

  - whether gpt-4o shows the documented POSITIVE length bias, giving
    two judges with opposite biases and no threshold that fixes either
  - the disagreement set between them, which is where reading meaning
    and counting words diverge, and is the most interesting output here

'''}STILL OPEN — carried into Phase 7:

  - Self-preference bias needs two generation models, not two judges.
    A judge scoring its own output family higher is a separate effect
    from length, and the current design cannot isolate it.
  - The probe set is {len(LENGTH_PROBES) + len(ORDER_PROBES)} hand-written pairs. Enough to detect a
    gross bias, not enough to estimate its size precisely. The reported
    per-word figure is an order of magnitude, not a coefficient.
  - mh-011, still. Both refund windows retrieve at depth {DEPTH}; whether
    any variant FLAGS the conflict rather than silently choosing one
    remains unanswered and needs the LLM arms.

HANDOFF TO PHASE 7: the deployment decision memo — what ships, at what
context budget, with which known failure modes and which measurement
caveats attached to each claim.
""")
