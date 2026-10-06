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
import json
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
from src.evaluation.judged_arms import write_metrics
from src.generation.extractive import ExtractiveAnswerer
from src.generation.pipeline import RAGPipeline
from src.generation.prompts import VARIANTS, format_context
from src.generation.refusal import check_refusal
from src.llm.provider import provider_ready
from src.retrieval.chunking import STRATEGIES
from src.retrieval.retrievers import BM25Retriever

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR = Path("reports/metrics")

STRATEGY = "markdown_section"
DEPTH = 15


def _record(name: str) -> dict | None:
    """A measured record from an earlier phase, if one has been written.

    The judged LLM arms cannot be recomputed without a credential, so
    where this audit needs them it reads the committed record instead of
    silently skipping the check.
    """
    path = METRICS_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None

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

if has_key:
    # A semantic judge returns claim counts and a stated reason. Printing
    # them is what separates "responds to length" from "found something
    # in the longer answer", which the score alone cannot distinguish.
    print("  What the judge said about each longer version:\n")
    for r in length_report.results:
        j = judge.faithfulness(r.probe.answer_b, r.probe.context)
        print(f"    {r.probe.probe_id}: {j.n_supported} of {j.n_claims} claims supported")
        print(f"      {j.reasoning}\n")

if has_key and length_report.is_biased and length_report.mean_delta < 0:
    print(f""">>> The longer version scored lower on {length_report.n_material} of {len(length_report.results)} probes, by {abs(length_report.mean_delta):.3f} on
    average. Read the judge's reasons above before calling that a
    length bias.

  The probes were written on the assumption that their padding asserts
  nothing new. That assumption was safe for a lexical judge, for which
  filler is simply vocabulary absent from the context. A judge that
  reads meaning does not see filler: it splits the answer into claims,
  and where it marks one unsupported it names the sentence. If the
  sentence it names is an inference or a piece of advice the context
  never states — "which means up to four devices in your household...",
  "acting promptly is advisable" — then the judge is not responding to
  length. It is responding to an added claim, and the probe contained
  one.

  So this number is two things at once, and only one of them is about
  the judge:

    - As a LENGTH bias it is an upper bound, not an estimate. These
      probes do not hold claims fixed for a judge that counts claims.
    - As a description of the judge it is informative: it is strict
      about elaboration. An answer that adds a helpful-sounding
      inference loses faithfulness. For a support assistant that is the
      direction to be strict in.

  Section F tests the length reading directly, against real answers.
""")
elif length_report.is_biased and length_report.mean_delta < 0:
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
if agreement.has_variance:
    print(f"    Cohen's kappa : {agreement.kappa:.3f}   ({agreement.interpretation})")
else:
    print(f"    Cohen's kappa : undefined")
    print(f"                    {agreement.interpretation}")
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

n_faithful = sum(judge.faithfulness(a, c).score >= 0.7 for _, a, c in items)

if len(agreement.disagreements) == 0 and probe_gap >= MATERIAL_DELTA:
    print(f"""  So they ARE different raters — up to {probe_gap:.3f} apart on the probes —
  and they still agree on all {agreement.n} golden-set answers, including
  which {agreement.n - n_faithful} to reject. Both raters vary ({n_faithful}/{agreement.n} judged faithful),
  so this is genuine perfect agreement rather than the degenerate case
  where kappa is undefined because nobody ever disagreed with anything.

  Why they agree here despite diverging on the probes: the disagreement
  the probes expose is about PADDING. Term overlap punishes filler words
  because they are absent from the context; sentence coverage does not,
  because a padded sentence can still be mostly grounded. The extractive
  baseline never pads — it copies sentences verbatim — so the one thing
  these two formulations disagree about does not occur in this data.

  That is worth stating carefully, because it is the opposite of a
  reassuring result. Agreement here is evidence that the DATA lacks the
  feature that separates the raters, not evidence that either rater is
  right. Both are still lexical, and both still score contradictions as
  faithful (Phase 5, Section B).

  Agreement becomes informative once answers are PARAPHRASED. A
  generator that restates rather than copies produces exactly the
  padding-like variation these formulations treat differently — and a
  semantic judge would then disagree with both.
""")
elif len(agreement.disagreements) == 0:
    print("""  The two raters agree everywhere AND barely differ on the probes, so
  this comparison is close to tautological. Treat it as a check that
  the machinery computes, not as evidence about reliability.
""")
elif has_key:
    ref_validation = validate_judge(reference_judge)
    answers_by_id = {qid: ans for qid, ans, _ in items}
    on_refusals = sum(1 for qid, _, _ in agreement.disagreements
                      if check_refusal(answers_by_id[qid]).is_refusal)
    print(f"""  {len(agreement.disagreements)} disagreements out of {agreement.n}.

  This is not two competent raters splitting hard cases, and reading it
  that way would be a mistake. The raters are not peers: on the Phase 5
  validation suite this judge scored {validation.overall_accuracy:.0%} and the lexical judge
  {ref_validation.overall_accuracy:.0%}. Agreement with an instrument known to be wrong is not a
  virtue, so low kappa here counts against the lexical judge rather
  than against either rater equally.

  The disagreements also have a shape. {on_refusals} of the {len(agreement.disagreements)} are on REFUSALS, which
  the lexical judge scores as unfaithful (a refusal shares no words
  with the context) and a semantic judge scores as vacuously faithful
  (it asserts nothing). The rest run the other way: answers stitched
  from copied fragments, which overlap the context almost perfectly and
  no longer say what the context says.

  What kappa between a validated and an unvalidated rater cannot do is
  certify the validated one. That would need a second rater that also
  passes the gate — a different model family, or human labels on a
  sample — and this project has neither.
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
ax1.set_xlabel("verbose score − terse score   (0 = scored the same)")
ax1.set_title(f"Length probes: {judge.name}\ndashed = material threshold",
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

gen_record = _record("04_generation_arms.json")
judged_record = _record("05_judged_arms.json")
measured_words = {
    name: (gen_record or {}).get("arms", {}).get(f"llm_{name}", {}).get("mean_answer_words")
    for name in VARIANTS}
have_lengths = all(v is not None for v in measured_words.values())

if have_lengths:
    print("  Instruction length by rung, beside the answer length it actually")
    print("  produced (measured, from the Phase 4 record):\n")
    print(f"    {'rung':<18} {'instruction':>12} {'mean answer':>12}")
    for name in ("naive", "grounded", "grounded_refusal", "cited", "strict"):
        print(f"    {name:<18} {prompt_words[name]:>6} words {measured_words[name]:>6.0f} words")
    longest = max(measured_words, key=measured_words.get)
    which = ("the rung with the SHORTEST instruction"
             if prompt_words[longest] == min(prompt_words.values())
             else "not the rung with the longest instruction")
    print(f"""
  This notebook used to treat instruction length as a proxy for answer
  length, on the reasoning that the upper rungs ask for more. Measured,
  the longest answers come from `{longest}` at {measured_words[longest]:.0f} words:
  {which}. An unconstrained model
  elaborates; an instruction to stay inside the context and cite it
  makes answers shorter. The proxy pointed the wrong way, so any
  contamination argument has to use these lengths.
""")
else:
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

  So a Phase 4 ladder comparison run through THIS judge would, if the
  per-word figure held on real answers, be substantially contaminated:
  a variant could win or lose on verbosity alone. A faithfulness
  comparison across the ladder would then need either a judge without
  the effect, or answers length-normalised before scoring.

  Stating the exposure precisely: Phase 4's JUDGE-FREE results —
  refusal rates, citation integrity, and blame attribution — are counts
  and regexes, untouched by any of this. What is exposed is
  faithfulness compared ACROSS variants, which is a Phase 5 output.
""")
else:
    print(f"""  That is below the {MATERIAL_DELTA} material threshold, so a ~30-word
  difference between variants would not move the verdict. The ladder
  comparison survives this judge's length effect.
""")


# =====================================================================
# G. Verdict
# =====================================================================
# --- The extrapolation, tested ------------------------------------------
pairs = (judged_record or {}).get("same_question_comparison", {})
length_check = None
if pairs and judge.name in str((judged_record or {}).get("judge", "")):
    print("""  THE EXTRAPOLATION, TESTED. A per-word effect read off three probes
  is a prediction, and Phase 5 recorded the data to check it: the same
  questions answered by two arms whose answers differ in length.
""")
    for label, row in pairs.items():
        a_name, b_name = [s.strip() for s in label.split(" - ")]
        words = row["mean_answer_words"]
        gap_words = words[a_name] - words[b_name]
        predicted = per_word * gap_words
        f = row["faithfulness"]
        inside = f["ci_low"] <= predicted <= f["ci_high"]
        length_check = {"pair": label, "word_gap": gap_words,
                        "predicted": predicted, "observed": f["difference"],
                        "ci_low": f["ci_low"], "ci_high": f["ci_high"],
                        "consistent": inside}
        print(f"    {label}, {row['n_both_answered']} questions both answered")
        print(f"      answer length          : {words[a_name]:.0f} vs {words[b_name]:.0f} words ({gap_words:+.0f})")
        print(f"      predicted from probes  : {predicted:+.3f} faithfulness")
        print(f"      observed               : {f['difference']:+.3f} [{f['ci_low']:+.3f}, {f['ci_high']:+.3f}]")
        if inside:
            print("""
  The prediction falls inside the observed interval, so the per-word
  reading is not contradicted by real answers. The contamination
  warning above stands.
""")
        else:
            print(f"""
>>> The prediction is outside the observed interval. Real answers that
    differ by {abs(gap_words):.0f} words do not show the gap a per-word length penalty
    requires.

  That supports the second reading of Section B. The judge does not
  charge by the word. It charged the probes for specific added claims,
  and real answers that are longer without adding unsupported claims
  are not penalised. The contamination warning above does not apply to
  this judge on this data.

  The caveat runs the other way from the usual one. This comparison is
  observational: the two arms differ in more than length, so it cannot
  prove the absence of a small length effect. What it rules out is an
  effect of the size the probes implied.
""")

if has_key:
    # Recorded for the same reason as the Phase 4 and 5 results: CI has
    # no key, so the memo's claims about the judge are checked against
    # this file. Written only when an LLM judge was actually audited.
    write_metrics(METRICS_DIR / "06_judge_audit.json", {
        "judge": repr(judge),
        "length_probes": {
            "n": len(length_report.results),
            "n_material": length_report.n_material,
            "mean_delta": round(length_report.mean_delta, 6),
            "per_word": round(per_word, 5),
        },
        "order_probes": {"n": len(order_report.results),
                         "max_abs_delta": round(order_report.max_abs_delta, 6)},
        "self_consistency_max_spread": round(consistency.max_abs_delta, 6),
        "agreement_with_lexical": {
            "n": agreement.n,
            "raw_agreement": round(agreement.raw_agreement, 6),
            "kappa": (round(agreement.kappa, 6) if agreement.has_variance else None),
            "disagreements": len(agreement.disagreements),
        },
        "length_extrapolation_check": None if length_check is None else {
            "pair": length_check["pair"],
            "word_gap": round(length_check["word_gap"], 1),
            "predicted": round(length_check["predicted"], 6),
            "observed": length_check["observed"],
            "ci_low": length_check["ci_low"],
            "ci_high": length_check["ci_high"],
            "consistent": bool(length_check["consistent"]),
        },
    })
    print(f"  Saved -> {METRICS_DIR}/06_judge_audit.json\n")

print("=" * 78)
print("PHASE 6 VERDICT")
print("=" * 78)

if has_key:
    length_verdict = (
        f"The longer versions scored lower on {length_report.n_material} of "
        f"{len(length_report.results)} probes. The judge's reasons name\n"
        "   specific added claims, so as a LENGTH bias this is an upper bound; "
        "as a\n   description of the judge it shows strictness about elaboration."
        if length_report.mean_delta < -0.05 else
        "Rewards verbosity — the documented LLM-judge failure mode."
        if length_report.mean_delta > 0.05 else "No material effect.")
    if length_check is not None:
        length_verdict += (
            f"\n   Tested on real answers {abs(length_check['word_gap']):.0f} words apart: predicted "
            f"{length_check['predicted']:+.3f}, observed {length_check['observed']:+.3f}\n"
            f"   [{length_check['ci_low']:+.3f}, {length_check['ci_high']:+.3f}] — "
            + ("consistent with a per-word effect." if length_check["consistent"]
               else "a per-word penalty of that size is ruled out."))
    agreement_verdict = (
        "Against a rater that failed the accuracy gate, so low agreement counts\n"
        "   against that rater. It does not certify this one: that needs a second\n"
        "   rater that also passes.")
else:
    length_verdict = (
        "Penalises verbosity — the opposite of the documented LLM-judge bias, and a direct consequence of scoring by term overlap."
        if length_report.mean_delta < -0.05 else
        "Rewards verbosity — the documented LLM-judge failure mode."
        if length_report.mean_delta > 0.05 else "No material effect.")
    agreement_verdict = (
        "Chance-corrected rather than raw, because raw agreement is inflated whenever one verdict dominates."
        if agreement.has_variance else
        "Both raters called every answer faithful, so expected agreement is 1.0 and kappa is 0/0. Reporting the raw 100% as reliability would be the error this metric exists to prevent.")

mh011_record = (judged_record or {}).get("mh011", {})
llm_mh011 = {k: v for k, v in mh011_record.items() if k.startswith("llm_")}
if llm_mh011:
    stances = "; ".join(f"{k} {v['stance']} (correctness {v['correctness']:.2f})"
                        for k, v in llm_mh011.items())
    surfaced = [k for k, v in llm_mh011.items() if v["stance"] == "states_both"]
    mh011_open = f"""  - mh-011 is answered, from the Phase 5 record: {stances}.
    {'At least one judged arm states both refund windows.' if surfaced else 'No judged arm states both refund windows.'} The judge's correctness
    scores are against a reference that names the conflict, so a high
    score for an answer that omits it is the judge being lenient on
    omission: it checks the claims an answer makes, and a missing
    caveat is not a false claim. That is a blind spot to design for."""
else:
    mh011_open = f"""  - mh-011, still. Both refund windows retrieve at depth {DEPTH}; whether
    any variant FLAGS the conflict rather than silently choosing one
    remains unanswered and needs the LLM arms."""

audited = [("length", length_report), ("context order", order_report),
           ("consistency", consistency)]
biased_dims = [name for name, rep in audited if rep.is_biased]

print(f"""
Audited {judge.name} on three dimensions beyond accuracy.

1. ACCURACY IS NOT ENOUGH.
   Phase 5 asked whether the judge gets known cases right; this phase
   asks what else it responds to. {'It showed a material effect' if biased_dims else 'It showed no material bias'} on {len(biased_dims)} of 3
   dimensions{': ' + ', '.join(biased_dims) if biased_dims else ''}.

2. PAIRED PROBES, NOT CORRELATIONS.
   Correlating score against length across the golden set would
   confound bias with quality — longer answers may simply be better.
   Each probe holds claims, context and citations fixed and varies one
   thing, so a gap is attributable rather than suggestive.

3. LENGTH EFFECT ON THE PROBES: {length_report.mean_delta:+.3f} MEAN.
   {length_verdict}

4. THE FLOOR ON MEASURABLE DIFFERENCES IS {consistency.max_abs_delta:.3f}.
   Gaps smaller than the judge's own run-to-run variation are not small
   effects, they are unmeasurable ones.

5. AGREEMENT: {"kappa " + format(agreement.kappa, ".3f") if agreement.has_variance else "UNDEFINED"}.
   {agreement_verdict}

{'' if has_key else '''NOT MEASURED IN THIS RUN — no credential. The lexical judge was audited
here, which is genuinely informative about IT. The audit of the LLM
judge — its probe results, the test of the length reading against real
answers, and its agreement with the lexical judge — comes from the
credentialed run and is recorded in reports/metrics/06_judge_audit.json.

'''}STILL OPEN — carried into Phase 7:

  - Self-preference bias needs two generation models, not two judges.
    A judge scoring its own output family higher is a separate effect
    from length, and the current design cannot isolate it.
  - The probe set is {len(LENGTH_PROBES) + len(ORDER_PROBES)} hand-written pairs. Enough to detect a
    gross bias, not enough to estimate its size precisely. The reported
    per-word figure is an order of magnitude, not a coefficient.
{mh011_open}

HANDOFF TO PHASE 7: the deployment decision memo — what ships, at what
context budget, with which known failure modes and which measurement
caveats attached to each claim.
""")
