"""Tests for the judge bias audit.

The probes are the instrument here, so most of these check the
instrument rather than any particular judge: do the pairs actually hold
everything else constant, does the measurement detect a bias that is
known to be present, and does it report zero when there is none.
"""
import pytest

from src.evaluation.judge import CoverageJudge, Judgement, LexicalJudge
from src.evaluation.judge_bias import (
    LENGTH_PROBES,
    MATERIAL_DELTA,
    ORDER_PROBES,
    _shuffle_context_blocks,
    cohens_kappa,
    judge_agreement,
    measure_length_bias,
    measure_order_sensitivity,
    measure_self_consistency,
)


# ---------------------------------------------------------------------
# Probe construction — the pairs must isolate ONE variable
# ---------------------------------------------------------------------
def test_length_probes_actually_differ_in_length():
    for p in LENGTH_PROBES:
        short, long = len(p.answer_a.split()), len(p.answer_b.split())
        assert long > short * 2, f"{p.probe_id}: length gap too small to probe"


def test_length_probes_share_their_citations():
    """If the verbose version cited more sources it would be a different
    answer, not a longer one — and the probe would measure citation
    count rather than length."""
    from src.generation.prompts import extract_citations
    for p in LENGTH_PROBES:
        assert set(extract_citations(p.answer_a)) == \
               set(extract_citations(p.answer_b)), p.probe_id


def test_length_probes_share_their_context():
    """Only the answer may vary. Different context would confound."""
    for p in LENGTH_PROBES:
        assert p.context, p.probe_id


def test_order_probes_use_identical_answers():
    """For the order probe the ANSWER must be constant — only the
    context ordering is manipulated."""
    for p in ORDER_PROBES:
        assert p.answer_a == p.answer_b, p.probe_id


def test_shuffle_changes_order_but_preserves_every_block():
    context = "[a-001] One\nAlpha.\n\n[b-002] Two\nBeta.\n\n[c-003] Three\nGamma."
    shuffled = _shuffle_context_blocks(context)
    assert shuffled != context, "shuffle must actually reorder"
    assert sorted(shuffled.split("\n\n")) == sorted(context.split("\n\n")), \
        "shuffle must preserve every block exactly"


def test_shuffle_is_deterministic():
    context = "[a-001] One\nAlpha.\n\n[b-002] Two\nBeta.\n\n[c-003] Three\nGamma."
    assert _shuffle_context_blocks(context) == _shuffle_context_blocks(context)


def test_shuffle_handles_a_single_block():
    """Nothing to reorder — must not loop forever looking for a change."""
    single = "[a-001] Only\nOne block."
    assert _shuffle_context_blocks(single) == single


# ---------------------------------------------------------------------
# Measurement must detect known bias and known absence of bias
# ---------------------------------------------------------------------
def test_length_bias_detected_in_a_judge_known_to_have_it():
    """LexicalJudge scores the fraction of ANSWER terms found in the
    context, so padding necessarily lowers the score. The measurement
    must find that."""
    report = measure_length_bias(LexicalJudge())
    assert report.is_biased
    assert report.mean_delta < -MATERIAL_DELTA
    assert "penalises" in report.direction


def test_no_length_bias_reported_for_a_length_blind_judge():
    """A judge ignoring its inputs has no bias, and the measurement must
    say so rather than finding one anyway."""
    class Constant(LexicalJudge):
        name = "constant"
        def faithfulness(self, a, c): return Judgement("faithfulness", 0.8)

    report = measure_length_bias(Constant())
    assert not report.is_biased
    assert report.mean_delta == 0.0
    assert report.direction == "none detected"


def test_order_sensitivity_is_zero_for_a_set_based_judge():
    """Set overlap cannot depend on order. This judge is exempt from the
    test rather than passing it, which the notebook states explicitly."""
    assert not measure_order_sensitivity(LexicalJudge()).is_biased


def test_order_sensitivity_detected_when_present():
    class PositionSensitive(LexicalJudge):
        name = "position_sensitive"
        def faithfulness(self, answer, context):
            # Score by which block happens to come first.
            first = context.split("\n\n")[0]
            return Judgement("faithfulness", 1.0 if "strm" in first else 0.2)

    assert measure_order_sensitivity(PositionSensitive()).is_biased


def test_self_consistency_is_perfect_for_a_deterministic_judge():
    report = measure_self_consistency(LexicalJudge(), n_repeats=3)
    assert report.max_abs_delta == 0.0
    assert not report.is_biased


def test_self_consistency_detects_a_wobbling_judge():
    class Wobbly(LexicalJudge):
        name = "wobbly"
        def __init__(self):
            self._n = 0
        def faithfulness(self, a, c):
            self._n += 1
            return Judgement("faithfulness", 0.9 if self._n % 2 else 0.2)

    assert measure_self_consistency(Wobbly(), n_repeats=3).is_biased


# ---------------------------------------------------------------------
# Cohen's kappa
# ---------------------------------------------------------------------
def test_kappa_is_one_for_perfect_agreement():
    assert cohens_kappa([True, False, True], [True, False, True]) == 1.0


def test_kappa_is_negative_for_systematic_disagreement():
    assert cohens_kappa([True, False, True, False],
                        [False, True, False, True]) == -1.0


def test_kappa_corrects_for_chance():
    """The property that makes kappa worth using: two raters that both
    say True almost always agree ~constantly by coincidence, and kappa
    must not reward that."""
    a = [True] * 18 + [False, True]
    b = [True] * 18 + [True, False]
    raw = sum(x == y for x, y in zip(a, b)) / len(a)
    assert raw >= 0.85
    assert cohens_kappa(a, b) < raw, "kappa must discount chance agreement"


def test_kappa_is_undefined_when_a_rater_has_no_variance():
    """Regression, and the more important of the two.

    Phase 6 originally reported 'Cohen's kappa: 1.000 (almost perfect)'
    for two judges that agreed on all 40 answers — but BOTH had called
    every answer faithful, so expected agreement was 1.0 and kappa was
    0/0. Returning 1.0 there claims near-perfect reliability from two
    raters that never said anything else, which is exactly the
    overstatement the notebook's own caveat warns against two
    paragraphs later.
    """
    import math
    assert math.isnan(cohens_kappa([True] * 5, [True] * 5))
    assert math.isnan(cohens_kappa([False] * 5, [False] * 5))

    # But only when BOTH are constant. With one rater constant and the
    # other varying, expected agreement equals the varying rater's rate
    # and kappa is a well-defined 0 — "no better than chance", which is
    # the correct reading: a constant rater carries no information, but
    # the quantity is still computable.
    assert cohens_kappa([True] * 5, [True, True, True, True, False]) == 0.0


def test_undefined_kappa_is_reported_as_such():
    from src.evaluation.judge_bias import AgreementReport
    report = AgreementReport(judge_a="a", judge_b="b", n=5,
                             raw_agreement=1.0, kappa=float("nan"),
                             disagreements=[])
    assert not report.has_variance
    assert "UNDEFINED" in report.interpretation


def test_consistency_direction_is_not_described_as_length():
    """The consistency delta is max-min and cannot be negative, so a
    shared 'favours the longer version' label was meaningless there."""
    from src.evaluation.judge_bias import BiasReport, ProbeResult, LENGTH_PROBES
    rep = BiasReport("j", "consistency",
                     [ProbeResult(LENGTH_PROBES[0], 0.2, 0.9)])
    assert "longer" not in rep.direction
    assert "itself" in rep.direction


def test_kappa_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="same length"):
        cohens_kappa([True], [True, False])


# ---------------------------------------------------------------------
# Agreement, and the two raters being genuinely distinct
# ---------------------------------------------------------------------
def test_the_two_keyless_judges_are_genuinely_different():
    """Regression: agreement was first computed between two LexicalJudge
    instances distinguished by a `support_threshold` argument that was
    stored and never read. They were one function, kappa came back at
    exactly 1.000, and a perfect score between a thing and itself looked
    like a result.
    """
    lex, cov = LexicalJudge(), CoverageJudge()
    gaps = [abs(lex.faithfulness(p.answer_b, p.context).score
                - cov.faithfulness(p.answer_b, p.context).score)
            for p in LENGTH_PROBES]
    assert max(gaps) >= MATERIAL_DELTA, \
        "the two keyless raters must actually diverge somewhere"


def test_lexical_judge_has_no_dead_threshold_parameter():
    """The dead parameter must stay gone."""
    with pytest.raises(TypeError):
        LexicalJudge(support_threshold=0.4)


def test_agreement_reports_disagreements_with_both_scores():
    class AlwaysHigh(LexicalJudge):
        name = "hi"
        def faithfulness(self, a, c): return Judgement("faithfulness", 0.95)

    class AlwaysLow(LexicalJudge):
        name = "lo"
        def faithfulness(self, a, c): return Judgement("faithfulness", 0.10)

    items = [(f"q{i}", "an answer", "some context") for i in range(6)]
    report = judge_agreement(AlwaysHigh(), AlwaysLow(), items)
    assert len(report.disagreements) == 6
    assert report.raw_agreement == 0.0
    for item_id, sa, sb in report.disagreements:
        assert sa == pytest.approx(0.95) and sb == pytest.approx(0.10)


def test_agreement_binarises_at_the_stated_threshold():
    """Two judges 0.4 apart but on the same side of the line agree; two
    judges 0.05 apart straddling it do not. That is intended — the
    harness's actual decision is 'faithful or not'."""
    class At(LexicalJudge):
        def __init__(self, score, name):
            self._s, self.name = score, name
        def faithfulness(self, a, c): return Judgement("faithfulness", self._s)

    items = [("q1", "a", "c")]
    # 0.75 and 0.99 both clear the 0.7 line, so the raters agree. Kappa
    # is undefined on a single item with no variance, which is why the
    # assertion is on raw agreement — the quantity that is defined here.
    assert judge_agreement(At(0.75, "a"), At(0.99, "b"), items).raw_agreement == 1.0
    # 0.68 and 0.72 straddle it, so a 0.04 gap is a real disagreement
    # while the 0.24 gap above was not.
    assert judge_agreement(At(0.68, "a"), At(0.72, "b"), items).raw_agreement == 0.0


def test_coverage_judge_scores_a_refusal_as_vacuously_faithful():
    """No sentences means no claims, which is faithful by default."""
    assert CoverageJudge().faithfulness("", "some context").score == 1.0
