"""Tests for the judged-arms helpers.

These cover the decisions that are easy to get quietly wrong: which arms
run, whether a planned run fits its budget, and what the contradiction
check does and does not claim.
"""
import json
import math

import pytest

from src.evaluation.judged_arms import (
    DEFAULT_ARMS,
    JudgePlan,
    contradiction_stance,
    estimate_usd,
    paired_mean_difference,
    parse_arms,
    plan_judging,
    report_to_dict,
    write_metrics,
)
from src.evaluation.rag_metrics import RunReport
from src.generation.prompts import VARIANTS
from src.generation.refusal import CANONICAL_REFUSAL

AVAILABLE = list(VARIANTS)


# ---------------------------------------------------------------------
# Arm selection
# ---------------------------------------------------------------------
def test_default_arms_exist_as_prompt_variants():
    """A default naming a variant that was renamed would judge nothing."""
    assert set(DEFAULT_ARMS) <= set(AVAILABLE)


@pytest.mark.parametrize("spec", [None, "", "   "])
def test_unset_spec_gives_the_defaults(spec):
    assert parse_arms(spec, AVAILABLE) == list(DEFAULT_ARMS)


def test_all_expands_to_every_variant():
    assert parse_arms("all", AVAILABLE) == AVAILABLE
    assert parse_arms(" ALL ", AVAILABLE) == AVAILABLE


def test_explicit_list_is_respected_in_order_without_duplicates():
    assert parse_arms("strict, cited ,strict", AVAILABLE) == ["strict", "cited"]


def test_unknown_arm_raises_rather_than_being_dropped():
    """Dropping it silently would leave a table with a row missing and
    no indication an arm was ever requested."""
    with pytest.raises(ValueError, match="citted"):
        parse_arms("naive,citted", AVAILABLE)


# ---------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------
def test_plan_counts_every_call():
    plan = plan_judging(n_questions=120, n_references=120,
                        n_validation_cases=9, n_llm_arms=2)
    assert plan.validation_calls == 18
    assert plan.calls_per_run == 360          # 120 x (faith + rel + correct)
    assert plan.baseline_calls == 360
    assert plan.arm_calls == 720
    assert plan.total_calls == 18 + 360 + 720


def test_plan_without_references_skips_correctness_calls():
    plan = plan_judging(n_questions=10, n_references=0,
                        n_validation_cases=0, n_llm_arms=1)
    assert plan.calls_per_run == 20


def test_plan_fits_is_checked_against_the_ceiling():
    plan = JudgePlan(validation_calls=18, calls_per_run=360, n_llm_arms=5)
    assert plan.total_calls == 2178
    assert not plan.fits(2000)        # all five arms overflow the default
    assert plan.fits(2178)
    assert plan.fits(0)               # 0 disables the ceiling


def test_cost_estimate_scales_linearly_and_is_positive():
    assert estimate_usd(0) == 0
    assert estimate_usd(100) > 0
    assert math.isclose(estimate_usd(200), 2 * estimate_usd(100))


# ---------------------------------------------------------------------
# The contradiction check
# ---------------------------------------------------------------------
def test_stance_states_both():
    a = "One article says 14 days [bill-003], another says 30 days [bill-002]."
    assert contradiction_stance(a) == "states_both"


def test_stance_states_one():
    assert contradiction_stance(
        "You can get a refund within 30 days [bill-002].") == "states_one"
    assert contradiction_stance(
        "Refunds are available for 14 days from the charge.") == "states_one"


def test_stance_states_neither():
    assert contradiction_stance(
        "Contact support with the charge date and amount.") == "states_neither"


def test_stance_refused():
    assert contradiction_stance(CANONICAL_REFUSAL) == "refused"


def test_stance_does_not_match_figures_inside_other_numbers():
    """'2014' and '$130' are not refund windows."""
    a = "Since 2014 the fee has been $130; contact support for refunds."
    assert contradiction_stance(a) == "states_neither"


def test_stance_ignores_figures_in_citation_ids():
    """A citation id is a source label, not a claim about a window —
    even when its number happens to equal one of the figures."""
    a = "Refunds are covered in [bill-014] and [trial-030]."
    assert contradiction_stance(a) == "states_neither"
    b = "You have 30 days [bill-014]."
    assert contradiction_stance(b) == "states_one"


# ---------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------
def test_paired_difference_of_identical_runs_is_exactly_zero():
    scores = [0.2, 0.9, 1.0, 0.5]
    assert paired_mean_difference(scores, scores) == (0.0, 0.0, 0.0)


def test_paired_difference_recovers_a_constant_shift():
    """A constant per-question gap has no sampling variation, so the
    interval collapses onto it. That is the pairing doing its job."""
    b = [0.1, 0.4, 0.7, 0.3, 0.6]
    a = [x + 0.2 for x in b]
    diff, lo, hi = paired_mean_difference(a, b)
    assert diff == pytest.approx(0.2)
    assert lo == pytest.approx(0.2) and hi == pytest.approx(0.2)


def test_paired_difference_interval_brackets_the_estimate():
    a = [1.0, 0.0, 1.0, 1.0, 0.0, 1.0, 0.5, 1.0]
    b = [0.0, 0.0, 1.0, 0.5, 0.0, 0.0, 0.5, 1.0]
    diff, lo, hi = paired_mean_difference(a, b)
    assert lo <= diff <= hi
    assert lo < hi


def test_paired_difference_rejects_misaligned_runs():
    with pytest.raises(ValueError):
        paired_mean_difference([1.0, 0.5], [1.0])
    with pytest.raises(ValueError):
        paired_mean_difference([], [])


# ---------------------------------------------------------------------
# The results record
# ---------------------------------------------------------------------
def _report(**overrides) -> RunReport:
    base = dict(answerer="llm_cited", n=120, n_answered=77,
                context_precision=0.21111, context_recall=0.88222,
                faithfulness=0.93456, faithfulness_all=0.95,
                relevancy=0.9, correctness=float("nan"), judged=True,
                failure_modes={"ok": 70, "generation_failure": 7})
    base.update(overrides)
    return RunReport(**base)


def test_report_to_dict_rounds_and_maps_nan_to_null():
    d = report_to_dict(_report())
    assert d["faithfulness_answered"] == 0.9346
    assert d["correctness"] is None           # NaN is not valid JSON
    assert list(d["failure_modes"]) == ["generation_failure", "ok"]
    json.dumps(d, allow_nan=False)            # must not raise


def test_write_metrics_is_byte_stable(tmp_path):
    """Identical results must produce an identical file, so that a diff
    in the record always means a number moved."""
    payload = {"b": 1, "a": {"y": 2.5, "x": [1, 2]}}
    p1, p2 = tmp_path / "one.json", tmp_path / "nested" / "two.json"
    write_metrics(p1, payload)
    write_metrics(p2, {"a": {"x": [1, 2], "y": 2.5}, "b": 1})
    assert p1.read_bytes() == p2.read_bytes()
    assert p1.read_bytes().endswith(b"}\n")
    assert b"\r" not in p1.read_bytes()
    assert json.loads(p1.read_text(encoding="utf-8")) == payload
