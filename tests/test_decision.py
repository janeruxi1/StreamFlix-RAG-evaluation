"""Tests for the decision arithmetic.

Each function is small enough to check against a value worked by hand,
and that is what these do. The decision model is only as trustworthy as
its algebra, and a sign error here would produce a confident wrong
recommendation with nothing to flag it.
"""
import math

import pytest

from src.evaluation.decision import (
    breakeven_cost_ratio,
    clean_sweep_n,
    design_effect,
    expected_value,
    n_for_upper_bound,
    required_refusal_rate,
    two_proportion_n,
    wilson_interval,
)
from src.evaluation.judged_arms import wilson_lower_bound


# ---------------------------------------------------------------------
# Wilson interval
# ---------------------------------------------------------------------
def test_wilson_interval_matches_the_existing_lower_bound():
    for k, n in [(25, 25), (3, 76), (0, 10), (50, 100)]:
        assert wilson_interval(k, n)[0] == pytest.approx(wilson_lower_bound(k, n))


def test_wilson_interval_known_values():
    lo, hi = wilson_interval(3, 76)
    assert lo == pytest.approx(0.0135, abs=1e-4)
    assert hi == pytest.approx(0.1097, abs=1e-4)
    assert wilson_interval(25, 25)[1] == 1.0
    assert wilson_interval(0, 25)[0] == 0.0


# ---------------------------------------------------------------------
# Expected value
# ---------------------------------------------------------------------
def test_a_perfect_system_is_worth_its_answer_rate():
    """No bad answers, perfect refusal, free refusals: every answerable
    query answered is one ticket deflected."""
    assert expected_value(a=1.0, b=0.0, r=1.0, p=0.0, k=10, f=0.0) == 1.0
    assert expected_value(a=0.8, b=0.0, r=1.0, p=0.25, k=10, f=0.0) == pytest.approx(0.6)


def test_expected_value_worked_by_hand():
    # (1-0.1) * [0.8*(0.95 - 0.05*10) - 0.2*0.1]  -  0.1 * [0.1*10 + 0.9*0.1]
    # = 0.9 * [0.36 - 0.02] - 0.1 * [1.09] = 0.306 - 0.109
    assert expected_value(a=0.8, b=0.05, r=0.9, p=0.1, k=10, f=0.1) == pytest.approx(0.197)


def test_never_refusing_is_charged_for_every_unanswerable_query():
    assert expected_value(a=1.0, b=0.0, r=0.0, p=1.0, k=7, f=0.0) == -7.0


def test_value_rises_with_refusal_rate_and_falls_with_bad_answers():
    base = dict(a=0.8, b=0.04, r=0.9, p=0.1, k=10, f=0.1)
    assert expected_value(**{**base, "r": 0.99}) > expected_value(**base)
    assert expected_value(**{**base, "b": 0.10}) < expected_value(**base)


# ---------------------------------------------------------------------
# Break-even and required refusal
# ---------------------------------------------------------------------
def test_breakeven_cost_ratio_is_where_the_answerable_side_is_zero():
    a, b, f = 0.8, 0.04, 0.1
    k = breakeven_cost_ratio(a, b, f)
    assert a * ((1 - b) - b * k) - (1 - a) * f == pytest.approx(0.0, abs=1e-12)
    assert breakeven_cost_ratio(0.8, 0.0) == math.inf


def test_required_refusal_rate_sits_exactly_at_zero_value():
    a, b, p, k, f = 0.8, 0.04, 0.1, 10, 0.1
    r = required_refusal_rate(a, b, p, k, f)
    assert 0 < r < 1
    assert expected_value(a, b, r, p, k, f) == pytest.approx(0.0, abs=1e-12)
    assert expected_value(a, b, r + 0.01, p, k, f) > 0
    assert expected_value(a, b, r - 0.01, p, k, f) < 0


def test_required_refusal_rate_is_none_when_nothing_can_save_it():
    """Past the break-even cost ratio the answerable side is negative,
    and refusing unanswerable questions perfectly does not change that."""
    k = breakeven_cost_ratio(0.8, 0.04) * 2
    assert required_refusal_rate(0.8, 0.04, p=0.1, k=k) is None


def test_required_refusal_rate_is_zero_when_any_rate_pays():
    assert required_refusal_rate(a=0.9, b=0.0, p=0.01, k=1.0, f=0.0) == 0.0


# ---------------------------------------------------------------------
# Sample sizes
# ---------------------------------------------------------------------
def test_clean_sweep_n_reproduces_the_bound_it_targets():
    for target in (0.867, 0.95, 0.99):
        n = clean_sweep_n(target)
        assert wilson_lower_bound(n, n) >= target
        assert wilson_lower_bound(n - 1, n - 1) < target


def test_clean_sweep_n_for_twenty_five_questions():
    """25 of 25 gives 86.7%; asking for exactly that returns 25 or 26."""
    assert clean_sweep_n(0.8668) in (25, 26)
    assert clean_sweep_n(0.99) == 381


def test_n_for_upper_bound_reaches_the_target_and_not_before():
    n = n_for_upper_bound(0.04, 0.06)
    assert wilson_interval(round(0.04 * n), n)[1] <= 0.06
    assert n > 100
    assert n_for_upper_bound(0.04, 0.03) is None       # rate above target
    assert n_for_upper_bound(0.04, 0.04) is None


def test_two_proportion_n_matches_the_textbook_figure():
    # 70% vs 73%, alpha 0.05 two-sided, power 0.80
    assert two_proportion_n(0.70, 0.73) == pytest.approx(3550, abs=10)
    assert two_proportion_n(0.73, 0.70) == two_proportion_n(0.70, 0.73)
    with pytest.raises(ValueError):
        two_proportion_n(0.5, 0.5)


def test_smaller_effects_need_more_data():
    assert two_proportion_n(0.70, 0.72) > two_proportion_n(0.70, 0.75)


def test_design_effect():
    assert design_effect(1, 0.3) == 1.0                 # no clustering
    assert design_effect(50, 0.0) == 1.0                # no correlation
    assert design_effect(50, 0.05) == pytest.approx(3.45)
