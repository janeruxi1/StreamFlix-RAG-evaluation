"""Turning measured rates into a deployment decision.

Phases 1 to 7 end with rates: how often the system answers, how often an
answer is unsupported, how often it refuses a question it cannot answer.
A rate is not a decision. "It refused 25 of 25" does not say whether to
ship until someone states what a wrong answer costs relative to what a
right one is worth.

This module holds that arithmetic, kept small enough to check by hand:

    value per query =
        (1 - p) * [ a * ((1 - b) * 1  -  b * k)  -  (1 - a) * f ]
      -      p  * [ (1 - r) * k  +  r * f ]

    p   share of traffic the help centre cannot answer
    a   answer rate on answerable questions
    b   share of those answers that are unsupported ("bad")
    r   refusal rate on unanswerable questions
    k   cost of one bad answer, in units of one deflected ticket
    f   cost of a refusal (the customer waited, then reached a person)

Everything is measured against sending every ticket to a person, which
is zero. a, b and r are measured. p, k and f are ASSUMPTIONS about the
business, and the functions take them as arguments so that nothing here
can quietly bake one in.
"""
from __future__ import annotations

import math
from statistics import NormalDist


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for a proportion. Stays inside [0, 1] and is
    not zero-width at 0% or 100%, which is where these rates live."""
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)
    return (max(0.0, (centre - spread) / denom), min(1.0, (centre + spread) / denom))


def expected_value(a: float, b: float, r: float, p: float, k: float,
                   f: float = 0.1) -> float:
    """Value per query against a human-only baseline of zero."""
    in_scope = a * ((1 - b) - b * k) - (1 - a) * f
    out_of_scope = (1 - r) * k + r * f
    return (1 - p) * in_scope - p * out_of_scope


def breakeven_cost_ratio(a: float, b: float, f: float = 0.1) -> float:
    """The k at which answering ANSWERABLE questions stops paying.

    Above this, the system loses money before a single unanswerable
    question arrives, so no refusal rate can rescue it. It depends only
    on the bad-answer rate: roughly (1 - b) / b.
    """
    if b <= 0:
        return math.inf
    return ((1 - b) - (1 - a) * f / a) / b


def required_refusal_rate(a: float, b: float, p: float, k: float,
                          f: float = 0.1) -> float | None:
    """Smallest refusal rate at which the system is worth deploying.

    0.0  it pays at any refusal rate
    None it does not pay even with perfect refusal (the answerable side
         is already negative)
    """
    in_scope = (1 - p) * (a * ((1 - b) - b * k) - (1 - a) * f)
    if in_scope - p * f < 0:              # negative even at r = 1
        return None
    if p <= 0 or k <= f:
        return 0.0
    needed = (k - in_scope / p) / (k - f)
    return min(1.0, max(0.0, needed))


def clean_sweep_n(target: float, z: float = 1.96) -> int:
    """How many consecutive successes put the Wilson lower bound at
    `target`. For n of n, the bound is n / (n + z^2)."""
    if not 0 <= target < 1:
        raise ValueError("target must be in [0, 1)")
    return math.ceil(z * z * target / (1 - target))


def n_for_upper_bound(rate: float, target: float, z: float = 1.96,
                      max_n: int = 200_000) -> int | None:
    """Smallest sample at which an observed `rate` has a Wilson upper
    bound at or below `target`. None if the rate is not below the target,
    in which case no sample size helps."""
    if rate >= target:
        return None
    lo, hi = 1, max_n
    if wilson_interval(round(rate * hi), hi, z)[1] > target:
        return None
    while lo < hi:                         # the bound shrinks with n
        mid = (lo + hi) // 2
        if wilson_interval(round(rate * mid), mid, z)[1] <= target:
            hi = mid
        else:
            lo = mid + 1
    return lo


def two_proportion_n(p1: float, p2: float, alpha: float = 0.05,
                     power: float = 0.80) -> int:
    """Per-arm sample size to detect p1 vs p2, two-sided."""
    if p1 == p2:
        raise ValueError("the two proportions are equal: no effect to detect")
    z_a = NormalDist().inv_cdf(1 - alpha / 2)
    z_b = NormalDist().inv_cdf(power)
    variance = p1 * (1 - p1) + p2 * (1 - p2)
    return math.ceil((z_a + z_b) ** 2 * variance / (p1 - p2) ** 2)


def design_effect(cluster_size: float, icc: float) -> float:
    """Variance inflation from randomising clusters (agents) instead of
    tickets: 1 + (m - 1) * rho."""
    return 1 + (cluster_size - 1) * icc
