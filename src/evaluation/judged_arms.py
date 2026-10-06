"""Judging the LLM arms — the step Phase 5 was missing.

Phase 5 built the harness and exercised it on the extractive baseline.
That was the right subject for building an instrument: free,
deterministic, with failure modes known in advance. It was the wrong
subject for a deployment decision, because nobody proposes to ship the
baseline.

For a long time the notebook stopped there even when a credential was
present. Adding a key swapped the lexical judge for an LLM judge and
then pointed it at the same extractive answers. The memo meanwhile said
the generation layer was one judge run away from being measured. It was
not: the run it described would have paid a model to grade a copier and
left every LLM answer unjudged. Nothing failed, because nothing checks
that a harness is aimed at the thing the write-up says it is aimed at.

This module holds the pieces that close that gap, kept out of the
notebook so they can be tested:

  - which arms get judged, and what that costs BEFORE anything is spent
  - a mechanical read of the planted contradiction (mh-011)
  - a stable on-disk record of the judged results, so a memo can be
    verified against measured numbers in a CI job that has no key
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from src.evaluation.rag_metrics import RunReport
from src.generation.refusal import check_refusal

# The judge is the expensive tier, so "judge everything" is not the
# default. Two arms answer the deployment question:
#
#   cited   the shipping candidate — best refusal F1 in Phase 4
#   naive   the tutorial default — the contrast that shows what the
#           grounding, refusal and citation instructions actually buy
#
# Set JUDGE_ARMS=all (or a comma-separated list) to judge more.
DEFAULT_ARMS: tuple[str, ...] = ("naive", "cited")

# Planning figures for the cost estimate. Averages across the three
# judge prompts: faithfulness carries the full retrieved context,
# relevancy and correctness are short. Prices are list prices for the
# default judge model at the time of writing and WILL drift — the
# estimate exists so a run is never started blind, not as a quote.
JUDGE_INPUT_TOKENS_PER_CALL = 800
JUDGE_OUTPUT_TOKENS_PER_CALL = 90
USD_PER_M_INPUT = 2.50
USD_PER_M_OUTPUT = 10.00


def parse_arms(spec: str | None, available: Iterable[str]) -> list[str]:
    """Resolve a JUDGE_ARMS setting into prompt-variant names.

    Raises on an unknown name rather than skipping it. A typo that
    silently drops an arm produces a results table with a row missing,
    which reads as "that arm was not interesting" rather than "that arm
    never ran".
    """
    available = list(available)
    if spec is None or not spec.strip():
        return [a for a in DEFAULT_ARMS if a in available]
    if spec.strip().lower() == "all":
        return available

    names = [s.strip() for s in spec.split(",") if s.strip()]
    unknown = [n for n in names if n not in available]
    if unknown:
        raise ValueError(
            f"JUDGE_ARMS names unknown prompt variant(s) {unknown}. "
            f"Available: {', '.join(available)} (or 'all')."
        )
    seen: list[str] = []
    for n in names:
        if n not in seen:
            seen.append(n)
    return seen


def estimate_usd(n_calls: int) -> float:
    """Planning estimate for `n_calls` judge calls. See the caveat above."""
    return (n_calls * JUDGE_INPUT_TOKENS_PER_CALL / 1e6 * USD_PER_M_INPUT
            + n_calls * JUDGE_OUTPUT_TOKENS_PER_CALL / 1e6 * USD_PER_M_OUTPUT)


@dataclass(frozen=True)
class JudgePlan:
    """How many judge calls a run will make, known before it starts."""
    validation_calls: int
    calls_per_run: int          # one answerer over the whole golden set
    n_llm_arms: int

    @property
    def baseline_calls(self) -> int:
        return self.calls_per_run

    @property
    def arm_calls(self) -> int:
        return self.calls_per_run * self.n_llm_arms

    @property
    def total_calls(self) -> int:
        return self.validation_calls + self.baseline_calls + self.arm_calls

    @property
    def total_usd(self) -> float:
        return estimate_usd(self.total_calls)

    def fits(self, max_calls: int) -> bool:
        """Whether the run fits the per-process call ceiling.

        Checked up front. The ceiling raises mid-loop by design, which is
        the right behaviour for a runaway and the wrong way to discover
        that a planned run was never going to fit: the money for the
        first arms is already spent when it trips.
        """
        return max_calls <= 0 or self.total_calls <= max_calls


def plan_judging(n_questions: int, n_references: int, n_validation_cases: int,
                 n_llm_arms: int) -> JudgePlan:
    """Faithfulness and relevancy for every question, correctness where a
    reference answer exists, two calls per validation case."""
    return JudgePlan(
        validation_calls=n_validation_cases * 2,
        calls_per_run=n_questions * 2 + n_references,
        n_llm_arms=n_llm_arms,
    )


# ---------------------------------------------------------------------
# The planted contradiction
# ---------------------------------------------------------------------
def contradiction_stance(answer: str,
                         figures: tuple[str, str] = ("14", "30")) -> str:
    """What an answer does with two sources that disagree.

    mh-011 retrieves two articles stating different refund windows. This
    reads the answer mechanically:

      refused        declined to answer
      states_both    both figures appear
      states_one     exactly one appears — a figure asserted as though
                     the other source did not exist
      states_neither answered without committing to either

    Deliberately named for what is checked, not for what is hoped.
    `states_both` is NOT "flagged the conflict": an answer can list both
    windows without saying they disagree. Whether it does is a semantic
    question, which the judge's correctness score against the reference
    answer addresses. This check is the cheap one that needs no judge
    and cannot be talked round.
    """
    if check_refusal(answer).is_refusal:
        return "refused"
    # Citation ids are stripped first: "[trial-014]" is a source label,
    # not a statement about a 14-day window.
    text = re.sub(r"\[[a-z]+-\d+\]", " ", answer)
    present = [bool(re.search(rf"(?<!\d){re.escape(f)}(?!\d)", text))
               for f in figures]
    if all(present):
        return "states_both"
    if any(present):
        return "states_one"
    return "states_neither"


# ---------------------------------------------------------------------
# Comparing two arms
# ---------------------------------------------------------------------
def paired_mean_difference(a: list[float], b: list[float], n_boot: int = 2000,
                           alpha: float = 0.05,
                           seed: int = 42) -> tuple[float, float, float]:
    """Mean of (a - b) over paired items, with a bootstrap interval.

    Two arms answer the SAME questions, so the comparison is paired: the
    question-to-question variation is shared and cancels. Comparing two
    independent means would throw that away and report a wider interval
    than the data supports.

    Returns (difference, ci_low, ci_high), each rounded to six places.

    Rounded HERE, once, because these values are both printed by a
    notebook and written to a record that the memo quotes. A mean of
    simple fractions is often an exact tie in the third decimal (7.5/120
    is 0.0625), and floating point lands a hair to one side of it. Two
    consumers formatting the raw and the stored value then disagree in
    the last digit — "+0.063" in the notebook, "+0.062" in the memo —
    about the same measurement.
    """
    if len(a) != len(b):
        raise ValueError("paired comparison needs equal-length score lists")
    if not a:
        raise ValueError("nothing to compare")
    diffs = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    boot = diffs[idx].mean(axis=1)
    return (round(float(diffs.mean()), 6),
            round(float(np.percentile(boot, 100 * alpha / 2)), 6),
            round(float(np.percentile(boot, 100 * (1 - alpha / 2))), 6))


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    """Lower end of the Wilson score interval for a proportion.

    The number that turns "25 of 25" into a claim that can be defended.
    A perfect score on a small sample is not evidence of a perfect
    system: with n=25, an underlying failure rate of one in ten produces
    a clean sweep about 7% of the time. The Wilson bound stays inside
    [0, 1] and behaves at the extremes, where the usual normal
    approximation reports a zero-width interval at exactly 100%.
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if not 0 <= successes <= n:
        raise ValueError("successes must be between 0 and n")
    p = successes / n
    denom = 1 + z * z / n
    centre = p + z * z / (2 * n)
    spread = z * ((p * (1 - p) + z * z / (4 * n)) / n) ** 0.5
    return max(0.0, (centre - spread) / denom)


# ---------------------------------------------------------------------
# A record of the measured results
# ---------------------------------------------------------------------
def report_to_dict(report: RunReport) -> dict:
    """A RunReport as plain JSON types, rounded for a stable diff.

    Six decimals, not three or four: these values are quoted elsewhere to
    three places, and rounding twice can move the last digit (0.0395
    prints as 0.039 or 0.040 depending on which rounding came first).
    """
    def r(x: float) -> float | None:
        return None if x != x else round(float(x), 6)      # NaN -> null

    return {
        "answerer": report.answerer,
        "n": report.n,
        "n_answered": report.n_answered,
        "context_precision": r(report.context_precision),
        "context_recall": r(report.context_recall),
        "faithfulness_answered": r(report.faithfulness),
        "faithfulness_all": r(report.faithfulness_all),
        "relevancy": r(report.relevancy),
        "correctness": r(report.correctness),
        "failure_modes": dict(sorted(report.failure_modes.items())),
    }


def write_metrics(path: Path, payload: dict) -> None:
    """Write a results record deterministically.

    LLM-measured numbers cannot be recomputed in CI, which has no key by
    design. Committing the record lets the memo be verified against what
    was measured rather than against prose. Sorted keys and a fixed
    layout keep a re-run with identical results byte-identical, so a
    diff in this file always means a result changed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(payload, indent=2, sort_keys=True,
                           ensure_ascii=False) + "\n")
