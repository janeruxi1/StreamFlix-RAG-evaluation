"""Measuring what a judge responds to besides the answer's quality.

Phase 5 asked whether the judge gets known cases right. That is
necessary and not sufficient. A judge can score 100% on unambiguous
cases and still be unusable for comparing prompt variants, because it
responds to properties that have nothing to do with quality:

    length         longer answers rated higher regardless of content
    context order  the same evidence, reordered, scoring differently
    instability    the same input scored differently on a second call

Each of those corrupts a DIFFERENT downstream conclusion. Length bias
invalidates the Phase 4 prompt ladder specifically, because the ladder's
rungs produce systematically different answer lengths — `strict` asks
for evidence assessment and citations, so it writes more. If the judge
rewards length, `strict` wins the comparison for a reason unrelated to
being better.

The method: paired probes
-------------------------
Correlating score against length across the golden set does NOT measure
length bias. Longer answers may genuinely be more complete, so the
correlation confounds bias with quality — and the confound runs in the
direction that makes bias look real when it isn't.

Instead each probe is a PAIR that is identical in every respect except
the dimension under test. Same claims, same context, same ground truth;
only the phrasing length differs. Any score difference is attributable
to the manipulated variable, because nothing else varied. That is a
controlled experiment rather than an observational correlation, and it
is the reason these probes are hand-written rather than sampled.

The expectation is stated per probe. For length and context order the
correct behaviour is NO CHANGE, so the measurement is a deviation from
zero rather than a correlation coefficient.
"""
from __future__ import annotations

import random
from dataclasses import dataclass

import numpy as np

from src.evaluation.judge import Judge, Judgement

# Above this, two scores are treated as materially different. Set to the
# same value used for the trust gate in Phase 5, so "bias" and "wrong"
# are measured on one scale.
MATERIAL_DELTA = 0.15


# ---------------------------------------------------------------------
# Probe definitions
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class BiasProbe:
    """Two answers that should score identically, differing in one way."""
    probe_id: str
    dimension: str            # length | context_order | consistency
    question: str
    context: str
    answer_a: str
    answer_b: str
    note: str = ""


_CTX_PLANS = (
    "[strm-007] Simultaneous streams by plan\n"
    "Basic supports 1 simultaneous stream. Standard supports 2. "
    "Premium supports 4 simultaneous streams and includes 4K where "
    "available.\n\n"
    "[bill-001] Billing cycle\n"
    "StreamFlix bills on a monthly recurring cycle. Your billing date is "
    "the calendar day you first subscribed."
)

_CTX_REFUND = (
    "[bill-002] Refund policy\n"
    "Refunds are available within 30 days of a charge. Contact support "
    "with your account email and the charge date.\n\n"
    "[bill-003] Canceling your subscription\n"
    "You can cancel at any time from Account Settings. Access continues "
    "until the end of the paid period."
)


LENGTH_PROBES: list[BiasProbe] = [
    BiasProbe(
        probe_id="len-01", dimension="length",
        question="How many streams does Premium allow?",
        context=_CTX_PLANS,
        answer_a="Premium supports 4 simultaneous streams [strm-007].",
        answer_b=(
            "Thank you for asking about our Premium plan. According to the "
            "information available, the Premium plan supports 4 simultaneous "
            "streams [strm-007], which means up to four devices in your "
            "household can watch different titles at the same time. This is "
            "the highest number offered across our plans."
        ),
        note="Identical single claim, identical citation. Only length "
             "differs, and the long version adds no new facts.",
    ),
    BiasProbe(
        probe_id="len-02", dimension="length",
        question="When is my billing date?",
        context=_CTX_PLANS,
        answer_a="It is the calendar day you first subscribed [bill-001].",
        answer_b=(
            "Your billing date is determined by when you first signed up. "
            "Specifically, it is the calendar day you first subscribed "
            "[bill-001]. StreamFlix operates on a monthly recurring cycle "
            "[bill-001], so you can expect the charge to recur on that same "
            "calendar day each month going forward."
        ),
        note="The long version restates the same two facts at greater "
             "length. No additional claim is made.",
    ),
    BiasProbe(
        probe_id="len-03", dimension="length",
        question="How long do I have to request a refund?",
        context=_CTX_REFUND,
        answer_a="Within 30 days of the charge [bill-002].",
        answer_b=(
            "Refunds are available within 30 days of a charge [bill-002]. "
            "To request one, you should contact support and provide your "
            "account email along with the date of the charge in question "
            "[bill-002]. Acting promptly within that window is advisable."
        ),
        note="Second sentence is drawn from the context; the closing "
             "sentence is padding that asserts nothing checkable.",
    ),
]


def _shuffle_context_blocks(context: str, seed: int = 7) -> str:
    """Reorder the [id]-tagged blocks without changing any of them."""
    blocks = [b for b in context.split("\n\n") if b.strip()]
    rng = random.Random(seed)
    shuffled = blocks[:]
    while len(blocks) > 1 and shuffled == blocks:
        rng.shuffle(shuffled)
    return "\n\n".join(shuffled)


ORDER_PROBES: list[BiasProbe] = [
    BiasProbe(
        probe_id="ord-01", dimension="context_order",
        question="How many streams does Premium allow?",
        context=_CTX_PLANS,
        answer_a="Premium supports 4 simultaneous streams [strm-007].",
        answer_b="Premium supports 4 simultaneous streams [strm-007].",
        note="Same answer scored against the same context blocks in a "
             "different order. Evidence is identical; only position moves.",
    ),
    BiasProbe(
        probe_id="ord-02", dimension="context_order",
        question="How long do I have to request a refund?",
        context=_CTX_REFUND,
        answer_a="Refunds are available within 30 days [bill-002].",
        answer_b="Refunds are available within 30 days [bill-002].",
        note="If the score moves, the judge is reading position as "
             "relevance — and so, plausibly, is the generator.",
    ),
]


# ---------------------------------------------------------------------
# Measurement
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class ProbeResult:
    probe: BiasProbe
    score_a: float
    score_b: float

    @property
    def delta(self) -> float:
        """b - a. Positive means the manipulated version scored higher."""
        return self.score_b - self.score_a

    @property
    def is_material(self) -> bool:
        return abs(self.delta) >= MATERIAL_DELTA


@dataclass(frozen=True)
class BiasReport:
    """One dimension's worth of probe results."""
    judge: str
    dimension: str
    results: list[ProbeResult]

    @property
    def mean_delta(self) -> float:
        """Signed. The direction matters as much as the size."""
        return float(np.mean([r.delta for r in self.results]))

    @property
    def max_abs_delta(self) -> float:
        return float(max(abs(r.delta) for r in self.results))

    @property
    def n_material(self) -> int:
        return sum(r.is_material for r in self.results)

    @property
    def is_biased(self) -> bool:
        return self.n_material > 0

    @property
    def direction(self) -> str:
        if not self.is_biased:
            return "none detected"
        return "favours the longer/moved version" if self.mean_delta > 0 \
            else "penalises the longer/moved version"

    def __str__(self) -> str:
        return (f"{self.dimension}: mean delta {self.mean_delta:+.3f}, "
                f"max |delta| {self.max_abs_delta:.3f}, "
                f"{self.n_material}/{len(self.results)} material")


def measure_length_bias(judge: Judge,
                        probes: list[BiasProbe] | None = None) -> BiasReport:
    """Score terse vs verbose versions of identical content.

    Both members of each pair make the same claims from the same context,
    so a faithful judge must score them equally. Deviation from zero is
    the bias.
    """
    probes = probes or LENGTH_PROBES
    results = [
        ProbeResult(
            probe=p,
            score_a=judge.faithfulness(p.answer_a, p.context).score,
            score_b=judge.faithfulness(p.answer_b, p.context).score,
        )
        for p in probes
    ]
    return BiasReport(judge.name, "length", results)


def measure_order_sensitivity(judge: Judge,
                              probes: list[BiasProbe] | None = None) -> BiasReport:
    """Score the same answer against reordered context blocks.

    The evidence is byte-identical; only its position changes. Any score
    movement means the judge treats position as information, which makes
    every cross-configuration comparison partly a comparison of retrieval
    ordering rather than answer quality.
    """
    probes = probes or ORDER_PROBES
    results = [
        ProbeResult(
            probe=p,
            score_a=judge.faithfulness(p.answer_a, p.context).score,
            score_b=judge.faithfulness(
                p.answer_b, _shuffle_context_blocks(p.context)).score,
        )
        for p in probes
    ]
    return BiasReport(judge.name, "context_order", results)


def measure_self_consistency(judge: Judge, n_repeats: int = 3,
                             probes: list[BiasProbe] | None = None) -> BiasReport:
    """Score the identical input several times.

    At temperature 0 this should be exactly reproducible. It often is not
    for hosted models, and the size of the wobble sets a floor on what
    counts as a real difference between configurations: a gap smaller
    than the judge's own run-to-run variation is not measurable at all.
    """
    probes = probes or (LENGTH_PROBES + ORDER_PROBES)
    results = []
    for p in probes:
        scores = [judge.faithfulness(p.answer_a, p.context).score
                  for _ in range(n_repeats)]
        results.append(ProbeResult(probe=p, score_a=min(scores),
                                   score_b=max(scores)))
    return BiasReport(judge.name, "consistency", results)


# ---------------------------------------------------------------------
# Inter-judge agreement
# ---------------------------------------------------------------------
def cohens_kappa(labels_a: list[bool], labels_b: list[bool]) -> float:
    """Agreement between two raters, corrected for chance.

    Raw agreement is misleading when one verdict dominates: two judges
    that both call 90% of answers faithful agree 82% of the time by
    coincidence alone. Kappa subtracts that expected agreement, so 0
    means "no better than chance" rather than "never agrees".

    Returns 1.0 for perfect agreement, 0.0 for chance-level, negative for
    systematic disagreement.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("rater label lists must be the same length")
    n = len(labels_a)
    if n == 0:
        return float("nan")

    observed = sum(a == b for a, b in zip(labels_a, labels_b)) / n
    p_a = sum(labels_a) / n
    p_b = sum(labels_b) / n
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)

    if expected == 1.0:            # both raters constant and identical
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)


@dataclass(frozen=True)
class AgreementReport:
    judge_a: str
    judge_b: str
    n: int
    raw_agreement: float
    kappa: float
    disagreements: list[tuple[str, float, float]]

    @property
    def interpretation(self) -> str:
        """Landis & Koch benchmarks, stated as the convention they are."""
        k = self.kappa
        if k != k:                       # nan
            return "undefined"
        if k < 0:
            return "worse than chance"
        if k < 0.20:
            return "slight"
        if k < 0.40:
            return "fair"
        if k < 0.60:
            return "moderate"
        if k < 0.80:
            return "substantial"
        return "almost perfect"


def judge_agreement(judge_a: Judge, judge_b: Judge,
                    items: list[tuple[str, str, str]],
                    threshold: float = 0.7) -> AgreementReport:
    """Compare two judges on the same (id, answer, context) triples.

    Scores are binarised at `threshold` before comparing, because kappa
    is defined over categories. That discards information deliberately:
    the decision the harness actually makes is "faithful or not", so
    agreement on that call is what matters, and two judges differing by
    0.05 either side of the line is a real disagreement while two
    differing by 0.4 within the same category is not.
    """
    labels_a, labels_b, disagreements = [], [], []
    for item_id, answer, context in items:
        sa = judge_a.faithfulness(answer, context).score
        sb = judge_b.faithfulness(answer, context).score
        la, lb = sa >= threshold, sb >= threshold
        labels_a.append(la)
        labels_b.append(lb)
        if la != lb:
            disagreements.append((item_id, sa, sb))

    return AgreementReport(
        judge_a=judge_a.name, judge_b=judge_b.name, n=len(items),
        raw_agreement=(sum(a == b for a, b in zip(labels_a, labels_b))
                       / len(items) if items else float("nan")),
        kappa=cohens_kappa(labels_a, labels_b),
        disagreements=disagreements,
    )
