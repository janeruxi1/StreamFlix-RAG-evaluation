"""Does the judge actually work? A labelled suite that answers it.

An LLM judge is a measuring instrument, and an uncalibrated instrument
produces numbers, not measurements. The usual practice — report
faithfulness to three decimals from a model whose agreement with ground
truth was never checked — hides an unknown error rate inside the
headline result.

So before any judge score is believed, the judge is run against cases
where the correct verdict is known by construction. Each case is built so
the right answer follows from how it was written, not from an opinion:

    supported     every claim appears verbatim in the context
    fabricated    a specific fact is present that the context never states
    contradicted  a fact directly opposes what the context says
    refusal       no claims are made at all
    off_topic     fluent, drawn from the context, answers a different
                  question

The cases are deliberately short and unambiguous. A judge that cannot
separate these has no chance on real answers, so this is a floor test
rather than a hard one — failing it is disqualifying, passing it is not
proof of much.

The most diagnostic case is `contradicted`. A contradicting sentence
reuses the context's vocabulary almost perfectly, so any overlap-based
method scores it as highly faithful. It is the case that separates
semantic judgement from term matching, and the reason a lexical judge
cannot substitute for a model here.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from src.evaluation.judge import Judge, Judgement


@dataclass(frozen=True)
class ValidationCase:
    """One case where the correct verdict is known by construction."""
    case_id: str
    kind: str                     # supported | fabricated | contradicted | ...
    question: str
    context: str
    answer: str
    reference: str
    expect_faithful_high: bool    # should faithfulness be high?
    expect_relevant_high: bool    # should relevancy be high?
    note: str = ""


_CTX_BILLING = (
    "[bill-001] Billing cycle\n"
    "StreamFlix bills on a monthly recurring cycle. Your billing date is "
    "the calendar day you first subscribed. Payment is taken in advance "
    "for the coming month. Months with fewer days bill on the final day."
)

_CTX_PLANS = (
    "[strm-007] Simultaneous streams by plan\n"
    "Basic supports 1 simultaneous stream. Standard supports 2. "
    "Premium supports 4 simultaneous streams and includes 4K where "
    "available."
)

_CTX_REFUND = (
    "[bill-002] Refund policy\n"
    "Refunds are available within 30 days of a charge. Contact support "
    "with your account email and the charge date."
)


VALIDATION_CASES: list[ValidationCase] = [
    ValidationCase(
        case_id="val-supported-1", kind="supported",
        question="When is my billing date?",
        context=_CTX_BILLING,
        answer="Your billing date is the calendar day you first subscribed, "
               "and payment is taken in advance for the coming month.",
        reference="It is the calendar day you first subscribed; payment is "
                  "taken in advance.",
        expect_faithful_high=True, expect_relevant_high=True,
        note="Every claim is stated in the context.",
    ),
    ValidationCase(
        case_id="val-supported-2", kind="supported",
        question="How many streams does Premium allow?",
        context=_CTX_PLANS,
        answer="Premium supports 4 simultaneous streams.",
        reference="Four simultaneous streams.",
        expect_faithful_high=True, expect_relevant_high=True,
    ),
    ValidationCase(
        case_id="val-fabricated-1", kind="fabricated",
        question="How many streams does Premium allow?",
        context=_CTX_PLANS,
        answer="Premium supports 4 simultaneous streams and costs $19.99 "
               "per month.",
        reference="Four simultaneous streams.",
        expect_faithful_high=False, expect_relevant_high=True,
        note="The price is plausible and entirely absent from the context. "
             "Half the claims are supported, so faithfulness should drop "
             "but not to zero.",
    ),
    ValidationCase(
        case_id="val-fabricated-2", kind="fabricated",
        question="When is my billing date?",
        context=_CTX_BILLING,
        answer="Your billing date is the day you subscribed, and you can "
               "change it once per year from Account Settings.",
        reference="It is the calendar day you first subscribed.",
        expect_faithful_high=False, expect_relevant_high=True,
        note="Changing the billing date is never mentioned.",
    ),
    ValidationCase(
        case_id="val-contradicted-1", kind="contradicted",
        question="How many streams does Premium allow?",
        context=_CTX_PLANS,
        answer="Premium supports 2 simultaneous streams.",
        reference="Four simultaneous streams.",
        expect_faithful_high=False, expect_relevant_high=True,
        note="THE DIAGNOSTIC CASE. Near-perfect term overlap with the "
             "context, opposite meaning. Any overlap-based judge scores "
             "this as faithful.",
    ),
    ValidationCase(
        case_id="val-contradicted-2", kind="contradicted",
        question="How long do I have to request a refund?",
        context=_CTX_REFUND,
        answer="Refunds are available within 14 days of a charge. Contact "
               "support with your account email and the charge date.",
        reference="Within 30 days of the charge.",
        expect_faithful_high=False, expect_relevant_high=True,
        note="One number changed; every other word is lifted from the "
             "context verbatim.",
    ),
    ValidationCase(
        case_id="val-refusal-1", kind="refusal",
        question="Do you offer student discounts?",
        context=_CTX_PLANS,
        answer="I don't have enough information to answer that.",
        reference="The corpus does not cover student discounts.",
        expect_faithful_high=True, expect_relevant_high=True,
        note="Vacuously faithful — no claims were made. A judge that "
             "scores this low would punish correct refusals on the very "
             "metric meant to reward not inventing things.",
    ),
    ValidationCase(
        case_id="val-offtopic-1", kind="off_topic",
        question="How many streams does Premium allow?",
        context=_CTX_PLANS,
        answer="Basic supports 1 simultaneous stream.",
        reference="Four simultaneous streams.",
        expect_faithful_high=True, expect_relevant_high=False,
        note="Perfectly faithful and does not answer the question. "
             "Separates faithfulness from relevancy — a judge conflating "
             "them fails here.",
    ),
    ValidationCase(
        case_id="val-offtopic-2", kind="off_topic",
        question="When is my billing date?",
        context=_CTX_BILLING,
        answer="Months with fewer days bill on the final day of the month.",
        reference="It is the calendar day you first subscribed.",
        expect_faithful_high=True, expect_relevant_high=False,
        note="Lifted verbatim from the context, answers a different "
             "question.",
    ),
]


# ---------------------------------------------------------------------
# Second batch — built from corpus articles, one case per kind per topic.
#
# Nine hand-written cases were a floor test whose 80% gate meant 7 versus
# 8 correct: one flipped verdict moved the result across the line. Each
# topic below yields all five kinds from one context, so the suite grows
# without hand-authoring forty unrelated scenarios, and each kind has
# enough cases that per-kind accuracy is not a single coin flip.
#
# (context, question, supported answer, fabricated answer,
#  contradicted answer, off-topic answer, reference)
# ---------------------------------------------------------------------
_TOPICS = [
    ("[bill-004] Failed or declined payments\n"
     "We attempt the charge up to four times over eight days: on the "
     "billing date, then at days 3, 5, and 8. Your account remains active "
     "during this window. Access is suspended on day 9. Your profiles, "
     "viewing history, and My List are preserved.",
     "What happens when my payment fails?",
     "StreamFlix retries up to four times over eight days, and your "
     "account stays active meanwhile. Access is suspended on day 9.",
     "StreamFlix retries four times over eight days, and you will receive "
     "a $5 late fee if all attempts fail.",
     "StreamFlix retries once, and your account is suspended immediately.",
     "Your profiles, viewing history, and My List are preserved.",
     "Four retries over eight days; access suspended on day 9."),
    ("[acct-003] Managing profiles\n"
     "Every StreamFlix account supports up to 5 profiles, on all plans. "
     "Set a 4-digit PIN on any profile in Manage profiles. Deleting a "
     "profile permanently removes its viewing history. The primary "
     "account profile cannot be deleted.",
     "How many profiles can I have?",
     "An account supports up to 5 profiles, on all plans.",
     "An account supports up to 5 profiles, and each one can use a "
     "different payment method.",
     "An account supports up to 2 profiles, on Premium only.",
     "You can set a 4-digit PIN on any profile.",
     "Up to 5 profiles on every plan."),
    ("[dev-001] Supported devices\n"
     "Supported mobile systems are iOS 15 and later and Android 9 and "
     "later. Not supported: Windows Phone, devices running Android below "
     "9, smart TVs from before 2018, and rooted or jailbroken devices.",
     "Does StreamFlix work on an old Android phone?",
     "Android 9 and later is supported; devices running Android below 9 "
     "are not.",
     "Android 9 and later is supported, and older phones can use a "
     "legacy app from the website.",
     "Android 5 and later is supported, including rooted devices.",
     "Smart TVs from before 2018 are not supported.",
     "Only Android 9 or later."),
    ("[strm-002] Buffering and playback problems\n"
     "Try these in order: restart the app, restart your device, restart "
     "your router and modem waiting 30 seconds before powering on, move "
     "closer to the router or use a wired connection. Lower playback "
     "quality in Account > Playback settings.",
     "What should I do if a show keeps buffering?",
     "Restart the app, then your device, then your router, and consider "
     "lowering playback quality in Playback settings.",
     "Restart the app and your device, and call StreamFlix support, who "
     "will reset your streaming server.",
     "Do not restart anything; buffering is always caused by the "
     "StreamFlix service.",
     "Move closer to the router or use a wired connection.",
     "Restart app, device, router; lower quality."),
    ("[trial-001] How the 14-day free trial works\n"
     "New customers get a 14-day free trial. A payment method is required "
     "to start. You are charged the plan price when the trial ends unless "
     "you cancel before then. Each household can use one trial.",
     "How long is the free trial?",
     "The free trial lasts 14 days, and a payment method is required to "
     "start it.",
     "The free trial lasts 14 days, and you can extend it once by "
     "contacting support.",
     "The free trial lasts 30 days and requires no payment method.",
     "Each household can use one trial.",
     "14 days."),
    ("[bill-002] Refund policy\n"
     "Refunds are available within 30 days of a charge. Contact support "
     "with your account email and the charge date. Refunds return to the "
     "original payment method.",
     "Where does a refund go?",
     "Refunds return to the original payment method.",
     "Refunds return to the original payment method within 24 hours, "
     "guaranteed.",
     "Refunds are issued as StreamFlix credit only.",
     "Contact support with your account email and the charge date.",
     "To the original payment method."),
]


def _batch_two() -> list[ValidationCase]:
    cases = []
    for i, (ctx, q, ok, fab, contra, off, ref) in enumerate(_TOPICS, 1):
        cases += [
            ValidationCase(f"val2-supported-{i}", "supported", q, ctx, ok,
                           ref, True, True),
            ValidationCase(f"val2-fabricated-{i}", "fabricated", q, ctx, fab,
                           ref, False, True,
                           "Plausible extra fact absent from the context."),
            ValidationCase(f"val2-contradicted-{i}", "contradicted", q, ctx,
                           contra, ref, False, True,
                           "States the opposite of the context."),
            ValidationCase(f"val2-offtopic-{i}", "off_topic", q, ctx, off,
                           ref, True, False,
                           "Faithful to context, answers another question."),
            ValidationCase(f"val2-refusal-{i}", "refusal", q, ctx,
                           "I don't have enough information to answer "
                           "that.", ref, True, True,
                           "Vacuously faithful: no claims made."),
        ]
    return cases


VALIDATION_CASES += _batch_two()

HIGH_THRESHOLD = 0.7      # what counts as "high" for a 0-1 judgement
LOW_THRESHOLD = 0.5       # what counts as "low"


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95% Wilson score interval for a proportion.

    Wilson rather than the normal approximation: accuracy here sits near
    1.0 on a small n, exactly where the normal interval is wrong (it can
    exceed 1 and has poor coverage).
    """
    if n == 0:
        return (0.0, 1.0)
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


@dataclass(frozen=True)
class CaseResult:
    case: ValidationCase
    faithfulness: Judgement
    relevancy: Judgement

    @property
    def faithfulness_correct(self) -> bool:
        if self.case.expect_faithful_high:
            return self.faithfulness.score >= HIGH_THRESHOLD
        return self.faithfulness.score < LOW_THRESHOLD

    @property
    def relevancy_correct(self) -> bool:
        if self.case.expect_relevant_high:
            return self.relevancy.score >= HIGH_THRESHOLD
        return self.relevancy.score < LOW_THRESHOLD


@dataclass(frozen=True)
class ValidationReport:
    """How well a judge agrees with known-correct verdicts."""
    judge: str
    results: list[CaseResult]

    @property
    def faithfulness_accuracy(self) -> float:
        return sum(r.faithfulness_correct for r in self.results) / len(self.results)

    @property
    def relevancy_accuracy(self) -> float:
        return sum(r.relevancy_correct for r in self.results) / len(self.results)

    @property
    def overall_accuracy(self) -> float:
        n = 2 * len(self.results)
        correct = sum(r.faithfulness_correct + r.relevancy_correct
                      for r in self.results)
        return correct / n

    @property
    def overall_ci(self) -> tuple[float, float]:
        """95% Wilson interval on overall accuracy (judgements, n = 2/case)."""
        n = 2 * len(self.results)
        correct = sum(r.faithfulness_correct + r.relevancy_correct
                      for r in self.results)
        return wilson_interval(correct, n)

    @property
    def parse_failures(self) -> int:
        return sum(not r.faithfulness.parsed_ok or not r.relevancy.parsed_ok
                   for r in self.results)

    def accuracy_by_kind(self) -> dict[str, float]:
        kinds = sorted({r.case.kind for r in self.results})
        out = {}
        for kind in kinds:
            subset = [r for r in self.results if r.case.kind == kind]
            correct = sum(r.faithfulness_correct + r.relevancy_correct
                          for r in subset)
            out[kind] = correct / (2 * len(subset))
        return out

    def failures(self) -> list[CaseResult]:
        return [r for r in self.results
                if not (r.faithfulness_correct and r.relevancy_correct)]

    @property
    def is_trustworthy(self) -> bool:
        """Whether downstream scores from this judge mean anything.

        The bar is deliberately blunt: a judge scoring below 80% on cases
        this unambiguous is not measuring what it claims to, and its
        numbers on real answers should not be reported as evidence.

        The point estimate alone is not enough. On a small suite a lucky
        run can clear 80% while the plausible true accuracy is far lower,
        so the lower end of the 95% interval must also stay above 70%.
        """
        return (self.overall_accuracy >= 0.8
                and self.overall_ci[0] >= 0.7
                and self.parse_failures == 0)


def validate_judge(judge: Judge,
                   cases: list[ValidationCase] | None = None) -> ValidationReport:
    """Run a judge against the labelled suite."""
    cases = cases or VALIDATION_CASES
    results = [
        CaseResult(
            case=c,
            faithfulness=judge.faithfulness(c.answer, c.context),
            relevancy=judge.relevancy(c.answer, c.question),
        )
        for c in cases
    ]
    return ValidationReport(judge=judge.name, results=results)
