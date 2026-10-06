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

HIGH_THRESHOLD = 0.7      # what counts as "high" for a 0-1 judgement
LOW_THRESHOLD = 0.5       # what counts as "low" — INCLUSIVE, see below

# Why "low" is `<=` and not `<`.
#
# Faithfulness is a claim ratio: supported claims over total claims.
# Three of the four cases that should score low are two-claim answers
# with exactly one bad claim, so a judge that reads them CORRECTLY
# returns 1/2 = 0.5 — the note on val-fabricated-1 even says so.
#
# With a strict `<`, that correct verdict sat exactly on the boundary
# and was marked wrong. A flawless claim-counting judge would have
# scored 15/18 = 83% on this suite, three points above the trust
# threshold, and one unrelated slip would have failed the gate and
# reported a sound judge as untrustworthy. The suite's own "perfect
# judge passes" test could not see it, because its oracle answered 1.0
# or 0.0 and never produced the value a real judge produces.
#
# The gate had never been run against an LLM judge, so nothing had
# tripped it yet. An instrument that has only ever measured the thing it
# was built to fail has not been calibrated.


@dataclass(frozen=True)
class CaseResult:
    case: ValidationCase
    faithfulness: Judgement
    relevancy: Judgement

    @property
    def faithfulness_correct(self) -> bool:
        if self.case.expect_faithful_high:
            return self.faithfulness.score >= HIGH_THRESHOLD
        return self.faithfulness.score <= LOW_THRESHOLD

    @property
    def relevancy_correct(self) -> bool:
        if self.case.expect_relevant_high:
            return self.relevancy.score >= HIGH_THRESHOLD
        return self.relevancy.score <= LOW_THRESHOLD


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
        """
        return self.overall_accuracy >= 0.8 and self.parse_failures == 0


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
