"""Detecting whether an answer is a refusal.

This looks trivial and is not. The prompts ask for an exact refusal
string, but models paraphrase, hedge, and produce partial refusals
("I don't have specific details, but generally..."), which is the most
dangerous shape of all: it reads as a refusal while still hallucinating.

Detection is deliberately pattern-based rather than LLM-based:

  - It is free and instant, so it can run on every answer.
  - It is deterministic, so the same answer always scores the same way.
  - It is auditable — anyone can read the patterns and disagree with a
    specific one, which is not true of a judge model's opinion.

The cost is that it can be fooled by unusual phrasings. Phase 6 uses an
LLM judge for the cases where that matters, but paying a judge to detect
"I don't know" would be spending money to do worse than a regex.

Partial refusals are treated as NON-refusals on purpose. A response that
disclaims and then answers anyway has still made unsupported claims, and
scoring it as a refusal would hide exactly the failure worth catching.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Canonical string the prompts request verbatim.
CANONICAL_REFUSAL = "I don't have enough information to answer that."

# Phrasings that indicate the model declined to answer. Ordered roughly
# by how unambiguous they are.
REFUSAL_PATTERNS = [
    r"i don'?t have enough information",
    r"i do not have enough information",
    r"not enough information (?:to|in the)",
    r"insufficient information",
    r"the context does not (?:contain|provide|include|mention)",
    r"the context doesn'?t (?:contain|provide|include|mention)",
    r"(?:is |are )?not (?:covered|mentioned|available|addressed) in the (?:context|provided)",
    r"i don'?t know",
    r"i do not know",
    r"i'?m (?:not able|unable) to answer",
    r"cannot (?:be )?answer(?:ed)? (?:from|using|based on) the (?:context|information|provided)",
    r"no information (?:about|on|regarding) .{0,40} in the (?:context|provided)",
    r"unable to (?:find|locate) .{0,40} in the (?:context|provided)",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in REFUSAL_PATTERNS]

# Signals that the model refused and then answered anyway.
_HEDGE_CONTINUATION = re.compile(
    r"\b(?:but|however|though|generally|typically|usually|in general)\b",
    re.IGNORECASE,
)

# A grounded claim. Imported rather than duplicated so the two modules
# cannot drift — a mismatch here would silently change what counts as a
# refusal. (prompts does not import refusal, so there is no cycle.)
from src.generation.prompts import CITATION_PATTERN  # noqa: E402

# How much text before the refusal phrase counts as "it answered first".
# Short preambles like "Based on the context," are not answers; a couple
# of clauses of substance are.
_LEAD_WORDS_THRESHOLD = 8

# How much hedged text after the refusal phrase counts as answering anyway.
_TAIL_WORDS_THRESHOLD = 12


@dataclass(frozen=True)
class RefusalCheck:
    """Verdict on one answer."""
    is_refusal: bool
    is_partial: bool          # refused, then answered anyway
    matched_pattern: str | None

    @property
    def label(self) -> str:
        if self.is_partial:
            return "partial_refusal"
        return "refusal" if self.is_refusal else "answer"


def check_refusal(answer: str) -> RefusalCheck:
    """Classify an answer as refusal, partial refusal, or answer.

    A partial refusal is a response that contains refusal language but
    still makes substantive claims. It is reported separately and counted
    as an ANSWER, because unsupported claims were made either way.

    It comes in two shapes, and BOTH must be caught:

        refuse-then-answer   "I don't have enough information, but
                              generally most services..."
        answer-then-refuse   "Premium costs $19.99 [bill-001]. I don't
                              have enough information about student
                              discounts."

    An earlier version only inspected the text AFTER the refusal phrase,
    so it caught the first shape and scored the second as a clean
    refusal. That error runs in the dangerous direction: a model that
    answers an out-of-scope question and appends a hedge would be counted
    as having correctly refused, so the safety metric would report the
    opposite of what happened.

    Evidence that the model answered rather than refused:
      - substantive text BEFORE the refusal phrase
      - a citation before it, which is a grounded claim by definition
      - a hedge plus substantial text AFTER it
    """
    text = answer.strip()
    if not text:
        # An empty response is a failure to answer, but it is not a
        # refusal — nothing was communicated to the user.
        return RefusalCheck(False, False, None)

    pattern = next((p for p in _COMPILED if p.search(text)), None)
    if pattern is None:
        return RefusalCheck(False, False, None)

    match = pattern.search(text)
    lead, tail = text[:match.start()], text[match.end():]

    answered_first = (
        len(lead.split()) >= _LEAD_WORDS_THRESHOLD
        or bool(CITATION_PATTERN.search(lead))
    )
    answered_after = bool(
        _HEDGE_CONTINUATION.search(tail)
        and len(tail.split()) > _TAIL_WORDS_THRESHOLD
    )
    is_partial = answered_first or answered_after

    return RefusalCheck(
        is_refusal=not is_partial,
        is_partial=is_partial,
        matched_pattern=pattern.pattern,
    )


def is_refusal(answer: str) -> bool:
    """Convenience wrapper. Partial refusals count as answers."""
    return check_refusal(answer).is_refusal
