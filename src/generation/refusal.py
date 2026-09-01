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

    A partial refusal is one containing refusal language followed by a
    hedge word and substantially more text — the "I don't have specifics,
    but generally..." shape. It is reported separately and counted as an
    ANSWER, because unsupported claims were still made.
    """
    text = answer.strip()
    if not text:
        # An empty response is a failure to answer, but it is not a
        # refusal — nothing was communicated to the user.
        return RefusalCheck(False, False, None)

    match = next((p for p in _COMPILED if p.search(text)), None)
    if match is None:
        return RefusalCheck(False, False, None)

    # Where does the refusal language end, and is there substantive text
    # after it?
    m = match.search(text)
    tail = text[m.end():]
    is_partial = bool(
        _HEDGE_CONTINUATION.search(tail) and len(tail.split()) > 12
    )

    return RefusalCheck(
        is_refusal=not is_partial,
        is_partial=is_partial,
        matched_pattern=match.pattern,
    )


def is_refusal(answer: str) -> bool:
    """Convenience wrapper. Partial refusals count as answers."""
    return check_refusal(answer).is_refusal
