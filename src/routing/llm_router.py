"""Zero-shot routing with an LLM: the intent list goes in the prompt.

The comparison this arm exists for: a general model with no training
data against small models trained on 9,000 labelled tickets. It goes
through the project's provider, so responses are cached on disk and a
re-run costs nothing, and the per-run call budget applies.

The LLM gives a label but no probability, so it cannot take part in the
confidence-threshold routing that the other arms are scored on. That is
reported as a limitation of the arm, not papered over with a verbalised
"confidence: 0.9" that nothing has calibrated.
"""
from __future__ import annotations

import difflib
import re
import time

PROMPT = """You route customer-support messages for a bank to the right team.

Choose exactly one intent from this list:
{intents}

Message: {text}

Reply with the intent name only, exactly as written in the list."""

# Published list price, US$ per million tokens, used only to turn token
# counts into a cost per 1,000 tickets. Check the provider's pricing page
# before quoting the result; the run records the values it used.
PRICE_PER_M = {"gpt-4o-mini": {"input": 0.15, "output": 0.60}}

UNPARSED = -1


def build_prompt(text: str, labels: list[str]) -> str:
    return PROMPT.format(intents="\n".join(f"- {lab}" for lab in labels),
                         text=text.strip())


def _norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def parse_label(response: str, labels: list[str]) -> int:
    """Map a model reply to a label id, or UNPARSED.

    Accepts the exact name, the name with quotes, bullets, a trailing
    full stop or spaces instead of underscores, and a reply that wraps
    the name in a short sentence. A near-miss spelling is accepted only
    when it is unambiguous (difflib ratio >= 0.9). Anything else is
    UNPARSED and scored as wrong: a router that answers off-list has
    sent the ticket nowhere.
    """
    index = {_norm(lab): i for i, lab in enumerate(labels)}
    first = response.strip().splitlines()[0] if response.strip() else ""
    cleaned = _norm(first.lstrip("-*• ").strip("`'\". "))
    if cleaned in index:
        return index[cleaned]

    # The name embedded in a sentence: take the longest label that appears.
    whole = "_" + _norm(response) + "_"
    found = [lab for lab in index if f"_{lab}_" in whole]
    if found:
        return index[max(found, key=len)]

    close = difflib.get_close_matches(cleaned, list(index), n=2, cutoff=0.9)
    if len(close) == 1:
        return index[close[0]]
    return UNPARSED


def approx_tokens(text: str) -> int:
    """Rough token count (4 characters per token) for the cost estimate."""
    return max(1, len(text) // 4)


def estimate_cost(texts: list[str], labels: list[str], model: str) -> float:
    """Upper-bound US$ for routing `texts` with no cache hits."""
    price = PRICE_PER_M.get(model)
    if price is None:
        return float("nan")
    tokens_in = sum(approx_tokens(build_prompt(t, labels)) for t in texts)
    tokens_out = 8 * len(texts)
    return (tokens_in * price["input"] + tokens_out * price["output"]) / 1e6


def route(provider, texts: list[str], labels: list[str], model: str | None = None,
          log=print) -> dict:
    """Route every text. Returns predictions, raw replies and timings.

    Latency is measured per call, so cached replies (which return in
    microseconds) are excluded from the latency figure; they are counted
    separately so the record says how many calls were live.
    """
    preds, replies, live_seconds = [], [], []
    for i, text in enumerate(texts, 1):
        started = time.perf_counter()
        reply = provider.complete(build_prompt(text, labels), model=model,
                                  max_tokens=20, temperature=0.0)
        took = time.perf_counter() - started
        if took > 0.05:             # a cache hit is far below this
            live_seconds.append(took)
        replies.append(reply)
        preds.append(parse_label(reply, labels))
        if i % 100 == 0:
            log(f"    {i}/{len(texts)} routed")
    return {"pred": preds, "replies": replies, "live_seconds": live_seconds}
