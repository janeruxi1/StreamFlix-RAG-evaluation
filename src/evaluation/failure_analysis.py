"""Phase 6 — where do failures come from, and what fixes each one?

A score says how often the system fails. It does not say what to change.
Failures here are sorted into three owners, because the fixes differ:

    retrieval   the evidence was never put in front of the generator
                (recall 0), or only part of it was (multi-hop)
    generation  the evidence was present and the answer was wrong:
                over-refusal, answering an unanswerable question, or
                (with a judge) an unfaithful answer
    corpus      the question touches a documented corpus flaw (a
                contradiction, a stale article, a near-duplicate), so the
                knowledge base itself is the root cause

Corpus exposure is a *tag*, not a cause: a question can fail through
retrieval and also involve a contradictory article, and collapsing the two
would hide the fix that actually applies.

Everything here works without a judge. Judged signals (faithfulness) sharpen
the generation bucket when available but are never required.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field

from src.corpus.seed_articles import KNOWN_CORPUS_FLAWS
from src.evaluation.rag_metrics import EvaluatedAnswer

OWNER = {
    "ok": "none",
    "correct_refusal": "none",
    "retrieval_miss": "retrieval",
    "retrieval_partial": "retrieval",
    "over_refusal": "generation",
    "answered_oos": "generation",
    "unfaithful": "generation",
}


@dataclass(frozen=True)
class FailureRecord:
    question_id: str
    category: str
    cause: str                         # key of OWNER
    flaw_tags: tuple[str, ...] = ()    # KNOWN_CORPUS_FLAWS ids touched

    @property
    def owner(self) -> str:
        return OWNER[self.cause]

    @property
    def is_failure(self) -> bool:
        return self.cause not in ("ok", "correct_refusal")


def _flaw_tags(ev: EvaluatedAnswer) -> tuple[str, ...]:
    """Flaws whose articles are in the ground truth of this question."""
    gt = set(ev.result.gt_article_ids)
    return tuple(f["flaw_id"] for f in KNOWN_CORPUS_FLAWS
                 if gt & set(f["article_ids"]))


def classify(ev: EvaluatedAnswer, faithful_threshold: float = 0.7) -> FailureRecord:
    """Assign one primary cause to one evaluated answer."""
    r = ev.result
    tags = _flaw_tags(ev)

    if r.is_out_of_scope:
        cause = "correct_refusal" if ev.is_refusal else "answered_oos"
    else:
        recall = ev.context_recall or 0.0
        if ev.is_refusal:
            cause = "over_refusal" if recall > 0 else "retrieval_miss"
        elif recall == 0:
            cause = "retrieval_miss"
        elif ev.faithfulness is not None and \
                ev.faithfulness.score < faithful_threshold:
            cause = "unfaithful"
        elif recall < 1.0:
            cause = "retrieval_partial"
        else:
            cause = "ok"
    return FailureRecord(r.question_id, r.category, cause, tags)


def classify_run(evaluated: list[EvaluatedAnswer]) -> list[FailureRecord]:
    return [classify(e) for e in evaluated]


@dataclass
class FailureSummary:
    n: int
    by_cause: Counter = field(default_factory=Counter)
    by_owner: Counter = field(default_factory=Counter)
    by_category: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    flaw_exposure: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))

    @property
    def n_failures(self) -> int:
        return sum(v for k, v in self.by_cause.items()
                   if k not in ("ok", "correct_refusal"))


def summarise(records: list[FailureRecord]) -> FailureSummary:
    s = FailureSummary(n=len(records))
    for rec in records:
        s.by_cause[rec.cause] += 1
        s.by_owner[rec.owner] += 1
        s.by_category[rec.category][rec.cause] += 1
        for tag in rec.flaw_tags:
            s.flaw_exposure[tag]["failed" if rec.is_failure else "passed"] += 1
    return s


def render_markdown(summary: FailureSummary, title: str) -> str:
    """Human-readable failure report, suitable for committing to reports/."""
    lines = [f"## {title}", "",
             f"{summary.n} questions, {summary.n_failures} failures "
             f"({summary.n_failures / summary.n:.0%}).", "",
             "| Cause | Owner | n |", "|---|---|---:|"]
    for cause, n in summary.by_cause.most_common():
        lines.append(f"| `{cause}` | {OWNER[cause]} | {n} |")
    lines += ["", "| Category | " + " | ".join(
        sorted({c for cs in summary.by_category.values() for c in cs})) + " |"]
    causes = sorted({c for cs in summary.by_category.values() for c in cs})
    lines.append("|---|" + "---:|" * len(causes))
    for cat, cs in sorted(summary.by_category.items()):
        lines.append(f"| {cat} | " + " | ".join(str(cs.get(c, 0)) for c in causes) + " |")
    if summary.flaw_exposure:
        lines += ["", "| Corpus flaw | failed | passed |", "|---|---:|---:|"]
        for flaw, c in sorted(summary.flaw_exposure.items()):
            lines.append(f"| `{flaw}` | {c.get('failed', 0)} | {c.get('passed', 0)} |")
    return "\n".join(lines) + "\n"
