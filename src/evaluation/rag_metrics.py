"""End-to-end RAG evaluation: context quality and answer quality.

Four metrics, in the RAGAS tradition but with one deliberate departure.

    context_precision   of the retrieved context, how much was relevant
    context_recall      of the needed sources, how much was retrieved
    faithfulness        are the answer's claims supported by the context
    answer_relevancy    does the answer address the question

RAGAS computes ALL FOUR with an LLM, because it assumes you have no
ground-truth labels — which is the normal situation and a reasonable
default. This project has exact labels: every golden question names its
source articles.

So the first two are computed EXACTLY here, not judged. Using an LLM to
estimate a quantity you can compute is strictly worse on every axis: it
is noisier, it costs money per question, it is not reproducible across
model versions, and it introduces the judge's error into a number that
had none. The LLM is reserved for the two metrics that genuinely require
reading comprehension.

That split also makes the harness partially runnable with no credential
at all, and it means a judge outage degrades the evaluation rather than
stopping it.

The joint reading that matters
------------------------------
Faithfulness and context_recall have to be read together, and this is the
Phase 3/4 thread arriving at its conclusion:

    high recall + low faithfulness   the evidence was there and the model
                                     ignored it -> GENERATION failure
    low recall + low faithfulness    the evidence was never retrieved and
                                     the model answered anyway -> a
                                     RETRIEVAL failure that will be
                                     misread as hallucination
    low recall + high faithfulness   the model stayed faithful to wrong
                                     context, or correctly refused

Only the first is fixed by touching the generator. Reporting a single
"hallucination rate" collapses all three and points the fix in an
arbitrary direction.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from src.evaluation.judge import Judge, Judgement
from src.generation.pipeline import RAGResult
from src.generation.prompts import format_context


# ---------------------------------------------------------------------
# Context metrics — exact, no judge
# ---------------------------------------------------------------------
def context_precision(result: RAGResult) -> float | None:
    """Fraction of retrieved CHUNKS that came from a ground-truth article.

    Chunk-level rather than article-level on purpose. The generator reads
    chunks, so the proportion of its context window that is noise is what
    actually dilutes its attention — and a strategy producing many small
    chunks can score well on article recall while filling the context
    with irrelevant fragments.

    None for out-of-scope questions: with no ground truth, no retrieved
    chunk can be "relevant", and scoring them 0.0 would penalise the
    retriever for a question that has no right answer.
    """
    if not result.gt_article_ids:
        return None
    if not result.hits:
        return 0.0
    gt = set(result.gt_article_ids)
    return sum(h.chunk.article_id in gt for h in result.hits) / len(result.hits)


def context_recall(result: RAGResult) -> float | None:
    """Fraction of ground-truth articles present in the retrieved context.

    Identical to the retrieval recall from Phase 3, restated here so the
    evaluation report is self-contained.
    """
    return result.retrieval_recall


def context_precision_ceiling(result: RAGResult,
                              chunks_per_article: dict[str, int]) -> float | None:
    """The highest context_precision this question could possibly score.

    Reporting raw context_precision without this is misleading, and the
    reason is structural rather than a modelling choice.

    A question with one ground-truth article, retrieved at depth 15, on a
    strategy that splits that article into 4 chunks, can reach at most
    4/15 = 0.27 — because only 4 relevant chunks EXIST. The remaining 11
    slots must be filled with something irrelevant. A score of 0.21
    against a ceiling of 0.27 is a retriever doing well; read against an
    implicit ceiling of 1.0 it looks broken.

    The ceiling depends on retrieval depth and chunking granularity, both
    of which Phase 3 chose on other grounds. So this is the cost of those
    choices made visible, not a defect to fix.
    """
    if not result.gt_article_ids or not result.hits:
        return None
    available = sum(chunks_per_article.get(a, 0)
                    for a in set(result.gt_article_ids))
    return min(available, len(result.hits)) / len(result.hits)


# ---------------------------------------------------------------------
# Per-question evaluation
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class EvaluatedAnswer:
    """One question, fully scored."""
    result: RAGResult
    faithfulness: Judgement | None = None
    relevancy: Judgement | None = None
    correctness: Judgement | None = None

    @property
    def question_id(self) -> str:
        return self.result.question_id

    @property
    def category(self) -> str:
        return self.result.category

    @property
    def context_precision(self) -> float | None:
        return context_precision(self.result)

    @property
    def context_recall(self) -> float | None:
        return context_recall(self.result)

    @property
    def is_refusal(self) -> bool:
        return self.result.refusal.is_refusal

    @property
    def failure_mode(self) -> str:
        """Where the fault lies, when the answer is unfaithful.

        This is the field that turns a score into an action. It is
        deliberately conservative: anything it cannot attribute is
        labelled 'unattributed' rather than guessed at.
        """
        if self.result.is_out_of_scope:
            return "correct_refusal" if self.is_refusal else "answered_oos"
        if self.is_refusal:
            recall = self.context_recall or 0.0
            return "over_refusal" if recall > 0 else "refused_no_evidence"

        if self.faithfulness is None:
            return "unjudged"
        faithful = self.faithfulness.score >= 0.7
        recall = self.context_recall or 0.0

        if faithful and recall > 0:
            return "ok"
        if not faithful and recall > 0:
            return "generation_failure"      # evidence present, ignored
        if not faithful and recall == 0:
            return "retrieval_failure"       # answered without evidence
        return "faithful_to_wrong_context"


def evaluate_answer(result: RAGResult, judge: Judge | None,
                    reference: str | None = None) -> EvaluatedAnswer:
    """Score one answer. Context metrics always; judged metrics if a judge.

    The judge is shown the context in EXACTLY the form the generator saw
    it, article-id tags included. That is not cosmetic.

    An earlier version passed bare chunk text with the ids stripped. An
    answer citing "[bill-007]" then presented the judge with a token
    absent from its context, which a careful judge should mark as an
    unsupported claim. The effect is systematic and directional: it
    deflates faithfulness for precisely the prompt variants that follow
    the citation instruction, so the `cited` and `strict` rungs of the
    Phase 4 ladder would have been penalised for complying with it.

    Evaluating against a different context than the model was given is
    not evaluating the model.
    """
    if judge is None:
        return EvaluatedAnswer(result=result)

    context = format_context(result.hits)
    return EvaluatedAnswer(
        result=result,
        faithfulness=judge.faithfulness(result.answer, context),
        relevancy=judge.relevancy(result.answer, result.question),
        correctness=(judge.correctness(result.answer, result.question, reference)
                     if reference else None),
    )


def evaluate_run(results: list[RAGResult], judge: Judge | None = None,
                 references: dict[str, str] | None = None,
                 progress_every: int = 0) -> list[EvaluatedAnswer]:
    """Score a full run."""
    references = references or {}
    out = []
    for i, r in enumerate(results, 1):
        out.append(evaluate_answer(r, judge, references.get(r.question_id)))
        if progress_every and i % progress_every == 0:
            print(f"    judged {i}/{len(results)} ...", flush=True)
    return out


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------
def _mean(values: list[float | None]) -> float:
    present = [v for v in values if v is not None]
    return float(np.mean(present)) if present else float("nan")


@dataclass(frozen=True)
class RunReport:
    """Aggregate scores for one answerer.

    `faithfulness` deliberately covers ANSWERED questions only. A refusal
    asserts nothing, so a correct judge scores it vacuously faithful at
    1.0 — which means averaging refusals in rewards a system for
    declining to be useful. Taken to its limit, a system that refuses
    every question reports perfect faithfulness.

    `faithfulness_all` keeps the all-questions figure so the difference
    is visible rather than hidden, but the answered-only number is the
    one that means "when this system makes claims, are they supported".
    """
    answerer: str
    n: int
    n_answered: int
    context_precision: float
    context_recall: float
    faithfulness: float            # answered questions only
    faithfulness_all: float        # including refusals — inflated
    relevancy: float
    correctness: float
    judged: bool
    failure_modes: dict[str, int] = field(default_factory=dict)

    def __str__(self) -> str:
        judged = "" if self.judged else "  (context metrics only — no judge)"
        return (f"{self.answerer}: ctx_prec={self.context_precision:.3f} "
                f"ctx_rec={self.context_recall:.3f} "
                f"faith={self.faithfulness:.3f} (answered) "
                f"rel={self.relevancy:.3f}{judged}")


def aggregate_run(evaluated: list[EvaluatedAnswer],
                  answerer: str | None = None) -> RunReport:
    """Mean each metric over the questions where it is defined.

    In-scope questions only for context metrics — out-of-scope questions
    have no ground truth, so including them would average in a quantity
    that is undefined rather than zero.
    """
    in_scope = [e for e in evaluated if not e.result.is_out_of_scope]
    answered = [e for e in evaluated if not e.is_refusal]
    judged = any(e.faithfulness is not None for e in evaluated)

    modes: dict[str, int] = {}
    for e in evaluated:
        modes[e.failure_mode] = modes.get(e.failure_mode, 0) + 1

    return RunReport(
        answerer=answerer or (evaluated[0].result.answerer if evaluated else "?"),
        n=len(evaluated),
        n_answered=len(answered),
        context_precision=_mean([e.context_precision for e in in_scope]),
        context_recall=_mean([e.context_recall for e in in_scope]),
        faithfulness=_mean([e.faithfulness.score if e.faithfulness else None
                            for e in answered]),
        faithfulness_all=_mean([e.faithfulness.score if e.faithfulness else None
                                for e in evaluated]),
        relevancy=_mean([e.relevancy.score if e.relevancy else None
                         for e in evaluated]),
        correctness=_mean([e.correctness.score if e.correctness else None
                           for e in evaluated]),
        judged=judged,
        failure_modes=dict(sorted(modes.items(), key=lambda kv: -kv[1])),
    )


def attribution_table(evaluated: list[EvaluatedAnswer]) -> dict[str, int]:
    """Cross-tabulate faithfulness against whether evidence was retrieved.

    The point of the whole harness: separating 'the model ignored good
    evidence' from 'the model was handed nothing and answered anyway'.
    Those need opposite fixes, and a single hallucination rate hides which
    one you have.
    """
    table = {"evidence+faithful": 0, "evidence+unfaithful": 0,
             "no_evidence+faithful": 0, "no_evidence+unfaithful": 0,
             "refused": 0, "out_of_scope": 0, "unjudged": 0}
    for e in evaluated:
        if e.faithfulness is None:
            table["unjudged"] += 1
            continue
        if e.result.is_out_of_scope:
            table["out_of_scope"] += 1
            continue
        if e.is_refusal:
            # A refusal asserts nothing, so attributing its faithfulness
            # to retrieval or generation is meaningless. Bucketing it
            # here keeps it out of both numerators — otherwise a judge
            # that scores refusals low (the lexical one does) would fill
            # the 'unfaithful' cells with responses that made no claims.
            table["refused"] += 1
            continue
        has_evidence = (e.context_recall or 0) > 0
        faithful = e.faithfulness.score >= 0.7
        key = ("evidence" if has_evidence else "no_evidence") + \
              ("+faithful" if faithful else "+unfaithful")
        table[key] += 1
    return table
