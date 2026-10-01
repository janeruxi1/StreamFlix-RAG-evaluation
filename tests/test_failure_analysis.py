"""Failure-analysis taxonomy: each cause assigned for the right reason."""
import pytest

from src.corpus.build import load_corpus
from src.evaluation.failure_analysis import (
    classify,
    classify_run,
    render_markdown,
    summarise,
)
from src.evaluation.judge import Judgement
from src.evaluation.rag_metrics import EvaluatedAnswer
from src.generation.pipeline import RAGResult
from src.generation.refusal import CANONICAL_REFUSAL
from src.retrieval.chunking import whole_article
from src.retrieval.vectorstore import SearchHit


@pytest.fixture(scope="module")
def by_article():
    return {c.article_id: c for c in whole_article(load_corpus())}


def ev(by_article, gt, retrieved, answer="Premium costs $19.", faith=None):
    r = RAGResult(
        question_id="q", question="Q?", category="single_hop", answer=answer,
        answerer="t",
        hits=[SearchHit(chunk=by_article[a], score=1.0, rank=i)
              for i, a in enumerate(retrieved)],
        gt_article_ids=tuple(gt))
    f = Judgement("faithfulness", faith, "") if faith is not None else None
    return EvaluatedAnswer(result=r, faithfulness=f)


def test_correct_answer_is_ok(by_article):
    assert classify(ev(by_article, ["bill-001"], ["bill-001"])).cause == "ok"


def test_no_evidence_is_retrieval_miss_even_if_answered(by_article):
    assert classify(ev(by_article, ["bill-001"], ["dev-001"])).cause == "retrieval_miss"


def test_refusal_with_evidence_is_over_refusal(by_article):
    rec = classify(ev(by_article, ["bill-001"], ["bill-001"], CANONICAL_REFUSAL))
    assert (rec.cause, rec.owner) == ("over_refusal", "generation")


def test_refusal_without_evidence_blames_retrieval_not_generator(by_article):
    rec = classify(ev(by_article, ["bill-001"], ["dev-001"], CANONICAL_REFUSAL))
    assert rec.owner == "retrieval"


def test_partial_recall_is_retrieval_partial(by_article):
    rec = classify(ev(by_article, ["bill-002", "bill-003"], ["bill-002"]))
    assert rec.cause == "retrieval_partial"


def test_unfaithful_answer_with_evidence_is_generation(by_article):
    rec = classify(ev(by_article, ["bill-001"], ["bill-001"], faith=0.2))
    assert rec.cause == "unfaithful"


def test_oos_answered_is_failure_and_oos_refused_is_not(by_article):
    assert classify(ev(by_article, [], ["bill-001"])).cause == "answered_oos"
    good = classify(ev(by_article, [], ["bill-001"], CANONICAL_REFUSAL))
    assert good.cause == "correct_refusal" and not good.is_failure


def test_corpus_flaw_is_a_tag_not_a_cause(by_article):
    rec = classify(ev(by_article, ["bill-002", "bill-003"], ["bill-002", "bill-003"]))
    assert rec.cause == "ok"
    assert "contradiction-refund-window" in rec.flaw_tags


def test_summary_counts_and_markdown(by_article):
    evs = [ev(by_article, ["bill-001"], ["bill-001"]),
           ev(by_article, ["bill-001"], ["dev-001"]),
           ev(by_article, [], ["bill-001"])]
    s = summarise(classify_run(evs))
    assert s.n == 3 and s.n_failures == 2
    assert s.by_owner["retrieval"] == 1 and s.by_owner["generation"] == 1
    md = render_markdown(s, "T")
    assert "retrieval_miss" in md and "answered_oos" in md
