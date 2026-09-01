"""Tests for the evaluation harness.

The judge is a measuring instrument, so most of these tests are about the
instrument rather than the readings: does the validation suite actually
discriminate, does parsing fail loudly, are undefined metrics reported as
undefined rather than zero.
"""
import pytest

from src.corpus.build import load_corpus, load_golden_set
from src.evaluation.judge import (
    Judgement,
    LexicalJudge,
    get_judge,
    parse_faithfulness,
    parse_score,
)
from src.evaluation.judge_validation import (
    VALIDATION_CASES,
    validate_judge,
)
from src.evaluation.rag_metrics import (
    aggregate_run,
    attribution_table,
    context_precision,
    context_precision_ceiling,
    context_recall,
    evaluate_run,
)
from src.generation.pipeline import RAGResult
from src.generation.refusal import CANONICAL_REFUSAL
from src.retrieval.chunking import markdown_section, whole_article
from src.retrieval.vectorstore import SearchHit


@pytest.fixture(scope="module")
def articles():
    return load_corpus()


@pytest.fixture(scope="module")
def by_article(articles):
    return {c.article_id: c for c in whole_article(articles)}


def hits(article_ids, by_article):
    return [SearchHit(chunk=by_article[a], score=1.0 - i * 0.01, rank=i)
            for i, a in enumerate(article_ids)]


def result(gt=("bill-001",), retrieved=("bill-001",), answer="An answer.",
           by_article=None, category="single_hop"):
    return RAGResult(question_id="q1", question="Q?", category=category,
                     answer=answer, answerer="t",
                     hits=hits(retrieved, by_article),
                     gt_article_ids=tuple(gt))


# ---------------------------------------------------------------------
# Parsing — a format violation is not a zero
# ---------------------------------------------------------------------
def test_faithfulness_parses_claim_counts():
    j = parse_faithfulness("CLAIMS_TOTAL: 4\nCLAIMS_SUPPORTED: 3\nREASONING: x")
    assert j.score == 0.75
    assert j.n_claims == 4 and j.n_supported == 3
    assert j.parsed_ok


def test_zero_claims_is_vacuously_faithful():
    """A refusal asserts nothing. Scoring it 0 would punish correct
    refusals on the metric meant to reward not inventing things."""
    j = parse_faithfulness("CLAIMS_TOTAL: 0\nCLAIMS_SUPPORTED: 0")
    assert j.score == 1.0


def test_supported_cannot_exceed_total():
    """Guard against a judge miscount producing a score above 1."""
    j = parse_faithfulness("CLAIMS_TOTAL: 2\nCLAIMS_SUPPORTED: 5")
    assert j.score == 1.0


def test_unparseable_response_is_flagged_not_scored_zero():
    """A parse failure and a genuine zero are different facts about the
    system; collapsing them would hide judge malfunctions as bad answers."""
    j = parse_faithfulness("I think the answer is mostly fine actually.")
    assert not j.parsed_ok
    assert j.score == 0.0


def test_score_is_clamped_to_unit_interval():
    assert parse_score("SCORE: 1.7", "relevancy").score == 1.0
    assert parse_score("SCORE: -0.3", "relevancy").score == 0.0


def test_reasoning_is_extracted():
    j = parse_score("SCORE: 0.5\nREASONING: partially addresses it", "relevancy")
    assert "partially addresses" in j.reasoning


# ---------------------------------------------------------------------
# The validation suite must actually discriminate
# ---------------------------------------------------------------------
def test_validation_suite_covers_every_failure_kind():
    kinds = {c.kind for c in VALIDATION_CASES}
    assert kinds == {"supported", "fabricated", "contradicted",
                     "refusal", "off_topic"}


def test_validation_cases_have_both_expectations_represented():
    """A suite where every case expects 'high' could be passed by a judge
    that always returns 1.0."""
    assert any(not c.expect_faithful_high for c in VALIDATION_CASES)
    assert any(not c.expect_relevant_high for c in VALIDATION_CASES)


def test_an_always_high_judge_fails_validation():
    """The suite must not be passable by a constant."""
    class AlwaysHigh(LexicalJudge):
        name = "always_high"
        def faithfulness(self, a, c): return Judgement("faithfulness", 1.0)
        def relevancy(self, a, q): return Judgement("relevancy", 1.0)

    report = validate_judge(AlwaysHigh())
    assert not report.is_trustworthy


def test_an_always_low_judge_fails_validation():
    class AlwaysLow(LexicalJudge):
        name = "always_low"
        def faithfulness(self, a, c): return Judgement("faithfulness", 0.0)
        def relevancy(self, a, q): return Judgement("relevancy", 0.0)

    assert not validate_judge(AlwaysLow()).is_trustworthy


def test_a_perfect_judge_passes_validation():
    """Sanity check on the harness itself: an oracle must score 100%."""
    class Oracle(LexicalJudge):
        name = "oracle"
        def __init__(self):
            self._by_answer = {c.answer: c for c in VALIDATION_CASES}
        def faithfulness(self, answer, context):
            c = self._by_answer[answer]
            return Judgement("faithfulness", 1.0 if c.expect_faithful_high else 0.0)
        def relevancy(self, answer, question):
            c = self._by_answer[answer]
            return Judgement("relevancy", 1.0 if c.expect_relevant_high else 0.0)

    report = validate_judge(Oracle())
    assert report.overall_accuracy == 1.0
    assert report.is_trustworthy


def test_lexical_judge_fails_on_contradictions(articles):
    """The documented reason a lexical judge cannot replace a model.

    A contradicting sentence reuses nearly every term of the context it
    contradicts, so overlap scores it as faithful. This is not a bug to
    fix — it is what 'semantic' means, demonstrated rather than asserted.
    """
    report = validate_judge(LexicalJudge())
    contradicted = [r for r in report.results if r.case.kind == "contradicted"]
    assert contradicted
    assert all(r.faithfulness.score >= 0.7 for r in contradicted), \
        "expected overlap to rate contradictions as faithful"
    assert not any(r.faithfulness_correct for r in contradicted)


def test_lexical_judge_is_not_trustworthy():
    assert not validate_judge(LexicalJudge()).is_trustworthy


def test_validation_report_lists_its_failures():
    report = validate_judge(LexicalJudge())
    assert len(report.failures()) > 0
    assert len(report.failures()) <= len(report.results)


# ---------------------------------------------------------------------
# Context metrics — exact, and undefined where they should be
# ---------------------------------------------------------------------
def test_context_precision_counts_chunks_not_articles(by_article):
    r = result(gt=("bill-001",), retrieved=("bill-001", "dev-001", "dev-002"),
               by_article=by_article)
    assert context_precision(r) == pytest.approx(1 / 3)


def test_context_precision_is_none_for_out_of_scope(by_article):
    """No ground truth means no chunk can be relevant. Scoring 0 would
    penalise the retriever for a question with no right answer."""
    r = result(gt=(), retrieved=("bill-001",), by_article=by_article,
               category="out_of_scope")
    assert context_precision(r) is None
    assert context_recall(r) is None


def test_context_precision_is_zero_with_no_relevant_chunks(by_article):
    r = result(gt=("bill-001",), retrieved=("dev-001", "dev-002"),
               by_article=by_article)
    assert context_precision(r) == 0.0


def test_ceiling_reflects_available_relevant_chunks(articles, by_article):
    """The ceiling is arithmetic: you cannot retrieve more relevant
    chunks than exist."""
    chunks = markdown_section(articles)
    per_article = {}
    for c in chunks:
        per_article[c.article_id] = per_article.get(c.article_id, 0) + 1

    r = result(gt=("bill-001",), retrieved=("bill-001",) * 10,
               by_article=by_article)
    ceiling = context_precision_ceiling(r, per_article)
    assert ceiling == pytest.approx(min(per_article["bill-001"], 10) / 10)


def test_ceiling_rises_with_more_ground_truth_articles(articles, by_article):
    per_article = {a.article_id: 4 for a in articles}
    one = result(gt=("bill-001",), retrieved=("bill-001",) * 12,
                 by_article=by_article)
    three = result(gt=("bill-001", "bill-002", "dev-001"),
                   retrieved=("bill-001",) * 12, by_article=by_article)
    assert context_precision_ceiling(three, per_article) > \
           context_precision_ceiling(one, per_article)


def test_ceiling_is_none_without_ground_truth(by_article):
    r = result(gt=(), retrieved=("bill-001",), by_article=by_article)
    assert context_precision_ceiling(r, {"bill-001": 4}) is None


# ---------------------------------------------------------------------
# Attribution
# ---------------------------------------------------------------------
def test_refusals_are_excluded_from_attribution(by_article):
    """A refusal asserts nothing, so attributing its faithfulness to
    retrieval or generation is meaningless. Regression: refusals used to
    land in the 'unfaithful' cells because the lexical judge scores them
    zero."""
    r = result(gt=("bill-001",), retrieved=("bill-001",),
               answer=CANONICAL_REFUSAL, by_article=by_article)
    table = attribution_table(evaluate_run([r], judge=LexicalJudge()))
    assert table["refused"] == 1
    assert table["evidence+unfaithful"] == 0


def test_out_of_scope_excluded_from_attribution(by_article):
    r = result(gt=(), retrieved=("bill-001",), by_article=by_article,
               category="out_of_scope")
    table = attribution_table(evaluate_run([r], judge=LexicalJudge()))
    assert table["out_of_scope"] == 1


def test_unjudged_run_is_marked_not_scored(by_article):
    r = result(gt=("bill-001",), retrieved=("bill-001",), by_article=by_article)
    ev = evaluate_run([r], judge=None)
    assert ev[0].faithfulness is None
    assert ev[0].failure_mode == "unjudged"
    assert attribution_table(ev)["unjudged"] == 1


def test_aggregate_reports_whether_it_was_judged(by_article):
    r = result(gt=("bill-001",), retrieved=("bill-001",), by_article=by_article)
    assert not aggregate_run(evaluate_run([r], judge=None)).judged
    assert aggregate_run(evaluate_run([r], judge=LexicalJudge())).judged


def test_context_metrics_exclude_out_of_scope_from_the_mean(by_article):
    """Averaging in an undefined quantity is not the same as averaging
    in a zero."""
    good = result(gt=("bill-001",), retrieved=("bill-001",),
                  by_article=by_article)
    oos = result(gt=(), retrieved=("dev-001",), by_article=by_article,
                 category="out_of_scope")
    rep = aggregate_run(evaluate_run([good, oos], judge=None))
    assert rep.context_precision == 1.0      # not dragged down by the oos row


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
def test_get_judge_returns_lexical_without_a_provider():
    assert isinstance(get_judge(None), LexicalJudge)


def test_end_to_end_evaluation_runs_without_a_credential():
    """The whole harness must work keyless, or CI cannot exercise it."""
    from src.retrieval.retrievers import BM25Retriever
    from src.generation.extractive import ExtractiveAnswerer
    from src.generation.pipeline import RAGPipeline

    arts = load_corpus()
    golden = load_golden_set()[:12]
    pipe = RAGPipeline(BM25Retriever(markdown_section(arts)),
                       ExtractiveAnswerer(), depth=10)
    rep = aggregate_run(evaluate_run(pipe.run(golden), judge=LexicalJudge()))
    assert rep.n == 12
    assert rep.judged


# ---------------------------------------------------------------------
# Regressions from the Phase 5 review
# ---------------------------------------------------------------------
def test_judge_sees_the_same_context_the_generator_saw(by_article):
    """Regression: the judge was passed bare chunk text with article ids
    stripped, while the generator saw id-tagged blocks.

    An answer citing [bill-001] then handed the judge a token absent from
    its context — an unsupported claim by construction. The bias is
    systematic and directional: it penalises exactly the prompt variants
    that comply with the citation instruction.
    """
    captured = {}

    class Recording(LexicalJudge):
        name = "recording"
        def faithfulness(self, answer, context):
            captured["context"] = context
            return Judgement("faithfulness", 1.0)

    r = result(gt=("bill-001",), retrieved=("bill-001", "dev-001"),
               answer="A claim [bill-001].", by_article=by_article)
    evaluate_run([r], judge=Recording())

    for article_id in ("bill-001", "dev-001"):
        assert f"[{article_id}]" in captured["context"], \
            "judge context must carry article ids, as the generator's did"


def test_cited_answer_is_not_penalised_for_its_citation(by_article):
    """The consequence of the bug above, stated as a behaviour: every id
    an answer can legitimately cite must be present in the judge's view
    of the context."""
    r = result(gt=("bill-001",), retrieved=("bill-001",),
               answer="Premium costs $19.99 [bill-001].", by_article=by_article)
    ev = evaluate_run([r], judge=LexicalJudge())[0]
    cited = ev.result.citations
    assert cited == ["bill-001"]
    assert all(c in ev.result.context_article_ids for c in cited)


def test_faithfulness_excludes_refusals_from_the_headline_mean(by_article):
    """Regression: refusals were averaged into faithfulness.

    A refusal asserts nothing, so a correct judge scores it vacuously
    faithful at 1.0 — meaning a system that refuses everything would
    report perfect faithfulness. The headline number must cover answered
    questions only.
    """
    class RefusalIsFaithful(LexicalJudge):
        name = "vacuous"
        def faithfulness(self, answer, context):
            from src.generation.refusal import is_refusal
            return Judgement("faithfulness", 1.0 if is_refusal(answer) else 0.2)

    rows = [
        result(gt=("bill-001",), retrieved=("bill-001",),
               answer=CANONICAL_REFUSAL, by_article=by_article),
        result(gt=("bill-001",), retrieved=("bill-001",),
               answer="An unsupported claim.", by_article=by_article),
    ]
    rep = aggregate_run(evaluate_run(rows, judge=RefusalIsFaithful()))

    assert rep.n_answered == 1
    assert rep.faithfulness == pytest.approx(0.2)        # answered only
    assert rep.faithfulness_all == pytest.approx(0.6)    # inflated by refusal
    assert rep.faithfulness < rep.faithfulness_all


def test_refuse_everything_does_not_score_perfect_faithfulness(by_article):
    """The limiting case the fix exists to prevent."""
    class RefusalIsFaithful(LexicalJudge):
        name = "vacuous"
        def faithfulness(self, answer, context):
            return Judgement("faithfulness", 1.0)

    rows = [result(gt=("bill-001",), retrieved=("bill-001",),
                   answer=CANONICAL_REFUSAL, by_article=by_article)
            for _ in range(5)]
    rep = aggregate_run(evaluate_run(rows, judge=RefusalIsFaithful()))
    assert rep.n_answered == 0
    assert rep.faithfulness != rep.faithfulness or rep.faithfulness == 0.0 \
        or str(rep.faithfulness) == "nan"   # undefined, not 1.0
