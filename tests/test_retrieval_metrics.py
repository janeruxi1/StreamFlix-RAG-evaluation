"""Tests for retrieval metrics.

Metric bugs are the most dangerous kind in this project: they produce a
complete, plausible, entirely wrong results table with no error anywhere.
So these tests pin the arithmetic against hand-computed values rather
than only checking that outputs look reasonable.
"""
import math

import numpy as np
import pytest

from src.corpus.build import load_corpus
from src.retrieval.chunking import whole_article
from src.retrieval.vectorstore import SearchHit
from src.evaluation.retrieval_metrics import (
    CostedConfig,
    aggregate,
    best_under_budget,
    bootstrap_ci,
    context_tokens,
    paired_bootstrap,
    pareto_frontier,
    score_question,
    score_run,
    selection_optimism,
    stratified_split,
)


@pytest.fixture(scope="module")
def chunk_by_article():
    return {c.article_id: c for c in whole_article(load_corpus())}


def hits_for(article_ids, chunk_by_article):
    """Build a fake ranking from article ids, best first."""
    return [SearchHit(chunk=chunk_by_article[a], score=1.0 - i * 0.01, rank=i)
            for i, a in enumerate(article_ids)]


def question(qid="q1", gt=("bill-001",), category="single_hop"):
    return {"question_id": qid, "question": "?", "category": category,
            "gt_article_ids": list(gt)}


# ---------------------------------------------------------------------
# Per-question metrics — pinned to hand-computed values
# ---------------------------------------------------------------------
def test_perfect_retrieval_scores_one(chunk_by_article):
    s = score_question(question(gt=("bill-001",)),
                       hits_for(["bill-001", "bill-002"], chunk_by_article), k=5)
    assert s.recall_at_k == 1.0
    assert s.hit_at_k == 1.0
    assert s.mrr == 1.0
    assert s.ndcg_at_k == 1.0


def test_complete_miss_scores_zero(chunk_by_article):
    s = score_question(question(gt=("bill-001",)),
                       hits_for(["dev-001", "dev-002"], chunk_by_article), k=5)
    assert s.recall_at_k == 0.0
    assert s.hit_at_k == 0.0
    assert s.mrr == 0.0
    assert s.ndcg_at_k == 0.0


def test_partial_recall_on_multi_source(chunk_by_article):
    """2 of 4 ground-truth articles found -> recall 0.5, hit 1.0."""
    s = score_question(
        question(gt=("bill-001", "bill-002", "dev-001", "dev-002")),
        hits_for(["bill-001", "strm-001", "bill-002"], chunk_by_article), k=5)
    assert s.recall_at_k == 0.5
    assert s.hit_at_k == 1.0


def test_mrr_uses_first_correct_rank(chunk_by_article):
    """Correct article third -> MRR = 1/3."""
    s = score_question(question(gt=("bill-005",)),
                       hits_for(["dev-001", "dev-002", "bill-005"],
                                chunk_by_article), k=5)
    assert s.mrr == pytest.approx(1 / 3)


def test_mrr_ignores_later_correct_hits(chunk_by_article):
    """Only the FIRST correct rank matters, by definition."""
    a = score_question(question(gt=("bill-001", "bill-002")),
                       hits_for(["bill-001", "dev-001"], chunk_by_article), k=5)
    b = score_question(question(gt=("bill-001", "bill-002")),
                       hits_for(["bill-001", "bill-002"], chunk_by_article), k=5)
    assert a.mrr == b.mrr == 1.0
    assert b.recall_at_k > a.recall_at_k      # recall does distinguish them


def test_ndcg_matches_manual_computation(chunk_by_article):
    """Ground truth at ranks 1 and 3 of 3, two relevant docs.

        DCG  = 1/log2(2) + 0/log2(3) + 1/log2(4) = 1 + 0 + 0.5 = 1.5
        IDCG = 1/log2(2) + 1/log2(3)             = 1 + 0.6309  = 1.6309
    """
    s = score_question(question(gt=("bill-001", "bill-003")),
                       hits_for(["bill-001", "dev-001", "bill-003"],
                                chunk_by_article), k=5)
    expected = (1 + 0 + 1 / math.log2(4)) / (1 + 1 / math.log2(3))
    assert s.ndcg_at_k == pytest.approx(expected)


def test_ndcg_penalises_late_placement(chunk_by_article):
    """Same recall, worse ranking -> strictly lower nDCG. This is the
    property that makes nDCG worth reporting alongside recall."""
    early = score_question(question(gt=("bill-001",)),
                           hits_for(["bill-001", "dev-001", "dev-002"],
                                    chunk_by_article), k=5)
    late = score_question(question(gt=("bill-001",)),
                          hits_for(["dev-001", "dev-002", "bill-001"],
                                   chunk_by_article), k=5)
    assert early.recall_at_k == late.recall_at_k
    assert early.ndcg_at_k > late.ndcg_at_k


def test_k_truncates_before_scoring(chunk_by_article):
    """A correct article beyond k must not count."""
    s = score_question(question(gt=("bill-009",)),
                       hits_for(["dev-001", "dev-002", "dev-003", "bill-009"],
                                chunk_by_article), k=3)
    assert s.recall_at_k == 0.0


def test_out_of_scope_question_is_rejected(chunk_by_article):
    """Scoring a no-ground-truth question would silently drag every
    average toward zero. It must fail loudly instead."""
    with pytest.raises(ValueError, match="out-of-scope"):
        score_question(question(gt=()), hits_for(["bill-001"], chunk_by_article))


def test_duplicate_chunks_from_one_article_count_once(chunk_by_article):
    """Article-level ground truth means three chunks of the same article
    are one hit, not three."""
    c = chunk_by_article["bill-001"]
    hits = [SearchHit(chunk=c, score=1.0, rank=i) for i in range(3)]
    s = score_question(question(gt=("bill-001",)), hits, k=5)
    assert s.recall_at_k == 1.0
    assert len(s.retrieved) == 1


def test_score_run_rejects_length_mismatch(chunk_by_article):
    with pytest.raises(ValueError, match="mismatch"):
        score_run([question()], [], k=5)


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------
def test_aggregate_splits_by_category(chunk_by_article):
    scores = [
        score_question(question("a", ("bill-001",), "single_hop"),
                       hits_for(["bill-001"], chunk_by_article), k=5),
        score_question(question("b", ("bill-002",), "multi_hop"),
                       hits_for(["dev-001"], chunk_by_article), k=5),
    ]
    agg = aggregate("cfg", scores)
    assert agg.overall["recall_at_k"] == 0.5
    assert agg.by_category["single_hop"]["recall_at_k"] == 1.0
    assert agg.by_category["multi_hop"]["recall_at_k"] == 0.0


# ---------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------
def test_bootstrap_ci_brackets_the_point_estimate(chunk_by_article):
    scores = [
        score_question(question(f"q{i}", ("bill-001",)),
                       hits_for(["bill-001"] if i % 2 else ["dev-001"],
                                chunk_by_article), k=5)
        for i in range(40)
    ]
    point, lo, hi = bootstrap_ci(scores, n_boot=500)
    assert lo <= point <= hi
    assert 0.0 <= lo <= hi <= 1.0


def test_bootstrap_ci_is_deterministic(chunk_by_article):
    scores = [score_question(question(f"q{i}", ("bill-001",)),
                             hits_for(["bill-001"], chunk_by_article), k=5)
              for i in range(10)]
    assert bootstrap_ci(scores, seed=7) == bootstrap_ci(scores, seed=7)


def test_identical_configs_produce_zero_difference(chunk_by_article):
    scores = [score_question(question(f"q{i}", ("bill-001",)),
                             hits_for(["bill-001"], chunk_by_article), k=5)
              for i in range(20)]
    cmp = paired_bootstrap(scores, scores, n_boot=200)
    assert cmp.difference == 0.0
    assert cmp.n_differing == 0
    assert not cmp.significant


def test_paired_bootstrap_detects_a_real_gap(chunk_by_article):
    """One config right on every question, the other wrong on every one:
    the difference must be significant."""
    good = [score_question(question(f"q{i}", ("bill-001",)),
                           hits_for(["bill-001"], chunk_by_article), k=5)
            for i in range(30)]
    bad = [score_question(question(f"q{i}", ("bill-001",)),
                          hits_for(["dev-001"], chunk_by_article), k=5)
           for i in range(30)]
    cmp = paired_bootstrap(good, bad, n_boot=500)
    assert cmp.difference == 1.0
    assert cmp.significant
    assert cmp.n_differing == 30


def test_paired_bootstrap_rejects_misaligned_runs(chunk_by_article):
    """Comparing different question sets would yield a confident,
    meaningless number — it must raise instead."""
    a = [score_question(question("q1", ("bill-001",)),
                        hits_for(["bill-001"], chunk_by_article), k=5)]
    b = [score_question(question("q2", ("bill-001",)),
                        hits_for(["bill-001"], chunk_by_article), k=5)]
    with pytest.raises(ValueError, match="misaligned"):
        paired_bootstrap(a, b)


# ---------------------------------------------------------------------
# Cost and the frontier
# ---------------------------------------------------------------------
def test_context_tokens_counts_generation_text(chunk_by_article):
    hits = hits_for(["bill-001", "bill-002"], chunk_by_article)
    assert context_tokens(hits) > 0
    assert context_tokens(hits) > context_tokens(hits[:1])


def test_pareto_frontier_drops_dominated_configs():
    configs = [
        CostedConfig("cheap_good", 0.80, 100),
        CostedConfig("dominated", 0.70, 200),      # worse AND costlier
        CostedConfig("expensive_better", 0.90, 500),
    ]
    names = [c.config for c in pareto_frontier(configs)]
    assert "dominated" not in names
    assert set(names) == {"cheap_good", "expensive_better"}


def test_pareto_frontier_is_cost_ordered():
    configs = [CostedConfig("c", 0.9, 500), CostedConfig("a", 0.6, 50),
               CostedConfig("b", 0.8, 200)]
    costs = [c.mean_tokens for c in pareto_frontier(configs)]
    assert costs == sorted(costs)


def test_best_under_budget_breaks_ties_on_cost():
    """Without a tie-break, the 'best' config is whichever was
    enumerated first — which is not a decision."""
    configs = [CostedConfig("pricey", 0.85, 400),
               CostedConfig("cheap", 0.85, 150)]
    assert best_under_budget(configs, 600).config == "cheap"


def test_best_under_budget_returns_none_when_nothing_fits():
    assert best_under_budget([CostedConfig("x", 0.9, 1000)], 100) is None


# ---------------------------------------------------------------------
# Selection honesty
# ---------------------------------------------------------------------
def test_stratified_split_is_exhaustive_and_disjoint():
    questions = [question(f"q{i}", ("bill-001",),
                          ["single_hop", "multi_hop", "ambiguous"][i % 3])
                 for i in range(60)]
    dev, test = stratified_split(questions, test_fraction=0.4)
    ids_dev = {q["question_id"] for q in dev}
    ids_test = {q["question_id"] for q in test}
    assert not (ids_dev & ids_test)
    assert len(ids_dev | ids_test) == 60


def test_stratified_split_preserves_category_proportions():
    """A uniform split can leave a rare category almost absent from test,
    which makes that column meaningless."""
    questions = [question(f"q{i}", ("bill-001",),
                          "rare" if i < 10 else "common")
                 for i in range(100)]
    dev, test = stratified_split(questions, test_fraction=0.4)
    assert sum(q["category"] == "rare" for q in test) == 4
    assert sum(q["category"] == "rare" for q in dev) == 6


def test_stratified_split_is_deterministic():
    questions = [question(f"q{i}", ("bill-001",)) for i in range(30)]
    a, _ = stratified_split(questions, seed=1)
    b, _ = stratified_split(questions, seed=1)
    assert [q["question_id"] for q in a] == [q["question_id"] for q in b]


def test_selection_optimism_picks_the_dev_winner(chunk_by_article):
    def run(correct):
        return [score_question(question(f"q{i}", ("bill-001",)),
                               hits_for(["bill-001"] if correct else ["dev-001"],
                                        chunk_by_article), k=5)
                for i in range(10)]

    dev = {"good": run(True), "bad": run(False)}
    test = {"good": run(True), "bad": run(False)}
    result = selection_optimism(dev, test)
    assert result["winner"] == "good"
    assert result["optimism"] == 0.0
    assert result["regret"] == 0.0
