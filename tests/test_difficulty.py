"""Tests for the benchmark-difficulty analysis.

The BM25 baseline is the floor every Phase 3 retrieval result is
reported against. If it silently breaks, later comparisons become
meaningless without failing anything visibly.
"""
from __future__ import annotations

import pytest

from src.corpus.build import load_corpus, load_golden_set, write_corpus, write_golden_set
from src.corpus.difficulty import (
    BM25,
    count_tokens,
    evaluate_lexical_baseline,
    jaccard,
    most_similar_pairs,
    question_article_overlap,
    tokenize,
)


@pytest.fixture(scope="module")
def corpus_and_golden(tmp_path_factory):
    root = tmp_path_factory.mktemp("difficulty")
    write_corpus(root / "corpus")
    write_golden_set(root / "golden")
    return load_corpus(root / "corpus"), load_golden_set(root / "golden")


@pytest.fixture(scope="module")
def documents(corpus_and_golden):
    articles, _ = corpus_and_golden
    return {a.article_id: a.as_document() for a in articles}


# ---------------------------------------------------------------------
# Tokenization
# ---------------------------------------------------------------------
def test_tokenize_strips_stopwords():
    assert "the" not in tokenize("the refund policy")
    assert "refund" in tokenize("the refund policy")


def test_tokenize_drops_short_tokens():
    assert tokenize("my tv is on") == []


def test_tokenize_is_case_insensitive():
    assert tokenize("Refund POLICY") == tokenize("refund policy")


# ---------------------------------------------------------------------
# BM25 correctness
# ---------------------------------------------------------------------
def test_bm25_ranks_exact_match_first():
    docs = {
        "a": "refund policy and how refunds are processed",
        "b": "video quality and streaming resolution settings",
        "c": "installing the app on a smart television",
    }
    assert BM25(docs).rank("refund")[0] == "a"


def test_bm25_returns_every_document():
    docs = {"a": "alpha content", "b": "beta content", "c": "gamma content"}
    assert len(BM25(docs).rank("content")) == 3


def test_bm25_respects_top_k():
    docs = {"a": "alpha", "b": "beta", "c": "gamma"}
    assert len(BM25(docs).rank("alpha", top_k=2)) == 2


def test_bm25_scores_zero_for_absent_terms():
    bm25 = BM25({"a": "refund policy"})
    assert bm25.score(tokenize("quantum astrophysics"), "a") == 0.0


def test_bm25_rare_terms_outweigh_common_ones():
    """IDF must make a distinctive term dominate a ubiquitous one."""
    docs = {
        "common": "streaming streaming streaming service",
        "rare": "streaming chromecast",
    }
    assert BM25(docs).rank("chromecast")[0] == "rare"


# ---------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------
def test_baseline_covers_in_scope_categories(documents, corpus_and_golden):
    _, golden = corpus_and_golden
    per_cat, overall = evaluate_lexical_baseline(documents, golden)
    assert {r.category for r in per_cat} == {
        "single_hop", "multi_hop", "ambiguous"
    }
    assert overall.n_questions == 95  # 120 minus 25 out-of-scope


def test_baseline_excludes_out_of_scope(documents, corpus_and_golden):
    """Out-of-scope questions have no ground truth, so retrieval metrics
    are undefined for them. Including them would corrupt the floor."""
    _, golden = corpus_and_golden
    _, overall = evaluate_lexical_baseline(documents, golden)
    n_in_scope = sum(1 for q in golden if q["category"] != "out_of_scope")
    assert overall.n_questions == n_in_scope


def test_baseline_metrics_are_valid_proportions(documents, corpus_and_golden):
    _, golden = corpus_and_golden
    per_cat, overall = evaluate_lexical_baseline(documents, golden)
    for r in [*per_cat, overall]:
        for value in (r.hit_at_1, r.hit_at_3, r.hit_at_5, r.recall_at_5):
            assert 0.0 <= value <= 1.0, r.category


def test_hit_rate_is_monotonic_in_k(documents, corpus_and_golden):
    """hit@1 <= hit@3 <= hit@5 by construction — a violation means the
    ranking or the slicing is wrong."""
    _, golden = corpus_and_golden
    per_cat, overall = evaluate_lexical_baseline(documents, golden)
    for r in [*per_cat, overall]:
        assert r.hit_at_1 <= r.hit_at_3 <= r.hit_at_5, r.category


def test_baseline_beats_chance(documents, corpus_and_golden):
    """Sanity floor: BM25 must substantially outperform random ranking
    over 45 documents (~11% hit@5 by chance)."""
    _, golden = corpus_and_golden
    _, overall = evaluate_lexical_baseline(documents, golden)
    assert overall.hit_at_5 > 0.5


def test_single_hop_is_easier_than_ambiguous(documents, corpus_and_golden):
    """The category design should produce a difficulty gradient. If
    ambiguous questions were as easy as single-hop, the labels would not
    mean what they claim."""
    _, golden = corpus_and_golden
    per_cat, _ = evaluate_lexical_baseline(documents, golden)
    by_cat = {r.category: r for r in per_cat}
    assert by_cat["single_hop"].recall_at_5 > by_cat["ambiguous"].recall_at_5


# ---------------------------------------------------------------------
# Question / article overlap
# ---------------------------------------------------------------------
def test_questions_are_not_keyword_matched(corpus_and_golden):
    """Questions must not simply echo article titles, or the benchmark
    measures string matching rather than retrieval."""
    articles, golden = corpus_and_golden
    titles = {a.article_id: a.title for a in articles}
    report = question_article_overlap(titles, golden)
    assert report.mean_title_overlap < 0.35
    assert report.zero_overlap_rate > 0.30


def test_overlap_report_counts_in_scope_only(corpus_and_golden):
    articles, golden = corpus_and_golden
    titles = {a.article_id: a.title for a in articles}
    report = question_article_overlap(titles, golden)
    assert report.n_questions == 95


# ---------------------------------------------------------------------
# Near-duplicate detection
# ---------------------------------------------------------------------
def test_jaccard_bounds():
    assert jaccard("refund policy", "refund policy") == 1.0
    assert jaccard("refund policy", "chromecast setup") == 0.0


def test_planted_cluster_is_highly_confusable(documents):
    """All three device-setup pairs must sit in the most-similar region.

    Originally asserted the cluster held 2 of the top 3 pairs. It does
    not — and the failure was informative rather than a bad threshold.
    An UNPLANTED pair (bill-003 cancel-subscription vs trial-004
    cancel-during-trial) scores higher than any planted pair, because
    both describe the same cancellation flow for different account
    states. See test_emergent_near_duplicate_is_documented.

    The correct assertion is that all three planted pairs are in the top
    tier of confusability, not that they monopolise it.
    """
    cluster = {"dev-003", "dev-004", "dev-005"}
    top_pairs = most_similar_pairs(documents, top_n=8)
    from_cluster = sum(1 for a, b, _ in top_pairs if {a, b} <= cluster)
    assert from_cluster == 3, "all 3 planted device pairs should rank in top 8"


def test_emergent_near_duplicate_is_documented(documents):
    """The most-similar pair in the corpus emerged by accident and is
    now documented as a discovered flaw.

    Guards against a future corpus edit silently removing it — the
    Phase 3 retrieval analysis references it as a real confusion case.
    """
    from src.corpus.seed_articles import KNOWN_CORPUS_FLAWS

    top_pair = most_similar_pairs(documents, top_n=1)[0]
    a, b, similarity = top_pair
    assert {a, b} == {"bill-003", "trial-004"}
    assert similarity > 0.25

    documented = {
        frozenset(f["article_ids"]) for f in KNOWN_CORPUS_FLAWS
    }
    assert frozenset({"bill-003", "trial-004"}) in documented


# ---------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------
def test_count_tokens_is_positive():
    assert count_tokens("the refund policy applies for 30 days") > 0


def test_count_tokens_scales_with_length():
    short = count_tokens("refund policy")
    long = count_tokens("refund policy " * 50)
    assert long > short * 10


def test_every_article_fits_embedding_window(corpus_and_golden):
    """512 tokens is the limit for BGE-small and most sentence-transformers
    models. An article over the limit would be silently truncated."""
    articles, _ = corpus_and_golden
    for a in articles:
        assert count_tokens(a.as_document()) <= 512, a.article_id
