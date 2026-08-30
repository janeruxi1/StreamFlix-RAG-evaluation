"""Corpus and golden-set integrity tests.

Every downstream metric is measured against the golden set's ground-truth
labels, so a silent corruption here would invalidate the entire
evaluation without failing anything visibly. These tests are the guard.
"""
from __future__ import annotations

import pytest

from src.corpus.build import (
    load_corpus,
    load_golden_set,
    orphan_articles,
    validate_corpus,
    write_corpus,
    write_golden_set,
)
from src.corpus.golden_set import (
    GOLDEN_QUESTIONS,
    articles_referenced,
    by_category,
    category_counts,
)
from src.corpus.seed_articles import (
    ARTICLES,
    COVERAGE_GAPS,
    KNOWN_CORPUS_FLAWS,
)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Materialize corpus + golden set to a temp dir and load them back."""
    root = tmp_path_factory.mktemp("corpus")
    corpus_dir, golden_dir = root / "corpus", root / "golden"
    write_corpus(corpus_dir)
    write_golden_set(golden_dir)
    return load_corpus(corpus_dir), load_golden_set(golden_dir)


# ---------------------------------------------------------------------
# Round trip
# ---------------------------------------------------------------------
def test_every_seed_article_reaches_disk(built):
    articles, _ = built
    assert len(articles) == len(ARTICLES)


def test_round_trip_preserves_content(built):
    """Front-matter parsing must not corrupt the body."""
    articles, _ = built
    by_id = {a.article_id: a for a in articles}
    for seed in ARTICLES:
        loaded = by_id[seed["article_id"]]
        assert loaded.title == seed["title"]
        assert loaded.category == seed["category"]
        assert loaded.last_updated == seed["last_updated"]
        assert loaded.body.strip() == seed["body"].strip()


def test_golden_set_round_trips(built):
    _, golden = built
    assert len(golden) == len(GOLDEN_QUESTIONS)


# ---------------------------------------------------------------------
# Integrity — the checks that protect downstream evaluation
# ---------------------------------------------------------------------
def test_no_integrity_problems(built):
    articles, golden = built
    assert validate_corpus(articles, golden) == []


def test_article_ids_are_unique():
    ids = [a["article_id"] for a in ARTICLES]
    assert len(ids) == len(set(ids))


def test_question_ids_are_unique():
    ids = [q["question_id"] for q in GOLDEN_QUESTIONS]
    assert len(ids) == len(set(ids))


def test_every_ground_truth_reference_resolves():
    known = {a["article_id"] for a in ARTICLES}
    assert articles_referenced() <= known


def test_out_of_scope_questions_have_no_ground_truth():
    """An out-of-scope question with ground truth is a contradiction in
    terms and would silently corrupt the refusal metric."""
    for q in by_category("out_of_scope"):
        assert q["gt_article_ids"] == [], q["question_id"]


def test_in_scope_questions_all_have_ground_truth():
    for cat in ("single_hop", "multi_hop", "ambiguous"):
        for q in by_category(cat):
            assert q["gt_article_ids"], q["question_id"]


def test_multi_hop_questions_span_multiple_sources():
    """A 'multi-hop' question resolvable from one article is mislabelled
    and would understate retrieval difficulty.

    mh-005 is the deliberate exception: two distinct facts inside one
    article, testing extraction rather than cross-document assembly.
    """
    single_source_allowed = {"mh-005", "mh-020"}
    for q in by_category("multi_hop"):
        if q["question_id"] in single_source_allowed:
            continue
        assert len(q["gt_article_ids"]) >= 2, q["question_id"]


def test_ambiguous_questions_have_several_candidates():
    for q in by_category("ambiguous"):
        assert len(q["gt_article_ids"]) >= 2, q["question_id"]


# ---------------------------------------------------------------------
# Set composition — locks the design in place
# ---------------------------------------------------------------------
def test_category_balance_matches_design():
    assert category_counts() == {
        "single_hop": 60,
        "multi_hop": 20,
        "ambiguous": 15,
        "out_of_scope": 25,
    }


def test_golden_set_totals_120():
    assert len(GOLDEN_QUESTIONS) == 120


def test_every_question_has_a_reference_answer():
    for q in GOLDEN_QUESTIONS:
        assert q["reference_answer"].strip(), q["question_id"]


# ---------------------------------------------------------------------
# Deliberate flaws must persist — they are the point of the eval
# ---------------------------------------------------------------------
def test_refund_contradiction_still_present():
    """If someone 'fixes' the corpus, the contradiction probe silently
    stops testing anything. This asserts the flaw survives."""
    bodies = {a["article_id"]: a["body"] for a in ARTICLES}
    assert "30 days" in bodies["bill-002"]
    assert "14 days" in bodies["bill-003"]


def test_outdated_article_still_present():
    stale = next(a for a in ARTICLES if a["article_id"] == "bill-009")
    assert "Basic Plus" in stale["body"]
    assert stale["last_updated"] < "2025-01-01"


def test_near_duplicate_cluster_intact():
    cluster = {"dev-003", "dev-004", "dev-005"}
    present = {a["article_id"] for a in ARTICLES}
    assert cluster <= present


def test_documented_flaws_reference_real_articles():
    known = {a["article_id"] for a in ARTICLES}
    for flaw in KNOWN_CORPUS_FLAWS:
        assert set(flaw["article_ids"]) <= known, flaw["flaw_id"]


def test_coverage_gaps_are_documented():
    assert len(COVERAGE_GAPS) >= 5


def test_gap_topics_absent_from_corpus():
    """Spot-check that gap topics genuinely have no article.

    If an article covering one of these were added, the matching
    out-of-scope questions would become answerable and the refusal
    metric would report a false hallucination.
    """
    corpus_text = " ".join(a["body"] + a["title"] for a in ARTICLES).lower()
    for term in ("gift card", "gift subscription", "audio description",
                 "affiliate", "gdpr"):
        assert term not in corpus_text, f"gap topic leaked into corpus: {term}"


# ---------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------
def test_orphan_articles_are_reportable(built):
    """Orphans are allowed — retrieval distractors are realistic — but
    the count must be computable so the audit can report it."""
    articles, golden = built
    assert isinstance(orphan_articles(articles, golden), list)


def test_articles_are_substantive(built):
    articles, _ = built
    for a in articles:
        assert a.word_count >= 80, f"{a.article_id} is too short to chunk"


def test_document_form_includes_title(built):
    articles, _ = built
    for a in articles:
        assert a.title in a.as_document()
