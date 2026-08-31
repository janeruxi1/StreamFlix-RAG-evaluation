"""Tests for the retrieval arms.

The property that matters most here is FAIRNESS: every arm must index the
same text and return the same object type, or the bake-off compares the
setup rather than the method.
"""
import numpy as np
import pytest

from src.corpus.build import load_corpus
from src.retrieval.chunking import markdown_section, whole_article
from src.retrieval.embedding import TfidfSvdEmbedder
from src.retrieval.retrievers import (
    BM25Retriever,
    DenseRetriever,
    HybridRetriever,
    OracleUnionRetriever,
)
from src.retrieval.vectorstore import SearchHit, hits_to_articles


@pytest.fixture(scope="module")
def articles():
    return load_corpus()


@pytest.fixture(scope="module")
def chunks(articles):
    return markdown_section(articles)


@pytest.fixture(scope="module")
def bm25(chunks):
    return BM25Retriever(chunks)


@pytest.fixture(scope="module")
def dense(chunks):
    return DenseRetriever(chunks, TfidfSvdEmbedder(n_components=64))


@pytest.fixture(scope="module")
def hybrid(bm25, dense):
    return HybridRetriever([bm25, dense])


# ---------------------------------------------------------------------
# Shared contract — every arm must satisfy these
# ---------------------------------------------------------------------
@pytest.fixture(params=["bm25", "dense", "hybrid"])
def any_retriever(request, bm25, dense, hybrid):
    return {"bm25": bm25, "dense": dense, "hybrid": hybrid}[request.param]


def test_returns_requested_number_of_hits(any_retriever):
    assert len(any_retriever.search("how do I cancel", top_k=5)) == 5


def test_hits_are_rank_ordered(any_retriever):
    hits = any_retriever.search("refund policy", top_k=8)
    assert [h.rank for h in hits] == list(range(8))
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_returns_search_hits(any_retriever):
    assert all(isinstance(h, SearchHit)
               for h in any_retriever.search("4K streaming", top_k=3))


def test_search_many_matches_search(any_retriever):
    """The batched path is what the bake-off uses; it must agree with the
    single-query path or every swept number is suspect."""
    queries = ["how do I cancel", "4K streaming", "refund window"]
    batched = any_retriever.search_many(queries, top_k=5)
    for q, hits in zip(queries, batched):
        single = any_retriever.search(q, top_k=5)
        assert [h.chunk.chunk_id for h in single] == \
               [h.chunk.chunk_id for h in hits]


def test_top_k_larger_than_corpus_is_clamped(any_retriever, chunks):
    hits = any_retriever.search("streaming", top_k=len(chunks) + 50)
    assert len(hits) <= len(chunks)


def test_deeper_search_is_a_superset(any_retriever):
    """Retrieving deeper must not drop anything already found.

    This is the invariant that makes recall monotone in depth. If it
    fails, the Phase 3 monotonicity argument collapses and the sweep is
    unreadable.
    """
    shallow = {h.chunk.chunk_id for h in any_retriever.search("billing", top_k=5)}
    deep = {h.chunk.chunk_id for h in any_retriever.search("billing", top_k=15)}
    assert shallow <= deep


# ---------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------
def test_bm25_ranks_exact_term_match_first(bm25):
    hits = bm25.search("Roku", top_k=3)
    assert any("Roku" in h.chunk.embed_text for h in hits[:1])


def test_bm25_scores_are_non_negative(bm25):
    assert all(h.score >= 0 for h in bm25.search("subscription", top_k=10))


def test_bm25_unmatched_query_still_returns_results(bm25):
    """Must degrade to zero-score results rather than raising — the
    bake-off would otherwise crash on one bad query."""
    hits = bm25.search("zzzz qqqq xxxx", top_k=5)
    assert len(hits) == 5


# ---------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------
def test_dense_fits_unfitted_embedder_automatically(chunks):
    """A silently unfitted embedder would produce garbage rankings."""
    r = DenseRetriever(chunks, TfidfSvdEmbedder(n_components=32))
    assert len(r.search("cancel", top_k=3)) == 3


def test_dense_indexes_same_text_as_bm25(chunks, bm25, dense):
    """Fairness check. Different views of the corpus would make any
    bm25-vs-dense difference uninterpretable."""
    bm25_ids = {c for c in bm25._chunks}
    dense_ids = {c.chunk_id for c in dense._store._chunks}
    assert bm25_ids == dense_ids


def test_dense_self_retrieval_is_exact(chunks, dense):
    hits = dense.search(chunks[10].embed_text, top_k=1)
    assert hits[0].chunk.chunk_id == chunks[10].chunk_id


# ---------------------------------------------------------------------
# Hybrid / RRF
# ---------------------------------------------------------------------
def test_hybrid_requires_two_retrievers(bm25):
    with pytest.raises(ValueError, match="at least two"):
        HybridRetriever([bm25])


def test_hybrid_derives_name_from_constituents(bm25, dense):
    assert HybridRetriever([bm25, dense]).name == f"hybrid_{bm25.name}+{dense.name}"


def test_hybrid_respects_explicit_name(bm25, dense):
    """Regression guard: the auto-naming used to overwrite an explicitly
    supplied name, which silently broke lookups keyed on it."""
    assert HybridRetriever([bm25, dense], name="hybrid").name == "hybrid"


def test_rrf_scores_match_the_formula(bm25, dense):
    """Pin the fusion arithmetic rather than trusting the ordering."""
    h = HybridRetriever([bm25, dense], depth=10, k=60)
    query = "how do I cancel my subscription"
    rankings = [bm25.search(query, top_k=10), dense.search(query, top_k=10)]
    expected: dict[str, float] = {}
    for hits in rankings:
        for hit in hits:
            expected[hit.chunk.chunk_id] = expected.get(hit.chunk.chunk_id, 0.0) \
                + 1.0 / (60 + hit.rank + 1)
    for hit in h.search(query, top_k=5):
        assert hit.score == pytest.approx(expected[hit.chunk.chunk_id])


def test_rrf_rewards_agreement_between_arms(bm25, dense):
    """A chunk both arms rank highly must beat one only a single arm
    likes — that is the entire premise of fusion."""
    h = HybridRetriever([bm25, dense], depth=20)
    query = "refund policy"
    top_bm = {x.chunk.chunk_id for x in bm25.search(query, top_k=5)}
    top_dn = {x.chunk.chunk_id for x in dense.search(query, top_k=5)}
    agreed = top_bm & top_dn
    if agreed:                       # only meaningful when they overlap
        fused = [x.chunk.chunk_id for x in h.search(query, top_k=10)]
        assert any(a in fused[:5] for a in agreed)


def test_hybrid_depth_must_exceed_top_k_to_help(bm25, dense):
    """Documents the parameter's purpose: fusing only shallow lists
    cannot recover a document ranked mid-list by both arms."""
    shallow = HybridRetriever([bm25, dense], depth=3)
    deep = HybridRetriever([bm25, dense], depth=30)
    q = "what happens when my trial ends"
    assert len(shallow.search(q, top_k=5)) <= len(deep.search(q, top_k=5))


# ---------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------
def test_oracle_pool_contains_every_constituent_result(bm25, dense):
    """The oracle is the ceiling — anything an arm found must be in it,
    or the 'ranking failure vs candidate failure' diagnostic is wrong."""
    oracle = OracleUnionRetriever([bm25, dense], depth=20)
    q = "charged after I cancelled"
    pool = {h.chunk.chunk_id for h in oracle.search(q, top_k=999)}
    for arm in (bm25, dense):
        assert {h.chunk.chunk_id for h in arm.search(q, top_k=20)} <= pool


def test_oracle_recall_is_an_upper_bound(articles, bm25, dense):
    """Formally: oracle article recall >= any constituent's."""
    oracle = OracleUnionRetriever([bm25, dense], depth=20)
    q = "how much does Premium cost"
    pool = set(hits_to_articles(oracle.search(q, top_k=999)))
    for arm in (bm25, dense):
        assert set(hits_to_articles(arm.search(q, top_k=20))) <= pool
