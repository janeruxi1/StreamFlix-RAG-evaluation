"""Tests for embeddings and the vector store.

These run on the TF-IDF/SVD fallback so CI needs no model download. The
transformer path is exercised by the same interface contract — tests
here assert on the Embedder ABC, not on a specific backend.
"""
import numpy as np
import pytest

from src.corpus.build import load_corpus
from src.retrieval.chunking import markdown_section, whole_article
from src.retrieval.embedding import (
    EmbeddingCache,
    TfidfSvdEmbedder,
    encode_cached,
    get_embedder,
    transformer_available,
)
from src.retrieval.vectorstore import VectorStore, hits_to_articles


@pytest.fixture(scope="module")
def articles():
    return load_corpus()


@pytest.fixture(scope="module")
def chunks(articles):
    return markdown_section(articles)


@pytest.fixture(scope="module")
def fitted(chunks):
    texts = [c.embed_text for c in chunks]
    return TfidfSvdEmbedder(n_components=64).fit(texts)


@pytest.fixture(scope="module")
def store(chunks, fitted):
    vectors = fitted.encode([c.embed_text for c in chunks])
    return VectorStore().add(chunks, vectors)


# ---------------------------------------------------------------------
# Embedder contract
# ---------------------------------------------------------------------
def test_embeddings_are_l2_normalised(fitted, chunks):
    """Cosine similarity reduces to a dot product only if this holds.
    Without it, the vector store silently returns wrong rankings."""
    vectors = fitted.encode([c.embed_text for c in chunks[:20]])
    norms = np.linalg.norm(vectors, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5)


def test_encode_shape_matches_input(fitted, chunks):
    vectors = fitted.encode([c.embed_text for c in chunks[:10]])
    assert vectors.shape == (10, fitted.dimension)


def test_encoding_is_deterministic(fitted):
    a = fitted.encode(["how do I cancel my subscription"])
    b = fitted.encode(["how do I cancel my subscription"])
    assert np.allclose(a, b)


def test_unfitted_tfidf_embedder_raises_clearly(chunks):
    """Silent garbage vectors would be far worse than an exception."""
    with pytest.raises(RuntimeError, match="fitted"):
        TfidfSvdEmbedder().encode(["anything"])


def test_semantically_related_text_scores_higher(fitted):
    """Minimum bar for calling it an embedding at all."""
    vectors = fitted.encode([
        "How do I cancel my subscription?",
        "How do I end my subscription?",
        "What resolution does 4K streaming use?",
    ])
    similar = float(vectors[0] @ vectors[1])
    unrelated = float(vectors[0] @ vectors[2])
    assert similar > unrelated


def test_svd_components_capped_by_matrix_rank(articles):
    """Asking for more components than the corpus can support must
    degrade gracefully, not crash."""
    tiny = [a.body for a in articles[:5]]
    embedder = TfidfSvdEmbedder(n_components=512).fit(tiny)
    assert embedder.dimension < 512
    assert embedder.encode(["test"]).shape == (1, embedder.dimension)


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
def test_explicit_backend_is_honoured():
    assert isinstance(get_embedder(backend="tfidf_svd"), TfidfSvdEmbedder)


def test_unknown_backend_raises():
    with pytest.raises(ValueError, match="Unknown embedding backend"):
        get_embedder(backend="word2vec")


def test_env_var_selects_backend(monkeypatch):
    monkeypatch.setenv("EMBEDDING_BACKEND", "tfidf_svd")
    assert isinstance(get_embedder(), TfidfSvdEmbedder)


def test_factory_falls_back_without_transformer(monkeypatch):
    """A fresh clone must get a working system before installing torch."""
    monkeypatch.delenv("EMBEDDING_BACKEND", raising=False)
    monkeypatch.setattr(
        "src.retrieval.embedding.transformer_available", lambda: False
    )
    assert isinstance(get_embedder(), TfidfSvdEmbedder)


def test_transformer_available_returns_bool():
    assert isinstance(transformer_available(), bool)


# ---------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------
def test_cache_roundtrip(tmp_path, fitted):
    cache = EmbeddingCache(cache_dir=tmp_path)
    texts = ["hello world"]
    vectors = fitted.encode(texts)
    assert cache.get("tfidf_svd", "m", texts) is None
    cache.put("tfidf_svd", "m", texts, vectors)
    assert np.allclose(cache.get("tfidf_svd", "m", texts), vectors)


def test_cache_key_depends_on_text(tmp_path, fitted):
    """Different inputs must not collide — a collision would return
    another query's embeddings with no error."""
    cache = EmbeddingCache(cache_dir=tmp_path)
    cache.put("tfidf_svd", "m", ["a"], fitted.encode(["a"]))
    assert cache.get("tfidf_svd", "m", ["b"]) is None


def test_cache_key_depends_on_backend(tmp_path, fitted):
    """Switching backends must invalidate — otherwise a bake-off would
    compare a model against its own cached rival."""
    cache = EmbeddingCache(cache_dir=tmp_path)
    cache.put("tfidf_svd", "m", ["a"], fitted.encode(["a"]))
    assert cache.get("sentence_transformer", "m", ["a"]) is None


def test_disabled_cache_is_a_no_op(tmp_path, fitted):
    cache = EmbeddingCache(cache_dir=tmp_path, enabled=False)
    cache.put("tfidf_svd", "m", ["a"], fitted.encode(["a"]))
    assert cache.get("tfidf_svd", "m", ["a"]) is None


def test_encode_cached_matches_direct_encode(tmp_path, fitted):
    cache = EmbeddingCache(cache_dir=tmp_path)
    texts = ["how do I cancel"]
    direct = fitted.encode(texts)
    assert np.allclose(encode_cached(fitted, texts, cache), direct)
    assert np.allclose(encode_cached(fitted, texts, cache), direct)  # hit


# ---------------------------------------------------------------------
# VectorStore
# ---------------------------------------------------------------------
def test_empty_store_raises_rather_than_returning_nothing():
    with pytest.raises(RuntimeError, match="empty"):
        VectorStore().search(np.zeros(64))


def test_add_rejects_length_mismatch(chunks):
    with pytest.raises(ValueError, match="mismatch"):
        VectorStore().add(chunks, np.zeros((len(chunks) - 1, 64)))


def test_search_returns_k_results_ranked(store, fitted):
    hits = store.search(fitted.encode(["cancel subscription"])[0], top_k=5)
    assert len(hits) == 5
    assert [h.rank for h in hits] == [0, 1, 2, 3, 4]
    scores = [h.score for h in hits]
    assert scores == sorted(scores, reverse=True)


def test_top_k_larger_than_corpus_is_clamped(fitted, articles):
    chunks = whole_article(articles[:3])
    vectors = fitted.encode([c.embed_text for c in chunks])
    store = VectorStore().add(chunks, vectors)
    assert len(store.search(vectors[0], top_k=99)) == 3


def test_self_retrieval_is_exact(store, chunks, fitted):
    """A chunk's own vector must rank that chunk first with score ~1.
    This is the single strongest signal that indexing is not scrambled."""
    for i in (0, 25, 60):
        hits = store.search(fitted.encode([chunks[i].embed_text])[0], top_k=1)
        assert hits[0].chunk.chunk_id == chunks[i].chunk_id
        assert hits[0].score == pytest.approx(1.0, abs=1e-4)


def test_search_batch_matches_single_search(store, fitted):
    """The batched path is used for the whole Phase 3 sweep, so it must
    be provably identical to the single-query path."""
    queries = ["cancel subscription", "4K streaming", "refund window"]
    vectors = fitted.encode(queries)
    batched = store.search_batch(vectors, top_k=5)
    for query_vector, batch_hits in zip(vectors, batched):
        single = store.search(query_vector, top_k=5)
        assert [h.chunk.chunk_id for h in single] == \
               [h.chunk.chunk_id for h in batch_hits]
        assert [h.score for h in single] == \
               pytest.approx([h.score for h in batch_hits], abs=1e-5)


def test_store_reports_size_and_dimension(store, chunks, fitted):
    assert len(store) == len(chunks)
    assert store.dimension == fitted.dimension


# ---------------------------------------------------------------------
# Article aggregation
# ---------------------------------------------------------------------
def test_hits_to_articles_deduplicates_preserving_rank(store, fitted):
    hits = store.search(fitted.encode(["how do I cancel"])[0], top_k=5)
    articles_out = hits_to_articles(hits)
    assert len(articles_out) == len(set(articles_out))
    # first article must come from the top-ranked chunk
    assert articles_out[0] == hits[0].article_id


def test_hits_to_articles_respects_max(store, fitted):
    hits = store.search(fitted.encode(["streaming quality"])[0], top_k=10)
    assert len(hits_to_articles(hits, max_articles=3)) <= 3


def test_article_count_never_exceeds_chunk_count(store, fitted):
    """The dedup effect that makes 'top-5 chunks' and 'top-5 articles'
    different quantities — Phase 3 must not conflate them."""
    hits = store.search(fitted.encode(["roku setup steps"])[0], top_k=5)
    assert len(hits_to_articles(hits)) <= len(hits)


# ---------------------------------------------------------------------
# End-to-end sanity
# ---------------------------------------------------------------------
def test_retrieval_beats_random_on_the_golden_set(store, fitted):
    """Guards against a pipeline that runs cleanly but retrieves noise —
    the failure mode where every unit test passes and the system is
    still useless."""
    from src.corpus.build import load_golden_set

    golden = [q for q in load_golden_set()
              if q["category"] == "single_hop"][:40]
    vectors = fitted.encode([q["question"] for q in golden])
    results = store.search_batch(vectors, top_k=15)

    hits = sum(
        bool(set(q["gt_article_ids"]) & set(hits_to_articles(r, 5)))
        for q, r in zip(golden, results)
    )
    accuracy = hits / len(golden)
    # random 5-of-45 would be ~11%; anything near that means broken
    assert accuracy > 0.5, f"hit@5 was {accuracy:.1%} — retrieval is broken"
