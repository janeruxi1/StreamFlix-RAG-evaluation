"""Tests for chunking strategies.

The property that matters most: no strategy may lose text or lose the
article_id link. Losing text silently caps recall at a ceiling no amount
of retrieval tuning can lift, and losing article_id makes every Phase 3
metric wrong while still producing plausible numbers.
"""
import pytest

from src.corpus.build import load_corpus
from src.retrieval.chunking import (
    STRATEGIES,
    Chunk,
    fixed_token,
    markdown_section,
    profile_strategy,
    sentence_window,
    whole_article,
)


@pytest.fixture(scope="module")
def articles():
    return load_corpus()


# ---------------------------------------------------------------------
# Chunk dataclass
# ---------------------------------------------------------------------
def test_embed_text_prepends_title():
    c = Chunk("a#0", "a", "Refund policy", "billing", "Body text.")
    assert c.embed_text.startswith("Refund policy")
    assert "Body text." in c.embed_text


def test_generation_text_defaults_to_text():
    c = Chunk("a#0", "a", "T", "billing", "Body.")
    assert c.generation_text == "Body."


def test_generation_text_uses_context_when_present():
    c = Chunk("a#0", "a", "T", "billing", "Body.",
              context_text="Before. Body. After.")
    assert c.generation_text == "Before. Body. After."
    assert c.text == "Body."          # embed text stays narrow


def test_chunk_is_immutable():
    c = Chunk("a#0", "a", "T", "billing", "Body.")
    with pytest.raises(Exception):
        c.text = "mutated"


# ---------------------------------------------------------------------
# Universal invariants — every strategy must satisfy these
# ---------------------------------------------------------------------
@pytest.mark.parametrize("name", list(STRATEGIES))
def test_every_article_is_represented(name, articles):
    """No article may vanish. An unchunked article is unretrievable."""
    chunks = STRATEGIES[name](articles)
    covered = {c.article_id for c in chunks}
    missing = {a.article_id for a in articles} - covered
    assert not missing, f"{name} dropped articles: {sorted(missing)}"


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_chunk_ids_are_unique(name, articles):
    """Duplicate ids would silently collapse rows in Phase 3 joins."""
    ids = [c.chunk_id for c in STRATEGIES[name](articles)]
    assert len(ids) == len(set(ids))


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_no_empty_chunks(name, articles):
    """Empty text embeds to a meaningless vector that still gets ranked."""
    for c in STRATEGIES[name](articles):
        assert c.text.strip(), f"{name} produced empty chunk {c.chunk_id}"


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_metadata_matches_source_article(name, articles):
    """article_id/title/category must survive chunking — ground truth
    in the golden set is keyed on article_id."""
    by_id = {a.article_id: a for a in articles}
    for c in STRATEGIES[name](articles):
        src = by_id[c.article_id]
        assert c.title == src.title
        assert c.category == src.category


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_strategies_are_deterministic(name, articles):
    """Two runs must be identical, or cached embeddings go stale
    without any signal."""
    first = STRATEGIES[name](articles)
    second = STRATEGIES[name](articles)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


@pytest.mark.parametrize("name", list(STRATEGIES))
def test_chunks_fit_embedding_window(name, articles):
    """Anything over 512 tokens gets silently truncated by the encoder —
    a data-loss bug that produces no error."""
    oversized = [c for c in STRATEGIES[name](articles) if c.n_tokens > 512]
    assert not oversized, f"{name}: {len(oversized)} chunks exceed 512 tokens"


# ---------------------------------------------------------------------
# Strategy-specific behaviour
# ---------------------------------------------------------------------
def test_whole_article_is_one_per_article(articles):
    chunks = whole_article(articles)
    assert len(chunks) == len(articles)


def test_whole_article_preserves_body_exactly(articles):
    """The no-split arm must genuinely not split."""
    by_id = {a.article_id: a for a in articles}
    for c in whole_article(articles):
        assert c.text == by_id[c.article_id].body.strip()


def test_markdown_section_splits_multi_section_articles(articles):
    """If this collapsed to one chunk each it would be whole_article
    wearing a different name."""
    chunks = markdown_section(articles)
    assert len(chunks) > len(articles)


def test_markdown_section_merges_tiny_fragments(articles):
    """Sub-threshold sections merge forward rather than embedding to noise.

    Only the FIRST section of an article may fall below the threshold —
    it has nothing to merge backwards into.
    """
    chunks = markdown_section(articles, min_tokens=20)
    undersized = [c for c in chunks if c.n_tokens < 20 and c.position > 0]
    assert not undersized, \
        f"unmerged tiny chunks: {[c.chunk_id for c in undersized]}"


def test_fixed_token_respects_size_budget(articles):
    chunks = fixed_token(articles, chunk_tokens=128, overlap_tokens=32)
    assert max(c.n_tokens for c in chunks) <= 200   # 128 tokens + title


def test_fixed_token_rejects_overlap_larger_than_chunk(articles):
    with pytest.raises(ValueError):
        fixed_token(articles, chunk_tokens=64, overlap_tokens=64)


def test_smaller_chunk_size_produces_more_chunks(articles):
    """Sanity check on the core parameter — if this failed, the size
    argument would be doing nothing."""
    small = fixed_token(articles, chunk_tokens=64, overlap_tokens=16)
    large = fixed_token(articles, chunk_tokens=256, overlap_tokens=64)
    assert len(small) > len(large)


def test_sentence_window_attaches_context(articles):
    """The whole point of the strategy: embed narrow, generate wide."""
    chunks = sentence_window(articles, window=2)
    assert all(c.context_text is not None for c in chunks)
    wider = [c for c in chunks if len(c.context_text) > len(c.text)]
    assert len(wider) > 0.8 * len(chunks)


def test_sentence_window_context_contains_the_sentence(articles):
    for c in sentence_window(articles, window=2)[:50]:
        assert c.text in c.context_text


def test_sentence_window_is_the_finest_strategy(articles):
    """Ordering check across strategies — catches a splitter that
    silently stopped splitting."""
    counts = {n: len(fn(articles)) for n, fn in STRATEGIES.items()}
    assert counts["sentence_window"] == max(counts.values())
    assert counts["whole_article"] == min(counts.values())


# ---------------------------------------------------------------------
# The degenerate-arm finding from the Phase 2 notebook
# ---------------------------------------------------------------------
def test_fixed_token_256_is_degenerate_at_this_corpus_size(articles):
    """Documents a real finding rather than asserting a design intent.

    Every article is shorter than 256 tokens, so a 256-token window never
    splits anything and the arm collapses into whole_article. The Phase 2
    notebook detects this dynamically and drops the arm; this test pins
    the fact so that if the corpus later grows and the arm stops being
    degenerate, the failure prompts a re-read of that notebook section
    rather than passing unnoticed.
    """
    assert len(STRATEGIES["fixed_token_256"](articles)) == len(articles)


def test_profile_strategy_reports_full_coverage(articles):
    chunks = markdown_section(articles)
    profile = profile_strategy("markdown_section", chunks, len(articles))
    assert profile.articles_covered == len(articles)
    assert profile.n_chunks == len(chunks)
    assert profile.min_tokens <= profile.median_tokens <= profile.max_tokens
