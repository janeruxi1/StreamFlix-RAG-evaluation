"""Chunking strategies for the StreamFlix help corpus.

Phase 1 established that every article fits inside a 512-token embedding
window (longest is well under the limit). That matters: we are never
FORCED to chunk. Splitting is therefore a genuine experimental choice,
and `whole_article` is a legitimate arm in the Phase 3 bake-off rather
than a strawman.

The trade-off each strategy navigates:

    Larger chunks  -> more context per hit, but the embedding is an
                      average over more topics, so it matches queries
                      less sharply (topic dilution).
    Smaller chunks -> sharper topical match, but an answer spanning two
                      chunks needs both retrieved, and each chunk carries
                      less context for the generator to ground on.

Four strategies, each a different point on that trade-off:

    whole_article    No split. Maximum context, minimum precision.
    markdown_section Split on the article's own '**Heading**' structure.
                     Respects author-intended boundaries.
    fixed_token      Fixed-size windows with overlap. The generic default
                     most RAG tutorials use.
    sentence_window  One sentence per chunk for matching, but neighbours
                     attached for context. Sharp retrieval, wide context.

All strategies return `Chunk` objects carrying their source article_id,
which is what lets Phase 3 score retrieval against ground truth.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Callable, Iterable

from src.corpus.build import Article
from src.corpus.difficulty import count_tokens


@dataclass(frozen=True)
class Chunk:
    """A retrievable unit of text.

    `article_id` is the ground-truth link — retrieval is scored by
    whether a chunk from the correct article was returned, so this field
    must survive every transformation.
    """
    chunk_id: str
    article_id: str
    title: str
    category: str
    text: str
    position: int = 0            # index of this chunk within its article
    context_text: str | None = None   # wider text passed to the generator

    @property
    def embed_text(self) -> str:
        """Text used to build the embedding.

        The title is prepended because it carries topical signal the body
        often assumes ("Steps" means nothing without "Setting up Roku").
        """
        return f"{self.title}\n\n{self.text}"

    @property
    def generation_text(self) -> str:
        """Text handed to the generator — wider than the embed text when
        the strategy attaches surrounding context."""
        return self.context_text if self.context_text is not None else self.text

    @property
    def n_tokens(self) -> int:
        return count_tokens(self.embed_text)


# ---------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------
def whole_article(articles: Iterable[Article]) -> list[Chunk]:
    """One chunk per article. No splitting.

    Viable here only because Phase 1 confirmed every article fits the
    embedding window. On a corpus of long documents this arm would be
    invalid rather than merely weak.
    """
    return [
        Chunk(
            chunk_id=f"{a.article_id}#whole",
            article_id=a.article_id,
            title=a.title,
            category=a.category,
            text=a.body.strip(),
            position=0,
        )
        for a in articles
    ]


def _split_markdown_sections(body: str) -> list[str]:
    """Split on the corpus's '**Bold heading**' convention.

    Content before the first heading becomes its own section so no text
    is silently dropped.
    """
    parts = re.split(r"\n(?=\*\*[^*\n]+\*\*\s*\n)", body.strip())
    return [p.strip() for p in parts if p.strip()]


def markdown_section(articles: Iterable[Article],
                     min_tokens: int = 20) -> list[Chunk]:
    """Split on author-authored headings.

    Rationale: the person who wrote the article already decided where the
    topic boundaries are. Fixed-size windows ignore that and routinely cut
    mid-table or mid-list.

    Sections below `min_tokens` are merged forward — a two-line heading
    stub embeds to noise.
    """
    chunks: list[Chunk] = []
    for a in articles:
        sections = _split_markdown_sections(a.body)

        merged: list[str] = []
        for section in sections:
            if merged and count_tokens(section) < min_tokens:
                merged[-1] = merged[-1] + "\n\n" + section
            else:
                merged.append(section)

        for i, section in enumerate(merged):
            chunks.append(Chunk(
                chunk_id=f"{a.article_id}#sec{i}",
                article_id=a.article_id,
                title=a.title,
                category=a.category,
                text=section,
                position=i,
            ))
    return chunks


def fixed_token(articles: Iterable[Article],
                chunk_tokens: int = 128,
                overlap_tokens: int = 32) -> list[Chunk]:
    """Fixed-size windows with overlap, measured in words as a proxy.

    The generic default in most RAG stacks. Included because it is the
    thing a reviewer will expect to see compared against, and because it
    is a fair test of whether structure-aware splitting actually earns
    its complexity.

    Overlap exists so a fact straddling a boundary appears whole in at
    least one chunk.
    """
    if overlap_tokens >= chunk_tokens:
        raise ValueError("overlap_tokens must be smaller than chunk_tokens")

    chunks: list[Chunk] = []
    for a in articles:
        words = a.body.split()
        # ~1.33 tokens per word for English prose; converting here keeps
        # the public API in tokens, which is how chunk size is normally
        # specified, while the split itself stays cheap.
        step_words = max(1, int((chunk_tokens - overlap_tokens) / 1.33))
        window_words = max(1, int(chunk_tokens / 1.33))

        position = 0
        for start in range(0, len(words), step_words):
            window = words[start:start + window_words]
            if not window:
                break
            # Skip a trailing fragment already fully covered by overlap
            if start > 0 and len(window) <= overlap_tokens / 1.33:
                break
            chunks.append(Chunk(
                chunk_id=f"{a.article_id}#tok{position}",
                article_id=a.article_id,
                title=a.title,
                category=a.category,
                text=" ".join(window),
                position=position,
            ))
            position += 1
            if start + window_words >= len(words):
                break
    return chunks


def _split_sentences(text: str) -> list[str]:
    """Lightweight sentence splitter.

    Avoids a spacy/nltk dependency. Protects the corpus's markdown
    artifacts (tables, bullets) by treating a line starting with '|' or
    '-' as its own unit rather than splitting it on periods.
    """
    units: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("|", "-", "*", "#")) or stripped.endswith(":"):
            units.append(stripped)
            continue
        for sentence in re.split(r"(?<=[.!?])\s+(?=[A-Z])", stripped):
            if sentence.strip():
                units.append(sentence.strip())
    return units


def sentence_window(articles: Iterable[Article],
                    window: int = 2) -> list[Chunk]:
    """Embed one sentence, but hand the generator its neighbours.

    Decouples the two jobs a chunk normally does at once:
      - matching a query wants SMALL, topically sharp text
      - grounding an answer wants LARGE, contextually complete text

    The embedding sees a single sentence; `context_text` carries the
    surrounding `window` sentences on each side. Retrieval precision
    without starving the generator.
    """
    chunks: list[Chunk] = []
    for a in articles:
        sentences = _split_sentences(a.body)
        for i, sentence in enumerate(sentences):
            lo = max(0, i - window)
            hi = min(len(sentences), i + window + 1)
            chunks.append(Chunk(
                chunk_id=f"{a.article_id}#sent{i}",
                article_id=a.article_id,
                title=a.title,
                category=a.category,
                text=sentence,
                position=i,
                context_text=" ".join(sentences[lo:hi]),
            ))
    return chunks


# ---------------------------------------------------------------------
# Registry — Phase 3 iterates over this
# ---------------------------------------------------------------------
ChunkStrategy = Callable[..., list[Chunk]]

STRATEGIES: dict[str, ChunkStrategy] = {
    "whole_article": whole_article,
    "markdown_section": markdown_section,
    "fixed_token_128": lambda arts: fixed_token(arts, 128, 32),
    "fixed_token_256": lambda arts: fixed_token(arts, 256, 64),
    "sentence_window": sentence_window,
}


@dataclass(frozen=True)
class ChunkProfile:
    """Summary statistics for one chunking strategy."""
    strategy: str
    n_chunks: int
    chunks_per_article: float
    min_tokens: int
    median_tokens: int
    max_tokens: int
    total_tokens: int
    articles_covered: int

    def __str__(self) -> str:
        return (f"{self.strategy:<18} {self.n_chunks:>5} chunks  "
                f"{self.chunks_per_article:>5.1f}/article  "
                f"tokens: {self.min_tokens:>4}/{self.median_tokens:>4}/"
                f"{self.max_tokens:>4} (min/med/max)")


def profile_strategy(name: str, chunks: list[Chunk],
                     n_articles: int) -> ChunkProfile:
    """Compute size statistics for a set of chunks."""
    token_counts = sorted(c.n_tokens for c in chunks)
    return ChunkProfile(
        strategy=name,
        n_chunks=len(chunks),
        chunks_per_article=len(chunks) / n_articles,
        min_tokens=token_counts[0],
        median_tokens=token_counts[len(token_counts) // 2],
        max_tokens=token_counts[-1],
        total_tokens=sum(token_counts),
        articles_covered=len({c.article_id for c in chunks}),
    )
