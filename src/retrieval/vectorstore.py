"""Vector store — exact cosine search over the chunked corpus.

Why numpy rather than Chroma / FAISS / pgvector
-----------------------------------------------
This corpus produces roughly 50-400 chunks depending on strategy. At
that scale an approximate-nearest-neighbour index is the wrong tool:

  - Exact search over 400 x 384 floats is a single matrix multiply,
    well under a millisecond. An ANN index cannot beat that; it can only
    approximate it while adding build time.
  - ANN indexes trade recall for speed. Accepting *any* recall loss to
    speed up a sub-millisecond operation would be a strictly bad trade,
    and it would contaminate the Phase 3 retrieval measurements with
    index error.
  - Every extra dependency is a reproducibility risk for anyone cloning
    the repo.

The honest engineering answer is that ANN indexes earn their complexity
somewhere north of ~100k vectors. Below that, exact search is faster,
simpler, and exactly correct.

`VectorStore` deliberately mirrors the interface a real vector database
exposes (`add`, `search`, `search_batch`), so swapping in Chroma or
pgvector later is a backend change rather than a rewrite.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.retrieval.chunking import Chunk


@dataclass(frozen=True)
class SearchHit:
    """One retrieved chunk with its similarity score."""
    chunk: Chunk
    score: float
    rank: int

    @property
    def article_id(self) -> str:
        return self.chunk.article_id


class VectorStore:
    """Exact cosine-similarity search over embedded chunks.

    Vectors are expected L2-normalised, so cosine similarity reduces to a
    dot product and the whole search is one matrix multiply.
    """

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._matrix: np.ndarray | None = None

    def add(self, chunks: list[Chunk], vectors: np.ndarray) -> "VectorStore":
        """Index chunks alongside their embeddings."""
        if len(chunks) != len(vectors):
            raise ValueError(
                f"chunk/vector count mismatch: {len(chunks)} vs {len(vectors)}"
            )
        self._chunks = list(chunks)
        self._matrix = np.ascontiguousarray(vectors, dtype=np.float32)
        return self

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def dimension(self) -> int:
        return 0 if self._matrix is None else int(self._matrix.shape[1])

    def search(self, query_vector: np.ndarray,
               top_k: int = 5) -> list[SearchHit]:
        """Top-k most similar chunks to a single query vector."""
        if self._matrix is None:
            raise RuntimeError("VectorStore is empty — call add() first")

        query = np.asarray(query_vector, dtype=np.float32).ravel()
        scores = self._matrix @ query

        k = min(top_k, len(self._chunks))
        # argpartition finds the top-k without fully sorting, then we
        # sort only that slice.
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]

        return [
            SearchHit(chunk=self._chunks[i], score=float(scores[i]), rank=r)
            for r, i in enumerate(top_idx)
        ]

    def search_batch(self, query_vectors: np.ndarray,
                     top_k: int = 5) -> list[list[SearchHit]]:
        """Vectorised search for many queries at once.

        One matrix multiply for the whole golden set — the Phase 3
        bake-off runs this hundreds of times across configurations, so
        batching matters more than it looks.
        """
        if self._matrix is None:
            raise RuntimeError("VectorStore is empty — call add() first")

        queries = np.asarray(query_vectors, dtype=np.float32)
        scores = queries @ self._matrix.T          # (n_queries, n_chunks)

        k = min(top_k, len(self._chunks))
        results: list[list[SearchHit]] = []
        for row in scores:
            top_idx = np.argpartition(-row, k - 1)[:k]
            top_idx = top_idx[np.argsort(-row[top_idx])]
            results.append([
                SearchHit(chunk=self._chunks[i], score=float(row[i]), rank=r)
                for r, i in enumerate(top_idx)
            ])
        return results


# ---------------------------------------------------------------------
# Article-level aggregation
# ---------------------------------------------------------------------
def hits_to_articles(hits: list[SearchHit],
                     max_articles: int | None = None) -> list[str]:
    """Collapse chunk hits to a deduplicated, rank-ordered article list.

    Ground truth in the golden set is expressed as article ids, but
    retrieval returns chunks — and a strategy producing many small chunks
    can fill top-k with several pieces of the same article.

    Deduplicating by first appearance means "top-5 chunks" and "top-5
    articles" are different quantities, which is exactly the effect
    Phase 3 needs to measure rather than accidentally hide.
    """
    seen: list[str] = []
    for hit in hits:
        if hit.article_id not in seen:
            seen.append(hit.article_id)
    return seen[:max_articles] if max_articles else seen
