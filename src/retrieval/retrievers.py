"""Retrievers behind one interface, so the bake-off compares like with like.

Phase 1 measured BM25 over whole ARTICLES. Phase 2 measured dense
retrieval over CHUNKS. Those numbers are not directly comparable — the
unit of retrieval differed — which is fine for establishing a floor but
useless for choosing a configuration.

This module fixes that. Every retriever here takes the same chunk list,
returns the same `SearchHit` objects, and is scored by the same code. The
only thing that varies is the ranking function, which is the thing under
test.

Three families:

    BM25Retriever      Lexical. Exact term matching, no training, no
                       embedding. Phase 1 showed it is strong on this
                       corpus (93% recall@5 on single-hop), so it is a
                       first-class arm rather than a formality.

    DenseRetriever     Embedding + cosine similarity. Wraps any Embedder,
                       so both the transformer and the TF-IDF/SVD
                       fallback plug in unchanged.

    HybridRetriever    Reciprocal rank fusion over any set of retrievers.
                       Motivated by a specific Phase 2 observation: BM25
                       and dense retrieval failed on DIFFERENT questions,
                       which is precisely the condition under which
                       fusion helps.

Why reciprocal rank fusion rather than score blending
-----------------------------------------------------
BM25 scores are unbounded sums of IDF terms; cosine similarities live in
[-1, 1]. Blending them requires normalising two quantities with different
distributions and no shared scale — and the usual fixes (min-max per
query, z-scoring) are unstable when one retriever returns a flat score
profile, which happens on exactly the hard queries.

RRF sidesteps this by discarding scores and using only RANK:

    score(d) = sum over retrievers of  1 / (k + rank(d))

It needs no tuning beyond `k`, is robust to scale mismatch by
construction, and is the method most production hybrid search actually
uses. `k=60` is the value from the original Cormack et al. paper and is
used here as a documented default rather than a tuned parameter — tuning
it on 95 questions would be fitting noise.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from src.corpus.difficulty import BM25, tokenize
from src.retrieval.chunking import Chunk
from src.retrieval.embedding import Embedder, TfidfSvdEmbedder, encode_cached
from src.retrieval.vectorstore import SearchHit, VectorStore


class Retriever(ABC):
    """Common interface. Every arm in the bake-off implements this."""

    name: str = "base"

    @abstractmethod
    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        """Return the top-k chunks for one query, best first."""

    def search_many(self, queries: list[str],
                    top_k: int = 5) -> list[list[SearchHit]]:
        """Batch search. Overridden where a vectorised path exists."""
        return [self.search(q, top_k) for q in queries]

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


# ---------------------------------------------------------------------
# Lexical
# ---------------------------------------------------------------------
class BM25Retriever(Retriever):
    """Okapi BM25 over chunks.

    Note this indexes `embed_text` (title + body), the same text the dense
    arm embeds. Giving BM25 a different view of the corpus than the dense
    retriever would make the comparison meaningless — any difference could
    be the text rather than the method.
    """

    name = "bm25"

    def __init__(self, chunks: list[Chunk], k1: float = 1.5, b: float = 0.75):
        self._chunks = {c.chunk_id: c for c in chunks}
        self._bm25 = BM25({c.chunk_id: c.embed_text for c in chunks},
                          k1=k1, b=b)

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        tokens = tokenize(query)
        ranked = self._bm25.rank(query, top_k=top_k)
        return [
            SearchHit(chunk=self._chunks[cid],
                      score=self._bm25.score(tokens, cid), rank=r)
            for r, cid in enumerate(ranked)
        ]


# ---------------------------------------------------------------------
# Dense
# ---------------------------------------------------------------------
class DenseRetriever(Retriever):
    """Embedding similarity over chunks.

    Holds the embedder so queries are encoded with the same model that
    built the index — a mismatch there produces silently wrong rankings
    rather than an error.
    """

    def __init__(self, chunks: list[Chunk], embedder: Embedder,
                 name: str | None = None, use_cache: bool = True):
        self.embedder = embedder
        self.name = name or f"dense_{embedder.name}"
        self._use_cache = use_cache

        texts = [c.embed_text for c in chunks]
        if isinstance(embedder, TfidfSvdEmbedder) and not embedder._fitted:
            embedder.fit(texts)

        vectors = (encode_cached(embedder, texts) if use_cache
                   else embedder.encode(texts))
        self._store = VectorStore().add(chunks, vectors)

    def _encode(self, texts: list[str]) -> np.ndarray:
        return (encode_cached(self.embedder, texts) if self._use_cache
                else self.embedder.encode(texts))

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        return self._store.search(self._encode([query])[0], top_k=top_k)

    def search_many(self, queries: list[str],
                    top_k: int = 5) -> list[list[SearchHit]]:
        return self._store.search_batch(self._encode(queries), top_k=top_k)


# ---------------------------------------------------------------------
# Hybrid
# ---------------------------------------------------------------------
RRF_K = 60          # Cormack et al. default; deliberately not tuned


@dataclass
class HybridRetriever(Retriever):
    """Reciprocal rank fusion over two or more retrievers.

    `depth` is how far down each constituent ranking is considered before
    fusing. It must exceed top_k — fusing only the top-5 of each arm
    cannot recover a document that sits at rank 8 in both, which is the
    main case fusion is supposed to fix.
    """

    retrievers: list[Retriever]
    depth: int = 30
    k: int = RRF_K
    name: str = ""          # empty => derive from constituents

    def __post_init__(self):
        if len(self.retrievers) < 2:
            raise ValueError("HybridRetriever needs at least two retrievers")
        if not self.name:
            self.name = "hybrid_" + "+".join(r.name for r in self.retrievers)

    @staticmethod
    def _fuse(rankings: list[list[SearchHit]], k: int,
              top_k: int) -> list[SearchHit]:
        scores: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for hits in rankings:
            for hit in hits:
                cid = hit.chunk.chunk_id
                scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + hit.rank + 1)
                chunks[cid] = hit.chunk
        ordered = sorted(scores, key=lambda c: -scores[c])[:top_k]
        return [SearchHit(chunk=chunks[c], score=scores[c], rank=r)
                for r, c in enumerate(ordered)]

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        rankings = [r.search(query, top_k=self.depth) for r in self.retrievers]
        return self._fuse(rankings, self.k, top_k)

    def search_many(self, queries: list[str],
                    top_k: int = 5) -> list[list[SearchHit]]:
        per_retriever = [r.search_many(queries, top_k=self.depth)
                         for r in self.retrievers]
        return [self._fuse([pr[i] for pr in per_retriever], self.k, top_k)
                for i in range(len(queries))]


# ---------------------------------------------------------------------
# Oracle — the ceiling any fusion of these arms could reach
# ---------------------------------------------------------------------
class OracleUnionRetriever(Retriever):
    """Upper bound: a perfect reranker over the union of candidate pools.

    Not shippable — it cannot rank, it only reports whether the right
    chunk was ANYWHERE in the pooled candidates. That makes it the
    ceiling for the whole family: if the oracle misses a question, no
    amount of fusion tuning or reranking over these retrievers will
    recover it, and the fix has to be upstream (query rewriting, a
    different embedding, or a corpus change).

    Separating "the candidates were never there" from "the candidates
    were there but ranked badly" is the single most useful diagnostic
    in a retrieval bake-off, because the two failures have completely
    different remedies.
    """

    name = "oracle_union"

    def __init__(self, retrievers: list[Retriever], depth: int = 30):
        self.retrievers = retrievers
        self.depth = depth

    def search(self, query: str, top_k: int = 5) -> list[SearchHit]:
        """Pooled candidates, best-known score first, truncated to top_k.

        Ordering by each candidate's best score across arms is not a real
        ranking — the oracle exists to answer "was it in the pool at
        all", not "where would it rank". But it must still honour top_k,
        or callers silently receive the whole pool and any metric
        computed from it is meaningless.
        """
        seen: dict[str, SearchHit] = {}
        for r in self.retrievers:
            for hit in r.search(query, top_k=self.depth):
                prev = seen.get(hit.chunk.chunk_id)
                if prev is None or hit.score > prev.score:
                    seen[hit.chunk.chunk_id] = hit
        ordered = sorted(seen.values(), key=lambda h: -h.score)[:top_k]
        return [SearchHit(chunk=h.chunk, score=h.score, rank=i)
                for i, h in enumerate(ordered)]

    def pool_size(self, query: str) -> int:
        """How many distinct chunks the arms produced between them."""
        return len(self.search(query, top_k=10**9))
