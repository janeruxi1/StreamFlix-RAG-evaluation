"""Embedding backends.

Two implementations behind one interface:

    SentenceTransformerEmbedder   The real one. BGE-small by default.
                                  Local, free, no API key, ~130MB model.

    TfidfSvdEmbedder              Deterministic fallback. sklearn only.
                                  Exists so tests and CI can exercise the
                                  full retrieval pipeline without
                                  downloading a transformer.

The fallback is not a toy: TF-IDF + truncated SVD is latent semantic
analysis, a real (if dated) dense-retrieval method. Keeping it as a
first-class backend has three benefits:

  1. CI runs the whole pipeline in seconds with no model download.
  2. Anyone cloning the repo gets a working system before deciding
     whether to install torch.
  3. Phase 3 gains a legitimate third arm — BM25 (lexical), LSA (classic
     dense), transformer (modern dense) — which makes the bake-off a
     progression rather than a binary.

Embeddings are cached to disk keyed by (backend, model, text-hash), so
re-running a notebook after a kernel restart costs nothing.
"""
from __future__ import annotations

import hashlib
import json
import os
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

CACHE_DIR = Path("data/embedding_cache")


class Embedder(ABC):
    """Common interface for every embedding backend."""

    name: str = "base"

    @abstractmethod
    def encode(self, texts: list[str]) -> np.ndarray:
        """Return an (n_texts, dim) L2-normalised float array."""

    @property
    @abstractmethod
    def dimension(self) -> int: ...

    @property
    def cache_key(self) -> str:
        """Identity for caching — must capture anything that changes output.

        For a pretrained model the model name is sufficient. For a backend
        FITTED on the corpus it is not, and the override in
        TfidfSvdEmbedder explains why.
        """
        return f"{self.name}:{getattr(self, 'model_name', self.name)}"

    @staticmethod
    def _normalise(vectors: np.ndarray) -> np.ndarray:
        """L2-normalise rows so dot product equals cosine similarity."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return vectors / norms

    def __repr__(self) -> str:
        return f"<{type(self).__name__} dim={self.dimension}>"


class SentenceTransformerEmbedder(Embedder):
    """Transformer embeddings via sentence-transformers.

    BGE-small-en-v1.5 is the default: 384 dimensions, ~130MB, and
    competitive with much larger models on retrieval benchmarks. Runs on
    CPU in a few seconds for a corpus this size.
    """

    name = "sentence_transformer"

    def __init__(self, model_name: str | None = None):
        self.model_name = model_name or os.getenv(
            "EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5"
        )
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:
            raise ImportError(
                "sentence-transformers is not installed.\n"
                "  pip install sentence-transformers\n"
                "Or use the fallback backend:\n"
                "  get_embedder(backend='tfidf_svd')"
            ) from exc
        self._model = SentenceTransformer(self.model_name)

    @property
    def dimension(self) -> int:
        return int(self._model.get_sentence_embedding_dimension())

    def encode(self, texts: list[str]) -> np.ndarray:
        vectors = self._model.encode(
            texts, batch_size=32, show_progress_bar=False,
            convert_to_numpy=True,
        )
        return self._normalise(np.asarray(vectors, dtype=np.float32))


class TfidfSvdEmbedder(Embedder):
    """TF-IDF followed by truncated SVD — latent semantic analysis.

    Must be fitted on the corpus before encoding queries, which is a real
    limitation worth naming: unlike a transformer, it cannot embed text
    containing vocabulary it never saw at fit time. That is precisely why
    it underperforms on paraphrased questions, and Phase 3 will show it.
    """

    name = "tfidf_svd"

    def __init__(self, n_components: int = 128, random_state: int = 42):
        from sklearn.decomposition import TruncatedSVD
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.n_components = n_components
        self._vectorizer = TfidfVectorizer(
            lowercase=True, stop_words="english",
            ngram_range=(1, 2), min_df=1, sublinear_tf=True,
        )
        self._svd = TruncatedSVD(
            n_components=n_components, random_state=random_state
        )
        self._fitted = False

    def fit(self, corpus_texts: list[str]) -> "TfidfSvdEmbedder":
        """Fit the vocabulary and latent space on the corpus."""
        tfidf = self._vectorizer.fit_transform(corpus_texts)
        # SVD cannot produce more components than the feature matrix rank
        self._svd.n_components = min(
            self.n_components, min(tfidf.shape) - 1
        )
        self._svd.fit(tfidf)
        self._fitted = True
        self._fit_fingerprint = hashlib.sha256(
            json.dumps(sorted(corpus_texts)).encode()
        ).hexdigest()[:16]
        return self

    @property
    def cache_key(self) -> str:
        """Identity including WHAT THIS WAS FITTED ON.

        This override is load-bearing. Unlike a pretrained transformer,
        this embedder's output for a given text depends entirely on the
        corpus it was fitted to — the same query encodes differently under
        a vocabulary built from 45 whole articles versus 994 sentences.

        Without the fingerprint, the disk cache would key identical query
        text to a single entry and hand back vectors from whichever
        chunking strategy ran first. Every subsequent configuration in the
        bake-off would then be silently scored against the wrong query
        embeddings — producing a complete, plausible, entirely invalid
        results table with no error anywhere.
        """
        fp = getattr(self, "_fit_fingerprint", "unfitted")
        return f"{self.name}:svd{self.n_components}:{fp}"

    @property
    def dimension(self) -> int:
        return int(self._svd.n_components)

    def encode(self, texts: list[str]) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError(
                "TfidfSvdEmbedder must be fitted on the corpus before "
                "encoding. Call .fit(corpus_texts) first."
            )
        reduced = self._svd.transform(self._vectorizer.transform(texts))
        return self._normalise(np.asarray(reduced, dtype=np.float32))


# ---------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------
class EmbeddingCache:
    """Disk cache keyed by backend, model, and a hash of the texts.

    Embeddings are deterministic, so recomputing them across kernel
    restarts is pure waste — and the Phase 3 bake-off re-embeds the
    corpus once per chunking strategy.
    """

    def __init__(self, cache_dir: Path = CACHE_DIR, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        if enabled:
            cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(backend: str, model: str, texts: list[str]) -> str:
        digest = hashlib.sha256(
            json.dumps([backend, model, texts]).encode()
        ).hexdigest()
        return digest[:32]

    def get(self, backend: str, model: str,
            texts: list[str]) -> np.ndarray | None:
        if not self.enabled:
            return None
        path = self.cache_dir / f"{self._key(backend, model, texts)}.npy"
        return np.load(path) if path.exists() else None

    def put(self, backend: str, model: str, texts: list[str],
            vectors: np.ndarray) -> None:
        if not self.enabled:
            return
        path = self.cache_dir / f"{self._key(backend, model, texts)}.npy"
        np.save(path, vectors)


def encode_cached(embedder: Embedder, texts: list[str],
                  cache: EmbeddingCache | None = None) -> np.ndarray:
    """Encode with a disk cache in front."""
    cache = cache or EmbeddingCache()
    key = embedder.cache_key
    hit = cache.get(embedder.name, key, texts)
    if hit is not None:
        return hit
    vectors = embedder.encode(texts)
    cache.put(embedder.name, key, texts, vectors)
    return vectors


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
def transformer_available() -> bool:
    """Whether sentence-transformers can be imported."""
    try:
        import sentence_transformers  # noqa: F401
        return True
    except ImportError:
        return False


def get_embedder(backend: str | None = None,
                 model_name: str | None = None) -> Embedder:
    """Construct an embedder.

    Resolution order:
      1. explicit `backend` argument
      2. EMBEDDING_BACKEND environment variable
      3. sentence-transformers if importable
      4. TF-IDF + SVD fallback

    The fallback is chosen silently rather than raising, so a fresh
    clone runs end-to-end before anyone installs torch. Which backend
    ran is always reported in the notebook output.
    """
    backend = backend or os.getenv("EMBEDDING_BACKEND")

    if backend == "tfidf_svd":
        return TfidfSvdEmbedder()
    if backend == "sentence_transformer":
        return SentenceTransformerEmbedder(model_name)
    if backend is not None:
        raise ValueError(
            f"Unknown embedding backend {backend!r}. "
            f"Expected 'sentence_transformer' or 'tfidf_svd'."
        )

    return (SentenceTransformerEmbedder(model_name)
            if transformer_available() else TfidfSvdEmbedder())
