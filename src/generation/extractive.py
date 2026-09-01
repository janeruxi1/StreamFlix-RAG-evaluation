"""An extractive, non-LLM answerer.

Two jobs, and the second is the interesting one.

1. CI and reproducibility. Phase 4 onward needs a generator, and a
   generator normally needs an API key. This one needs nothing, so the
   whole pipeline stays runnable and testable without a credential —
   the same principle as the TF-IDF embedding fallback in Phase 2.

2. A baseline the LLM has to beat. This is the part most RAG projects
   skip. "The LLM answers the questions well" is not a finding unless
   you know what a non-LLM method achieves on the same inputs. If
   sentence extraction gets most of the way there, the LLM's cost and
   latency need a better justification than enthusiasm.

Method: score every sentence in the retrieved context by term overlap
with the question (IDF-weighted, so "the" counts for nothing and "Roku"
counts for a lot), return the best few, and refuse when the best score
falls below a threshold.

Its real weaknesses are the point of including it, not a defect:

  - It cannot synthesise across articles, so multi-hop questions get a
    fragment of one source rather than a combined answer.
  - It cannot paraphrase, so answers read like documentation, not like
    a support reply.
  - It cannot detect contradictions between sources.
  - Its refusal is a similarity threshold, which is exactly the wrong
    instrument for out-of-scope questions whose retrieved context is
    topically adjacent and lexically similar.

Those are precisely the capabilities an LLM is supposed to add, so the
gap between this baseline and the LLM is a measurement of what was
actually bought.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass

from src.corpus.difficulty import tokenize
from src.generation.refusal import CANONICAL_REFUSAL
from src.retrieval.vectorstore import SearchHit

# Reuse the exact string the prompts request, so the baseline and the
# LLM arms are scored by identical refusal detection.
REFUSAL_TEXT = CANONICAL_REFUSAL


def _split_sentences(text: str) -> list[str]:
    """Sentence-ish split that keeps markdown rows and bullets intact."""
    units: list[str] = []
    for line in text.split("\n"):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("|", "-", "*", "#")) or stripped.endswith(":"):
            units.append(stripped)
            continue
        for s in re.split(r"(?<=[.!?])\s+(?=[A-Z])", stripped):
            if s.strip():
                units.append(s.strip())
    return units


@dataclass
class ExtractiveAnswerer:
    """Answers by selecting sentences from the retrieved context.

    `min_score` is the refusal threshold. It is a free parameter and is
    tuned on the DEV split only — tuning it on the full golden set would
    let the baseline peek at the same questions it is scored on, which is
    the error Phase 3 spent a section avoiding.
    """

    max_sentences: int = 3
    min_score: float = 0.18
    cite: bool = True

    name: str = "extractive_baseline"

    def answer(self, question: str, hits: list[SearchHit]) -> str:
        if not hits:
            return REFUSAL_TEXT

        # IDF over the retrieved context: a term appearing in every chunk
        # carries no discriminative signal.
        docs = [tokenize(h.chunk.generation_text) for h in hits]
        n_docs = len(docs)
        df = Counter()
        for d in docs:
            df.update(set(d))
        idf = {t: math.log((n_docs + 1) / (c + 0.5)) for t, c in df.items()}

        q_terms = set(tokenize(question))
        if not q_terms:
            return REFUSAL_TEXT

        scored: list[tuple[float, str, str]] = []
        for hit in hits:
            for sentence in _split_sentences(hit.chunk.generation_text):
                s_terms = set(tokenize(sentence))
                if not s_terms:
                    continue
                overlap = q_terms & s_terms
                # Normalise by the question's own IDF mass so long
                # sentences do not win purely by covering more ground.
                num = sum(idf.get(t, 1.0) for t in overlap)
                den = sum(idf.get(t, 1.0) for t in q_terms) or 1.0
                scored.append((num / den, sentence, hit.chunk.article_id))

        if not scored:
            return REFUSAL_TEXT

        scored.sort(key=lambda x: -x[0])
        if scored[0][0] < self.min_score:
            return REFUSAL_TEXT

        chosen = [s for s in scored[:self.max_sentences]
                  if s[0] >= self.min_score * 0.5]

        parts = []
        seen: set[str] = set()
        for score, sentence, article_id in chosen:
            if sentence in seen:
                continue
            seen.add(sentence)
            parts.append(f"{sentence} [{article_id}]" if self.cite else sentence)
        return " ".join(parts)

    def __repr__(self) -> str:
        return (f"<ExtractiveAnswerer max_sentences={self.max_sentences} "
                f"min_score={self.min_score}>")
