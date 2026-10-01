"""Keyless end-to-end answer path used by the demo app.

Kept out of the Streamlit file so it is testable without Streamlit: the
app is a thin shell over `answer_question`.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

from src.corpus.build import load_corpus
from src.generation.extractive import ExtractiveAnswerer
from src.generation.refusal import check_refusal
from src.retrieval.chunking import STRATEGIES
from src.retrieval.retrievers import BM25Retriever

STRATEGY, DEPTH, THRESHOLD = "markdown_section", 15, 0.35   # Phase 3/4 choices


@dataclass(frozen=True)
class DemoAnswer:
    answer: str
    refused: bool
    sources: list[tuple[str, str, float]]    # (article_id, snippet, score)


@lru_cache(maxsize=1)
def _retriever() -> BM25Retriever:
    return BM25Retriever(STRATEGIES[STRATEGY](load_corpus()))


def answer_question(question: str, top_sources: int = 5) -> DemoAnswer:
    hits = _retriever().search(question, top_k=DEPTH)
    answer = ExtractiveAnswerer(min_score=THRESHOLD).answer(question, hits)
    sources = [(h.chunk.article_id, h.chunk.generation_text[:220], float(h.score))
               for h in hits[:top_sources]]
    return DemoAnswer(answer=answer, refused=check_refusal(answer).is_refusal,
                      sources=sources)
