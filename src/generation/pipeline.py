"""The RAG pipeline: retrieve, build context, generate, record.

One class, one responsibility per stage, and — importantly — every
intermediate kept. A pipeline that returns only the final answer makes
failure attribution impossible: when an answer is wrong, you cannot tell
whether retrieval missed the source, the context was assembled badly, or
the model ignored what it was given.

Phase 3 produced a concrete example of why this matters. Question mh-011
retrieves five plausible, on-topic, WRONG chunks at shallow depth. The
generator then answers fluently from them, and the result looks exactly
like a hallucination while the fault is entirely upstream. `RAGResult`
therefore carries the retrieved chunks alongside the answer, so Phase 6
can separate "the evidence was never there" from "the evidence was there
and was ignored".

Answerers are duck-typed rather than sharing a base class: anything with
`.name` and an `answer(question, hits) -> str` method works. That lets
the extractive baseline and the LLM variants run through identical code,
which is what makes their comparison meaningful.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

from src.corpus.difficulty import count_tokens
from src.generation.prompts import (
    PromptVariant,
    citation_precision,
    extract_citations,
    format_context,
)
from src.generation.refusal import RefusalCheck, check_refusal
from src.llm.provider import LLMProvider
from src.retrieval.retrievers import Retriever
from src.retrieval.vectorstore import SearchHit, hits_to_articles


@dataclass
class RAGResult:
    """Everything one question produced, at every stage."""
    question_id: str
    question: str
    category: str
    answer: str
    answerer: str
    hits: list[SearchHit] = field(repr=False, default_factory=list)
    latency_s: float = 0.0
    prompt_tokens: int = 0

    # ground truth, carried through for scoring
    gt_article_ids: tuple[str, ...] = ()

    @property
    def retrieved_article_ids(self) -> list[str]:
        return hits_to_articles(self.hits)

    @property
    def context_article_ids(self) -> set[str]:
        return {h.chunk.article_id for h in self.hits}

    @property
    def refusal(self) -> RefusalCheck:
        return check_refusal(self.answer)

    @property
    def citations(self) -> list[str]:
        return extract_citations(self.answer)

    @property
    def citation_precision(self) -> float | None:
        return citation_precision(self.answer, self.context_article_ids)

    @property
    def is_out_of_scope(self) -> bool:
        return not self.gt_article_ids

    @property
    def retrieval_recall(self) -> float | None:
        """Was the evidence even available? None for out-of-scope.

        This is the field that makes blame attribution possible: a wrong
        answer with recall 0.0 is a retrieval failure wearing a
        generation failure's clothes.
        """
        if not self.gt_article_ids:
            return None
        gt = set(self.gt_article_ids)
        return len(gt & set(self.retrieved_article_ids)) / len(gt)


class LLMAnswerer:
    """Wraps a provider and a prompt variant into an answerer."""

    def __init__(self, provider: LLMProvider, variant: PromptVariant,
                 model: str | None = None, max_tokens: int = 400):
        self.provider = provider
        self.variant = variant
        self.model = model
        self.max_tokens = max_tokens
        self.name = f"llm_{variant.name}"
        self.last_prompt: str = ""

    def answer(self, question: str, hits: list[SearchHit]) -> str:
        self.last_prompt = self.variant.render(
            question=question, context=format_context(hits))
        # temperature=0 throughout: this is measurement, and sampling
        # noise would be indistinguishable from a prompt's effect.
        return self.provider.complete(
            self.last_prompt, model=self.model,
            max_tokens=self.max_tokens, temperature=0.0).strip()


@dataclass
class RAGPipeline:
    """Retriever + answerer, run over the golden set."""

    retriever: Retriever
    answerer: object              # anything with .name and .answer()
    depth: int = 15

    def run_one(self, question: dict) -> RAGResult:
        started = time.perf_counter()
        hits = self.retriever.search(question["question"], top_k=self.depth)
        answer = self.answerer.answer(question["question"], hits)
        elapsed = time.perf_counter() - started

        prompt = getattr(self.answerer, "last_prompt", "")
        return RAGResult(
            question_id=question["question_id"],
            question=question["question"],
            category=question["category"],
            answer=answer,
            answerer=self.answerer.name,
            hits=hits,
            latency_s=elapsed,
            prompt_tokens=count_tokens(prompt) if prompt else 0,
            gt_article_ids=tuple(question.get("gt_article_ids", ())),
        )

    def run(self, questions: list[dict],
            progress_every: int = 0) -> list[RAGResult]:
        results = []
        for i, q in enumerate(questions, 1):
            results.append(self.run_one(q))
            if progress_every and i % progress_every == 0:
                print(f"    {i}/{len(questions)} ...", flush=True)
        return results


# ---------------------------------------------------------------------
# Scoring — mechanical only. No judge, no API calls.
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class GenerationScores:
    """What can be measured without an LLM judge.

    Deliberately excludes answer quality. Whether an answer is CORRECT
    needs either a judge or a human, and that is Phase 5/6. Everything
    here is free, deterministic, and auditable, and it already catches
    the failure modes that matter most for a support bot: answering when
    it should refuse, refusing when it should answer, and citing sources
    it was never shown.
    """
    answerer: str
    n_in_scope: int
    n_out_of_scope: int

    # The trade-off pair. Neither number means anything alone.
    refusal_rate_oos: float        # of out-of-scope, fraction refused
    answer_rate_in_scope: float    # of in-scope, fraction answered

    over_refusal_rate: float       # in-scope refused DESPITE good retrieval
    partial_refusal_rate: float    # "I don't know, but..." — worst shape
    mean_citation_precision: float | None
    uncited_answer_rate: float | None
    mean_latency_s: float
    mean_prompt_tokens: float

    @property
    def refusal_f1(self) -> float:
        """Harmonic mean of correct-refusal and correct-answer rates.

        A single number for ranking, but it must not be read alone — it
        hides which side a variant failed on, and the two failures have
        very different costs to a business.
        """
        p, r = self.refusal_rate_oos, self.answer_rate_in_scope
        return 2 * p * r / (p + r) if (p + r) else 0.0


def score_generation(results: list[RAGResult],
                     answerer: str | None = None) -> GenerationScores:
    """Compute every judge-free metric for one answerer's run."""
    in_scope = [r for r in results if not r.is_out_of_scope]
    oos = [r for r in results if r.is_out_of_scope]

    refused_oos = sum(r.refusal.is_refusal for r in oos)
    answered_in = sum(not r.refusal.is_refusal for r in in_scope)

    # Over-refusal is only meaningful where retrieval actually supplied
    # the evidence. Refusing when nothing relevant was retrieved is
    # correct behaviour, and charging the generator for it would blame
    # the wrong component.
    retrievable = [r for r in in_scope if (r.retrieval_recall or 0) > 0]
    over_refused = sum(r.refusal.is_refusal for r in retrievable)

    precisions = [r.citation_precision for r in results
                  if r.citation_precision is not None]
    answered = [r for r in results if not r.refusal.is_refusal]
    uncited = sum(1 for r in answered if not r.citations)

    return GenerationScores(
        answerer=answerer or (results[0].answerer if results else "unknown"),
        n_in_scope=len(in_scope),
        n_out_of_scope=len(oos),
        refusal_rate_oos=refused_oos / len(oos) if oos else float("nan"),
        answer_rate_in_scope=answered_in / len(in_scope) if in_scope else float("nan"),
        over_refusal_rate=over_refused / len(retrievable) if retrievable else float("nan"),
        partial_refusal_rate=sum(r.refusal.is_partial for r in results) / len(results),
        mean_citation_precision=(sum(precisions) / len(precisions)
                                 if precisions else None),
        uncited_answer_rate=uncited / len(answered) if answered else None,
        mean_latency_s=sum(r.latency_s for r in results) / len(results),
        mean_prompt_tokens=sum(r.prompt_tokens for r in results) / len(results),
    )
