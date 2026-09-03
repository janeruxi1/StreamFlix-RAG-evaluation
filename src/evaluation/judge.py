"""LLM-as-judge, and the machinery to check whether the judge works.

The standard failure of LLM-as-judge evaluation is that nobody validates
the judge. Scores get reported to three decimal places by a model whose
agreement with ground truth was never measured, and the whole evaluation
inherits an unknown error rate.

So this module ships two things, and the second is the point:

    Judge        faithfulness / relevancy / correctness scoring
    validation   a labelled suite where the right answer is known, used
                 to measure judge accuracy BEFORE any real score is
                 believed

Design decisions worth stating
------------------------------
JUDGE STRONGER THAN GENERATOR. The judge defaults to gpt-4o while
generation uses gpt-4o-mini. A model grading its own output family
exhibits self-preference bias — it rates text that looks like its own
output more highly — which would inflate exactly the headline metric this
project is built around. The asymmetry costs more per call, but the
harness makes far fewer judge calls than generation calls.

TEMPERATURE 0. This is measurement. Sampling noise would be
indistinguishable from a real difference between prompt variants.

STRUCTURED OUTPUT, PARSED DEFENSIVELY. The judge is asked for a rigid
format. Real models violate rigid formats, so parsing falls back rather
than raising, and an unparseable response is recorded as such instead of
being silently scored zero — a parse failure and a genuine zero are very
different facts about the system.

CLAIM-LEVEL FAITHFULNESS. Asking "is this answer faithful, 0 to 1"
invites an unanchored vibe. Asking the judge to enumerate the answer's
claims and mark each supported or unsupported produces a score with
defined units — supported claims over total claims — and makes the
reasoning auditable.
"""
from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from src.corpus.difficulty import tokenize
from src.llm.provider import LLMProvider


@dataclass(frozen=True)
class Judgement:
    """One metric's verdict on one answer."""
    metric: str
    score: float                  # 0.0 - 1.0
    reasoning: str = ""
    parsed_ok: bool = True        # False => format violation, not a real 0
    n_claims: int | None = None
    n_supported: int | None = None

    def __str__(self) -> str:
        flag = "" if self.parsed_ok else "  [PARSE FAILED]"
        return f"{self.metric}={self.score:.2f}{flag}"


# ---------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------
FAITHFULNESS_PROMPT = """You are evaluating whether an answer is supported by
the context it was given. You are NOT judging whether the answer is
helpful, well written, or correct in general — only whether every claim
it makes can be verified from the context below.

Context:
{context}

Answer to evaluate:
{answer}

Break the answer into individual factual claims. For each, decide whether
the context supports it. A claim that is plausible but absent from the
context is NOT supported.

Reply in exactly this format:
CLAIMS_TOTAL: <integer>
CLAIMS_SUPPORTED: <integer>
REASONING: <one sentence>"""

RELEVANCY_PROMPT = """You are evaluating whether an answer addresses the
question that was asked. You are NOT judging whether it is factually
correct — only whether it is responsive.

Question: {question}

Answer to evaluate:
{answer}

An answer that is fluent and on-topic but does not actually answer the
question scores low. An answer that declines to answer because
information was unavailable should be scored 1.0 if declining was
responsive, since it addresses the question honestly.

Reply in exactly this format:
SCORE: <number between 0 and 1>
REASONING: <one sentence>"""

CORRECTNESS_PROMPT = """You are comparing an answer against a reference
answer written by a domain expert.

Question: {question}

Reference answer:
{reference}

Answer to evaluate:
{answer}

Score how well the answer conveys the same information as the reference.
Differences in wording, length, or style do not matter. Missing facts and
contradictory facts do.

Reply in exactly this format:
SCORE: <number between 0 and 1>
REASONING: <one sentence>"""


# ---------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------
def _extract_number(text: str, field: str) -> float | None:
    m = re.search(rf"{field}\s*:\s*([0-9]*\.?[0-9]+)", text, re.IGNORECASE)
    return float(m.group(1)) if m else None


def _extract_reasoning(text: str) -> str:
    m = re.search(r"REASONING\s*:\s*(.+)", text, re.IGNORECASE | re.DOTALL)
    return m.group(1).strip().split("\n")[0][:300] if m else ""


def parse_faithfulness(raw: str) -> Judgement:
    """Parse a claim-count response into a ratio.

    Zero total claims is a real case, not an error: a refusal makes no
    claims. Scoring it 0.0 would punish correct refusals on the metric
    that is supposed to reward not making things up, so it scores 1.0 —
    vacuously faithful, having asserted nothing.
    """
    total = _extract_number(raw, "CLAIMS_TOTAL")
    supported = _extract_number(raw, "CLAIMS_SUPPORTED")

    if total is None or supported is None:
        return Judgement("faithfulness", 0.0, _extract_reasoning(raw),
                         parsed_ok=False)
    if total == 0:
        return Judgement("faithfulness", 1.0, _extract_reasoning(raw),
                         n_claims=0, n_supported=0)

    supported = min(supported, total)      # guard against a miscount
    return Judgement("faithfulness", supported / total,
                     _extract_reasoning(raw),
                     n_claims=int(total), n_supported=int(supported))


def parse_score(raw: str, metric: str) -> Judgement:
    score = _extract_number(raw, "SCORE")
    if score is None:
        return Judgement(metric, 0.0, _extract_reasoning(raw), parsed_ok=False)
    return Judgement(metric, max(0.0, min(1.0, score)), _extract_reasoning(raw))


# ---------------------------------------------------------------------
# Judges
# ---------------------------------------------------------------------
class Judge(ABC):
    """Common interface so the LLM and lexical judges are interchangeable."""

    name: str = "base"

    @abstractmethod
    def faithfulness(self, answer: str, context: str) -> Judgement: ...

    @abstractmethod
    def relevancy(self, answer: str, question: str) -> Judgement: ...

    @abstractmethod
    def correctness(self, answer: str, question: str,
                    reference: str) -> Judgement: ...

    def __repr__(self) -> str:
        return f"<{type(self).__name__} name={self.name!r}>"


class LLMJudge(Judge):
    """Judge backed by a model, deliberately stronger than the generator."""

    def __init__(self, provider: LLMProvider, model: str | None = None):
        self.provider = provider
        self.model = model
        self.name = f"llm_judge_{model or provider.default_model}"

    def _ask(self, prompt: str) -> str:
        return self.provider.complete(prompt, model=self.model,
                                      max_tokens=250, temperature=0.0)

    def faithfulness(self, answer: str, context: str) -> Judgement:
        return parse_faithfulness(self._ask(
            FAITHFULNESS_PROMPT.format(context=context, answer=answer)))

    def relevancy(self, answer: str, question: str) -> Judgement:
        return parse_score(self._ask(
            RELEVANCY_PROMPT.format(question=question, answer=answer)),
            "relevancy")

    def correctness(self, answer: str, question: str,
                    reference: str) -> Judgement:
        return parse_score(self._ask(
            CORRECTNESS_PROMPT.format(question=question, reference=reference,
                                      answer=answer)),
            "correctness")


class LexicalJudge(Judge):
    """A judge with no model behind it. Overlap statistics only.

    Exists for the same reason the extractive answerer does: it keeps the
    harness runnable without a credential, AND it is a baseline the LLM
    judge has to beat on the validation suite below.

    It is expected to FAIL specific cases, and that failure is the useful
    part. It cannot detect a contradiction, because a contradicting
    sentence reuses the context's vocabulary almost perfectly — high
    overlap, opposite meaning. Demonstrating that concretely is what
    justifies paying for the LLM judge rather than asserting it is better.
    """

    name = "lexical_judge"

    @staticmethod
    def _overlap(a: str, b: str) -> float:
        """Fraction of a's content terms that appear in b."""
        ta, tb = set(tokenize(a)), set(tokenize(b))
        return len(ta & tb) / len(ta) if ta else 0.0

    def faithfulness(self, answer: str, context: str) -> Judgement:
        score = self._overlap(answer, context)
        return Judgement("faithfulness", score,
                         "lexical overlap of answer terms with context")

    def relevancy(self, answer: str, question: str) -> Judgement:
        return Judgement("relevancy", self._overlap(question, answer),
                         "lexical overlap of question terms with answer")

    def correctness(self, answer: str, question: str,
                    reference: str) -> Judgement:
        return Judgement("correctness", self._overlap(reference, answer),
                         "lexical overlap of reference terms with answer")


class CoverageJudge(Judge):
    """A second non-LLM judge using a genuinely different formulation.

    Exists so inter-judge agreement can be exercised without a
    credential — but only if the two judges actually differ. An earlier
    version compared two LexicalJudge instances distinguished by a
    `support_threshold` parameter that was stored and never read, so the
    "two raters" were one function and Cohen's kappa came out at exactly
    1.000. A perfect agreement score between a thing and itself measures
    nothing, and it looked like a result.

    LexicalJudge asks: what fraction of the ANSWER's terms appear in the
    context? That is precision-flavoured — it punishes an answer for
    saying anything the context does not, including filler.

    This asks the reverse: what fraction of the answer's SENTENCES are
    each substantially grounded in the context? That is closer to how
    faithfulness is actually defined — per claim rather than per token —
    and it disagrees with term overlap on exactly the cases where an
    answer is verbose but every sentence is supported.

    Still not semantic, and still fails contradictions for the same
    reason. It is a second flawed rater, which is what makes the
    agreement number non-trivial rather than tautological.
    """

    name = "coverage_judge"

    def __init__(self, sentence_support: float = 0.6):
        self.sentence_support = sentence_support

    def _sentences(self, text: str) -> list[str]:
        return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text)
                if s.strip()]

    def faithfulness(self, answer: str, context: str) -> Judgement:
        sentences = self._sentences(answer)
        if not sentences:
            return Judgement("faithfulness", 1.0, "no claims made")
        ctx_terms = set(tokenize(context))
        supported = 0
        for sentence in sentences:
            terms = set(tokenize(sentence))
            if not terms:
                supported += 1
                continue
            if len(terms & ctx_terms) / len(terms) >= self.sentence_support:
                supported += 1
        return Judgement("faithfulness", supported / len(sentences),
                         f"{supported}/{len(sentences)} sentences grounded",
                         n_claims=len(sentences), n_supported=supported)

    def relevancy(self, answer: str, question: str) -> Judgement:
        q, a = set(tokenize(question)), set(tokenize(answer))
        return Judgement("relevancy", len(q & a) / len(q) if q else 0.0,
                         "question-term coverage in the answer")

    def correctness(self, answer: str, question: str,
                    reference: str) -> Judgement:
        r, a = set(tokenize(reference)), set(tokenize(answer))
        return Judgement("correctness", len(r & a) / len(r) if r else 0.0,
                         "reference-term coverage in the answer")


def get_judge(provider: LLMProvider | None = None,
              model: str | None = None) -> Judge:
    """LLM judge when a provider is supplied, lexical otherwise."""
    return LLMJudge(provider, model) if provider is not None else LexicalJudge()
