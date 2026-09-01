"""Prompt variants — the experimental axis of Phase 4.

Retrieval is now fixed by Phase 3. The remaining question is what to do
with the retrieved context, and prompting is where most RAG systems
silently succeed or fail. So prompts are treated as configurations to be
compared, not as a detail written once and never revisited.

The variants form a ladder, each adding exactly one mechanism so the
contribution of each is attributable:

    naive             Question + context. No instructions at all. The
                      thing most tutorials ship.
    grounded          Adds an explicit instruction to use only the
                      context.
    grounded_refusal  Adds permission and instruction to say "I don't
                      know" when the context does not support an answer.
    cited             Adds a requirement to cite article ids inline, so
                      faithfulness becomes mechanically checkable rather
                      than requiring a judge for every claim.
    strict            All of the above plus a step of explicit evidence
                      assessment before answering.

The central tension this phase measures
---------------------------------------
Every instruction that makes the model more willing to refuse also makes
it more likely to refuse a question it COULD have answered. That is a
precision/recall trade-off on refusal, and there is no prompt that wins
both ends of it. Reporting only "hallucination went down" while hiding
that helpful answers also went down is the standard way this gets
misreported.

So Phase 4 scores every variant on BOTH:
    - refusal rate on the 25 out-of-scope questions   (higher is better)
    - answer rate on the 95 in-scope questions        (higher is better)

A variant that refuses everything scores perfectly on the first and is
useless. The pair has to be read together.

Citation format
---------------
Citations are requested as [article-id] inline. That format is chosen
because it is trivially parseable with a regex, which means citation
precision — did the model cite articles that were actually in its
context — can be computed for every answer at zero cost, with no LLM
judge involved. Cheap mechanical checks should always be exhausted
before paying for a judge.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from src.retrieval.vectorstore import SearchHit

# Article ids are <prefix>-<3 digits>, and the prefixes vary in length:
# acct, bill, cont, strm (4), dev (3), trial (5). An earlier version of
# this pattern hardcoded {4} and silently dropped every dev- and trial-
# citation — producing understated citation counts and a corrupted
# precision metric with no error anywhere. Kept deliberately loose in
# length and validated against the real corpus in the tests.
CITATION_PATTERN = re.compile(r"\[([a-z]{3,6}-\d{3})\]")


def format_context(hits: list[SearchHit], max_chunks: int | None = None) -> str:
    """Render retrieved chunks as numbered, id-tagged context blocks.

    Each block carries its article id so the model has something concrete
    to cite. Without the id in the context, a citation instruction asks
    the model to invent an identifier, which it will happily do.

    Uses `generation_text`, not `embed_text` — the sentence-window
    strategy deliberately passes wider context than it embeds, and the
    generator should see the wider version.
    """
    selected = hits[:max_chunks] if max_chunks else hits
    return "\n\n".join(
        f"[{h.chunk.article_id}] {h.chunk.title}\n{h.chunk.generation_text}"
        for h in selected
    )


@dataclass(frozen=True)
class PromptVariant:
    """One prompting strategy under test."""
    name: str
    template: str
    description: str
    requires_citations: bool = False

    def render(self, question: str, context: str) -> str:
        return self.template.format(question=question, context=context)


NAIVE = PromptVariant(
    name="naive",
    description="Question + context, no instructions. The tutorial default.",
    template="""Context:
{context}

Question: {question}

Answer:""",
)

GROUNDED = PromptVariant(
    name="grounded",
    description="Adds an instruction to use only the provided context.",
    template="""You are a StreamFlix customer support assistant. Answer the
question using ONLY the information in the context below. Do not use any
outside knowledge about streaming services.

Context:
{context}

Question: {question}

Answer:""",
)

GROUNDED_REFUSAL = PromptVariant(
    name="grounded_refusal",
    description="Grounded, plus explicit permission and instruction to refuse.",
    template="""You are a StreamFlix customer support assistant. Answer the
question using ONLY the information in the context below.

If the context does not contain enough information to answer the
question, reply exactly: "I don't have enough information to answer
that." Do not guess, and do not use outside knowledge. It is better to
say you don't know than to give an answer the context does not support.

Context:
{context}

Question: {question}

Answer:""",
)

CITED = PromptVariant(
    name="cited",
    description="Grounded + refusal + inline [article-id] citations.",
    requires_citations=True,
    template="""You are a StreamFlix customer support assistant. Answer the
question using ONLY the information in the context below.

Cite your sources inline using the article id in square brackets, for
example [bill-001]. Every factual claim must carry a citation to the
context block it came from.

If the context does not contain enough information to answer the
question, reply exactly: "I don't have enough information to answer
that." Do not guess, and do not use outside knowledge.

Context:
{context}

Question: {question}

Answer:""",
)

STRICT = PromptVariant(
    name="strict",
    description="Cited, plus an explicit evidence-assessment step first.",
    requires_citations=True,
    template="""You are a StreamFlix customer support assistant.

Follow these steps:
1. Read the context blocks below.
2. Decide whether they contain enough information to answer the question.
3. If they do NOT, reply exactly: "I don't have enough information to
   answer that." and stop.
4. If they do, answer using ONLY the context, citing the article id for
   each claim inline in square brackets, for example [bill-001].

If the context contains CONFLICTING information, say so explicitly and
cite both sources rather than silently choosing one.

Context:
{context}

Question: {question}

Answer:""",
)

VARIANTS: dict[str, PromptVariant] = {
    v.name: v for v in (NAIVE, GROUNDED, GROUNDED_REFUSAL, CITED, STRICT)
}


# ---------------------------------------------------------------------
# Mechanical checks on generated answers
# ---------------------------------------------------------------------
def extract_citations(answer: str) -> list[str]:
    """Pull [article-id] citations out of an answer, in order."""
    return CITATION_PATTERN.findall(answer)


def citation_precision(answer: str, context_article_ids: set[str]) -> float | None:
    """Fraction of cited ids that were actually present in the context.

    A citation to an article the model was never shown is a fabricated
    source — arguably the most damaging failure mode for a support bot,
    because it looks like evidence. This catches it without an LLM judge.

    Returns None when no citations were made, which is different from
    scoring zero: a variant that does not request citations should not be
    penalised for not producing them.
    """
    cited = extract_citations(answer)
    if not cited:
        return None
    valid = sum(1 for c in cited if c in context_article_ids)
    return valid / len(cited)
