"""Tests for the generation layer.

Refusal detection is the highest-stakes logic in this phase: it decides
what counts as a hallucination, so a bug there corrupts every downstream
safety number. It gets the most tests, including the partial-refusal
cases that are easy to get wrong.
"""
import pytest

from src.corpus.build import load_corpus, load_golden_set
from src.generation.extractive import ExtractiveAnswerer
from src.generation.pipeline import RAGPipeline, RAGResult, score_generation
from src.generation.prompts import (
    VARIANTS,
    citation_precision,
    extract_citations,
    format_context,
)
from src.generation.refusal import CANONICAL_REFUSAL, check_refusal, is_refusal
from src.retrieval.chunking import markdown_section
from src.retrieval.retrievers import BM25Retriever


@pytest.fixture(scope="module")
def articles():
    return load_corpus()


@pytest.fixture(scope="module")
def chunks(articles):
    return markdown_section(articles)


@pytest.fixture(scope="module")
def retriever(chunks):
    return BM25Retriever(chunks)


@pytest.fixture(scope="module")
def golden():
    return load_golden_set()


# ---------------------------------------------------------------------
# Refusal detection
# ---------------------------------------------------------------------
def test_canonical_refusal_is_detected():
    assert is_refusal(CANONICAL_REFUSAL)


@pytest.mark.parametrize("text", [
    "I don't have enough information to answer that.",
    "I do not have enough information to answer this question.",
    "The context does not contain information about gift cards.",
    "The context doesn't mention business accounts.",
    "I don't know.",
    "That is not covered in the provided context.",
    "Insufficient information to answer.",
    "This cannot be answered from the context provided.",
])
def test_refusal_phrasings_are_detected(text):
    """Models paraphrase the requested refusal string constantly."""
    assert is_refusal(text), f"missed refusal: {text!r}"


@pytest.mark.parametrize("text", [
    "The Premium plan costs $19.99 per month.",
    "You can cancel from Account Settings at any time.",
    "Refunds are available within 30 days of purchase.",
])
def test_real_answers_are_not_refusals(text):
    assert not is_refusal(text)


def test_partial_refusal_is_flagged_and_counted_as_an_answer():
    """The most dangerous shape: disclaims, then asserts anyway.

    Counting this as a refusal would hide unsupported claims behind
    language that merely SOUNDS careful.
    """
    text = ("I don't have enough information to answer that, but generally "
            "most streaming services allow you to cancel at any time from "
            "your account settings and receive a prorated refund.")
    check = check_refusal(text)
    assert check.is_partial
    assert not check.is_refusal          # counted as an answer
    assert check.label == "partial_refusal"


def test_short_hedge_after_refusal_is_still_a_refusal():
    """A brief qualifier is not the same as answering anyway."""
    check = check_refusal(
        "I don't have enough information to answer that, but I'm sorry.")
    assert check.is_refusal
    assert not check.is_partial


def test_empty_answer_is_not_a_refusal():
    """Nothing was communicated — a failure, but not a refusal, and
    scoring it as one would flatter the safety numbers."""
    check = check_refusal("")
    assert not check.is_refusal
    assert check.label == "answer"


def test_refusal_detection_is_case_insensitive():
    assert is_refusal("I DON'T HAVE ENOUGH INFORMATION TO ANSWER THAT.")


def test_refusal_check_reports_matched_pattern():
    assert check_refusal(CANONICAL_REFUSAL).matched_pattern is not None
    assert check_refusal("The plan costs $9.99.").matched_pattern is None


# ---------------------------------------------------------------------
# Prompts and citations
# ---------------------------------------------------------------------
def test_every_variant_renders_both_fields():
    for name, v in VARIANTS.items():
        out = v.render(question="Q?", context="CTX")
        assert "Q?" in out and "CTX" in out, name


def test_variants_form_a_ladder_of_increasing_instruction():
    """Each step should add instruction, not rewrite from scratch."""
    lengths = [len(VARIANTS[n].template) for n in
               ("naive", "grounded", "grounded_refusal", "cited")]
    assert lengths == sorted(lengths)


def test_citation_variants_are_marked():
    assert VARIANTS["cited"].requires_citations
    assert VARIANTS["strict"].requires_citations
    assert not VARIANTS["naive"].requires_citations


def test_format_context_tags_every_block_with_an_article_id(retriever):
    """Without the id in context, a citation instruction asks the model
    to invent an identifier."""
    hits = retriever.search("refund policy", top_k=5)
    ctx = format_context(hits)
    for h in hits:
        assert f"[{h.chunk.article_id}]" in ctx


def test_format_context_uses_generation_text(articles):
    """sentence_window passes wider context than it embeds; charging the
    generator the narrow text would understate what it sees."""
    from src.retrieval.chunking import sentence_window
    from src.retrieval.vectorstore import SearchHit
    c = sentence_window(articles)[5]
    ctx = format_context([SearchHit(chunk=c, score=1.0, rank=0)])
    assert c.generation_text in ctx


def test_extract_citations_finds_ids():
    assert extract_citations("Costs $9.99 [bill-001] and see [dev-003].") == \
        ["bill-001", "dev-003"]


def test_extract_citations_ignores_non_ids():
    assert extract_citations("See [the docs] and [123] and [BILL-001].") == []


def test_citation_precision_detects_fabrication():
    """The failure that matters: a source the model was never shown."""
    assert citation_precision("A [bill-001] B [fake-999]",
                              {"bill-001"}) == 0.5


def test_citation_precision_is_none_without_citations():
    """Distinct from zero — a variant that never asked for citations
    must not be penalised for not producing them."""
    assert citation_precision("No citations here.", {"bill-001"}) is None


# ---------------------------------------------------------------------
# Extractive baseline
# ---------------------------------------------------------------------
def test_extractive_refuses_with_no_hits():
    assert ExtractiveAnswerer().answer("anything", []) == CANONICAL_REFUSAL


def test_extractive_answers_an_easy_question(retriever):
    hits = retriever.search("How do I cancel my subscription?", top_k=10)
    answer = ExtractiveAnswerer(min_score=0.1).answer(
        "How do I cancel my subscription?", hits)
    assert not is_refusal(answer)
    assert len(answer) > 20


def test_extractive_only_cites_articles_it_was_shown(retriever):
    """It copies ids off retrieved chunks, so it is structurally
    incapable of fabricating one. Pinning that guarantee."""
    q = "What does the Premium plan cost?"
    hits = retriever.search(q, top_k=10)
    answer = ExtractiveAnswerer(min_score=0.05).answer(q, hits)
    shown = {h.chunk.article_id for h in hits}
    assert all(c in shown for c in extract_citations(answer))


def test_higher_threshold_refuses_more(retriever, golden):
    """The core trade-off must actually be monotone in the knob."""
    qs = golden[:40]
    def refusal_rate(thr):
        a = ExtractiveAnswerer(min_score=thr)
        return sum(is_refusal(a.answer(q["question"],
                                       retriever.search(q["question"], top_k=10)))
                   for q in qs)
    assert refusal_rate(0.6) >= refusal_rate(0.05)


def test_extractive_is_deterministic(retriever):
    q = "How much is Premium?"
    hits = retriever.search(q, top_k=10)
    a = ExtractiveAnswerer(min_score=0.1)
    assert a.answer(q, hits) == a.answer(q, hits)


# ---------------------------------------------------------------------
# Pipeline and scoring
# ---------------------------------------------------------------------
def test_pipeline_preserves_retrieval_for_blame_attribution(retriever, golden):
    """The whole point of keeping hits: telling a retrieval failure from
    a generation failure."""
    r = RAGPipeline(retriever, ExtractiveAnswerer(), depth=10).run_one(golden[0])
    assert r.hits
    assert r.retrieval_recall is not None
    assert r.question_id == golden[0]["question_id"]


def test_out_of_scope_result_has_no_retrieval_recall(retriever, golden):
    oos = next(q for q in golden if q["category"] == "out_of_scope")
    r = RAGPipeline(retriever, ExtractiveAnswerer(), depth=10).run_one(oos)
    assert r.is_out_of_scope
    assert r.retrieval_recall is None


def _result(qid, category, answer, gt=(), hits=()):
    return RAGResult(question_id=qid, question="?", category=category,
                     answer=answer, answerer="t", hits=list(hits),
                     gt_article_ids=tuple(gt))


def test_scores_separate_in_scope_and_out_of_scope():
    results = [
        _result("a", "single_hop", "The plan costs $9.99.", gt=("bill-001",)),
        _result("b", "out_of_scope", CANONICAL_REFUSAL),
    ]
    s = score_generation(results)
    assert s.n_in_scope == 1 and s.n_out_of_scope == 1
    assert s.refusal_rate_oos == 1.0
    assert s.answer_rate_in_scope == 1.0


def test_refusal_f1_punishes_refusing_everything():
    """A variant that refuses every question is perfectly safe and
    useless; F1 must reflect that."""
    always_refuse = [_result("a", "single_hop", CANONICAL_REFUSAL, gt=("bill-001",)),
                     _result("b", "out_of_scope", CANONICAL_REFUSAL)]
    s = score_generation(always_refuse)
    assert s.refusal_rate_oos == 1.0
    assert s.answer_rate_in_scope == 0.0
    assert s.refusal_f1 == 0.0


def test_refusal_f1_punishes_never_refusing():
    never = [_result("a", "single_hop", "An answer.", gt=("bill-001",)),
             _result("b", "out_of_scope", "A confident wrong answer.")]
    s = score_generation(never)
    assert s.refusal_rate_oos == 0.0
    assert s.refusal_f1 == 0.0


def test_over_refusal_only_counts_when_evidence_was_retrieved(articles):
    """Refusing when retrieval returned nothing relevant is CORRECT.
    Charging the generator for it would blame the wrong component."""
    from src.retrieval.chunking import whole_article
    from src.retrieval.vectorstore import SearchHit
    by_id = {c.article_id: c for c in whole_article(articles)}
    good_hit = SearchHit(chunk=by_id["bill-001"], score=1.0, rank=0)
    wrong_hit = SearchHit(chunk=by_id["dev-001"], score=1.0, rank=0)

    results = [
        # refused despite having the right evidence -> over-refusal
        _result("a", "single_hop", CANONICAL_REFUSAL,
                gt=("bill-001",), hits=[good_hit]),
        # refused with no relevant evidence -> not counted
        _result("b", "single_hop", CANONICAL_REFUSAL,
                gt=("bill-002",), hits=[wrong_hit]),
    ]
    s = score_generation(results)
    assert s.over_refusal_rate == 1.0        # 1 of 1 retrievable, not 2


def test_partial_refusal_rate_is_tracked():
    results = [_result("a", "single_hop",
                       "I don't know, but generally most services let you "
                       "cancel any time and refund you within thirty days.",
                       gt=("bill-001",))]
    s = score_generation(results)
    assert s.partial_refusal_rate == 1.0
    assert s.answer_rate_in_scope == 1.0     # counted as an answer


def test_citation_pattern_matches_every_real_article_id(articles):
    """Regression for a silent bug: the pattern hardcoded a 4-letter
    prefix and dropped every dev- and trial- citation. Validate against
    the actual corpus rather than a hand-picked example."""
    for a in articles:
        assert extract_citations(f"claim [{a.article_id}]") == [a.article_id], \
            f"pattern failed on {a.article_id}"


def test_citation_pattern_covers_all_prefix_lengths(articles):
    prefixes = {a.article_id.split("-")[0] for a in articles}
    assert {len(p) for p in prefixes} & {3, 5}, \
        "corpus should contain non-4-letter prefixes for this to be a real test"
