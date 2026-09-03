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
    rungs = ("naive", "grounded", "grounded_refusal", "cited", "strict")
    lengths = [len(VARIANTS[n].template) for n in rungs]
    assert lengths == sorted(lengths)
    assert len(rungs) == len(VARIANTS), "a variant is missing from the ladder"


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


# ---------------------------------------------------------------------
# Answer-then-refuse — regression for a latent scoring bug
# ---------------------------------------------------------------------
@pytest.mark.parametrize("text", [
    "The Premium plan costs $19.99/month [bill-001]. I don't have enough "
    "information about student discounts.",
    "You can cancel from Account Settings [bill-003]. The context does not "
    "mention refund timing.",
    "Downloads are available on Premium [strm-006]. I don't know about the "
    "exact device limit.",
    "Refunds are issued within 30 days [bill-002], though the context doesn't "
    "mention partial months.",
    "You can change plans at any time from your account page, and the change "
    "applies next cycle. I don't know about mid-cycle proration.",
])
def test_answer_then_refuse_is_not_scored_as_a_refusal(text):
    """Regression. check_refusal originally inspected only the text AFTER
    the refusal phrase, so a response that ANSWERED and then hedged about
    a sub-part was scored as a clean refusal.

    That error runs in the dangerous direction: a model that answers an
    out-of-scope question and appends a hedge would be counted as having
    correctly refused, making the safety metric report the opposite of
    what happened. It was latent only because the extractive baseline
    never produces this shape — LLM arms routinely do.
    """
    check = check_refusal(text)
    assert not check.is_refusal, f"scored as refusal: {text!r}"
    assert check.is_partial


def test_citation_before_refusal_phrase_means_it_answered():
    """A citation is a grounded claim by definition, so one appearing
    before the refusal language proves the model answered first."""
    check = check_refusal("Premium is $19.99 [bill-001]. I don't know more.")
    assert not check.is_refusal


def test_short_preamble_before_refusal_is_still_a_refusal():
    """'Based on the context,' is a preamble, not an answer — the lead
    threshold must not be so low that it catches these."""
    for text in ("Based on the context, I don't have enough information "
                 "to answer that.",
                 "Unfortunately, I don't have enough information to answer that.",
                 "After reviewing, I don't know."):
        assert check_refusal(text).is_refusal, text


def test_refusal_followed_by_a_pointer_stays_ambiguous_and_is_documented():
    """A known boundary of pattern-based detection.

    "I don't have enough information. The closest article is [bill-001]
    but it doesn't cover this." is arguably a refusal with a helpful
    pointer, and it is scored as a partial refusal (i.e. an answer)
    because the tail is long and hedged.

    Distinguishing it needs negation understanding, which is beyond a
    regex and is what the Phase 6 judge is for. The misclassification
    runs in the SAFE direction — it under-counts refusals, so it
    under-claims safety rather than overstating it. Pinned here so the
    behaviour is a documented choice rather than an accident.
    """
    check = check_refusal(
        "I don't have enough information to answer that. The closest "
        "article is [bill-001] but it doesn't cover this.")
    assert check.is_partial
    assert not check.is_refusal


# ---------------------------------------------------------------------
# Notebook bootstrap — regression for two real breakages
# ---------------------------------------------------------------------
def test_notebook_setup_cell_resolves_from_any_launch_directory(tmp_path):
    """The .ipynb setup cell must find the project root wherever Jupyter
    was started.

    Two wrong assumptions shipped here in succession, each breaking a
    real workflow and neither caught by CI, because the .py files use
    __file__ and are immune:

      Path.cwd().parent   broke when Jupyter was launched from the
                          project root
      upward search only  broke when Jupyter was launched from a parent
                          folder holding several projects, which puts
                          the project BELOW cwd

    This runs the actual setup cell from each location.
    """
    import json
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "notebooks" / "02_chunking_embedding.ipynb")
                    .read_text(encoding="utf-8"))
    setup = "".join(nb["cells"][1]["source"])
    probe = setup + "\nprint(PROJECT_ROOT.name)"

    for cwd in (root / "notebooks", root, root.parent):
        r = subprocess.run(["python", "-c", probe], cwd=cwd,
                           capture_output=True, text=True, timeout=120)
        assert r.returncode == 0, f"setup cell failed from {cwd}: {r.stderr[-300:]}"
        assert r.stdout.strip().splitlines()[-1] == root.name


def test_notebook_setup_cell_fails_clearly_when_outside_the_project(tmp_path):
    """From somewhere unrelated it must raise something actionable, not
    silently resolve to the wrong directory."""
    import json
    import subprocess
    from pathlib import Path

    root = Path(__file__).resolve().parents[1]
    nb = json.loads((root / "notebooks" / "02_chunking_embedding.ipynb")
                    .read_text(encoding="utf-8"))
    setup = "".join(nb["cells"][1]["source"])

    r = subprocess.run(["python", "-c", setup], cwd=tmp_path,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode != 0
    assert "Could not locate the project root" in r.stderr


# ---------------------------------------------------------------------
# Provider preflight — every precondition, not just the credential
# ---------------------------------------------------------------------
def test_provider_ready_reports_a_missing_sdk(monkeypatch):
    """Regression: notebook 04 checked only for a key, announced
    'running 120 questions', then died on question one with
    ModuleNotFoundError because the openai package was absent.

    A credential is necessary and not sufficient.
    """
    from src.llm import provider as prov

    monkeypatch.setattr(prov, "_load_dotenv_if_present", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "x" * 40)
    monkeypatch.setattr(prov.importlib.util, "find_spec", lambda name: None)

    ready, reason = prov.provider_ready()
    assert not ready
    assert "openai" in reason
    assert "pip install" in reason, "the reason must name the fix"


def test_provider_ready_reports_a_missing_key(monkeypatch):
    from src.llm import provider as prov

    monkeypatch.setattr(prov, "_load_dotenv_if_present", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    ready, reason = prov.provider_ready()
    assert not ready
    assert "OPENAI_API_KEY" in reason


def test_provider_ready_rejects_the_placeholder_key(monkeypatch):
    """.env.example ships a placeholder; copying it without editing is a
    normal mistake and must not look like a working setup."""
    from src.llm import provider as prov

    monkeypatch.setattr(prov, "_load_dotenv_if_present", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-your-key-here")

    ready, reason = prov.provider_ready()
    assert not ready
    assert "placeholder" in reason.lower()


def test_provider_ready_is_true_when_key_and_sdk_are_both_present(monkeypatch):
    from src.llm import provider as prov

    monkeypatch.setattr(prov, "_load_dotenv_if_present", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "x" * 40)
    monkeypatch.setattr(prov.importlib.util, "find_spec", lambda name: object())

    ready, reason = prov.provider_ready()
    assert ready and reason == ""


def test_provider_ready_makes_no_network_call(monkeypatch):
    """It runs at the top of every notebook, so it must be cheap and must
    never construct a client."""
    from src.llm import provider as prov

    monkeypatch.setattr(prov, "_load_dotenv_if_present", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "x" * 40)
    monkeypatch.setattr(prov, "get_provider", lambda *a, **k:
                        (_ for _ in ()).throw(AssertionError("built a provider")))
    prov.provider_ready()


def test_provider_ready_rejects_an_unknown_provider(monkeypatch):
    from src.llm import provider as prov

    monkeypatch.setattr(prov, "_load_dotenv_if_present", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "gemini")
    ready, reason = prov.provider_ready()
    assert not ready
    assert "not recognised" in reason
