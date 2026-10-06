"""StreamFlix Support RAG — interactive demo.

Most RAG demos show you an answer. This one shows you the answer AND the
evidence for whether to believe it, because that is what the project is
about: a fluent answer is easy, and knowing whether it is trustworthy is
the hard part.

Every panel is computed by the same `src/` modules the notebooks and CI
use, so nothing here is a special demo path that could disagree with the
measured results.

Three modes:

  Ask                  type any question, see retrieval, the answer, and
                       every judge-free check that applies to it
  Golden set           pick a labelled question and see the evaluation
                       against its ground truth
  What the harness     the audit that decides whether any judge score is
  measures             worth reading at all

Runs with no API key: the extractive baseline and every judge-free metric
work offline. A credential adds the LLM arms and the semantic judge.

    streamlit run app/streamlit_app.py
"""
import sys
from pathlib import Path


def _find_project_root(start: Path) -> Path:
    """Same up-then-down search the notebooks use.

    Streamlit is launched from wherever the user happens to be, so this
    cannot assume a working directory any more than the notebooks can.
    """
    marker = Path("src") / "corpus" / "seed_articles.py"
    start = start.resolve()
    for candidate in [start, *start.parents]:
        if (candidate / marker).is_file():
            return candidate
    for pattern in ("*", "*/*"):
        found = sorted({m.parents[2]
                        for m in start.glob(f"{pattern}/{marker.as_posix()}")})
        if len(found) == 1:
            return found[0]
    raise RuntimeError(f"Could not locate the project root from {start}")


PROJECT_ROOT = _find_project_root(Path(__file__).resolve().parent)
sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

from src.corpus.build import load_corpus, load_golden_set
from src.evaluation.judge import LexicalJudge, get_judge
from src.evaluation.judge_bias import (
    measure_length_bias,
    measure_order_sensitivity,
)
from src.evaluation.judge_validation import VALIDATION_CASES, validate_judge
from src.evaluation.rag_metrics import context_precision, evaluate_answer
from src.evaluation.retrieval_metrics import context_tokens
from src.generation.extractive import ExtractiveAnswerer
from src.generation.pipeline import LLMAnswerer, RAGPipeline
from src.generation.prompts import VARIANTS, format_context
from src.llm.provider import provider_ready
from src.retrieval.chunking import STRATEGIES
from src.retrieval.retrievers import BM25Retriever

# Locked in by the Phase 3 bake-off, chosen on the Pareto frontier under
# a 600-token context budget rather than at the top of the recall table.
STRATEGY = "markdown_section"
DEPTH = 15
EXTRACTIVE_THRESHOLD = 0.35

st.set_page_config(page_title="StreamFlix Support RAG", page_icon="🔍",
                   layout="wide")


@st.cache_resource
def build_system():
    articles = load_corpus()
    chunks = STRATEGIES[STRATEGY](articles)
    return articles, chunks, BM25Retriever(chunks)


@st.cache_data
def load_questions():
    return load_golden_set()


articles, chunks, retriever = build_system()
golden = load_questions()
has_key, blocker = provider_ready()


# ---------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------
with st.sidebar:
    st.title("🔍 Support RAG")
    st.caption("Retrieval-augmented answering, with the evaluation attached.")

    mode = st.radio("Mode", ["Ask anything", "Golden set", "Is the judge "
                             "trustworthy?"])

    st.divider()
    st.subheader("Configuration")
    st.markdown(
        f"""
        **Chunking** `{STRATEGY}`
        **Retrieval** BM25 @ depth {DEPTH}
        **Corpus** {len(articles)} articles
        **Golden set** {len(golden)} questions
        """
    )
    st.caption(
        "Chosen on the Pareto frontier under a 600-token context budget — "
        "not the top of the recall table. The highest-recall configuration "
        "costs 4.5× more context for a difference that fails a significance "
        "test."
    )

    st.divider()
    if has_key:
        st.success("LLM available")
        answerer_choice = st.selectbox(
            "Answerer",
            ["extractive (no LLM)"] + [f"llm — {n}" for n in VARIANTS],
        )
    else:
        st.info("Running without an API key")
        st.caption(
            f"{blocker}\n\nThe extractive baseline and every judge-free "
            f"metric work offline. That is deliberate: the harness degrades "
            f"rather than stops."
        )
        answerer_choice = "extractive (no LLM)"


def build_answerer():
    if answerer_choice.startswith("llm"):
        from src.llm.provider import get_provider
        variant = VARIANTS[answerer_choice.split("— ")[1]]
        return LLMAnswerer(get_provider(), variant)
    return ExtractiveAnswerer(min_score=EXTRACTIVE_THRESHOLD)


def show_evaluation(result, reference: str | None = None):
    """Every check that applies, and an honest label on each."""
    judge = get_judge(None) if not has_key else get_judge(
        __import__("src.llm.provider", fromlist=["get_provider"]).get_provider())
    ev = evaluate_answer(result, judge, reference)

    st.subheader("Was it a refusal?")
    refusal = result.refusal
    label = {"refusal": "🛑 Refused", "partial_refusal": "⚠️ Partial refusal",
             "answer": "💬 Answered"}[refusal.label]
    st.markdown(f"**{label}**")
    if refusal.is_partial:
        st.warning(
            "A partial refusal disclaims and then asserts anyway. It reads "
            "as caution while still making unsupported claims, so the "
            "harness counts it as an ANSWER."
        )

    st.subheader("Citations")
    cited = result.citations
    if not cited:
        st.caption("No citations in this answer.")
    else:
        shown = result.context_article_ids
        fabricated = [c for c in cited if c not in shown]
        st.markdown(f"Cited: {', '.join(f'`{c}`' for c in cited)}")
        if fabricated:
            st.error(
                f"**Fabricated source(s):** {', '.join(fabricated)} — cited "
                f"but never shown to the model. A fabricated citation is "
                f"worse than an uncited claim, because it looks like evidence."
            )
        else:
            st.success("Every cited article was actually in the context.")
            if isinstance(build_answerer(), ExtractiveAnswerer):
                st.caption(
                    "Note: the extractive answerer copies ids off the chunks "
                    "it lifted from, so it is structurally incapable of "
                    "fabricating one. Perfect precision here measures the "
                    "architecture, not the behaviour."
                )

    if result.gt_article_ids:
        st.subheader("Against ground truth")
        c1, c2 = st.columns(2)
        c1.metric("context recall", f"{ev.context_recall:.2f}")
        c2.metric("context precision", f"{context_precision(result):.2f}")
        st.caption(
            "Precision looks low because it is capped: only ~4.5 relevant "
            "chunks exist per article, so at depth 15 most slots must hold "
            "something irrelevant. Read it against its ceiling, not against 1.0."
        )
        st.markdown(f"**Failure mode:** `{ev.failure_mode}`")

    if ev.faithfulness is not None:
        st.subheader("Judged")
        c1, c2 = st.columns(2)
        c1.metric("faithfulness", f"{ev.faithfulness.score:.2f}")
        c2.metric("relevancy", f"{ev.relevancy.score:.2f}")
        if not has_key:
            st.warning(
                "These come from the LEXICAL judge, which scores **50%** on "
                "the validation gate and fails it. They are harness output, "
                "not evidence about the answer. See the judge tab."
            )


# ---------------------------------------------------------------------
# Mode: Ask anything
# ---------------------------------------------------------------------
if mode == "Ask anything":
    st.title("Ask the help centre")
    st.caption(
        "Anything the corpus does not cover *should* produce a refusal. "
        "Try 'How do I buy a gift card?' — deliberately uncovered."
    )

    question = st.text_input("Question", "How many streams does Premium allow?")

    if question:
        pipeline = RAGPipeline(retriever, build_answerer(), depth=DEPTH)
        result = pipeline.run_one(
            {"question_id": "adhoc", "question": question,
             "category": "adhoc", "gt_article_ids": []}
        )

        st.subheader("Answer")
        st.info(result.answer)

        left, right = st.columns([3, 2])
        with left:
            st.subheader(f"Retrieved context — {len(result.hits)} chunks")
            st.caption(
                f"~{context_tokens(result.hits)} estimated tokens handed to the generator."
            )
            for hit in result.hits[:8]:
                with st.expander(
                    f"`{hit.chunk.article_id}` · {hit.chunk.title} "
                    f"(score {hit.score:.2f})"
                ):
                    st.text(hit.chunk.generation_text[:600])
        with right:
            show_evaluation(result)


# ---------------------------------------------------------------------
# Mode: Golden set
# ---------------------------------------------------------------------
elif mode == "Golden set":
    st.title("Golden set — evaluation against ground truth")
    st.caption(
        "120 questions with hand-labelled sources, written BEFORE any "
        "retrieval code existed. A golden set written afterwards describes "
        "your system rather than testing it."
    )

    category = st.selectbox(
        "Category",
        ["single_hop", "multi_hop", "ambiguous", "out_of_scope"],
        help="out_of_scope questions have no answer in the corpus. The "
             "correct behaviour is refusal.",
    )
    subset = [q for q in golden if q["category"] == category]
    picked = st.selectbox(
        "Question", subset,
        format_func=lambda q: f"[{q['question_id']}] {q['question']}",
    )

    if picked:
        if category == "out_of_scope":
            st.warning(
                "**Unanswerable by design.** The corpus is silent on this. "
                "A confident answer here is the most expensive failure this "
                "system can produce."
            )

        pipeline = RAGPipeline(retriever, build_answerer(), depth=DEPTH)
        result = pipeline.run_one(picked)

        st.subheader("Answer")
        st.info(result.answer)

        if picked.get("reference_answer"):
            with st.expander("Reference answer (expert-written)"):
                st.write(picked["reference_answer"])

        left, right = st.columns([3, 2])
        with left:
            st.subheader("Ground truth")
            if picked["gt_article_ids"]:
                got = set(result.retrieved_article_ids)
                for article_id in picked["gt_article_ids"]:
                    mark = "✅" if article_id in got else "❌"
                    st.markdown(f"{mark} `{article_id}`")
            else:
                st.caption("None — the corpus cannot answer this.")

            st.subheader("Retrieved")
            for hit in result.hits[:6]:
                relevant = hit.chunk.article_id in set(picked["gt_article_ids"])
                st.markdown(
                    f"{'🟢' if relevant else '⚪'} `{hit.chunk.article_id}` "
                    f"· {hit.chunk.title}"
                )
        with right:
            show_evaluation(result, picked.get("reference_answer"))


# ---------------------------------------------------------------------
# Mode: judge audit
# ---------------------------------------------------------------------
else:
    st.title("Is the judge trustworthy?")
    st.caption(
        "An LLM judge is a measuring instrument. An uncalibrated instrument "
        "produces numbers, not measurements — so it gets audited before any "
        "score it emits is believed."
    )

    judge = get_judge(None) if not has_key else get_judge(
        __import__("src.llm.provider", fromlist=["get_provider"]).get_provider())

    report = validate_judge(judge)

    st.header("1. Accuracy on cases with known verdicts")
    c1, c2, c3 = st.columns(3)
    c1.metric("overall accuracy", f"{report.overall_accuracy:.0%}")
    c2.metric("cases", len(VALIDATION_CASES))
    c3.metric("trustworthy", "yes" if report.is_trustworthy else "NO")

    if not report.is_trustworthy:
        st.error(
            f"**This judge fails the gate ({report.overall_accuracy:.0%} < 80%).** "
            f"Any faithfulness number it produces is harness output, not "
            f"evidence about answer quality."
        )

    st.caption("Accuracy by case kind:")
    for kind, acc in report.accuracy_by_kind().items():
        st.progress(acc, text=f"{kind} — {acc:.0%}")

    st.info(
        "**The diagnostic case is `contradicted`.** A contradicting sentence "
        "reuses nearly every term of the context it contradicts — high "
        "overlap, opposite meaning. It is the case that separates semantic "
        "judgement from term matching, and it is why a lexical judge cannot "
        "substitute for a model."
    )

    st.header("2. Bias — what else does it respond to?")
    st.write(
        "Accuracy is not enough. A judge can score perfectly on unambiguous "
        "cases and still be useless for comparing prompt variants, because it "
        "may respond to properties unrelated to quality."
    )

    length = measure_length_bias(judge)
    order = measure_order_sensitivity(judge)

    c1, c2 = st.columns(2)
    c1.metric("length bias (mean Δ)", f"{length.mean_delta:+.3f}",
              delta=f"{length.n_material}/{len(length.results)} material",
              delta_color="inverse")
    c2.metric("context-order sensitivity", f"{order.mean_delta:+.3f}",
              delta=f"{order.n_material}/{len(order.results)} material",
              delta_color="inverse")

    if length.is_biased and length.mean_delta < 0:
        st.warning(
            f"**This judge penalises length by {abs(length.mean_delta):.3f}** — the "
            f"*opposite* of the documented LLM-judge tendency to reward "
            f"verbosity. Faithfulness here is the fraction of answer terms "
            f"found in context, so padding drags it down. Two judges with "
            f"opposite biases are both wrong, and neither is fixable by "
            f"choosing a threshold."
        )

    st.caption(
        "Each probe is a **pair** — same claims, same context, same "
        "citations, only phrasing length differs. Correlating score against "
        "length across the golden set would confound bias with quality, "
        "because longer answers may genuinely be more complete."
    )

    with st.expander("See the probes"):
        for r in length.results:
            st.markdown(f"**{r.probe.probe_id}** · Δ {r.delta:+.3f}")
            st.text(f"terse   ({len(r.probe.answer_a.split())}w): "
                    f"{r.probe.answer_a}")
            st.text(f"verbose ({len(r.probe.answer_b.split())}w): "
                    f"{r.probe.answer_b}")
            st.caption(r.probe.note)
            st.divider()

st.sidebar.divider()
st.sidebar.caption(
    "Every panel uses the same `src/` modules the notebooks and CI use — "
    "there is no separate demo path that could disagree with the measured "
    "results."
)
