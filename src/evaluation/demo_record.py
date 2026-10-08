"""The per-question record behind the demo's "Measured results" view.

The other records in reports/metrics/ are aggregates: enough to verify a
memo, not enough to show anyone an answer. This one keeps every measured
answer with the judge's score and stated reason, so the demo can replay
the credentialed run with no API key and no cost.

A replay is only honest if it is the same run the memo reports. So the
record is never trusted on its own: `check_against_records` recomputes
the headline aggregates from the per-question rows and compares them with
the records notebooks 05 and 08 wrote. CI runs that check, which means
the demo cannot drift from the memo without failing the build.
"""
from __future__ import annotations

import json
from pathlib import Path

RECORD_PATH = Path("reports/metrics/10_measured_answers.json")
FAITHFUL = 0.7

# Display order. `judged` says which scores exist for the arm.
ARMS: dict[str, dict] = {
    "rag_naive": {
        "title": "naive prompt",
        "context": "retrieval (BM25, depth 15)",
        "note": "No instruction to refuse. The control.",
    },
    "rag_cited": {
        "title": "cited prompt",
        "context": "retrieval (BM25, depth 15)",
        "note": "Must cite a source or decline. The arm the memo recommends piloting.",
    },
    "full_cited": {
        "title": "cited prompt, no retrieval",
        "context": "all 45 articles in the prompt",
        "note": "The Phase 8 baseline: does retrieval matter at this size?",
    },
    "full_naive": {
        "title": "naive prompt, no retrieval",
        "context": "all 45 articles in the prompt",
        "note": "Generated but not judged. Shown for its refusal behaviour only.",
    },
}


def _judgement(j) -> tuple[float | None, str]:
    if j is None:
        return None, ""
    return round(float(j.score), 6), (j.reasoning or "").strip()


def answer_entry(result, faithfulness=None, correctness=None, relevancy=None,
                 known_ids: set[str] | None = None) -> dict:
    """One answer, flattened to plain JSON."""
    refusal = result.refusal
    cited = list(result.citations)
    f_score, f_reason = _judgement(faithfulness)
    c_score, c_reason = _judgement(correctness)
    r_score, _ = _judgement(relevancy)
    return {
        "answer": result.answer,
        "status": refusal.label,                    # answer | partial_refusal | refusal
        "is_refusal": bool(refusal.is_refusal),     # what the harness counts
        "words": len(result.answer.split()),
        "citations": cited,
        "citations_not_in_context": sorted(set(cited) - set(result.context_article_ids)),
        "citations_unknown": sorted(set(cited) - (known_ids or set(cited))),
        "faithfulness": f_score,
        "faithfulness_reason": f_reason,
        "correctness": c_score,
        "correctness_reason": c_reason,
        "relevancy": r_score,
    }


def load_record(path: Path = RECORD_PATH) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def arm_summary(record: dict, arm: str, golden: list[dict]) -> dict:
    """Headline aggregates for one arm, recomputed from the rows."""
    category = {q["question_id"]: q["category"] for q in golden}
    rows = {qid: a[arm] for qid, a in record["answers"].items() if arm in a}
    in_scope = [q for q in rows if category[q] != "out_of_scope"]
    oos = [q for q in rows if category[q] == "out_of_scope"]
    answered = [q for q in rows if not rows[q]["is_refusal"]]
    answered_in = [q for q in in_scope if not rows[q]["is_refusal"]]
    judged = [q for q in answered if rows[q]["faithfulness"] is not None]
    scored = [q for q in rows if rows[q]["correctness"] is not None]
    return {
        "n": len(rows),
        "out_of_scope_n": len(oos),
        "out_of_scope_refused": sum(rows[q]["is_refusal"] for q in oos),
        "out_of_scope_answers_unsupported": sum(
            1 for q in oos if not rows[q]["is_refusal"]
            and rows[q]["faithfulness"] is not None
            and rows[q]["faithfulness"] < FAITHFUL),
        "in_scope_n": len(in_scope),
        "in_scope_answered": len(answered_in),
        "in_scope_answers_unfaithful": sum(
            1 for q in answered_in if rows[q]["faithfulness"] is not None
            and rows[q]["faithfulness"] < FAITHFUL),
        "faithfulness_answered": _mean([rows[q]["faithfulness"] for q in judged]),
        "correctness": _mean([rows[q]["correctness"] for q in scored]),
        "correctness_in_scope": _mean(
            [rows[q]["correctness"] for q in in_scope if q in scored]),
        "correctness_out_of_scope": _mean(
            [rows[q]["correctness"] for q in oos if q in scored]),
        "citations_not_in_context": sum(
            len(rows[q]["citations_not_in_context"]) for q in rows),
        "citations_unknown": sum(len(rows[q]["citations_unknown"]) for q in rows),
    }


def agreement(record: dict, golden: list[dict], a: str = "rag_cited",
              b: str = "full_cited") -> dict:
    """Which answerable questions each of two arms answered."""
    out = {"both_answered": 0, "both_refused": 0, "only_a_answered": 0,
           "only_b_answered": 0}
    for q in golden:
        if q["category"] == "out_of_scope":
            continue
        row = record["answers"][q["question_id"]]
        ra, rb = row[a]["is_refusal"], row[b]["is_refusal"]
        key = ("both_refused" if ra and rb else "both_answered" if not ra and not rb
               else "only_a_answered" if not ra else "only_b_answered")
        out[key] += 1
    return out


def check_against_records(record: dict, golden: list[dict], r05: dict,
                          r08: dict) -> list[tuple[str, object, object, bool]]:
    """(label, expected from 05/08, recomputed from rows, match) per check."""
    naive = arm_summary(record, "rag_naive", golden)
    cited = arm_summary(record, "rag_cited", golden)
    full = arm_summary(record, "full_cited", golden)
    full_naive = arm_summary(record, "full_naive", golden)
    agree = agreement(record, golden)
    ref5, runs5 = r05["refusals"], r05["runs"]
    f8, a8 = r08["full_corpus"], r08["in_scope_agreement"]

    pairs = [
        ("questions", r05["n_questions"], len(record["answers"])),
        ("cited: unanswerable refused", ref5["llm_cited"]["out_of_scope_refused"], cited["out_of_scope_refused"]),
        ("cited: answerable answered", ref5["llm_cited"]["in_scope_answered"], cited["in_scope_answered"]),
        ("cited: answers below faithfulness threshold", ref5["llm_cited"]["in_scope_answers_unfaithful"], cited["in_scope_answers_unfaithful"]),
        ("cited: faithfulness, answered", runs5["llm_cited"]["faithfulness_answered"], cited["faithfulness_answered"]),
        ("cited: correctness, all questions", runs5["llm_cited"]["correctness"], cited["correctness"]),
        ("cited: citations of articles never shown", 0, cited["citations_not_in_context"]),
        ("naive: unanswerable refused", ref5["llm_naive"]["out_of_scope_refused"], naive["out_of_scope_refused"]),
        ("naive: unanswerable answers unsupported", ref5["llm_naive"]["out_of_scope_answers_judged_unsupported"], naive["out_of_scope_answers_unsupported"]),
        ("naive: faithfulness, answered", runs5["llm_naive"]["faithfulness_answered"], naive["faithfulness_answered"]),
        ("naive: correctness, all questions", runs5["llm_naive"]["correctness"], naive["correctness"]),
        ("full corpus: unanswerable refused", f8["out_of_scope_refused"], full["out_of_scope_refused"]),
        ("full corpus: answerable answered", f8["in_scope_answered"], full["in_scope_answered"]),
        ("full corpus: answers below threshold", f8["in_scope_answers_unfaithful"], full["in_scope_answers_unfaithful"]),
        ("full corpus: faithfulness, answered", f8["faithfulness_answered"], full["faithfulness_answered"]),
        ("full corpus: correctness, answerable", f8["correctness_in_scope"], full["correctness_in_scope"]),
        ("full corpus: correctness, unanswerable", f8["correctness_out_of_scope"], full["correctness_out_of_scope"]),
        ("full corpus, naive: unanswerable refused", r08["full_corpus_naive_out_of_scope_refused"], full_naive["out_of_scope_refused"]),
        ("both arms answered", a8["both_answered"], agree["both_answered"]),
        ("both arms refused", a8["both_refused"], agree["both_refused"]),
        ("only retrieval answered", a8["only_retrieval_answered"], agree["only_a_answered"]),
        ("only the full corpus answered", a8["only_full_corpus_answered"], agree["only_b_answered"]),
    ]
    out = []
    for label, expected, actual in pairs:
        if isinstance(expected, float) or isinstance(actual, float):
            ok = actual is not None and abs(float(expected) - float(actual)) < 1e-5
            shown = None if actual is None else round(float(actual), 6)
        else:
            ok, shown = expected == actual, actual
        out.append((label, expected, shown, ok))
    return out
