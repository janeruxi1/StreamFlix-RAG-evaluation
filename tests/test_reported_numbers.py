"""The headline numbers in the README and the decision memo are read from the recorded results.

Each check formats a value from reports/metrics/ the way the docs write it and asserts that the
text appears in the doc, so a rerun that changes a result fails CI until the docs are updated.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
README = (ROOT / "README.md").read_text(encoding="utf-8")
MEMO = (ROOT / "reports" / "decision_memo.md").read_text(encoding="utf-8")


def rec(name):
    return json.loads((ROOT / "reports" / "metrics" / f"{name}.json").read_text(encoding="utf-8"))


R3, R5, R6, R8 = rec("03_retrieval_arms"), rec("05_judged_arms"), rec("06_judge_audit"), rec("08_full_corpus")
CITED, NAIVE = R5["refusals"]["llm_cited"], R5["refusals"]["llm_naive"]
RAG = R8["rag"]
IN_SCOPE = R5["paired_correctness_by_scope"]["llm_naive - llm_cited (in-scope)"]
FULL = R8["full_minus_rag"]["correctness_in_scope"]
PICK = R3["budget_pick_vs_bm25"]
VAL = R5["judge_validation"]


def signed(x, d=3):
    return f"{x:+.{d}f}"


README_CLAIMS = [
    f"`cited` refused **{CITED['out_of_scope_refused']} of {CITED['out_of_scope_n']}**",
    f"`naive` declined **{NAIVE['out_of_scope_refused']} of {NAIVE['out_of_scope_n']}**",
    f"{NAIVE['out_of_scope_answers_judged_unsupported']} of them with unsupported claims",
    f"**{RAG['out_of_scope_refusal_wilson_lower_95']:.1%}**",
    f"**{RAG['answers_citing_an_unknown_article']}** fabricated citations",
    f"**{RAG['in_scope_answers_unfaithful']} of {RAG['in_scope_answered']}**",
    f"(faithfulness {RAG['faithfulness_answered']:.3f})",
    f"**{CITED['in_scope_refused']} of {RAG['in_scope_n']}**",
    f"{signed(IN_SCOPE['difference'])} [{signed(IN_SCOPE['ci_low'])}, {signed(IN_SCOPE['ci_high'])}]",
    f"{signed(FULL['difference'])} [{FULL['ci_low']:+.3f}, {signed(FULL['ci_high'])}]".replace("+-", "-"),
    f"{R8['full_corpus']['mean_context_tokens_estimated'] / RAG['mean_context_tokens_estimated']:.1f}x the context",
    f"{R3['shipping']['recall']:.3f} context recall for about {R3['shipping']['tokens']:.0f} tokens",
    f"{signed(PICK['difference'])} [{PICK['ci_low']:+.3f}, {signed(PICK['ci_high'])}]".replace("+-", "-"),
    f"{R3['n_configurations']} configurations",
    f"**{VAL['overall_accuracy']:.0%} on {VAL['n_cases']} cases**",
]


@pytest.mark.parametrize("text", README_CLAIMS, ids=[c[:40] for c in README_CLAIMS])
def test_readme_quotes_the_recorded_result(text):
    assert text in README, f"README should contain {text!r}"


def test_memo_uses_the_same_refusal_and_faithfulness_results():
    for text in (f"{CITED['out_of_scope_refused']} of {CITED['out_of_scope_n']}",
                 f"{RAG['out_of_scope_refusal_wilson_lower_95']:.1%}",
                 f"{RAG['in_scope_answers_unfaithful']} of {RAG['in_scope_answered']}"):
        assert text in MEMO, f"decision memo should contain {text!r}"


def test_the_judge_was_trusted_before_use():
    assert VAL["trustworthy"] and VAL["parse_failures"] == 0
