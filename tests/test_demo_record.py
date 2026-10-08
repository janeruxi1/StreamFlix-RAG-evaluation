"""The demo replays a per-question record of the credentialed run. These
tests are what stops that record drifting from the aggregates the memo is
verified against: every headline number is recomputed from the rows."""
import json
import re
from pathlib import Path

import pytest

from src.corpus.build import load_corpus, load_golden_set
from src.evaluation.demo_record import (
    ARMS,
    RECORD_PATH,
    arm_summary,
    check_against_records,
    load_record,
)

METRICS = Path("reports/metrics")
record = load_record(RECORD_PATH)
pytestmark = pytest.mark.skipif(
    record is None, reason="no credentialed run has written the demo record")
golden = load_golden_set()


def test_one_row_per_golden_question_and_one_entry_per_arm():
    assert set(record["answers"]) == {q["question_id"] for q in golden}
    for qid, row in record["answers"].items():
        assert set(row) == set(ARMS), qid
        for arm, entry in row.items():
            assert entry["answer"].strip(), (qid, arm)
            assert entry["status"] in {"answer", "partial_refusal", "refusal"}


def test_rows_add_up_to_the_records_the_memo_is_checked_against():
    r05 = json.loads((METRICS / "05_judged_arms.json").read_text(encoding="utf-8"))
    r08 = json.loads((METRICS / "08_full_corpus.json").read_text(encoding="utf-8"))
    checks = check_against_records(record, golden, r05, r08)
    assert len(checks) >= 20
    wrong = [(label, expected, actual) for label, expected, actual, ok in checks if not ok]
    assert not wrong, wrong


def test_judged_answers_carry_the_judges_reason():
    """A score with no stated reason cannot be audited by a reader."""
    for qid, row in record["answers"].items():
        for arm in ("rag_naive", "rag_cited", "full_cited"):
            entry = row[arm]
            assert entry["correctness"] is not None and entry["correctness_reason"], (qid, arm)
            if entry["faithfulness"] is not None:
                assert entry["faithfulness_reason"], (qid, arm)
            else:
                assert arm == "full_cited" and entry["is_refusal"], (qid, arm)
        assert row["full_naive"]["correctness"] is None      # generated, not judged


def test_the_cited_arms_cite_only_real_articles_they_were_shown():
    known = {a.article_id for a in load_corpus()}
    for arm in ("rag_cited", "full_cited"):
        summary = arm_summary(record, arm, golden)
        assert summary["citations_not_in_context"] == 0
        assert summary["citations_unknown"] == 0
        for row in record["answers"].values():
            assert set(row[arm]["citations"]) <= known


def test_the_record_holds_no_credential():
    text = RECORD_PATH.read_text(encoding="utf-8")
    assert not re.search(r"sk-[A-Za-z0-9_-]{16,}", text)
    assert "api_key" not in text.lower()
