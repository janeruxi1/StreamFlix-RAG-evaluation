"""The refusal check, validated against answers a person has read.

This file exists because of a measurement error. The first pattern list
recognised the refusal sentence the cited prompts are told to use and
missed the ways a prompt with no refusal instruction declines in its own
words. The naive prompt was reported as refusing 0 of 25 unanswerable
questions; read by hand, it declined 9. The fixture holds every such
answer from the credentialed run with a hand label, so the check is now
scored against labels instead of trusted.
"""
import json
from collections import Counter
from pathlib import Path

import pytest

from src.generation.refusal import CANONICAL_REFUSAL, check_refusal

FIXTURE = Path(__file__).parent / "fixtures" / "refusal_hand_labels.json"
LABELS = json.loads(FIXTURE.read_text(encoding="utf-8"))["labels"]


def _id(row) -> str:
    return f"{row['prompt']}/{row['context'].split()[0]}/{row['question_id']}"


@pytest.mark.parametrize("row", LABELS, ids=_id)
def test_check_agrees_with_the_hand_label(row):
    verdict = check_refusal(row["answer"])
    assert verdict.is_refusal == row["declined"], (verdict.label, row["answer"][:160])


def test_the_fixture_covers_every_unanswerable_question_for_each_arm():
    """25 unanswerable questions, three arms with no refusal instruction."""
    seen = Counter((r["prompt"], r["context"]) for r in LABELS
                   if r["category"] == "out_of_scope")
    assert seen == {("naive", "retrieval"): 25, ("grounded", "retrieval"): 25,
                    ("naive", "all 45 articles"): 25}


def test_hand_counts_are_the_ones_the_memo_reports():
    """If these change, the memo's refusal counts for the naive prompt
    have changed too and it needs re-reading, not just re-running."""
    declined = Counter((r["prompt"], r["context"]) for r in LABELS
                       if r["category"] == "out_of_scope" and r["declined"])
    assert declined[("naive", "retrieval")] == 9
    assert declined[("naive", "all 45 articles")] == 6
    assert declined[("grounded", "retrieval")] == 20


@pytest.mark.parametrize("answer", [
    "The provided context does not include information about gift cards. "
    "Please check the StreamFlix website.",
    "The context provided does not specify the cost of the add-on.",
    "I'm sorry, but the information provided does not include any details "
    "about investing in StreamFlix.",
    "There is no mention of an education plan in the provided information.",
    "I don't have information about hiring. Please check the official website.",
    "I'm sorry, but I can't assist with that.",
    "I'm sorry, but I can't provide information about the weather. However, "
    "if you have any questions about your StreamFlix account or services, "
    "feel free to ask!",
    CANONICAL_REFUSAL,
])
def test_declines_in_the_models_own_words_are_refusals(answer):
    assert check_refusal(answer).is_refusal


@pytest.mark.parametrize("answer", [
    # disclaims, then asserts anyway
    "The provided context does not mention audio descriptions. However, audio "
    "descriptions are typically available for certain titles and can be "
    "enabled in the audio settings of the streaming service.",
    # an answer that happens to deny something
    "No, the basic Fire TV Stick outputs a maximum of 1080p and does not "
    "support 4K playback.",
    "No, deleting your account does not automatically stop billing. You must "
    "cancel any active subscription separately first.",
    "No, you cannot use StreamFlix on Xbox 360. The app was retired in 2024.",
    "The extra member add-on costs $5 per month.",
])
def test_answers_that_assert_something_are_not_refusals(answer):
    assert not check_refusal(answer).is_refusal
