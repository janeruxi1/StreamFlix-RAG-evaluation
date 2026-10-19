"""The refusal check, validated against answers a person has read.

This file exists because of a measurement error. The first pattern list
recognised the refusal sentence the cited prompts are told to use and
missed the ways a prompt with no refusal instruction declines in its own
words. The naive prompt was reported as refusing 0 of 25 unanswerable
questions; read by hand, it declined 9. The labels hold every such
answer from the credentialed run with a hand label, so the check is now
scored against labels instead of trusted.

A blind second reading changed three labels to declined. Those three
answers decline without any refusal language, so the pattern check
misses them; they are pinned below as known misses, and the evaluation
uses the hand label for them through `refusal_verdict`.
"""
import json
from collections import Counter

import pytest

from src.generation.refusal import (CANONICAL_REFUSAL, HAND_LABELS, check_refusal,
                                    refusal_verdict)

LABELS = json.loads(HAND_LABELS.read_text(encoding="utf-8"))["labels"]

# Declines with no refusal language. A pattern that caught these would
# also catch real answers, so they stay misses and are counted as such.
KNOWN_PATTERN_MISSES = {
    ("naive", "retrieval", "oos-024"),
    ("naive", "all 45 articles", "oos-020"),
    ("naive", "all 45 articles", "oos-024"),
}


def _key(row) -> tuple[str, str, str]:
    return row["prompt"], row["context"], row["question_id"]


def _id(row) -> str:
    return f"{row['prompt']}/{row['context'].split()[0]}/{row['question_id']}"


@pytest.mark.parametrize("row", LABELS, ids=_id)
def test_the_counted_verdict_is_the_hand_label(row):
    assert refusal_verdict(row["answer"]).is_refusal == row["declined"]


def test_the_pattern_check_misses_only_the_known_cases():
    """85 of 88. A new miss, or a pattern that starts catching one of
    these, changes what the memo says about the check."""
    misses = {_key(r) for r in LABELS
              if check_refusal(r["answer"]).is_refusal != r["declined"]}
    assert misses == KNOWN_PATTERN_MISSES
    assert len(LABELS) - len(misses) == 85


def test_known_misses_are_declines_the_check_counts_as_answers():
    for r in LABELS:
        if _key(r) in KNOWN_PATTERN_MISSES:
            assert r["declined"] and not check_refusal(r["answer"]).is_refusal
            assert refusal_verdict(r["answer"]).reviewed


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
    assert declined[("naive", "retrieval")] == 10
    assert declined[("naive", "all 45 articles")] == 8
    assert declined[("grounded", "retrieval")] == 20


def test_unlabelled_answers_fall_through_to_the_pattern_check():
    text = "The extra member add-on costs $5 per month, billed monthly."
    assert refusal_verdict(text) == check_refusal(text)


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
