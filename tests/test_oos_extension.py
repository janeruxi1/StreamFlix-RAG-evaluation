"""The drafted out-of-scope extension is evidence only if every question
in it really is unanswerable from the corpus. These tests are that
claim, checked, so the set cannot rot when an article is edited."""
import re
from collections import Counter

from src.corpus.build import load_corpus, load_golden_set
from src.corpus.oos_extension import load_oos_extension

QUESTIONS = load_oos_extension()
CORPUS_TEXT = " ".join(f"{a.title} {a.body}" for a in load_corpus()).lower()


def test_size_ids_and_shape():
    assert len(QUESTIONS) == 275
    ids = [q["question_id"] for q in QUESTIONS]
    assert len(set(ids)) == len(ids)
    assert all(re.fullmatch(r"oosx-\d{3}", i) for i in ids)
    for q in QUESTIONS:
        assert q["category"] == "out_of_scope"
        assert q["gt_article_ids"] == []
        assert q["tier"] in {"near_miss", "gap", "off_domain"}
        assert len(q["question"].split()) >= 3


def test_no_question_is_repeated_or_already_in_the_golden_set():
    texts = [q["question"].strip().lower() for q in QUESTIONS]
    assert len(set(texts)) == len(texts)
    golden = {q["question"].strip().lower() for q in load_golden_set()}
    assert not golden & set(texts)


def test_the_thing_each_question_asks_about_is_absent_from_the_corpus():
    """A question is out of scope because its subject is not in the help
    centre. If an article later mentions it, the question is answerable
    and scoring a refusal as correct would be wrong."""
    leaks = []
    for q in QUESTIONS:
        assert q["absent_terms"], q["question_id"]
        for term in q["absent_terms"]:
            pattern = rf"(?<![a-z0-9]){re.escape(term.lower())}(?![a-z0-9])"
            if re.search(pattern, CORPUS_TEXT):
                leaks.append((q["question_id"], term))
    assert not leaks, leaks


def test_no_topic_dominates():
    """275 paraphrases of ten questions would be ten data points."""
    per_topic = Counter(q["topic"] for q in QUESTIONS)
    assert max(per_topic.values()) <= 3
    assert len(per_topic) >= 200
