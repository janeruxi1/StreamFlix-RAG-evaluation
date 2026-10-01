from src.generation.demo import answer_question


def test_in_scope_question_is_answered_with_citation():
    res = answer_question("My card was declined. What happens to my account?")
    assert not res.refused
    assert "[bill-004]" in res.answer
    assert res.sources


def test_out_of_scope_question_is_refused():
    assert answer_question("How do I buy a StreamFlix gift card?").refused
