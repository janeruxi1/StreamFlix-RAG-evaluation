"""Ticket routing (notebook 11): data splits, metrics, LLM reply parsing.

No torch and no key: these run in CI's minimal install. The fine-tune
loop has its own file, skipped when torch is absent.
"""
import json

import numpy as np
import pandas as pd
import pytest

from src.routing import baselines, data, llm_router, metrics


# ---------------------------------------------------------------- data
def test_labels_and_splits_are_complete_and_disjoint():
    labels = data.load_labels()
    sp = data.splits()
    assert len(labels) == 77 and len(set(labels)) == 77
    assert len(sp["fit"]) + len(sp["val"]) == 10003
    assert len(sp["test"]) == 3080
    assert not set(sp["fit"].text) & set(sp["val"].text)
    # every intent is in every split
    for part in sp.values():
        assert part.label.nunique() == 77


def test_split_is_stratified_and_fixed():
    a, b = data.splits(), data.splits()
    assert a["val"].text.tolist() == b["val"].text.tolist()
    share = a["val"].label.value_counts() / (
        a["val"].label.value_counts() + a["fit"].label.value_counts())
    assert share.between(0.07, 0.13).all()


def test_llm_subset_is_balanced():
    test = data.load_split("test")
    rows = data.llm_subset(test)
    assert len(rows) == 77 * data.LLM_PER_CLASS
    assert (test.label.iloc[rows].value_counts() == data.LLM_PER_CLASS).all()


def test_stratified_indices_rejects_oversampling():
    with pytest.raises(ValueError):
        data.stratified_indices(np.array([0, 0, 1]), 3, seed=0)


def test_unknown_split_name():
    with pytest.raises(ValueError):
        data.load_split("validation")


# ------------------------------------------------------------- metrics
def test_accuracy_and_macro_f1():
    y = np.array([0, 0, 1, 1])
    assert metrics.accuracy(y, np.array([0, 0, 1, 0])) == 0.75
    assert metrics.macro_f1(y, y) == 1.0
    # an unparsed answer (-1) is simply wrong
    assert metrics.macro_f1(y, np.array([0, 0, 1, -1]), labels=[0, 1]) < 1.0


def test_paired_bootstrap_sign_and_ci():
    rng = np.random.default_rng(1)
    y = rng.integers(0, 5, 2000)
    good = y.copy()
    bad = y.copy()
    bad[:200] = (bad[:200] + 1) % 5
    out = metrics.paired_bootstrap(y, good, bad, n=500)
    assert out["diff"] == pytest.approx(0.10)
    assert out["ci_low"] > 0.08 and out["ci_high"] < 0.12
    assert out["p_value"] <= 1 / 500 + 1e-12
    same = metrics.paired_bootstrap(y, good, good, n=200)
    assert same["diff"] == 0 and same["p_value"] == 1.0


def test_bootstrap_ci_brackets_accuracy():
    y = np.zeros(1000, dtype=int)
    pred = np.where(np.arange(1000) < 900, 0, 1)
    lo, hi = metrics.bootstrap_ci(y, pred, n=500)
    assert lo < 0.9 < hi


def test_ece_perfect_and_overconfident():
    correct = np.array([1, 1, 0, 0] * 50)
    assert metrics.ece(np.full(200, 0.5), correct) == pytest.approx(0.0)
    assert metrics.ece(np.full(200, 0.99), correct) == pytest.approx(0.49)


def test_threshold_for_target_picks_lowest_qualifying():
    conf = np.array([0.99, 0.95, 0.90, 0.80, 0.70, 0.60])
    corr = np.array([1, 1, 1, 0, 1, 0])
    # top 3 are all right; adding the 4th drops to 75%
    assert metrics.threshold_for_target(conf, corr, 1.0) == 0.90
    assert metrics.threshold_for_target(conf, corr, 0.80) == 0.70
    assert metrics.threshold_for_target(conf, np.zeros(6), 0.5) == float("inf")


def test_threshold_respects_ties():
    # two tickets share 0.9; admitting one admits both
    conf = np.array([0.95, 0.9, 0.9])
    corr = np.array([1, 1, 0])
    assert metrics.threshold_for_target(conf, corr, 1.0) == 0.95


def test_selective_and_risk_coverage():
    conf = np.array([0.9, 0.8, 0.3, 0.2])
    corr = np.array([1, 1, 0, 1])
    s = metrics.selective(conf, corr, 0.5)
    assert s["coverage"] == 0.5 and s["accuracy_routed"] == 1.0
    assert s["accuracy_to_people"] == 0.5
    cov, risk = metrics.risk_coverage(conf, corr)
    assert cov[-1] == 1.0 and risk[0] == 0.0 and risk[-1] == 0.25


def test_confused_pairs_counts_and_ignores_unparsed():
    labels = ["a", "b", "c"]
    y = np.array([0, 0, 0, 1, 2])
    pred = np.array([1, 1, -1, 0, 2])
    assert metrics.confused_pairs(y, pred, labels) == [("a", "b", 2), ("b", "a", 1)]


# ------------------------------------------------------------ baselines
def test_tfidf_router_learns_a_toy_problem():
    texts = ["card not arrived", "where is my card", "card still not here",
             "exchange rate today", "what rate do you use", "currency rate"] * 5
    y = [0, 0, 0, 1, 1, 1] * 5
    model = baselines.tfidf_router().fit(texts, y)
    pred, conf = baselines.predict(model, ["my card has not arrived",
                                           "which exchange rate"])
    assert pred.tolist() == [0, 1]
    assert ((conf > 0.5) & (conf <= 1)).all()


# ----------------------------------------------------------- llm parse
LABELS = ["card_arrival", "card_payment_fee_charged", "card_payment_not_recognised",
          "Refund_not_showing_up", "top_up_failed"]


@pytest.mark.parametrize("reply,expected", [
    ("card_arrival", 0),
    ("  `card_arrival`.\n", 0),
    ("- card arrival", 0),
    ("refund_not_showing_up", 3),
    ("The intent is card_payment_fee_charged.", 1),
    ("card_payment_not_recognized", 2),           # US spelling, unambiguous
    ("transfer_timing", llm_router.UNPARSED),
    ("", llm_router.UNPARSED),
])
def test_parse_label(reply, expected):
    assert llm_router.parse_label(reply, LABELS) == expected


def test_prompt_lists_every_intent_and_the_text():
    prompt = llm_router.build_prompt("  Where is my card? ", LABELS)
    assert all(f"- {lab}" in prompt for lab in LABELS)
    assert "Message: Where is my card?" in prompt


def test_cost_estimate_scales_with_tickets():
    one = llm_router.estimate_cost(["hello"], LABELS, "gpt-4o-mini")
    ten = llm_router.estimate_cost(["hello"] * 10, LABELS, "gpt-4o-mini")
    assert ten == pytest.approx(10 * one) and one > 0
    assert np.isnan(llm_router.estimate_cost(["x"], LABELS, "unknown-model"))


class FakeProvider:
    """Answers from a lookup, like a cached provider would."""

    name = "fake"
    default_model = "fake-model"

    def __init__(self, answers):
        self.answers, self.calls = answers, 0

    def complete(self, prompt, model=None, max_tokens=512, temperature=0.0):
        self.calls += 1
        assert temperature == 0.0 and max_tokens <= 32
        text = prompt.split("Message: ")[1].split("\n")[0]
        return self.answers[text]


def test_route_with_fake_provider():
    provider = FakeProvider({"card late": "card_arrival",
                             "fee?": "card_payment_fee_charged",
                             "huh": "I am not sure"})
    out = llm_router.route(provider, ["card late", "fee?", "huh"], LABELS,
                           log=lambda *_: None)
    assert out["pred"] == [0, 1, llm_router.UNPARSED]
    assert provider.calls == 3
    assert out["live_seconds"] == []          # instant replies count as cached


# ------------------------------------------------- committed predictions
def test_committed_predictions_are_well_formed():
    from pathlib import Path
    path = Path("reports/metrics/11_routing_predictions.csv")
    if not path.exists():
        pytest.skip("no routing run committed yet")
    preds = pd.read_csv(path)
    record = json.loads(Path("reports/metrics/11_routing.json").read_text())
    sp = data.splits()
    assert set(preds.columns) == {"split", "idx", "arm", "seed", "pred", "conf"}
    assert set(preds.arm) <= {"tfidf", "frozen", "finetune", "llm"}
    assert record["labels"] == data.load_labels()
    for (arm, seed, split), grp in preds.groupby(["arm", "seed", "split"]):
        assert grp.idx.is_unique
        assert grp.idx.max() < len(sp[split])
        assert grp.pred.between(-1, 76).all()
        if arm != "llm":
            assert len(grp) == len(sp[split])
            assert grp.conf.between(0, 1).all()
