"""Scoring a router: accuracy, macro-F1, calibration, selective routing.

Everything here takes plain arrays, so the notebook can recompute the
reported numbers from the stored per-ticket predictions without the
models that made them.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def accuracy(y: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean(np.asarray(y) == np.asarray(pred)))


def macro_f1(y: np.ndarray, pred: np.ndarray, labels=None) -> float:
    """Mean per-intent F1. Unparsed LLM answers (-1) count as wrong."""
    return float(f1_score(y, pred, average="macro", labels=labels,
                          zero_division=0))


def bootstrap_ci(y: np.ndarray, pred: np.ndarray, n: int = 2000,
                 seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile CI for accuracy, resampling tickets."""
    correct = (np.asarray(y) == np.asarray(pred)).astype(float)
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(correct), size=(n, len(correct)))
    stats = correct[draws].mean(axis=1)
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def paired_bootstrap(y: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray,
                     n: int = 2000, seed: int = 0) -> dict:
    """Accuracy of A minus accuracy of B, resampling the SAME tickets.

    Pairing matters: the two routers are scored on identical questions,
    so most of the per-ticket noise cancels. An unpaired comparison of
    two separate CIs would call real differences "overlapping".
    """
    y = np.asarray(y)
    diff = ((np.asarray(pred_a) == y).astype(float)
            - (np.asarray(pred_b) == y).astype(float))
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(diff), size=(n, len(diff)))
    stats = diff[draws].mean(axis=1)
    lo, hi = np.quantile(stats, [0.025, 0.975])
    # Two-sided: how often the resampled difference lands on the other
    # side of zero, doubled. Floor at 1/n so it never prints as 0.
    p = 2 * min(np.mean(stats <= 0), np.mean(stats >= 0))
    return {"diff": float(diff.mean()), "ci_low": float(lo),
            "ci_high": float(hi), "p_value": float(max(min(p, 1.0), 1 / n))}


def ece(confidence: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error with equal-width confidence bins.

    The weighted gap between how sure the router says it is and how
    often it is right. A router used with a confidence threshold is only
    as good as this number: if "0.95" does not mean 95% right, the
    threshold is routing on a fiction.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    edges = np.linspace(0.0, 1.0, bins + 1)
    which = np.clip(np.digitize(confidence, edges[1:-1], right=True), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        in_bin = which == b
        if in_bin.any():
            gap = abs(confidence[in_bin].mean() - correct[in_bin].mean())
            total += in_bin.mean() * gap
    return float(total)


def threshold_for_target(confidence: np.ndarray, correct: np.ndarray,
                         target: float = 0.95) -> float:
    """Lowest confidence threshold whose auto-routed tickets meet `target`.

    Chosen on validation data, then applied unchanged to test. Tickets at
    or above the threshold are routed automatically; the rest go to a
    person. Lower threshold = more automation, so take the lowest one
    that still meets the accuracy bar. Returns inf if none does.
    """
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    order = np.argsort(-confidence, kind="stable")
    conf_sorted, corr_sorted = confidence[order], correct[order]
    running = np.cumsum(corr_sorted) / np.arange(1, len(corr_sorted) + 1)
    ok = np.flatnonzero(running >= target)
    if not len(ok):
        return float("inf")
    # Ties: a threshold admits every ticket with that confidence, so only
    # stop at positions where the next ticket is strictly less sure.
    last = len(conf_sorted) - 1
    boundary = [i for i in ok if i == last or conf_sorted[i + 1] < conf_sorted[i]]
    if not boundary:
        return float("inf")
    return float(conf_sorted[max(boundary)])


def selective(confidence: np.ndarray, correct: np.ndarray,
              threshold: float) -> dict:
    """Coverage and accuracy of the tickets routed automatically."""
    confidence = np.asarray(confidence, dtype=float)
    correct = np.asarray(correct, dtype=float)
    routed = confidence >= threshold
    return {
        "threshold": float(threshold),
        "coverage": float(routed.mean()),
        "accuracy_routed": float(correct[routed].mean()) if routed.any() else float("nan"),
        "accuracy_to_people": float(correct[~routed].mean()) if (~routed).any() else float("nan"),
    }


def risk_coverage(confidence: np.ndarray, correct: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Coverage and error rate as the threshold is lowered ticket by ticket."""
    order = np.argsort(-np.asarray(confidence, dtype=float), kind="stable")
    corr = np.asarray(correct, dtype=float)[order]
    k = np.arange(1, len(corr) + 1)
    return k / len(corr), 1 - np.cumsum(corr) / k


def confused_pairs(y: np.ndarray, pred: np.ndarray, labels: list[str],
                   top: int = 8) -> list[tuple[str, str, int]]:
    """Most frequent (true, predicted) mistakes."""
    y, pred = np.asarray(y), np.asarray(pred)
    wrong = (y != pred) & (pred >= 0)
    pairs: dict[tuple[int, int], int] = {}
    for t, p in zip(y[wrong], pred[wrong]):
        pairs[(int(t), int(p))] = pairs.get((int(t), int(p)), 0) + 1
    ranked = sorted(pairs.items(), key=lambda kv: (-kv[1], kv[0]))[:top]
    return [(labels[t], labels[p], n) for (t, p), n in ranked]
