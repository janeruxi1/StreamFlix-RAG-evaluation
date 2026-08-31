"""Retrieval metrics, with uncertainty.

The thing that makes a retrieval bake-off misleading is reporting point
estimates. With 95 in-scope questions, a 3-point difference in recall@5
is roughly three questions changing answer, which is well inside noise.
Ranking a table of configurations by a point estimate and declaring a
winner is how you ship a configuration that is no better than the one it
replaced.

So everything here comes with an interval, and comparisons between
configurations are PAIRED.

Why paired
----------
Every configuration is evaluated on the same questions. That makes the
comparison a within-subjects design: question difficulty is a shared
nuisance factor, and differencing per question removes it. An unpaired
interval on each configuration separately would be dominated by
variation in question difficulty — which is identical across arms and
therefore irrelevant to which arm is better.

Concretely: two configurations might each have wide, heavily overlapping
marginal CIs while the CI on their DIFFERENCE excludes zero, because they
disagree on only a handful of questions and agree everywhere else.
Overlapping marginal CIs do not imply a non-significant difference, and
reading them that way is a common error.

Metrics
-------
    recall@k     fraction of a question's ground-truth articles found
    hit@k        did ANY ground-truth article appear (per-question 0/1)
    MRR          1 / rank of the first correct article — rewards putting
                 the right thing first, which matters because the
                 generator reads top-ranked context most attentively
    nDCG@k       graded, position-discounted; the right metric when a
                 question has several ground-truth articles and finding
                 them EARLY matters

All are computed at ARTICLE level even though retrieval returns chunks,
because ground truth is labelled per article. `hits_to_articles` does the
deduplication, which is why "top-5 chunks" and "top-5 articles" are
different quantities.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from src.retrieval.vectorstore import SearchHit, hits_to_articles


# ---------------------------------------------------------------------
# Per-question scoring
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class QuestionScore:
    """Every metric for one question under one configuration."""
    question_id: str
    category: str
    recall_at_k: float
    hit_at_k: float
    mrr: float
    ndcg_at_k: float
    n_ground_truth: int
    retrieved: tuple[str, ...] = ()

    def get(self, metric: str) -> float:
        return getattr(self, metric)


def _dcg(relevances: list[float]) -> float:
    return sum(rel / math.log2(i + 2) for i, rel in enumerate(relevances))


def score_question(question: dict, hits: list[SearchHit],
                   k: int = 5) -> QuestionScore:
    """Score one question's retrieved chunks against its ground truth."""
    gt = set(question["gt_article_ids"])
    retrieved = hits_to_articles(hits, max_articles=k)

    if not gt:
        # Out-of-scope questions have no retrievable ground truth. They
        # are scored in Phase 5 on REFUSAL, not retrieval — including
        # them here would silently drag every average toward zero.
        raise ValueError(
            f"{question['question_id']} has no ground truth; out-of-scope "
            f"questions are evaluated on refusal behaviour, not retrieval"
        )

    found = gt & set(retrieved)
    rr = next((1.0 / (i + 1) for i, a in enumerate(retrieved) if a in gt), 0.0)

    gains = [1.0 if a in gt else 0.0 for a in retrieved]
    ideal = [1.0] * min(len(gt), k)
    idcg = _dcg(ideal)

    return QuestionScore(
        question_id=question["question_id"],
        category=question["category"],
        recall_at_k=len(found) / len(gt),
        hit_at_k=1.0 if found else 0.0,
        mrr=rr,
        ndcg_at_k=_dcg(gains) / idcg if idcg else 0.0,
        n_ground_truth=len(gt),
        retrieved=tuple(retrieved),
    )


def score_run(questions: list[dict], all_hits: list[list[SearchHit]],
              k: int = 5) -> list[QuestionScore]:
    """Score a full pass over the golden set."""
    if len(questions) != len(all_hits):
        raise ValueError(
            f"question/hit count mismatch: {len(questions)} vs {len(all_hits)}"
        )
    return [score_question(q, h, k) for q, h in zip(questions, all_hits)]


# ---------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------
METRICS = ("recall_at_k", "hit_at_k", "mrr", "ndcg_at_k")


@dataclass(frozen=True)
class ConfigResult:
    """One configuration's aggregate scores, overall and per category."""
    config: str
    n_questions: int
    overall: dict[str, float]
    by_category: dict[str, dict[str, float]]
    scores: list[QuestionScore] = field(repr=False, default_factory=list)

    def metric(self, name: str, category: str | None = None) -> float:
        return (self.overall[name] if category is None
                else self.by_category[category][name])


def aggregate(config: str, scores: list[QuestionScore]) -> ConfigResult:
    """Mean each metric overall and within each question category."""
    def means(subset: list[QuestionScore]) -> dict[str, float]:
        if not subset:
            return {m: float("nan") for m in METRICS}
        return {m: float(np.mean([s.get(m) for s in subset])) for m in METRICS}

    categories = sorted({s.category for s in scores})
    return ConfigResult(
        config=config,
        n_questions=len(scores),
        overall=means(scores),
        by_category={c: means([s for s in scores if s.category == c])
                     for c in categories},
        scores=scores,
    )


# ---------------------------------------------------------------------
# Uncertainty
# ---------------------------------------------------------------------
def bootstrap_ci(scores: list[QuestionScore], metric: str = "recall_at_k",
                 n_boot: int = 2000, alpha: float = 0.05,
                 seed: int = 42) -> tuple[float, float, float]:
    """Percentile bootstrap CI for one configuration's mean.

    Resamples QUESTIONS, not retrieved documents — the questions are the
    sampling unit, and the golden set is one draw from the population of
    questions users might ask.

    Returns (point_estimate, lower, upper).
    """
    values = np.array([s.get(metric) for s in scores], dtype=float)
    if len(values) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(n_boot, len(values)))
    means = values[idx].mean(axis=1)
    return (float(values.mean()),
            float(np.percentile(means, 100 * alpha / 2)),
            float(np.percentile(means, 100 * (1 - alpha / 2))))


@dataclass(frozen=True)
class PairedComparison:
    """Paired bootstrap comparison of two configurations."""
    config_a: str
    config_b: str
    metric: str
    mean_a: float
    mean_b: float
    difference: float          # a - b
    ci_low: float
    ci_high: float
    p_value: float
    n_questions: int
    n_differing: int           # questions where the two actually disagree

    @property
    def significant(self) -> bool:
        """CI on the difference excludes zero."""
        return self.ci_low > 0 or self.ci_high < 0

    def __str__(self) -> str:
        verdict = "significant" if self.significant else "not distinguishable"
        return (f"{self.config_a} - {self.config_b}: "
                f"{self.difference:+.3f} "
                f"[{self.ci_low:+.3f}, {self.ci_high:+.3f}] "
                f"p={self.p_value:.3f}  ({verdict}, "
                f"{self.n_differing}/{self.n_questions} questions differ)")


def paired_bootstrap(a: list[QuestionScore], b: list[QuestionScore],
                     metric: str = "recall_at_k", n_boot: int = 2000,
                     alpha: float = 0.05, seed: int = 42,
                     label_a: str = "A", label_b: str = "B") -> PairedComparison:
    """Paired bootstrap on the per-question difference between two configs.

    Requires both runs to cover the same questions in the same order, and
    checks it — silently comparing misaligned runs would produce a
    confident, meaningless number.

    The p-value is a two-sided bootstrap test of H0: mean difference = 0,
    computed as the proportion of resampled means falling on the other
    side of zero.
    """
    ids_a = [s.question_id for s in a]
    ids_b = [s.question_id for s in b]
    if ids_a != ids_b:
        raise ValueError(
            "paired comparison requires identical question sets in the same "
            "order — the runs being compared are misaligned"
        )

    diffs = np.array([x.get(metric) - y.get(metric) for x, y in zip(a, b)],
                     dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diffs), size=(n_boot, len(diffs)))
    boot = diffs[idx].mean(axis=1)

    observed = float(diffs.mean())
    # two-sided: how often does the bootstrap distribution cross zero
    p = 2.0 * min(float((boot <= 0).mean()), float((boot >= 0).mean()))

    return PairedComparison(
        config_a=label_a, config_b=label_b, metric=metric,
        mean_a=float(np.mean([s.get(metric) for s in a])),
        mean_b=float(np.mean([s.get(metric) for s in b])),
        difference=observed,
        ci_low=float(np.percentile(boot, 100 * alpha / 2)),
        ci_high=float(np.percentile(boot, 100 * (1 - alpha / 2))),
        p_value=min(1.0, p),
        n_questions=len(diffs),
        n_differing=int((diffs != 0).sum()),
    )


# ---------------------------------------------------------------------
# Cost — the axis that makes retrieval depth a trade-off
# ---------------------------------------------------------------------
def context_tokens(hits: list[SearchHit]) -> int:
    """Tokens the generator will actually be handed for one question.

    This is the missing axis in most retrieval bake-offs. recall@k is
    monotonically non-decreasing in k, so sweeping k and ranking by
    recall always crowns the largest k — a conclusion that is arithmetic,
    not evidence.

    Every retrieved chunk costs tokens, and tokens cost money per query
    and dilute the generator's attention with distractors. Measuring cost
    turns "retrieve more" from a free win into a real trade-off.

    Uses `generation_text`, not `embed_text` — the sentence-window
    strategy deliberately passes a wider context than it embeds, and
    charging it for the narrow text would understate what it costs.
    """
    from src.corpus.difficulty import count_tokens
    return sum(count_tokens(h.chunk.generation_text) for h in hits)


@dataclass(frozen=True)
class CostedConfig:
    """One configuration's quality and cost, for frontier analysis."""
    config: str
    recall: float
    mean_tokens: float

    @property
    def recall_per_1k_tokens(self) -> float:
        return 1000.0 * self.recall / self.mean_tokens if self.mean_tokens else 0.0


def pareto_frontier(configs: list[CostedConfig]) -> list[CostedConfig]:
    """Configurations no other configuration strictly beats.

    A config is dominated when another achieves at least as much recall
    for strictly fewer tokens (or strictly more recall for no more
    tokens). What survives is the set of defensible choices; picking
    anything off the frontier means knowingly paying more for less.

    Returned in increasing cost order, so reading down the list shows
    exactly what each additional token of context buys.
    """
    frontier = [
        c for c in configs
        if not any(
            (o.recall >= c.recall and o.mean_tokens < c.mean_tokens)
            or (o.recall > c.recall and o.mean_tokens <= c.mean_tokens)
            for o in configs if o is not c
        )
    ]
    return sorted(frontier, key=lambda c: c.mean_tokens)


def best_under_budget(configs: list[CostedConfig],
                      budget: float) -> CostedConfig | None:
    """Highest-recall configuration fitting a context budget.

    Ties on recall are broken by lower cost — without that, the choice
    among equal-recall configurations is whichever happened to be
    enumerated first, which is not a decision.
    """
    affordable = [c for c in configs if c.mean_tokens <= budget]
    if not affordable:
        return None
    return min(affordable, key=lambda c: (-c.recall, c.mean_tokens))


# ---------------------------------------------------------------------
# Selection honesty
# ---------------------------------------------------------------------
def stratified_split(questions: list[dict], test_fraction: float = 0.4,
                     seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Split the golden set into dev and test, stratified by category.

    Necessary because the bake-off evaluates dozens of configurations on
    95 questions. The maximum of many noisy estimates is biased upward —
    the winner's score is part skill, part luck, and reporting it as an
    expected production number overstates the system.

    Stratifying keeps the rare categories (20 multi-hop, 15 ambiguous)
    represented in both halves; a uniform random split can easily put 2
    of 15 ambiguous questions in test and make that column meaningless.
    """
    rng = np.random.default_rng(seed)
    dev: list[dict] = []
    test: list[dict] = []
    for category in sorted({q["category"] for q in questions}):
        group = [q for q in questions if q["category"] == category]
        order = rng.permutation(len(group))
        n_test = int(round(test_fraction * len(group)))
        for rank, i in enumerate(order):
            (test if rank < n_test else dev).append(group[i])
    return dev, test


def selection_optimism(dev_scores: dict[str, list[QuestionScore]],
                       test_scores: dict[str, list[QuestionScore]],
                       metric: str = "recall_at_k") -> dict[str, float]:
    """Quantify how much the dev-set winner's score was luck.

    Returns the winner's dev score, its test score, and the drop. A large
    drop means the bake-off was mostly fitting noise and the configuration
    ranking should not be trusted beyond its top group.
    """
    dev_means = {c: float(np.mean([s.get(metric) for s in v]))
                 for c, v in dev_scores.items()}
    winner = max(dev_means, key=dev_means.get)
    test_mean = float(np.mean([s.get(metric)
                               for s in test_scores[winner]]))
    all_test = {c: float(np.mean([s.get(metric) for s in v]))
                for c, v in test_scores.items()}
    best_test = max(all_test.values())
    return {
        "winner": winner,
        "dev_score": dev_means[winner],
        "test_score": test_mean,
        "optimism": dev_means[winner] - test_mean,
        "best_possible_test": best_test,
        "regret": best_test - test_mean,
    }
