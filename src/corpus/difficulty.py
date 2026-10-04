"""Benchmark difficulty analysis — how hard is this golden set, really?

Motivation
----------
A RAG project that reports "our retriever achieves 94% recall" without
saying what a trivial baseline achieves has reported nothing. If keyword
matching already gets 91%, the dense retriever earned 3 points, not 94.

This module answers two questions before any embedding model is trained
or any API key is spent:

  1. What does a pure lexical baseline (BM25) score on this golden set?
     That number is the floor every later result must be compared to.

  2. Do the questions actually avoid keyword-matching their ground-truth
     articles? If questions echo article titles, retrieval is trivial
     and the benchmark measures nothing.

BM25 is implemented here rather than pulled from a library so the
scoring is inspectable and has no extra dependency. It is the standard
Okapi BM25 with k1=1.5, b=0.75.

Token counts come in two kinds, kept deliberately apart. Anything that
DECIDES something — a chunk boundary, a context budget, a number the memo
asserts — uses `estimate_tokens`, which is pure arithmetic and identical
on every machine. `exact_token_count` uses tiktoken when it is installed
and is for display only. See the note above `estimate_tokens` for the
bug that separation fixes.
"""
from __future__ import annotations

import math
import re
import statistics
from collections import Counter
from dataclasses import dataclass

# Common English function words. Removed before scoring so that overlap
# reflects topical content, not grammar.
_STOPWORDS = set("""
a an the is are was were be been being do does did doing have has had having
i you he she it we they me him her us them my your his its our their this that
these those what which who whom how when where why can could will would shall
should may might must of in on at to for with about from by as if then than so
and or but not no nor too very just get got make makes made want wants need
needs there here also more most some any all each other into out up down over
under again further once only own same such
""".split())


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokens, stopwords and 1-2 char words removed."""
    return [
        w for w in re.findall(r"[a-z0-9]+", text.lower())
        if w not in _STOPWORDS and len(w) > 2
    ]


# ---------------------------------------------------------------------
# BM25
# ---------------------------------------------------------------------
class BM25:
    """Okapi BM25 over a small in-memory corpus.

    Deliberately dependency-free and readable: this is the baseline every
    later retrieval result gets measured against, so it should be
    auditable rather than a black box.
    """

    def __init__(self, documents: dict[str, str], k1: float = 1.5,
                 b: float = 0.75):
        self.k1, self.b = k1, b
        self.docs = {doc_id: tokenize(text)
                     for doc_id, text in documents.items()}
        self.doc_len = {i: len(d) for i, d in self.docs.items()}
        self.avgdl = (sum(self.doc_len.values()) / len(self.docs)
                      if self.docs else 0.0)

        n_docs = len(self.docs)
        doc_freq: Counter[str] = Counter()
        for tokens in self.docs.values():
            doc_freq.update(set(tokens))
        self.idf = {
            term: math.log((n_docs - freq + 0.5) / (freq + 0.5) + 1)
            for term, freq in doc_freq.items()
        }
        self._tf = {i: Counter(d) for i, d in self.docs.items()}

    def score(self, query_tokens: list[str], doc_id: str) -> float:
        tf, dl = self._tf[doc_id], self.doc_len[doc_id]
        total = 0.0
        for term in query_tokens:
            freq = tf.get(term, 0)
            if not freq:
                continue
            denom = freq + self.k1 * (1 - self.b + self.b * dl / self.avgdl)
            total += self.idf.get(term, 0.0) * (freq * (self.k1 + 1)) / denom
        return total

    def rank(self, query: str, top_k: int | None = None) -> list[str]:
        """Document ids ordered by descending relevance."""
        q = tokenize(query)
        ordered = sorted(self.docs, key=lambda d: -self.score(q, d))
        return ordered[:top_k] if top_k else ordered


# ---------------------------------------------------------------------
# Baseline evaluation
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class BaselineResult:
    """Retrieval scores for one question category."""
    category: str
    n_questions: int
    hit_at_1: float      # fraction with ANY ground-truth doc in top 1
    hit_at_3: float
    hit_at_5: float
    recall_at_5: float   # mean fraction of ground-truth docs found in top 5

    def __str__(self) -> str:
        return (f"{self.category:<14} n={self.n_questions:<4} "
                f"hit@1={self.hit_at_1:>6.1%}  hit@3={self.hit_at_3:>6.1%}  "
                f"hit@5={self.hit_at_5:>6.1%}  recall@5={self.recall_at_5:>6.1%}")


def evaluate_lexical_baseline(
    documents: dict[str, str],
    golden: list[dict],
) -> tuple[list[BaselineResult], BaselineResult]:
    """Score BM25 on the golden set, per category and overall.

    Out-of-scope questions are excluded — they have no ground truth, so
    retrieval metrics are undefined for them. Their handling is a
    generation-side concern measured in Phase 6.

    Returns (per_category_results, overall_result).
    """
    bm25 = BM25(documents)
    in_scope = [q for q in golden if q["category"] != "out_of_scope"]

    by_cat: dict[str, list[dict]] = {}
    for q in in_scope:
        by_cat.setdefault(q["category"], []).append(q)

    def _score(questions: list[dict], label: str) -> BaselineResult:
        hits = {1: 0, 3: 0, 5: 0}
        recalls: list[float] = []
        for q in questions:
            ranked = bm25.rank(q["question"], top_k=5)
            gt = set(q["gt_article_ids"])
            for k in hits:
                if gt & set(ranked[:k]):
                    hits[k] += 1
            recalls.append(len(gt & set(ranked[:5])) / len(gt))
        n = len(questions)
        return BaselineResult(
            category=label,
            n_questions=n,
            hit_at_1=hits[1] / n,
            hit_at_3=hits[3] / n,
            hit_at_5=hits[5] / n,
            recall_at_5=statistics.mean(recalls),
        )

    order = ["single_hop", "multi_hop", "ambiguous"]
    per_cat = [_score(by_cat[c], c) for c in order if c in by_cat]
    overall = _score(in_scope, "ALL in-scope")
    return per_cat, overall


# ---------------------------------------------------------------------
# Question / article lexical overlap
# ---------------------------------------------------------------------
@dataclass(frozen=True)
class OverlapReport:
    """How much questions echo the wording of their ground-truth articles.

    High overlap means the benchmark is testing string matching rather
    than retrieval quality.
    """
    mean_title_overlap: float
    zero_overlap_count: int
    n_questions: int

    @property
    def zero_overlap_rate(self) -> float:
        return self.zero_overlap_count / self.n_questions


def question_article_overlap(
    titles: dict[str, str],
    golden: list[dict],
) -> OverlapReport:
    """Fraction of a question's content words that appear in its
    ground-truth article titles."""
    in_scope = [q for q in golden if q["category"] != "out_of_scope"]
    overlaps: list[float] = []
    for q in in_scope:
        q_terms = set(tokenize(q["question"]))
        title_terms: set[str] = set()
        for aid in q["gt_article_ids"]:
            title_terms |= set(tokenize(titles[aid]))
        overlaps.append(len(q_terms & title_terms) / max(len(q_terms), 1))
    return OverlapReport(
        mean_title_overlap=statistics.mean(overlaps),
        zero_overlap_count=sum(1 for o in overlaps if o == 0.0),
        n_questions=len(in_scope),
    )


# ---------------------------------------------------------------------
# Near-duplicate quantification
# ---------------------------------------------------------------------
def jaccard(a: str, b: str) -> float:
    """Token-set Jaccard similarity — a dependency-free stand-in for
    embedding cosine, adequate for confirming near-duplication."""
    ta, tb = set(tokenize(a)), set(tokenize(b))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def most_similar_pairs(documents: dict[str, str],
                       top_n: int = 5) -> list[tuple[str, str, float]]:
    """The `top_n` most lexically similar document pairs.

    Used to confirm the planted near-duplicate cluster really is the most
    confusable region of the corpus, rather than an unverified claim.
    """
    ids = sorted(documents)
    pairs = [
        (a, b, jaccard(documents[a], documents[b]))
        for i, a in enumerate(ids) for b in ids[i + 1:]
    ]
    return sorted(pairs, key=lambda p: -p[2])[:top_n]


# ---------------------------------------------------------------------
# Token counting — Phase 2 needs tokens, not words
# ---------------------------------------------------------------------
# There used to be one function here, `count_tokens`, which asked
# tiktoken for an exact count and silently fell back to a word-based
# estimate when tiktoken was missing or could not load its encoding.
#
# That fallback was not cosmetic. The count decides which markdown
# sections get merged, so the SAME code produced 202 chunks without
# tiktoken and 209 with it. CI has no tiktoken; `pip install -r
# requirements.txt` installs it. Every downstream number therefore
# depended on an optional package, the memo's verification passed in CI
# and described a pipeline nobody following the README would get, and
# nothing raised an error at any point.
#
# The fix is to stop letting an optional dependency make decisions:
#
#   estimate_tokens     deterministic, dependency-free. Used for every
#                       chunk boundary, every context cost, and every
#                       number the memo asserts.
#   exact_token_count   tiktoken when available, else None. Display only.
#
# The estimate runs low. Measured on this corpus it gives 7,888 tokens
# against tiktoken's 8,323, so exact counts are about 5.5% higher. Token
# figures in this project are estimates and are labelled as such.
TOKENS_PER_WORD = 1.3


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate: whitespace words x 1.3, truncated.

    Identical on every machine and every run, which is the property that
    matters for anything a result depends on. It is an estimate, not a
    tokenizer — see the calibration note above.
    """
    return int(len(text.split()) * TOKENS_PER_WORD)


def exact_token_count(text: str, model: str = "cl100k_base") -> int | None:
    """Exact count via tiktoken, or None when it is unavailable.

    For display only. Returning None rather than falling back is the
    point: a caller cannot mistake an estimate for an exact count, and
    cannot build behaviour on a value that changes with the environment.
    """
    try:
        import tiktoken
        return len(tiktoken.get_encoding(model).encode(text))
    except Exception:
        return None


def tokenizer_available() -> bool:
    """Whether exact token counts are available in this environment."""
    return exact_token_count("probe") is not None
