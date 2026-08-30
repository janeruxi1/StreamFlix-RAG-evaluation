"""Materialize the seed corpus to disk and load it back.

The corpus is authored in `seed_articles.py` as structured Python, then
written out as one markdown file per article with YAML front matter.
Two reasons for the round trip rather than embedding straight into the
vector store:

  1. The retrieval pipeline should read documents the way a real one
     would — files on disk, parsed at ingestion — not a Python import.
  2. Markdown files are diffable and reviewable, so a corpus change is
     visible in a pull request.

Usage:
    python -m src.corpus.build           # write data/corpus/*.md
    python -m src.corpus.build --stats   # write, then print a summary
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from src.corpus.seed_articles import (
    ARTICLES,
    COVERAGE_GAPS,
    KNOWN_CORPUS_FLAWS,
)
from src.corpus.golden_set import GOLDEN_QUESTIONS

CORPUS_DIR = Path("data/corpus")
GOLDEN_DIR = Path("data/golden_set")


@dataclass(frozen=True)
class Article:
    """One help-centre article, as loaded from disk."""
    article_id: str
    title: str
    category: str
    last_updated: str
    body: str

    @property
    def word_count(self) -> int:
        return len(self.body.split())

    def as_document(self) -> str:
        """Full text used for embedding — title included for retrieval.

        The title carries strong topical signal that the body sometimes
        assumes. Prepending it measurably helps retrieval on short,
        keyword-light queries.
        """
        return f"# {self.title}\n\n{self.body.strip()}"


# ---------------------------------------------------------------------
# Write
# ---------------------------------------------------------------------
def _front_matter(article: dict) -> str:
    return (
        "---\n"
        f"article_id: {article['article_id']}\n"
        f"title: {article['title']}\n"
        f"category: {article['category']}\n"
        f"last_updated: {article['last_updated']}\n"
        "---\n\n"
    )


def write_corpus(corpus_dir: Path = CORPUS_DIR) -> int:
    """Write every seed article to `corpus_dir` as markdown. Returns count."""
    corpus_dir.mkdir(parents=True, exist_ok=True)
    for article in ARTICLES:
        path = corpus_dir / f"{article['article_id']}.md"
        path.write_text(
            _front_matter(article) + article["body"].strip() + "\n",
            encoding="utf-8",
        )
    return len(ARTICLES)


def write_golden_set(golden_dir: Path = GOLDEN_DIR) -> int:
    """Write the golden question set to JSON. Returns count."""
    golden_dir.mkdir(parents=True, exist_ok=True)
    (golden_dir / "golden_questions.json").write_text(
        json.dumps(GOLDEN_QUESTIONS, indent=2), encoding="utf-8"
    )
    return len(GOLDEN_QUESTIONS)


# ---------------------------------------------------------------------
# Read
# ---------------------------------------------------------------------
def _parse_markdown(text: str) -> Article:
    """Parse a corpus markdown file with YAML-ish front matter."""
    if not text.startswith("---"):
        raise ValueError("Corpus file is missing front matter")
    _, raw_meta, body = text.split("---", 2)
    meta: dict[str, str] = {}
    for line in raw_meta.strip().splitlines():
        key, _, value = line.partition(":")
        meta[key.strip()] = value.strip()
    return Article(
        article_id=meta["article_id"],
        title=meta["title"],
        category=meta["category"],
        last_updated=meta["last_updated"],
        body=body.strip(),
    )


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[Article]:
    """Load every article from disk, sorted by article_id."""
    if not corpus_dir.exists():
        raise FileNotFoundError(
            f"{corpus_dir} not found. Build it first:\n"
            f"    python -m src.corpus.build"
        )
    articles = [
        _parse_markdown(p.read_text(encoding="utf-8"))
        for p in sorted(corpus_dir.glob("*.md"))
    ]
    if not articles:
        raise FileNotFoundError(f"No markdown files in {corpus_dir}")
    return articles


def load_golden_set(golden_dir: Path = GOLDEN_DIR) -> list[dict]:
    """Load the golden question set from disk."""
    path = golden_dir / "golden_questions.json"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Build it first:\n"
            f"    python -m src.corpus.build"
        )
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------
# Integrity checks — run by the Phase 1 audit and by the test suite
# ---------------------------------------------------------------------
def validate_corpus(articles: list[Article],
                    golden: list[dict]) -> list[str]:
    """Return a list of integrity problems. Empty list means clean.

    Checks that would invalidate downstream evaluation if violated:
      - duplicate article ids
      - a golden question citing an article that does not exist
      - an out-of-scope question that mistakenly cites ground truth
      - an in-scope question with no ground truth
      - duplicate question ids
    """
    problems: list[str] = []
    ids = [a.article_id for a in articles]

    dupes = {i for i in ids if ids.count(i) > 1}
    if dupes:
        problems.append(f"Duplicate article_ids: {sorted(dupes)}")

    known = set(ids)
    for q in golden:
        missing = [a for a in q["gt_article_ids"] if a not in known]
        if missing:
            problems.append(
                f"{q['question_id']} cites unknown article(s): {missing}"
            )
        if q["category"] == "out_of_scope" and q["gt_article_ids"]:
            problems.append(
                f"{q['question_id']} is out_of_scope but cites ground truth"
            )
        if q["category"] != "out_of_scope" and not q["gt_article_ids"]:
            problems.append(
                f"{q['question_id']} is in-scope but has no ground truth"
            )

    qids = [q["question_id"] for q in golden]
    qdupes = {q for q in qids if qids.count(q) > 1}
    if qdupes:
        problems.append(f"Duplicate question_ids: {sorted(qdupes)}")

    return problems


def orphan_articles(articles: list[Article],
                    golden: list[dict]) -> list[str]:
    """Articles no golden question exercises.

    Not an error — corpus breadth is realistic and some articles exist
    only as retrieval distractors. Worth reporting so the coverage gap
    is a deliberate choice rather than an oversight.
    """
    cited = {aid for q in golden for aid in q["gt_article_ids"]}
    return sorted(a.article_id for a in articles if a.article_id not in cited)


def _stats() -> None:
    articles = load_corpus()
    golden = load_golden_set()

    print("=" * 66)
    print("  CORPUS SUMMARY")
    print("=" * 66)
    print(f"  Articles: {len(articles)}")
    by_cat: dict[str, int] = {}
    for a in articles:
        by_cat[a.category] = by_cat.get(a.category, 0) + 1
    for cat, n in sorted(by_cat.items()):
        print(f"    {cat:<12} {n:>3}")
    words = [a.word_count for a in articles]
    print(f"  Words: total {sum(words):,}  "
          f"min {min(words)}  median {sorted(words)[len(words)//2]}  "
          f"max {max(words)}")

    print(f"\n  Golden questions: {len(golden)}")
    qc: dict[str, int] = {}
    for q in golden:
        qc[q["category"]] = qc.get(q["category"], 0) + 1
    for cat, n in sorted(qc.items()):
        print(f"    {cat:<14} {n:>3}")

    print(f"\n  Documented flaws: {len(KNOWN_CORPUS_FLAWS)}")
    for f in KNOWN_CORPUS_FLAWS:
        print(f"    [{f['kind']}] {f['flaw_id']} -> {f['article_ids']}")
    print(f"\n  Coverage gaps: {len(COVERAGE_GAPS)}")

    problems = validate_corpus(articles, golden)
    print(f"\n  Integrity problems: {len(problems)}")
    for p in problems:
        print(f"    ✗ {p}")
    if not problems:
        print("    ✓ none")
    print("=" * 66)


if __name__ == "__main__":
    import sys

    n_articles = write_corpus()
    n_golden = write_golden_set()
    print(f"Wrote {n_articles} articles -> {CORPUS_DIR}/")
    print(f"Wrote {n_golden} golden questions -> {GOLDEN_DIR}/")
    if "--stats" in sys.argv:
        print()
        _stats()
