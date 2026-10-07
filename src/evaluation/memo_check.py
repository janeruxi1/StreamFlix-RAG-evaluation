"""Checking that a document says the numbers the code produces.

Shared by the follow-up notebooks. Phase 7 established the pattern: a
figure in a markdown file is a snapshot that rots silently, so each one
is recomputed and the build fails when the prose and the code disagree.
"""
from __future__ import annotations

from pathlib import Path


def normalise(text: str) -> str:
    """Drop line wrapping and bold markers. A phrase that breaks across
    two lines of markdown still states its number."""
    return " ".join(text.replace("**", "").split())


def missing_claims(path: Path, claims: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """The (label, needle) pairs whose needle is absent from the file."""
    text = normalise(path.read_text(encoding="utf-8"))
    return [(label, needle) for label, needle in claims
            if normalise(needle) not in text]


def report(path: Path, claims: list[tuple[str, str]]) -> int:
    """Print the check and return how many claims failed."""
    missing = missing_claims(path, claims)
    absent = {label for label, _ in missing}
    for label, needle in claims:
        shown = needle if len(needle) <= 44 else needle[:41] + "..."
        print(f"    {label:<34}{shown:<46}{'NO  <-- DRIFT' if label in absent else 'yes'}")
    return len(missing)
