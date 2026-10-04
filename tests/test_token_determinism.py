"""Results must not depend on whether tiktoken is installed.

The bug these tests exist for: token counting used tiktoken when it was
present and silently fell back to a word estimate when it was not. The
count decides which markdown sections merge, so the same code produced
202 chunks in CI (no tiktoken) and 209 on a machine that had run
`pip install -r requirements.txt`. Every downstream number moved with
it, the memo's verification passed in CI regardless, and nothing raised.

A test that only runs in one environment cannot see that. So these tests
SIMULATE both environments inside one process — a fake tiktoken that
returns wildly different counts, and a blocked import — and require that
nothing behavioural changes between them.
"""
import sys
import types
from pathlib import Path

import pytest

from src.corpus.build import load_corpus, load_golden_set
from src.corpus.difficulty import (
    estimate_tokens,
    exact_token_count,
    tokenizer_available,
)
from src.evaluation.retrieval_metrics import context_tokens
from src.retrieval.chunking import STRATEGIES, markdown_section
from src.retrieval.retrievers import BM25Retriever

SRC = Path(__file__).resolve().parents[1] / "src"


@pytest.fixture(scope="module")
def articles():
    return load_corpus()


@pytest.fixture
def fake_tiktoken(monkeypatch):
    """A tiktoken whose counts are 10x the word count.

    Deliberately far from the estimate: if any code path still consults
    the tokenizer, chunk boundaries and costs move visibly rather than
    by a rounding error that a tolerance could hide.
    """
    module = types.ModuleType("tiktoken")

    class _Encoding:
        def encode(self, text):
            return [0] * (len(text.split()) * 10)

    module.get_encoding = lambda name: _Encoding()
    monkeypatch.setitem(sys.modules, "tiktoken", module)
    return module


@pytest.fixture
def no_tiktoken(monkeypatch):
    """Make `import tiktoken` raise ImportError, as on the CI runner."""
    monkeypatch.setitem(sys.modules, "tiktoken", None)


# ---------------------------------------------------------------------
# The two functions do what their names say
# ---------------------------------------------------------------------
def test_estimate_is_plain_arithmetic():
    assert estimate_tokens("") == 0
    assert estimate_tokens("one two three four five six seven eight nine ten") == 13
    assert estimate_tokens("  spaced \n\n out\twords  ") == 3   # int(3 * 1.3)


def test_estimate_ignores_an_installed_tokenizer(fake_tiktoken):
    text = "the refund policy applies for thirty days after purchase"
    assert exact_token_count(text) == 90          # the fake IS live
    assert estimate_tokens(text) == int(9 * 1.3)  # and is not consulted


def test_exact_count_is_none_without_tiktoken(no_tiktoken):
    """None, not a fallback number — a caller cannot mistake an estimate
    for an exact count if no number is returned."""
    assert exact_token_count("refund policy") is None
    assert tokenizer_available() is False


def test_exact_count_is_none_when_the_encoding_cannot_load(monkeypatch):
    """tiktoken imports fine but downloads its encoding on first use; a
    blocked network makes get_encoding raise. That must also yield None
    rather than a silently different number."""
    module = types.ModuleType("tiktoken")

    def _raise(name):
        raise ConnectionError("encoding download blocked")

    module.get_encoding = _raise
    monkeypatch.setitem(sys.modules, "tiktoken", module)
    assert exact_token_count("refund policy") is None


# ---------------------------------------------------------------------
# Nothing behavioural changes between the two environments
# ---------------------------------------------------------------------
def _snapshot(articles):
    """Everything downstream results are built from."""
    chunks = {
        name: [(c.chunk_id, c.text, c.n_tokens) for c in strategy(articles)]
        for name, strategy in STRATEGIES.items()
    }
    section_chunks = markdown_section(articles)
    retriever = BM25Retriever(section_chunks)
    in_scope = [q for q in load_golden_set() if q["category"] != "out_of_scope"]
    hits = [retriever.search(q["question"], top_k=15) for q in in_scope]
    costs = [context_tokens(h) for h in hits]
    ranked = [[x.chunk.chunk_id for x in h] for h in hits]
    return chunks, costs, ranked


def test_chunks_costs_and_rankings_ignore_tiktoken(articles, monkeypatch):
    monkeypatch.setitem(sys.modules, "tiktoken", None)
    without = _snapshot(articles)

    module = types.ModuleType("tiktoken")

    class _Encoding:
        def encode(self, text):
            return [0] * (len(text.split()) * 10)

    module.get_encoding = lambda name: _Encoding()
    monkeypatch.setitem(sys.modules, "tiktoken", module)
    assert exact_token_count("a b c") == 30       # the fake IS live
    with_fake = _snapshot(articles)

    assert without[0] == with_fake[0], "chunking changed with tiktoken"
    assert without[1] == with_fake[1], "context cost changed with tiktoken"
    assert without[2] == with_fake[2], "retrieval ranking changed with tiktoken"


def test_markdown_section_chunk_count_is_pinned(articles):
    """202 on the shipped corpus, in every environment.

    A pin, not a law: change the corpus or the merge threshold and this
    number should change, and the memo's figures with it. What it must
    never do is change because of what is installed. The value that
    leaked out of the old tiktoken path was 209.
    """
    assert len(markdown_section(articles)) == 202


# ---------------------------------------------------------------------
# The guard that keeps it fixed
# ---------------------------------------------------------------------
def test_only_the_token_module_touches_the_exact_tokenizer():
    """Pipeline code may not read an exact token count at all.

    `exact_token_count` is for display. If anything under src/ other than
    its own module references it or tiktoken, a result can depend on the
    environment again — and, as the first time, nothing would fail.
    """
    offenders = []
    for path in sorted(SRC.rglob("*.py")):
        if path.name == "difficulty.py":
            continue
        text = path.read_text(encoding="utf-8")
        if "tiktoken" in text or "exact_token_count" in text:
            offenders.append(str(path.relative_to(SRC)))
    assert not offenders, f"exact tokenizer used outside difficulty.py: {offenders}"
