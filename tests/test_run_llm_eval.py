"""Tests for the credentialed-run script's safety properties.

The script's job is to keep evidence from a paid run. The property that
must hold above all others is that the evidence it writes can be
committed: no credential in any log, whatever a notebook printed.
"""
import importlib.util
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def runner():
    spec = importlib.util.spec_from_file_location(
        "run_llm_eval", ROOT / "scripts" / "run_llm_eval.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


FAKE_KEY = "sk-test-" + "a1B2c3D4" * 5          # key-shaped, not a key


def test_scrub_removes_a_credential_wherever_it_appears(runner):
    text = (f"Authorization: Bearer {FAKE_KEY}\n"
            f"repr: OpenAI(api_key='{FAKE_KEY}')\n"
            f"url: https://x/?key={FAKE_KEY}&q=1\n")
    out = runner.scrub(text, [FAKE_KEY])
    assert FAKE_KEY not in out
    assert out.count("[REDACTED]") == 3


def test_scrub_handles_one_secret_containing_another(runner):
    """Longest first, or the shorter replacement leaves the tail of the
    longer secret behind in the log."""
    long_key = FAKE_KEY + "-suffix-9z8y7x"
    secrets = sorted([FAKE_KEY, long_key], key=len, reverse=True)
    out = runner.scrub(f"k={long_key}", secrets)
    assert "suffix" not in out
    assert out == "k=[REDACTED]\n"


def test_scrub_strips_trailing_whitespace_and_ends_in_one_newline(runner):
    out = runner.scrub("a   \nb\t\n\n", [])
    assert out == "a\nb\n\n"


def test_secrets_are_found_by_name_and_sorted_longest_first(runner, monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("SOME_SERVICE_TOKEN", FAKE_KEY + "-longer-value")
    monkeypatch.setenv("ENABLE_LLM_CACHE", "true")          # not a secret
    monkeypatch.setenv("SHORT_KEY", "abc")                   # too short to redact
    found = runner._secrets()
    assert FAKE_KEY in found
    assert FAKE_KEY + "-longer-value" in found
    assert "true" not in found and "abc" not in found
    assert found == sorted(found, key=len, reverse=True)


def test_dry_run_executes_nothing(runner, monkeypatch, capsys, tmp_path):
    """--dry-run must not create the output directory, let alone spend."""
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    monkeypatch.setenv("LLM_PROVIDER", "ollama")     # keyless, needs only `requests`
    monkeypatch.setattr(runner, "OUT_DIR", tmp_path / "llm_run")

    def _fail(*a, **k):
        raise AssertionError("dry run started a notebook")

    monkeypatch.setattr(runner, "run_notebook", _fail)
    code = runner.main(["--dry-run"])
    out = capsys.readouterr().out
    assert code in (0, 1)        # 1 when the provider SDK is absent, as in CI
    assert "markdown_section chunks : 202" in out
    assert not (tmp_path / "llm_run").exists()
