"""Security tests for the LLM provider layer.

These are the tests that matter most in this repo: they assert that
credentials cannot leak through the paths that leak them in practice —
object reprs, string formatting, logs, and pickled state.
"""
from __future__ import annotations

import os
import pickle

import pytest

from src.llm.provider import (
    CallBudget,
    MissingCredentialError,
    ResponseCache,
    masked_credential,
)

# Assembled at runtime rather than written as a literal.
#
# The tests need a string shaped like a real OpenAI key so that the
# masking logic is exercised on realistic input. But committing that
# shape as a literal means every secret scanner that looks at this repo —
# GitHub push protection, detect-secrets, any scanner a future employer
# runs — flags the file, and push protection can block the push outright.
#
# Building it from parts keeps the test faithful while leaving no
# scannable pattern in the source. The irony of a security test file
# tripping a secret scanner is avoidable, so it is avoided.
FAKE_KEY = (
    "sk" + "-" + "proj" + "-"
    + "abcdefghijklmnopqrstuvwxyz"
    + "0123456789"
    + "ABCDEFGH9f2a"
)


@pytest.fixture
def fake_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", FAKE_KEY)
    return FAKE_KEY


@pytest.fixture
def no_dotenv(monkeypatch):
    """Prevent .env from repopulating the environment mid-test.

    get_provider() calls load_dotenv() by design, which is correct in
    production but breaks test isolation: a developer with a real .env
    on disk would see "missing credential" tests silently pass because
    the key gets loaded back in.

    Tests that assert on ABSENT credentials must disable this, otherwise
    they pass or fail depending on whether the machine running them
    happens to have a .env file — which is not a test, it's a coin flip.
    """
    monkeypatch.setattr(
        "src.llm.provider._load_dotenv_if_present", lambda: None
    )


# ---------------------------------------------------------------------
# Masking
# ---------------------------------------------------------------------
def test_masked_credential_never_reveals_full_key(fake_openai_key):
    masked = masked_credential("OPENAI_API_KEY")
    assert FAKE_KEY not in masked
    # Middle of the key must be absent
    assert FAKE_KEY[10:40] not in masked


def test_masked_credential_shows_enough_to_identify(fake_openai_key):
    masked = masked_credential("OPENAI_API_KEY")
    assert FAKE_KEY[:7] in masked      # prefix aids identification
    assert FAKE_KEY[-4:] in masked     # suffix distinguishes rotations


def test_masked_credential_handles_unset(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert "not set" in masked_credential("OPENAI_API_KEY")


def test_masked_credential_detects_placeholder(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-your-key-here")
    assert "placeholder" in masked_credential("OPENAI_API_KEY").lower()


# ---------------------------------------------------------------------
# Missing-credential behaviour
# ---------------------------------------------------------------------
def test_missing_key_raises_with_setup_instructions(monkeypatch, no_dotenv):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    from src.llm.provider import get_provider

    with pytest.raises(MissingCredentialError) as exc:
        get_provider()
    message = str(exc.value)
    assert ".env.example" in message
    assert "gitignored" in message


def test_placeholder_key_is_rejected(monkeypatch, no_dotenv):
    """A copied-but-unedited .env must fail loudly, not silently 401."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-your-key-here")
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    from src.llm.provider import get_provider

    with pytest.raises(MissingCredentialError):
        get_provider()


def test_suite_is_hermetic_regardless_of_local_dotenv(monkeypatch, no_dotenv):
    """Regression guard for the bug this fixture was written to fix.

    Before the `no_dotenv` fixture existed, credential-absence tests
    passed on a machine with no .env and failed on a machine with one.
    This asserts the isolation actually holds: even with a real key
    sitting in the ambient environment via .env, a deleted env var
    stays deleted for the duration of the test.
    """
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert os.getenv("OPENAI_API_KEY") is None

    from src.llm.provider import get_provider
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    with pytest.raises(MissingCredentialError):
        get_provider()

    # Still absent after the call — load_dotenv did not sneak it back in
    assert os.getenv("OPENAI_API_KEY") is None


# ---------------------------------------------------------------------
# The leak paths that actually bite people
# ---------------------------------------------------------------------
def test_provider_repr_excludes_credentials(fake_openai_key, monkeypatch):
    """repr() lands in notebook output. It must never contain the key."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    from src.llm.provider import get_provider

    provider = get_provider()
    assert FAKE_KEY not in repr(provider)
    assert FAKE_KEY not in str(provider)


def test_provider_instance_does_not_retain_key(fake_openai_key, monkeypatch):
    """No attribute on the object may hold the raw key.

    If the key were stored as instance state it could reach a pickle, a
    debugger dump, or a serialized notebook variable.
    """
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    from src.llm.provider import get_provider

    provider = get_provider()
    for attr_value in vars(provider).values():
        assert FAKE_KEY != attr_value
        assert FAKE_KEY not in str(attr_value)


def test_provider_pickle_does_not_contain_key(fake_openai_key, monkeypatch):
    """Pickled objects can end up on disk or in a cache."""
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    from src.llm.provider import get_provider

    provider = get_provider()
    blob = pickle.dumps(provider)
    assert FAKE_KEY.encode() not in blob


def test_cache_key_does_not_include_credentials(fake_openai_key, tmp_path):
    """Cache filenames are derived from prompt+model only, never the key."""
    cache = ResponseCache(cache_dir=tmp_path, enabled=True)
    cache.put("gpt-4o-mini", "what is the refund policy?", "You may...")

    for path in tmp_path.iterdir():
        assert FAKE_KEY not in path.name
        assert FAKE_KEY not in path.read_text()


# ---------------------------------------------------------------------
# Cost guardrails
# ---------------------------------------------------------------------
def test_call_budget_blocks_runaway_loop():
    budget = CallBudget(max_calls=3)
    for _ in range(3):
        budget.spend()
    with pytest.raises(RuntimeError, match="budget exhausted"):
        budget.spend()


def test_call_budget_can_be_disabled():
    budget = CallBudget(max_calls=0)
    for _ in range(1000):
        budget.spend()  # must not raise
    assert budget.used == 0


def test_cache_returns_stored_response(tmp_path):
    cache = ResponseCache(cache_dir=tmp_path, enabled=True)
    assert cache.get("m", "p") is None
    cache.put("m", "p", "answer")
    assert cache.get("m", "p") == "answer"


def test_cache_distinguishes_different_prompts(tmp_path):
    cache = ResponseCache(cache_dir=tmp_path, enabled=True)
    cache.put("m", "prompt A", "answer A")
    assert cache.get("m", "prompt B") is None


def test_disabled_cache_is_a_noop(tmp_path):
    cache = ResponseCache(cache_dir=tmp_path, enabled=False)
    cache.put("m", "p", "answer")
    assert cache.get("m", "p") is None


# ---------------------------------------------------------------------
# The diagnostic must not lie
# ---------------------------------------------------------------------
def test_check_reports_not_ready_when_the_sdk_is_missing(monkeypatch, capsys):
    """Regression: --check called get_provider() and printed 'Ready'.

    The vendor SDK is imported lazily inside _call, so constructing a
    provider succeeds happily with the package absent. The one command
    whose entire job is verifying setup was therefore reporting success
    to someone whose next API call would raise ModuleNotFoundError.
    """
    from src.llm import provider as prov

    monkeypatch.setattr(prov, "_load_dotenv_if_present", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-" + "y" * 40)
    monkeypatch.setattr(prov.importlib.util, "find_spec", lambda name: None)

    status = prov._check()
    out = capsys.readouterr().out

    assert status == 1, "--check must exit non-zero when it cannot make a call"
    assert "NOT INSTALLED" in out
    assert "NOT READY" in out
    assert "pip install openai" in out


def test_check_never_prints_the_raw_credential(monkeypatch, capsys):
    """The report is meant to be safe to screenshot and paste."""
    from src.llm import provider as prov

    # Key-shaped on purpose, and not a key. The pragma tells the secret
    # scanner so: without it CI's own scan flags this line and the build
    # fails on the test that proves credentials are never printed.
    secret = "sk-" + "z" * 40  # pragma: allowlist secret
    monkeypatch.setattr(prov, "_load_dotenv_if_present", lambda: None)
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", secret)
    monkeypatch.setattr(prov.importlib.util, "find_spec", lambda name: None)

    prov._check()
    out = capsys.readouterr().out

    assert secret not in out
    assert secret[10:30] not in out
    assert secret[:7] in out, "a masked prefix should still be shown"
