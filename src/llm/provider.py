"""LLM provider abstraction with safe credential handling.

Design goals:
    1. SECRETS NEVER LEAVE THE ENVIRONMENT. The API key is read from
       os.environ at call time and is never returned, logged, printed,
       serialized, or stored on any object that could end up in a
       notebook output or a pickle.
    2. SWAPPABLE BACKEND. OpenAI, Anthropic, and Ollama all satisfy the
       same `LLMProvider` interface, so the retrieval/generation/eval
       layers never import a vendor SDK directly.
    3. COST GUARDRAILS. A per-run call ceiling and an on-disk response
       cache prevent a runaway loop from burning budget.

Why the judge model is stronger than the generation model:
    The evaluation harness uses an LLM to score the generation model's
    output (faithfulness, relevancy, hallucination). If the judge were
    the same model as the system under test, evaluation quality would be
    bounded by the thing being evaluated — a weak model can't reliably
    detect its own errors. We deliberately use a stronger judge tier.

Usage:
    from src.llm.provider import get_provider

    llm = get_provider()               # reads LLM_PROVIDER from env
    answer = llm.complete("Summarize: ...", max_tokens=300)

Verify your key loaded without exposing it:
    python -m src.llm.provider --check
"""
from __future__ import annotations

import hashlib
import json
import importlib.util
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------
# Credential handling — the only place secrets are touched
# ---------------------------------------------------------------------
class MissingCredentialError(RuntimeError):
    """Raised when a required API key is absent from the environment."""


def _require_env(var_name: str, setup_hint: str) -> str:
    """Fetch a required environment variable or fail with a clear message.

    The value is returned to the caller but never logged. Callers must
    treat the return value as a secret: do not print it, do not store it
    on a long-lived object, do not include it in a repr.
    """
    value = os.getenv(var_name)
    if not value or value.startswith("sk-your-key") or value.endswith("-here"):
        raise MissingCredentialError(
            f"\n{var_name} is not set (or still holds the placeholder value).\n"
            f"\n  {setup_hint}\n"
            f"\nSetup:\n"
            f"  1. cp .env.example .env\n"
            f"  2. Paste your real key into .env\n"
            f"  3. python -m src.llm.provider --check\n"
            f"\n.env is gitignored — it will not be committed.\n"
        )
    return value


def masked_credential(var_name: str = "OPENAI_API_KEY") -> str:
    """Return a masked form of a credential — SAFE to print or log.

    Shows enough to confirm the right key is loaded, not enough to use.
    Example: 'sk-proj...9f2a'

    Use this instead of printing the key when debugging. Never call
    _require_env() in a notebook cell that displays its result.
    """
    raw = os.getenv(var_name, "")
    if not raw:
        return f"{var_name}: (not set)"
    if raw.startswith("sk-your-key") or raw.endswith("-here"):
        return f"{var_name}: (placeholder — not a real key)"
    if len(raw) <= 12:
        return f"{var_name}: (set, but suspiciously short)"
    return f"{var_name}: {raw[:7]}...{raw[-4:]}  (length {len(raw)})"


# ---------------------------------------------------------------------
# Cost guardrails
# ---------------------------------------------------------------------
@dataclass
class CallBudget:
    """Hard ceiling on API calls within a single process run.

    Prevents a runaway loop (bad retry logic, an accidental nested loop
    over the golden set) from burning through the account's usage cap.
    """
    max_calls: int
    used: int = 0

    def spend(self, n: int = 1) -> None:
        if self.max_calls <= 0:
            return  # ceiling disabled
        if self.used + n > self.max_calls:
            raise RuntimeError(
                f"LLM call budget exhausted: {self.used}/{self.max_calls} calls "
                f"used in this run.\n"
                f"Raise MAX_LLM_CALLS_PER_RUN in .env if this is expected, "
                f"or check for a runaway loop."
            )
        self.used += n


# ---------------------------------------------------------------------
# Response cache — avoid re-paying for identical calls
# ---------------------------------------------------------------------
class ResponseCache:
    """On-disk cache keyed by a hash of (model, prompt, params).

    Re-running a notebook after a kernel restart shouldn't re-pay for
    identical calls. The cache directory is gitignored.

    NOTE: the cache key hashes the prompt, never the API key.
    """

    def __init__(self, cache_dir: Path, enabled: bool = True):
        self.cache_dir = cache_dir
        self.enabled = enabled
        if self.enabled:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _key(model: str, prompt: str, **params) -> str:
        payload = json.dumps(
            {"model": model, "prompt": prompt, **params}, sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()[:32]

    def get(self, model: str, prompt: str, **params) -> Optional[str]:
        if not self.enabled:
            return None
        path = self.cache_dir / f"{self._key(model, prompt, **params)}.json"
        if path.exists():
            return json.loads(path.read_text())["response"]
        return None

    def put(self, model: str, prompt: str, response: str, **params) -> None:
        if not self.enabled:
            return
        path = self.cache_dir / f"{self._key(model, prompt, **params)}.json"
        path.write_text(json.dumps({"response": response}))


# ---------------------------------------------------------------------
# Provider interface
# ---------------------------------------------------------------------
class LLMProvider(ABC):
    """Common interface every backend satisfies.

    Downstream code (retrieval, generation, evaluation) depends only on
    this interface — never on a vendor SDK. Swapping providers is a
    one-line change in .env.
    """

    name: str = "base"

    def __init__(self, budget: CallBudget, cache: ResponseCache):
        self._budget = budget
        self._cache = cache

    @abstractmethod
    def _call(self, prompt: str, model: str, max_tokens: int,
              temperature: float) -> str:
        """Vendor-specific API call. Implementations must not log the key."""

    def complete(self, prompt: str, model: Optional[str] = None,
                 max_tokens: int = 512, temperature: float = 0.0) -> str:
        """Generate a completion, respecting cache and budget."""
        model = model or self.default_model
        cached = self._cache.get(model, prompt, max_tokens=max_tokens,
                                 temperature=temperature)
        if cached is not None:
            return cached

        self._budget.spend(1)
        response = self._call(prompt, model, max_tokens, temperature)
        self._cache.put(model, prompt, response, max_tokens=max_tokens,
                        temperature=temperature)
        return response

    @property
    @abstractmethod
    def default_model(self) -> str: ...

    def __repr__(self) -> str:
        # Deliberately excludes any credential material.
        return f"<{type(self).__name__} model={self.default_model!r}>"


class OpenAIProvider(LLMProvider):
    name = "openai"

    def __init__(self, budget: CallBudget, cache: ResponseCache,
                 model: Optional[str] = None):
        super().__init__(budget, cache)
        self._model = model or os.getenv("GENERATION_MODEL", "gpt-4o-mini")
        # Validate the key exists NOW so failures surface at construction
        # time, not mid-loop. The value is not retained on the instance.
        _require_env(
            "OPENAI_API_KEY",
            "Get a key at https://platform.openai.com/api-keys",
        )

    @property
    def default_model(self) -> str:
        return self._model

    def _call(self, prompt: str, model: str, max_tokens: int,
              temperature: float) -> str:
        from openai import OpenAI
        # Key is read fresh from env and scoped to this call.
        client = OpenAI(api_key=_require_env("OPENAI_API_KEY", ""))
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return resp.choices[0].message.content or ""


class AnthropicProvider(LLMProvider):
    name = "anthropic"

    def __init__(self, budget: CallBudget, cache: ResponseCache,
                 model: Optional[str] = None):
        super().__init__(budget, cache)
        self._model = model or os.getenv("GENERATION_MODEL",
                                         "claude-haiku-4-5-20251001")
        _require_env(
            "ANTHROPIC_API_KEY",
            "Get a key at https://console.anthropic.com/settings/keys",
        )

    @property
    def default_model(self) -> str:
        return self._model

    def _call(self, prompt: str, model: str, max_tokens: int,
              temperature: float) -> str:
        import anthropic
        client = anthropic.Anthropic(
            api_key=_require_env("ANTHROPIC_API_KEY", "")
        )
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}],
        )
        return resp.content[0].text


class OllamaProvider(LLMProvider):
    """Local models — no API key, no cost, fully reproducible.

    Included so anyone can clone the repo and run the pipeline without
    credentials. Answer quality is lower than the hosted tiers, which
    the evaluation harness will surface honestly.
    """
    name = "ollama"

    def __init__(self, budget: CallBudget, cache: ResponseCache,
                 model: Optional[str] = None):
        super().__init__(budget, cache)
        self._model = model or os.getenv("GENERATION_MODEL", "llama3.2")
        self._base_url = os.getenv("OLLAMA_BASE_URL",
                                   "http://localhost:11434")

    @property
    def default_model(self) -> str:
        return self._model

    def _call(self, prompt: str, model: str, max_tokens: int,
              temperature: float) -> str:
        import requests
        resp = requests.post(
            f"{self._base_url}/api/generate",
            json={
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": temperature,
                            "num_predict": max_tokens},
            },
            timeout=120,
        )
        resp.raise_for_status()
        return resp.json()["response"]


# ---------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------
_PROVIDERS = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "ollama": OllamaProvider,
}


def get_provider(provider_name: Optional[str] = None,
                 model: Optional[str] = None) -> LLMProvider:
    """Construct the configured provider.

    Reads LLM_PROVIDER from the environment unless overridden. Raises
    MissingCredentialError with setup instructions if the required key
    is absent.
    """
    _load_dotenv_if_present()

    name = (provider_name or os.getenv("LLM_PROVIDER", "openai")).lower()
    if name not in _PROVIDERS:
        raise ValueError(
            f"Unknown LLM_PROVIDER={name!r}. "
            f"Expected one of: {', '.join(_PROVIDERS)}"
        )

    budget = CallBudget(
        max_calls=int(os.getenv("MAX_LLM_CALLS_PER_RUN", "2000"))
    )
    cache = ResponseCache(
        cache_dir=Path(".llm_cache"),
        enabled=os.getenv("ENABLE_LLM_CACHE", "true").lower() == "true",
    )
    return _PROVIDERS[name](budget, cache, model=model)


_SDK_MODULE = {"openai": "openai", "anthropic": "anthropic", "ollama": "requests"}
_SDK_INSTALL = {"openai": "pip install openai",
                "anthropic": "pip install anthropic",
                "ollama": "pip install requests"}


def provider_ready(provider_name: Optional[str] = None) -> tuple[bool, str]:
    """Can we actually make a call? Checks EVERY precondition.

    A credential in the environment is necessary and not sufficient — the
    vendor SDK also has to be importable. Checking only the key means a
    notebook announces "running 120 questions" and then dies on question
    one with ModuleNotFoundError, after the reader has been told the run
    started.

    Returns (ready, reason). `reason` is empty when ready, and otherwise
    names the missing piece and the command that fixes it, so callers can
    print something actionable instead of a traceback.

    Deliberately does NOT construct a provider or make a network call:
    this must be cheap enough to run at the top of every notebook.
    """
    _load_dotenv_if_present()

    name = (provider_name or os.getenv("LLM_PROVIDER", "openai")).lower()
    if name not in _PROVIDERS:
        return False, (f"LLM_PROVIDER={name!r} is not recognised. "
                       f"Expected one of: {', '.join(_PROVIDERS)}")

    if name != "ollama":                      # ollama is keyless and local
        env_var = "OPENAI_API_KEY" if name == "openai" else "ANTHROPIC_API_KEY"
        raw = os.getenv(env_var, "").strip()
        if not raw:
            return False, (f"{env_var} is not set. Copy .env.example to .env "
                           f"and add your key.")
        if raw.startswith("sk-your-key") or raw.endswith("-here"):
            return False, (f"{env_var} still holds the placeholder from "
                           f".env.example, not a real key.")

    module = _SDK_MODULE[name]
    if importlib.util.find_spec(module) is None:
        return False, (f"the {module!r} package is not installed, so no call "
                       f"can be made even though the credential is present. "
                       f"Fix: {_SDK_INSTALL[name]}  (or "
                       f"pip install -r requirements.txt)")

    return True, ""


def get_judge(model: Optional[str] = None) -> LLMProvider:
    """Provider configured with the JUDGE model (stronger tier).

    Used by the evaluation harness. Deliberately a stronger model than
    the generation tier so eval quality isn't bounded by the system
    under test.
    """
    judge_model = model or os.getenv("JUDGE_MODEL", "gpt-4o")
    return get_provider(model=judge_model)


def _load_dotenv_if_present() -> None:
    """Load .env into the environment if python-dotenv is installed.

    Safe no-op when the package or the file is absent, so CI (which
    injects env vars directly) works without a .env file.
    """
    try:
        from dotenv import load_dotenv
        load_dotenv(override=False)
    except ImportError:
        pass


# ---------------------------------------------------------------------
# CLI: verify credentials WITHOUT exposing them
# ---------------------------------------------------------------------
def _check() -> int:
    """Print a masked credential report. Safe to run and screenshot."""
    _load_dotenv_if_present()

    provider = os.getenv("LLM_PROVIDER", "openai")
    print("=" * 62)
    print("  Credential check — values are MASKED and safe to share")
    print("=" * 62)
    print(f"  LLM_PROVIDER      : {provider}")
    print(f"  GENERATION_MODEL  : {os.getenv('GENERATION_MODEL', '(default)')}")
    print(f"  JUDGE_MODEL       : {os.getenv('JUDGE_MODEL', '(default)')}")
    print(f"  EMBEDDING_MODEL   : {os.getenv('EMBEDDING_MODEL', '(default)')}")
    print()
    for var in ("OPENAI_API_KEY", "ANTHROPIC_API_KEY"):
        print(f"  {masked_credential(var)}")
    print()

    # Report SDK availability explicitly. Constructing a provider is NOT
    # sufficient evidence that a call can be made: the vendor SDK is
    # imported lazily inside _call, so get_provider() succeeds happily
    # with the package absent. An earlier version of this function did
    # exactly that and printed "Ready" to someone whose next API call
    # would raise ModuleNotFoundError — a diagnostic that lied about the
    # thing it existed to diagnose.
    module = _SDK_MODULE.get(provider, provider)
    installed = importlib.util.find_spec(module) is not None
    mark = "✓" if installed else "✗"
    print(f"  {mark} SDK package {module!r}: "
          f"{'installed' if installed else 'NOT INSTALLED'}")
    print()

    ready, blocker = provider_ready()
    if ready:
        try:
            llm = get_provider()
            print(f"  ✓ Provider constructed: {llm!r}")
            print("  ✓ READY — credential and SDK both present, and the "
                  "credential was\n      resolved without being exposed.")
            status = 0
        except Exception as exc:  # noqa: BLE001 - surface any config problem
            print(f"  ✗ Provider construction failed: "
                  f"{type(exc).__name__}: {exc}")
            status = 1
    else:
        print(f"  ✗ NOT READY — {blocker}")
        status = 1

    print("=" * 62)
    return status


if __name__ == "__main__":
    import sys
    if "--check" in sys.argv:
        raise SystemExit(_check())
    print(__doc__)
