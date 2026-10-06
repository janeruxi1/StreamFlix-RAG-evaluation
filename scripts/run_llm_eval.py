"""Run the credentialed evaluation end to end and keep the evidence.

    python scripts/run_llm_eval.py              # plan, confirm, run 01-07
    python scripts/run_llm_eval.py --dry-run    # plan only, spends nothing
    python scripts/run_llm_eval.py --from 05    # resume at notebook 05
    python scripts/run_llm_eval.py --yes        # skip the confirmation

Why this exists. The notebooks strip their outputs on commit (a cell
output can bake a leaked key into git history), so an interactive run
leaves its results in a kernel that is about to be closed. The first
credentialed run of this project survived only as staged blobs in the
git index. This script runs the same .py sources CI runs, writes each
one's output to reports/llm_run/, and records what produced it.

It runs the .py files, not the .ipynb, so it never touches a notebook
and cannot put the CI sync gate out of step.

What it guarantees:

  - Nothing is spent before the plan is shown and confirmed.
  - It stops at the first notebook that fails, rather than spending on
    later ones whose inputs are now suspect.
  - Logs are scrubbed of every credential in the environment before
    they are written, and the scrub is verified.
  - The manifest records the environment, so a number that later fails
    to reproduce can be traced to what was installed when it was made.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NOTEBOOKS = [
    "01_corpus_construction",
    "02_chunking_embedding",
    "03_retrieval_bakeoff",
    "04_generation",
    "05_evaluation",
    "06_judge_analysis",
    "07_decision_memo",
]
OUT_DIR = ROOT / "reports" / "llm_run"
CACHE_DIR = ROOT / ".llm_cache"
PACKAGES = ["numpy", "pandas", "scikit-learn", "matplotlib", "openai",
            "anthropic", "sentence-transformers", "tiktoken"]
SECRET_MARKERS = ("KEY", "TOKEN", "SECRET", "PASSWORD")
# Lines the notebooks print while a long loop is running. Matched on
# their shape, not on "indented and contains an ellipsis", which also
# matched ordinary prose in the output and echoed it as if it were progress.
PROGRESS_LINE = re.compile(
    r"^\s+(running \S+ over \d+ questions|\S+: (generating|judging)\b.*|"
    r"(judged )?\d+/\d+) \.\.\.\s*$")


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _git(*args: str) -> str:
    try:
        return subprocess.run(["git", *args], cwd=ROOT, capture_output=True,
                              text=True, check=False).stdout.strip()
    except OSError:
        return ""


def commit_id() -> str:
    """The short commit hash, seven characters.

    Short on purpose. The full 40-character hash is a high-entropy hex
    string, which is exactly what the secret scanner looks for: with it
    in the manifest, the commit hook and CI both refuse the file this
    script exists to produce. Seven characters identify the commit and
    sit below the scanner's entropy threshold.
    """
    return _git("rev-parse", "--short=7", "HEAD")[:7]


def environment_drift(manifest_path: Path) -> list[str]:
    """Packages the last completed run had that this interpreter lacks.

    A machine usually has several Pythons, and the one a new terminal
    picks is not always the one the last run used. Started from the
    wrong one, this script either stops for a missing SDK or — worse —
    runs without the transformer arm and quietly regenerates figures
    that no longer match the memo. Comparing against the last run's
    manifest turns "the package is not installed" into "you are in a
    different environment from the one that produced the results".
    """
    if not manifest_path.exists():
        return []
    try:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return sorted(name for name, version in previous.get("packages", {}).items()
                  if version is not None and _version(name) is None)


def _cache_entries() -> int:
    return len(list(CACHE_DIR.glob("*.json"))) if CACHE_DIR.exists() else 0


def _secrets() -> list[str]:
    """Every credential-shaped value in the environment.

    Long values only: redacting a short one (a flag set to "true") would
    shred the log without protecting anything.
    """
    return sorted(
        {v for k, v in os.environ.items()
         if any(m in k.upper() for m in SECRET_MARKERS) and len(v) >= 12},
        key=len, reverse=True)


def scrub(text: str, secrets: list[str]) -> str:
    """Remove credentials and trailing whitespace from captured output.

    The result ends in exactly one newline. The notebooks finish on a
    blank line, and a log that keeps it is rewritten by the commit hook's
    end-of-file fixer — which blocks the commit, leaves the files
    half-staged, and happens again on every run. Evidence the hooks
    refuse is not evidence anyone gets to commit.
    """
    for s in secrets:
        text = text.replace(s, "[REDACTED]")
    return "\n".join(line.rstrip() for line in text.splitlines()).rstrip("\n") + "\n"


def run_notebook(stem: str, env: dict, secrets: list[str]) -> dict:
    """Run one notebook source, streaming its output and keeping a copy."""
    started = time.perf_counter()
    cache_before = _cache_entries()
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "notebooks" / f"{stem}.py")],
        cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding="utf-8", errors="replace", bufsize=1)
    captured: list[str] = []
    assert proc.stdout is not None
    for line in proc.stdout:
        captured.append(line)
        if PROGRESS_LINE.match(line):
            print(f"      {line.strip()}", flush=True)       # progress only
    code = proc.wait()

    text = scrub("".join(captured), secrets)
    leaked = [s for s in secrets if s in text]
    if leaked:                                   # cannot happen; verified anyway
        raise RuntimeError("a credential survived scrubbing — log not written")
    with open(OUT_DIR / f"{stem}.txt", "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    return {
        "exit_code": code,
        "seconds": round(time.perf_counter() - started, 1),
        "new_api_calls": _cache_entries() - cache_before,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--from", dest="start", default="01", metavar="NN",
                        help="first notebook to run, e.g. 05 (default 01)")
    parser.add_argument("--dry-run", action="store_true",
                        help="show the plan and exit without running anything")
    parser.add_argument("--yes", action="store_true",
                        help="do not ask for confirmation before spending")
    args = parser.parse_args(argv)

    # The notebooks print non-ASCII (arrows, check marks). A Windows
    # console defaults to a legacy code page and would raise on them.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")

    to_run = [n for n in NOTEBOOKS if n[:2] >= args.start.zfill(2)]
    if not to_run:
        print(f"No notebook at or after {args.start!r}.")
        return 2

    os.chdir(ROOT)
    from src.corpus.build import load_corpus, load_golden_set
    from src.corpus.difficulty import tokenizer_available
    from src.evaluation.judge_validation import VALIDATION_CASES
    from src.evaluation.judged_arms import estimate_usd, parse_arms, plan_judging
    from src.generation.prompts import VARIANTS
    from src.llm.provider import provider_ready
    from src.retrieval.chunking import STRATEGIES

    ready, blocker = provider_ready()            # also loads .env
    golden = load_golden_set()
    arms = parse_arms(os.getenv("JUDGE_ARMS"), VARIANTS)
    n_refs = sum(1 for q in golden if q.get("reference_answer"))
    plan = plan_judging(len(golden), n_refs, len(VALIDATION_CASES), len(arms))
    n_chunks = len(STRATEGIES["markdown_section"](load_corpus()))

    print("=" * 72)
    print("  Credentialed evaluation run")
    print("=" * 72)
    drift = environment_drift(OUT_DIR / "manifest.json")
    print(f"""
  Python           : {sys.executable}
  Notebooks        : {to_run[0][:2]} to {to_run[-1][:2]}
  Provider ready   : {ready}{'' if ready else '  — ' + blocker}
  Generation model : {os.getenv('GENERATION_MODEL', 'gpt-4o-mini')}
  Judge model      : {os.getenv('JUDGE_MODEL', 'gpt-4o')}
  LLM arms judged  : {', '.join(arms)}
  Cached responses : {_cache_entries()}  (identical calls are free)

  Judge calls if nothing is cached: {plan.total_calls}  (~${plan.total_usd:.2f})
    validation {plan.validation_calls}, extractive baseline {plan.baseline_calls}, LLM arms {plan.arm_calls}
  Generation: up to {len(golden) * len(VARIANTS)} calls on the generation model in notebook 04,
    typically cents in total. Notebook 06 adds a small number of judge
    calls for its bias probes.

  The dollar figure is an estimate from average prompt sizes and list
  prices, not a quote. If the judge fails its validation gate in
  notebook 05, the LLM arms are not judged and ~${estimate_usd(plan.arm_calls):.2f} is not spent.

  Reproducibility fingerprint:
    markdown_section chunks : {n_chunks}   (must be 202)
    tiktoken installed      : {tokenizer_available()}   (must not change any result)
""")

    if drift:
        print(f"""  STOP: this is not the environment the last run used. That run had
  these packages and this interpreter does not:

    {', '.join(drift)}

  Running here would not reproduce it: without sentence-transformers
  the retrieval sweep drops the transformer arm and its figure is
  redrawn without it, and without the provider SDK no cached response
  can be read back. Switch to the interpreter that produced the results
  (the Python line above shows which one this is) and run again.
  Installing the packages here also works, but is slower and downloads
  the embedding model again.
""")
        return 1
    if n_chunks != 202:
        print("  STOP: the chunk count is not 202, so this environment would "
              "produce numbers\n  the memo does not describe. Run the test "
              "suite before spending anything.")
        return 1
    if not ready:
        print("  No credential path is ready, so this would be a keyless run "
              "and measure\n  nothing new. Fix the line above, or run the "
              "notebooks directly.")
        return 1
    if args.dry_run:
        print("  Dry run — nothing was executed.")
        return 0
    if not args.yes:
        reply = input("  Proceed? This will make paid API calls. [y/N] ")
        if reply.strip().lower() not in ("y", "yes"):
            print("  Cancelled — nothing was executed.")
            return 0

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    secrets = _secrets()
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8",
               MPLBACKEND="Agg")

    manifest = {
        "started_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": commit_id(),
        "git_dirty": bool(_git("status", "--porcelain")),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: _version(p) for p in PACKAGES},
        "provider": os.getenv("LLM_PROVIDER", "openai"),
        "generation_model": os.getenv("GENERATION_MODEL", "gpt-4o-mini"),
        "judge_model": os.getenv("JUDGE_MODEL", "gpt-4o"),
        "judged_arms": arms,
        "markdown_section_chunks": n_chunks,
        "tiktoken_available": tokenizer_available(),
        "planned_judge_calls": plan.total_calls,
        "notebooks": {},
    }

    failed = None
    for stem in to_run:
        print(f"  [{stem}] running ...", flush=True)
        result = run_notebook(stem, env, secrets)
        manifest["notebooks"][stem] = result
        status = "ok" if result["exit_code"] == 0 else f"FAILED (exit {result['exit_code']})"
        print(f"  [{stem}] {status} in {result['seconds']}s, "
              f"{result['new_api_calls']} new API calls", flush=True)
        if result["exit_code"] != 0:
            failed = stem
            break

    manifest["total_new_api_calls"] = sum(
        r["new_api_calls"] for r in manifest["notebooks"].values())
    manifest["completed"] = failed is None
    with open(OUT_DIR / "manifest.json", "w", encoding="utf-8", newline="\n") as f:
        f.write(scrub(json.dumps(manifest, indent=2, sort_keys=True), secrets))

    print()
    if failed:
        print(f"  Stopped at {failed}. Its output is in "
              f"reports/llm_run/{failed}.txt — the last lines say why.")
        print("  Later notebooks were not run. Completed calls are cached, "
              "so re-running\n  repeats nothing that already succeeded.")
        return 1

    print(f"  Done. {manifest['total_new_api_calls']} new API calls.")
    print("  Outputs  : reports/llm_run/*.txt and manifest.json")
    print("  Measured : reports/metrics/*.json")
    print("  Figures  : reports/figures/")
    return 0


if __name__ == "__main__":
    sys.exit(main())
