"""Repo-wide invariant check — scans EVERY file, not just changed ones.

Written after a real miss: notebook 02 was committed with 18 embedded
outputs and stayed that way for several commits, because the pre-commit
checks being replicated by hand only ever looked at the files in the
current diff. A violation introduced once and then left alone is
invisible to a diff-scoped check forever.

`pre-commit run --all-files` does this properly on a machine where the
hooks are installed. This script is the portable equivalent, so the same
guarantee holds in CI and in environments where the hooks cannot run.
"""
from __future__ import annotations

import ast
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
problems: list[str] = []

# Directories that are never source and must not be walked.
#
# The exclusion is not just tidiness. This repo lives on a cloud-synced
# folder where files can be on-demand placeholders, and reading one
# blocks until it downloads. An unbounded rglob over caches and .git
# turns a one-second check into a multi-minute stall that looks like a
# hang. Scanning only source keeps it fast and predictable.
SKIP_DIRS = {".git", "__pycache__", ".pytest_cache", ".venv", "venv",
             ".ipynb_checkpoints", "embedding_cache", ".llm_cache",
             "vectorstore", ".mypy_cache"}
BINARY_SUFFIXES = {".png", ".jpg", ".jpeg", ".gif", ".npy", ".faiss",
                   ".index", ".pkl", ".joblib", ".zip", ".pdf"}
MAX_BYTES = 2_000_000


def source_files() -> list[Path]:
    """Every text file that is genuinely part of the project."""
    out: list[Path] = []
    for path in ROOT.rglob("*"):
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if not path.is_file() or path.suffix in BINARY_SUFFIXES:
            continue
        try:
            if path.stat().st_size > MAX_BYTES:
                continue
        except OSError:
            continue
        out.append(path)
    return sorted(out)


def check_notebooks() -> None:
    """Outputs stripped, cell ids present, and .py/.ipynb in sync."""
    for nb_path in sorted((ROOT / "notebooks").glob("*.ipynb")):
        nb = json.loads(nb_path.read_text(encoding="utf-8"))

        outputs = sum(len(c.get("outputs", [])) for c in nb["cells"])
        if outputs:
            problems.append(
                f"{nb_path.name}: {outputs} embedded outputs — a cell output "
                f"can bake a leaked credential into git history permanently")

        if any("id" not in c for c in nb["cells"]):
            problems.append(f"{nb_path.name}: cells missing ids (nbformat 4.5+)")

        setup = "".join(nb["cells"][1]["source"]) if len(nb["cells"]) > 1 else ""
        # Match the ASSIGNMENT, not the bare substring — the corrected
        # setup cell explains the old bug in a comment, and a naive
        # substring check flags its own documentation.
        code_lines = [ln for ln in setup.split("\n") if not ln.lstrip().startswith("#")]
        if any(re.search(r"PROJECT_ROOT\s*=\s*Path\.cwd\(\)\.parent", ln)
               for ln in code_lines):
            problems.append(
                f"{nb_path.name}: setup cell uses Path.cwd().parent, which "
                f"assumes the kernel started in notebooks/. Launch Jupyter "
                f"from the project root and it resolves one level too high, "
                f"so data/ and src/ both vanish. Walk up for a marker instead.")

        py_path = nb_path.with_suffix(".py")
        if not py_path.exists():
            problems.append(f"{nb_path.name}: no paired .py source")
            continue

        src = py_path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        body = "\n".join(src.split("\n")[tree.body[1].lineno - 1:])
        code = "".join("".join(c["source"]) for c in nb["cells"]
                       if c["cell_type"] == "code")
        def norm(s):
            return re.sub(r"\s+", "", s)
        # Normalise the notebook ONCE. Recomputing it per source line is
        # O(lines x notebook_size) and turns a fast check into a hang on
        # the larger notebooks.
        norm_code = norm(code)
        missing = [
            line for line in body.split("\n")
            if line.strip() and not line.strip().startswith("#")
            and "Path(__file__)" not in line
            and norm(line) not in norm_code
        ]
        if missing:
            problems.append(
                f"{nb_path.name}: {len(missing)} code lines in .py absent from "
                f".ipynb — the notebook is stale relative to its source")


def check_text_files(files: list[Path]) -> None:
    """Encoding, trailing newline, whitespace, merge markers."""
    exts = {".py", ".md", ".yml", ".yaml", ".json", ".txt", ".example",
            ".cfg", ".toml"}
    for path in files:
        if path.suffix in exts or path.name in {".gitignore", ".gitattributes",
                                                ".env.example"}:
            raw = path.read_bytes()
            if raw[:2] in (b"\xff\xfe", b"\xfe\xff"):
                problems.append(f"{path.relative_to(ROOT)}: UTF-16 (PowerShell "
                                f"redirect artifact) — tools expect UTF-8")
            try:
                text = raw.decode("utf-8")
            except UnicodeDecodeError:
                problems.append(f"{path.relative_to(ROOT)}: not valid UTF-8")
                continue
            if raw and not raw.endswith(b"\n"):
                problems.append(f"{path.relative_to(ROOT)}: no trailing newline")
            if any(ln != ln.rstrip() for ln in text.split("\n")):
                problems.append(f"{path.relative_to(ROOT)}: trailing whitespace")
            if re.search(r"^(<{7} |={7}$|>{7} )", text, re.M):
                problems.append(f"{path.relative_to(ROOT)}: merge conflict marker")


def check_secrets(files: list[Path]) -> None:
    """Credential shapes anywhere in the source tree."""
    shapes = re.compile(
        r"sk-[a-zA-Z0-9]{20,}|sk-ant-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}"
        r"|BEGIN [A-Z ]*PRIVATE KEY")
    for path in files:
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for m in shapes.finditer(text):
            problems.append(f"{path.relative_to(ROOT)}: credential-shaped "
                            f"string {m.group(0)[:12]}...")


def main() -> int:
    files = source_files()
    check_notebooks()
    check_text_files(files)
    check_secrets(files)
    print(f"Scanned {len(files)} source files.")

    if problems:
        print(f"FAILED — {len(problems)} problem(s):\n")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("All repo-wide invariants hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
