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

        py_path = nb_path.with_suffix(".py")
        if not py_path.exists():
            problems.append(f"{nb_path.name}: no paired .py source")
            continue

        src = py_path.read_text(encoding="utf-8")
        tree = ast.parse(src)
        body = "\n".join(src.split("\n")[tree.body[1].lineno - 1:])
        code = "".join("".join(c["source"]) for c in nb["cells"]
                       if c["cell_type"] == "code")
        norm = lambda s: re.sub(r"\s+", "", s)
        missing = [
            line for line in body.split("\n")
            if line.strip() and not line.strip().startswith("#")
            and "Path(__file__)" not in line
            and norm(line) not in norm(code)
        ]
        if missing:
            problems.append(
                f"{nb_path.name}: {len(missing)} code lines in .py absent from "
                f".ipynb — the notebook is stale relative to its source")


def check_text_files() -> None:
    """Encoding, trailing newline, whitespace, merge markers."""
    patterns = ("*.py", "*.md", "*.yml", "*.yaml", "*.json", "*.txt", "*.example")
    for pattern in patterns:
        for path in sorted(ROOT.rglob(pattern)):
            if any(part in {".git", ".venv", "__pycache__", ".pytest_cache"}
                   for part in path.parts):
                continue
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
            if any(l != l.rstrip() for l in text.split("\n")):
                problems.append(f"{path.relative_to(ROOT)}: trailing whitespace")
            if re.search(r"^(<{7} |={7}$|>{7} )", text, re.M):
                problems.append(f"{path.relative_to(ROOT)}: merge conflict marker")


def check_secrets() -> None:
    """Credential shapes anywhere in the tree."""
    shapes = re.compile(
        r"sk-[a-zA-Z0-9]{20,}|sk-ant-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16}"
        r"|BEGIN [A-Z ]*PRIVATE KEY")
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file() or any(
                p in {".git", "__pycache__", ".pytest_cache"} for p in path.parts):
            continue
        if path.suffix in {".png", ".jpg", ".npy", ".faiss"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            continue
        for m in shapes.finditer(text):
            problems.append(f"{path.relative_to(ROOT)}: credential-shaped "
                            f"string {m.group(0)[:12]}...")


def main() -> int:
    check_notebooks()
    check_text_files()
    check_secrets()

    if problems:
        print(f"FAILED — {len(problems)} problem(s):\n")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("All repo-wide invariants hold.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
