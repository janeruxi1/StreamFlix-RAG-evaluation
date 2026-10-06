"""Tests for the notebook sync check.

The check has two ways to be wrong and both have happened here. Too
strict, it fails on differences that are not staleness and the build is
red for no reason. Too loose, it passes a notebook whose code no longer
matches its source. These pin both edges.
"""
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def builder(tmp_path, monkeypatch):
    """The build script, pointed at a scratch copy of notebooks/."""
    spec = importlib.util.spec_from_file_location(
        "build_notebooks", ROOT / "scripts" / "build_notebooks.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    scratch = tmp_path / "notebooks"
    shutil.copytree(ROOT / "notebooks", scratch,
                    ignore=shutil.ignore_patterns("__pycache__", ".ipynb_checkpoints"))
    monkeypatch.setattr(module, "NOTEBOOKS", scratch)
    return module


def _stems(builder):
    return sorted(p.stem for p in builder.NOTEBOOKS.glob("*.py"))


def _load(builder, stem):
    return json.loads((builder.NOTEBOOKS / f"{stem}.ipynb").read_text(encoding="utf-8"))


def _save(builder, stem, notebook):
    (builder.NOTEBOOKS / f"{stem}.ipynb").write_text(
        json.dumps(notebook, indent=1), encoding="utf-8")


def test_committed_notebooks_match_their_sources(builder):
    """The repository as committed must pass its own gate."""
    assert [builder.check(s) for s in _stems(builder)] == [None] * len(_stems(builder))
    assert builder.main(["build_notebooks.py", "--check"]) == 0


def test_kernel_metadata_is_not_staleness(builder):
    """Opening a notebook rewrites this block with the local kernel. That
    is what kept CI red: identical cells, a kernel called "base"."""
    stem = _stems(builder)[0]
    nb = _load(builder, stem)
    nb["metadata"] = {
        "kernelspec": {"display_name": "base", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.13.12",
                          "codemirror_mode": {"name": "ipython", "version": 3},
                          "file_extension": ".py", "pygments_lexer": "ipython3"},
    }
    _save(builder, stem, nb)
    assert builder.check(stem) is None


def test_cell_ids_and_layout_are_not_staleness(builder):
    stem = _stems(builder)[0]
    nb = _load(builder, stem)
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"renamed-{i}"
    (builder.NOTEBOOKS / f"{stem}.ipynb").write_text(
        json.dumps(nb, indent=4, sort_keys=False), encoding="utf-8")
    assert builder.check(stem) is None


def test_an_edited_cell_is_staleness(builder):
    """The case the gate exists for: the notebook's code no longer says
    what its source says."""
    stem = _stems(builder)[0]
    nb = _load(builder, stem)
    code = next(i for i, c in enumerate(nb["cells"]) if c["cell_type"] == "code")
    nb["cells"][code]["source"] = ["print('edited in the notebook only')\n"]
    _save(builder, stem, nb)
    problem = builder.check(stem)
    assert problem is not None and f"cell {code}" in problem
    assert builder.main(["build_notebooks.py", "--check"]) == 1


def test_an_edited_source_is_staleness(builder):
    """And the mirror image: the .py moved on and the notebook did not."""
    stem = _stems(builder)[0]
    path = builder.NOTEBOOKS / f"{stem}.py"
    path.write_text(path.read_text(encoding="utf-8") + "\nprint('added to the source')\n",
                    encoding="utf-8")
    assert builder.check(stem) is not None


def test_a_missing_or_extra_cell_is_staleness(builder):
    stem = _stems(builder)[0]
    nb = _load(builder, stem)
    nb["cells"].pop()
    _save(builder, stem, nb)
    assert "cells" in builder.check(stem)


def test_check_writes_nothing(builder):
    before = {p.name: p.read_bytes() for p in builder.NOTEBOOKS.glob("*.ipynb")}
    builder.main(["build_notebooks.py", "--check"])
    after = {p.name: p.read_bytes() for p in builder.NOTEBOOKS.glob("*.ipynb")}
    assert before == after


def test_build_then_check_round_trips(builder):
    stem = _stems(builder)[0]
    nb = _load(builder, stem)
    nb["cells"] = nb["cells"][:2]
    _save(builder, stem, nb)
    assert builder.check(stem) is not None
    builder.build(stem)
    assert builder.check(stem) is None
