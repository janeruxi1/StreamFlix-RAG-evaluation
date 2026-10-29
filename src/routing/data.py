"""Loading Banking77 and making the splits every arm shares."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path("data/routing")
VAL_FRACTION = 0.10
SPLIT_SEED = 0
LLM_PER_CLASS = 13          # 13 x 77 = 1,001 test tickets for the LLM arm


def load_labels(data_dir: Path = DATA_DIR) -> list[str]:
    """The 77 intent names, in the fixed order that defines label ids."""
    return json.loads((data_dir / "categories.json").read_text(encoding="utf-8"))


def load_split(name: str, data_dir: Path = DATA_DIR) -> pd.DataFrame:
    """One CSV with an integer `label` column added."""
    if name not in {"train", "test"}:
        raise ValueError(f"Unknown split {name!r}; expected 'train' or 'test'")
    df = pd.read_csv(data_dir / f"{name}.csv")
    index = {lab: i for i, lab in enumerate(load_labels(data_dir))}
    unknown = set(df["category"]) - set(index)
    if unknown:
        raise ValueError(f"{name}.csv has labels not in categories.json: "
                         f"{sorted(unknown)[:5]}")
    df["label"] = df["category"].map(index).astype(int)
    return df.reset_index(drop=True)


def stratified_indices(labels: np.ndarray, per_class_or_frac: float,
                       seed: int) -> np.ndarray:
    """Row positions drawn per class, sorted.

    A float in (0, 1) takes that fraction of each class (rounded, at
    least one); an integer takes that many from each class.
    """
    rng = np.random.default_rng(seed)
    picked: list[np.ndarray] = []
    for lab in np.unique(labels):
        rows = np.flatnonzero(labels == lab)
        if 0 < per_class_or_frac < 1:
            k = max(1, int(round(len(rows) * per_class_or_frac)))
        else:
            k = int(per_class_or_frac)
        if k > len(rows):
            raise ValueError(f"class {lab} has {len(rows)} rows, asked for {k}")
        picked.append(rng.choice(rows, size=k, replace=False))
    return np.sort(np.concatenate(picked))


def splits(data_dir: Path = DATA_DIR) -> dict[str, pd.DataFrame]:
    """fit / val / test, identical for every arm.

    The published train set is split 90/10, stratified by intent. Models
    fit on `fit`; anything tuned (early stopping, the routing threshold)
    is tuned on `val`; `test` is touched once, for the reported numbers.
    """
    train = load_split("train", data_dir)
    val_rows = stratified_indices(train["label"].to_numpy(), VAL_FRACTION,
                                  SPLIT_SEED)
    mask = np.zeros(len(train), dtype=bool)
    mask[val_rows] = True
    return {
        "fit": train[~mask].reset_index(drop=True),
        "val": train[mask].reset_index(drop=True),
        "test": load_split("test", data_dir),
    }


def llm_subset(test: pd.DataFrame, per_class: int = LLM_PER_CLASS,
               seed: int = SPLIT_SEED) -> np.ndarray:
    """Test row positions the LLM arm is scored on (balanced, fixed)."""
    return stratified_indices(test["label"].to_numpy(), per_class, seed)
