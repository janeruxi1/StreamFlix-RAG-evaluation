"""Train and run the four ticket routers; write their per-ticket predictions.

This is the heavy half of notebook 11. It needs torch, transformers and
sentence-transformers, and for the LLM arm an API key. The notebook
needs none of those: it reads what this script writes.

    python scripts/run_routing.py                      # all four arms
    python scripts/run_routing.py --arms tfidf,frozen  # just these
    python scripts/run_routing.py --seeds 1            # quicker fine-tune

Outputs (both committed):
    reports/metrics/11_routing_predictions.csv  split, idx, arm, seed, pred, conf
    reports/metrics/11_routing.json             config, timings, versions

Re-running an arm replaces that arm's rows and leaves the others alone,
so the arms can be run on different days. LLM replies are cached under
.llm_cache/, so re-running the LLM arm is free after the first time.

Runs on a laptop CPU. The fine-tune is the slow part: a few minutes per
epoch, up to 5 epochs, per seed.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.routing import baselines, data  # noqa: E402
from src.routing.metrics import accuracy, macro_f1  # noqa: E402

PRED_PATH = Path("reports/metrics/11_routing_predictions.csv")
RECORD_PATH = Path("reports/metrics/11_routing.json")
ARMS = ("tfidf", "frozen", "finetune", "llm")
LATENCY_SAMPLE = 50


def versions() -> dict:
    out = {"python": platform.python_version()}
    for pkg in ("numpy", "pandas", "scikit-learn", "torch", "transformers",
                "sentence-transformers", "openai"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def rows(split: str, idx, arm: str, seed: int, pred, conf) -> pd.DataFrame:
    return pd.DataFrame({"split": split, "idx": np.asarray(idx, dtype=int),
                         "arm": arm, "seed": seed,
                         "pred": np.asarray(pred, dtype=int),
                         "conf": np.round(np.asarray(conf, dtype=float), 5)})


def latency(fn, texts: list[str]) -> dict:
    """Batched throughput over all texts, and single-ticket median."""
    started = time.perf_counter()
    fn(texts)
    batched = time.perf_counter() - started
    singles = []
    for t in texts[:LATENCY_SAMPLE]:
        s = time.perf_counter()
        fn([t])
        singles.append(time.perf_counter() - s)
    return {"ms_per_ticket_batched": round(1000 * batched / len(texts), 3),
            "ms_single_ticket_median": round(1000 * float(np.median(singles)), 2)}


def run_tfidf(sp) -> tuple[list[pd.DataFrame], dict]:
    started = time.perf_counter()
    model = baselines.tfidf_router().fit(sp["fit"].text, sp["fit"].label)
    fit_s = time.perf_counter() - started
    out = []
    for split in ("val", "test"):
        pred, conf = baselines.predict(model, sp[split].text)
        out.append(rows(split, sp[split].index, "tfidf", 0, pred, conf))
    lat = latency(lambda ts: model.predict_proba(ts), list(sp["test"].text))
    return out, {"train_seconds": round(fit_s, 1), **lat}


def run_frozen(sp, model_name: str) -> tuple[list[pd.DataFrame], dict]:
    from sentence_transformers import SentenceTransformer

    encoder = SentenceTransformer(model_name, device="cpu")
    encode = lambda ts: encoder.encode(list(ts), batch_size=64,  # noqa: E731
                                       normalize_embeddings=True,
                                       show_progress_bar=False)
    started = time.perf_counter()
    X = {s: encode(sp[s].text) for s in ("fit", "val", "test")}
    encode_s = time.perf_counter() - started
    started = time.perf_counter()
    clf = baselines.embedding_router().fit(X["fit"], sp["fit"].label)
    fit_s = time.perf_counter() - started
    out = []
    for split in ("val", "test"):
        pred, conf = baselines.predict(clf, X[split])
        out.append(rows(split, sp[split].index, "frozen", 0, pred, conf))
    lat = latency(lambda ts: clf.predict_proba(encode(ts)), list(sp["test"].text))
    return out, {"model": model_name, "encode_seconds": round(encode_s, 1),
                 "train_seconds": round(fit_s, 1), **lat}


def run_finetune(sp, n_labels: int, seeds: list[int],
                 model_name: str) -> tuple[list[pd.DataFrame], dict]:
    import torch

    from src.routing import finetune as ft

    torch.set_num_threads(max(1, torch.get_num_threads()))
    out, per_seed = [], []
    for seed in seeds:
        cfg = ft.FinetuneConfig(model_name=model_name, seed=seed)
        print(f"  seed {seed}")
        tokenizer, model = ft.load_pretrained(cfg, n_labels)
        started = time.perf_counter()
        history = ft.train(model, tokenizer, list(sp["fit"].text),
                           sp["fit"].label.to_numpy(), list(sp["val"].text),
                           sp["val"].label.to_numpy(), cfg)
        history["train_seconds"] = round(time.perf_counter() - started, 1)
        for split in ("val", "test"):
            proba = ft.predict_proba(model, tokenizer, list(sp[split].text),
                                     max_len=cfg.max_len)
            out.append(rows(split, sp[split].index, "finetune", seed,
                            proba.argmax(1), proba.max(1)))
        if seed == seeds[0]:
            history.update(latency(
                lambda ts: ft.predict_proba(model, tokenizer, ts,
                                            max_len=cfg.max_len),
                list(sp["test"].text)))
            history["n_parameters"] = int(sum(p.numel() for p in model.parameters()))
        per_seed.append({"seed": seed, **history})
    return out, {"config": ft.FinetuneConfig(model_name=model_name).as_dict(),
                 "seeds": per_seed}


def run_llm(sp, labels: list[str], assume_yes: bool) -> tuple[list[pd.DataFrame], dict]:
    from src.llm.provider import get_provider, provider_ready
    from src.routing import llm_router

    ready, blocker = provider_ready()
    if not ready:
        print(f"  skipped: {blocker}")
        return [], {}
    provider = get_provider()
    model = provider.default_model
    subset = data.llm_subset(sp["test"])
    texts = list(sp["test"].text.iloc[subset])
    cost = llm_router.estimate_cost(texts, labels, model)
    print(f"  {len(texts)} tickets with {model}; at most ~${cost:.2f} "
          f"if nothing is cached (list price, see PRICE_PER_M).")
    if not assume_yes and input("  Proceed? [y/N] ").strip().lower() != "y":
        print("  skipped by user")
        return [], {}
    started = time.perf_counter()
    result = llm_router.route(provider, texts, labels, model=model)
    unparsed = [r for r, p in zip(result["replies"], result["pred"])
                if p == llm_router.UNPARSED]
    live = result["live_seconds"]
    detail = {
        "provider": provider.name, "model": model,
        "n_tickets": len(texts), "per_class": data.LLM_PER_CLASS,
        "wall_seconds": round(time.perf_counter() - started, 1),
        "live_calls": len(live),
        "ms_single_ticket_median": round(1000 * float(np.median(live)), 1) if live else None,
        "unparsed": len(unparsed), "unparsed_examples": unparsed[:10],
        "est_cost_usd_per_1000": round(1000 * cost / len(texts), 4),
        "price_per_m_tokens": llm_router.PRICE_PER_M.get(model),
    }
    return [rows("test", subset, "llm", 0, result["pred"],
                 np.full(len(texts), np.nan))], detail


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--arms", default=",".join(ARMS))
    parser.add_argument("--seeds", type=int, default=3)
    parser.add_argument("--model", default="BAAI/bge-small-en-v1.5")
    parser.add_argument("--yes", action="store_true",
                        help="do not ask before paid LLM calls")
    args = parser.parse_args()

    import os
    os.chdir(ROOT)
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    bad = set(arms) - set(ARMS)
    if bad:
        parser.error(f"unknown arm(s): {sorted(bad)}; choose from {ARMS}")

    labels = data.load_labels()
    sp = data.splits()
    print(f"Banking77: fit {len(sp['fit'])}, val {len(sp['val'])}, "
          f"test {len(sp['test'])}, {len(labels)} intents")

    record = (json.loads(RECORD_PATH.read_text(encoding="utf-8"))
              if RECORD_PATH.exists() else {"arms": {}})
    preds = pd.read_csv(PRED_PATH) if PRED_PATH.exists() else pd.DataFrame()

    for arm in arms:
        print(f"\n== {arm} ==")
        if arm == "tfidf":
            frames, detail = run_tfidf(sp)
        elif arm == "frozen":
            frames, detail = run_frozen(sp, args.model)
        elif arm == "finetune":
            frames, detail = run_finetune(sp, len(labels),
                                          list(range(args.seeds)), args.model)
        else:
            frames, detail = run_llm(sp, labels, args.yes)
        if not frames:
            continue
        new = pd.concat(frames, ignore_index=True)
        if len(preds):
            preds = preds[preds.arm != arm]
        preds = pd.concat([preds, new], ignore_index=True)
        record["arms"][arm] = detail

        test = new[new.split == "test"]
        for seed, grp in test.groupby("seed"):
            y = sp["test"].label.to_numpy()[grp.idx.to_numpy()]
            print(f"  test (seed {seed}): accuracy {accuracy(y, grp.pred):.4f}, "
                  f"macro-F1 {macro_f1(y, grp.pred):.4f}")

        # Write after every arm, so a later failure keeps the earlier work.
        PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
        order = {a: i for i, a in enumerate(ARMS)}
        preds = preds.sort_values(
            ["arm", "seed", "split", "idx"],
            key=lambda c: c.map(order) if c.name == "arm" else c,
        ).reset_index(drop=True)
        preds.to_csv(PRED_PATH, index=False, lineterminator="\n")
        record.update({
            "dataset": "Banking77 (PolyAI, CC BY 4.0)",
            "splits": {k: len(v) for k, v in sp.items()},
            "split_seed": data.SPLIT_SEED, "val_fraction": data.VAL_FRACTION,
            "labels": labels,
            "versions": versions(),
            "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        # newline="\n": on Windows write_text would otherwise write CRLF,
        # which the repository checks reject as trailing whitespace.
        RECORD_PATH.write_text(json.dumps(record, indent=1) + "\n",
                               encoding="utf-8", newline="\n")

    print(f"\nWrote {PRED_PATH} and {RECORD_PATH}.")
    print("Next: python notebooks/11_ticket_routing.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
