"""Fine-tuning a sentence encoder as an intent classifier, in plain PyTorch.

No Trainer, no Lightning: the loop is short enough to read in one
sitting, and every choice in it is one an interviewer can ask about.

    encoder (BGE-small) -> [CLS] vector -> dropout -> linear(77)

  - AdamW, with weight decay off for biases and LayerNorm weights
  - linear warmup over the first 6% of steps, then linear decay
  - gradient clipping at 1.0
  - dynamic padding: each batch is padded to its own longest ticket,
    not to max_len. Banking77 questions average 12 words, so this is
    most of the difference between minutes and an hour on a laptop CPU
  - early stopping on VALIDATION macro-F1, restoring the best weights

torch and transformers are imported inside the functions, so the rest
of the package (and CI's minimal install) never needs them.
"""
from __future__ import annotations

import copy
import random
import time
from dataclasses import asdict, dataclass

import numpy as np


@dataclass
class FinetuneConfig:
    model_name: str = "BAAI/bge-small-en-v1.5"
    max_len: int = 64
    batch_size: int = 32
    lr: float = 5e-5
    head_lr: float = 1e-3
    weight_decay: float = 0.01
    warmup_frac: float = 0.06
    max_epochs: int = 5
    patience: int = 2
    dropout: float = 0.1
    pooling: str = "cls"          # BGE is trained with [CLS] pooling
    seed: int = 0

    def as_dict(self) -> dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def build_classifier(encoder, n_labels: int, dropout: float, pooling: str):
    """Wrap any Hugging Face encoder with a pooling step and a linear head.

    The head is initialised from torch's current random state; call
    set_seed first for a reproducible run.
    """
    from torch import nn

    if pooling not in {"cls", "mean"}:
        raise ValueError(f"pooling must be 'cls' or 'mean', got {pooling!r}")

    class Classifier(nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = encoder
            self.dropout = nn.Dropout(dropout)
            self.head = nn.Linear(encoder.config.hidden_size, n_labels)

        def forward(self, input_ids, attention_mask, **kwargs):
            hidden = self.encoder(input_ids=input_ids,
                                  attention_mask=attention_mask).last_hidden_state
            if pooling == "cls":
                pooled = hidden[:, 0]
            else:
                mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp(min=1.0)
            return self.head(self.dropout(pooled))

    return Classifier()


def _batches(texts: list[str], labels, tokenizer, batch_size: int,
             max_len: int, shuffle: bool, rng: np.random.Generator | None):
    import torch

    order = np.arange(len(texts))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), batch_size):
        rows = order[start:start + batch_size]
        enc = tokenizer([texts[i] for i in rows], padding=True, truncation=True,
                        max_length=max_len, return_tensors="pt")
        batch = {"input_ids": enc["input_ids"],
                 "attention_mask": enc["attention_mask"]}
        if labels is not None:
            batch["labels"] = torch.as_tensor(np.asarray(labels)[rows],
                                              dtype=torch.long)
        yield batch


def predict_proba(model, tokenizer, texts: list[str], batch_size: int = 64,
                  max_len: int = 64) -> np.ndarray:
    """Softmax probabilities, (n_texts, n_labels)."""
    import torch

    model.eval()
    out = []
    with torch.no_grad():
        for batch in _batches(texts, None, tokenizer, batch_size, max_len,
                              shuffle=False, rng=None):
            out.append(torch.softmax(model(**batch), dim=-1).numpy())
    return np.concatenate(out) if out else np.zeros((0, 0))


def train(model, tokenizer, fit_texts: list[str], fit_labels,
          val_texts: list[str], val_labels, cfg: FinetuneConfig,
          log=print) -> dict:
    """Train in place; restore the best epoch by validation macro-F1.

    Returns the training history: per-epoch loss, validation accuracy and
    macro-F1, seconds, and which epoch was kept.
    """
    import torch
    from torch.nn.utils import clip_grad_norm_

    from src.routing.metrics import accuracy, macro_f1

    set_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    no_decay = ("bias", "LayerNorm.weight", "layer_norm.weight")
    groups = [
        {"params": [p for n, p in model.encoder.named_parameters()
                    if not any(k in n for k in no_decay)],
         "lr": cfg.lr, "weight_decay": cfg.weight_decay},
        {"params": [p for n, p in model.encoder.named_parameters()
                    if any(k in n for k in no_decay)],
         "lr": cfg.lr, "weight_decay": 0.0},
        # The head starts from random weights, so it gets a larger step.
        {"params": list(model.head.parameters()),
         "lr": cfg.head_lr, "weight_decay": cfg.weight_decay},
    ]
    optimizer = torch.optim.AdamW(groups)

    steps_per_epoch = -(-len(fit_texts) // cfg.batch_size)
    total = steps_per_epoch * cfg.max_epochs
    warmup = max(1, int(cfg.warmup_frac * total))
    schedule = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: (step + 1) / warmup if step < warmup
        else max(0.0, (total - step) / max(1, total - warmup)),
    )
    loss_fn = torch.nn.CrossEntropyLoss()

    history: list[dict] = []
    best_f1, best_state, best_epoch, stale = -1.0, None, 0, 0
    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        started, losses = time.perf_counter(), []
        for batch in _batches(fit_texts, fit_labels, tokenizer, cfg.batch_size,
                              cfg.max_len, shuffle=True, rng=rng):
            labels = batch.pop("labels")
            loss = loss_fn(model(**batch), labels)
            optimizer.zero_grad()
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            losses.append(loss.item())

        proba = predict_proba(model, tokenizer, val_texts, max_len=cfg.max_len)
        pred = proba.argmax(axis=1)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)),
               "val_accuracy": accuracy(val_labels, pred),
               "val_macro_f1": macro_f1(val_labels, pred),
               "seconds": round(time.perf_counter() - started, 1)}
        history.append(row)
        log(f"    epoch {epoch}: loss {row['train_loss']:.3f}  "
            f"val acc {row['val_accuracy']:.3f}  "
            f"val F1 {row['val_macro_f1']:.3f}  ({row['seconds']:.0f}s)")

        if row["val_macro_f1"] > best_f1:
            best_f1, best_epoch, stale = row["val_macro_f1"], epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            stale += 1
            if stale >= cfg.patience:
                log(f"    no gain for {cfg.patience} epochs; stopping")
                break

    model.load_state_dict(best_state)
    return {"epochs": history, "best_epoch": best_epoch,
            "best_val_macro_f1": best_f1}


def load_pretrained(cfg: FinetuneConfig, n_labels: int):
    """Tokenizer and classifier around the pretrained encoder.

    Downloads the encoder on first use (about 130 MB); after that it is
    read from the Hugging Face cache, the same copy notebooks 02-03 use.
    """
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name)
    encoder = AutoModel.from_pretrained(cfg.model_name)
    set_seed(cfg.seed)              # the head's initial weights depend on it
    return tokenizer, build_classifier(encoder, n_labels, cfg.dropout,
                                       cfg.pooling)
