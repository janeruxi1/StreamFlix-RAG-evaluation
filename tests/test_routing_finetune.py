"""The PyTorch fine-tune loop, on a tiny randomly-initialised BERT.

Nothing is downloaded: the encoder is built from a config and the
tokenizer from a vocabulary written to a temp folder. That makes this a
test of the LOOP (batching, padding, the optimiser groups, the schedule,
early stopping, restoring the best weights), which is the part written
here; the pretrained weights are someone else's.

Skipped where torch or transformers is missing, which includes CI's
main job. A separate CI job installs the CPU wheels and runs this file.
"""
import numpy as np
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from src.routing import finetune as ft  # noqa: E402

WORDS = ["card", "arrive", "late", "where", "rate", "exchange", "currency",
         "today", "my", "is", "the", "what", "not", "yet", "fee", "charged"]


@pytest.fixture()
def tiny(tmp_path):
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *WORDS]
    (tmp_path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    # transformers 5 takes the vocabulary as `vocab`; 4.x takes a file
    # path as `vocab_file` and silently ignores `vocab`. A tokenizer that
    # maps every word to [UNK] still trains (on ticket length alone), so
    # the check below is what stops this test passing for the wrong reason.
    tokenizer = transformers.BertTokenizerFast(
        vocab={w: i for i, w in enumerate(vocab)}, do_lower_case=True)
    if "[UNK]" in tokenizer.tokenize("card"):
        tokenizer = transformers.BertTokenizerFast(
            vocab_file=str(tmp_path / "vocab.txt"), do_lower_case=True)
    assert tokenizer.tokenize("Where is my card") == ["where", "is", "my", "card"]
    config = transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=32, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=64,
        max_position_embeddings=64)

    def make(seed=0, pooling="cls"):
        ft.set_seed(seed)
        encoder = transformers.BertModel(config)
        return ft.build_classifier(encoder, 3, dropout=0.0, pooling=pooling)

    return tokenizer, make


def toy_data(n=60, seed=0):
    rng = np.random.default_rng(seed)
    bank = [["where", "is", "my", "card"], ["card", "not", "arrive", "yet"],
            ["exchange", "rate", "today"], ["what", "currency", "rate"],
            ["fee", "charged"], ["the", "fee", "is", "charged"]]
    labels = [0, 0, 1, 1, 2, 2]
    picks = rng.integers(0, len(bank), n)
    return [" ".join(bank[i]) for i in picks], np.array([labels[i] for i in picks])


CFG = dict(batch_size=8, lr=3e-3, head_lr=3e-3, max_epochs=8, patience=3,
           warmup_frac=0.1, max_len=16)


def test_loop_learns_a_separable_problem(tiny):
    tokenizer, make = tiny
    model = make()
    texts, y = toy_data()
    vt, vy = toy_data(30, seed=1)
    hist = ft.train(model, tokenizer, texts, y, vt, vy,
                    ft.FinetuneConfig(**CFG), log=lambda *_: None)
    assert hist["best_val_macro_f1"] >= 0.9
    proba = ft.predict_proba(model, tokenizer, vt, max_len=16)
    assert proba.shape == (30, 3)
    assert np.allclose(proba.sum(1), 1, atol=1e-5)
    assert (proba.argmax(1) == vy).mean() >= 0.9


def test_best_epoch_weights_are_restored(tiny):
    tokenizer, make = tiny
    model = make()
    texts, y = toy_data()
    vt, vy = toy_data(30, seed=1)
    hist = ft.train(model, tokenizer, texts, y, vt, vy,
                    ft.FinetuneConfig(**CFG), log=lambda *_: None)
    best = hist["epochs"][hist["best_epoch"] - 1]["val_macro_f1"]
    from src.routing.metrics import macro_f1
    pred = ft.predict_proba(model, tokenizer, vt, max_len=16).argmax(1)
    assert macro_f1(vy, pred) == pytest.approx(best)


def test_early_stopping_stops(tiny):
    tokenizer, make = tiny
    model = make()
    texts, y = toy_data()
    # learning rate 0: validation can never improve after epoch 1
    cfg = ft.FinetuneConfig(**{**CFG, "lr": 0.0, "head_lr": 0.0,
                               "max_epochs": 10, "patience": 2})
    hist = ft.train(model, tokenizer, texts, y, texts, y, cfg, log=lambda *_: None)
    assert len(hist["epochs"]) == 3 and hist["best_epoch"] == 1


def test_same_seed_same_result(tiny):
    tokenizer, make = tiny
    texts, y = toy_data()
    outs = []
    for _ in range(2):
        model = make(seed=7)
        ft.train(model, tokenizer, texts, y, texts[:20], y[:20],
                 ft.FinetuneConfig(**{**CFG, "max_epochs": 2, "seed": 7}),
                 log=lambda *_: None)
        outs.append(ft.predict_proba(model, tokenizer, texts[:10], max_len=16))
    assert np.allclose(outs[0], outs[1], atol=1e-6)


def test_padding_does_not_change_predictions(tiny):
    """Dynamic padding is only safe if pooling ignores pad tokens."""
    tokenizer, make = tiny
    for pooling in ("cls", "mean"):
        model = make(pooling=pooling).eval()
        short, long = "fee charged", "where is my card not arrive yet today"
        alone = ft.predict_proba(model, tokenizer, [short], max_len=16)
        padded = ft.predict_proba(model, tokenizer, [short, long], max_len=16)[:1]
        assert np.allclose(alone, padded, atol=1e-5), pooling


def test_bad_pooling_rejected(tiny):
    with pytest.raises(ValueError):
        ft.build_classifier(transformers.BertModel(transformers.BertConfig(
            vocab_size=30, hidden_size=8, num_hidden_layers=1,
            num_attention_heads=1, intermediate_size=8)), 2, 0.0, "max")


def _subset_splits(n_per_class=2):
    """A few tickets per intent, enough to exercise the script's glue."""
    from src.routing import data
    sp = data.splits()
    out = {}
    for name, df in sp.items():
        keep = data.stratified_indices(df.label.to_numpy(), n_per_class, seed=0)
        # keep the ORIGINAL row positions as the index: the script writes
        # them as `idx`, and the notebook looks tickets up by them
        out[name] = df.iloc[keep]
    return out


def _save_tiny_model(path):
    from src.routing import data
    words = sorted({w for t in data.load_split("train").text.head(500)
                    for w in t.lower().replace("?", " ").split()})
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", *words]
    (path / "vocab.txt").write_text("\n".join(vocab) + "\n")
    tok = transformers.BertTokenizerFast(
        vocab={w: i for i, w in enumerate(vocab)}, do_lower_case=True)
    if "[UNK]" in tok.tokenize(words[0]):
        tok = transformers.BertTokenizerFast(vocab_file=str(path / "vocab.txt"))
    tok.save_pretrained(path)
    transformers.BertModel(transformers.BertConfig(
        vocab_size=len(vocab), hidden_size=16, num_hidden_layers=1,
        num_attention_heads=2, intermediate_size=32,
        max_position_embeddings=128)).save_pretrained(path)
    return str(path)


def test_script_finetune_arm_end_to_end(tmp_path, monkeypatch):
    """run_routing's fine-tune arm, with a local tiny model for BGE."""
    import scripts.run_routing as rr
    from src.routing import finetune

    monkeypatch.setattr(finetune.FinetuneConfig, "max_epochs", 1)
    monkeypatch.setattr(rr, "LATENCY_SAMPLE", 3)
    sp = _subset_splits()
    frames, detail = rr.run_finetune(sp, 77, [0, 1], _save_tiny_model(tmp_path))
    import pandas as pd
    out = pd.concat(frames)
    assert set(out.seed) == {0, 1} and set(out.split) == {"val", "test"}
    assert sorted(out[(out.seed == 0) & (out.split == "test")].idx) == sorted(sp["test"].index)
    assert out.conf.between(0, 1).all()
    assert [s["seed"] for s in detail["seeds"]] == [0, 1]
    assert "ms_per_ticket_batched" in detail["seeds"][0]


def test_script_frozen_arm_end_to_end(tmp_path, monkeypatch):
    pytest.importorskip("sentence_transformers")
    import scripts.run_routing as rr

    monkeypatch.setattr(rr, "LATENCY_SAMPLE", 3)
    sp = _subset_splits()
    frames, detail = rr.run_frozen(sp, _save_tiny_model(tmp_path))
    assert {f.split.iloc[0] for f in frames} == {"val", "test"}
    assert detail["encode_seconds"] >= 0
