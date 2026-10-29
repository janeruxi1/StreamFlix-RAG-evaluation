"""
Phase 11 — Routing the Ticket Before Anyone Answers It
=======================================================

Notebooks 01–10 measured what the assistant says once a question has
reached it. A support operation makes an earlier decision on every
ticket: which queue it goes to. Get that wrong and the best answer in
the world is written by the wrong team, a day late.

This phase compares four ways to make that decision, on 3,080 held-out
customer questions labelled with 77 intents:

    tfidf      word and character n-grams + logistic regression
    frozen     BGE-small embeddings (the project's retriever model),
               frozen, + logistic regression
    finetune   BGE-small fine-tuned end to end in a PyTorch loop
    llm        a general LLM, zero-shot, with the intent list in the prompt

and asks the question an operations lead would: which router lets us
send the most tickets automatically while keeping misroutes under 5%,
and what does each one cost to run?

The models are trained by scripts/run_routing.py, which needs torch and
(for the LLM arm) a key. This notebook needs neither. It reads the
per-ticket predictions that script wrote and recomputes every number
from them, and it retrains the TF-IDF arm live to show the stored
predictions are what the code produces.

Sections
--------
  A. The data, and why it is not StreamFlix
  B. The baseline, recomputed live
  C. Head to head on the full test set
  D. Confidence: calibration and how much can be routed automatically
  E. Cost and latency
  F. Where the routers fail
  G. The decision, and the note checked against these numbers
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src.evaluation.memo_check import report
from src.routing import baselines, data
from src.routing.metrics import (
    accuracy,
    bootstrap_ci,
    confused_pairs,
    ece,
    macro_f1,
    paired_bootstrap,
    risk_coverage,
    selective,
    threshold_for_target,
)

PRED_PATH = Path("reports/metrics/11_routing_predictions.csv")
RECORD_PATH = Path("reports/metrics/11_routing.json")
NOTE = Path("reports/routing_note.md")
FIG_DIR = Path("reports/figures")
TARGET = 0.95                    # auto-routed tickets must be >= 95% right
ARM_TITLES = {"tfidf": "TF-IDF + LR", "frozen": "BGE-small frozen + LR",
              "finetune": "BGE-small fine-tuned", "llm": "LLM zero-shot"}
ARM_COLOURS = {"tfidf": "#5B8FF9", "frozen": "#F6BD16",
               "finetune": "#5AD8A6", "llm": "#F6735B"}

labels = data.load_labels()
sp = data.splits()
y = {s: sp[s].label.to_numpy() for s in ("val", "test")}

if not PRED_PATH.exists() or not RECORD_PATH.exists():
    print(f"No routing predictions yet ({PRED_PATH}).\n"
          f"Run `python scripts/run_routing.py` first; it needs torch, and a\n"
          f"key for the LLM arm. Nothing else in the project depends on it.")
    raise SystemExit(0)

preds = pd.read_csv(PRED_PATH)
record = json.loads(RECORD_PATH.read_text(encoding="utf-8"))
arms = [a for a in ARM_TITLES if a in set(preds.arm)]


def arm_rows(arm: str, split: str, seed: int = 0) -> pd.DataFrame:
    """One arm's predictions for one split, in ticket order."""
    out = preds[(preds.arm == arm) & (preds.split == split) & (preds.seed == seed)]
    return out.sort_values("idx").reset_index(drop=True)


# =====================================================================
# A. The data, and why it is not StreamFlix
# =====================================================================
print("=" * 78)
print("A. THE DATA, AND WHY IT IS NOT STREAMFLIX")
print("=" * 78)
counts = sp["test"].label.value_counts()
print(f"""
  Banking77 (PolyAI, CC BY 4.0): real customer questions to an online
  bank, each labelled with one of {len(labels)} intents.

    fit   {len(sp['fit']):>6,}   models learn from these
    val   {len(sp['val']):>6,}   10% of the published train set, stratified;
                   early stopping and the routing threshold are
                   chosen here
    test  {len(sp['test']):>6,}   {counts.min()} per intent; touched once, for the
                   numbers below

  This is a different domain from the StreamFlix help centre in
  notebooks 01–10, and that is deliberate. Routing is a supervised
  problem: it needs thousands of labelled tickets. The StreamFlix
  corpus has 45 articles and 117 golden questions. Writing 13,000
  synthetic tickets to fill the gap would test how well a generator
  imitates itself, not how well a router works. A public set of real
  questions answers the methodological question honestly; what carries
  over to another help desk is the method, not the accuracy figure.

  77 intents is also the hard version of the problem. Many differ by a
  single detail a customer may not state:
""")
for a, b in [("card_arrival", "card_delivery_estimate"),
             ("top_up_failed", "top_up_reverted"),
             ("pending_card_payment", "card_payment_not_recognised")]:
    ex = sp["test"][sp["test"].category == a].text.iloc[0]
    print(f"    {a:<30} vs {b:<30}\n      e.g. \"{ex}\"")

missing = [a for a in ARM_TITLES if a not in arms]
print(f"\n  Arms in this run: {', '.join(arms)}"
      + (f"   (not run: {', '.join(missing)})" if missing else ""))


# =====================================================================
# B. The baseline, recomputed live
# =====================================================================
print("\n" + "=" * 78)
print("B. THE BASELINE, RECOMPUTED LIVE")
print("=" * 78)
model = baselines.tfidf_router().fit(sp["fit"].text, sp["fit"].label)
live_pred, live_conf = baselines.predict(model, sp["test"].text)
stored = arm_rows("tfidf", "test")
agree = float(np.mean(live_pred == stored.pred.to_numpy()))
print(f"""
  The TF-IDF router was retrained just now from the committed data and
  code, in seconds, and compared with the predictions the run stored:

    live accuracy     {accuracy(y['test'], live_pred):.4f}
    stored accuracy   {accuracy(y['test'], stored.pred):.4f}
    same prediction on {agree:.2%} of {len(stored):,} tickets
""")
if agree < 0.99:
    print("  FAILED — the stored TF-IDF predictions are not what this code\n"
          "  produces. Re-run scripts/run_routing.py --arms tfidf.\n")
    raise SystemExit(1)
print("  The other arms need torch or a key, so they are not retrained here;\n"
      "  their predictions are scored exactly as stored.")


# =====================================================================
# C. Head to head on the full test set
# =====================================================================
print("\n" + "=" * 78)
print("C. HEAD TO HEAD ON THE FULL TEST SET")
print("=" * 78)
results: dict[str, dict] = {}
for arm in [a for a in arms if a != "llm"]:
    seeds = sorted(preds[(preds.arm == arm)].seed.unique())
    per_seed = [arm_rows(arm, "test", s) for s in seeds]
    accs = [accuracy(y["test"], r.pred) for r in per_seed]
    f1s = [macro_f1(y["test"], r.pred) for r in per_seed]
    lo, hi = bootstrap_ci(y["test"], per_seed[0].pred)
    results[arm] = {"seeds": seeds, "acc": accs[0], "f1": f1s[0],
                    "acc_ci": (lo, hi), "acc_seeds": accs, "f1_seeds": f1s}

print(f"\n    {'router':<24}{'accuracy':>9}{'95% CI':>18}{'macro-F1':>10}   seeds")
print("    " + "-" * 72)
for arm, r in results.items():
    spread = (f"{min(r['acc_seeds']):.3f}–{max(r['acc_seeds']):.3f}"
              if len(r["seeds"]) > 1 else "1")
    print(f"    {ARM_TITLES[arm]:<24}{r['acc']:>9.3f}   "
          f"[{r['acc_ci'][0]:.3f}, {r['acc_ci'][1]:.3f}]{r['f1']:>10.3f}   {spread}")
print("""
  Seed 0 is the reported model for every arm, fixed before the run. For
  the fine-tune, the other seeds show how much of a difference is the
  random start rather than the method.""")

comparisons = {}
if "finetune" in results:
    ft_pred = arm_rows("finetune", "test").pred.to_numpy()
    for other in [a for a in ("tfidf", "frozen") if a in results]:
        comparisons[other] = paired_bootstrap(
            y["test"], ft_pred, arm_rows(other, "test").pred.to_numpy())
    if comparisons:
        print("\n  Fine-tuned minus each alternative, paired on the same tickets:")
        for other, c in comparisons.items():
            print(f"    vs {ARM_TITLES[other]:<24}{c['diff']*100:+6.1f} pts  "
                  f"[{c['ci_low']*100:+.1f}, {c['ci_high']*100:+.1f}]  "
                  f"p = {c['p_value']:.3f}")

    # Seed 0 is one draw. The same comparison for every seed shows which
    # of the gains survive a different random start.
    if "frozen" in results and len(results["finetune"]["seeds"]) > 1:
        frozen_pred = arm_rows("frozen", "test").pred.to_numpy()
        tfidf_pred = arm_rows("tfidf", "test").pred.to_numpy()
        per_seed_cmp = []
        print("\n  The same comparisons for every fine-tune seed:")
        print(f"    {'seed':<6}{'accuracy':>10}{'vs TF-IDF':>22}{'vs frozen':>26}")
        for seed in results["finetune"]["seeds"]:
            sp_pred = arm_rows("finetune", "test", seed).pred.to_numpy()
            ct = paired_bootstrap(y["test"], sp_pred, tfidf_pred)
            cf = paired_bootstrap(y["test"], sp_pred, frozen_pred)
            per_seed_cmp.append({"seed": seed, "tfidf": ct, "frozen": cf})
            print(f"    {seed:<6}{accuracy(y['test'], sp_pred):>10.3f}"
                  f"{ct['diff']*100:>+9.1f} (p={ct['p_value']:.3f})"
                  f"{cf['diff']*100:>+13.1f} (p={cf['p_value']:.3f})")
        results["finetune"]["per_seed_cmp"] = per_seed_cmp
        sig = lambda key: sum(c[key]["ci_low"] > 0 for c in per_seed_cmp)  # noqa: E731
        print(f"""
  The gain over TF-IDF is significant for {sig('tfidf')} of {len(per_seed_cmp)} seeds; the gain over
  frozen embeddings for {sig('frozen')} of {len(per_seed_cmp)}. A difference that depends on which
  random start was drawn is not one to plan a 100-minute training job
  around on accuracy grounds alone. Section D shows what does hold.""")

llm = None
if "llm" in arms:
    llm_rows = preds[(preds.arm == "llm") & (preds.split == "test")].sort_values("idx")
    sub = llm_rows.idx.to_numpy()
    y_sub = y["test"][sub]
    llm = {"n": len(sub), "acc": accuracy(y_sub, llm_rows.pred),
           "f1": macro_f1(y_sub, llm_rows.pred, labels=list(range(len(labels)))),
           "unparsed": int((llm_rows.pred == -1).sum()),
           "ci": bootstrap_ci(y_sub, llm_rows.pred.to_numpy())}
    print(f"""
  The LLM arm ran on a balanced {llm['n']:,}-ticket subset of the test set
  ({data.LLM_PER_CLASS} per intent) to keep the paid run small. Every arm is
  scored on that same subset here, so the comparison is like for like:
""")
    print(f"    {'router':<24}{'accuracy on subset':>20}")
    for arm in results:
        sub_pred = arm_rows(arm, "test").pred.to_numpy()[sub]
        results[arm]["acc_llm_subset"] = accuracy(y_sub, sub_pred)
        print(f"    {ARM_TITLES[arm]:<24}{results[arm]['acc_llm_subset']:>20.3f}")
    print(f"    {ARM_TITLES['llm']:<24}{llm['acc']:>20.3f}"
          f"   ({llm['unparsed']} replies were not an intent name)")
    if "finetune" in results:
        llm["vs_finetune"] = paired_bootstrap(
            y_sub, arm_rows("finetune", "test").pred.to_numpy()[sub],
            llm_rows.pred.to_numpy())

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

FIG_DIR.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 3.2))
rows_plot = [(a, r["acc"], r["acc_ci"]) for a, r in results.items()]
if llm:
    rows_plot.append(("llm", llm["acc"], llm["ci"]))
for i, (arm, acc, (lo, hi)) in enumerate(rows_plot):
    ax.plot([lo, hi], [i, i], color=ARM_COLOURS[arm], linewidth=2)
    ax.plot(acc, i, "o", color=ARM_COLOURS[arm], markersize=9,
            markeredgecolor="white", markeredgewidth=2)
    note = "  (1,001-ticket subset)" if arm == "llm" else ""
    ax.annotate(f"{acc:.1%}{note}", (hi, i), xytext=(8, 0),
                textcoords="offset points", va="center", fontsize=9,
                color="#2B2B2B")
ax.set_yticks(range(len(rows_plot)), [ARM_TITLES[a] for a, *_ in rows_plot])
ax.invert_yaxis()
ax.set_xlabel("test accuracy, 77 intents (95% bootstrap CI)")
ax.grid(axis="x", color="#E8E8E8")
ax.set_axisbelow(True)
for side in ("top", "right", "left"):
    ax.spines[side].set_visible(False)
ax.set_xlim(right=min(1.0, ax.get_xlim()[1] + 0.04))
ax.set_title("Phase 11 — routing accuracy by approach", loc="left", fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "11_routing_accuracy.png", dpi=140, bbox_inches="tight",
            facecolor="white")
plt.close()


# =====================================================================
# D. Confidence: calibration and how much can be routed automatically
# =====================================================================
print("\n" + "=" * 78)
print("D. CONFIDENCE: CALIBRATION AND AUTOMATIC ROUTING")
print("=" * 78)
print(f"""
  Accuracy averages over every ticket. Operations does not have to
  route every ticket automatically: a confident router sends the easy
  ones straight to the right queue and hands the rest to a person. The
  useful number is COVERAGE — the share routed automatically while the
  automatic ones stay at least {TARGET:.0%} correct.

  The threshold is chosen on the validation split and applied unchanged
  to test, so the reported coverage is not tuned on the data it is
  measured on.
""")
print(f"    {'router':<24}{'ECE':>7}{'threshold':>11}{'coverage':>10}"
      f"{'acc. routed':>13}{'acc. to people':>16}")
print("    " + "-" * 81)
for arm in results:
    v, t = arm_rows(arm, "val"), arm_rows(arm, "test")
    thr = threshold_for_target(v.conf, v.pred.to_numpy() == y["val"], TARGET)
    correct_t = t.pred.to_numpy() == y["test"]
    sel = selective(t.conf, correct_t, thr)
    results[arm].update({"ece": ece(t.conf, correct_t), "sel": sel})
    print(f"    {ARM_TITLES[arm]:<24}{results[arm]['ece']:>7.3f}{thr:>11.3f}"
          f"{sel['coverage']:>10.1%}{sel['accuracy_routed']:>13.1%}"
          f"{sel['accuracy_to_people']:>16.1%}")
if llm:
    print(f"    {ARM_TITLES['llm']:<24}{'—':>7}{'—':>11}{'—':>10}   "
          f"no probability to threshold on")

if "finetune" in results and len(results["finetune"]["seeds"]) > 1:
    seed_sel = []
    for seed in results["finetune"]["seeds"]:
        v, t = arm_rows("finetune", "val", seed), arm_rows("finetune", "test", seed)
        thr = threshold_for_target(v.conf, v.pred.to_numpy() == y["val"], TARGET)
        correct_t = t.pred.to_numpy() == y["test"]
        seed_sel.append({"seed": seed, "ece": ece(t.conf, correct_t),
                         **selective(t.conf, correct_t, thr)})
    results["finetune"]["seed_sel"] = seed_sel
    print("\n  Fine-tune, every seed:")
    for r in seed_sel:
        print(f"    seed {r['seed']}{r['ece']:>13.3f}{r['threshold']:>11.3f}"
              f"{r['coverage']:>10.1%}{r['accuracy_routed']:>13.1%}")
    held = sum(r["accuracy_routed"] >= TARGET for r in seed_sel)
    print(f"""
  This is the difference that survives the seeds. Fine-tuning leaves
  the router's confidence calibrated (ECE {min(r['ece'] for r in seed_sel):.3f}–{max(r['ece'] for r in seed_sel):.3f} for every seed,
  against {results.get('frozen', {}).get('ece', float('nan')):.3f} for frozen embeddings), so a threshold chosen on validation
  keeps its promise on test: {held} of {len(seed_sel)} seeds route at or above {TARGET:.0%}.""")
    if "frozen" in results:
        fz = results["frozen"]["sel"]
        verdict = ("holds" if fz["accuracy_routed"] >= TARGET
                   else f"misses: {fz['accuracy_routed']:.1%} on test")
        t_fz = arm_rows("frozen", "test")
        print(f"""  The frozen router's threshold {verdict}. Its probabilities run low
  (average confidence {t_fz.conf.mean():.1%} for a router that is {results['frozen']['acc']:.1%} right), so
  the cut lands at {fz['threshold']:.3f}, in a thinly populated low-confidence band
  whose accuracy shifts between validation and test. A recalibration
  step (temperature scaling on validation) is the obvious fix and was
  not run here.""")
print("""
  ECE is the expected calibration error: the average gap between the
  router's stated confidence and how often it is right. Low ECE is what
  makes a threshold mean what it says.

  The LLM returns a label with no probability, so it cannot be used
  this way without a second mechanism (sampling several answers and
  measuring agreement, say), which multiplies its cost.""")

fig, ax = plt.subplots(figsize=(8, 4))
for arm in results:
    t = arm_rows(arm, "test")
    cov, risk = risk_coverage(t.conf, t.pred.to_numpy() == y["test"])
    ax.plot(cov, risk, color=ARM_COLOURS[arm], linewidth=2, label=ARM_TITLES[arm])
    s = results[arm]["sel"]
    ax.plot(s["coverage"], 1 - s["accuracy_routed"], "o", color=ARM_COLOURS[arm],
            markersize=8, markeredgecolor="white", markeredgewidth=2)
ax.axhline(1 - TARGET, linestyle="--", color="#2B2B2B", linewidth=1)
ax.annotate(f"{1 - TARGET:.0%} misroute limit", (0.02, 1 - TARGET),
            xytext=(0, 4), textcoords="offset points", fontsize=9,
            color="#2B2B2B")
ax.set_xlabel("share of tickets routed automatically (most confident first)")
ax.set_ylabel("misroute rate among them")
ax.set_ylim(0, max(0.12, ax.get_ylim()[1]))
ax.grid(color="#E8E8E8")
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, loc="upper left")
ax.set_title("Phase 11 — how much can be routed without a person "
             "(dots: threshold chosen on validation)", loc="left", fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / "11_routing_coverage.png", dpi=140, bbox_inches="tight",
            facecolor="white")
plt.close()


# =====================================================================
# E. Cost and latency
# =====================================================================
print("\n" + "=" * 78)
print("E. COST AND LATENCY")
print("=" * 78)
print(f"""
  Measured on the machine that ran scripts/run_routing.py (CPU only;
  Python {record.get('versions', {}).get('python', '?')}). Batched = the whole test set at once,
  per ticket; single = one ticket at a time, median, which is what a
  live queue sees.
""")
print(f"    {'router':<24}{'train':>10}{'batched ms':>12}{'single ms':>11}"
      f"{'$ per 1,000':>13}")
print("    " + "-" * 70)
ops = {}
for arm in arms:
    d = record["arms"].get(arm, {})
    if arm == "finetune":
        first = d["seeds"][0]
        train_s = first["train_seconds"]
        batched, single = first["ms_per_ticket_batched"], first["ms_single_ticket_median"]
    elif arm == "llm":
        train_s, batched, single = 0.0, None, d.get("ms_single_ticket_median")
    else:
        train_s = d.get("train_seconds", 0) + d.get("encode_seconds", 0)
        batched, single = d["ms_per_ticket_batched"], d["ms_single_ticket_median"]
    cost = d.get("est_cost_usd_per_1000", 0.0) if arm == "llm" else 0.0
    ops[arm] = {"train_s": train_s, "batched": batched, "single": single,
                "cost": cost}
    fmt = lambda v, w: f"{'—':>{w}}" if v is None else f"{v:>{w}.2f}"  # noqa: E731
    print(f"    {ARM_TITLES[arm]:<24}{train_s / 60:>8.1f} m{fmt(batched, 12)}"
          f"{fmt(single, 11)}{'$' + format(cost, '.3f'):>13}")
if "llm" in arms:
    d = record["arms"]["llm"]
    print(f"""
  The LLM cost is an estimate at list price for {d['model']}
  (${d['price_per_m_tokens']['input']:.2f} / ${d['price_per_m_tokens']['output']:.2f} per million input / output
  tokens, ~4 characters per token), recorded with the run. Most of each
  prompt is the 77-intent list, repeated on every call.""")
print("""
  The local routers cost electricity. At a few milliseconds per ticket
  on a laptop CPU, one machine routes millions of tickets a day.""")


# =====================================================================
# F. Where the routers fail
# =====================================================================
print("\n" + "=" * 78)
print("F. WHERE THE ROUTERS FAIL")
print("=" * 78)
best = max(results, key=lambda a: results[a]["acc"])
best_pred = arm_rows(best, "test").pred.to_numpy()
print(f"\n  Most frequent mistakes of the best router ({ARM_TITLES[best]}):\n")
pairs = confused_pairs(y["test"], best_pred, labels, top=6)
for true, pred, n in pairs:
    print(f"    {n:>3} x  {true:<34} -> {pred}")
wrong_best = best_pred != y["test"]
if best != "tfidf":
    tfidf_pred = arm_rows("tfidf", "test").pred.to_numpy()
    shared = wrong_best & (tfidf_pred != y["test"])
    print(f"""
  {int(wrong_best.sum())} tickets are misrouted by the best router; {int(shared.sum())} of them are
  also misrouted by TF-IDF. Mistakes both make are the ones to read
  first: a different model family did not fix them, so the remedy is in the
  queue design (merge two queues, or ask one clarifying question). The
  most common of them:
""")
    for true, pred, n in confused_pairs(y["test"][shared], best_pred[shared],
                                        labels, top=5):
        print(f"    {n:>3} x  {true:<34} -> {pred}")


# =====================================================================
# G. The decision, and the note checked against these numbers
# =====================================================================
print("\n" + "=" * 78)
print("G. THE DECISION")
print("=" * 78)
all_arms = set(ARM_TITLES) <= set(arms)
if not all_arms:
    print("\n  Not every arm has run yet, so there is no decision to check.\n")
    raise SystemExit(0)

ft, tf, fz = results["finetune"], results["tfidf"], results["frozen"]
c = comparisons["tfidf"]
seed_sel = ft["seed_sel"]
print(f"""
  Against TF-IDF, fine-tuning is better on every seed and in every
  measure. Against frozen embeddings its accuracy edge is within seed
  noise; its calibration edge is not, and calibration is what decides
  how many tickets can skip a person. The LLM, the only arm needing no
  training data, scores {(llm['acc'] - ft['acc_llm_subset'])*100:+.1f} points against the fine-tune on the same
  tickets, costs money per ticket, and gives no confidence to threshold.

  The write-up is {NOTE}; its figures are checked here.
""")
if not NOTE.exists():
    print(f"  FAILED — {NOTE} is missing.\n")
    raise SystemExit(1)

claims = [
    ("fine-tune accuracy", f"{ft['acc']:.1%}"),
    ("fine-tune seed range", f"{min(ft['acc_seeds']):.1%}–{max(ft['acc_seeds']):.1%}"),
    ("TF-IDF accuracy", f"{tf['acc']:.1%}"),
    ("frozen accuracy", f"{results['frozen']['acc']:.1%}"),
    ("LLM accuracy on subset", f"{llm['acc']:.1%}"),
    ("fine-tune on LLM subset", f"{ft['acc_llm_subset']:.1%}"),
    ("gain over TF-IDF", f"{c['diff']*100:+.1f} points"),
    ("gain CI", f"[{c['ci_low']*100:+.1f}, {c['ci_high']*100:+.1f}]"),
    ("fine-tune coverage", f"{ft['sel']['coverage']:.1%}"),
    ("TF-IDF coverage", f"{tf['sel']['coverage']:.1%}"),
    ("frozen coverage", f"{results['frozen']['sel']['coverage']:.1%}"),
    ("fine-tune ECE", f"{ft['ece']:.3f}"),
    ("frozen ECE", f"{fz['ece']:.3f}"),
    ("frozen routed accuracy", f"{fz['sel']['accuracy_routed']:.1%}"),
    ("fine-tune routed acc., seeds",
     f"{min(r['accuracy_routed'] for r in seed_sel):.1%}–{max(r['accuracy_routed'] for r in seed_sel):.1%}"),
    ("fine-tune coverage, seeds",
     f"{min(r['coverage'] for r in seed_sel):.1%}–{max(r['coverage'] for r in seed_sel):.1%}"),
    ("vs frozen, seed 0", f"{comparisons['frozen']['diff']*100:+.1f} points"),
    ("seeds beating frozen", f"{sum(x['frozen']['ci_low'] > 0 for x in ft['per_seed_cmp'])} of {len(seed_sel)} seeds"),
    ("fine-tune training time", f"{ops['finetune']['train_s'] / 60:.0f} minutes"),
    ("TF-IDF routed accuracy", f"{tf['sel']['accuracy_routed']:.1%}"),
    ("TF-IDF ECE", f"{tf['ece']:.3f}"),
    *[(f"{a} single-ticket latency", f"{ops[a]['single']:.0f} ms") for a in ARM_TITLES],
    ("LLM cost per 1,000", f"${ops['llm']['cost']:.3f}"),
    ("LLM unparsed", f"{llm['unparsed']} of {llm['n']:,}"),
]
print(f"    {'claim':<34}{'the note must contain':<46}found")
print("    " + "-" * 86)
failed = report(NOTE, claims)
print()
if failed:
    print(f"  FAILED — {failed} figure(s) in {NOTE} no longer match this notebook.\n")
    raise SystemExit(1)
print(f"  All {len(claims)} figures in {NOTE} match, recomputed live.\n")

# The README repeats the headline. Same check, same failure.
README = Path("README.md")
readme_claims = [c for c in claims if c[0] in {
    "fine-tune accuracy", "TF-IDF accuracy", "frozen accuracy",
    "LLM accuracy on subset", "gain over TF-IDF", "gain CI",
    "fine-tune coverage", "TF-IDF coverage", "frozen coverage",
    "frozen routed accuracy", "fine-tune ECE", "frozen ECE", "TF-IDF ECE",
    "LLM cost per 1,000", "LLM unparsed", "seeds beating frozen"}]
readme_claims.append(("fine-tune routed accuracy",
                      f"{ft['sel']['accuracy_routed']:.1%} correct"))
print(f"    {'claim':<34}{'the README must contain':<46}found")
print("    " + "-" * 86)
failed = report(README, readme_claims)
print()
if failed:
    print(f"  FAILED — {failed} figure(s) in {README} no longer match this notebook.\n")
    raise SystemExit(1)
print(f"  All {len(readme_claims)} routing figures in {README} match.\n")
