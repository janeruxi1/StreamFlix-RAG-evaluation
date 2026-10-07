"""
Phase 9 — What the Numbers Are Worth
=====================================

Phase 7 recommended a pilot because 25 of 25 refusals only bounds the
true rate at 86.7%. That argument is incomplete. A bound is too low or
high enough only relative to what a wrong answer costs, and the memo
never said.

This phase states the cost model, then asks three questions of it:

  - At what cost of a bad answer does the system stop paying at all?
  - What refusal rate does the business need, and has it been shown?
  - Of the two things still uncertain, which one actually moves the
    decision — and so what should a pilot be sized to measure?

Everything here is arithmetic on rates measured in Phase 5. Nothing
needs a credential, so CI recomputes all of it. The business inputs
(what a bad answer costs, how much traffic is unanswerable) are
ASSUMPTIONS, shown as ranges, and the conclusions are stated as "for
which assumptions does the decision hold" rather than as one number.

Sections
--------
  A. Inputs: what is measured, what is assumed
  B. Where the system stops paying
  C. The refusal rate the business needs
  D. Which uncertainty moves the decision
  E. What the pilot has to measure
  F. Verification of the pilot brief
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluation.decision import (
    breakeven_cost_ratio,
    clean_sweep_n,
    design_effect,
    expected_value,
    n_for_upper_bound,
    required_refusal_rate,
    two_proportion_n,
    wilson_interval,
)
from src.evaluation.memo_check import report

RECORD = Path("reports/metrics/05_judged_arms.json")
BRIEF = Path("reports/pilot_design.md")
ARM = "llm_cited"

# --- Assumptions. None of these is measured. ---------------------------
COST_RATIOS = [1, 5, 10, 20, 50]        # k: one bad answer, in deflected tickets
OOS_SHARES = [0.05, 0.10, 0.21]         # p: share of traffic that is unanswerable
REFUSAL_COST = 0.10                     # f: a refusal, in deflected tickets
BASE_K, BASE_P = 10, 0.10               # the case worked through in D and E
# Pilot planning inputs, placeholders until real figures replace them.
BASELINE_RESOLUTION = 0.70              # first-contact resolution today
MIN_EFFECT = 0.03                       # smallest lift worth detecting
TICKETS_PER_AGENT = 50
AGENT_ICC = 0.05
DAILY_TICKETS = 1_000


# =====================================================================
# A. Inputs
# =====================================================================
print("=" * 78)
print("A. INPUTS — MEASURED AND ASSUMED")
print("=" * 78)

if not RECORD.exists():
    print(f"""
  {RECORD} is missing, so there are no measured rates to reason from.
  Run `python scripts/run_llm_eval.py` first.
""")
    raise SystemExit(0)

ref = json.loads(RECORD.read_text(encoding="utf-8"))["refusals"][ARM]
n_in = ref["in_scope_answered"] + ref["in_scope_refused"]
n_ans, n_bad = ref["in_scope_answered"], ref["in_scope_answers_unfaithful"]
n_oos, n_ref = ref["out_of_scope_n"], ref["out_of_scope_refused"]

a = n_ans / n_in
b, (b_lo, b_hi) = n_bad / n_ans, wilson_interval(n_bad, n_ans)
r, (r_lo, r_hi) = n_ref / n_oos, wilson_interval(n_ref, n_oos)

print(f"""
  Measured, for `{ARM}` (Phase 5 record):

    a  answers an answerable question      {f'{n_ans} of {n_in}':<10} {a:>6.1%}
    b  an answer is unsupported            {f'{n_bad} of {n_ans}':<10} {b:>6.1%}   95% interval [{b_lo:.1%}, {b_hi:.1%}]
    r  refuses an unanswerable question    {f'{n_ref} of {n_oos}':<10} {r:>6.1%}   95% interval [{r_lo:.1%}, {r_hi:.1%}]

  Assumed, and varied:

    k  cost of one bad answer, in deflected tickets   {COST_RATIOS}
    p  share of traffic that is unanswerable          {OOS_SHARES}
    f  cost of a refusal, in deflected tickets        {REFUSAL_COST}

  Value per query, against sending every ticket to a person (zero):

    (1 - p) * [ a * ((1 - b) - b*k) - (1 - a)*f ]  -  p * [ (1 - r)*k + r*f ]

  "Bad" means the judge found a claim the context does not support. The
  share of traffic that is unanswerable is NOT the golden set's 21%:
  that mix was designed, and is included only as the pessimistic end.
""")


# =====================================================================
# B. Where the system stops paying
# =====================================================================
print("=" * 78)
print("B. WHERE THE SYSTEM STOPS PAYING")
print("=" * 78)

k_point = breakeven_cost_ratio(a, b, REFUSAL_COST)
k_worst = breakeven_cost_ratio(a, b_hi, REFUSAL_COST)
k_best = breakeven_cost_ratio(a, b_lo, REFUSAL_COST)

print(f"""
  Ignore unanswerable questions entirely. Answering ANSWERABLE ones
  stops paying once a bad answer costs more than this many deflected
  tickets:

    at the measured bad-answer rate   {b:>6.1%}     k = {k_point:.1f}
    at the top of its interval        {b_hi:>6.1%}     k = {k_worst:.1f}
    at the bottom of its interval     {b_lo:>6.1%}     k = {k_best:.1f}

  Past that point no refusal rate helps: the system loses on the
  questions it is supposed to be good at. So the first thing the
  business has to say is whether one wrong billing answer costs more or
  less than about {k_worst:.0f} tickets' worth of saved effort — and {n_bad} bad
  answers in {n_ans} cannot place that threshold more precisely than
  somewhere between {k_worst:.0f} and {k_best:.0f}.
""")


# =====================================================================
# C. The refusal rate the business needs
# =====================================================================
print("=" * 78)
print("C. THE REFUSAL RATE THE BUSINESS NEEDS")
print("=" * 78)


def _cell(k: float, p: float, bad: float) -> str:
    need = required_refusal_rate(a, bad, p, k, REFUSAL_COST)
    if need is None:
        return "never"
    if need == 0:
        return "any"
    mark = "" if need <= r_lo else " *"
    return f"{need:.1%}{mark}"


for label, bad in (("measured bad-answer rate", b), ("top of its interval", b_hi)):
    print(f"\n  Refusal rate needed to break even, at the {label} ({bad:.1%}):\n")
    print(f"    {'bad answer costs':<20}" + "".join(f"{f'p = {p:.0%}':>12}" for p in OOS_SHARES))
    print("    " + "-" * (20 + 12 * len(OOS_SHARES)))
    for k in COST_RATIOS:
        print(f"    {f'k = {k}':<20}" + "".join(f"{_cell(k, p, bad):>12}" for p in OOS_SHARES))

need_base = required_refusal_rate(a, b, BASE_P, BASE_K, REFUSAL_COST)
need_base_worst = required_refusal_rate(a, b_hi, BASE_P, BASE_K, REFUSAL_COST)
print(f"""
    any    pays at every refusal rate
    never  does not pay even with perfect refusal
    *      higher than the {r_lo:.1%} that {n_ref} of {n_oos} has established

  Read down a column. In the base case (k = {BASE_K}, p = {BASE_P:.0%}) the system
  needs to refuse {need_base:.1%} of unanswerable questions, and {n_ref} of {n_oos} already
  shows at least {r_lo:.1%}. The sample the memo called too small is, for that
  case, enough. Where the table says "never", it is not the refusal
  rate that fails.
""")


# =====================================================================
# D. Which uncertainty moves the decision
# =====================================================================
print("=" * 78)
print("D. WHICH UNCERTAINTY MOVES THE DECISION")
print("=" * 78)

ev_point = expected_value(a, b, r, BASE_P, BASE_K, REFUSAL_COST)
swing_r = (expected_value(a, b, r_hi, BASE_P, BASE_K, REFUSAL_COST)
           - expected_value(a, b, r_lo, BASE_P, BASE_K, REFUSAL_COST))
swing_b = (expected_value(a, b_lo, r, BASE_P, BASE_K, REFUSAL_COST)
           - expected_value(a, b_hi, r, BASE_P, BASE_K, REFUSAL_COST))
ev_worst = expected_value(a, b_hi, r_lo, BASE_P, BASE_K, REFUSAL_COST)
ev_best = expected_value(a, b_lo, r_hi, BASE_P, BASE_K, REFUSAL_COST)

print(f"""
  Two rates are uncertain: the refusal rate ({n_ref} of {n_oos}) and the
  bad-answer rate ({n_bad} of {n_ans}). Moving each across its own 95% interval,
  holding the other at its measured value, in the base case:

    value per query at the measured rates              {ev_point:+.3f}
    swing from the refusal rate   [{r_lo:.1%} to {r_hi:.1%}]     {swing_r:.3f}
    swing from the bad-answer rate [{b_lo:.1%} to {b_hi:.1%}]     {swing_b:.3f}

>>> The bad-answer rate moves the result {swing_b / swing_r:.1f}x as much as the refusal
    rate does.

  The memo's headline caveat was the refusal sample. On this model that
  is the smaller of the two uncertainties. The larger one is how often
  an answer to an ANSWERABLE question contains an unsupported claim,
  and it is larger for a plain reason: at p = {BASE_P:.0%}, nine queries in ten
  are answerable, so an error rate on that side is multiplied by nine
  times the traffic.

  Both at their unfavourable ends at once, the base case is worth
  {ev_worst:+.3f} per query; both favourable, {ev_best:+.3f}.""")

print(f"""
  Does the decision survive the uncertainty? Value per query with both
  rates at their unfavourable ends, by assumption:

    {'bad answer costs':<20}""" + "".join(f"{f'p = {p:.0%}':>12}" for p in OOS_SHARES))
print("    " + "-" * (20 + 12 * len(OOS_SHARES)))
verdicts = {}
for k in COST_RATIOS:
    row = []
    for p in OOS_SHARES:
        worst = expected_value(a, b_hi, r_lo, p, k, REFUSAL_COST)
        point = expected_value(a, b, r, p, k, REFUSAL_COST)
        verdicts[(k, p)] = ("holds" if worst > 0 else
                            "open" if point > 0 else "fails")
        row.append(f"{worst:+.2f} {verdicts[(k, p)]}")
    print(f"    {f'k = {k}':<20}" + "".join(f"{c:>12}" for c in row))

n_holds = sum(v == "holds" for v in verdicts.values())
n_open = sum(v == "open" for v in verdicts.values())
n_fails = sum(v == "fails" for v in verdicts.values())
print(f"""
    holds  positive even at the unfavourable ends: deploy on this evidence
    open   positive at the measured rates, negative at the unfavourable
           ends: the evidence cannot decide, and more data would
    fails  negative at the measured rates: more data will not help

  Of {len(verdicts)} combinations, {n_holds} hold, {n_open} are open and {n_fails} fail. The pilot is
  only worth running for the open ones — and what it must tighten
  there is mostly the bad-answer rate.
""")


# =====================================================================
# E. What the pilot has to measure
# =====================================================================
print("=" * 78)
print("E. WHAT THE PILOT HAS TO MEASURE")
print("=" * 78)

# The bad-answer rate must be shown to sit below the rate at which the
# base case breaks even with refusal at its unfavourable end.
lo_b, hi_b = 0.0, 1.0
for _ in range(60):                       # bisect for the break-even b
    mid = (lo_b + hi_b) / 2
    if expected_value(a, mid, r_lo, BASE_P, BASE_K, REFUSAL_COST) > 0:
        lo_b = mid
    else:
        hi_b = mid
b_max = lo_b
n_drafts = n_for_upper_bound(b, b_max)
n_clean = clean_sweep_n(need_base) if need_base and need_base > 0 else 0
n_resolution = two_proportion_n(BASELINE_RESOLUTION, BASELINE_RESOLUTION + MIN_EFFECT)
deff = design_effect(TICKETS_PER_AGENT, AGENT_ICC)
n_clustered = int(round(n_resolution * deff))
total_tickets = 2 * n_clustered
days = total_tickets / DAILY_TICKETS
# Early stop: the bad-draft count in the first 100 at which even the
# LOWER end of the interval is above the ceiling, so more data cannot
# bring the rate back under it.
STOP_WINDOW = 100
stop_count = next(c for c in range(STOP_WINDOW + 1)
                  if wilson_interval(c, STOP_WINDOW)[0] > b_max)

print(f"""
  1. THE BAD-ANSWER RATE. In the base case the system breaks even while
     no more than {b_max:.1%} of its answers are unsupported. The measured
     rate is {b:.1%} with an upper bound of {b_hi:.1%}. To bring the upper bound
     under {b_max:.1%}, if the true rate is what was measured, the pilot needs
     agents to review about {n_drafts:,} drafted answers.

     And a rule for stopping early: {stop_count} or more bad drafts in the first
     {STOP_WINDOW} puts even the lower end of the interval above {b_max:.1%}. At that
     point more data cannot rescue the base case, and the pilot ends.

  2. THE REFUSAL RATE. The base case needs {need_base:.1%}. A clean run of
     {n_clean} unanswerable tickets establishes that; {n_oos} already have.
     Nothing further is needed here unless a bad answer costs far more
     than assumed.

  3. WHETHER IT HELPS AT ALL. Neither rate says the assistant improves
     support. That needs a controlled comparison: tickets handled with
     a drafted answer against tickets handled without. To detect a lift
     in first-contact resolution from {BASELINE_RESOLUTION:.0%} to {BASELINE_RESOLUTION + MIN_EFFECT:.0%} (two-sided, 5%
     significance, 80% power) takes {n_resolution:,} tickets per arm if tickets are
     randomised individually.

     Randomising by AGENT is cleaner, because an agent who has seen
     drafts cannot unsee them. It costs precision: at {TICKETS_PER_AGENT} tickets per
     agent and an intra-agent correlation of {AGENT_ICC}, the design effect is
     {deff:.2f}, so {n_clustered:,} tickets per arm — {total_tickets:,} in total, about
     {days:.0f} days at {DAILY_TICKETS:,} tickets a day.

  The resolution baseline, the smallest lift worth detecting, the
  correlation and the ticket volume are placeholders. They are inputs
  to replace with the support team's figures, and the notebook
  recomputes everything when they change.
""")


# =====================================================================
# F. Verification
# =====================================================================
print("=" * 78)
print("F. VERIFICATION — THE PILOT BRIEF SAYS WHAT THIS NOTEBOOK COMPUTES")
print("=" * 78)

if not BRIEF.exists():
    print(f"\n  {BRIEF} not found — nothing to verify.\n")
    raise SystemExit(0)

claims = [
    ("answer rate", f"{n_ans} of {n_in}"),
    ("bad answers", f"{n_bad} of {n_ans}"),
    ("bad-answer interval", f"[{b_lo:.1%}, {b_hi:.1%}]"),
    ("refusals", f"{n_ref} of {n_oos}"),
    ("refusal lower bound", f"{r_lo:.1%}"),
    ("break-even k, measured", f"k = {k_point:.1f}"),
    ("break-even k, unfavourable", f"k = {k_worst:.1f}"),
    ("refusal needed, base case", f"{need_base:.1%}"),
    ("swing from refusal rate", f"{swing_r:.3f}"),
    ("swing from bad-answer rate", f"{swing_b:.3f}"),
    ("ratio of swings", f"{swing_b / swing_r:.1f}x"),
    ("value at measured rates", f"{ev_point:+.3f}"),
    ("value at unfavourable ends", f"{ev_worst:+.3f}"),
    ("combinations", f"{n_holds} hold, {n_open} are open and {n_fails} fail"),
    ("bad-answer ceiling", f"{b_max:.1%}"),
    ("drafts to review", f"{n_drafts:,} drafted answers"),
    ("early stop", f"{stop_count} or more bad drafts in the first {STOP_WINDOW}"),
    ("tickets per arm, individual", f"{n_resolution:,} tickets per arm"),
    ("design effect", f"{deff:.2f}"),
    ("tickets per arm, by agent", f"{n_clustered:,} tickets per arm"),
    ("total tickets", f"{total_tickets:,} in total"),
    ("duration", f"{days:.0f} days"),
]

print(f"\n    {'claim':<34}{'the brief must contain':<46}found")
print("    " + "-" * 86)
failed = report(BRIEF, claims)
print()
if failed:
    print(f"  FAILED — {failed} figure(s) in {BRIEF} no longer match this notebook.\n"
          f"  Update the brief, or the assumptions at the top of this file.\n")
    raise SystemExit(1)
print(f"  All {len(claims)} figures in {BRIEF} match, recomputed live.\n")

# The decision memo quotes four of these. Same check, same failure.
MEMO = Path("reports/decision_memo.md")
memo_claims = [
    ("bad-answer interval", f"[{b_lo:.1%}, {b_hi:.1%}]"),
    ("ratio of swings", f"{swing_b / swing_r:.1f}x"),
    ("refusal needed, base case", f"{need_base:.1%}"),
    ("drafts to review", f"{n_drafts:,} drafted answers"),
]
if MEMO.exists():
    print(f"    {'claim':<34}{'the memo must contain':<46}found")
    print("    " + "-" * 86)
    failed = report(MEMO, memo_claims)
    print()
    if failed:
        print(f"  FAILED — {failed} figure(s) in {MEMO} no longer match this notebook.\n")
        raise SystemExit(1)
    print(f"  All {len(memo_claims)} figures the memo takes from this phase match.\n")
