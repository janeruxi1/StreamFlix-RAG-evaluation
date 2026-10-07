# Pilot Brief — Cited Answers With an Agent in the Loop

**From:** Xi Ru, Data Science
**Companion to:** [`decision_memo.md`](decision_memo.md)

Every figure here is recomputed by `notebooks/09_decision_model.py`,
which CI runs and which fails the build if this brief and the code
disagree. Change an assumption at the top of that notebook and the
numbers below must be updated to match.

---

## The decision this pilot informs

Whether the `cited` answering arm can go to customers unattended. The
memo recommended a pilot first. This brief says what the pilot has to
measure, how much of it, and when to stop.

## What is already measured

| | Count | Rate | 95% interval |
|---|---:|---:|---:|
| Answers an answerable question | 76 of 95 | 80.0% | |
| An answer contains an unsupported claim | 3 of 76 | 3.9% | [1.4%, 11.0%] |
| Refuses an unanswerable question | 25 of 25 | 100.0% | [86.7%, 100.0%] |

## The cost model

A rate is not a decision until someone says what a wrong answer costs.
Value per query, measured against sending every ticket to a person:

```
(1 - p) * [ a * ((1 - b) - b*k) - (1 - a)*f ]  -  p * [ (1 - r)*k + r*f ]
```

`a`, `b` and `r` are the three measured rates above. The rest are
**assumptions**, and nothing below should be read as a forecast:

| | Meaning | Values tried |
|---|---|---|
| `k` | cost of one bad answer, in deflected tickets | 1, 5, 10, 20, 50 |
| `p` | share of traffic the help centre cannot answer | 5%, 10%, 21% |
| `f` | cost of a refusal, in deflected tickets | 0.1 |

The worked case uses `k` = 10 and `p` = 10%.

## What the model says

**1. There is a cost at which the system does not pay, whatever it
refuses.** At the measured bad-answer rate, answering answerable
questions stops paying at k = 23.7. At the top of that rate's interval
it stops at k = 7.9. So the first question for the business is whether
one wrong billing answer costs more or less than about eight tickets'
worth of saved effort.

**2. The refusal evidence is already enough for the worked case.** It
needs the system to refuse 61.7% of unanswerable questions, and 25 of 25
establishes at least 86.7%.

**3. The larger uncertainty is not the one the memo led with.** Moving
each rate across its own interval, the refusal rate swings the value per
query by 0.132 and the bad-answer rate swings it by 0.762: 5.8x as much.
Most traffic is answerable, so an error rate on that side is multiplied
by most of the traffic.

**4. Whether to deploy depends on the assumptions, and the evidence
settles some cases and not others.** The worked case is worth +0.379 per
query at the measured rates and -0.309 with both rates at their
unfavourable ends. Across the 15 combinations of `k` and `p`, 6 hold, 6
are open and 3 fail: six are positive even at the unfavourable ends, six
cannot be decided on this evidence, and three are negative at the
measured rates, where more data will not help.

The pilot is worth running for the open cases, and what it has to
tighten there is mostly the bad-answer rate.

## Design

**Arms.** Tickets handled with a cited draft and its sources shown to
the agent, against tickets handled as today. The agent decides what the
customer receives in both.

**Unit of randomisation: the agent.** An agent who has seen drafts
cannot unsee them, so randomising ticket by ticket contaminates the
control arm.

**Primary metric.** First-contact resolution.

**Measured on the treatment arm only.** For every draft, the agent
records whether it was sent as written, edited, or discarded, and
whether it contained a claim the cited article does not support. That
label is the pilot's measurement of the bad-answer rate, and it gives
the LLM judge the human rater it currently lacks.

**Guardrails.** Customer satisfaction and re-contact rate must not fall
in the treatment arm.

## How much data

**To settle the bad-answer rate.** In the worked case the system breaks
even while no more than 7.1% of answers are unsupported. Bringing the
upper bound under that, if the true rate is what was measured, takes
about 273 drafted answers reviewed by agents.

**To stop early.** 13 or more bad drafts in the first 100 puts even the
lower end of the interval above 7.1%. More data cannot rescue the worked
case from there, and the pilot ends.

**To detect an effect on resolution.** A lift from 70% to 73%,
two-sided at 5% significance with 80% power, takes 3,551 tickets per arm
if tickets are randomised individually. Randomising by agent costs
precision: at 50 tickets per agent and an intra-agent correlation of
0.05 the design effect is 3.45, so 12,251 tickets per arm — 24,502 in
total, about 25 days at 1,000 tickets a day.

The resolution baseline, the smallest lift worth detecting, the
correlation and the ticket volume are placeholders for the support
team's own figures.

## Limits of this brief

- **The three rates come from 120 synthetic questions** written by the
  author of the corpus. The pilot exists to replace them with real ones.
- **"Bad" is the LLM judge's verdict.** It found a claim the context does
  not support. Some such claims are harmless, and the judge misses
  omissions. Agent labels in the pilot are the correction.
- **One cost for every bad answer.** A wrong refund window and a wrong
  subtitle setting are priced the same here. They should not be, and a
  second version would weight by topic.
- **A refusal is assumed cheap.** If customers who are refused leave
  rather than wait for a person, `f` is too low and the model flatters
  a cautious system.
