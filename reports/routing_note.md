# Routing support tickets: which router, and how much can skip a person?

*Notebook 11. Every figure below is recomputed from the stored
predictions by `notebooks/11_ticket_routing.py`, and CI fails if this
note and the notebook disagree.*

## The question

Before an assistant or an agent answers a ticket, someone decides which
team gets it. An operations lead wants to send as many tickets as
possible straight to the right queue while keeping misroutes among
those below 5%, and to know what each option costs to run.

## The data

Banking77 (PolyAI, CC BY 4.0): real customer questions to an online
bank, each labelled with one of 77 intents. 9,000 to train, 1,003 held
back to tune on, 3,080 to test (40 per intent). It is a different
domain from the StreamFlix help centre in notebooks 01–10, used because
routing needs thousands of labelled tickets and synthetic ones would
test the generator, not the router.

## Four routers

| Router | Test accuracy | Auto-routed at the 95% bar | Accuracy of those | Calibration (ECE) | Cost per 1,000 | Single ticket |
|---|---|---|---|---|---|---|
| TF-IDF + logistic regression | 91.4% | 89.5% | 95.5% | 0.097 | $0 | 2 ms |
| BGE-small, frozen, + logistic regression | 93.0% | 97.4% | 94.4% | 0.106 | $0 | 25 ms |
| BGE-small, fine-tuned (seed 0) | 93.8% | 97.0% | 95.3% | 0.016 | $0 | 34 ms |
| gpt-4o-mini, zero-shot | 66.6% on a 1,001-ticket subset | — | — | — | $0.081 | 684 ms |

The threshold for automatic routing is chosen on the tuning split and
applied unchanged to test. Times are a laptop CPU.

## What holds up

- **Fine-tuning beats TF-IDF, robustly.** +2.3 points for the reported
  seed, [+1.4, +3.2] paired on the same tickets, and significant for
  all three seeds (accuracy 92.9%–93.8% across seeds).
- **Its accuracy edge over frozen embeddings does not.** +0.8 points
  for seed 0, but significant for only 1 of 3 seeds. On accuracy alone,
  the frozen model gets nearly all of the benefit in two minutes of
  training instead of 108 minutes.
- **What fine-tuning does buy is calibration.** Its confidence means
  what it says (ECE 0.016, against 0.106 frozen). Every seed's
  validation-chosen threshold kept its promise on test, routing
  95.0%–95.3% correctly while sending 95.7%–97.0% of tickets without a
  person. The frozen router's threshold missed, at 94.4%: its
  probabilities run low, so the cut sits in a sparse band that does
  not transfer from tuning data to test.
- **The general LLM is the wrong tool here.** 66.6% on a balanced
  subset where the fine-tuned model scores 94.1%, with 14 of 1,001
  replies naming an intent that does not exist, no confidence to
  threshold on, and a cost on every ticket. Nine thousand labelled
  examples beat a much larger model that has seen none.

## Recommendation

Route with the fine-tuned model behind a 95% threshold: about 97% of
tickets go straight to a queue and the rest to a person. If the
training cost matters more than the threshold's reliability, the
frozen model is the fallback, but only after recalibrating it
(temperature scaling on the tuning split, not run here) and checking
again that the threshold holds.

The most common mistakes are shared by every model: card arrival
against delivery estimate, a failed top-up against a reverted one. No
model recovers a detail the customer did not write, so those pairs are
for the queue design, by merging them or asking one clarifying
question.

## Limits

- One public dataset in one domain; 77 intents is the hard version of
  routing, and a real help desk's label set and traffic mix will differ.
- One LLM, one prompt, zero-shot. Few-shot examples or a larger model
  would narrow the gap; whether they close it at an acceptable cost is
  untested.
- The 95% bar is an assumption. Section D of the notebook shows the
  whole coverage curve for other bars.
