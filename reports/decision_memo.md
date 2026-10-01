# Decision memo: StreamFlix support RAG

**Status: provisional.** The retrieval recommendation rests on measured results.
The generation recommendation cannot be finalised because the LLM prompt
variants and LLM judge have **not been run** (no API key was available).
Sections marked **[PENDING]** are filled in from `reports/results/llm_eval.json`
after `python scripts/run_llm_eval.py`.

## Recommendation

| Component | Decision | Confidence |
|---|---|---|
| Chunking | `markdown_section` | High on cost, medium on recall (see below) |
| Retrieval | BM25, depth 15 | Medium: dense is not distinguishable from BM25 here |
| Generation | **[PENDING]**: choose among `grounded_refusal`, `cited`, `strict` by the refusal/answer trade-off | Not measured |
| Judge | LLM judge, only if it passes validation (80% accuracy and 95% CI lower bound ≥ 70%) | Not measured |
| Ship to customers? | **No, not yet.** Refusal behaviour is unmeasured for an LLM | n/a |

## What the evidence supports

1. **Retrieval config.** `markdown_section` + BM25 at depth 15 gives recall
   0.882 at 571 tokens/query. The recall-optimal config (`whole_article` +
   LSA, depth 15) reaches 0.918 but costs 2,543 tokens/query: 4.5× more. The
   paired recall difference is +0.040 (95% CI [−0.003, +0.084]), so the
   saving is certain and the recall cost is bounded at 0.084. This is a cost
   decision, since no quality-only metric can pick depth (finding 3).
2. **Dense vs lexical.** Not statistically distinguishable (+0.009, 95% CI
   [−0.023, +0.042]). With consistent vocabulary, BM25 is the cheaper,
   simpler choice.
3. **Refusal is the unsolved problem.** The keyless baseline answers 11 of 25
   out-of-scope questions it should refuse, and over-refuses 34 in-scope
   questions it could answer. Raising its threshold trades one for the other
   (OOS refusal 33%→80% while in-scope answering falls 100%→37%).
4. **Ambiguous queries** have the weakest recall (0.56) and reach 22% of
   their precision ceiling. Retrieval tuning will not fix this; it needs
   query clarification or rewriting.
5. **The corpus itself causes failures.** Questions touching the planted
   refund contradiction fail 3 of 7 times for the baseline. A policy fix
   (reconcile bill-002/bill-003) is cheaper than any model change.

## Failure ownership (keyless baseline, 120 questions)

| Owner | Failures | Typical fix |
|---|---:|---|
| Generation (over-refusal 34, answered OOS 11) | 45 | Prompting/threshold; the LLM variants are designed to address this |
| Retrieval (partial recall 9, miss 3) | 12 | Rerank/deeper retrieval for multi-hop; query rewriting for ambiguous |
| Corpus | tagged, not exclusive | Reconcile contradiction, retire `bill-009` |

## What would change the recommendation

- An LLM variant that refuses ≥ 90% of OOS questions while answering ≥ 85%
  in-scope would make shipping with a human-escalation path reasonable.
- A judge that fails validation means faithfulness numbers must not be
  reported.
- A larger golden set (≥ 300 questions) is needed to rank the top retrieval
  configurations; today they cannot be separated.

## Risks and monitoring (if deployed)

- Log refusal rate and answer rate weekly; a drift in either is the first
  sign of corpus or traffic change.
- Sample 50 answers a month for human faithfulness review to re-check the
  judge.
- Re-run the golden set after every corpus update; add each production
  failure as a new golden question.
- Escalate to a human on refusal, on conflicting sources (contradiction
  articles), and on low-confidence answers.

## Cost

Retrieval and the extractive baseline are free (~7 ms/query). LLM cost per
query is **[PENDING]**: the prompt averages about 570 context tokens at the
recommended config, so cost scales with the generation model's price; the
run script prints a dry-run estimate (`--dry-run`).
