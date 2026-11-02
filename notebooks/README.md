# Notebooks: reading guide

Read them in order. Each notebook opens with **At a glance**: the question, the answer, the
figure it produces and a link to its full printed output.

**Why there are no outputs in the notebooks.** Several notebooks call an LLM API. A notebook
output can capture a credential and keep it in git history for good, so the notebooks are
published without outputs. The figures are in [`reports/figures/`](../reports/figures/) and each
notebook's full printed run is in [`reports/llm_run/`](../reports/llm_run/).

| # | Notebook | The question in plain words |
|---|---|---|
| 01 | [Corpus construction](01_corpus_construction.ipynb) | What does the assistant answer from, and how will answers be judged? |
| 02 | [Chunking and embeddings](02_chunking_embedding.ipynb) | How should the articles be split and searched? |
| 03 | [Retrieval bake-off](03_retrieval_bakeoff.ipynb) | Which retrieval setup should ship? |
| 04 | [Generation](04_generation.ipynb) | How to make the model answer from the help centre and decline otherwise? |
| 05 | [Evaluation](05_evaluation.ipynb) | How good are the answers, by a judge checked first? |
| 06 | [Judge analysis](06_judge_analysis.ipynb) | Is the judge biased? |
| 07 | [Decision memo](07_decision_memo.ipynb) | What to deploy, and does the memo match the measurements? |
| 08 | [Full-corpus baseline](08_full_corpus_baseline.ipynb) | Is retrieval needed at this size? |
| 09 | [Decision model](09_decision_model.ipynb) | What are the error rates worth? |
| 10 | [Measured answers](10_measured_answers.ipynb) | Does the demo replay the reported run? |
| 11 | [Ticket routing](11_ticket_routing.ipynb) | Which team should each question go to, and when automatically? |

**Running them.** Each `.ipynb` is generated from the `.py` file beside it; see the main
[README](../README.md) for setup.
