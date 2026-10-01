## Failure analysis: extractive baseline

120 questions, 57 failures (48%).

| Cause | Owner | n |
|---|---|---:|
| `ok` | none | 49 |
| `over_refusal` | generation | 34 |
| `correct_refusal` | none | 14 |
| `answered_oos` | generation | 11 |
| `retrieval_partial` | retrieval | 9 |
| `retrieval_miss` | retrieval | 3 |

| Category | answered_oos | correct_refusal | ok | over_refusal | retrieval_miss | retrieval_partial |
|---|---:|---:|---:|---:|---:|---:|
| ambiguous | 0 | 0 | 4 | 4 | 2 | 5 |
| multi_hop | 0 | 0 | 7 | 9 | 0 | 4 |
| out_of_scope | 11 | 14 | 0 | 0 | 0 | 0 |
| single_hop | 0 | 0 | 38 | 21 | 1 | 0 |

| Corpus flaw | failed | passed |
|---|---:|---:|
| `contradiction-refund-window` | 3 | 4 |
| `near-duplicate-cancellation` | 2 | 4 |
| `near-duplicate-device-setup` | 3 | 4 |
