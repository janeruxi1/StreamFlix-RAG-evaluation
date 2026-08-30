# Data Card — StreamFlix Support Corpus & Golden Set

Two synthetic artifacts underpin this project: a help-centre **corpus**
(the knowledge base the RAG system retrieves from) and a **golden set**
(the labelled questions every metric is measured against).

Both are generated from source in `src/corpus/`. Regenerate with:

```bash
python -m src.corpus.build --stats
```

---

## 1. Corpus — `data/corpus/*.md`

45 help-centre articles, one markdown file per article.

### Schema

Each file carries YAML front matter followed by a markdown body:

```markdown
---
article_id: bill-002
title: Refund policy
category: billing
last_updated: 2025-08-02
---

StreamFlix offers refunds on subscription charges in limited...
```

| Field | Type | Description |
|---|---|---|
| `article_id` | string | Stable identifier. Used as retrieval ground truth — **never renumber**. Prefix encodes category (`bill-`, `strm-`, `acct-`, `dev-`, `cont-`, `trial-`). |
| `title` | string | Human-readable heading. Prepended to the body when embedding, since titles carry topical signal the body sometimes assumes. |
| `category` | string | One of `billing`, `streaming`, `account`, `devices`, `content`, `trial`. |
| `last_updated` | ISO date | Deliberately stale for one article — see Known Flaws. |
| body | markdown | Article content, 107–162 words. |

### Composition

| Category | Articles |
|---|---:|
| billing | 10 |
| devices | 9 |
| account | 8 |
| streaming | 8 |
| content | 6 |
| trial | 4 |
| **Total** | **45** |

Total ~5,870 words. Every article fits inside a 512-token embedding
window, so chunking is an experimental choice in Phase 2 rather than a
forced workaround.

---

## 2. Golden set — `data/golden_set/golden_questions.json`

120 questions with ground-truth source labels.

### Schema

```json
{
  "question_id": "mh-011",
  "question": "Charged after I cancelled — what do I do?",
  "category": "multi_hop",
  "gt_article_ids": ["bill-003", "bill-002"],
  "reference_answer": "A charge after a confirmed cancellation is...",
  "notes": "CONTRADICTION PROBE — a good answer surfaces the conflict."
}
```

| Field | Type | Description |
|---|---|---|
| `question_id` | string | Stable identifier, prefixed by category (`sh-`, `mh-`, `am-`, `oos-`). |
| `question` | string | Phrased as a subscriber would ask, deliberately not echoing article titles. |
| `category` | string | `single_hop`, `multi_hop`, `ambiguous`, or `out_of_scope`. |
| `gt_article_ids` | list[string] | Ground-truth source articles. **Empty for `out_of_scope`** — that invariant is enforced by tests. |
| `reference_answer` | string | What a correct answer contains. The judge scores against this, not against retrieved context. |
| `notes` | string | Why the question earns its place; flags contradiction and gap probes. |

### Composition

| Category | Count | Purpose |
|---|---:|---|
| `single_hop` | 60 | Answer in one article — baseline retrieval + faithfulness |
| `multi_hop` | 20 | Two or more articles — context assembly, not just top-1 |
| `ambiguous` | 15 | Several plausible articles — precision/recall trade-off |
| `out_of_scope` | 25 | No article covers it — **must refuse**; the hallucination guardrail |
| **Total** | **120** | |

33 of the 95 in-scope questions require more than one source article.

---

## 3. Known flaws — deliberate, documented, tested

The corpus contains planted imperfections. A clean, internally
consistent, fully-covering corpus produces uniformly high scores and
teaches nothing. Real help centres contradict themselves, go stale, and
have gaps.

| Flaw | Articles | What it tests |
|---|---|---|
| **Contradiction** | `bill-002` (30-day refund) vs `bill-003` (14-day refund) | Does the system surface the conflict, or confidently assert one figure? Probed by `mh-011`, `am-001`. |
| **Outdated** | `bill-009` documents the discontinued "Basic Plus" tier; `last_updated: 2024-06-03` | Does retrieval present stale content as current? |
| **Near-duplicates** | `dev-003` / `dev-004` / `dev-005` (Roku / Apple TV / Fire TV setup) | Retrieval precision — does a Roku question return Roku, or a plausible sibling? Confirmed as the corpus's most lexically similar cluster. |

`tests/test_corpus.py` asserts these flaws still exist. If someone
"fixes" the corpus, those tests fail — otherwise the contradiction probe
and refusal metric would silently become tests of nothing.

### Coverage gaps

Eight topics have **no article at all**, generating the 25 out-of-scope
questions:

gift subscriptions · business and education accounts · accessibility
features · live sports and events · password-sharing policy · GDPR data
export · affiliate programmes · physical merchandise

Four out-of-scope questions have a **topically adjacent** article that
does not actually answer them — the hardest refusals, where over-answering
is most likely:

| Question | Adjacent but silent article |
|---|---|
| `oos-007` audio description | `strm-005` subtitles |
| `oos-010` password sharing | `strm-007` simultaneous streams |
| `oos-024` season release date | `cont-003` new releases |
| `oos-025` goodwill credit | `bill-002` refund policy |

---

## 4. Benchmark difficulty

Measured before any embedding model was chosen, so every later result
has a floor to be compared against.

**BM25 lexical baseline, recall@5** (out-of-scope excluded — retrieval
metrics are undefined without ground truth):

| Category | recall@5 | Headroom |
|---|---:|---:|
| single_hop | 93.3% | ~7 pts |
| multi_hop | 67.9% | ~32 pts |
| ambiguous | 48.9% | ~51 pts |
| **All in-scope** | **81.0%** | |

Consequences for Phase 3: BM25 is a first-class arm in the retrieval
bake-off, and results are reported per category — the aggregate is
dominated by single-hop questions where the baseline is already strong.

**Question phrasing check:** 49% of in-scope questions share no content
word with their ground-truth article title (mean overlap 15.7%). The
benchmark tests retrieval, not string matching.

---

## 5. Limitations

1. **Single annotator.** Corpus and questions were written by the same
   author, so ground-truth labels are self-assigned. No inter-annotator
   agreement statistic exists because there is one annotator. Structural
   errors are caught mechanically by `validate_corpus()`; a question
   whose "correct" article is arguably wrong would not be.

2. **Ambiguous labels are judgement calls.** For the 15 ambiguous
   questions, which articles count as relevant is genuinely debatable.
   Context-precision on that slice is directional, not exact.

3. **Synthetic corpus is cleaner than reality.** One author, one sitting,
   consistent phrasing. Real help centres are written by many people over
   years. This likely makes retrieval easier here than in production.

4. **Question phrasing is clean.** No typos, no fragments. Real support
   queries look like "cant login on tv". Phase 6 revisits this as a
   robustness check rather than pretending the current set covers it.

---

## 6. Regeneration

```bash
python -m src.corpus.build            # write files
python -m src.corpus.build --stats    # write, then print a summary
pytest tests/test_corpus.py -q        # 20+ integrity tests
```

Source of truth is `src/corpus/seed_articles.py` and
`src/corpus/golden_set.py`. Files under `data/` are build output —
edit the source, not the output.
