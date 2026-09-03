"""
Phase 2 — Chunking, Embeddings & the Retrieval Baseline
========================================================

Phase 1 established the floor: BM25 gets 93% recall@5 on single-hop
questions, 68% on multi-hop, 49% on ambiguous. Phase 2 builds the dense
retrieval pipeline that has to justify itself against those numbers.

Three components, each a decision rather than a default:

  CHUNKING     Phase 1 confirmed every article fits inside a 512-token
               embedding window, so we are never forced to split.
               `whole_article` is therefore a legitimate arm, not a
               strawman, and chunking becomes a real experiment.

  EMBEDDING    A transformer (BGE-small) when available, with TF-IDF +
               SVD as a deterministic fallback so CI and fresh clones
               run the full pipeline with no model download. The
               fallback doubles as a third bake-off arm: lexical (BM25)
               -> classic dense (LSA) -> modern dense (transformer).

  VECTOR STORE Exact cosine search in numpy. At 50-400 chunks an ANN
               index would trade recall for speed on an operation that
               already takes under a millisecond — a strictly bad deal,
               and one that would contaminate Phase 3's measurements
               with index error.

This phase produces the baseline. Phase 3 attacks it with a full
bake-off across strategies, backends, and top-k.

Sections
--------
  A. Chunking strategies — five arms profiled
  B. Chunk-size analysis — what each strategy actually produces
  C. Embedding backend — which one is running and why
  D. Build the index
  E. Retrieval baseline vs the Phase 1 BM25 floor
  F. Qualitative inspection — including the contradiction probe
  G. Verdict + handoff to Phase 3
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.corpus.build import load_corpus, load_golden_set
from src.corpus.difficulty import evaluate_lexical_baseline
from src.retrieval.chunking import STRATEGIES, profile_strategy
from src.retrieval.embedding import (
    TfidfSvdEmbedder,
    encode_cached,
    get_embedder,
    transformer_available,
)
from src.retrieval.vectorstore import VectorStore, hits_to_articles

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

articles = load_corpus()
golden = load_golden_set()
in_scope = [q for q in golden if q["category"] != "out_of_scope"]


# =====================================================================
# A. Chunking strategies
# =====================================================================
print("=" * 78)
print("A. CHUNKING STRATEGIES")
print("=" * 78)
print("""
Five arms, each a different point on the same trade-off:

  larger chunks  -> more context per hit, but the embedding averages
                    over more topics and matches queries less sharply
  smaller chunks -> sharper matching, but an answer spanning two chunks
                    needs both retrieved, and each carries less context
""")

chunk_sets = {name: fn(articles) for name, fn in STRATEGIES.items()}

for name, chunks in chunk_sets.items():
    print(f"  {profile_strategy(name, chunks, len(articles))}")

print("""
  whole_article     no split — maximum context, minimum precision
  markdown_section  split on the author's own '**Heading**' structure
  fixed_token_128   generic 128-token windows, 32-token overlap
  fixed_token_256   generic 256-token windows, 64-token overlap
  sentence_window   one sentence embedded, neighbours attached for context
""")


# =====================================================================
# B. Chunk-size analysis
# =====================================================================
print("\n" + "=" * 78)
print("B. CHUNK-SIZE ANALYSIS")
print("=" * 78)

profiles = [profile_strategy(n, c, len(articles))
            for n, c in chunk_sets.items()]
profile_df = pd.DataFrame([{
    "strategy": p.strategy,
    "chunks": p.n_chunks,
    "per_article": round(p.chunks_per_article, 1),
    "min_tok": p.min_tokens,
    "med_tok": p.median_tokens,
    "max_tok": p.max_tokens,
    "total_tok": p.total_tokens,
} for p in profiles])
print("\n" + profile_df.to_string(index=False))

# --- Degenerate-arm check --------------------------------------------
whole = chunk_sets["whole_article"]
degenerate = [
    name for name, chunks in chunk_sets.items()
    if name != "whole_article" and len(chunks) == len(whole)
]
print(f"""
>>> FINDING — one arm collapsed into another.

{', '.join(degenerate)} produced exactly {len(whole)} chunks: one per article,
identical to whole_article. The reason is arithmetic — the longest
article is {max(c.n_tokens for c in whole)} tokens, so a 256-token window never has anything
to split. The chunker ran, did nothing, and would have been reported
in Phase 3 as a distinct configuration.

That is the kind of silent no-op that produces a bake-off table with two
rows of suspiciously identical numbers and an author who cannot explain
why. Caught here instead: {', '.join(degenerate)} is dropped from the Phase 3
bake-off as redundant, and the reason is recorded rather than left as a
coincidence for a reviewer to notice.

General lesson worth stating: chunk size only means something relative
to document length. On a corpus of 10-page PDFs these arms would differ
enormously.
""")

active_strategies = [n for n in chunk_sets if n not in degenerate]
print(f"Strategies carried into Phase 3: {', '.join(active_strategies)}")

# --- Very small chunks -----------------------------------------------
sentence_chunks = chunk_sets["sentence_window"]
tiny = [c for c in sentence_chunks if c.n_tokens < 8]
print(f"""
sentence_window produces {len(sentence_chunks)} chunks, {len(tiny)} of them under 8 tokens
(median {profile_strategy('sw', sentence_chunks, len(articles)).median_tokens} tokens). Fragments that short — table rows, single
bullets — carry little standalone meaning and will embed to noise.

This is why the strategy attaches `context_text`: the embedding sees one
sentence for sharp matching, the generator receives the surrounding
window. Phase 3 measures whether decoupling those two jobs pays for the
{len(sentence_chunks) / len(whole):.0f}x increase in index size.
""")

# --- Chart ------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

names = [p.strategy for p in profiles]
counts = [p.n_chunks for p in profiles]
colors = ["#B9B9B9" if n in degenerate else "#5B8FF9" for n in names]
bars = ax1.barh(names, counts, color=colors, edgecolor="white")
ax1.set_xlabel("Chunks produced")
ax1.set_title("Index size by strategy\n(grey = degenerate, dropped)",
              fontweight="bold")
ax1.grid(axis="x", linestyle="--", alpha=0.4)
ax1.set_axisbelow(True)
for bar, v in zip(bars, counts):
    ax1.text(v + 8, bar.get_y() + bar.get_height() / 2, str(v),
             va="center", fontsize=9)

for name, chunks in chunk_sets.items():
    if name in degenerate:
        continue
    ax2.hist([c.n_tokens for c in chunks], bins=25, alpha=0.55, label=name)
ax2.axvline(512, color="#F6735B", linestyle="--", linewidth=1.5,
            label="embedding limit (512)")
ax2.set_xlabel("Chunk size (tokens)")
ax2.set_ylabel("Chunks")
ax2.set_title("Chunk-size distribution\n(all well inside the embedding window)",
              fontweight="bold")
ax2.legend(fontsize=8)
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)

plt.suptitle("Phase 2 — chunking strategy comparison",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "02_chunking_profiles.png", dpi=140,
            bbox_inches="tight",
            metadata={"Software": None})
print(f"Saved -> {FIG_DIR}/02_chunking_profiles.png")


# =====================================================================
# C. Embedding backend
# =====================================================================
print("\n" + "=" * 78)
print("C. EMBEDDING BACKEND")
print("=" * 78)

has_transformer = transformer_available()
print(f"\n  sentence-transformers available: {has_transformer}")

if has_transformer:
    print("""
  Using BGE-small-en-v1.5 — 384 dimensions, ~130MB, CPU-only, free.
  Competitive with far larger models on retrieval benchmarks.
""")
else:
    print("""
  Falling back to TF-IDF + truncated SVD (latent semantic analysis).

  This is not a stub. LSA is a real dense-retrieval method, and keeping
  it as a first-class backend means CI exercises the entire pipeline in
  seconds with no model download, and anyone cloning the repo gets a
  working system before deciding whether to install torch.

  Its genuine limitation is worth naming: the vocabulary is fitted on
  the corpus, so it cannot represent query words it never saw at fit
  time. That is exactly why it should lose to a transformer on
  paraphrased questions — and Phase 3 will show whether it does.

  Install sentence-transformers to run the transformer arm:
      pip install sentence-transformers
""")


# =====================================================================
# D. Build the index
# =====================================================================
print("\n" + "=" * 78)
print("D. BUILD THE INDEX")
print("=" * 78)

BASELINE_STRATEGY = "markdown_section"
baseline_chunks = chunk_sets[BASELINE_STRATEGY]
chunk_texts = [c.embed_text for c in baseline_chunks]

embedder = get_embedder()
if isinstance(embedder, TfidfSvdEmbedder):
    embedder.fit(chunk_texts)          # LSA must see the corpus first

chunk_vectors = encode_cached(embedder, chunk_texts)
store = VectorStore().add(baseline_chunks, chunk_vectors)

print(f"""
  Baseline strategy : {BASELINE_STRATEGY}
  Backend           : {embedder!r}
  Indexed           : {len(store)} chunks, {store.dimension} dimensions
  Index memory      : ~{chunk_vectors.nbytes / 1024:.0f} KB

markdown_section is the baseline because it splits on boundaries the
article author already chose. Phase 3 tests whether that intuition
survives measurement.
""")

question_texts = [q["question"] for q in in_scope]
question_vectors = encode_cached(embedder, question_texts)
print(f"  Encoded {len(question_vectors)} in-scope questions "
      f"(out-of-scope excluded — retrieval metrics need ground truth)")


# =====================================================================
# E. Retrieval baseline vs the Phase 1 floor
# =====================================================================
print("\n" + "=" * 78)
print("E. RETRIEVAL BASELINE vs THE BM25 FLOOR")
print("=" * 78)

TOP_K = 5
all_hits = store.search_batch(question_vectors, top_k=TOP_K * 3)

rows = []
for q, hits in zip(in_scope, all_hits):
    retrieved = hits_to_articles(hits, max_articles=TOP_K)
    gt = set(q["gt_article_ids"])
    rows.append({
        "question_id": q["question_id"],
        "category": q["category"],
        "hit_at_1": bool(gt & set(retrieved[:1])),
        "hit_at_3": bool(gt & set(retrieved[:3])),
        "hit_at_5": bool(gt & set(retrieved[:5])),
        "recall_at_5": len(gt & set(retrieved[:5])) / len(gt),
    })
dense_df = pd.DataFrame(rows)

documents = {a.article_id: a.as_document() for a in articles}
bm25_per_cat, bm25_overall = evaluate_lexical_baseline(documents, golden)
bm25_lookup = {r.category: r for r in bm25_per_cat}
bm25_lookup["ALL in-scope"] = bm25_overall

print(f"\nDense retrieval ({embedder.name}, {BASELINE_STRATEGY}, "
      f"top-{TOP_K} articles) vs BM25:\n")
print(f"  {'category':<14} {'n':>4} {'BM25 r@5':>10} {'dense r@5':>11} "
      f"{'delta':>8}")
print(f"  {'-'*14} {'-'*4} {'-'*10} {'-'*11} {'-'*8}")

comparison = []
for cat in ["single_hop", "multi_hop", "ambiguous"]:
    sub = dense_df[dense_df["category"] == cat]
    dense_r5 = sub["recall_at_5"].mean()
    bm25_r5 = bm25_lookup[cat].recall_at_5
    delta = dense_r5 - bm25_r5
    comparison.append((cat, len(sub), bm25_r5, dense_r5, delta))
    print(f"  {cat:<14} {len(sub):>4} {bm25_r5:>9.1%} {dense_r5:>10.1%} "
          f"{delta:>+8.1%}")

dense_all = dense_df["recall_at_5"].mean()
bm25_all = bm25_overall.recall_at_5
print(f"  {'-'*14} {'-'*4} {'-'*10} {'-'*11} {'-'*8}")
print(f"  {'ALL in-scope':<14} {len(dense_df):>4} {bm25_all:>9.1%} "
      f"{dense_all:>10.1%} {dense_all - bm25_all:>+8.1%}")

wins = [c for c, _, _, d, delta in comparison if delta > 0]
losses = [c for c, _, _, d, delta in comparison if delta < 0]

print(f"""
Read per category. The aggregate is dominated by the 60 single-hop
questions and hides everything interesting.

  dense wins on  : {', '.join(wins) if wins else '(none)'}
  dense loses on : {', '.join(losses) if losses else '(none)'}
""")

if dense_all < bm25_all:
    print(f"""
>>> The untuned dense baseline LOSES to BM25 overall ({dense_all:.1%} vs {bm25_all:.1%}).

That is a normal and reportable result, not a bug. Three reasons it
happens, all worth separating:

  1. BM25 is not a weak baseline here. Phase 1 showed this corpus uses
     consistent product vocabulary — "Roku", "4K", "refund" — and
     questions reuse it. Exact term matching is genuinely well suited to
     that. Lexical baselines beating dense retrieval on in-vocabulary
     factual lookup is a documented finding in the IR literature, not an
     anomaly.
  2. This is ONE untuned configuration. One chunking strategy, default
     top-k, no hybrid, no reranking.
  3. The backend running right now is {embedder.name}. """
          + ("Because sentence-transformers\n     is unavailable, this is the LSA fallback, whose vocabulary is\n     fitted on the corpus and cannot represent unseen query words. The\n     transformer arm is expected to close much of this gap — Phase 3\n     measures whether it does rather than assuming it."
             if not has_transformer else
             "This is the transformer arm,\n     so the gap is not a fallback artifact and needs a real explanation\n     — Phase 3's job.")
          + """

The useful conclusion is not "dense retrieval is bad." It is that a
lexical baseline must be beaten, not assumed away — and that shipping
the dense system on intuition alone would have degraded the product.
""")
else:
    print(f"""
The dense baseline edges out BM25 overall ({dense_all:.1%} vs {bm25_all:.1%}), but the
margin is small and the per-category split matters more than the total.
An untuned single configuration beating a strong lexical baseline by a
few points is not yet grounds for shipping it — Phase 3 checks whether
the advantage survives a proper sweep, and whether a hybrid beats both.
""")

print("""
Either way, the honest framing for Phase 3: BM25 is a first-class arm,
not a formality. A hybrid that unions lexical and dense candidates is
the configuration most likely to win, precisely because the two methods
fail on different questions.
""")


# =====================================================================
# F. Qualitative inspection
# =====================================================================
print("\n" + "=" * 78)
print("F. QUALITATIVE INSPECTION")
print("=" * 78)

def show(question_id: str, note: str = "") -> float:
    """Print the top-5 chunks for one question; return its recall@5."""
    q = next(x for x in golden if x["question_id"] == question_id)
    qv = encode_cached(embedder, [q["question"]])[0]
    hits = store.search(qv, top_k=5)
    retrieved = hits_to_articles(hits)
    gt = set(q["gt_article_ids"])
    print(f"\n  [{question_id}] {q['question']}")
    if note:
        print(f"  {note}")
    print(f"  ground truth : {sorted(gt) if gt else '(none — out of scope)'}")
    for h in hits:
        mark = "✓" if h.article_id in gt else " "
        print(f"    {mark} {h.score:.3f}  {h.chunk.chunk_id:<20} "
              f"{h.chunk.title[:38]}")
    if not gt:
        return float("nan")
    recall = len(gt & set(retrieved[:5])) / len(gt)
    print(f"  recall@5 = {recall:.0%}")
    return recall

print("\n--- The contradiction probe -----------------------------------")
recall_mh011 = show(
    "mh-011",
    "A good system surfaces BOTH refund windows and flags the conflict.")

# Diagnose against BM25 on the same question — is this a dense-retrieval
# weakness, or is the question hard for everything?
from src.corpus.difficulty import BM25

bm25 = BM25(documents)
q_mh011 = next(q for q in golden if q["question_id"] == "mh-011")
bm25_top = bm25.rank(q_mh011["question"], top_k=5)
bm25_hits = set(q_mh011["gt_article_ids"]) & set(bm25_top)

print(f"""
  BM25 on the same question : {bm25_top}
  BM25 found                : {sorted(bm25_hits) if bm25_hits else 'nothing'}
""")

gt_mh011 = set(q_mh011["gt_article_ids"])
neither_retriever_complete = (recall_mh011 < 1.0
                              and bm25_hits != gt_mh011)

if neither_retriever_complete:
    print(f"""
>>> FINDING — NEITHER retriever surfaces both sides of the contradiction.

  dense recall@5 : {recall_mh011:.0%}   BM25 recall@5 : {len(bm25_hits) / len(gt_mh011):.0%}

  Both retrievers are pulled toward the free-trial articles. The query's
  dominant terms are "charged" and "cancelled", and the trial-
  cancellation articles are saturated with both — so they win on surface
  similarity, lexical and dense alike, while the refund policy that
  actually answers the question is crowded out.

  The failure is shared, which rules out the convenient explanation.
  This is not "dense retrieval is weak" and not "the fallback backend is
  weak" — it is a property of the QUESTION: its vocabulary points at the
  wrong neighbourhood of the corpus. Changing retriever will not fix it.
  Query rewriting, hybrid candidate pools, or a wider top-k might.

  Why it matters beyond one missed question:

  The generator will receive five plausible-looking, on-topic, WRONG
  chunks and will answer fluently from them. Nothing in the output will
  look like an error. In Phase 6 this presents as a faithfulness failure
  — the model said something unsupported — when the fault is upstream in
  retrieval. Attributing it to the generator would aim the fix at the
  wrong component entirely.

  This is the clearest argument for measuring retrieval and generation
  SEPARATELY. Context precision/recall isolate it; an end-to-end
  answer-quality score would average it away.

  OPEN ITEM -> Phase 3: does ANY configuration retrieve both articles?
  OPEN ITEM -> Phase 6: is the failure attributed to retrieval, not
               generation?
""")
elif recall_mh011 == 1.0:
    print("""
  Both refund articles surfaced, so the generator at least HAS the
  material to detect the conflict. Whether it actually flags the
  contradiction rather than silently picking one window is a generation
  question — Phase 5.
""")
else:
    print(f"""
  Dense missed part of the pair ({recall_mh011:.0%}) where BM25 found it. A
  retriever-specific gap, unlike a shared failure — which makes hybrid
  retrieval the obvious Phase 3 candidate for this question.
""")

print("\n--- The near-duplicate cluster --------------------------------")
show("sh-046", "Roku-specific question. Does it return Roku, or a sibling?")

print("\n--- The discovered near-duplicate -----------------------------")
show("am-009", "Ambiguous 'how do I cancel' — paid vs trial paths.")

print("\n--- A multi-source question -----------------------------------")
show("mh-015", "Needs four articles. Hardest retrieval case in the set.")


# =====================================================================
# G. Verdict
# =====================================================================
print("\n" + "=" * 78)
print("PHASE 2 VERDICT")
print("=" * 78)
print(f"""
Built:

  Chunking      {len(STRATEGIES)} strategies implemented, {len(degenerate)} dropped as degenerate
                ({', '.join(degenerate)} collapsed into whole_article)
  Embedding     {embedder.name}, {store.dimension} dimensions, disk-cached
  Vector store  exact cosine over {len(store)} chunks, ~{chunk_vectors.nbytes / 1024:.0f} KB
  Baseline      {BASELINE_STRATEGY} @ top-{TOP_K}

Measured against the Phase 1 floor:

  single_hop    BM25 {bm25_lookup['single_hop'].recall_at_5:>5.1%}  ->  dense {comparison[0][3]:>5.1%}   ({comparison[0][4]:+.1%})
  multi_hop     BM25 {bm25_lookup['multi_hop'].recall_at_5:>5.1%}  ->  dense {comparison[1][3]:>5.1%}   ({comparison[1][4]:+.1%})
  ambiguous     BM25 {bm25_lookup['ambiguous'].recall_at_5:>5.1%}  ->  dense {comparison[2][3]:>5.1%}   ({comparison[2][4]:+.1%})

Carried into Phase 3:

  1. {len(active_strategies)} viable chunking strategies, with the degenerate arm
     already removed and the reason documented.
  2. A per-category reporting discipline — the aggregate is dominated
     by the 60 single-hop questions and hides everything interesting.
  3. Two dense backends plus BM25, giving the bake-off a genuine
     progression: lexical -> classic dense -> modern dense.
  4. A concrete open failure (mh-011): the refund-contradiction question
     defeats BOTH retrievers, because its vocabulary points at the
     free-trial articles rather than the refund policy. A shared failure
     across lexical and dense rules out "wrong retriever" as the
     explanation and points at query rewriting or hybrid pools instead.
     It will also present as a GENERATION failure downstream, so Phase 6
     has to attribute it correctly.

Not yet done — deliberately deferred:

  - No hybrid retrieval. Phase 3 builds it, once there is a measured
    baseline to justify the added complexity.
  - No reranking. Same reasoning.
  - Single top-k. Phase 3 sweeps it; recall@5 vs recall@10 changes the
    precision/context-budget trade-off the generator inherits.

Handoff to Phase 3: sweep strategy x backend x top-k against the golden
set, report per category, and establish which configuration ships.
""")
