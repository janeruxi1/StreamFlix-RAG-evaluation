"""
Phase 1 — Corpus Construction & Audit
======================================

Before any retrieval or generation, two artifacts have to exist and be
trustworthy:

  A. THE CORPUS — 45 StreamFlix help-centre articles across 6 categories.
  B. THE GOLDEN SET — 120 questions with ground-truth source labels.

The golden set is the load-bearing piece. Every retrieval and generation
metric in Phases 3, 5, and 6 is measured against its labels, so an
undetected error here silently invalidates the whole evaluation. This
notebook builds both and audits them.

Design decision worth stating up front: the corpus contains DELIBERATE
FLAWS. A clean, internally consistent, fully-covering corpus would make
the evaluation phases report uniformly high scores and teach nothing.
Real help centres contradict themselves, go stale, and have gaps. Ours
does too, on purpose, and the flaws are documented so the eval can
measure whether the system handles them.

Sections
--------
  A.  Build the corpus + golden set
  B.  Corpus profile — size, categories, word and token distribution
  C.  Golden-set profile — category balance and difficulty
  D.  Ground-truth coverage — which articles get exercised
  E.  Deliberate flaws — the traps, and what each one tests
  E2. Benchmark difficulty — BM25 baseline, the floor for every later result
  F.  Integrity audit + stated limitations
  G.  Verdict + handoff to Phase 2
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt
import pandas as pd

from src.corpus.build import (
    load_corpus,
    load_golden_set,
    orphan_articles,
    validate_corpus,
    write_corpus,
    write_golden_set,
)
from src.corpus.difficulty import (
    BM25,
    count_tokens,
    evaluate_lexical_baseline,
    most_similar_pairs,
    question_article_overlap,
    tokenizer_available,
)
from src.corpus.seed_articles import COVERAGE_GAPS, KNOWN_CORPUS_FLAWS

FIG_DIR = Path("reports/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)


# =====================================================================
# A. Build
# =====================================================================
print("=" * 78)
print("A. BUILD — materialize corpus and golden set to disk")
print("=" * 78)

n_articles = write_corpus()
n_golden = write_golden_set()
print(f"\nWrote {n_articles} articles      -> data/corpus/*.md")
print(f"Wrote {n_golden} golden questions -> data/golden_set/golden_questions.json")

articles = load_corpus()
golden = load_golden_set()
print(f"\nLoaded back: {len(articles)} articles, {len(golden)} questions")
print("""
The round trip is deliberate. The retrieval pipeline reads markdown
files from disk the way a real ingestion job would, rather than
importing Python objects. It also makes corpus changes reviewable as
a diff.
""")


# =====================================================================
# B. Corpus profile
# =====================================================================
print("\n" + "=" * 78)
print("B. CORPUS PROFILE")
print("=" * 78)

corpus_df = pd.DataFrame([
    {
        "article_id": a.article_id,
        "title": a.title,
        "category": a.category,
        "last_updated": a.last_updated,
        "words": a.word_count,
    }
    for a in articles
])

print(f"\nArticles by category:")
cat_counts = corpus_df["category"].value_counts().sort_index()
for cat, n in cat_counts.items():
    bar = "█" * n
    print(f"  {cat:<12} {n:>3}  {bar}")

corpus_df["tokens"] = [count_tokens(a.as_document()) for a in articles]

w = corpus_df["words"]
t = corpus_df["tokens"]
exact = "exact (tiktoken)" if tokenizer_available() else "approximate (fallback)"

print(f"\nArticle length:")
print(f"  {'':<10} {'words':>8} {'tokens':>8}")
print(f"  {'-'*10} {'-'*8} {'-'*8}")
print(f"  {'total':<10} {w.sum():>8,} {t.sum():>8,}")
print(f"  {'min':<10} {w.min():>8} {t.min():>8}")
print(f"  {'median':<10} {int(w.median()):>8} {int(t.median()):>8}")
print(f"  {'mean':<10} {w.mean():>8.0f} {t.mean():>8.0f}")
print(f"  {'max':<10} {w.max():>8} {t.max():>8}")
print(f"\n  Token counts are {exact}.")

# Embedding-window headroom. BGE-small and most sentence-transformers
# models truncate at 512 tokens; anything above that is silently cut.
EMBED_LIMIT = 512
over_limit = corpus_df[corpus_df["tokens"] > EMBED_LIMIT]
print(f"\nEmbedding-window check (limit {EMBED_LIMIT} tokens):")
print(f"  Articles exceeding the limit: {len(over_limit)}")
if len(over_limit):
    print(over_limit[["article_id", "tokens"]].to_string(index=False))
    print("  ⚠️  These would be silently truncated if embedded whole.")
else:
    print(f"  ✓ Longest article is {t.max()} tokens — "
          f"{EMBED_LIMIT - t.max()} tokens of headroom.")

print(f"""
Why this matters for Phase 2: every article fits inside a single
embedding window, so we are never FORCED to chunk. That makes the
chunking ablation a genuine experiment — does splitting help or hurt
retrieval? — rather than a workaround for a hard constraint. Whole-article
embedding becomes a legitimate arm in the Phase 3 bake-off.
""")

# Staleness — which articles have not been touched recently
corpus_df["updated_dt"] = pd.to_datetime(corpus_df["last_updated"])
stale = corpus_df.nsmallest(5, "updated_dt")[
    ["article_id", "title", "last_updated"]
]
print("Five least-recently-updated articles:")
print(stale.to_string(index=False))
print("""
bill-009 is the deliberately stale one — see Section E.
""")


# =====================================================================
# C. Golden-set profile
# =====================================================================
print("\n" + "=" * 78)
print("C. GOLDEN-SET PROFILE")
print("=" * 78)

golden_df = pd.DataFrame([
    {
        "question_id": q["question_id"],
        "category": q["category"],
        "n_sources": len(q["gt_article_ids"]),
        "question_words": len(q["question"].split()),
    }
    for q in golden
])

print("\nQuestions by category:")
design = {"single_hop": 60, "multi_hop": 20, "ambiguous": 15,
          "out_of_scope": 25}
print(f"  {'category':<14} {'n':>4} {'designed':>9} {'match':>7}")
print(f"  {'-'*14} {'-'*4} {'-'*9} {'-'*7}")
for cat, target in design.items():
    actual = int((golden_df["category"] == cat).sum())
    ok = "✓" if actual == target else "✗"
    print(f"  {cat:<14} {actual:>4} {target:>9} {ok:>7}")

print("\nGround-truth sources per question (in-scope only):")
in_scope = golden_df[golden_df["category"] != "out_of_scope"]
src_dist = in_scope["n_sources"].value_counts().sort_index()
for n_src, count in src_dist.items():
    label = f"{n_src} article" + ("s" if n_src != 1 else "")
    print(f"  {label:<12} {count:>3} questions  {'█' * count}")

print(f"""
Read: {int((in_scope['n_sources'] >= 2).sum())} of {len(in_scope)} in-scope questions require more than one
article. A retriever that only ever returns top-1 will fail those
regardless of how good its ranking is — which is exactly what the
top-k sweep in Phase 3 is designed to expose.
""")


# =====================================================================
# D. Ground-truth coverage
# =====================================================================
print("\n" + "=" * 78)
print("D. GROUND-TRUTH COVERAGE")
print("=" * 78)

cited: dict[str, int] = {}
for q in golden:
    for aid in q["gt_article_ids"]:
        cited[aid] = cited.get(aid, 0) + 1

orphans = orphan_articles(articles, golden)
covered = len(articles) - len(orphans)

print(f"\nArticles exercised by at least one question: {covered}/{len(articles)} "
      f"({covered/len(articles):.0%})")
print(f"Articles never cited (retrieval distractors):  {len(orphans)}")
if orphans:
    titles = {a.article_id: a.title for a in articles}
    for aid in orphans:
        print(f"    {aid:<12} {titles[aid]}")

print(f"""
Uncited articles are not a defect. A corpus where every document is
someone's answer is unrealistically easy — real retrieval has to
reject plausible-but-wrong neighbours. These are the distractors.
""")

print("Most-cited articles (highest retrieval pressure):")
top_cited = sorted(cited.items(), key=lambda kv: -kv[1])[:8]
titles = {a.article_id: a.title for a in articles}
for aid, n in top_cited:
    print(f"  {aid:<12} {n:>2} questions   {titles[aid][:44]}")


# =====================================================================
# E. Deliberate flaws
# =====================================================================
print("\n" + "=" * 78)
print("E. DELIBERATE FLAWS — the traps and what each tests")
print("=" * 78)
print("""
A corpus without flaws produces an evaluation without findings. These
are planted, documented, and each has matching golden questions that
probe it.
""")

# Article-level BM25, matching how Phase 1 frames retrieval.
bm25 = BM25({a.article_id: a.as_document() for a in articles})

for i, flaw in enumerate(KNOWN_CORPUS_FLAWS, 1):
    print(f"{i}. [{flaw['kind'].upper()}] {flaw['flaw_id']}")
    print(f"   Articles: {', '.join(flaw['article_ids'])}")
    print(f"   {flaw['description']}")
    probes = [
        q["question_id"] for q in golden
        if set(q["gt_article_ids"]) & set(flaw["article_ids"])
    ]
    if probes:
        print(f"   Probed by {len(probes)} question(s): {', '.join(probes[:6])}"
              + (" ..." if len(probes) > 6 else ""))
    else:
        # A flaw with no question pointing at it is not necessarily an
        # oversight. Some flaws are DISTRACTOR-type: the correct
        # behaviour is that the article is never retrieved as an answer,
        # so there is nothing to ask about it. Printing a bare "0" makes
        # that design look like a gap, so measure the thing that
        # actually matters — how often it pollutes retrieved context.
        distractor_hits = [
            q["question_id"] for q in golden
            if set(flaw["article_ids"]) & set(
                bm25.rank(q["question"], top_k=5))
        ]
        top_three = [
            q["question_id"] for q in golden
            if set(flaw["article_ids"]) & set(
                bm25.rank(q["question"], top_k=3))
        ]
        print(f"   Probed by 0 questions — this is a DISTRACTOR-type flaw.")
        print(f"   Its job is to be retrieved and be wrong, so it is measured")
        print(f"   by contamination rather than by recall:")
        print(f"     pulled into the top 5 for {len(distractor_hits)}/{len(golden)} questions")
        print(f"     reached the top 3 for     {len(top_three)}/{len(golden)} questions")
        if top_three:
            print(f"     e.g. {', '.join(top_three[:5])}")
        print(f"   It is never a correct answer to anything, so every one of")
        print(f"   those is context the generator has to ignore.")
    print()

print(f"COVERAGE GAPS ({len(COVERAGE_GAPS)} topics with no article at all):")
for gap in COVERAGE_GAPS:
    print(f"  - {gap}")

n_oos = sum(1 for q in golden if q["category"] == "out_of_scope")
print(f"""
These gaps generate the {n_oos} out-of-scope questions. A system that
answers them is hallucinating — it has no source to ground on. This
is the single most important guardrail in the project, because a
confidently wrong answer about a refund or a charge is worse for a
support team than no answer at all.

The hardest refusals are the ones with an ADJACENT article:
  oos-007  audio description   -> subtitles article is nearby but silent
  oos-010  password sharing    -> stream limits are nearby but not policy
  oos-024  season release date -> releases article explains where to look
  oos-025  goodwill credit     -> refund policy is nearby but narrower

Those four are where a weak system will most likely over-answer.
""")


# =====================================================================
# E2. Benchmark difficulty — what does a trivial baseline score?
# =====================================================================
print("\n" + "=" * 78)
print("E2. BENCHMARK DIFFICULTY — the floor every later result is measured against")
print("=" * 78)
print("""
A RAG project that reports "our retriever achieves 94% recall" without
saying what keyword matching achieves has reported nothing. If BM25
already gets 91%, the dense retriever earned 3 points, not 94.

So before any embedding model is chosen or any API key is spent, we
establish the floor: pure lexical BM25, no embeddings, no LLM.
""")

documents = {a.article_id: a.as_document() for a in articles}
titles_map = {a.article_id: a.title for a in articles}
per_cat_results, overall = evaluate_lexical_baseline(documents, golden)

print("BM25 lexical baseline (out-of-scope excluded — no ground truth):\n")
print(f"  {'category':<14} {'n':>4} {'hit@1':>8} {'hit@3':>8} "
      f"{'hit@5':>8} {'recall@5':>9}")
print(f"  {'-'*14} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")
for r in per_cat_results:
    print(f"  {r.category:<14} {r.n_questions:>4} {r.hit_at_1:>7.1%} "
          f"{r.hit_at_3:>7.1%} {r.hit_at_5:>7.1%} {r.recall_at_5:>8.1%}")
print(f"  {'-'*14} {'-'*4} {'-'*8} {'-'*8} {'-'*8} {'-'*9}")
print(f"  {overall.category:<14} {overall.n_questions:>4} "
      f"{overall.hit_at_1:>7.1%} {overall.hit_at_3:>7.1%} "
      f"{overall.hit_at_5:>7.1%} {overall.recall_at_5:>8.1%}")

single = next(r for r in per_cat_results if r.category == "single_hop")
multi = next(r for r in per_cat_results if r.category == "multi_hop")
ambig = next(r for r in per_cat_results if r.category == "ambiguous")

print(f"""
>>> FINDING — this reshapes the Phase 3 experiment design.

BM25 reaches {single.recall_at_5:.1%} recall@5 on single-hop questions. Dense
retrieval has roughly {(1 - single.recall_at_5) * 100:.0f} points of headroom there — thin.

The headroom lives elsewhere:
    multi_hop   {multi.recall_at_5:>6.1%}  ->  {(1 - multi.recall_at_5) * 100:>4.0f} points available
    ambiguous   {ambig.recall_at_5:>6.1%}  ->  {(1 - ambig.recall_at_5) * 100:>4.0f} points available

Two consequences, both of which change what Phase 3 does:

  1. BM25 becomes a FIRST-CLASS ARM in the retrieval bake-off, not a
     footnote. Reporting a dense-retrieval number without the lexical
     floor beside it would overstate the contribution.

  2. Results get reported PER CATEGORY, never as a single aggregate.
     The aggregate ({overall.recall_at_5:.1%}) hides the entire story — it is
     dominated by the 60 single-hop questions where the baseline is
     already strong.

This is also why production RAG systems use hybrid retrieval. Lexical
search is genuinely hard to beat on direct factual lookup; dense
retrieval earns its cost on paraphrase, multi-source, and ambiguity.
""")

# --- Are the questions actually keyword-matched to their articles? ---
overlap = question_article_overlap(titles_map, golden)
print("Question / ground-truth-title lexical overlap:")
print(f"  Mean overlap of question content words with GT titles: "
      f"{overlap.mean_title_overlap:.1%}")
print(f"  Questions sharing NO title word with ground truth:     "
      f"{overlap.zero_overlap_count}/{overlap.n_questions} "
      f"({overlap.zero_overlap_rate:.0%})")
print(f"""
{overlap.zero_overlap_rate:.0%} of in-scope questions share no content word with their
ground-truth article title. The questions were written to sound like a
subscriber, not to echo the documentation — "Why is the picture blurry
partway through a movie?" rather than "video quality settings". That is
what keeps BM25 honest at {single.recall_at_5:.0%} rather than near-perfect.
""")

# --- Are the planted near-duplicates really the most confusable? ---
print("Most lexically similar article pairs (Jaccard on content words):\n")
device_cluster = {"dev-003", "dev-004", "dev-005"}
for a, b, sim in most_similar_pairs(documents, top_n=8):
    if {a, b} <= device_cluster:
        tag = "  <- planted device cluster"
    elif {a, b} == {"bill-003", "trial-004"}:
        tag = "  <- DISCOVERED, not planted"
    else:
        tag = ""
    print(f"  {sim:>5.1%}  {a:<10} / {b:<10}{tag}")

print("""
>>> FINDING — the analysis caught a near-duplicate I did not plant.

All three planted device-setup pairs rank in the top 8, so that trap
works. But the single most confusable pair in the corpus is one I never
designed: bill-003 (canceling your subscription) and trial-004
(canceling during your trial).

That makes sense in hindsight. Both describe the same cancellation flow
for different account states, and differ mainly in the consequences —
a paid cancellation keeps access to period end, a trial cancellation
avoids the charge entirely. Which is exactly the distinction a user
asking the bare question "how do I cancel?" needs drawn for them, and
exactly what am-009 probes.

It is now recorded in KNOWN_CORPUS_FLAWS as a discovered rather than
planted flaw, with a regression test so a later corpus edit cannot
silently remove it. Worth stating plainly: the planted traps test what I
anticipated, and this one tests something I did not.
""")


# =====================================================================
# F. Integrity audit
# =====================================================================
print("\n" + "=" * 78)
print("F. INTEGRITY AUDIT")
print("=" * 78)

problems = validate_corpus(articles, golden)

checks = [
    ("Article IDs unique",
     len({a.article_id for a in articles}) == len(articles)),
    ("Question IDs unique",
     len({q["question_id"] for q in golden}) == len(golden)),
    ("All ground-truth references resolve",
     not any("unknown article" in p for p in problems)),
    ("Out-of-scope questions carry no ground truth",
     not any("out_of_scope but cites" in p for p in problems)),
    ("In-scope questions all have ground truth",
     not any("no ground truth" in p for p in problems)),
    ("Category balance matches design",
     all(int((golden_df['category'] == c).sum()) == t
         for c, t in design.items())),
    ("All articles substantive (>= 80 words)",
     bool((corpus_df["words"] >= 80).all())),
]

for label, passed in checks:
    print(f"  {'✓' if passed else '✗'} {label}")

print(f"\nIntegrity problems: {len(problems)}")
for p in problems:
    print(f"  ✗ {p}")
if not problems:
    print("  ✓ none — golden-set labels are safe to measure against")

print("""
--------------------------------------------------------------------------
LIMITATIONS OF THIS GOLDEN SET — stated plainly

  1. SINGLE ANNOTATOR. The corpus and the questions were written by the
     same author, so ground-truth labels are self-assigned. There is no
     inter-annotator agreement statistic because there is one annotator.
     A production benchmark would have at least two people label
     independently and report Cohen's kappa. The mitigation here is
     mechanical rather than statistical: the integrity checks above
     catch structural errors (dangling references, mislabelled scope),
     but they cannot catch a question whose "correct" article is
     arguably wrong.

  2. AMBIGUOUS LABELS ARE JUDGEMENT CALLS. For the 15 ambiguous
     questions, "which articles are relevant" is genuinely debatable.
     Context-precision scores on that slice should be read as
     directional, not exact.

  3. SYNTHETIC CORPUS. Article phrasing is consistent and well-formed
     because one author wrote it in one sitting. Real help centres are
     written by many people over years and are messier. This likely
     makes retrieval EASIER here than in production.

  4. CLEAN QUESTION PHRASING. Questions are grammatical and typo-free.
     Real support queries are terse and misspelled ("cant login on tv").
     Phase 6 revisits this as a robustness check rather than pretending
     the current set covers it.

Naming these matters more than fixing them. A benchmark whose limits are
documented can be reasoned about; one presented as flawless cannot.
--------------------------------------------------------------------------
""")


# =====================================================================
# Chart — corpus and golden-set composition
# =====================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

cat_counts.plot(kind="barh", ax=ax1, color="#5B8FF9", edgecolor="white")
ax1.set_xlabel("Articles")
ax1.set_ylabel("")
ax1.set_title("Corpus composition by category", fontweight="bold")
ax1.grid(axis="x", linestyle="--", alpha=0.4)
ax1.set_axisbelow(True)
for i, v in enumerate(cat_counts.values):
    ax1.text(v + 0.1, i, str(v), va="center", fontsize=9)

order = ["single_hop", "multi_hop", "ambiguous", "out_of_scope"]
colors = ["#5AD8A6", "#5B8FF9", "#F6BD16", "#F6735B"]
counts = [int((golden_df["category"] == c).sum()) for c in order]
bars = ax2.bar(range(len(order)), counts, color=colors, edgecolor="white")
ax2.set_xticks(range(len(order)))
ax2.set_xticklabels([o.replace("_", "\n") for o in order], fontsize=9)
ax2.set_ylabel("Questions")
ax2.set_title("Golden set by question type\n"
              "(out-of-scope = the refusal guardrail)",
              fontweight="bold")
ax2.grid(axis="y", linestyle="--", alpha=0.4)
ax2.set_axisbelow(True)
for bar, v in zip(bars, counts):
    ax2.text(bar.get_x() + bar.get_width() / 2, v + 0.6, str(v),
             ha="center", fontsize=10, fontweight="bold")

plt.suptitle("Phase 1 — corpus and evaluation set composition",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / "01_corpus_composition.png", dpi=140,
            bbox_inches="tight",
            metadata={"Software": None})
print(f"\nSaved -> {FIG_DIR}/01_corpus_composition.png")


# =====================================================================
# G. Verdict
# =====================================================================
print("\n" + "=" * 78)
print("PHASE 1 VERDICT")
print("=" * 78)

n_multi = int((in_scope["n_sources"] >= 2).sum())
print(f"""
Built and audited:

  Corpus         {len(articles)} articles, {corpus_df['words'].sum():,} words / {corpus_df['tokens'].sum():,} tokens, {len(cat_counts)} categories
  Golden set     {len(golden)} questions across 4 difficulty types
  Distractors    {len(orphans)} articles never cited as ground truth
  Planted flaws  {len(KNOWN_CORPUS_FLAWS)} documented, {len(COVERAGE_GAPS)} coverage gaps
  Integrity      {len(problems)} problems

  BM25 floor     single-hop {single.recall_at_5:.0%} | multi-hop {multi.recall_at_5:.0%} | ambiguous {ambig.recall_at_5:.0%} recall@5
                 Every Phase 3 result is reported against this baseline.

What this buys the later phases:

  Phase 3 (retrieval bake-off) can compute context precision and
  recall against real labels rather than eyeballing results, and the
  {n_multi} multi-source questions make top-k a measurable trade-off
  rather than a guess.

  Phase 5 (evaluation harness) has a reference answer for every
  question, so faithfulness is scored against ground truth instead of
  the model's own retrieved context.

  Phase 6 (failure analysis) has {n_oos} out-of-scope questions to
  measure hallucination, and 4 of them sit next to a topically
  adjacent article — the hardest refusals, where over-answering is
  most likely.

Handoff to Phase 2: chunking strategy, embedding model, and vector
store, producing the retrieval baseline that Phase 3 then attacks.
""")
