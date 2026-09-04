# 📝 Brief — Help Centre Answering Assistant

**From:** Priya Nandakumar, Director of Customer Experience
**To:** Data Science Team
**Re:** Whether we can put an AI assistant in front of the help centre

---

## Background

StreamFlix Support handles roughly 48,000 contacts a month. About 60% are
questions the help centre already answers — billing dates, device setup,
plan limits — but customers either can't find the article or don't trust
that they've found the right one. Those contacts cost us ~$4.10 each to
resolve through a human agent.

The obvious move is an assistant that answers from the help centre
directly. Every vendor we've spoken to can demo one in a week.

**That's exactly what worries me.**

## The problem I actually want solved

We ran a proof-of-concept with a vendor last quarter. It gave confident,
well-written answers, and we killed it in week three — because we couldn't
answer a simple question from Legal:

> *"How often does it make something up, and how would you know?"*

The vendor's answer was a demo transcript and a satisfaction score. Not
good enough for a system that tells customers when they'll be charged.

So the deliverable I want is not a working assistant. It's **a defensible
answer to whether an assistant is safe to deploy**, with the assistant
built only as far as needed to produce that answer.

## What I need to be able to say

To the exec team, and to Legal, without hedging:

1. **How often it answers something the help centre doesn't cover.** This
   is my biggest exposure. A confidently wrong answer about refund
   eligibility is worse than no assistant at all — it's a written statement
   from StreamFlix that a customer can act on and hold us to.

2. **How often it refuses when it shouldn't.** The failure nobody
   measures. If it says "I don't know" to questions we *do* answer, we've
   spent money to make self-service worse, and it won't show up in any
   hallucination metric.

3. **Whether its sources are real.** If it cites an article, that article
   must exist and must actually say what's claimed. Fabricated citations
   are the worst case: they look like evidence, so they survive the
   scrutiny that would catch a bare assertion.

4. **What it costs per question, and what that buys.** Finance will ask.
   I'd rather bring the trade-off than have it extracted from me.

## Constraints

- **Our help centre contradicts itself.** We know of at least one case —
  two articles quote different refund windows. Rewriting the corpus is a
  six-month content project that isn't funded. The assistant has to behave
  sensibly on a corpus that is genuinely imperfect, and I'd rather know how
  it behaves than pretend the corpus is clean.
- **Coverage gaps are permanent.** Gift cards, business accounts, and
  regional pricing questions come up constantly and we deliberately don't
  document them. Those are exactly the questions that need a refusal.
- **Budget is small and the bar is high.** This is a feasibility study.
  I'd rather have a rigorous answer about a simple system than a vague
  answer about a sophisticated one.

## What "done" looks like

A recommendation I can take to the exec team that states plainly:

- what we'd deploy, and at what cost per question
- what's been measured, with the evidence
- **what hasn't been measured, and what it would take** — I would much
  rather see "we didn't measure this and here's what it costs to" than a
  number nobody can defend

If the answer is "not yet," that's a legitimate outcome. What I can't use
is a system that looks impressive and can't tell me when it's wrong.

---

## Success criteria

| | |
|---|---|
| **Primary** | A defensible deploy / don't-deploy recommendation, with the reasoning legible to a non-specialist |
| **Secondary** | Measured refusal behaviour on questions the corpus can't answer |
| **Explicitly not** | A polished chat UI, or the highest possible score on any single metric |
