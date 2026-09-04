# Interactive demo

```bash
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

Runs from any directory — the app locates the project root the same way
the notebooks do.

## What it shows

Most RAG demos show you an answer. This one shows the answer **and the
evidence for whether to believe it**, because that is what the project is
about. A fluent answer is easy; knowing whether it's trustworthy is the
hard part.

| Mode | What it does |
|---|---|
| **Ask anything** | Type a question. See the retrieved context, the answer, refusal detection, citation integrity, and the token cost of the context handed to the generator. |
| **Golden set** | Pick one of the 120 labelled questions and see the evaluation against its ground truth — which articles *should* have been retrieved, which were, and the failure mode if they weren't. |
| **Is the judge trustworthy?** | The audit that decides whether any judge score is worth reading: accuracy on cases with known verdicts, plus the bias probes. |

## No API key required

The extractive baseline and every judge-free metric work offline —
refusal detection, citation integrity, context precision and recall are
all regexes, set operations and counts against ground truth.

A credential adds the LLM answerer arms and the semantic judge. The app
says which mode it's in and never implies more evidence than it has: with
no key, judged scores are labelled as coming from a judge that **fails its
own validation gate at 50%**.

## Things worth trying

- **`How do I buy a gift card?`** — deliberately uncovered by the corpus.
  The correct behaviour is a refusal. Answering it confidently is the most
  expensive failure this system can produce.
- **`mh-011`** in the Golden set tab — the planted contradiction. Two
  articles state different refund windows (30 vs 14 days). Both retrieve at
  depth 15, so the evidence is present; whether an answer *flags* the
  conflict rather than silently picking one is the open question.
- **The judge tab's `contradicted` row** — the lexical judge scores
  contradictions as *fully faithful*, because a contradicting sentence
  reuses nearly every term of the context it contradicts. That single row
  is the argument for paying for a semantic judge.

## Where the numbers come from

Every panel calls the same `src/` modules the notebooks and CI use. There
is no separate demo path that could quietly disagree with the measured
results — if the app shows a number, the test suite covers the code that
produced it.
