"""Demo: ask the StreamFlix support bot a question.

    pip install streamlit
    streamlit run app/streamlit_app.py

Runs on the keyless extractive baseline (BM25 retrieval + sentence
selection + threshold refusal). It shows the system the evaluation harness
measures, including the refusal behaviour, with no API key.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from src.generation.demo import answer_question

st.set_page_config(page_title="StreamFlix support bot", page_icon="🎬")
st.title("StreamFlix support bot")
st.caption("Extractive baseline over a synthetic help centre. Try the gift-card "
           "example to see it refuse; it still wrongly answers about half of "
           "out-of-scope questions, which is what the evaluation measures.")

EXAMPLES = ["When does StreamFlix take my monthly payment?",
            "My card was declined. What happens to my account?",
            "How do I buy a StreamFlix gift card?",
            "Charged after I cancelled — what do I do?"]
choice = st.selectbox("Examples", [""] + EXAMPLES)
question = st.text_input("Your question", value=choice)

if question:
    res = answer_question(question)
    if res.refused:
        st.warning(res.answer)
    else:
        st.success(res.answer)
    with st.expander("Retrieved sources"):
        for article_id, snippet, score in res.sources:
            st.markdown(f"**{article_id}** (score {score:.2f})")
            st.text(snippet)
