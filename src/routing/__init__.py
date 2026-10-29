"""Ticket routing: which queue should a support question go to?

Notebook 11 compares four routers on Banking77 (77 intents):

    tfidf        TF-IDF word + character n-grams, logistic regression
    frozen       BGE-small sentence embeddings (frozen), logistic regression
    finetune     BGE-small fine-tuned end to end with a PyTorch loop
    llm          zero-shot LLM with the intent list in the prompt

Heavy work (embedding, fine-tuning, LLM calls) runs once in
scripts/run_routing.py and writes per-ticket predictions. The notebook
recomputes every number from those predictions, so it needs neither
torch nor a key.
"""
