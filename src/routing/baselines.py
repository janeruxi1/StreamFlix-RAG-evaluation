"""The two scikit-learn routers. Both return class probabilities."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union


def tfidf_router(C: float = 10.0):
    """Word 1-2 grams plus character 2-5 grams, then logistic regression.

    Character n-grams carry the typos and word fragments that support
    tickets are full of ("recieved", "top-up", "topup"). This is the arm
    to beat: it trains in seconds and needs nothing but scikit-learn.
    """
    features = make_union(
        TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1),
        TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5),
                        sublinear_tf=True, min_df=2),
    )
    return make_pipeline(features, LogisticRegression(C=C, max_iter=3000))


def embedding_router(C: float = 10.0) -> LogisticRegression:
    """Logistic regression over fixed sentence embeddings."""
    return LogisticRegression(C=C, max_iter=3000)


def predict(model, X) -> tuple[np.ndarray, np.ndarray]:
    """(predicted label, top-class probability) for each row."""
    proba = model.predict_proba(X)
    classes = np.asarray(model.classes_)
    return classes[proba.argmax(axis=1)], proba.max(axis=1)
