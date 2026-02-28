import os
import re
from typing import List, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


def clean_text(text: str) -> str:
    """
    Basic text cleaning used by the TF-IDF vectorizer:
    - lowercase
    - remove punctuation and non-letter characters
    - collapse extra whitespace
    """
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def get_small_fake_news_dataset() -> Tuple[List[str], List[str]]:
    """
    Returns a tiny, hard-coded fake news dataset.
    In a real project you would load this from a CSV or database.
    Labels are simple strings: 'FAKE' or 'REAL'.
    """
    data = [
        (
            "Government report confirms steady economic growth and rising employment over the last year.",
            "REAL",
        ),
        (
            "Scientists at a respected university discover a new treatment that significantly reduces flu symptoms.",
            "REAL",
        ),
        (
            "Local city council approves funding for new public transport routes to reduce traffic congestion.",
            "REAL",
        ),
        (
            "New study shows that regular exercise improves mental health and sleep quality.",
            "REAL",
        ),
        (
            "International health organization warns about the spread of a new seasonal virus and suggests precautions.",
            "REAL",
        ),
        (
            "Breaking: drinking only water for one week guarantees you will never get sick again.",
            "FAKE",
        ),
        (
            "Scientists prove the moon is made entirely of cheese and plan a tasting mission.",
            "FAKE",
        ),
        (
            "Government secretly gives every citizen a million dollars but asks people not to tell anyone.",
            "FAKE",
        ),
        (
            "New smartphone automatically reads your thoughts and sends messages without your permission.",
            "FAKE",
        ),
        (
            "Experts say eating chocolate for every meal makes you live 50 years longer.",
            "FAKE",
        ),
    ]

    texts = [item[0] for item in data]
    labels = [item[1] for item in data]
    return texts, labels


def train_and_save_model() -> None:
    """
    Trains a simple Logistic Regression classifier with TF-IDF features
    on a small, in-memory dataset and saves both the model and vectorizer
    as .pkl files.
    """
    texts, labels = get_small_fake_news_dataset()

    vectorizer = TfidfVectorizer(
        preprocessor=clean_text,
        stop_words="english",
        max_features=2000,
    )

    X = vectorizer.fit_transform(texts)

    model = LogisticRegression(max_iter=1000)
    model.fit(X, labels)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.pkl")
    vectorizer_path = os.path.join(base_dir, "vectorizer.pkl")

    joblib.dump(model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    print(f"Saved model to: {model_path}")
    print(f"Saved vectorizer to: {vectorizer_path}")


if __name__ == "__main__":
    train_and_save_model()

