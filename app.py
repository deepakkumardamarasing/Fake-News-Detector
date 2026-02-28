import os
from collections import Counter
from typing import List

import joblib
from flask import Flask, render_template, request

from train import clean_text


app = Flask(__name__)


def load_artifacts():
    """
    Load the trained model and vectorizer from disk once
    so they can be reused for each request.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, "model.pkl")
    vectorizer_path = os.path.join(base_dir, "vectorizer.pkl")

    if not os.path.exists(model_path) or not os.path.exists(vectorizer_path):
        raise FileNotFoundError(
            "model.pkl or vectorizer.pkl not found. "
            "Please run 'python train.py' first to train and save the model."
        )

    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    return model, vectorizer


model, vectorizer = load_artifacts()


def split_into_sentences(text: str) -> List[str]:
    """
    Very simple sentence splitter based on punctuation.
    This is not perfect, but good enough for a demo.
    """
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text.strip())
    return [s for s in sentences if s]


def summarize_text(text: str, max_sentences: int = 2) -> str:
    """
    Create a short extractive summary by:
    - splitting the text into sentences
    - scoring each sentence based on word frequency
    - returning the top N sentences in their original order
    """
    sentences = split_into_sentences(text)
    if not sentences:
        return ""

    if len(sentences) <= max_sentences:
        return " ".join(sentences)

    words = clean_text(text).split()
    if not words:
        return sentences[0]

    freq = Counter(words)
    sentence_scores = []

    for sent in sentences:
        sent_words = clean_text(sent).split()
        if not sent_words:
            continue
        score = sum(freq[w] for w in sent_words)
        sentence_scores.append((score, sent))

    if not sentence_scores:
        return sentences[0]

    top_sentences = [
        s for _, s in sorted(sentence_scores, key=lambda x: x[0], reverse=True)[:max_sentences]
    ]

    ordered = [s for s in sentences if s in top_sentences]
    return " ".join(ordered)


@app.route("/", methods=["GET", "POST"])
def index():
    """
    Main page: shows a text area, handles form submission,
    and displays prediction + confidence + summary.
    """
    prediction_label = None
    confidence = None
    summary = None
    user_text = ""
    error_message = None

    if request.method == "POST":
        user_text = request.form.get("news_text", "").strip()

        if not user_text:
            error_message = "Please paste or type a news article before submitting."
        else:
            try:
                features = vectorizer.transform([user_text])
                probabilities = model.predict_proba(features)[0]
                classes = list(model.classes_)

                best_index = int(probabilities.argmax())
                prediction_label = classes[best_index]
                confidence = float(probabilities[best_index] * 100.0)
                summary = summarize_text(user_text)
            except Exception as exc:  # noqa: BLE001
                error_message = f"An error occurred while making a prediction: {exc}"

    return render_template(
        "index.html",
        prediction=prediction_label,
        confidence=confidence,
        summary=summary,
        user_text=user_text,
        error_message=error_message,
    )


if __name__ == "__main__":
    app.run(debug=True)

