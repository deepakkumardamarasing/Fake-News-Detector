# Fake News Detector (Flask + Logistic Regression)

Simple, beginner‑friendly fake news detection web app built with **Flask**, **TF‑IDF**, and **Logistic Regression**.

The app:

- accepts a news article as plain text
- preprocesses it (lowercase, remove punctuation, English stopwords via TF‑IDF)
- predicts whether the news is **REAL** or **FAKE**
- shows a **confidence score**
- generates a short **extractive summary** (top sentences)

---

## Folder structure

The project root acts as `fake-news-detector/`:

- `app.py` – Flask web application (loads model + vectorizer and serves the UI)
- `train.py` – trains a small Logistic Regression model on a tiny in‑memory dataset and saves:
  - `model.pkl`
  - `vectorizer.pkl`
- `model.pkl` – trained Logistic Regression model (created after running `train.py`)
- `vectorizer.pkl` – TF‑IDF vectorizer (created after running `train.py`)
- `templates/index.html` – HTML UI (Jinja2 template)
- `static/style.css` – basic styling for the app
- `requirements.txt` – Python dependencies
- `README.md` – this file

---

## Prerequisites

- Python **3.9+** installed
- `pip` available on your PATH

On Windows, you can check:

```bash
python --version
pip --version
```

If you have both `python` and `python3` installed, use whichever matches your environment.

---

## 1. Create and activate a virtual environment (recommended)

From the project folder (where `app.py` and `train.py` live):

```bash
cd "c:\Users\deepa\OneDrive\Deepak Personal Ino\Projects\Facke news detection"
python -m venv .venv
```

Activate it (PowerShell):

```bash
.venv\Scripts\Activate.ps1
```

or (Command Prompt):

```bash
.venv\Scripts\activate.bat
```

You should see `(.venv)` in your terminal prompt.

---

## 2. Install dependencies

With the virtual environment active:

```bash
pip install -r requirements.txt
```

This installs Flask, scikit‑learn, and other required libraries.

---

## 3. Train the model

Run the training script **once** to create `model.pkl` and `vectorizer.pkl`:

```bash
python train.py
```

You should see messages like:

```text
Saved model to: ...\model.pkl
Saved vectorizer to: ...\vectorizer.pkl
```

After this step, `app.py` can load the saved artifacts.

---

## 4. Start the Flask web app

Still in the same folder and virtual environment, run:

```bash
python app.py
```

By default Flask runs in development mode at:

- `http://127.0.0.1:5000/`

Open that URL in your browser.

---

## 5. Using the app

1. Open `http://127.0.0.1:5000/` in a browser.
2. Paste or type a news article into the large text area.
3. Click **Analyze**.
4. The app will display:
   - **Prediction**: `REAL` or `FAKE`
   - **Confidence**: probability (%) of the predicted class
   - **Short Summary**: 1–2 key sentences extracted from the article

If you submit an empty form, an error message will ask you to enter text.

---

## Implementation details (high‑level)

- **Preprocessing**
  - `train.py` defines `clean_text`:
    - converts text to lowercase
    - removes punctuation and non‑alphabet characters
    - collapses multiple spaces
  - `TfidfVectorizer` uses this function as its `preprocessor` and removes English stopwords.

- **Model**
  - Uses `LogisticRegression` from scikit‑learn.
  - Trained on a tiny, hard‑coded dataset (`REAL` vs `FAKE` examples) in `train.py`.
  - Saves the trained model to `model.pkl`.

- **Backend (Flask)**
  - `app.py` loads `model.pkl` and `vectorizer.pkl` once at startup.
  - `POST /`:
    - takes the raw article text
    - transforms it with the TF‑IDF vectorizer
    - calls `model.predict_proba(...)` to get class probabilities
    - picks the class with highest probability:
      - label becomes `REAL` or `FAKE`
      - confidence is shown as a percentage
  - Uses a small extractive summarizer:
    - splits the text into sentences
    - scores each sentence using word frequencies
    - returns the top 1–2 sentences.

- **Frontend**
  - `templates/index.html` is a simple Jinja2 template (no React).
  - `static/style.css` adds a clean, modern dark‑mode look while staying simple.

---

## Notes and possible extensions

- The dataset in `train.py` is intentionally tiny and only for demonstration.
- For better performance:
  - replace the in‑memory dataset with a real fake‑news dataset (CSV)
  - tweak vectorizer settings (n‑grams, min_df, etc.)
  - experiment with other classifiers (e.g., LinearSVC).

