"""
Email Spam Classifier
TF-IDF + Logistic Regression pipeline for spam/ham detection.
Includes: preprocessing, training, evaluation, FastAPI inference server.
"""

import re
import json
import pickle
import os
from typing import List

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

MODEL_PATH = "./spam_model.pkl"

# ─────────────────────────────────────────────────────────────────────────────
# Sample data (extend with real dataset: SMS Spam Collection / Enron)
# ─────────────────────────────────────────────────────────────────────────────
SAMPLE_EMAILS = [
    ("Congratulations! You've won a $1,000 Walmart gift card. Click here NOW!", "spam"),
    ("URGENT: Your account will be suspended. Verify immediately at http://phish.io", "spam"),
    ("FREE iPhone! Just fill out this survey and claim your prize!", "spam"),
    ("Make $5000 a week working from home. No experience needed!", "spam"),
    ("WINNER: You have been selected for a cash prize of $500,000!", "spam"),
    ("Cheap Viagra! Order now and get 70% off. No prescription needed.", "spam"),
    ("Hi, are you free for a call tomorrow at 2pm?", "ham"),
    ("The meeting has been rescheduled to Friday. Please update your calendar.", "ham"),
    ("Thanks for sending over the report. I'll review it by end of day.", "ham"),
    ("Can you pick up milk on your way home?", "ham"),
    ("Your order #12345 has shipped and will arrive by Thursday.", "ham"),
    ("Reminder: your dentist appointment is tomorrow at 10am.", "ham"),
    ("Please find attached the Q3 financial summary for your review.", "ham"),
    ("Hey! Long time no talk. How have you been?", "ham"),
    ("Your password was changed successfully. If this wasn't you, contact support.", "ham"),
    ("Team lunch is at noon today. See you there!", "ham"),
    ("You are PRE-APPROVED for a loan up to $50,000. Apply now!", "spam"),
    ("Earn easy money by clicking ads! $10 per click guaranteed!", "spam"),
    ("Your subscription has been renewed. $9.99 charged to your card.", "ham"),
    ("Project deadline is next Monday. Please send your updates.", "ham"),
]


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def preprocess(text: str) -> str:
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " URL ", text)          # mask URLs
    text = re.sub(r"\$[\d,]+", " MONEY ", text)               # mask dollar amounts
    text = re.sub(r"\d+", " NUM ", text)                      # mask numbers
    text = re.sub(r"[^a-z\s]", " ", text)                    # remove special chars
    text = re.sub(r"\s+", " ", text).strip()
    return text


# ─────────────────────────────────────────────────────────────────────────────
# Build pipeline
# ─────────────────────────────────────────────────────────────────────────────
def build_pipeline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            preprocessor=preprocess,
            ngram_range=(1, 2),
            max_features=10_000,
            sublinear_tf=True,
            min_df=1,
        )),
        ("clf", LogisticRegression(
            C=1.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
        )),
    ])


# ─────────────────────────────────────────────────────────────────────────────
# Train
# ─────────────────────────────────────────────────────────────────────────────
def train(data=None, model_path: str = MODEL_PATH):
    data = data or SAMPLE_EMAILS
    texts  = [d[0] for d in data]
    labels = [d[1] for d in data]

    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )

    model = build_pipeline()
    model.fit(X_train, y_train)

    # Evaluation
    y_pred = model.predict(X_test)
    print("=== Test Set Performance ===")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # Cross-val on full set
    cv_scores = cross_val_score(model, texts, labels, cv=min(5, len(data) // 2), scoring="accuracy")
    print(f"\nCross-val accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    # Save
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"Model saved to {model_path}")
    return model


# ─────────────────────────────────────────────────────────────────────────────
# Predict
# ─────────────────────────────────────────────────────────────────────────────
def load_model(model_path: str = MODEL_PATH):
    with open(model_path, "rb") as f:
        return pickle.load(f)


def predict(texts: List[str], model=None, model_path: str = MODEL_PATH) -> List[dict]:
    if model is None:
        model = load_model(model_path)
    labels = model.predict(texts)
    probs  = model.predict_proba(texts)
    classes = model.classes_
    return [
        {
            "text": t,
            "label": label,
            "confidence": round(float(max(prob)), 4),
            "scores": {c: round(float(p), 4) for c, p in zip(classes, prob)},
        }
        for t, label, prob in zip(texts, labels, probs)
    ]


def get_top_features(model, n: int = 20):
    """Return top spam/ham indicator words."""
    tfidf = model.named_steps["tfidf"]
    clf   = model.named_steps["clf"]
    feature_names = np.array(tfidf.get_feature_names_out())
    coefs = clf.coef_[0]
    top_spam = feature_names[np.argsort(coefs)[-n:][::-1]].tolist()
    top_ham  = feature_names[np.argsort(coefs)[:n]].tolist()
    return {"spam_indicators": top_spam, "ham_indicators": top_ham}


# ─────────────────────────────────────────────────────────────────────────────
# FastAPI server
# ─────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Email Spam Classifier API", version="1.0")

_model = None


class ClassifyRequest(BaseModel):
    texts: List[str]


class ClassifyResponse(BaseModel):
    results: List[dict]


@app.on_event("startup")
def load_on_startup():
    global _model
    if os.path.exists(MODEL_PATH):
        _model = load_model(MODEL_PATH)
        print("Spam model loaded.")
    else:
        print("No model found. Training on sample data...")
        _model = train()


@app.post("/classify", response_model=ClassifyResponse)
def api_classify(req: ClassifyRequest):
    if _model is None:
        raise HTTPException(503, "Model not ready.")
    return {"results": predict(req.texts, _model)}


@app.post("/train")
def api_train():
    global _model
    _model = train()
    return {"status": "training complete"}


@app.get("/features")
def api_features(n: int = 20):
    if _model is None:
        raise HTTPException(503, "Model not ready.")
    return get_top_features(_model, n)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": _model is not None}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "predict", "serve"], default="serve")
    parser.add_argument("--texts", nargs="+",
                        default=["Congratulations! You won a free iPhone!",
                                 "Can we reschedule the meeting to Thursday?"])
    parser.add_argument("--port", type=int, default=8001)
    args = parser.parse_args()

    if args.mode == "train":
        train()
    elif args.mode == "predict":
        if not os.path.exists(MODEL_PATH):
            print("No model found, training first...")
            train()
        results = predict(args.texts)
        print(json.dumps(results, indent=2))
    else:
        uvicorn.run("spam_classifier:app", host="0.0.0.0", port=args.port, reload=True)
