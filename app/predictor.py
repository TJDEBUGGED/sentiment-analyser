"""
predictor.py — Loads the saved model and runs predictions.
"""

import pickle
import numpy as np
from pathlib import Path

MODEL_DIR = Path(__file__).parent.parent / "model"

class SentimentPredictor:
    def __init__(self):
        with open(MODEL_DIR / "vectorizer.pkl", "rb") as f:
            self.vectorizer = pickle.load(f)
        with open(MODEL_DIR / "classifier.pkl", "rb") as f:
            self.model = pickle.load(f)

    def predict(self, text: str) -> dict:
        X = self.vectorizer.transform([text])
        label_idx = self.model.predict(X)[0]
        proba = self.model.predict_proba(X)[0]
        confidence = float(proba[label_idx])

        label = "positive" if label_idx == 1 else "negative"

        # Map to three-class output for display
        if confidence < 0.60:
            sentiment = "neutral"
        else:
            sentiment = label

        return {
            "sentiment": sentiment,
            "confidence": round(confidence, 4),
            "scores": {
                "negative": round(float(proba[0]), 4),
                "positive": round(float(proba[1]), 4),
            }
        }

# Singleton — loaded once when the app starts
predictor = SentimentPredictor()
