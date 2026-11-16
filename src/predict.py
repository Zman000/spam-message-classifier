import os
import sys
sys.path.insert(0, ".")

import joblib

MODEL_PATH = "models/best_model.pkl"
_model = None

def load_model(path=MODEL_PATH):
    global _model
    if _model is None:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Model not found at {path}. Run train.py first.")
        _model = joblib.load(path)
    return _model

def predict_message(text: str, model=None):
    if not text or not text.strip():
        return {"label": "Ham", "confidence": 0.0, "is_spam": False}

    if model is None:
        model = load_model()

    pred = int(model.predict([text])[0])
    proba = model.predict_proba([text])[0]
    confidence = float(proba[pred]) * 100

    return {
        "label": "Spam" if pred == 1 else "Ham",
        "category": "Potential Spam / Scam" if pred == 1 else "Legitimate Message",
        "confidence": round(confidence, 2),
        "spam_probability": round(float(proba[1]), 4),
        "is_spam": pred == 1
    }

if __name__ == "__main__":
    test = "Win £1000 cash prize! Text CLAIM to 8888 now."
    print(predict_message(test))
