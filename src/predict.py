"""
predict.py
==========
Inference module for SMS Spam classification.

Loads the serialized Scikit-learn Pipeline and predicts whether
incoming messages are Legitimate (Ham) or Potential Spam / Scam.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Union, List
import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing import clean_text_standard

DEFAULT_MODEL_PATH = os.path.join("models", "best_model.pkl")
_LOADED_MODEL = None


def load_model(model_path: str = DEFAULT_MODEL_PATH):
    """
    Loads and caches the serialized model pipeline.

    Parameters
    ----------
    model_path : str
        Path to the joblib serialized pipeline file.

    Returns
    -------
    Pipeline
        The loaded Scikit-learn pipeline.
    """
    global _LOADED_MODEL
    if _LOADED_MODEL is not None and model_path == DEFAULT_MODEL_PATH:
        return _LOADED_MODEL

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at '{model_path}'. "
            "Please run 'python src/train.py' first to train and save the model."
        )

    model = joblib.load(model_path)
    if model_path == DEFAULT_MODEL_PATH:
        _LOADED_MODEL = model
    return model


def predict_message(text: str, model=None) -> Dict[str, Any]:
    """
    Classifies an input SMS text as Legitimate or Spam.

    Parameters
    ----------
    text : str
        Raw input SMS text.
    model : optional
        Preloaded pipeline. If None, loads from DEFAULT_MODEL_PATH.

    Returns
    -------
    Dict[str, Any]
        Dictionary with classification details:
        - raw_text: original input
        - cleaned_text: preprocessed text
        - prediction_num: 0 or 1
        - label: 'Ham' or 'Spam'
        - category: 'Legitimate Message' or 'Potential Spam / Scam'
        - probability: float or None
        - confidence_percent: str or None
        - top_contributing_words: list of detected vocabulary words
    """
    if not isinstance(text, str) or not text.strip():
        return {
            "raw_text": text,
            "cleaned_text": "",
            "prediction_num": 0,
            "label": "Ham",
            "category": "Legitimate Message",
            "probability": 0.0,
            "confidence_percent": "N/A (Empty Input)",
            "top_contributing_words": [],
        }

    if model is None:
        model = load_model()

    # Inference using complete pipeline
    pred_num = int(model.predict([text])[0])
    has_proba = hasattr(model, "predict_proba")

    spam_prob = None
    confidence_str = "N/A"
    if has_proba:
        probas = model.predict_proba([text])[0]
        spam_prob = float(probas[1])
        # Confidence is probability of the predicted class
        conf = probas[pred_num] * 100
        confidence_str = f"{conf:.2f}%"

    label = "Spam" if pred_num == 1 else "Ham"
    category = "Potential Spam / Scam" if pred_num == 1 else "Legitimate Message"

    # Inspect matched vocabulary features to assist student explanation
    detected_words = []
    try:
        vec = model.named_steps.get("vectorizer")
        if vec:
            cleaned = clean_text_standard(text)
            words = set(cleaned.split())
            vocab = set(vec.vocabulary_.keys())
            detected_words = sorted(list(words.intersection(vocab)))
    except Exception:
        pass

    return {
        "raw_text": text,
        "cleaned_text": clean_text_standard(text),
        "prediction_num": pred_num,
        "label": label,
        "category": category,
        "spam_probability": spam_prob,
        "confidence_percent": confidence_str,
        "detected_vocab_words": detected_words,
    }


if __name__ == "__main__":
    test_samples = [
        "Congratulations! You've won a £1,000 Walmart gift card. Call 0800-999-123 to claim your prize now!",
        "Hey, are you free to grab coffee after the ML lecture tomorrow?",
        "URGENT! Your banking password has expired. Click http://secure-update-login.com to reset immediately.",
        "Can you send me the lecture notes from Monday?",
    ]

    print("--- INFERENCE DEMO ---")
    for msg in test_samples:
        res = predict_message(msg)
        print(f"\nMessage: \"{msg}\"")
        print(f"Prediction:  {res['category']} ({res['label']})")
        print(f"Confidence:  {res['confidence_percent']} (Spam prob: {res['spam_probability']})")
        print(f"Vocab Words: {res['detected_vocab_words'][:10]}")
