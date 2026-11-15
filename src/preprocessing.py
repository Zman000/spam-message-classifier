"""
preprocessing.py
================
Module for SMS text cleaning and preprocessing.

Key Concepts:
- Text Normalization: Converting diverse casing, irregular spacing, and special characters
  into a consistent representation so the vocabulary captures word meaning rather than noise.
- Preservation of Scam Indicators: In spam detection, numbers (e.g. phone numbers, prize amounts),
  currency symbols ($/£/€), and urgent action keywords ('claim', 'win', 'call') provide strong
  discriminative signal. Aggressive lemmatization or over-cleaning can destroy these signals.
- Modes for Experimentation:
    Mode A ('minimal'): Keeps original text untouched (just stripped string).
    Mode B ('standard'): Lowercase, removes punctuation, normalizes whitespace, preserves numbers & currency.
    Mode C ('with_stopwords'): Mode B plus standard English stopword filtering.
"""

import re
import string
import sys
from pathlib import Path
from typing import List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Standard English stop words defined as a lightweight, explainable set
# (Avoids requiring external NLTK downloads while maintaining coursework transparency)
ENGLISH_STOPWORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can't", "cannot", "could", "couldn't",
    "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down", "during",
    "each", "few", "for", "from", "further", "had", "hadn't", "has", "hasn't",
    "have", "haven't", "having", "he", "he'd", "he'll", "he's", "her", "here",
    "here's", "hers", "herself", "him", "himself", "his", "how", "how's", "i",
    "i'd", "i'll", "i'm", "i've", "if", "in", "into", "is", "isn't", "it", "it's",
    "its", "itself", "let's", "me", "more", "most", "mustn't", "my", "myself",
    "no", "nor", "not", "of", "off", "on", "once", "only", "or", "other", "ought",
    "our", "ours", "ourselves", "out", "over", "own", "same", "shan't", "she",
    "she'd", "she'll", "she's", "should", "shouldn't", "so", "some", "such",
    "than", "that", "that's", "the", "their", "theirs", "them", "themselves",
    "then", "there", "there's", "these", "they", "they'd", "they'll", "they're",
    "they've", "this", "those", "through", "to", "too", "under", "until", "up",
    "very", "was", "wasn't", "we", "we'd", "we'll", "we're", "we've", "were",
    "weren't", "what", "what's", "when", "when's", "where", "where's", "which",
    "while", "who", "who's", "whom", "why", "why's", "with", "won't", "would",
    "wouldn't", "you", "you'd", "you'll", "you're", "you've", "your", "yours",
    "yourself", "yourselves"
}


def clean_text_standard(text: str) -> str:
    """
    Standard cleaning for SMS messages (Mode B):
    1. Converts text to lowercase.
    2. Normalizes URLs to 'http_url' token to preserve link signals.
    3. Normalizes currency symbols (£, $, €) to 'currency_symbol' token.
    4. Removes non-alphanumeric punctuation.
    5. Normalizes whitespace.

    Parameters
    ----------
    text : str
        Input raw SMS message.

    Returns
    -------
    str
        Cleaned text string.
    """
    if not isinstance(text, str):
        return ""

    # Step 1: Lowercase conversion
    text = text.lower()

    # Step 2: Normalize URLs to keep signal without sparse unique URLs
    text = re.sub(r"https?://\S+|www\.\S+", "http_url", text)

    # Step 3: Normalize currency symbols (often indicative of lottery/prize scams)
    text = re.sub(r"[$£€]", " currency_symbol ", text)

    # Step 4: Remove punctuation (replace with single space to avoid merging words)
    # Retain alphanumeric characters and spaces
    text = re.sub(r"[^\w\s]", " ", text)

    # Step 5: Whitespace normalization
    text = re.sub(r"\s+", " ", text).strip()

    return text


def clean_text_with_stopwords(text: str) -> str:
    """
    Cleaning with stop word removal (Mode C):
    Performs standard cleaning, then removes common English stopwords.

    Parameters
    ----------
    text : str
        Input raw SMS message.

    Returns
    -------
    str
        Cleaned text with stopwords removed.
    """
    cleaned = clean_text_standard(text)
    tokens = cleaned.split()
    filtered_tokens = [tok for tok in tokens if tok not in ENGLISH_STOPWORDS]
    return " ".join(filtered_tokens)


def preprocess_text(text: str, mode: str = "standard") -> str:
    """
    Unified entry point for preprocessing a single SMS text.

    Parameters
    ----------
    text : str
        The raw SMS string.
    mode : str
        'minimal'        -> Mode A (just raw string stripped)
        'standard'       -> Mode B (lowercase, punctuation, currency/url normalizer)
        'with_stopwords' -> Mode C (standard + stopword filtering)

    Returns
    -------
    str
        Preprocessed text.
    """
    if mode == "minimal":
        return str(text).strip()
    elif mode == "with_stopwords":
        return clean_text_with_stopwords(text)
    elif mode == "standard":
        return clean_text_standard(text)
    else:
        raise ValueError(f"Unknown preprocessing mode: '{mode}'. Choose 'minimal', 'standard', or 'with_stopwords'.")


def preprocess_corpus(corpus: List[str], mode: str = "standard") -> List[str]:
    """
    Preprocesses a list or Series of texts.

    Parameters
    ----------
    corpus : List[str]
        Collection of raw SMS messages.
    mode : str
        Preprocessing mode.

    Returns
    -------
    List[str]
        Preprocessed texts.
    """
    return [preprocess_text(msg, mode=mode) for msg in corpus]


if __name__ == "__main__":
    sample_texts = [
        "Congratulations! You won a £1,000 Walmart gift card. Go to http://claim-prize.com to claim NOW!",
        "Hey bro, are you coming to class today? Let me know.",
        "URGENT! Your account 39482 has been suspended. Call 0800-123456 immediately.",
    ]

    print("--- PREPROCESSING DEMO ---")
    for sample in sample_texts:
        print(f"\nOriginal:        {sample}")
        print(f"Mode A (Minimal):       {preprocess_text(sample, mode='minimal')}")
        print(f"Mode B (Standard):      {preprocess_text(sample, mode='standard')}")
        print(f"Mode C (With Stopwords): {preprocess_text(sample, mode='with_stopwords')}")
