"""
feature_extraction.py
=====================
Bag-of-Words (BoW) feature extraction using Scikit-Learn's CountVectorizer.

Key Concepts:
- Bag of Words (BoW): Represents text as a collection (or multiset) of words,
  disregarding grammar and word order but keeping track of token frequency.
- Vocabulary Construction: The set of unique words discovered in the training corpus.
- max_features: Restricts vocabulary to the top-N most frequent words across the corpus,
  reducing dimensionality and preventing sparse, rare tokens from inflating feature space.
- Sparse Matrix Representation: Since any single SMS message contains only a few dozen words
  out of thousands of vocabulary words, most entries in the matrix are 0. Storing this as a
  SciPy CSR sparse matrix saves memory and accelerates linear algebra computations.
- Strict Train/Test Separation (Preventing Data Leakage):
  The vectorizer must ALWAYS be fitted ONLY on the training data (`fit_transform`).
  The test data must ONLY be transformed (`transform`), ensuring the model never sees
  test vocabulary or word statistics during training.
"""

import sys
from pathlib import Path
from typing import Tuple
import pandas as pd
from scipy.sparse import spmatrix
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.preprocessing import preprocess_corpus


def split_data(
    df: pd.DataFrame,
    test_size: float = 0.20,
    random_state: int = 42
) -> Tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits dataset into stratified train and test partitions (80:20).

    Stratification ensures both splits have the exact same proportion of
    spam vs ham messages as the original dataset.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataframe with 'message' and 'label_num' columns.
    test_size : float, default 0.20
        Proportion of samples reserved for testing.
    random_state : int, default 42
        Seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : Tuple[pd.Series, pd.Series, pd.Series, pd.Series]
    """
    X = df["message"]
    y = df["label_num"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    return X_train, X_test, y_train, y_test


def build_vectorizer(
    max_features: int = 3000,
    stop_words: str = None
) -> CountVectorizer:
    """
    Constructs a Scikit-Learn CountVectorizer configured for Bag of Words.

    Parameters
    ----------
    max_features : int, default 3000
        Maximum size of vocabulary.
    stop_words : str or list, optional
        Stop words list, or None.

    Returns
    -------
    CountVectorizer
        Configured, unfitted vectorizer.
    """
    return CountVectorizer(
        max_features=max_features,
        stop_words=stop_words,
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b"
    )


def extract_bow_features(
    X_train_raw: pd.Series,
    X_test_raw: pd.Series,
    max_features: int = 3000,
    preprocessing_mode: str = "standard"
) -> Tuple[spmatrix, spmatrix, CountVectorizer]:
    """
    Applies text preprocessing and extracts Bag-of-Words features.
    
    CRITICAL: Fits vectorizer ONLY on X_train. Transforms X_test using the fitted vectorizer.

    Parameters
    ----------
    X_train_raw : pd.Series
        Raw training SMS texts.
    X_test_raw : pd.Series
        Raw testing SMS texts.
    max_features : int, default 3000
        Maximum vocabulary size.
    preprocessing_mode : str, default 'standard'
        Mode passed to preprocess_corpus ('minimal', 'standard', 'with_stopwords').

    Returns
    -------
    X_train_vec : spmatrix
        Sparse document-term matrix for training set.
    X_test_vec : spmatrix
        Sparse document-term matrix for test set.
    vectorizer : CountVectorizer
        The fitted vectorizer instance.
    """
    # 1. Preprocess texts
    X_train_clean = preprocess_corpus(X_train_raw.tolist(), mode=preprocessing_mode)
    X_test_clean = preprocess_corpus(X_test_raw.tolist(), mode=preprocessing_mode)

    # 2. Build CountVectorizer
    vectorizer = build_vectorizer(max_features=max_features)

    # 3. Fit ONLY on training data, then transform training data
    X_train_vec = vectorizer.fit_transform(X_train_clean)

    # 4. Transform test data using vocabulary learned from training data
    X_test_vec = vectorizer.transform(X_test_clean)

    return X_train_vec, X_test_vec, vectorizer


if __name__ == "__main__":
    from src.data_loader import load_raw_data, clean_dataset

    print("Loading data...")
    raw_df = load_raw_data()
    clean_df, stats = clean_dataset(raw_df)

    print(f"Total clean records: {len(clean_df)}")
    X_train, X_test, y_train, y_test = split_data(clean_df, test_size=0.20, random_state=42)

    print(f"Train samples: {len(X_train)} (Ham: {(y_train == 0).sum()}, Spam: {(y_train == 1).sum()})")
    print(f"Test samples:  {len(X_test)} (Ham: {(y_test == 0).sum()}, Spam: {(y_test == 1).sum()})")

    X_train_vec, X_test_vec, vec = extract_bow_features(
        X_train, X_test, max_features=3000, preprocessing_mode="standard"
    )

    print("\n--- FEATURE EXTRACTION SUMMARY ---")
    print(f"Vocabulary size:         {len(vec.vocabulary_)}")
    print(f"X_train_vec shape:       {X_train_vec.shape} (Sparse matrix: {type(X_train_vec).__name__})")
    print(f"X_test_vec shape:        {X_test_vec.shape}")
    sparsity = 100.0 * (1 - X_train_vec.nnz / (X_train_vec.shape[0] * X_train_vec.shape[1]))
    print(f"Matrix Sparsity:         {sparsity:.2f}% zeros")
    print(f"Sample vocabulary words: {list(vec.vocabulary_.keys())[:15]}")
