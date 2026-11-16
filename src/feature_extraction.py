import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from src.preprocessing import preprocess_corpus

def split_data(df, test_size=0.2, random_state=42):
    """80:20 stratified split."""
    return train_test_split(
        df["message"],
        df["label_num"],
        test_size=test_size,
        random_state=random_state,
        stratify=df["label_num"]
    )

def extract_bow_features(X_train, X_test, max_features=3000, preprocessing_mode="standard"):
    """Fit CountVectorizer on train data only, then transform test data."""
    X_train_clean = preprocess_corpus(X_train, mode=preprocessing_mode)
    X_test_clean = preprocess_corpus(X_test, mode=preprocessing_mode)

    vectorizer = CountVectorizer(max_features=max_features, lowercase=True)
    X_train_vec = vectorizer.fit_transform(X_train_clean)
    X_test_vec = vectorizer.transform(X_test_clean)

    return X_train_vec, X_test_vec, vectorizer
