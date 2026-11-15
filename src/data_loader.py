"""
data_loader.py
==============
Module for loading, inspecting, and cleaning the SMS Spam dataset.

Key Concepts:
- Binary Classification: Mapping text classes ('ham', 'spam') to numeric labels (0, 1).
- Data Leakage Prevention: Removing duplicate messages before train-test split
  so identical messages don't appear in both training and test sets.
"""

import os
from typing import Tuple, Dict, Any
import pandas as pd


def load_raw_data(data_path: str = None) -> pd.DataFrame:
    """
    Loads the SMS Spam Collection dataset from a CSV or tab-separated text file.
    
    Handles both:
    1. CSV files with columns ['label', 'message'] or ['v1', 'v2'].
    2. Original UCI tab-separated text file 'SMSSpamCollection'.
    
    Parameters
    ----------
    data_path : str, optional
        Path to the dataset file. If None, checks default locations in dataset/.
        
    Returns
    -------
    pd.DataFrame
        DataFrame containing 'label' and 'message' columns.
    """
    if data_path is None:
        # Check standard locations
        if os.path.exists(os.path.join("dataset", "spam.csv")):
            data_path = os.path.join("dataset", "spam.csv")
        elif os.path.exists(os.path.join("dataset", "SMSSpamCollection")):
            data_path = os.path.join("dataset", "SMSSpamCollection")
        else:
            raise FileNotFoundError(
                "SMS dataset not found in dataset/. Please place 'spam.csv' or "
                "'SMSSpamCollection' inside the dataset/ directory."
            )

    # Detect file type
    if data_path.endswith(".csv"):
        # Try utf-8 first, fallback to latin-1 (common with kaggle export of UCI)
        try:
            df = pd.read_csv(data_path, encoding="utf-8")
        except UnicodeDecodeError:
            df = pd.read_csv(data_path, encoding="latin-1")
    else:
        # UCI raw format: tab-separated without headers
        try:
            df = pd.read_csv(
                data_path,
                sep="\t",
                names=["label", "message"],
                quoting=3,
                encoding="utf-8",
            )
        except UnicodeDecodeError:
            df = pd.read_csv(
                data_path,
                sep="\t",
                names=["label", "message"],
                quoting=3,
                encoding="latin-1",
            )

    # Standardize column names
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "message"})
        # Drop extra columns that Kaggle CSVs sometimes contain
        df = df[["label", "message"]]
    elif "label" not in df.columns or "message" not in df.columns:
        # Fallback: assume first column is label, second is message
        cols = list(df.columns)
        df = df.rename(columns={cols[0]: "label", cols[1]: "message"})
        df = df[["label", "message"]]

    return df


def clean_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Cleans raw dataset and prepares it for machine learning:
    1. Handles missing values.
    2. Identifies and removes duplicate messages.
    3. Maps binary labels: 'ham' -> 0, 'spam' -> 1.
    4. Computes message length and statistics.

    Parameters
    ----------
    df : pd.DataFrame
        Raw DataFrame with 'label' and 'message' columns.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any]]
        Cleaned DataFrame and a summary dictionary of dataset statistics.
    """
    initial_count = len(df)

    # 1. Handle missing values
    df = df.dropna(subset=["label", "message"]).copy()
    null_count = initial_count - len(df)

    # Standardize labels to lowercase strings
    df["label"] = df["label"].astype(str).str.strip().str.lower()
    df["message"] = df["message"].astype(str).str.strip()

    # Filter to only valid 'ham' and 'spam'
    df = df[df["label"].isin(["ham", "spam"])].copy()

    # 2. Check and remove duplicate messages
    duplicate_count = df.duplicated(subset=["message"]).sum()
    df_clean = df.drop_duplicates(subset=["message"]).reset_index(drop=True)

    # 3. Label mapping: ham -> 0, spam -> 1
    # 0 = Negative class (Ham / Legitimate)
    # 1 = Positive class (Spam / Scam)
    label_mapping = {"ham": 0, "spam": 1}
    df_clean["label_num"] = df_clean["label"].map(label_mapping)

    # 4. Feature engineering for EDA: message length
    df_clean["char_length"] = df_clean["message"].apply(len)
    df_clean["word_count"] = df_clean["message"].apply(lambda x: len(x.split()))

    # Compute statistics
    total_messages = len(df_clean)
    ham_count = int((df_clean["label"] == "ham").sum())
    spam_count = int((df_clean["label"] == "spam").sum())

    ham_pct = (ham_count / total_messages) * 100
    spam_pct = (spam_count / total_messages) * 100

    avg_len_ham = float(df_clean[df_clean["label"] == "ham"]["char_length"].mean())
    avg_len_spam = float(df_clean[df_clean["label"] == "spam"]["char_length"].mean())

    stats = {
        "initial_rows": initial_count,
        "null_rows_removed": null_count,
        "duplicates_removed": int(duplicate_count),
        "total_clean_messages": total_messages,
        "ham_count": ham_count,
        "spam_count": spam_count,
        "ham_percentage": round(ham_pct, 2),
        "spam_percentage": round(spam_pct, 2),
        "avg_length_ham_chars": round(avg_len_ham, 2),
        "avg_length_spam_chars": round(avg_len_spam, 2),
    }

    return df_clean, stats


if __name__ == "__main__":
    print("Loading raw dataset...")
    raw_df = load_raw_data()
    print(f"Loaded raw dataset with {len(raw_df)} rows.")

    print("\nCleaning dataset...")
    clean_df, stats = clean_dataset(raw_df)

    print("\n--- DATASET STATISTICS ---")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    print("\nSample records:")
    print(clean_df[["label", "label_num", "message"]].head(5))
