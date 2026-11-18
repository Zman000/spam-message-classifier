import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB

from src.data_loader import load_data
from src.feature_extraction import split_data, extract_bow_features
from src.evaluate import calculate_metrics

fig_dir = PROJECT_ROOT / "reports" / "figures"
res_dir = PROJECT_ROOT / "reports" / "results"
os.makedirs(fig_dir, exist_ok=True)
os.makedirs(res_dir, exist_ok=True)

def run_vocab_size_experiment():
    """Experiment 2: Effect of vocabulary size on Multinomial Naive Bayes."""
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    vocab_sizes = [500, 1000, 3000, 5000]
    records = []

    for size in vocab_sizes:
        X_tr, X_te, _ = extract_bow_features(X_train, X_test, max_features=size)
        nb = MultinomialNB()
        nb.fit(X_tr, y_train)
        y_pred = nb.predict(X_te)
        
        m = calculate_metrics(y_test, y_pred, model_name=f"Vocab {size}")
        m["Vocab Size"] = size
        records.append(m)

    res_df = pd.DataFrame(records)
    res_df.to_csv(res_dir / "experiment_vocab_size.csv", index=False)

    plt.figure(figsize=(7, 4))
    plt.plot(res_df["Vocab Size"], res_df["F1-Score"], marker="o", color="#2b5c8f")
    plt.title("Vocabulary Size vs F1-Score")
    plt.xlabel("Vocabulary Size")
    plt.ylabel("F1-Score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fig_dir / "07_vocab_size_comparison.png", dpi=200)
    plt.close()
    return res_df

def run_preprocessing_experiment():
    """Experiment 3: Compare raw, cleaned, and stopword-filtered text."""
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    modes = [("Minimal", "minimal"), ("Standard", "standard"), ("Stopwords Removed", "with_stopwords")]
    records = []

    for label, mode in modes:
        X_tr, X_te, _ = extract_bow_features(X_train, X_test, max_features=3000, preprocessing_mode=mode)
        nb = MultinomialNB()
        nb.fit(X_tr, y_train)
        y_pred = nb.predict(X_te)
        
        m = calculate_metrics(y_test, y_pred, model_name=label)
        m["Strategy"] = label
        records.append(m)

    res_df = pd.DataFrame(records)
    res_df.to_csv(res_dir / "experiment_preprocessing.csv", index=False)

    plt.figure(figsize=(7, 4))
    sns.barplot(data=res_df, x="Strategy", y="F1-Score", color="#2b5c8f")
    plt.title("Preprocessing Strategy vs F1-Score")
    plt.ylim(0.85, 1.0)
    plt.tight_layout()
    plt.savefig(fig_dir / "08_preprocessing_comparison.png", dpi=200)
    plt.close()
    return res_df

def run_misclassification_analysis():
    """Experiment 4: Identify false positives and false negatives."""
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    X_tr, X_te, _ = extract_bow_features(X_train, X_test, max_features=3000)
    nb = MultinomialNB()
    nb.fit(X_tr, y_train)
    y_pred = nb.predict(X_te)

    analysis_df = pd.DataFrame({
        "message": X_test.values,
        "actual": y_test.values,
        "predicted": y_pred
    })
    errors = analysis_df[analysis_df["actual"] != analysis_df["predicted"]].copy()
    errors["error_type"] = errors.apply(
        lambda r: "False Positive (Ham as Spam)" if r["actual"] == 0 else "False Negative (Spam as Ham)", axis=1
    )
    errors.to_csv(res_dir / "misclassifications.csv", index=False)
    print(f"Total misclassifications: {len(errors)}")
    return errors

if __name__ == "__main__":
    print("\nRunning Vocab Size Experiment...")
    print(run_vocab_size_experiment())
    print("\nRunning Preprocessing Experiment...")
    print(run_preprocessing_experiment())
    print("\nRunning Error Analysis...")
    print(run_misclassification_analysis().head())
