"""
experiments.py
==============
Implementation of ablation studies and detailed empirical experiments:
- Experiment 2: Effect of Bag-of-Words vocabulary size (500, 1000, 3000, 5000 words).
- Experiment 3: Impact of text preprocessing levels (Minimal, Standard, With Stopwords).
- Experiment 4: Qualitative and quantitative misclassification (error) analysis.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_raw_data, clean_dataset
from src.feature_extraction import split_data, extract_bow_features
from src.evaluate import calculate_metrics

PALETTE = ["#2b5c8f", "#41b6c4", "#e7298a", "#d95f02"]


def run_vocab_size_experiment(
    vocab_sizes: List[int] = [500, 1000, 3000, 5000]
) -> pd.DataFrame:
    """
    Experiment 2: Evaluates the impact of vocabulary size on Multinomial Naive Bayes.
    
    Vocabulary determines the dimensionality of the Bag-of-Words feature space:
    - Too small: Informative rare spam tokens are dropped (underfitting).
    - Too large: Very rare words inflate sparsity without adding generalization value.
    """
    print("\n" + "=" * 60)
    print(" EXPERIMENT 2: VOCABULARY SIZE IMPACT (Multinomial Naive Bayes)")
    print("=" * 60)

    raw_df = load_raw_data()
    clean_df, _ = clean_dataset(raw_df)
    X_train, X_test, y_train, y_test = split_data(clean_df, test_size=0.20, random_state=42)

    results = []

    for size in vocab_sizes:
        print(f"Testing vocabulary size: {size} words...")
        # Fit CountVectorizer ONLY on X_train, transform X_test
        X_train_vec, X_test_vec, vec = extract_bow_features(
            X_train, X_test, max_features=size, preprocessing_mode="standard"
        )

        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        m = calculate_metrics(y_test, y_pred, model_name=f"Vocab {size}")
        m["Vocab Size"] = size
        m["Actual Features"] = X_train_vec.shape[1]
        results.append(m)

    df_results = pd.DataFrame(results)

    # Save CSV
    out_csv = os.path.join("reports", "results", "experiment_vocab_size.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"Saved vocabulary experiment results to: {out_csv}")

    # Plot line chart
    plt.figure(figsize=(9, 5))
    plt.plot(df_results["Vocab Size"], df_results["Accuracy"], marker="o", linewidth=2, label="Accuracy")
    plt.plot(df_results["Vocab Size"], df_results["Precision"], marker="s", linewidth=2, label="Precision")
    plt.plot(df_results["Vocab Size"], df_results["Recall"], marker="^", linewidth=2, label="Recall")
    plt.plot(df_results["Vocab Size"], df_results["F1-Score"], marker="d", linewidth=2.5, color="#d95f02", label="F1-Score")

    plt.title("Impact of Bag-of-Words Vocabulary Size on Classification (Multinomial NB)",
              fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Vocabulary Size (Max Features)", fontsize=11)
    plt.ylabel("Score", fontsize=11)
    plt.xticks(vocab_sizes)
    plt.ylim(0.85, 1.0)
    plt.legend(frameon=True)
    plt.tight_layout()

    out_fig = os.path.join("reports", "figures", "07_vocab_size_comparison.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved vocabulary comparison plot to: {out_fig}")

    return df_results


def run_preprocessing_experiment() -> pd.DataFrame:
    """
    Experiment 3: Evaluates the impact of three preprocessing strategies:
    A. 'minimal': Raw stripped text.
    B. 'standard': Lowercase, punctuation cleaning, currency & URL tokens.
    C. 'with_stopwords': Standard cleaning + English stopword removal.
    
    Classifier: Fixed Multinomial Naive Bayes with max_features=3000.
    """
    print("\n" + "=" * 60)
    print(" EXPERIMENT 3: TEXT PREPROCESSING STRATEGIES")
    print("=" * 60)

    raw_df = load_raw_data()
    clean_df, _ = clean_dataset(raw_df)
    X_train, X_test, y_train, y_test = split_data(clean_df, test_size=0.20, random_state=42)

    configs = [
        ("A: Minimal (Raw)", "minimal"),
        ("B: Standard (Punct/Case/Tokens)", "standard"),
        ("C: With Stopwords Removed", "with_stopwords"),
    ]

    results = []

    for label, mode in configs:
        print(f"Testing Preprocessing Strategy [{label}]...")
        X_train_vec, X_test_vec, _ = extract_bow_features(
            X_train, X_test, max_features=3000, preprocessing_mode=mode
        )

        model = MultinomialNB()
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        m = calculate_metrics(y_test, y_pred, model_name=label)
        m["Strategy"] = label
        results.append(m)

    df_results = pd.DataFrame(results)

    # Save CSV
    out_csv = os.path.join("reports", "results", "experiment_preprocessing.csv")
    df_results.to_csv(out_csv, index=False)
    print(f"Saved preprocessing experiment results to: {out_csv}")

    # Plot bar comparison
    plt.figure(figsize=(9, 5))
    melted = df_results.melt(
        id_vars=["Strategy"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score"
    )

    sns.barplot(
        data=melted,
        x="Strategy",
        y="Score",
        hue="Metric",
        palette="crest"
    )
    plt.title("Effect of Preprocessing Pipeline on Classification Performance",
              fontsize=13, fontweight="bold", pad=12)
    plt.ylim(0.85, 1.02)
    plt.ylabel("Score", fontsize=11)
    plt.xlabel("Preprocessing Strategy", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    out_fig = os.path.join("reports", "figures", "08_preprocessing_comparison.png")
    plt.savefig(out_fig, dpi=300)
    plt.close()
    print(f"Saved preprocessing comparison plot to: {out_fig}")

    return df_results


def run_misclassification_analysis() -> pd.DataFrame:
    """
    Experiment 4: In-depth error analysis on the best performing model (Multinomial NB).
    Identifies and analyzes:
    - False Positives (Ham messages mistakenly classified as Spam)
    - False Negatives (Spam messages mistakenly classified as Ham)
    """
    print("\n" + "=" * 60)
    print(" EXPERIMENT 4: MISCLASSIFICATION & ERROR ANALYSIS")
    print("=" * 60)

    raw_df = load_raw_data()
    clean_df, _ = clean_dataset(raw_df)
    X_train, X_test, y_train, y_test = split_data(clean_df, test_size=0.20, random_state=42)

    X_train_vec, X_test_vec, vec = extract_bow_features(
        X_train, X_test, max_features=3000, preprocessing_mode="standard"
    )

    model = MultinomialNB()
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    y_proba = model.predict_proba(X_test_vec)[:, 1]  # Probability of being spam

    # Build evaluation DataFrame from test set
    test_eval_df = pd.DataFrame({
        "message": X_test.values,
        "actual_label": ["spam" if y == 1 else "ham" for y in y_test],
        "actual_num": y_test.values,
        "predicted_label": ["spam" if p == 1 else "ham" for p in y_pred],
        "predicted_num": y_pred,
        "spam_probability": np.round(y_proba, 4)
    })

    # Filter False Positives & False Negatives
    false_positives = test_eval_df[
        (test_eval_df["actual_num"] == 0) & (test_eval_df["predicted_num"] == 1)
    ].copy()
    false_positives["error_type"] = "False Positive (Ham as Spam)"

    false_negatives = test_eval_df[
        (test_eval_df["actual_num"] == 1) & (test_eval_df["predicted_num"] == 0)
    ].copy()
    false_negatives["error_type"] = "False Negative (Spam as Ham)"

    all_errors = pd.concat([false_positives, false_negatives], ignore_index=True)

    # Save to CSV
    out_csv = os.path.join("reports", "results", "misclassifications.csv")
    all_errors.to_csv(out_csv, index=False)
    print(f"Identified {len(false_positives)} False Positives and {len(false_negatives)} False Negatives.")
    print(f"Saved misclassified samples to: {out_csv}")

    print("\n--- SAMPLE FALSE POSITIVES (Legitimate Ham flagged as Spam) ---")
    for idx, row in false_positives.head(5).iterrows():
        print(f"[{idx+1}] Prob: {row['spam_probability']:.4f} | Text: \"{row['message']}\"")

    print("\n--- SAMPLE FALSE NEGATIVES (Spam messages missed by model) ---")
    for idx, row in false_negatives.head(5).iterrows():
        print(f"[{idx+1}] Prob: {row['spam_probability']:.4f} | Text: \"{row['message']}\"")

    return all_errors


if __name__ == "__main__":
    v_df = run_vocab_size_experiment()
    print("\nVocabulary Size Results:\n", v_df[["Vocab Size", "Accuracy", "Precision", "Recall", "F1-Score"]])

    p_df = run_preprocessing_experiment()
    print("\nPreprocessing Results:\n", p_df[["Strategy", "Accuracy", "Precision", "Recall", "F1-Score"]])

    e_df = run_misclassification_analysis()
