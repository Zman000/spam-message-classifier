"""
train.py
========
Model training, evaluation, cross-validation, and comparison pipeline.

Algorithms implemented:
1. Multinomial Naive Bayes:
   - Probabilistic classifier based on Bayes' theorem with feature independence assumption.
   - High speed, highly effective for word-count features in text classification.
2. Logistic Regression:
   - Linear model that outputs posterior probabilities using the sigmoid function.
   - Learns positive and negative feature weights for individual vocabulary words.
3. Decision Tree:
   - Non-linear rule-based tree splitting on word occurrences (bounded depth=30 to curb overfitting).
4. Random Forest:
   - Ensemble of 100 decorrelated decision trees using bagging and feature subsampling.
   - Reduces decision tree variance and enhances generalization.
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.data_loader import load_raw_data, clean_dataset
from src.preprocessing import clean_text_standard
from src.feature_extraction import split_data, extract_bow_features
from src.evaluate import calculate_metrics, print_evaluation_report, plot_confusion_matrix


def get_default_models() -> Dict[str, Any]:
    """
    Returns standard instances of all 4 supervised classifiers.
    Hyperparameters follow coursework specifications.
    """
    return {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            random_state=42
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=30,
            random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=100,
            max_depth=30,
            random_state=42,
            n_jobs=-1
        ),
    }


def train_and_evaluate_all(
    models: Dict[str, Any] = None,
    max_features: int = 3000,
    preprocessing_mode: str = "standard",
    run_cv: bool = True
) -> Tuple[pd.DataFrame, Dict[str, Any], Any]:
    """
    Trains all models on identical Bag-of-Words features, evaluates them on
    the unseen test set, generates plots, and records performance.

    Parameters
    ----------
    models : Dict[str, Any], optional
        Dictionary mapping model names to sklearn estimator instances.
    max_features : int, default 3000
        Maximum vocabulary size for CountVectorizer.
    preprocessing_mode : str, default 'standard'
        Text preprocessing pipeline mode.
    run_cv : bool, default True
        Whether to perform 5-fold Stratified Cross Validation on training data.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, Any], Any]
        Comparison DataFrame, trained models dict, and the fitted vectorizer.
    """
    if models is None:
        models = get_default_models()

    # 1. Load and clean data
    raw_df = load_raw_data()
    clean_df, stats = clean_dataset(raw_df)

    # 2. Stratified train-test split (80:20)
    X_train, X_test, y_train, y_test = split_data(clean_df, test_size=0.20, random_state=42)

    # 3. Fit BoW ONLY on training data, transform test data
    X_train_vec, X_test_vec, vectorizer = extract_bow_features(
        X_train, X_test, max_features=max_features, preprocessing_mode=preprocessing_mode
    )

    results_list = []
    trained_models = {}

    figures_dir = os.path.join("reports", "figures")
    results_dir = os.path.join("reports", "results")
    models_dir = os.path.join("models")
    os.makedirs(figures_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Save vectorizer
    vec_path = os.path.join(models_dir, "count_vectorizer.pkl")
    joblib.dump(vectorizer, vec_path)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for name, model in models.items():
        print(f"\nTraining [{name}]...")
        model.fit(X_train_vec, y_train)
        trained_models[name] = model

        # Save individual model checkpoint
        slug = name.lower().replace(" ", "_")
        model_save_path = os.path.join(models_dir, f"model_{slug}.pkl")
        joblib.dump(model, model_save_path)

        # Predict on unseen test data
        y_pred = model.predict(X_test_vec)

        # Print detailed report
        print_evaluation_report(y_test, y_pred, name)

        # Calculate metrics
        metrics = calculate_metrics(y_test, y_pred, model_name=name)

        # Stratified 5-Fold Cross-Validation on training data (F1-score for Spam)
        if run_cv:
            cv_scores = cross_val_score(model, X_train_vec, y_train, cv=cv, scoring="f1")
            metrics["CV F1 (Mean)"] = round(float(cv_scores.mean()), 4)
            metrics["CV F1 (Std)"] = round(float(cv_scores.std()), 4)
            print(f"5-Fold CV F1 Score: {metrics['CV F1 (Mean)']:.4f} (+/- {metrics['CV F1 (Std)']:.4f})")

        results_list.append(metrics)

        # Plot confusion matrix heatmap
        cm = confusion_matrix(y_test, y_pred)
        cm_path = os.path.join(figures_dir, f"cm_{slug}.png")
        plot_confusion_matrix(cm, name, output_path=cm_path)
        print(f"Saved confusion matrix: {cm_path}")

    # Build comparison DataFrame
    comparison_df = pd.DataFrame(results_list)

    # Automatically select best model based on highest F1-Score
    best_row = comparison_df.sort_values(by="F1-Score", ascending=False).iloc[0]
    best_model_name = best_row["Model"]
    print(f"\n---> BEST MODEL SELECTED: [{best_model_name}] with Test F1-Score: {best_row['F1-Score']:.4f}")

    # Build and serialize a production Scikit-learn Pipeline
    # Encapsulating both CountVectorizer (with standard preprocessing) and the best classifier
    best_classifier_instance = trained_models[best_model_name]
    best_pipeline = Pipeline([
        ("vectorizer", CountVectorizer(
            preprocessor=clean_text_standard,
            max_features=max_features,
            token_pattern=r"(?u)\b\w+\b"
        )),
        ("classifier", best_classifier_instance.__class__(**best_classifier_instance.get_params()))
    ])

    # Fit complete pipeline on raw training texts (ensures identical behavior during inference)
    best_pipeline.fit(X_train, y_train)

    best_pipeline_path = os.path.join(models_dir, "best_model.pkl")
    joblib.dump(best_pipeline, best_pipeline_path)
    print(f"Saved complete end-to-end pipeline to: {best_pipeline_path}")

    # Save results to CSV
    csv_path = os.path.join(results_dir, "model_comparison.csv")
    comparison_df.to_csv(csv_path, index=False)
    print(f"Saved model comparison table to: {csv_path}")

    # -------------------------------------------------------------
    # Plot Model Performance Comparison Chart
    # -------------------------------------------------------------
    plt.figure(figsize=(10, 6))
    melted_df = comparison_df.melt(
        id_vars=["Model"],
        value_vars=["Accuracy", "Precision", "Recall", "F1-Score"],
        var_name="Metric",
        value_name="Score"
    )

    sns.barplot(
        data=melted_df,
        x="Model",
        y="Score",
        hue="Metric",
        palette="viridis"
    )
    plt.title("Supervised Classification Models: Performance Comparison on Test Set",
              fontsize=13, fontweight="bold", pad=12)
    plt.ylim(0.80, 1.02)
    plt.ylabel("Score", fontsize=11)
    plt.xlabel("Algorithm", fontsize=11)
    plt.legend(loc="lower right", frameon=True)
    plt.tight_layout()

    chart_path = os.path.join(figures_dir, "06_model_comparison.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"Saved comparison chart: {chart_path}")

    return comparison_df, trained_models, vectorizer


if __name__ == "__main__":
    print("=" * 60)
    print("      EXPERIMENT 1: MODEL TRAINING & COMPARISON")
    print("=" * 60)
    comp_df, models, vec = train_and_evaluate_all()
    print("\nFINAL MODEL COMPARISON SUMMARY:")
    display_cols = ["Model", "Accuracy", "Precision", "Recall", "F1-Score",
                    "False Positives", "False Negatives", "CV F1 (Mean)"]
    print(comp_df[display_cols].to_string(index=False))
