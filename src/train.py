import os
import sys
sys.path.insert(0, ".")

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

from src.data_loader import load_data
from src.preprocessing import clean_text
from src.feature_extraction import split_data, extract_bow_features
from src.evaluate import calculate_metrics, plot_confusion_matrix

def train_models():
    # 1. Load dataset
    df = load_data()
    X_train, X_test, y_train, y_test = split_data(df)

    # 2. Extract Bag-of-Words features
    X_train_vec, X_test_vec, _ = extract_bow_features(X_train, X_test, max_features=3000)

    # 3. Define the 4 classification algorithms
    models = {
        "Multinomial Naive Bayes": MultinomialNB(),
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=30, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=30, random_state=42, n_jobs=-1)
    }

    results = []
    os.makedirs("reports/figures", exist_ok=True)
    os.makedirs("reports/results", exist_ok=True)
    os.makedirs("models", exist_ok=True)

    # 4. Train and evaluate each model
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        y_pred = model.predict(X_test_vec)

        metrics = calculate_metrics(y_test, y_pred, model_name=name)
        results.append(metrics)

        # Save confusion matrix plot
        slug = name.lower().replace(" ", "_")
        plot_confusion_matrix(y_test, y_pred, name, save_path=f"reports/figures/cm_{slug}.png")

    results_df = pd.DataFrame(results)
    results_df.to_csv("reports/results/model_comparison.csv", index=False)
    print("\n--- Model Evaluation Results ---")
    print(results_df.to_string(index=False))

    # 5. Build and save the best pipeline (CountVectorizer + MultinomialNB)
    best_pipeline = Pipeline([
        ("vectorizer", CountVectorizer(preprocessor=clean_text, max_features=3000)),
        ("classifier", MultinomialNB())
    ])
    best_pipeline.fit(X_train, y_train)
    joblib.dump(best_pipeline, "models/best_model.pkl")
    print("\nSaved best model pipeline to models/best_model.pkl")

    # 6. Save comparison bar chart
    plt.figure(figsize=(8, 4))
    sns.barplot(data=results_df, x="Model", y="F1-Score", color="#2b5c8f")
    plt.title("Model Comparison (F1-Score)")
    plt.ylim(0.7, 1.0)
    plt.tight_layout()
    plt.savefig("reports/figures/06_model_comparison.png", dpi=200)
    plt.close()

    return results_df

# Alias for notebook compatibility
def train_and_evaluate_all():
    comp_df = train_models()
    return comp_df, None, None

if __name__ == "__main__":
    train_models()
