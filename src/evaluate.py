"""
evaluate.py
===========
Module for evaluating supervised classification models.

Key Concepts:
- Binary Classification Metrics:
    - Accuracy: Total correct predictions / Total samples. Can be misleading with imbalanced data!
    - Precision: Of all messages predicted as Spam, how many were actually Spam?
      High precision = Few False Positives (legitimate emails won't be wrongly marked as spam).
    - Recall (Sensitivity): Of all actual Spam messages, how many did the model catch?
      High recall = Few False Negatives (spam won't leak into the inbox).
    - F1-Score: Harmonic mean of Precision and Recall. Balances both, especially under class imbalance.
    - Confusion Matrix:
        [ [True Negatives (Ham as Ham),  False Positives (Ham as Spam)],
          [False Negatives (Spam as Ham), True Positives (Spam as Spam)] ]
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any, Tuple
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def calculate_metrics(y_true, y_pred, model_name: str = "Model") -> Dict[str, Any]:
    """
    Computes standard binary classification metrics treating Spam (1) as the positive class.

    Parameters
    ----------
    y_true : array-like
        True binary labels (0 or 1).
    y_pred : array-like
        Predicted binary labels (0 or 1).
    model_name : str
        Human-readable name of the model.

    Returns
    -------
    Dict[str, Any]
        Dictionary of metrics formatted for reporting.
    """
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    rec = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    f1 = f1_score(y_true, y_pred, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    tn, fp, fn, tp = cm.ravel()

    return {
        "Model": model_name,
        "Accuracy": round(acc, 4),
        "Precision": round(prec, 4),
        "Recall": round(rec, 4),
        "F1-Score": round(f1, 4),
        "True Negatives": int(tn),
        "False Positives": int(fp),
        "False Negatives": int(fn),
        "True Positives": int(tp),
    }


def print_evaluation_report(y_true, y_pred, model_name: str):
    """
    Prints a detailed classification report and confusion matrix breakdown.
    """
    print("\n" + "=" * 60)
    print(f" EVALUATION REPORT: {model_name}")
    print("=" * 60)
    
    report = classification_report(
        y_true, y_pred, target_names=["Ham (0)", "Spam (1)"], digits=4
    )
    print(report)
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix Breakdown:")
    print(f"  True Negatives  (Ham correctly identified):  {tn}")
    print(f"  False Positives (Ham mistaken for Spam):      {fp}  <-- Risk of missing important messages")
    print(f"  False Negatives (Spam mistaken for Ham):      {fn}  <-- Inconvenience of spam in inbox")
    print(f"  True Positives  (Spam correctly caught):     {tp}")
    print("=" * 60)


def plot_confusion_matrix(
    cm: np.ndarray,
    model_name: str,
    output_path: str = None
):
    """
    Plots and saves a professional heatmap for a model's confusion matrix.
    """
    plt.figure(figsize=(6, 5))
    labels = ["Ham (0)", "Spam (1)"]
    
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        cbar=False,
        annot_kws={"size": 13, "weight": "bold"}
    )
    plt.title(f"Confusion Matrix: {model_name}", fontsize=12, fontweight="bold", pad=12)
    plt.xlabel("Predicted Label", fontsize=11)
    plt.ylabel("Actual Label", fontsize=11)
    plt.tight_layout()

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=300)
        plt.close()
    else:
        plt.show()
