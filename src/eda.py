"""
eda.py
======
Module for Exploratory Data Analysis (EDA) of the SMS Spam dataset.

Generates and saves visual reports:
1. Class frequency count plot.
2. Class percentage distribution.
3. Message length distributions by class.
4. Top frequent words in spam messages.
5. Top frequent words in legitimate messages.
"""

import os
import sys
from pathlib import Path
from collections import Counter
import re

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

from src.data_loader import load_raw_data, clean_dataset

# Set style for professional academic presentation
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
PALETTE = {"ham": "#2b5c8f", "spam": "#d95f02"}


def run_eda(output_dir: str = os.path.join("reports", "figures")) -> pd.DataFrame:
    """
    Performs full exploratory data analysis and exports visualizations.
    
    Parameters
    ----------
    output_dir : str
        Directory where figures will be saved.
        
    Returns
    -------
    pd.DataFrame
        The cleaned dataset.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load and clean
    raw_df = load_raw_data()
    clean_df, stats = clean_dataset(raw_df)
    
    print("\n" + "=" * 50)
    print("        EXPLORATORY DATA ANALYSIS SUMMARY")
    print("=" * 50)
    print(f"Total clean messages:        {stats['total_clean_messages']}")
    print(f"Legitimate messages (Ham):   {stats['ham_count']} ({stats['ham_percentage']}%)")
    print(f"Spam / Scam messages:        {stats['spam_count']} ({stats['spam_percentage']}%)")
    print(f"Avg length (Ham):            {stats['avg_length_ham_chars']} characters")
    print(f"Avg length (Spam):           {stats['avg_length_spam_chars']} characters")
    print("=" * 50 + "\n")

    # -------------------------------------------------------------
    # FIGURE 1: Spam vs Legitimate Count Plot
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(
        data=clean_df,
        x="label",
        order=["ham", "spam"],
        hue="label",
        palette=PALETTE,
        legend=False
    )
    plt.title("Class Frequency: Legitimate (Ham) vs Spam Messages", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Class", fontsize=11)
    plt.ylabel("Number of Messages", fontsize=11)
    
    # Add count labels on top of bars
    for p in ax.patches:
        height = p.get_height()
        ax.annotate(f"{int(height)}",
                    (p.get_x() + p.get_width() / 2., height),
                    ha="center", va="bottom", fontsize=10, xytext=(0, 4),
                    textcoords="offset points", fontweight="bold")
    
    plt.tight_layout()
    fig1_path = os.path.join(output_dir, "01_class_counts.png")
    plt.savefig(fig1_path, dpi=300)
    plt.close()
    print(f"Saved: {fig1_path}")

    # -------------------------------------------------------------
    # FIGURE 2: Percentage Distribution (Donut Chart)
    # -------------------------------------------------------------
    plt.figure(figsize=(6, 6))
    counts = [stats["ham_count"], stats["spam_count"]]
    labels = [f"Ham ({stats['ham_percentage']}%)", f"Spam ({stats['spam_percentage']}%)"]
    colors = [PALETTE["ham"], PALETTE["spam"]]
    
    wedges, texts, autotexts = plt.pie(
        counts,
        labels=labels,
        colors=colors,
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        textprops={"fontsize": 11},
        wedgeprops=dict(width=0.45, edgecolor="white", linewidth=2)
    )
    for autotext in autotexts:
        autotext.set_color("white")
        autotext.set_weight("bold")
        
    plt.title("Class Imbalance: Percentage Distribution", fontsize=13, fontweight="bold", pad=12)
    plt.tight_layout()
    fig2_path = os.path.join(output_dir, "02_class_percentage_distribution.png")
    plt.savefig(fig2_path, dpi=300)
    plt.close()
    print(f"Saved: {fig2_path}")

    # -------------------------------------------------------------
    # FIGURE 3: Message Length Distribution
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5))
    sns.histplot(
        data=clean_df,
        x="char_length",
        hue="label",
        palette=PALETTE,
        bins=60,
        kde=True,
        stat="density",
        common_norm=False,
        alpha=0.5
    )
    plt.xlim(0, 300)
    plt.axvline(stats["avg_length_ham_chars"], color=PALETTE["ham"], linestyle="--", linewidth=1.5,
                label=f"Avg Ham ({stats['avg_length_ham_chars']} chars)")
    plt.axvline(stats["avg_length_spam_chars"], color=PALETTE["spam"], linestyle="--", linewidth=1.5,
                label=f"Avg Spam ({stats['avg_length_spam_chars']} chars)")
    plt.title("Distribution of Message Length (Characters): Ham vs Spam", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Message Length (Characters)", fontsize=11)
    plt.ylabel("Density", fontsize=11)
    plt.legend(title="Message Class", frameon=True)
    plt.tight_layout()
    fig3_path = os.path.join(output_dir, "03_message_length_distribution.png")
    plt.savefig(fig3_path, dpi=300)
    plt.close()
    print(f"Saved: {fig3_path}")

    # Helper function to extract basic word frequencies (simple tokenization)
    def get_top_words(texts, n=15):
        # Basic lowercase word tokenization (retaining alphanumeric tokens)
        words = []
        for text in texts:
            tokens = re.findall(r"\b[a-zA-Z0-9$£€]+\b", text.lower())
            words.extend(tokens)
        return Counter(words).most_common(n)

    # -------------------------------------------------------------
    # FIGURE 4: Most Frequent Words in Spam
    # -------------------------------------------------------------
    spam_words = get_top_words(clean_df[clean_df["label"] == "spam"]["message"], n=15)
    spam_word_df = pd.DataFrame(spam_words, columns=["Word", "Frequency"])

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=spam_word_df,
        y="Word",
        x="Frequency",
        color=PALETTE["spam"]
    )
    plt.title("Top 15 Most Frequent Tokens in Spam Messages", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Frequency Count", fontsize=11)
    plt.ylabel("Token", fontsize=11)
    
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{int(width)}",
                    (width, p.get_y() + p.get_height() / 2.),
                    ha="left", va="center", fontsize=9, xytext=(4, 0),
                    textcoords="offset points")
                    
    plt.tight_layout()
    fig4_path = os.path.join(output_dir, "04_top_words_spam.png")
    plt.savefig(fig4_path, dpi=300)
    plt.close()
    print(f"Saved: {fig4_path}")

    # -------------------------------------------------------------
    # FIGURE 5: Most Frequent Words in Ham (Legitimate)
    # -------------------------------------------------------------
    ham_words = get_top_words(clean_df[clean_df["label"] == "ham"]["message"], n=15)
    ham_word_df = pd.DataFrame(ham_words, columns=["Word", "Frequency"])

    plt.figure(figsize=(9, 5))
    ax = sns.barplot(
        data=ham_word_df,
        y="Word",
        x="Frequency",
        color=PALETTE["ham"]
    )
    plt.title("Top 15 Most Frequent Tokens in Legitimate (Ham) Messages", fontsize=13, fontweight="bold", pad=12)
    plt.xlabel("Frequency Count", fontsize=11)
    plt.ylabel("Token", fontsize=11)
    
    for p in ax.patches:
        width = p.get_width()
        ax.annotate(f"{int(width)}",
                    (width, p.get_y() + p.get_height() / 2.),
                    ha="left", va="center", fontsize=9, xytext=(4, 0),
                    textcoords="offset points")

    plt.tight_layout()
    fig5_path = os.path.join(output_dir, "05_top_words_ham.png")
    plt.savefig(fig5_path, dpi=300)
    plt.close()
    print(f"Saved: {fig5_path}")

    print("\nEDA completed. All figures saved in", output_dir)
    return clean_df


if __name__ == "__main__":
    run_eda()
