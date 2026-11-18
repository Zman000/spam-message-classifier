import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from src.data_loader import load_data

def run_eda(output_dir=None):
    if output_dir is None:
        out_path = PROJECT_ROOT / "reports" / "figures"
    else:
        out_path = Path(output_dir)
        if not out_path.is_absolute():
            # If path like '../reports/figures' or 'reports/figures'
            if (Path.cwd() / out_path).parent.exists():
                out_path = Path.cwd() / out_path
            else:
                out_path = PROJECT_ROOT / out_path
    
    os.makedirs(out_path, exist_ok=True)
    df = load_data()
    
    # 1. Print dataset summary
    total = len(df)
    ham_cnt = int((df["label"] == "ham").sum())
    spam_cnt = int((df["label"] == "spam").sum())
    df["length"] = df["message"].apply(len)
    avg_ham_len = df[df["label"] == "ham"]["length"].mean()
    avg_spam_len = df[df["label"] == "spam"]["length"].mean()

    print(f"Total: {total} | Ham: {ham_cnt} ({ham_cnt/total*100:.1f}%) | Spam: {spam_cnt} ({spam_cnt/total*100:.1f}%)")
    print(f"Avg Length -> Ham: {avg_ham_len:.1f} chars | Spam: {avg_spam_len:.1f} chars")

    # 2. Plot class counts
    plt.figure(figsize=(5, 4))
    sns.countplot(data=df, x="label", hue="label", palette=["#2b5c8f", "#d95f02"], legend=False)
    plt.title("Message Count by Class")
    plt.tight_layout()
    plt.savefig(out_path / "01_class_counts.png", dpi=200)
    plt.close()

    # 3. Plot class percentage pie chart
    plt.figure(figsize=(4, 4))
    plt.pie([ham_cnt, spam_cnt], labels=["Ham", "Spam"], colors=["#2b5c8f", "#d95f02"], autopct="%1.1f%%")
    plt.title("Class Distribution")
    plt.tight_layout()
    plt.savefig(out_path / "02_class_percentage_distribution.png", dpi=200)
    plt.close()

    # 4. Message length distribution
    plt.figure(figsize=(7, 4))
    sns.histplot(data=df, x="length", hue="label", bins=50, palette=["#2b5c8f", "#d95f02"])
    plt.xlim(0, 300)
    plt.title("Message Length Distribution")
    plt.tight_layout()
    plt.savefig(out_path / "03_message_length_distribution.png", dpi=200)
    plt.close()

    # 5. Top words in spam & ham
    for label, filename in [("spam", "04_top_words_spam.png"), ("ham", "05_top_words_ham.png")]:
        words = []
        for text in df[df["label"] == label]["message"]:
            words.extend(re.findall(r"\b[a-zA-Z]{3,}\b", text.lower()))
        top_words = Counter(words).most_common(12)
        top_df = pd.DataFrame(top_words, columns=["Word", "Count"])

        plt.figure(figsize=(6, 4))
        sns.barplot(data=top_df, y="Word", x="Count", color="#2b5c8f" if label == "ham" else "#d95f02")
        plt.title(f"Top Words in {label.capitalize()} Messages")
        plt.tight_layout()
        plt.savefig(out_path / filename, dpi=200)
        plt.close()

    print(f"EDA figures generated successfully in {out_path}.")
    return df

if __name__ == "__main__":
    run_eda()
