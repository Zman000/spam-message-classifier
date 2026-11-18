import os
from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def load_data(filepath=None):
    """Load the dataset and return a cleaned dataframe."""
    if filepath is None:
        candidates = [
            Path("dataset/spam.csv"),
            Path("dataset/SMSSpamCollection"),
            PROJECT_ROOT / "dataset" / "spam.csv",
            PROJECT_ROOT / "dataset" / "SMSSpamCollection",
        ]
        target = None
        for p in candidates:
            if p.exists():
                target = p
                break
        if target is None:
            raise FileNotFoundError(f"Dataset not found. Checked: {[str(p) for p in candidates]}")
    else:
        target = Path(filepath)
        if not target.exists():
            if (PROJECT_ROOT / filepath).exists():
                target = PROJECT_ROOT / filepath
            else:
                raise FileNotFoundError(f"Dataset not found at {filepath}")

    if target.suffix == ".csv":
        df = pd.read_csv(target, encoding="utf-8")
    else:
        df = pd.read_csv(target, sep="\t", names=["label", "message"], encoding="utf-8")

    # Rename columns if needed
    if "v1" in df.columns and "v2" in df.columns:
        df = df.rename(columns={"v1": "label", "v2": "message"})
    df = df[["label", "message"]].dropna()

    # Clean text and remove duplicates
    df["label"] = df["label"].str.strip().str.lower()
    df["message"] = df["message"].str.strip()
    df = df[df["label"].isin(["ham", "spam"])].drop_duplicates(subset=["message"]).reset_index(drop=True)

    # Encode labels: ham -> 0, spam -> 1
    df["label_num"] = df["label"].map({"ham": 0, "spam": 1})
    return df

# Alias for backwards compatibility
load_raw_data = load_data

def clean_dataset(df=None):
    clean_df = load_data()
    stats = {
        "total_messages": len(clean_df),
        "ham_count": int((clean_df["label_num"] == 0).sum()),
        "spam_count": int((clean_df["label_num"] == 1).sum()),
    }
    return clean_df, stats

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} messages ({df['label'].value_counts().to_dict()})")
