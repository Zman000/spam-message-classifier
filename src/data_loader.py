import os
import pandas as pd

def load_data(filepath="dataset/spam.csv"):
    """Load the dataset and return a cleaned dataframe."""
    if not os.path.exists(filepath):
        alt_path = "dataset/SMSSpamCollection"
        if os.path.exists(alt_path):
            df = pd.read_csv(alt_path, sep="\t", names=["label", "message"], encoding="utf-8")
        else:
            raise FileNotFoundError(f"Dataset not found at {filepath} or {alt_path}")
    else:
        df = pd.read_csv(filepath, encoding="utf-8")

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
def clean_dataset(df):
    clean_df = load_data()
    stats = {
        "total_messages": len(clean_df),
        "ham_count": (clean_df["label_num"] == 0).sum(),
        "spam_count": (clean_df["label_num"] == 1).sum(),
    }
    return clean_df, stats

if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} messages ({df['label'].value_counts().to_dict()})")
