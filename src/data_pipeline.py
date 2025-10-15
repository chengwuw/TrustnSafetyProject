"""
Ingest & split the open 'civil_comments' dataset from HuggingFace.
Creates data/train.csv and data/test.csv with columns: ['text', 'label'].
"""
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split
import os, json

TOXIC_THRESHOLD = float(os.getenv("TOXIC_THRESHOLD", "0.5"))
TEST_SIZE = float(os.getenv("TEST_SIZE", "0.2"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", "42"))

def binarize_label(x, thr=0.5):
    # civil_comments has a 'toxicity' float in [0,1]
    return 1 if x >= thr else 0

def main():
    print("Loading 'civil_comments' (this will download on first run)...")
    ds = load_dataset("civil_comments")
    # Use the 'train' split as full corpus (civil_comments provides a single split)
    df = ds['train'].to_pandas()[['text', 'toxicity']].dropna()
    df['label'] = df['toxicity'].apply(lambda v: binarize_label(v, TOXIC_THRESHOLD))
    df = df[['text', 'label']]
    print(df.head())

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df['label']
    )

    os.makedirs("data", exist_ok=True)
    train_df.to_csv("data/train.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    meta = {
        "dataset": "civil_comments",
        "toxic_threshold": TOXIC_THRESHOLD,
        "test_size": TEST_SIZE,
        "random_state": RANDOM_STATE,
        "n_train": len(train_df),
        "n_test": len(test_df),
        "pos_rate_train": float(train_df['label'].mean()),
        "pos_rate_test": float(test_df['label'].mean()),
    }
    with open("data/ingest_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved data/train.csv, data/test.csv and data/ingest_meta.json")
    print(meta)

if __name__ == "__main__":
    main()
