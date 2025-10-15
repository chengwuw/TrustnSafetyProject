"""
Score text(s) with the trained model.
Usage:
  python src/score.py --text "You are awful..."
  python src/score.py --csv path/to/file.csv --text_col text
"""
import argparse, sys
import pandas as pd
from joblib import load

def load_model(path="models/toxicity_clf.joblib"):
    return load(path)

def score_texts(model, texts):
    import numpy as np
    probas = model.predict_proba(texts)[:,1]
    return probas

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text", type=str, help="Single text to score")
    parser.add_argument("--csv", type=str, help="CSV file path containing texts")
    parser.add_argument("--text_col", type=str, default="text", help="Column name with text")
    args = parser.parse_args()

    model = load_model()

    if args.text:
        p = score_texts(model, [args.text])[0]
        print(f"toxicity_prob={p:.4f}")
        sys.exit(0)

    if args.csv:
        df = pd.read_csv(args.csv)
        if args.text_col not in df.columns:
            raise ValueError(f"Column '{args.text_col}' not in CSV columns: {df.columns.tolist()}")
        df["toxicity_prob"] = score_texts(model, df[args.text_col].astype(str).tolist())
        out = args.csv.replace(".csv", "_scored.csv")
        df.to_csv(out, index=False)
        print(f"Saved: {out}")
        sys.exit(0)

    parser.print_help()

if __name__ == "__main__":
    main()
