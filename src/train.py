"""
Train a simple TF-IDF + LogisticRegression toxicity classifier.
Saves: models/toxicity_clf.joblib and models/metrics.json
"""
import pandas as pd
import numpy as np
import json, os
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report, roc_auc_score, average_precision_score,
    precision_recall_curve, roc_curve
)
from joblib import dump
import matplotlib.pyplot as plt

def load_data():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    return train, test

def build_pipeline():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(max_features=100_000, ngram_range=(1,2), min_df=2)),
        ("lr", LogisticRegression(max_iter=1000, n_jobs=None, class_weight="balanced")),
    ])
    return pipe

def eval_and_plots(pipe, X_test, y_test):
    probas = pipe.predict_proba(X_test)[:,1]
    preds = (probas >= 0.5).astype(int)

    report = classification_report(y_test, preds, output_dict=True)
    roc_auc = roc_auc_score(y_test, probas)
    pr_auc = average_precision_score(y_test, probas)

    # PR curve
    precision, recall, _ = precision_recall_curve(y_test, probas)
    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curve")
    plt.savefig("models/pr_curve.png", bbox_inches="tight")
    plt.close()

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, probas)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.savefig("models/roc_curve.png", bbox_inches="tight")
    plt.close()

    return {
        "roc_auc": float(roc_auc),
        "pr_auc": float(pr_auc),
        "classification_report": report,
    }

def main():
    os.makedirs("models", exist_ok=True)
    train, test = load_data()
    X_train, y_train = train["text"].astype(str), train["label"].astype(int)
    X_test, y_test = test["text"].astype(str), test["label"].astype(int)

    pipe = build_pipeline()
    pipe.fit(X_train, y_train)

    metrics = eval_and_plots(pipe, X_test, y_test)

    dump(pipe, "models/toxicity_clf.joblib")
    with open("models/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("Saved model to models/toxicity_clf.joblib")
    print("Key metrics:", {k: metrics[k] for k in ["roc_auc", "pr_auc"]})

if __name__ == "__main__":
    main()
