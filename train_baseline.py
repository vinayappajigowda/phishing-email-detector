"""Train and evaluate the TF-IDF + Logistic Regression baseline.

Usage:
    python train_baseline.py                          # random 80/20 split
    python train_baseline.py --holdout-source CEAS_08 # train on the rest, test on one unseen source
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay, PrecisionRecallDisplay

from src.data import load_dataset, split
from src.model import build_baseline, evaluate, top_global_terms


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--holdout-source", default=None)
    args = ap.parse_args()

    Path("models").mkdir(exist_ok=True)
    Path("reports").mkdir(exist_ok=True)
    tag = f"holdout_{args.holdout_source}" if args.holdout_source else "random_split"

    df = load_dataset(args.data)
    print(f"Loaded {len(df):,} unique emails ({df.attrs['duplicates_removed']:,} duplicates removed)")
    print(df.groupby(["source", "label"]).size().unstack(fill_value=0), "\n")

    train, test = split(df, args.test_size, args.holdout_source)
    print(f"Train: {len(train):,}  Test: {len(test):,}")

    model = build_baseline()
    t0 = time.time()
    model.fit(train["clean"], train["label"])
    print(f"Trained in {time.time() - t0:.1f}s")

    prob = model.predict_proba(test["clean"])[:, 1]
    metrics = evaluate(test["label"], prob)
    metrics.update({"model": "tfidf_logreg", "split": tag, "n_train": len(train)})
    print(json.dumps(metrics, indent=2))

    (Path("reports") / f"baseline_metrics_{tag}.json").write_text(json.dumps(metrics, indent=2))

    fig, ax = plt.subplots(figsize=(4.5, 4))
    ConfusionMatrixDisplay.from_predictions(
        test["label"], (prob >= 0.5).astype(int), display_labels=["Legitimate", "Phishing"],
        cmap="Blues", values_format=",", ax=ax, colorbar=False)
    ax.set_title("TF-IDF + Logistic Regression")
    fig.tight_layout()
    fig.savefig(f"reports/confusion_matrix_baseline_{tag}.png", dpi=150)

    if test["label"].nunique() == 2:
        fig, ax = plt.subplots(figsize=(4.5, 4))
        PrecisionRecallDisplay.from_predictions(test["label"], prob, name="TF-IDF + LogReg", ax=ax)
        ax.set_title("Precision-recall curve")
        fig.tight_layout()
        fig.savefig(f"reports/pr_curve_baseline_{tag}.png", dpi=150)

    terms = top_global_terms(model)
    pd.DataFrame({
        "phishing_term": [t for t, _ in terms["phishing"]],
        "phishing_weight": [w for _, w in terms["phishing"]],
        "legit_term": [t for t, _ in terms["legitimate"]],
        "legit_weight": [w for _, w in terms["legitimate"]],
    }).to_csv(f"reports/top_terms_{tag}.csv", index=False)

    if not args.holdout_source:
        joblib.dump(model, "models/baseline.joblib")
        print("Saved models/baseline.joblib (used by the Streamlit app)")


if __name__ == "__main__":
    main()
