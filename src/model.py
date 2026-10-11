"""Baseline model (TF-IDF + Logistic Regression), evaluation and explanations."""
from __future__ import annotations

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score, precision_score,
    recall_score, roc_auc_score, roc_curve,
)
from sklearn.pipeline import Pipeline

from .data import clean_text


def build_baseline() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_df=0.9,
                                  max_features=200_000, sublinear_tf=True)),
        ("clf", LogisticRegression(C=4.0, class_weight="balanced", max_iter=2000)),
    ])


def evaluate(y_true, prob, threshold: float = 0.5) -> dict:
    """Metrics a security team cares about, not just accuracy."""
    y_true = np.asarray(y_true)
    pred = (np.asarray(prob) >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    out = {
        "n_test": int(len(y_true)),
        "phishing_in_test": int(y_true.sum()),
        "threshold": threshold,
        "precision": round(float(precision_score(y_true, pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, pred, zero_division=0)), 4),
        # False positive rate: share of legitimate emails wrongly flagged.
        # This is what annoys users and floods a SOC queue, so report it explicitly.
        "false_positive_rate": round(float(fp / (fp + tn)) if (fp + tn) else 0.0, 4),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }
    if len(np.unique(y_true)) == 2:
        out["roc_auc"] = round(float(roc_auc_score(y_true, prob)), 4)
        out["pr_auc"] = round(float(average_precision_score(y_true, prob)), 4)
        # Recall when we only allow 1% of legitimate mail to be flagged.
        fpr, tpr, _ = roc_curve(y_true, prob)
        out["recall_at_1pct_fpr"] = round(float(tpr[fpr <= 0.01].max()), 4)
    return out


def predict_proba(model: Pipeline, texts: list[str]) -> np.ndarray:
    return model.predict_proba([clean_text(t) for t in texts])[:, 1]


def explain(model: Pipeline, text: str, top_k: int = 8) -> list[tuple[str, float]]:
    """Words/phrases in this email that pushed the score towards phishing.

    For a linear model, each term's contribution is its TF-IDF weight times
    the model coefficient, so the explanation is exact rather than approximate.
    """
    vec, clf = model.named_steps["tfidf"], model.named_steps["clf"]
    x = vec.transform([clean_text(text)])
    contrib = x.multiply(clf.coef_[0]).tocsr()
    idx, vals = contrib.indices, contrib.data
    names = vec.get_feature_names_out()
    order = np.argsort(vals)[::-1]
    return [(names[idx[i]], round(float(vals[i]), 3)) for i in order[:top_k] if vals[i] > 0]


def top_global_terms(model: Pipeline, k: int = 25) -> dict[str, list[tuple[str, float]]]:
    names = model.named_steps["tfidf"].get_feature_names_out()
    coef = model.named_steps["clf"].coef_[0]
    hi, lo = np.argsort(coef)[::-1][:k], np.argsort(coef)[:k]
    return {
        "phishing": [(names[i], round(float(coef[i]), 3)) for i in hi],
        "legitimate": [(names[i], round(float(coef[i]), 3)) for i in lo],
    }
