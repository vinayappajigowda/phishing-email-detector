"""Load, clean and split the email datasets.

Expects CSV files in data/raw/ from the Kaggle "Phishing Email Dataset"
(naserabdullahalam/phishing-email-dataset): CEAS_08.csv, Enron.csv, Ling.csv,
Nazario.csv, Nigerian_Fraud.csv, SpamAssasin.csv. Label 1 = phishing/malicious, 0 = legitimate.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEED = 42
COMBINED_FILE = "phishing_email.csv"   # Kaggle's pre-merged file, a copy of the others

_URL = re.compile(r"(?i)\b(?:https?://|www\.)\S+")
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")
_NUM = re.compile(r"\b\d+(?:[.,]\d+)*\b")
_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Normalise text the same way for training and prediction.

    URLs, email addresses and numbers are replaced with tokens so the model
    learns patterns ("contains a link") rather than memorising specific addresses.
    """
    text = str(text).lower()
    text = _URL.sub(" urltoken ", text)
    text = _EMAIL.sub(" emailtoken ", text)
    text = _NUM.sub(" numtoken ", text)
    return _WS.sub(" ", text).strip()


def _read_one(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding_errors="replace", on_bad_lines="skip", low_memory=False)
    df.columns = [c.strip().lower() for c in df.columns]
    if "label" not in df.columns:
        raise ValueError(f"{path.name} has no 'label' column")
    if "text_combined" in df.columns:
        text = df["text_combined"].fillna("")
    else:
        subject = df["subject"].fillna("") if "subject" in df.columns else ""
        body = df["body"].fillna("") if "body" in df.columns else ""
        text = subject.astype(str) + " \n " + body.astype(str)
    out = pd.DataFrame({"text": text.astype(str), "label": df["label"], "source": path.stem})
    out["label"] = pd.to_numeric(out["label"], errors="coerce")
    return out.dropna(subset=["label"]).astype({"label": int})


def load_dataset(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    files = sorted(Path(data_dir).glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files in {data_dir}. See data/README.md.")
    # If the individual source files are present, skip the merged copy to avoid duplicates.
    if any(f.name != COMBINED_FILE for f in files):
        files = [f for f in files if f.name != COMBINED_FILE]

    df = pd.concat([_read_one(f) for f in files], ignore_index=True)
    df = df[df["label"].isin([0, 1])]
    df["clean"] = df["text"].map(clean_text)
    df = df[df["clean"].str.len() >= 20]

    # Remove exact duplicates BEFORE splitting. Otherwise the same email can land in
    # both train and test, and the scores look better than they really are.
    before = len(df)
    df = df.drop_duplicates(subset="clean").reset_index(drop=True)
    df.attrs["duplicates_removed"] = before - len(df)
    return df


def split(df: pd.DataFrame, test_size: float = 0.2, holdout_source: str | None = None):
    """Random stratified split, or hold out one whole source to test generalisation."""
    if holdout_source:
        test = df[df["source"] == holdout_source]
        train = df[df["source"] != holdout_source]
        if test.empty:
            raise ValueError(f"No rows from source '{holdout_source}'. Sources: {sorted(df['source'].unique())}")
        return train.reset_index(drop=True), test.reset_index(drop=True)
    train, test = train_test_split(df, test_size=test_size, stratify=df["label"], random_state=SEED)
    return train.reset_index(drop=True), test.reset_index(drop=True)
