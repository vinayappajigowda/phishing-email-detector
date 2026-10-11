"""End-to-end smoke test on a tiny made-up dataset. Checks the code runs, not model quality."""
import pandas as pd

from src.data import load_dataset, split
from src.model import build_baseline, evaluate, explain, predict_proba

PHISH = [
    "urgent verify your account now or it will be suspended click here {i}",
    "your password expires today login immediately to keep access {i}",
    "unusual activity detected confirm your identity within 24 hours {i}",
]
LEGIT = [
    "minutes from the team meeting are attached see you on thursday {i}",
    "the library book club meets next week bring a friend {i}",
    "quarterly report draft is ready for your review when you have time {i}",
]


def _tag(i):
    # Letters, not digits: cleaning turns digits into one token, which would make rows duplicates.
    return "ref" + chr(97 + i % 26) + chr(97 + i // 26)


def _write(tmp_path):
    rows_a = [{"subject": "s", "body": t.format(i=_tag(i)), "label": 1} for i in range(40) for t in PHISH]
    rows_b = [{"subject": "s", "body": t.format(i=_tag(i)), "label": 0} for i in range(40) for t in LEGIT]
    pd.DataFrame(rows_a[:60] + rows_b[:60]).to_csv(tmp_path / "SourceA.csv", index=False)
    pd.DataFrame(rows_a[60:] + rows_b[60:]).to_csv(tmp_path / "SourceB.csv", index=False)
    # The merged Kaggle file should be ignored when individual files exist.
    pd.DataFrame({"text_combined": ["dup"] * 5, "label": [1] * 5}).to_csv(
        tmp_path / "phishing_email.csv", index=False)


def test_train_predict_explain(tmp_path):
    _write(tmp_path)
    df = load_dataset(tmp_path)
    assert set(df["source"]) == {"SourceA", "SourceB"}
    assert df["clean"].is_unique

    train, test = split(df)
    model = build_baseline().fit(train["clean"], train["label"])
    m = evaluate(test["label"], model.predict_proba(test["clean"])[:, 1])
    assert m["f1"] > 0.9 and "roc_auc" in m

    p = predict_proba(model, ["URGENT verify your account now, click here"])[0]
    assert p > 0.5
    assert explain(model, "urgent verify your account")


def test_holdout_source(tmp_path):
    _write(tmp_path)
    train, test = split(load_dataset(tmp_path), holdout_source="SourceB")
    assert set(test["source"]) == {"SourceB"} and "SourceB" not in set(train["source"])
