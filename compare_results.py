"""Print a Markdown results table from reports/*metrics*.json, ready to paste into the README."""
import json
from pathlib import Path

COLS = [("model", "Model"), ("split", "Test set"), ("n_train", "Train rows"), ("precision", "Precision"),
        ("recall", "Recall"), ("f1", "F1"), ("false_positive_rate", "False positive rate"),
        ("recall_at_1pct_fpr", "Recall @ 1% FPR"), ("roc_auc", "ROC-AUC")]

rows = [json.loads(p.read_text()) for p in sorted(Path("reports").glob("*metrics*.json"))]
if not rows:
    raise SystemExit("No results yet. Run train_baseline.py first.")
print("| " + " | ".join(h for _, h in COLS) + " |")
print("|" + "---|" * len(COLS))
for r in rows:
    print("| " + " | ".join(str(r.get(k, "-")) for k, _ in COLS) + " |")
