"""Fine-tune DistilBERT on the same split as the baseline, so the results are comparable.

Needs a GPU. Run it in Google Colab (Runtime > Change runtime type > T4 GPU):
    !pip install -q transformers accelerate
    !python train_distilbert.py --max-train 30000

--max-train limits training rows so it finishes on the free Colab tier (~20-30 min).
The test set is never reduced, so the comparison with the baseline stays fair.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, Trainer, TrainingArguments,
)

from src.data import SEED, load_dataset, split
from src.model import evaluate

MODEL_NAME = "distilbert-base-uncased"


class EmailDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.enc = tokenizer(list(texts), truncation=True, max_length=max_len, padding="max_length")
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: torch.tensor(v[i]) for k, v in self.enc.items()}
        item["labels"] = torch.tensor(self.labels[i])
        return item


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--epochs", type=float, default=2)
    args = ap.parse_args()

    df = load_dataset(args.data)
    train, test = split(df)
    if args.max_train and len(train) > args.max_train:
        train = train.sample(args.max_train, random_state=SEED)
    print(f"Train: {len(train):,}  Test: {len(test):,}  GPU: {torch.cuda.is_available()}")

    tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)
    # Same cleaned text as the baseline (the model is uncased, so lowercasing loses nothing).
    train_ds = EmailDataset(train["clean"], train["label"], tok, args.max_len)
    test_ds = EmailDataset(test["clean"], test["label"], tok, args.max_len)

    targs = TrainingArguments(
        output_dir="models/distilbert_ckpt",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=64,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_ratio=0.06,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=200,
        fp16=torch.cuda.is_available(),
        seed=SEED,
        report_to=[],
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds)
    trainer.train()

    logits = trainer.predict(test_ds).predictions
    prob = torch.softmax(torch.tensor(logits), dim=-1)[:, 1].numpy()
    metrics = evaluate(test["label"].to_numpy(), prob)
    metrics.update({"model": "distilbert", "split": "random_split", "n_train": len(train)})
    print(json.dumps(metrics, indent=2))

    Path("reports").mkdir(exist_ok=True)
    Path("reports/distilbert_metrics_random_split.json").write_text(json.dumps(metrics, indent=2))
    np.save("reports/distilbert_test_probs.npy", prob)
    trainer.save_model("models/distilbert")
    tok.save_pretrained("models/distilbert")
    print("Saved models/distilbert")


if __name__ == "__main__":
    main()
