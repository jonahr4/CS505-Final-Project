#!/usr/bin/env python3
"""
Majority-class baseline.

Predicts the most frequent rating_bucket in the train split for every
val/test example. Any real model must beat this floor on both accuracy
and macro F1 to be meaningful.
"""

import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd

IN_CSV = "data/processed/movies_dataset.csv"
OUT_DIR = "results/baselines"
OUT_JSON = os.path.join(OUT_DIR, "majority_baseline.json")


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             n_classes: int) -> Dict[str, float]:
    """
    Compute accuracy, macro F1, and per-class precision/recall/F1.
    Written by hand (no sklearn) so we can eyeball the math.
    """
    n = len(y_true)
    acc = float(np.mean(y_true == y_pred))

    per_class = {}
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {
            "tp": tp, "fp": fp, "fn": fn,
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(np.sum(y_true == c)),
        }
        f1s.append(f1)

    macro_f1 = float(np.mean(f1s))
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": per_class,
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main() -> None:
    if not os.path.exists(IN_CSV):
        print(f"ERROR: {IN_CSV} not found. Run build_final_dataset.py first.")
        sys.exit(1)

    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} movies")

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]

    n_classes = int(df["rating_bucket"].max()) + 1
    majority_class = int(train["rating_bucket"].mode()[0])
    print(f"Majority class in train: {majority_class} "
          f"({train[train['rating_bucket'] == majority_class]['rating_bucket_label'].iloc[0]})")

    results: Dict[str, Dict] = {}
    for name, split in [("train", train), ("val", val), ("test", test)]:
        y_true = split["rating_bucket"].to_numpy()
        y_pred = np.full_like(y_true, majority_class)
        metrics = evaluate(y_true, y_pred, n_classes)
        cm = confusion_matrix(y_true, y_pred, n_classes)
        metrics["confusion_matrix"] = cm.tolist()
        results[name] = metrics

        print(f"\n=== {name.upper()} ({metrics['n']} movies) ===")
        print(f"  accuracy: {metrics['accuracy']}")
        print(f"  macro F1: {metrics['macro_f1']}")
        print(f"  per-class F1: " +
              ", ".join(f"c{c}={metrics['per_class'][c]['f1']}" for c in range(n_classes)))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        "model": "majority_class",
        "majority_class": majority_class,
        "n_classes": n_classes,
        "class_labels": df.drop_duplicates("rating_bucket")
                          .sort_values("rating_bucket")[["rating_bucket", "rating_bucket_label"]]
                          .set_index("rating_bucket")["rating_bucket_label"].to_dict(),
        "results": results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
