#!/usr/bin/env python3
"""
Compute mean absolute label distance (MAE) from saved confusion matrices.

The 5-class task is ordinal (bad < mediocre < average < good < great), so a
prediction of "great" when the truth is "good" should be penalized less than
"bad" when the truth is "great". Accuracy and macro F1 don't capture this;
MAE does.

For a confusion matrix M[i,j] (rows = true, cols = predicted):
    MAE = sum_{i,j} M[i,j] * |i - j|  /  sum_{i,j} M[i,j]

Reads every existing results JSON and writes results/mae_summary.json with
MAE for each (model, split) plus a one-table summary.
"""

import json
import os
from typing import Any, Dict, List, Tuple

import numpy as np

RESULT_FILES = [
    ("Majority baseline", "results/baselines/majority_baseline.json", "majority"),
    ("TF-IDF + LogReg",   "results/tfidf_logreg/results.json", "tfidf"),
    ("Structured (RF)",   "results/structured_features/results.json", "struct_rf"),
    ("Structured (LR)",   "results/structured_features/results.json", "struct_lr"),
    ("DistilBERT + LR",   "results/distilbert/results.json", "bert"),
]


def get_split_metrics(path: str, key: str) -> Dict[str, Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if key in ("majority", "tfidf", "bert"):
        return data["results"]
    if key == "struct_rf":
        return data["results"]["random_forest"]
    if key == "struct_lr":
        return data["results"]["logistic_regression"]
    raise ValueError(key)


def mae_from_confusion(cm: List[List[int]]) -> float:
    cm = np.array(cm, dtype=float)
    n_classes = cm.shape[0]
    weights = np.abs(np.arange(n_classes)[:, None] - np.arange(n_classes)[None, :])
    total = cm.sum()
    if total == 0:
        return float("nan")
    return float((cm * weights).sum() / total)


def random_baseline_mae(n_classes: int) -> float:
    """Expected MAE if predictions were uniform random over classes — useful reference."""
    s = 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            s += abs(i - j)
    return s / (n_classes * n_classes)


def main() -> None:
    out_rows = []
    summary = {}
    for name, path, key in RESULT_FILES:
        if not os.path.exists(path):
            print(f"[skip] {path} missing")
            continue
        per_split = get_split_metrics(path, key)
        per_split_mae = {}
        for split in ("train", "val", "test"):
            cm = per_split[split].get("confusion_matrix")
            if cm is None:
                continue
            per_split_mae[split] = round(mae_from_confusion(cm), 4)
        summary[name] = per_split_mae
        out_rows.append((name, per_split_mae))

    n_classes = 5
    rand_mae = random_baseline_mae(n_classes)

    # Print
    print(f"\nReference: random-uniform-prediction MAE on {n_classes}-class = {rand_mae:.3f}")
    print("Lower is better. Worst case (always wrong by max distance) = 4.0\n")
    print(f"{'Model':<22} {'train':>7} {'val':>7} {'test':>7}")
    print("-" * 46)
    for name, mae in out_rows:
        tr = mae.get("train", float("nan"))
        vl = mae.get("val", float("nan"))
        te = mae.get("test", float("nan"))
        print(f"{name:<22} {tr:>7.3f} {vl:>7.3f} {te:>7.3f}")

    out = {
        "metric": "mean_absolute_label_distance",
        "n_classes": n_classes,
        "random_uniform_reference_mae": round(rand_mae, 4),
        "by_model": summary,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/mae_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: results/mae_summary.json")


if __name__ == "__main__":
    main()
