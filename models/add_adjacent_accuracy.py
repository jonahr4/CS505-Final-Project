#!/usr/bin/env python3
"""
Compute adjacent accuracy (within ±1 bucket) for every model's saved
confusion matrix. The 5-class rating task is ordinal (bad < mediocre <
average < good < great), so a prediction one bucket off (good vs great)
should count as substantially better than one four buckets off (bad vs
great). Adjacent accuracy captures this human-readable interpretation
alongside the more granular MAE metric.

For a confusion matrix M[i,j] (rows = true, cols = predicted):
    adjacent_acc = sum_{|i-j| <= TOL} M[i,j]  /  sum_{i,j} M[i,j]

Reads every existing results JSON and writes
results/adjacent_accuracy_summary.json.
"""

import json
import os
from typing import Any, Dict, List

import numpy as np

TOL = 1   # off-by-≤1 counts as correct

RESULT_FILES = [
    ("Majority baseline",            "results/baselines/majority_baseline.json", "majority"),
    ("TF-IDF + LogReg",              "results/tfidf_logreg/results.json", "tfidf"),
    ("Structured (RF)",              "results/structured_features/results.json", "struct_rf"),
    ("Structured (LR)",              "results/structured_features/results.json", "struct_lr"),
    ("DistilBERT (frozen) + LR",     "results/distilbert/results.json", "bert_frozen"),
    ("DistilBERT FT (per-comment)",  "results/distilbert_finetune/results.json", "bert_ft_pc"),
    ("DistilBERT FT (movie-level)",  "results/distilbert_finetune_movie/results.json", "bert_ft_ml"),
]


def get_split_metrics(path: str, key: str) -> Dict[str, Dict[str, Any]]:
    with open(path) as f:
        data = json.load(f)
    if key in ("majority", "tfidf"):
        return data["results"]
    if key == "struct_rf":
        return data["results"]["random_forest"]
    if key == "struct_lr":
        return data["results"]["logistic_regression"]
    if key in ("bert_frozen", "bert_ft_pc", "bert_ft_ml"):
        return data["results"]
    raise ValueError(key)


def adjacent_acc_from_cm(cm: List[List[int]], tol: int = TOL) -> float:
    cm = np.array(cm, dtype=float)
    n_classes = cm.shape[0]
    weights = np.zeros_like(cm)
    for i in range(n_classes):
        for j in range(n_classes):
            if abs(i - j) <= tol:
                weights[i, j] = 1.0
    total = cm.sum()
    if total == 0:
        return float("nan")
    return float((cm * weights).sum() / total)


def random_baseline_adjacent_acc(n_classes: int = 5, tol: int = TOL) -> float:
    """Expected adjacent accuracy if predictions were uniform random."""
    matches = 0.0
    for i in range(n_classes):
        for j in range(n_classes):
            if abs(i - j) <= tol:
                matches += 1.0
    return matches / (n_classes * n_classes)


def main() -> None:
    out_rows = []
    summary: Dict[str, Dict[str, float]] = {}
    for name, path, key in RESULT_FILES:
        if not os.path.exists(path):
            print(f"[skip] {path} missing")
            continue
        per_split = get_split_metrics(path, key)
        per_split_adj: Dict[str, float] = {}
        for split in ("train", "val", "test"):
            cm = per_split.get(split, {}).get("confusion_matrix")
            if cm is None:
                continue
            per_split_adj[split] = round(adjacent_acc_from_cm(cm), 4)
        summary[name] = per_split_adj
        out_rows.append((name, per_split_adj))

    n_classes = 5
    rand_adj = random_baseline_adjacent_acc(n_classes, TOL)
    print(f"\nReference: random-uniform-prediction adjacent accuracy "
          f"(±{TOL} bucket) on {n_classes}-class = {rand_adj:.3f}")
    print("Higher is better. Max = 1.0 (always within ±1 bucket).\n")
    print(f"{'Model':<32} {'train':>7} {'val':>7} {'test':>7}")
    print("-" * 56)
    for name, adj in out_rows:
        tr = adj.get("train", float("nan"))
        vl = adj.get("val", float("nan"))
        te = adj.get("test", float("nan"))
        print(f"{name:<32} {tr:>7.3f} {vl:>7.3f} {te:>7.3f}")

    out = {
        "metric": "adjacent_accuracy",
        "tolerance_buckets": TOL,
        "n_classes": n_classes,
        "random_uniform_reference": round(rand_adj, 4),
        "by_model": summary,
    }
    os.makedirs("results", exist_ok=True)
    with open("results/adjacent_accuracy_summary.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nSaved: results/adjacent_accuracy_summary.json")


if __name__ == "__main__":
    main()
