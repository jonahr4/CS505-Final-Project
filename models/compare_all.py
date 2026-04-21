#!/usr/bin/env python3
"""
Aggregate all model results into a single comparison report.
"""

import json
import os

RESULT_FILES = [
    ("Majority baseline", "results/baselines/majority_baseline.json", "majority"),
    ("TF-IDF + LogReg",   "results/tfidf_logreg/results.json", "tfidf"),
    ("Structured (RF)",   "results/structured_features/results.json", "struct_rf"),
    ("Structured (LR)",   "results/structured_features/results.json", "struct_lr"),
    ("DistilBERT + LR",   "results/distilbert/results.json", "bert"),
]


def get_metrics(path: str, key: str):
    if not os.path.exists(path):
        return None
    with open(path) as f:
        data = json.load(f)
    if key == "majority":
        return data["results"]
    if key == "tfidf":
        return data["results"]
    if key == "struct_rf":
        return data["results"]["random_forest"]
    if key == "struct_lr":
        return data["results"]["logistic_regression"]
    if key == "bert":
        return data["results"]
    return None


def main():
    rows = []
    for name, path, key in RESULT_FILES:
        m = get_metrics(path, key)
        if m is None:
            print(f"[missing] {name} at {path}")
            continue
        rows.append({
            "model": name,
            "train_acc": m["train"]["accuracy"],
            "train_f1": m["train"]["macro_f1"],
            "val_acc": m["val"]["accuracy"],
            "val_f1": m["val"]["macro_f1"],
            "test_acc": m["test"]["accuracy"],
            "test_f1": m["test"]["macro_f1"],
        })

    if not rows:
        print("No results found.")
        return

    # Print table
    hdr = f"{'Model':<22} {'trAcc':>7} {'trF1':>7} {'valAcc':>7} {'valF1':>7} {'tstAcc':>7} {'tstF1':>7}"
    print("\n" + "=" * len(hdr))
    print(hdr)
    print("=" * len(hdr))
    for r in rows:
        print(f"{r['model']:<22} "
              f"{r['train_acc']:>7.3f} {r['train_f1']:>7.3f} "
              f"{r['val_acc']:>7.3f} {r['val_f1']:>7.3f} "
              f"{r['test_acc']:>7.3f} {r['test_f1']:>7.3f}")
    print("=" * len(hdr))

    # Highlight best test
    best_acc = max(rows, key=lambda r: r["test_acc"])
    best_f1 = max(rows, key=lambda r: r["test_f1"])
    print(f"\nBest test accuracy: {best_acc['model']} ({best_acc['test_acc']})")
    print(f"Best test macro F1: {best_f1['model']} ({best_f1['test_f1']})")

    # Save comparison
    os.makedirs("results", exist_ok=True)
    with open("results/comparison.json", "w") as f:
        json.dump(rows, f, indent=2)
    print("\nSaved: results/comparison.json")


if __name__ == "__main__":
    main()
