#!/usr/bin/env python3
"""
Item 2 (small) from midterm plan: per-window temporal ablation.

Train the structured-LR model with only the early-window features, only the
middle-window features, only the late-window features, and with all temporal
features removed entirely. Compare to the full model. Quantifies which
pre-release window carries the most signal.

Outputs:
    results/temporal_ablation.json
"""

import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

FEATS = "results/structured_features/features.csv"
OUT_JSON = "results/temporal_ablation.json"

# Non-temporal features that stay in every variant for fair comparison
NON_TEMPORAL_FEATURES = [
    "n_comments", "sent_mean", "sent_std", "sent_pos_frac", "sent_neg_frac",
    "hype_frac", "nega_frac", "avg_comment_len", "median_like_count",
    "mean_like_count", "reply_rate", "caps_frac", "emoji_density",
    "short_comment_frac", "cred_sent", "hype_minus_neg",
]
TEMPORAL_FEATURES = {
    "early":  ["sent_mean_early",  "hype_frac_early",  "n_frac_early"],
    "middle": ["sent_mean_middle", "hype_frac_middle", "n_frac_middle"],
    "late":   ["sent_mean_late",   "hype_frac_late",   "n_frac_late"],
}


def evaluate(y_true, y_pred, n_classes=5):
    n = len(y_true)
    acc = float(np.mean(y_true == y_pred))
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        rec = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        f1s.append(f1)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    weights = np.abs(np.arange(n_classes)[:, None] - np.arange(n_classes)[None, :])
    mae = float((cm*weights).sum() / max(cm.sum(), 1))
    return {"n": n, "accuracy": round(acc, 4),
            "macro_f1": round(float(np.mean(f1s)), 4),
            "mae": round(mae, 4)}


def fit_and_eval(feats: pd.DataFrame, cols, label: str):
    train = feats[feats["split"] == "train"]
    val = feats[feats["split"] == "val"]
    test = feats[feats["split"] == "test"]
    Xtr = train[cols].to_numpy()
    Xvl = val[cols].to_numpy()
    Xte = test[cols].to_numpy()
    y_tr = train["rating_bucket"].to_numpy()
    y_vl = val["rating_bucket"].to_numpy()
    y_te = test["rating_bucket"].to_numpy()
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=5000, class_weight="balanced",
                             solver="lbfgs", random_state=42)
    clf.fit(sc.transform(Xtr), y_tr)
    return {
        "label": label, "n_features": len(cols),
        "train": evaluate(y_tr, clf.predict(sc.transform(Xtr))),
        "val":   evaluate(y_vl, clf.predict(sc.transform(Xvl))),
        "test":  evaluate(y_te, clf.predict(sc.transform(Xte))),
    }


def main():
    if not os.path.exists(FEATS):
        print(f"ERROR: {FEATS} missing"); sys.exit(1)
    feats = pd.read_csv(FEATS)
    print(f"Loaded features: {feats.shape}")

    variants = [
        ("Full (all temporal windows)",
         NON_TEMPORAL_FEATURES + sum(TEMPORAL_FEATURES.values(), [])),
        ("No temporal features",        NON_TEMPORAL_FEATURES),
        ("Only EARLY window",           NON_TEMPORAL_FEATURES + TEMPORAL_FEATURES["early"]),
        ("Only MIDDLE window",          NON_TEMPORAL_FEATURES + TEMPORAL_FEATURES["middle"]),
        ("Only LATE window",            NON_TEMPORAL_FEATURES + TEMPORAL_FEATURES["late"]),
    ]

    results = []
    print(f"\n{'Variant':<32} {'#feat':>5}  {'tst Acc':>7} {'tst F1':>7} {'tst MAE':>7}")
    print("-" * 64)
    for label, cols in variants:
        r = fit_and_eval(feats, cols, label)
        results.append(r)
        t = r["test"]
        print(f"{label:<32} {r['n_features']:>5}  {t['accuracy']:>7.3f} {t['macro_f1']:>7.3f} {t['mae']:>7.3f}")

    out = {"variants": results}
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
