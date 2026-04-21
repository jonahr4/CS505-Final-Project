#!/usr/bin/env python3
"""
Model 1: TF-IDF + multinomial logistic regression.

For each movie, we have one document (all its pre-release YouTube comments
concatenated, space-separated). Vectorize with TF-IDF (word + bigrams,
English stopwords removed), then fit multinomial logistic regression on
rating_bucket.

Movie-level split from build_final_dataset.py is respected.
Hyperparameters are modest and held fixed (no sweep — this is the baseline
text model, not the final system).
"""

import json
import os
import sys
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

IN_CSV = "data/processed/movies_dataset.csv"
OUT_DIR = "results/tfidf_logreg"
OUT_JSON = os.path.join(OUT_DIR, "results.json")

# Vectorizer settings
MAX_FEATURES = 20000
NGRAM_RANGE = (1, 2)
MIN_DF = 2            # drop terms appearing in only 1 movie (noise)
MAX_DF = 0.95         # drop terms appearing in ≥95% of movies (boilerplate)

# Logreg settings
C = 1.0               # inverse regularization strength
MAX_ITER = 2000
RANDOM_SEED = 42


def evaluate(y_true: np.ndarray, y_pred: np.ndarray,
             n_classes: int) -> Dict[str, float]:
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
    return {
        "n": n,
        "accuracy": round(acc, 4),
        "macro_f1": round(float(np.mean(f1s)), 4),
        "per_class": per_class,
    }


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main() -> None:
    if not os.path.exists(IN_CSV):
        print(f"ERROR: {IN_CSV} not found.")
        sys.exit(1)

    df = pd.read_csv(IN_CSV)
    # comments_text may contain NaN if a movie had 0 kept comments in the CSV
    # but build_final_dataset already filtered those out; still be safe:
    df["comments_text"] = df["comments_text"].fillna("").astype(str)

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    n_classes = int(df["rating_bucket"].max()) + 1
    print(f"train={len(train)}  val={len(val)}  test={len(test)}  classes={n_classes}")

    # Fit TF-IDF on TRAIN ONLY — fitting on the full corpus would leak
    # test-side vocabulary/statistics into the model.
    print("\nFitting TF-IDF on train...")
    vec = TfidfVectorizer(
        max_features=MAX_FEATURES,
        ngram_range=NGRAM_RANGE,
        min_df=MIN_DF,
        max_df=MAX_DF,
        stop_words="english",
        lowercase=True,
        strip_accents="unicode",
    )
    X_train = vec.fit_transform(train["comments_text"])
    X_val = vec.transform(val["comments_text"])
    X_test = vec.transform(test["comments_text"])
    print(f"Vocabulary size: {len(vec.vocabulary_)}")
    print(f"Feature matrix shapes: train={X_train.shape}  val={X_val.shape}  test={X_test.shape}")

    y_train = train["rating_bucket"].to_numpy()
    y_val = val["rating_bucket"].to_numpy()
    y_test = test["rating_bucket"].to_numpy()

    print("\nFitting logistic regression...")
    clf = LogisticRegression(
        C=C,
        max_iter=MAX_ITER,
        class_weight="balanced",   # compensate for slight class imbalance
        solver="lbfgs",
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train, y_train)

    results: Dict[str, Dict] = {}
    for name, X, y in [
        ("train", X_train, y_train),
        ("val", X_val, y_val),
        ("test", X_test, y_test),
    ]:
        y_pred = clf.predict(X)
        metrics = evaluate(y, y_pred, n_classes)
        cm = confusion_matrix(y, y_pred, n_classes)
        metrics["confusion_matrix"] = cm.tolist()
        results[name] = metrics

        print(f"\n=== {name.upper()} ({metrics['n']} movies) ===")
        print(f"  accuracy: {metrics['accuracy']}")
        print(f"  macro F1: {metrics['macro_f1']}")
        per_f1 = ", ".join(f"c{c}={metrics['per_class'][c]['f1']}"
                           for c in range(n_classes))
        print(f"  per-class F1: {per_f1}")
        print(f"  confusion matrix (rows=true, cols=pred):")
        for row in cm:
            print("    " + "  ".join(f"{v:>3}" for v in row))

    # Look at what words are signaling each class (quick interpretability)
    print("\nTop features per class:")
    vocab = np.array(vec.get_feature_names_out())
    for c in range(n_classes):
        label = train[train["rating_bucket"] == c]["rating_bucket_label"].iloc[0]
        top_idx = np.argsort(clf.coef_[c])[-10:][::-1]
        print(f"  class {c} ({label}): " + ", ".join(vocab[top_idx]))

    os.makedirs(OUT_DIR, exist_ok=True)
    out = {
        "model": "tfidf_logreg",
        "config": {
            "max_features": MAX_FEATURES,
            "ngram_range": list(NGRAM_RANGE),
            "min_df": MIN_DF,
            "max_df": MAX_DF,
            "stop_words": "english",
            "C": C,
            "class_weight": "balanced",
            "max_iter": MAX_ITER,
            "seed": RANDOM_SEED,
        },
        "vocab_size": len(vec.vocabulary_),
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
