#!/usr/bin/env python3
"""
Model 2: Structured per-movie features + classifier.

For each movie, derive a small set of interpretable features from its
pre-release comments:

    SENTIMENT (VADER):
        sent_mean        avg VADER compound score across comments
        sent_std         std of VADER compound scores
        sent_pos_frac    fraction of comments with compound > 0.5
        sent_neg_frac    fraction of comments with compound < -0.5

    HYPE (keyword heuristics, proposal's "expected quality"):
        hype_frac        fraction of comments containing ANY hype term
                         (can't wait, excited, hyped, goat, peak, masterpiece,
                          perfection, day one, cinema, must watch, ...)
        nega_frac        fraction of comments with negative anticipation
                         (looks bad, gonna flop, looks cringe, no thanks, ...)

    VOLUME / ENGAGEMENT:
        n_comments          total pre-release comment count
        avg_comment_len     avg comment length (chars)
        median_like_count   median likes per comment
        mean_like_count     mean likes per comment
        reply_rate          mean replies per top-level comment

    SPAM-LIKE:
        caps_frac           fraction of comments that are >40% uppercase
        emoji_density       avg emoji count per comment
        short_comment_frac  fraction of comments with < 20 chars (low info)

    TEMPORAL (proposal's "temporal modeling"):
        for each of 3 pre-release windows (early/middle/late), compute
        sent_mean_{early,middle,late} and hype_frac_{early,middle,late}
        relative to the movie's own comment-time span.

    CREDIBILITY:
        cred_mean_likes     like-weighted avg sentiment
                            (implements proposal's credibility-weighted aggregation)

A RandomForestClassifier is trained on these features; movie-level split
is respected. We also train a LogisticRegression with the same features
for comparison.
"""

import json
import os
import re
import sys
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

IN_CSV = "data/processed/movies_dataset.csv"
IN_JSONL = "data/processed/movie_comments.jsonl"
OUT_DIR = "results/structured_features"
OUT_FEATS = os.path.join(OUT_DIR, "features.csv")
OUT_JSON = os.path.join(OUT_DIR, "results.json")

RANDOM_SEED = 42

HYPE_TERMS = [
    "can't wait", "cannot wait", "cant wait", "so excited", "hyped",
    "goat", "peak", "masterpiece", "perfection", "day one", "day 1",
    "cinema", "cinematic", "must watch", "must see", "top tier",
    "looks amazing", "looks incredible", "gonna be good", "gonna be great",
    "best movie", "fire", "banger", "no way", "chills",
]
NEGA_TERMS = [
    "looks bad", "looks awful", "looks terrible", "gonna flop", "will flop",
    "looks cringe", "looks dumb", "no thanks", "hard pass", "skip",
    "trash", "garbage", "mid", "cash grab", "woke", "who asked for this",
    "nobody asked", "ruined", "stop making", "another remake",
]
EMOJI_RE = re.compile(
    "["                                # emoji ranges (not exhaustive, good enough)
    "\U0001F600-\U0001F64F"
    "\U0001F300-\U0001F5FF"
    "\U0001F680-\U0001F6FF"
    "\U0001F700-\U0001F77F"
    "\U0001F780-\U0001F7FF"
    "\U0001F800-\U0001F8FF"
    "\U0001F900-\U0001F9FF"
    "\U0001FA00-\U0001FA6F"
    "\U0001FA70-\U0001FAFF"
    "☀-⛿✀-➿"
    "]", flags=re.UNICODE,
)

analyzer = SentimentIntensityAnalyzer()


def uppercase_ratio(s: str) -> float:
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return 0.0
    return sum(1 for c in letters if c.isupper()) / len(letters)


def contains_any(s: str, terms: List[str]) -> bool:
    s_low = s.lower()
    return any(t in s_low for t in terms)


def temporal_bin(pub: str, t_min: pd.Timestamp, t_max: pd.Timestamp) -> Optional[int]:
    """Return 0/1/2 for early/middle/late window. None if unparseable."""
    if not pub:
        return None
    try:
        t = pd.to_datetime(pub, utc=True)
    except Exception:
        return None
    span = (t_max - t_min).total_seconds()
    if span <= 0:
        return 0
    offset = (t - t_min).total_seconds()
    frac = offset / span
    if frac < 1 / 3:
        return 0
    if frac < 2 / 3:
        return 1
    return 2


def extract_movie_features(comments: List[Dict[str, Any]]) -> Dict[str, float]:
    """Given a list of comments for one movie, compute the feature dict."""
    n = len(comments)
    if n == 0:
        return {}

    # Per-comment fields
    texts = [(c.get("text") or "").strip() for c in comments]
    likes = np.array([int(c.get("like_count") or 0) for c in comments])
    replies = np.array([int(c.get("reply_count") or 0) for c in comments])
    lengths = np.array([len(t) for t in texts])
    pubs = [c.get("published_at") or "" for c in comments]

    # Sentiment
    compound = np.array([
        analyzer.polarity_scores(t)["compound"] if t else 0.0
        for t in texts
    ])

    # Hype / neg
    hype_mask = np.array([contains_any(t, HYPE_TERMS) for t in texts])
    nega_mask = np.array([contains_any(t, NEGA_TERMS) for t in texts])

    # Spam-like
    caps_mask = np.array([uppercase_ratio(t) > 0.4 for t in texts])
    emoji_counts = np.array([len(EMOJI_RE.findall(t)) for t in texts])
    short_mask = lengths < 20

    # Temporal windows
    parsed_times = pd.to_datetime(pubs, utc=True, errors="coerce")
    valid = parsed_times.notna()
    t_min = parsed_times[valid].min() if valid.any() else None
    t_max = parsed_times[valid].max() if valid.any() else None
    bins = np.array([
        temporal_bin(p, t_min, t_max) if (t_min is not None and t_max is not None) else None
        for p in pubs
    ], dtype=object)

    feats: Dict[str, float] = {
        "n_comments": float(n),
        "sent_mean": float(compound.mean()),
        "sent_std": float(compound.std()),
        "sent_pos_frac": float(np.mean(compound > 0.5)),
        "sent_neg_frac": float(np.mean(compound < -0.5)),

        "hype_frac": float(hype_mask.mean()),
        "nega_frac": float(nega_mask.mean()),

        "avg_comment_len": float(lengths.mean()),
        "median_like_count": float(np.median(likes)),
        "mean_like_count": float(likes.mean()),
        "reply_rate": float(replies.mean()),

        "caps_frac": float(caps_mask.mean()),
        "emoji_density": float(emoji_counts.mean()),
        "short_comment_frac": float(short_mask.mean()),

        # Credibility-weighted sentiment: comments with more likes get more weight.
        # Using log1p to damp the heavy-tailed like distribution.
        "cred_sent": float(
            np.average(compound, weights=np.log1p(likes) + 1.0)
        ),
    }

    # Temporal window features
    for window_idx, label in [(0, "early"), (1, "middle"), (2, "late")]:
        mask = bins == window_idx
        if mask.any():
            feats[f"sent_mean_{label}"] = float(compound[mask].mean())
            feats[f"hype_frac_{label}"] = float(hype_mask[mask].mean())
            feats[f"n_frac_{label}"] = float(mask.mean())
        else:
            feats[f"sent_mean_{label}"] = 0.0
            feats[f"hype_frac_{label}"] = 0.0
            feats[f"n_frac_{label}"] = 0.0

    # Novel: expectation-reality-proxy (just the hype minus neg, as a signed score)
    feats["hype_minus_neg"] = feats["hype_frac"] - feats["nega_frac"]

    return feats


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> Dict[str, Any]:
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
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": int(np.sum(y_true == c))}
        f1s.append(f1)
    return {"n": n, "accuracy": round(acc, 4),
            "macro_f1": round(float(np.mean(f1s)), 4), "per_class": per_class}


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main() -> None:
    if not os.path.exists(IN_JSONL):
        print(f"ERROR: {IN_JSONL} not found. Run build_final_dataset.py first.")
        sys.exit(1)

    print("Extracting per-movie features (VADER + heuristics)...")
    rows = []
    with open(IN_JSONL, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            rec = json.loads(line)
            feats = extract_movie_features(rec.get("comments", []))
            feats["tmdb_id"] = rec["tmdb_id"]
            feats["title"] = rec["title"]
            feats["rating_bucket"] = rec["rating_bucket"]
            feats["rating_bucket_label"] = rec["rating_bucket_label"]
            feats["split"] = rec["split"]
            rows.append(feats)
            if i % 20 == 0:
                print(f"  {i} movies processed")
    df = pd.DataFrame(rows)
    print(f"Feature matrix: {df.shape}")

    os.makedirs(OUT_DIR, exist_ok=True)
    df.to_csv(OUT_FEATS, index=False, encoding="utf-8")
    print(f"Saved features: {OUT_FEATS}")

    feature_cols = [c for c in df.columns if c not in
                    {"tmdb_id", "title", "rating_bucket", "rating_bucket_label", "split"}]
    print(f"Feature count: {len(feature_cols)}")

    train = df[df["split"] == "train"]
    val = df[df["split"] == "val"]
    test = df[df["split"] == "test"]
    n_classes = int(df["rating_bucket"].max()) + 1

    X_train = train[feature_cols].to_numpy()
    X_val = val[feature_cols].to_numpy()
    X_test = test[feature_cols].to_numpy()
    y_train = train["rating_bucket"].to_numpy()
    y_val = val["rating_bucket"].to_numpy()
    y_test = test["rating_bucket"].to_numpy()

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    results_all = {}

    # Random forest (no scaling needed but we have them; trees ignore scale)
    print("\nTraining RandomForest...")
    rf = RandomForestClassifier(
        n_estimators=500,
        max_depth=None,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf.fit(X_train, y_train)

    rf_results = {}
    for name, X, y in [("train", X_train, y_train),
                       ("val", X_val, y_val),
                       ("test", X_test, y_test)]:
        y_pred = rf.predict(X)
        m = evaluate(y, y_pred, n_classes)
        m["confusion_matrix"] = confusion_matrix(y, y_pred, n_classes).tolist()
        rf_results[name] = m
        print(f"  RF {name}: acc={m['accuracy']}  macroF1={m['macro_f1']}")

    # Top feature importances
    importances = sorted(zip(feature_cols, rf.feature_importances_),
                         key=lambda x: -x[1])
    print("\nTop 10 feature importances (RF):")
    for name, imp in importances[:10]:
        print(f"  {name:25} {imp:.4f}")

    # Logistic regression on scaled features
    print("\nTraining LogisticRegression on structured features...")
    lr = LogisticRegression(
        C=1.0, max_iter=5000, class_weight="balanced",
        solver="lbfgs", random_state=RANDOM_SEED,
    )
    lr.fit(X_train_s, y_train)
    lr_results = {}
    for name, X, y in [("train", X_train_s, y_train),
                       ("val", X_val_s, y_val),
                       ("test", X_test_s, y_test)]:
        y_pred = lr.predict(X)
        m = evaluate(y, y_pred, n_classes)
        m["confusion_matrix"] = confusion_matrix(y, y_pred, n_classes).tolist()
        lr_results[name] = m
        print(f"  LR {name}: acc={m['accuracy']}  macroF1={m['macro_f1']}")

    out = {
        "model": "structured_features",
        "n_features": len(feature_cols),
        "features": feature_cols,
        "results": {
            "random_forest": rf_results,
            "logistic_regression": lr_results,
        },
        "top_feature_importances_rf": [
            {"feature": n, "importance": round(float(i), 5)}
            for n, i in importances
        ],
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
