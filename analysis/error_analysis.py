#!/usr/bin/env python3
"""
Item 4 from midterm plan: qualitative error analysis.

Identify the test movies that EVERY model gets wrong, and look for patterns.

Loads the saved confusion matrices + per-movie predictions where available.
For models without per-movie predictions saved, recompute them quickly.

Outputs:
    results/error_analysis.json
    results/error_analysis.md  (human-readable summary for the report)
"""

import json
import os
import sys
from typing import Dict, List

import numpy as np
import pandas as pd

PROC_CSV = "data/processed/movies_dataset.csv"
PER_MOVIE_PREDS = {
    "DistilBERT_PerComment_FT":  "results/distilbert_finetune/per_movie_predictions.csv",
    "DistilBERT_MovieLevel_FT":  "results/distilbert_finetune_movie/per_movie_predictions.csv",
}
OUT_JSON = "results/error_analysis.json"
OUT_MD = "results/error_analysis.md"


def get_test_predictions_from_baseline_models() -> Dict[str, Dict[int, int]]:
    """
    Re-run TF-IDF and structured-LR on the dataset to get per-test-movie
    predictions (we didn't save them originally).
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(PROC_CSV)
    df["comments_text"] = df["comments_text"].fillna("").astype(str)
    train = df[df["split"] == "train"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    y_tr = train["rating_bucket"].to_numpy()
    y_te = test["rating_bucket"].to_numpy()

    out: Dict[str, Dict[int, int]] = {}

    # TF-IDF
    vec = TfidfVectorizer(max_features=20000, ngram_range=(1, 2),
                          min_df=2, max_df=0.95, stop_words="english",
                          lowercase=True, strip_accents="unicode")
    Xtr = vec.fit_transform(train["comments_text"])
    Xte = vec.transform(test["comments_text"])
    clf = LogisticRegression(C=1.0, max_iter=2000,
                             class_weight="balanced", solver="lbfgs",
                             random_state=42)
    clf.fit(Xtr, y_tr)
    preds = clf.predict(Xte)
    out["TF-IDF"] = {int(t): int(p) for t, p in zip(test["tmdb_id"], preds)}

    # Structured-LR
    feats = pd.read_csv("results/structured_features/features.csv")
    feat_cols = [c for c in feats.columns if c not in
                 {"tmdb_id", "title", "rating_bucket",
                  "rating_bucket_label", "split"}]
    feat_train = feats[feats["split"] == "train"].set_index("tmdb_id")
    feat_test = feats[feats["split"] == "test"].set_index("tmdb_id")
    Xtr_s = feat_train.loc[train["tmdb_id"]][feat_cols].to_numpy()
    Xte_s = feat_test.loc[test["tmdb_id"]][feat_cols].to_numpy()
    sc = StandardScaler().fit(Xtr_s)
    clf2 = LogisticRegression(C=1.0, max_iter=5000,
                              class_weight="balanced", solver="lbfgs",
                              random_state=42)
    clf2.fit(sc.transform(Xtr_s), y_tr)
    preds2 = clf2.predict(sc.transform(Xte_s))
    out["Structured-LR"] = {int(t): int(p) for t, p in zip(test["tmdb_id"], preds2)}

    return out


def main():
    if not os.path.exists(PROC_CSV):
        print(f"ERROR: {PROC_CSV} missing"); sys.exit(1)

    df = pd.read_csv(PROC_CSV)
    test_df = df[df["split"] == "test"].reset_index(drop=True)
    print(f"Test set: {len(test_df)} movies")

    # Collect predictions from all models we have
    all_preds: Dict[str, Dict[int, int]] = {}

    # Baseline models we re-run
    baseline_preds = get_test_predictions_from_baseline_models()
    all_preds.update(baseline_preds)

    # Saved per-movie predictions (fine-tuned BERTs)
    for name, path in PER_MOVIE_PREDS.items():
        if not os.path.exists(path):
            print(f"  [skip] {name} preds missing at {path}")
            continue
        p = pd.read_csv(path)
        p_test = p[p["split"] == "test"]
        all_preds[name] = {int(r["tmdb_id"]): int(r["predicted_bucket"])
                          for _, r in p_test.iterrows()}

    print(f"Models with predictions: {list(all_preds.keys())}")

    # Build per-movie error report
    rows = []
    for _, row in test_df.iterrows():
        tmdb_id = int(row["tmdb_id"])
        true_bucket = int(row["rating_bucket"])
        per_model = {}
        wrong_count = 0
        for model_name, preds in all_preds.items():
            pred = preds.get(tmdb_id, -1)
            per_model[model_name] = pred
            if pred != true_bucket:
                wrong_count += 1
        rows.append({
            "tmdb_id": tmdb_id, "title": row["title"], "year": int(row["year"]),
            "letterboxd_rating": float(row["letterboxd_rating"]),
            "true_bucket": true_bucket, "true_label": row["rating_bucket_label"],
            "n_comments": int(row["n_comments"]),
            "n_models_wrong": wrong_count, **per_model,
        })
    err_df = pd.DataFrame(rows).sort_values(
        ["n_models_wrong", "title"], ascending=[False, True]
    )

    print("\nTop 10 hardest movies (most models wrong):")
    print(err_df.head(10)[["title", "year", "true_label",
                           "letterboxd_rating", "n_models_wrong",
                           "n_comments"]].to_string(index=False))

    # Movies every model got wrong
    n_models = len(all_preds)
    universal_misses = err_df[err_df["n_models_wrong"] == n_models]
    print(f"\nMovies missed by ALL {n_models} models: {len(universal_misses)}")
    if len(universal_misses):
        print(universal_misses[["title", "year", "true_label",
                                "letterboxd_rating"]].to_string(index=False))

    # Direction-of-error analysis: do models tend to predict toward the mean?
    print("\nOff-by-X distribution across all (model, movie) pairs:")
    diffs = []
    for _, row in err_df.iterrows():
        for model_name in all_preds:
            pred = row[model_name]
            if pred >= 0:
                diffs.append(pred - row["true_bucket"])
    diffs = np.array(diffs)
    for d in range(-4, 5):
        n = int(np.sum(diffs == d))
        bar = "#" * (n // 2)
        print(f"  pred-true = {d:+d}  {n:>3}  {bar}")

    # Save
    out = {
        "n_test_movies": len(test_df),
        "n_models_compared": n_models,
        "models": list(all_preds.keys()),
        "movies_missed_by_all_models": universal_misses.to_dict("records"),
        "hardest_movies_top10": err_df.head(10).to_dict("records"),
        "off_by_X_counts": {int(d): int(np.sum(diffs == d)) for d in range(-4, 5)},
        "mean_signed_error": round(float(diffs.mean()), 4),
        "mean_abs_error": round(float(np.abs(diffs).mean()), 4),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    err_df.to_csv("results/per_movie_error_table.csv", index=False)

    # Markdown summary for the report
    lines = [
        "# Error Analysis (test set, n=" + str(len(test_df)) + ")\n",
        f"Models compared: {', '.join(all_preds.keys())}\n",
        f"\n## Movies missed by ALL {n_models} models ({len(universal_misses)})\n",
    ]
    for _, r in universal_misses.iterrows():
        preds_str = " | ".join(f"{m}={r[m]}" for m in all_preds)
        lines.append(f"- **{r['title']}** ({r['year']}) — true: {r['true_label']} "
                     f"(rating {r['letterboxd_rating']:.2f}); preds: {preds_str}\n")
    lines.append("\n## Top 10 hardest movies\n\n")
    lines.append("| Title | Year | True | Rating | # Models Wrong | # Comments |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for _, r in err_df.head(10).iterrows():
        lines.append(f"| {r['title']} | {r['year']} | {r['true_label']} | "
                     f"{r['letterboxd_rating']:.2f} | {r['n_models_wrong']}/{n_models} | "
                     f"{r['n_comments']} |\n")

    lines.append("\n## Off-by-X error distribution (across all model×movie pairs)\n\n")
    lines.append("| pred − true | count |\n|---|---|\n")
    for d in range(-4, 5):
        n = int(np.sum(diffs == d))
        lines.append(f"| {d:+d} | {n} |\n")
    lines.append(f"\nMean signed error: {diffs.mean():+.3f}  ")
    lines.append(f"(positive = models tend to over-predict the rating bucket)\n")
    with open(OUT_MD, "w") as f:
        f.writelines(lines)
    print(f"\nSaved: {OUT_JSON}")
    print(f"Saved: results/per_movie_error_table.csv")
    print(f"Saved: {OUT_MD}")


if __name__ == "__main__":
    main()
