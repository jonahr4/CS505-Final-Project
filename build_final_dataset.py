#!/usr/bin/env python3
"""
Step 3: Assemble the final per-movie dataset that models will train on.

Does three things in one pass:
    1. Filters movies with fewer than MIN_COMMENTS_PER_MOVIE pre-release comments.
    2. Assigns each movie a 5-class rating bucket based on Letterboxd rating.
    3. Creates a movie-level train/val/test split stratified by bucket.

Also aggregates each movie's pre-release comments into a single concatenated
document (for TF-IDF) and preserves the per-comment list (for transformer
models and structured feature extraction).

Outputs:
    data/processed/movies_dataset.csv
        One row per kept movie. Columns:
            tmdb_id, title, year, release_date,
            letterboxd_rating, letterboxd_rating_count,
            rating_bucket (int 0-4), rating_bucket_label,
            n_comments, n_trailers_used,
            split  ('train' / 'val' / 'test'),
            comments_text  (all comments concatenated, space-separated)

    data/processed/movie_comments.jsonl
        One JSON object per movie with the full list of its comments,
        including per-comment metadata (published_at, like_count, author).
        Used by the structured-features and transformer models which need
        per-comment data instead of one aggregated blob.

Choices (justified in comments where non-obvious):
    - MIN_COMMENTS_PER_MOVIE = 30 (drops the zero and very-thin movies; trade-off
      between dataset size and per-movie signal strength)
    - Rating thresholds chosen to roughly balance class sizes on this dataset;
      the top threshold is 3.5 rather than 3.6 so "great" has ~22 members
      instead of ~16 (critical for train/val/test splits staying meaningful)
    - 70/15/15 split, stratified by rating_bucket, seed=42
    - Split is at the MOVIE level — this is the whole point of the project,
      a comment-level split would leak a movie's signal across train and test
"""

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

IN_MOVIES_CSV = "data/raw/movies_selected.csv"
IN_COMMENTS_INDEX = "data/raw/comments_index.csv"
COMMENTS_DIR = "data/raw/comments"

OUT_DIR = "data/processed"
OUT_CSV = os.path.join(OUT_DIR, "movies_dataset.csv")
OUT_JSONL = os.path.join(OUT_DIR, "movie_comments.jsonl")

MIN_COMMENTS_PER_MOVIE = 30

# Rating buckets (lower bound inclusive, upper bound exclusive)
# Calibrated on actual distribution to roughly balance class sizes.
RATING_BUCKETS = [
    (None, 2.3,  0, "bad"),
    (2.3,  2.75, 1, "mediocre"),
    (2.75, 3.15, 2, "average"),
    (3.15, 3.5,  3, "good"),
    (3.5,  None, 4, "great"),
]

SPLIT_RATIOS = {"train": 0.70, "val": 0.15, "test": 0.15}
RANDOM_SEED = 42


def assign_bucket(rating: float) -> (int, str):
    for lo, hi, idx, label in RATING_BUCKETS:
        if (lo is None or rating >= lo) and (hi is None or rating < hi):
            return idx, label
    raise ValueError(f"No bucket for rating {rating}")


def load_movie_comments(tmdb_id: int) -> List[Dict[str, Any]]:
    """Load and flatten all comments for a movie across its trailer JSONs."""
    movie_dir = Path(COMMENTS_DIR) / str(tmdb_id)
    if not movie_dir.exists():
        return []
    comments: List[Dict[str, Any]] = []
    for trailer_file in sorted(movie_dir.glob("*.json")):
        with open(trailer_file, "r", encoding="utf-8") as f:
            payload = json.load(f)
        for c in payload.get("comments", []):
            c_out = dict(c)
            c_out["trailer_id"] = payload.get("trailer_id")
            comments.append(c_out)
    return comments


def stratified_split(df: pd.DataFrame, rng: np.random.Generator) -> pd.Series:
    """
    Stratified split by rating_bucket. Within each bucket, shuffle and split
    into train/val/test by SPLIT_RATIOS. Returns a Series of 'train'/'val'/'test'
    aligned with df.index.
    """
    splits = pd.Series(index=df.index, dtype=object)
    for bucket in sorted(df["rating_bucket"].unique()):
        members = df[df["rating_bucket"] == bucket].index.tolist()
        rng.shuffle(members)
        n = len(members)
        n_train = int(round(n * SPLIT_RATIOS["train"]))
        n_val = int(round(n * SPLIT_RATIOS["val"]))
        # any remainder goes to test
        splits.loc[members[:n_train]] = "train"
        splits.loc[members[n_train:n_train + n_val]] = "val"
        splits.loc[members[n_train + n_val:]] = "test"
    return splits


def main() -> None:
    if not os.path.exists(IN_MOVIES_CSV):
        print(f"ERROR: {IN_MOVIES_CSV} not found.")
        sys.exit(1)
    if not os.path.exists(IN_COMMENTS_INDEX):
        print(f"ERROR: {IN_COMMENTS_INDEX} not found.")
        sys.exit(1)

    movies = pd.read_csv(IN_MOVIES_CSV)
    idx = pd.read_csv(IN_COMMENTS_INDEX)
    print(f"Loaded {len(movies)} selected movies and {len(idx)} trailer rows")

    # Per-movie: total kept comments, number of trailers with at least 1 kept comment
    kept_per_movie = idx.groupby("tmdb_id")["kept_pre_release"].sum()
    trailers_used = idx[idx["kept_pre_release"] > 0].groupby("tmdb_id").size()

    movies["n_comments"] = movies["tmdb_id"].map(kept_per_movie).fillna(0).astype(int)
    movies["n_trailers_used"] = movies["tmdb_id"].map(trailers_used).fillna(0).astype(int)

    # Filter movies with enough comments
    before = len(movies)
    kept = movies[movies["n_comments"] >= MIN_COMMENTS_PER_MOVIE].copy()
    print(f"Filter: {before} -> {len(kept)} movies "
          f"(dropped {before - len(kept)} with < {MIN_COMMENTS_PER_MOVIE} comments)")

    # Parse numeric ratings + assign buckets
    kept["letterboxd_rating"] = pd.to_numeric(kept["letterboxd_rating"], errors="coerce")
    missing_rating = kept["letterboxd_rating"].isna().sum()
    if missing_rating:
        print(f"WARNING: {missing_rating} kept movies have no Letterboxd rating — dropping")
        kept = kept[kept["letterboxd_rating"].notna()]

    bucket_assignments = kept["letterboxd_rating"].apply(assign_bucket)
    kept["rating_bucket"] = [b[0] for b in bucket_assignments]
    kept["rating_bucket_label"] = [b[1] for b in bucket_assignments]

    print("\nBucket distribution (post-filter):")
    for lo, hi, idx_, label in RATING_BUCKETS:
        n = (kept["rating_bucket"] == idx_).sum()
        rng_str = f"[{lo if lo is not None else 'inf-'}, {hi if hi is not None else '+inf'})"
        print(f"  {idx_}={label:10} {rng_str:18}  {n:>3} movies")

    # Stratified movie-level split
    rng = np.random.default_rng(RANDOM_SEED)
    kept = kept.reset_index(drop=True)
    kept["split"] = stratified_split(kept, rng)

    print("\nSplit sizes:")
    print(kept["split"].value_counts().to_string())
    print("\nBucket × split:")
    print(pd.crosstab(kept["rating_bucket_label"], kept["split"]).to_string())

    # Aggregate comments per movie
    print("\nAggregating comments per movie...")
    os.makedirs(OUT_DIR, exist_ok=True)
    out_jsonl = open(OUT_JSONL, "w", encoding="utf-8")

    comments_texts: List[str] = []
    n_comments_verified: List[int] = []

    for i, row in enumerate(kept.itertuples(index=False), start=1):
        tmdb_id = int(row.tmdb_id)
        comments = load_movie_comments(tmdb_id)

        # Sanity: n_comments from index should match what we load
        if len(comments) != row.n_comments:
            print(f"  note: {tmdb_id} {row.title!r}: index said {row.n_comments}, "
                  f"loaded {len(comments)}")

        n_comments_verified.append(len(comments))

        # One concatenated text blob (for TF-IDF)
        blob = " ".join(
            (c.get("text") or "").replace("\n", " ").strip()
            for c in comments
        )
        comments_texts.append(blob)

        # Per-comment JSONL
        out_jsonl.write(json.dumps({
            "tmdb_id": tmdb_id,
            "title": row.title,
            "year": int(row.year),
            "release_date": str(row.release_date),
            "letterboxd_rating": float(row.letterboxd_rating),
            "rating_bucket": int(row.rating_bucket),
            "rating_bucket_label": row.rating_bucket_label,
            "split": row.split,
            "comments": comments,
        }, ensure_ascii=False) + "\n")

        if i % 20 == 0:
            print(f"  {i}/{len(kept)} movies aggregated")

    out_jsonl.close()

    kept["comments_text"] = comments_texts
    kept["n_comments_verified"] = n_comments_verified

    # Select/reorder final columns
    final_cols = [
        "tmdb_id", "title", "year", "release_date",
        "letterboxd_rating", "letterboxd_rating_count",
        "rating_bucket", "rating_bucket_label",
        "n_comments", "n_trailers_used",
        "split", "comments_text",
    ]
    final = kept[final_cols]
    final.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"\nWrote: {OUT_CSV} ({len(final)} rows)")
    print(f"Wrote: {OUT_JSONL}")

    # Quick sanity: total comment chars, avg text length
    total_chars = final["comments_text"].str.len().sum()
    print(f"\nTotal aggregated comment text: {total_chars:,} chars")
    print(f"Avg per movie: {int(total_chars / len(final)):,} chars")


if __name__ == "__main__":
    main()
