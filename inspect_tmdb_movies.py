#!/usr/bin/env python3
"""
Step 1b: Sanity-check the TMDB movie list before spending YouTube API quota.

Loads data/raw/movies_master_tmdb.csv and prints:
  - total row count
  - breakdown per year
  - language sanity check (should be all 'en')
  - trailer-count histogram
  - popularity / vote_count / vote_average distributions
  - release_date min / max / any missing
  - sample of 10 rows
  - list of any rows with suspicious values
"""

import json
import os
import sys

import pandas as pd

IN_CSV = "data/raw/movies_master_tmdb.csv"


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def main() -> None:
    if not os.path.exists(IN_CSV):
        print(f"ERROR: {IN_CSV} not found. Run collect_tmdb_movies.py first.")
        sys.exit(1)

    df = pd.read_csv(IN_CSV)

    section("Overview")
    print(f"Total movies: {len(df)}")
    print(f"Columns: {list(df.columns)}")

    section("Per-year counts")
    print(df["year"].value_counts().sort_index().to_string())

    section("Language sanity check (expect all 'en')")
    print(df["language"].value_counts().to_string())
    non_en = df[df["language"] != "en"]
    if len(non_en) > 0:
        print(f"\nWARNING: {len(non_en)} non-English rows leaked through:")
        print(non_en[["tmdb_id", "title", "language"]].to_string(index=False))

    section("Trailer-count histogram")
    print(df["trailer_count"].value_counts().sort_index().to_string())
    print(f"\nMin trailers: {df['trailer_count'].min()}")
    print(f"Max trailers: {df['trailer_count'].max()}")
    print(f"Mean trailers: {df['trailer_count'].mean():.2f}")

    section("Popularity distribution")
    print(df["popularity"].describe().to_string())

    section("TMDB vote count distribution")
    print(df["tmdb_vote_count"].describe().to_string())

    section("TMDB vote average distribution")
    print(df["tmdb_vote_average"].describe().to_string())

    section("Release date range")
    release_dates = pd.to_datetime(df["release_date"], errors="coerce")
    print(f"Earliest: {release_dates.min()}")
    print(f"Latest:   {release_dates.max()}")
    missing = release_dates.isna().sum()
    print(f"Missing release dates: {missing}")

    section("Suspicious rows")
    flags = []
    for _, row in df.iterrows():
        try:
            trailer_ids = json.loads(row["youtube_trailer_ids"])
        except Exception:
            flags.append((row["tmdb_id"], row["title"], "unparseable youtube_trailer_ids"))
            continue
        if not isinstance(trailer_ids, list) or len(trailer_ids) < 2:
            flags.append((row["tmdb_id"], row["title"], f"too few trailers: {len(trailer_ids)}"))
        if len(set(trailer_ids)) != len(trailer_ids):
            flags.append((row["tmdb_id"], row["title"], "duplicate trailer IDs"))
        if not row.get("release_date"):
            flags.append((row["tmdb_id"], row["title"], "missing release_date"))

    if flags:
        print(f"{len(flags)} suspicious rows:")
        for tmdb_id, title, reason in flags[:20]:
            print(f"  {tmdb_id}  {title!r}  -> {reason}")
        if len(flags) > 20:
            print(f"  ... and {len(flags) - 20} more")
    else:
        print("No suspicious rows found.")

    section("Sample (first 10)")
    preview_cols = [
        "tmdb_id", "title", "year", "release_date",
        "popularity", "tmdb_vote_count", "tmdb_vote_average", "trailer_count",
    ]
    print(df[preview_cols].head(10).to_string(index=False))

    section("Done")
    print(f"Ready for step 2 (YouTube comment scraping) on {len(df)} movies.")


if __name__ == "__main__":
    main()
