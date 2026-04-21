#!/usr/bin/env python3
"""
Step 1d: Select a stratified sample of 150 movies from the 388-candidate pool.

Strategy:
    - Target: 30 movies per year for 2020-2024 (150 total).
    - Within each year, split candidates into 3 popularity tiers (tertiles)
      and sample ~equally from each tier so the final set has a mix of
      blockbusters, mid-tier, and lower-popularity releases.
    - Use a fixed random seed so the selection is reproducible.

Output:
    data/raw/movies_selected.csv  -- same schema as movies_master_tmdb.csv,
                                     just the 150 rows that will be used
                                     for YouTube comment scraping.

The master CSV is untouched; the remaining 238 movies stay as a reserve pool
in case we want to expand the dataset later.
"""

import os
import sys

import numpy as np
import pandas as pd

RANDOM_SEED = 42
TARGET_TOTAL = 150
MOVIES_PER_YEAR = 30
TIERS_PER_YEAR = 3  # -> 10 movies per (year, tier)

IN_CSV = "data/raw/movies_master_tmdb.csv"
OUT_CSV = "data/raw/movies_selected.csv"


def main() -> None:
    if not os.path.exists(IN_CSV):
        print(f"ERROR: {IN_CSV} not found.")
        sys.exit(1)

    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} candidate movies")

    rng = np.random.default_rng(RANDOM_SEED)
    per_tier = MOVIES_PER_YEAR // TIERS_PER_YEAR  # 10

    selected_parts = []

    for year in sorted(df["year"].unique()):
        year_df = df[df["year"] == year].copy()
        n_year = len(year_df)

        # Assign to tertiles on popularity. qcut gives 3 equal-size buckets.
        # If a year has fewer than TIERS_PER_YEAR*2 movies, skip stratification
        # and just sample uniformly.
        if n_year >= TIERS_PER_YEAR * 2:
            year_df["pop_tier"] = pd.qcut(
                year_df["popularity"], q=TIERS_PER_YEAR, labels=False, duplicates="drop"
            )
        else:
            year_df["pop_tier"] = 0

        tiers_present = sorted(year_df["pop_tier"].dropna().unique())
        year_picks = []

        for tier in tiers_present:
            tier_df = year_df[year_df["pop_tier"] == tier]
            n_needed = per_tier
            if len(tier_df) <= n_needed:
                year_picks.append(tier_df)
                print(f"  year={year} tier={int(tier)}: took all {len(tier_df)} "
                      f"(wanted {n_needed})")
            else:
                pick_idx = rng.choice(tier_df.index, size=n_needed, replace=False)
                year_picks.append(tier_df.loc[pick_idx])
                print(f"  year={year} tier={int(tier)}: sampled {n_needed} "
                      f"of {len(tier_df)}")

        year_combined = pd.concat(year_picks)

        # If a year ended up short (e.g. a tier had fewer than 10 candidates),
        # top up from remaining movies in that year, preferring higher popularity.
        if len(year_combined) < MOVIES_PER_YEAR:
            shortfall = MOVIES_PER_YEAR - len(year_combined)
            remaining = year_df.drop(year_combined.index).sort_values(
                "popularity", ascending=False
            )
            topup = remaining.head(shortfall)
            year_combined = pd.concat([year_combined, topup])
            print(f"  year={year}: topped up with {shortfall} extra "
                  f"(needed {MOVIES_PER_YEAR}, tiers gave {MOVIES_PER_YEAR - shortfall})")

        print(f"  year={year}: selected {len(year_combined)} total")
        selected_parts.append(year_combined)

    selected = pd.concat(selected_parts)
    selected = selected.drop(columns=["pop_tier"], errors="ignore")
    selected = selected.sort_values(
        by=["year", "popularity"], ascending=[True, False]
    ).reset_index(drop=True)

    # Truncate or log if we're over/under target
    if len(selected) != TARGET_TOTAL:
        print(f"\nNote: selected {len(selected)} (target {TARGET_TOTAL})")

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    selected.to_csv(OUT_CSV, index=False, encoding="utf-8")

    print(f"\nWrote {len(selected)} movies to {OUT_CSV}")
    print("\nPer-year counts in selected set:")
    print(selected["year"].value_counts().sort_index().to_string())
    print("\nPopularity summary per year:")
    print(selected.groupby("year")["popularity"].describe()[["min", "50%", "max"]].to_string())
    print("\nTotal trailers to scrape:", int(selected["trailer_count"].sum()))


if __name__ == "__main__":
    main()
