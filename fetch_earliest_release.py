#!/usr/bin/env python3
"""
Step 1c: For each movie in movies_master_tmdb.csv, fetch TMDB /movie/{id}/release_dates
and determine the earliest theatrical release date across all regions.

This earliest date is the TRUE leakage boundary for step 2 (YouTube comment scraping):
any comment posted on or after this date could, in principle, be influenced by the movie
being viewable somewhere in the world (festival, early international, premiere, etc.).

TMDB release_dates "type" codes:
    1 = Premiere
    2 = Theatrical (limited)
    3 = Theatrical
    4 = Digital
    5 = Physical
    6 = TV

We consider types {1, 2, 3} as "the movie became publicly viewable" events.
Premiere (type 1) is included because it's the first public screening and comments
referencing it would be post-experience.

Output: overwrites data/raw/movies_master_tmdb.csv with two new columns added:
    - earliest_release_date    (YYYY-MM-DD, UTC interpretation)
    - earliest_release_type    (int: 1, 2, or 3 — which event was earliest)
    - earliest_release_country (ISO country code where it first happened)

If no theatrical/premiere date is found, falls back to the existing release_date.
"""

import json
import os
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
BASE_URL = "https://api.themoviedb.org/3"
SLEEP_SECS = 0.25

IN_CSV = "data/raw/movies_master_tmdb.csv"
OUT_CSV = "data/raw/movies_master_tmdb.csv"  # overwrite in place

RELEASE_TYPES_TO_CONSIDER = {1, 2, 3}  # Premiere, Theatrical (limited), Theatrical


def tmdb_get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{BASE_URL}{endpoint}"
    params = params.copy() if params else {}
    params["api_key"] = TMDB_API_KEY
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    time.sleep(SLEEP_SECS)
    return resp.json()


def extract_earliest_release(
    release_dates_payload: Dict[str, Any],
) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    """
    Walk the release_dates response and return (earliest_date, type, country).
    Dates in TMDB are ISO strings like '2020-07-29T00:00:00.000Z'; we keep the date part.
    """
    results = release_dates_payload.get("results") or []
    best: Optional[Tuple[str, int, str]] = None  # (date, type, country)

    for region in results:
        country = region.get("iso_3166_1") or ""
        for entry in region.get("release_dates", []) or []:
            rtype = entry.get("type")
            raw_date = entry.get("release_date") or ""
            if rtype not in RELEASE_TYPES_TO_CONSIDER:
                continue
            if not raw_date:
                continue
            date_part = raw_date[:10]
            candidate = (date_part, rtype, country)
            if best is None or candidate[0] < best[0]:
                best = candidate

    if best is None:
        return None, None, None
    return best[0], best[1], best[2]


def main() -> None:
    if not os.path.exists(IN_CSV):
        print(f"ERROR: {IN_CSV} not found.")
        sys.exit(1)

    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} movies from {IN_CSV}")

    earliest_dates: List[Optional[str]] = []
    earliest_types: List[Optional[int]] = []
    earliest_countries: List[Optional[str]] = []

    for i, row in enumerate(df.itertuples(index=False), start=1):
        tmdb_id = int(row.tmdb_id)
        try:
            print(f"[{i}/{len(df)}] tmdb_id={tmdb_id}  {row.title[:60]}")
            payload = tmdb_get(f"/movie/{tmdb_id}/release_dates")
            date, rtype, country = extract_earliest_release(payload)

            fallback_date = str(row.release_date) if row.release_date else None
            if date is None:
                date = fallback_date
                rtype = None
                country = None

            earliest_dates.append(date)
            earliest_types.append(rtype)
            earliest_countries.append(country)

        except Exception as e:
            print(f"  ERROR for {tmdb_id}: {e}")
            earliest_dates.append(str(row.release_date) if row.release_date else None)
            earliest_types.append(None)
            earliest_countries.append(None)

    df["earliest_release_date"] = earliest_dates
    df["earliest_release_type"] = earliest_types
    df["earliest_release_country"] = earliest_countries

    # Report how many earliest dates differ from the original TMDB release_date
    differs_mask = df["earliest_release_date"].astype(str) != df["release_date"].astype(str)
    n_differs = int(differs_mask.sum())

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    print(f"\nSaved updated CSV to {OUT_CSV}")
    print(f"Movies where earliest release date differs from TMDB release_date: {n_differs}/{len(df)}")

    if n_differs:
        print("\nExamples where the dates differ (first 10):")
        cols = ["tmdb_id", "title", "release_date", "earliest_release_date",
                "earliest_release_type", "earliest_release_country"]
        print(df.loc[differs_mask, cols].head(10).to_string(index=False))

    # Quick distribution of earliest_release_type
    print("\nEarliest release type distribution (1=Premiere, 2=LimitedTheatrical, 3=Theatrical):")
    print(df["earliest_release_type"].value_counts(dropna=False).to_string())


if __name__ == "__main__":
    main()
