#!/usr/bin/env python3

import os
import sys
import time
import json
import requests
import pandas as pd
from typing import Dict, List, Any, Optional

# ============================================================
# CONFIG
# ============================================================

# TMDB API key — read from environment. See .env.example for setup.
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")

# Years to scrape
YEARS = [2020, 2021, 2022, 2023, 2024]

# Each discover page has up to 20 movies
PAGES_PER_YEAR = 10

# Filters
LANGUAGE = "en-US"
ORIGINAL_LANGUAGE = "en"
MIN_VOTE_COUNT = 200
MIN_POPULARITY = 5.0
REQUIRE_RELEASED_ONLY = True
MIN_TRAILERS = 2
MAX_TRAILERS_TO_STORE = 4
SLEEP_SECS = 0.25

# Output paths
OUT_DIR = "data/raw"
OUT_CSV = os.path.join(OUT_DIR, "movies_master_tmdb.csv")
OUT_JSON = os.path.join(OUT_DIR, "movies_master_tmdb.json")

BASE_URL = "https://api.themoviedb.org/3"


def die(msg: str) -> None:
    print(f"ERROR: {msg}")
    sys.exit(1)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def tmdb_get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if not TMDB_API_KEY:
        die("TMDB_API_KEY is missing.")

    url = f"{BASE_URL}{endpoint}"
    params = params.copy() if params else {}
    params["api_key"] = TMDB_API_KEY

    resp = requests.get(url, params=params, timeout=30)
    if resp.status_code != 200:
        print(f"\nRequest failed: {resp.status_code}")
        print(f"URL: {resp.url}")
        print(f"Body: {resp.text[:1000]}")
        resp.raise_for_status()

    time.sleep(SLEEP_SECS)
    return resp.json()


def get_discover_movies_for_year(year: int, page: int) -> List[Dict[str, Any]]:
    params = {
        "language": LANGUAGE,
        "sort_by": "popularity.desc",
        "include_adult": "false",
        "include_video": "false",
        "page": page,
        "primary_release_year": year,
        "vote_count.gte": MIN_VOTE_COUNT,
    }
    data = tmdb_get("/discover/movie", params)
    return data.get("results", [])


def get_movie_details(tmdb_id: int) -> Dict[str, Any]:
    params = {"language": LANGUAGE}
    return tmdb_get(f"/movie/{tmdb_id}", params)


def get_movie_videos(tmdb_id: int) -> List[Dict[str, Any]]:
    params = {"language": LANGUAGE}
    data = tmdb_get(f"/movie/{tmdb_id}/videos", params)
    return data.get("results", [])


def extract_official_youtube_trailers(videos: List[Dict[str, Any]]) -> List[str]:
    scored = []

    for v in videos:
        site = (v.get("site") or "").strip()
        vtype = (v.get("type") or "").strip()
        official = bool(v.get("official", False))
        key = (v.get("key") or "").strip()
        name = (v.get("name") or "").strip().lower()

        if site != "YouTube" or not key:
            continue

        if not official:
            continue
        if vtype != "Trailer":
            continue

        score = 10
        if "official trailer" in name:
            score += 3
        elif "trailer" in name:
            score += 2

        scored.append((score, key))

    scored.sort(reverse=True, key=lambda x: x[0])

    out = []
    seen = set()
    for _, key in scored:
        if key not in seen:
            seen.add(key)
            out.append(key)

    return out[:MAX_TRAILERS_TO_STORE]


def movie_passes_basic_filters(details: Dict[str, Any]) -> bool:
    release_date = (details.get("release_date") or "").strip()
    original_language = (details.get("original_language") or "").strip()
    popularity = float(details.get("popularity") or 0.0)
    status = (details.get("status") or "").strip()

    if REQUIRE_RELEASED_ONLY and status.lower() != "released":
        return False
    if not release_date:
        return False
    if original_language != ORIGINAL_LANGUAGE:
        return False
    if popularity < MIN_POPULARITY:
        return False

    return True


def build_row(details: Dict[str, Any], trailer_ids: List[str]) -> Dict[str, Any]:
    genre_ids = [g["id"] for g in details.get("genres", []) if "id" in g]

    release_date = details.get("release_date", "")
    year = int(release_date[:4]) if release_date else None

    return {
        "tmdb_id": details.get("id"),
        "title": details.get("title"),
        "release_date": release_date,
        "year": year,
        "language": details.get("original_language"),
        "popularity": details.get("popularity"),
        "tmdb_vote_count": details.get("vote_count"),
        "tmdb_vote_average": details.get("vote_average"),
        "genre_ids": json.dumps(genre_ids, ensure_ascii=False),
        "trailer_count": len(trailer_ids),
        "youtube_trailer_ids": json.dumps(trailer_ids, ensure_ascii=False),
        "letterboxd_url": "",
        "letterboxd_rating": "",
        "rating_bucket": "",
        "include_flag": 1,
        "exclude_reason": "",
    }


def main() -> None:
    ensure_dir(OUT_DIR)

    print("Collecting candidate movies from TMDB...")
    candidate_ids = []
    seen = set()

    for year in YEARS:
        print(f"\nYear {year}")
        for page in range(1, PAGES_PER_YEAR + 1):
            print(f"  discover page {page}/{PAGES_PER_YEAR}")
            results = get_discover_movies_for_year(year, page)

            for item in results:
                tmdb_id = item.get("id")
                if tmdb_id and tmdb_id not in seen:
                    seen.add(tmdb_id)
                    candidate_ids.append(tmdb_id)

    print(f"\nDiscovered {len(candidate_ids)} unique candidate movies before detailed filtering.")

    rows = []
    kept = 0
    skipped = 0

    for i, tmdb_id in enumerate(candidate_ids, start=1):
        try:
            print(f"[{i}/{len(candidate_ids)}] movie_id={tmdb_id}")
            details = get_movie_details(tmdb_id)

            if not movie_passes_basic_filters(details):
                skipped += 1
                continue

            videos = get_movie_videos(tmdb_id)
            trailer_ids = extract_official_youtube_trailers(videos)

            if len(trailer_ids) < MIN_TRAILERS:
                skipped += 1
                continue

            row = build_row(details, trailer_ids)
            rows.append(row)
            kept += 1

        except Exception as e:
            print(f"Skipping {tmdb_id} due to error: {e}")
            skipped += 1
            continue

    if not rows:
        die("No movies collected. Lower filters like MIN_TRAILERS or MIN_VOTE_COUNT.")

    df = pd.DataFrame(rows)
    df = df.sort_values(
        by=["year", "popularity", "tmdb_vote_count"],
        ascending=[True, False, False]
    ).reset_index(drop=True)

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Kept: {kept}")
    print(f"Skipped: {skipped}")
    print(f"CSV saved to:  {OUT_CSV}")
    print(f"JSON saved to: {OUT_JSON}")
    print("\nPreview:")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()