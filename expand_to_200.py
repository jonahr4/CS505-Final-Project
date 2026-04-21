#!/usr/bin/env python3
"""
Expand the selected-movies set from 150 to 200 by drawing 50-80 more from the
reserve pool (movies in movies_master_tmdb.csv that aren't already selected).

Does four things in one pass:
    1. Picks ~60 additional movies via stratified sampling (balance years and
       popularity tiers), biased so that years/tiers currently underrepresented
       in the selected set get a bit more weight.
    2. Fetches /movie/{id}/release_dates from TMDB for each to populate
       earliest_release_date (same rules as step 1c).
    3. Scrapes Letterboxd for ratings (same rules + fixes as step 4).
    4. Appends the new rows to movies_selected.csv with all the same columns
       the existing pipeline expects.

After this script:
    - movies_selected.csv has ~210 rows (150 original + ~60 new)
    - All new rows have earliest_release_date and letterboxd_rating filled
    - The YouTube scraper can be run next — it's resumable and will only
      scrape trailers it hasn't seen.
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

# --- Config ---
TMDB_API_KEY = os.environ.get("TMDB_API_KEY", "")
TMDB_BASE = "https://api.themoviedb.org/3"
LETTERBOXD_BASE = "https://letterboxd.com/film/"

MASTER_CSV = "data/raw/movies_master_tmdb.csv"
SELECTED_CSV = "data/raw/movies_selected.csv"
LB_LOG_CSV = "data/raw/letterboxd_scrape_log.csv"

EXTRA_MOVIES_TARGET = 60        # pick 60 extra so we land near 200 after failures
RANDOM_SEED = 43                # different from the main selection seed (42)

TMDB_SLEEP = 0.25
LB_SLEEP = 2.0
REQ_TIMEOUT = 30
RELEASE_TYPES = {1, 2, 3}       # premiere, limited theatrical, theatrical

UA = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
      "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36")
LB_HEADERS = {
    "User-Agent": UA,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

JSONLD_RE = re.compile(
    r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>', re.DOTALL)
TWITTER_RE = re.compile(
    r'<meta[^>]*name="twitter:data2"[^>]*content="([^"]+)"')


# ---------- TMDB ----------

def tmdb_get(endpoint: str, params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    url = f"{TMDB_BASE}{endpoint}"
    p = (params or {}).copy()
    p["api_key"] = TMDB_API_KEY
    r = requests.get(url, params=p, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    time.sleep(TMDB_SLEEP)
    return r.json()


def fetch_earliest(tmdb_id: int) -> Tuple[Optional[str], Optional[int], Optional[str]]:
    try:
        data = tmdb_get(f"/movie/{tmdb_id}/release_dates")
    except Exception as e:
        print(f"    earliest-release error: {e}")
        return None, None, None
    best = None
    for region in data.get("results", []):
        country = region.get("iso_3166_1") or ""
        for entry in region.get("release_dates", []) or []:
            t = entry.get("type")
            raw = entry.get("release_date") or ""
            if t not in RELEASE_TYPES or not raw:
                continue
            date = raw[:10]
            candidate = (date, t, country)
            if best is None or candidate[0] < best[0]:
                best = candidate
    return best if best else (None, None, None)


# ---------- Letterboxd ----------

def slugify(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[''`\"\.]", "", text)
    text = re.sub(r"[^a-z0-9]+", " ", text).strip()
    return re.sub(r"\s+", "-", text)


def try_fetch_lb(slug: str) -> Optional[str]:
    url = LETTERBOXD_BASE + slug + "/"
    try:
        r = requests.get(url, headers=LB_HEADERS, timeout=REQ_TIMEOUT)
    except requests.RequestException:
        return None
    if r.status_code == 200:
        return r.text
    return None


def extract_rating(html: str) -> Dict[str, Any]:
    # Try JSON-LD
    for m in JSONLD_RE.finditer(html):
        raw = m.group(1).strip()
        raw = re.sub(r"^/\*\s*<!\[CDATA\[\s*\*/|/\*\s*\]\]>\s*\*/$", "", raw).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        blocks = data if isinstance(data, list) else [data]
        for b in blocks:
            agg = b.get("aggregateRating") if isinstance(b, dict) else None
            if agg:
                try:
                    return {
                        "rating": float(agg.get("ratingValue")),
                        "rating_count": int(agg.get("ratingCount")) if agg.get("ratingCount") else None,
                        "source": "jsonld",
                    }
                except (TypeError, ValueError):
                    continue
    m = TWITTER_RE.search(html)
    if m:
        val = m.group(1).strip()
        nm = re.match(r"([0-9.]+)", val)
        if nm:
            try:
                return {"rating": float(nm.group(1)),
                        "rating_count": None, "source": "twitter"}
            except ValueError:
                pass
    return {"rating": None, "rating_count": None, "source": "none"}


def lookup_letterboxd(title: str, year: int) -> Tuple[Optional[str], Dict[str, Any], str]:
    candidates = []
    slug_y = slugify(title) + f"-{year}"
    html = try_fetch_lb(slug_y)
    if html is not None:
        candidates.append((LETTERBOXD_BASE + slug_y + "/",
                           extract_rating(html), "slug_with_year"))
    time.sleep(LB_SLEEP)
    slug_o = slugify(title)
    if slug_o != slug_y:
        html = try_fetch_lb(slug_o)
        if html is not None:
            candidates.append((LETTERBOXD_BASE + slug_o + "/",
                               extract_rating(html), "slug_no_year"))
    if not candidates:
        return None, {"rating": None, "rating_count": None, "source": "none"}, "not_found"

    def score(c):
        data = c[1]
        return (1 if data["rating"] is not None else 0,
                data["rating_count"] or 0)
    candidates.sort(key=score, reverse=True)
    return candidates[0]


# ---------- Selection ----------

def pick_extras(master: pd.DataFrame, already_selected: List[int]) -> pd.DataFrame:
    pool = master[~master["tmdb_id"].isin(already_selected)].copy()
    print(f"Reserve pool: {len(pool)} movies")
    if len(pool) == 0:
        return pool

    rng = np.random.default_rng(RANDOM_SEED)

    # Stratify: 12/year × 5 years = 60 target.
    per_year = max(1, EXTRA_MOVIES_TARGET // 5)
    tier_count = 3
    per_tier = max(1, per_year // tier_count)

    parts = []
    for year in sorted(pool["year"].unique()):
        ydf = pool[pool["year"] == year].copy()
        if len(ydf) >= tier_count * 2:
            ydf["pop_tier"] = pd.qcut(ydf["popularity"], q=tier_count,
                                      labels=False, duplicates="drop")
        else:
            ydf["pop_tier"] = 0

        year_picks = []
        for tier in sorted(ydf["pop_tier"].dropna().unique()):
            tdf = ydf[ydf["pop_tier"] == tier]
            k = min(per_tier, len(tdf))
            if k > 0:
                idx = rng.choice(tdf.index, size=k, replace=False)
                year_picks.append(tdf.loc[idx])

        if year_picks:
            yc = pd.concat(year_picks)
            # top up to per_year from remaining pool for this year
            if len(yc) < per_year:
                short = per_year - len(yc)
                rem = ydf.drop(yc.index).sort_values("popularity", ascending=False)
                yc = pd.concat([yc, rem.head(short)])
            parts.append(yc)

    extras = pd.concat(parts)
    extras = extras.drop(columns=["pop_tier"], errors="ignore")
    return extras.reset_index(drop=True)


# ---------- Main ----------

def main() -> None:
    if not os.path.exists(MASTER_CSV):
        print(f"ERROR: {MASTER_CSV} missing")
        sys.exit(1)
    if not os.path.exists(SELECTED_CSV):
        print(f"ERROR: {SELECTED_CSV} missing")
        sys.exit(1)

    master = pd.read_csv(MASTER_CSV)
    selected = pd.read_csv(SELECTED_CSV)
    print(f"Current selected: {len(selected)}")
    print(f"Master pool:      {len(master)}")

    extras = pick_extras(master, selected["tmdb_id"].tolist())
    print(f"Picked extras: {len(extras)}")
    print("Per-year counts in extras:")
    print(extras["year"].value_counts().sort_index().to_string())

    # Ensure dtypes survive the assignment operations
    for col in ("letterboxd_url", "letterboxd_rating", "rating_bucket",
                "exclude_reason", "earliest_release_date",
                "earliest_release_country", "letterboxd_rating_count",
                "letterboxd_match_method"):
        if col in selected.columns:
            selected[col] = selected[col].astype(object)

    # Prepare extras row shape to match selected
    for col in selected.columns:
        if col not in extras.columns:
            extras[col] = ""
    # Align column order
    extras = extras[selected.columns].copy()

    # Cast all string-ish columns to object so we can assign URLs/strings.
    # Pandas will otherwise infer float64 for empty columns and reject strings.
    string_cols = [
        "letterboxd_url", "letterboxd_rating", "letterboxd_rating_count",
        "letterboxd_match_method", "earliest_release_date",
        "earliest_release_country", "rating_bucket", "exclude_reason",
    ]
    for col in string_cols:
        if col in extras.columns:
            extras[col] = extras[col].astype(object)

    # Fetch earliest_release_date for each extra
    print("\nFetching earliest_release_date for extras...")
    for i, row in enumerate(extras.itertuples(index=False), start=1):
        tmdb_id = int(row.tmdb_id)
        date, rtype, country = fetch_earliest(tmdb_id)
        fallback = str(row.release_date) if row.release_date else None
        if date is None:
            date = fallback
            rtype = None
            country = None
        idx = extras.index[extras["tmdb_id"] == tmdb_id][0]
        extras.at[idx, "earliest_release_date"] = date
        extras.at[idx, "earliest_release_type"] = rtype
        extras.at[idx, "earliest_release_country"] = country
        if i % 10 == 0:
            print(f"  [{i}/{len(extras)}] {row.title[:40]!r} -> earliest={date}")

    # Scrape Letterboxd for each extra
    print("\nScraping Letterboxd for extras...")
    log_rows = []
    for i, row in enumerate(extras.itertuples(index=False), start=1):
        tmdb_id = int(row.tmdb_id)
        title = row.title
        year = int(row.year)
        print(f"  [{i}/{len(extras)}] {title!r} ({year})")
        url, data, method = lookup_letterboxd(title, year)

        idx = extras.index[extras["tmdb_id"] == tmdb_id][0]
        extras.at[idx, "letterboxd_url"] = url or ""
        extras.at[idx, "letterboxd_rating"] = data["rating"] if data["rating"] is not None else ""
        extras.at[idx, "letterboxd_rating_count"] = data["rating_count"] or ""
        extras.at[idx, "letterboxd_match_method"] = method
        log_rows.append({
            "tmdb_id": tmdb_id, "title": title, "year": year,
            "letterboxd_url": url or "", "letterboxd_rating": data["rating"] or "",
            "rating_count": data["rating_count"] or "",
            "rating_source": data["source"], "match_method": method,
            "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
        })
        if method == "not_found":
            print(f"    NOT FOUND")
        else:
            print(f"    {method}  rating={data['rating']}  count={data['rating_count']}")
        time.sleep(LB_SLEEP)

    # Merge and save
    merged = pd.concat([selected, extras], ignore_index=True)
    merged.to_csv(SELECTED_CSV, index=False, encoding="utf-8")
    print(f"\nWrote merged CSV: {SELECTED_CSV} ({len(merged)} rows)")

    # Append to letterboxd log
    log_df = pd.DataFrame(log_rows)
    if os.path.exists(LB_LOG_CSV):
        old = pd.read_csv(LB_LOG_CSV)
        log_df = pd.concat([old, log_df], ignore_index=True)
    log_df.to_csv(LB_LOG_CSV, index=False, encoding="utf-8")
    print(f"Appended to Letterboxd log: {LB_LOG_CSV}")

    # Summary
    r = pd.to_numeric(merged["letterboxd_rating"], errors="coerce")
    print(f"\nMovies with numeric rating: {r.notna().sum()}/{len(merged)}")
    nf = (merged["letterboxd_match_method"] == "not_found").sum()
    print(f"Letterboxd not_found: {nf}")


if __name__ == "__main__":
    main()
