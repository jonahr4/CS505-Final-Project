#!/usr/bin/env python3
"""
Step 4: For each movie in movies_selected.csv, find its Letterboxd page and
scrape the average rating.

Strategy:
    1. Build a naive slug: slugify(title) + "-" + year  (e.g. "soul-2020")
    2. Try `https://letterboxd.com/film/{slug}/`. If HTTP 200, use it.
    3. Fallback: try slug without the year (e.g. "soul" — works for some
       movies that don't collide with older entries).
    4. If neither works, log it as 'not_found' and leave for manual review.

The Letterboxd page contains the rating in TWO places:
    - <script type="application/ld+json">: aggregateRating.ratingValue
    - <meta name="twitter:data2" content="3.94 out of 5">
We prefer the JSON-LD source (more precise, includes ratingCount/reviewCount).

Outputs:
    - Updates movies_selected.csv in place: fills letterboxd_url,
      letterboxd_rating, and adds letterboxd_rating_count + letterboxd_match_method
    - Writes data/raw/letterboxd_scrape_log.csv with per-movie status

Politeness:
    - ~2s sleep between requests (Letterboxd is a small operation)
    - Sets a real browser User-Agent
    - Resumable: if a movie already has letterboxd_rating filled, skip it
"""

import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import requests

IN_CSV = os.environ.get("LB_IN_CSV", "data/raw/movies_selected.csv")
OUT_CSV = os.environ.get("LB_OUT_CSV", IN_CSV)  # defaults to in-place
LOG_CSV = os.environ.get("LB_LOG_CSV", "data/raw/letterboxd_scrape_log.csv")

SLEEP_BETWEEN_REQUESTS = 2.0
REQUEST_TIMEOUT = 30

USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0 Safari/537.36"
)
HEADERS = {
    "User-Agent": USER_AGENT,
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

BASE = "https://letterboxd.com/film/"

# Regex-based extractors (fast, no HTML parser dependency)
JSONLD_BLOCK_RE = re.compile(
    r'<script[^>]*type="application/ld\+json"[^>]*>(.*?)</script>',
    re.DOTALL,
)
TWITTER_DATA2_RE = re.compile(
    r'<meta[^>]*name="twitter:data2"[^>]*content="([^"]+)"'
)


def slugify(text: str) -> str:
    """Letterboxd-style slug: lowercase, strip punctuation, collapse spaces to '-'."""
    text = text.lower()
    # Remove apostrophes, quotes, and periods (don't leave gaps — "P.S." -> "ps")
    text = re.sub(r"[''`\"\.]", "", text)
    # Replace anything not alphanumeric with a space
    text = re.sub(r"[^a-z0-9]+", " ", text)
    text = text.strip()
    return re.sub(r"\s+", "-", text)


def try_fetch(slug: str) -> Optional[str]:
    """GET letterboxd.com/film/{slug}/ — return HTML text if 200, else None."""
    url = BASE + slug + "/"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    except requests.RequestException as e:
        print(f"    network error on {url}: {e}")
        return None
    if resp.status_code == 200:
        return resp.text
    if resp.status_code != 404:
        print(f"    unexpected {resp.status_code} on {url}")
    return None


def extract_rating_from_html(html: str) -> Dict[str, Any]:
    """
    Extract aggregateRating from the JSON-LD block. Returns a dict with:
        rating: float or None
        rating_count: int or None
        review_count: int or None
        source: 'jsonld', 'twitter', or 'none'
    """
    # Try JSON-LD first (richer data)
    for match in JSONLD_BLOCK_RE.finditer(html):
        raw = match.group(1).strip()
        # Strip CDATA wrappers if present
        raw = re.sub(r"^/\*\s*<!\[CDATA\[\s*\*/|/\*\s*\]\]>\s*\*/$", "", raw).strip()
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        blocks = data if isinstance(data, list) else [data]
        for block in blocks:
            agg = block.get("aggregateRating") if isinstance(block, dict) else None
            if agg:
                try:
                    return {
                        "rating": float(agg.get("ratingValue")),
                        "rating_count": int(agg.get("ratingCount")) if agg.get("ratingCount") else None,
                        "review_count": int(agg.get("reviewCount")) if agg.get("reviewCount") else None,
                        "source": "jsonld",
                    }
                except (TypeError, ValueError):
                    continue

    # Fallback: twitter:data2
    m = TWITTER_DATA2_RE.search(html)
    if m:
        # content looks like "3.94 out of 5"
        val = m.group(1).strip()
        num_match = re.match(r"([0-9.]+)", val)
        if num_match:
            try:
                return {
                    "rating": float(num_match.group(1)),
                    "rating_count": None,
                    "review_count": None,
                    "source": "twitter",
                }
            except ValueError:
                pass

    return {"rating": None, "rating_count": None, "review_count": None, "source": "none"}


def lookup_movie(title: str, year: int) -> Tuple[Optional[str], Dict[str, Any], str]:
    """
    Try to find a Letterboxd page for (title, year). Fetches both slug variants
    (with year and without) and picks the one with the higher ratingCount —
    this avoids matching obscure collision entries (e.g. a short film that
    happens to share the title).

    Returns (letterboxd_url, extracted_data, match_method).
    match_method is one of: 'slug_with_year', 'slug_no_year', 'not_found'.
    """
    candidates = []  # list of (url, data, method)

    slug_year = slugify(title) + f"-{year}"
    html = try_fetch(slug_year)
    if html is not None:
        candidates.append((BASE + slug_year + "/",
                           extract_rating_from_html(html), "slug_with_year"))

    time.sleep(SLEEP_BETWEEN_REQUESTS)

    slug_only = slugify(title)
    if slug_only != slug_year:  # avoid redundant fetch if slugify didn't change
        html = try_fetch(slug_only)
        if html is not None:
            candidates.append((BASE + slug_only + "/",
                               extract_rating_from_html(html), "slug_no_year"))

    if not candidates:
        return None, {"rating": None, "rating_count": None, "review_count": None,
                      "source": "none"}, "not_found"

    # Prefer candidates that actually have a rating, then higher rating_count.
    def score(c):
        data = c[1]
        has_rating = 1 if data["rating"] is not None else 0
        rc = data["rating_count"] or 0
        return (has_rating, rc)

    candidates.sort(key=score, reverse=True)
    return candidates[0]


def main() -> None:
    if not os.path.exists(IN_CSV):
        print(f"ERROR: {IN_CSV} not found.")
        sys.exit(1)

    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} selected movies")

    # These columns are empty in the input CSV, so pandas infers float64 (NaN).
    # Force them to object so we can assign strings/numbers without dtype errors.
    for col in ("letterboxd_url", "letterboxd_rating", "rating_bucket", "exclude_reason"):
        if col in df.columns:
            df[col] = df[col].astype(object)

    # Add new columns if missing
    if "letterboxd_rating_count" not in df.columns:
        df["letterboxd_rating_count"] = ""
    if "letterboxd_match_method" not in df.columns:
        df["letterboxd_match_method"] = ""
    df["letterboxd_rating_count"] = df["letterboxd_rating_count"].astype(object)
    df["letterboxd_match_method"] = df["letterboxd_match_method"].astype(object)

    log_rows = []
    n_ok = 0
    n_notfound = 0

    for i, row in enumerate(df.itertuples(index=False), start=1):
        # Skip if we already have a rating for this movie (resumable)
        existing_rating = row.letterboxd_rating
        if pd.notna(existing_rating) and str(existing_rating).strip() not in ("", "nan"):
            print(f"[{i}/{len(df)}] SKIP {row.title!r} — already has rating "
                  f"{existing_rating}")
            continue

        title = row.title
        year = int(row.year)
        print(f"[{i}/{len(df)}] {title!r} ({year})")

        url, data, method = lookup_movie(title, year)

        idx = df.index[df["tmdb_id"] == row.tmdb_id][0]
        df.at[idx, "letterboxd_url"] = url or ""
        df.at[idx, "letterboxd_rating"] = data["rating"] if data["rating"] is not None else ""
        df.at[idx, "letterboxd_rating_count"] = data["rating_count"] or ""
        df.at[idx, "letterboxd_match_method"] = method

        log_rows.append({
            "tmdb_id": row.tmdb_id,
            "title": title,
            "year": year,
            "letterboxd_url": url or "",
            "letterboxd_rating": data["rating"] or "",
            "rating_count": data["rating_count"] or "",
            "review_count": data["review_count"] or "",
            "rating_source": data["source"],
            "match_method": method,
            "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
        })

        if method == "not_found":
            n_notfound += 1
            print(f"    NOT FOUND")
        else:
            n_ok += 1
            print(f"    {method}  rating={data['rating']}  "
                  f"count={data['rating_count']}  source={data['source']}")

        # Save incrementally every 10 movies so a crash doesn't lose work
        if i % 10 == 0:
            df.to_csv(OUT_CSV, index=False, encoding="utf-8")
            pd.DataFrame(log_rows).to_csv(LOG_CSV, index=False, encoding="utf-8")
            print(f"    [checkpoint saved at {i}]")

        time.sleep(SLEEP_BETWEEN_REQUESTS)

    df.to_csv(OUT_CSV, index=False, encoding="utf-8")
    pd.DataFrame(log_rows).to_csv(LOG_CSV, index=False, encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"Done. OK={n_ok}  NotFound={n_notfound}")
    print(f"Updated CSV: {OUT_CSV}")
    print(f"Log: {LOG_CSV}")
    print("=" * 60)


if __name__ == "__main__":
    main()
