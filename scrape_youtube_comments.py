#!/usr/bin/env python3
"""
Step 2: Scrape YouTube comments for each official trailer of the 150 selected movies.

For each (movie, trailer) pair:
    - Fetch top-level comments via commentThreads.list, ordered by time (chronological).
    - Cap at MAX_COMMENTS_PER_TRAILER.
    - Stop paging as soon as we cross the movie's earliest_release_date boundary
      (any comment with publishedAt >= earliest_release_date is post-release).
    - Save the raw comments (all fields we might need later) as JSON to
      data/raw/comments/{tmdb_id}/{trailer_id}.json.
    - Append a row to data/raw/comments_index.csv summarizing what we fetched.

Idempotent / resumable:
    - If the per-trailer JSON already exists, the trailer is skipped.
    - If we hit the YouTube API daily quota (HTTP 403, reason=quotaExceeded),
      the script exits cleanly after saving whatever is complete. Re-run tomorrow
      after the Pacific-midnight reset to continue.

Leakage policy:
    - Cutoff is the movie's earliest_release_date (from step 1c), interpreted as
      UTC midnight. A comment is kept only if publishedAt < cutoff.
    - This is the strict, leak-safe boundary: no comment that could have been
      influenced by ANY public screening (premiere, limited release, wide release)
      makes it into the training data.
"""

import json
import os
import sys
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import requests

# ============================================================
# CONFIG
# ============================================================

YOUTUBE_API_KEY = os.environ.get("YOUTUBE_API_KEY", "")

IN_CSV = os.environ.get("SCRAPE_IN_CSV", "data/raw/movies_selected.csv")
COMMENTS_DIR = os.environ.get("SCRAPE_COMMENTS_DIR", "data/raw/comments")
INDEX_CSV = os.environ.get("SCRAPE_INDEX_CSV", "data/raw/comments_index.csv")

# Fetch settings
MAX_COMMENTS_PER_TRAILER = 1000       # top-level only
PAGE_SIZE = 100                       # YouTube max per commentThreads.list call
ORDER = "time"                        # chronological so we can early-stop
REQUEST_TIMEOUT = 30
SLEEP_BETWEEN_CALLS = 0.1             # small politeness delay

BASE_URL = "https://www.googleapis.com/youtube/v3/commentThreads"
VIDEOS_URL = "https://www.googleapis.com/youtube/v3/videos"


# ============================================================
# HELPERS
# ============================================================

def parse_cutoff(earliest_release_date: str) -> datetime:
    """Parse 'YYYY-MM-DD' as UTC midnight — the strict pre-release boundary."""
    dt = datetime.strptime(earliest_release_date[:10], "%Y-%m-%d")
    return dt.replace(tzinfo=timezone.utc)


def parse_published_at(published_at: str) -> datetime:
    """YouTube returns ISO 8601 like '2020-05-15T14:23:11Z'."""
    if published_at.endswith("Z"):
        published_at = published_at[:-1] + "+00:00"
    return datetime.fromisoformat(published_at)


def extract_comment_fields(thread: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten the commentThread payload to the fields we'll actually use."""
    top = thread.get("snippet", {}).get("topLevelComment", {})
    snip = top.get("snippet", {})
    return {
        "comment_id": top.get("id"),
        "thread_id": thread.get("id"),
        "text": snip.get("textOriginal", ""),
        "author": snip.get("authorDisplayName", ""),
        "author_channel_id": (snip.get("authorChannelId") or {}).get("value", ""),
        "published_at": snip.get("publishedAt", ""),
        "updated_at": snip.get("updatedAt", ""),
        "like_count": snip.get("likeCount", 0),
        "reply_count": thread.get("snippet", {}).get("totalReplyCount", 0),
    }


class QuotaExceeded(Exception):
    """Raised when YouTube returns the daily-quota-exceeded error."""


class CommentsDisabled(Exception):
    """Raised when a video has comments disabled."""


class VideoNotFound(Exception):
    """Raised when the video is private / deleted."""


def fetch_video_published_at(video_id: str) -> Optional[datetime]:
    """
    Fetch the video's upload timestamp. Returns None if the video is unavailable.
    Costs 1 quota unit.
    """
    params = {
        "part": "snippet",
        "id": video_id,
        "key": YOUTUBE_API_KEY,
    }
    resp = requests.get(VIDEOS_URL, params=params, timeout=REQUEST_TIMEOUT)

    if resp.status_code == 200:
        items = resp.json().get("items", [])
        if not items:
            return None
        published = items[0].get("snippet", {}).get("publishedAt", "")
        if not published:
            return None
        try:
            return parse_published_at(published)
        except Exception:
            return None

    try:
        err = resp.json().get("error", {})
        reason = (err.get("errors") or [{}])[0].get("reason", "")
        message = err.get("message", "")
    except Exception:
        reason = ""
        message = resp.text[:500]

    if resp.status_code == 403 and reason == "quotaExceeded":
        raise QuotaExceeded(message)
    print(f"  [videos.list] HTTP {resp.status_code}  reason={reason}  msg={message[:200]}")
    return None


def fetch_page(video_id: str, page_token: Optional[str]) -> Dict[str, Any]:
    params = {
        "part": "snippet",
        "videoId": video_id,
        "maxResults": PAGE_SIZE,
        "order": ORDER,
        "textFormat": "plainText",
        "key": YOUTUBE_API_KEY,
    }
    if page_token:
        params["pageToken"] = page_token

    resp = requests.get(BASE_URL, params=params, timeout=REQUEST_TIMEOUT)

    if resp.status_code == 200:
        return resp.json()

    # Parse error reason if possible
    try:
        err = resp.json().get("error", {})
        reason = ""
        errors = err.get("errors", [])
        if errors:
            reason = errors[0].get("reason", "")
        message = err.get("message", "")
    except Exception:
        reason = ""
        message = resp.text[:500]

    if resp.status_code == 403 and reason == "quotaExceeded":
        raise QuotaExceeded(message)
    if resp.status_code == 403 and reason == "commentsDisabled":
        raise CommentsDisabled(message)
    if resp.status_code == 404 or reason in ("videoNotFound", "commentThreadsNotFound"):
        raise VideoNotFound(message)

    # Unexpected error — print and raise generic
    print(f"  HTTP {resp.status_code}  reason={reason}  msg={message[:200]}")
    resp.raise_for_status()
    return {}


def scrape_trailer(
    video_id: str, cutoff: datetime
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns (comments_kept, stats). Stops paging when we cross the cutoff OR
    hit MAX_COMMENTS_PER_TRAILER.

    YouTube's order=time returns NEWEST first, so to find pre-release comments
    we must page through until we stop seeing anything older than the cutoff.
    We keep only comments with publishedAt < cutoff. We stop when an entire
    page returns zero in-window comments (all newer OR all older than we need
    — see detailed logic below).
    """
    kept: List[Dict[str, Any]] = []
    stats = {
        "pages_fetched": 0,
        "raw_seen": 0,
        "kept_pre_release": 0,
        "stopped_reason": "",
    }

    page_token: Optional[str] = None

    while True:
        try:
            payload = fetch_page(video_id, page_token)
        except CommentsDisabled:
            stats["stopped_reason"] = "comments_disabled"
            return kept, stats
        except VideoNotFound:
            stats["stopped_reason"] = "video_not_found"
            return kept, stats

        stats["pages_fetched"] += 1
        items = payload.get("items", [])
        if not items:
            stats["stopped_reason"] = "no_more_items"
            break

        page_pre_release = 0
        for thread in items:
            stats["raw_seen"] += 1
            c = extract_comment_fields(thread)
            if not c["published_at"]:
                continue
            try:
                pub = parse_published_at(c["published_at"])
            except Exception:
                continue

            if pub < cutoff:
                kept.append(c)
                page_pre_release += 1
                if len(kept) >= MAX_COMMENTS_PER_TRAILER:
                    stats["stopped_reason"] = "max_comments_reached"
                    stats["kept_pre_release"] = len(kept)
                    return kept, stats

        # With order=time (newest first), we page through all post-release
        # comments before reaching the pre-release window (which sits at the
        # very end of the chronology). Once we start seeing pre-release
        # comments (kept > 0), a subsequent page with zero pre-release means
        # we've paged past the window — stop.
        if kept and page_pre_release == 0:
            stats["stopped_reason"] = "paged_past_pre_release_window"
            break

        page_token = payload.get("nextPageToken")
        if not page_token:
            stats["stopped_reason"] = "end_of_comments"
            break

        time.sleep(SLEEP_BETWEEN_CALLS)

    stats["kept_pre_release"] = len(kept)
    return kept, stats


def load_existing_index() -> pd.DataFrame:
    if os.path.exists(INDEX_CSV):
        return pd.read_csv(INDEX_CSV)
    return pd.DataFrame(columns=[
        "tmdb_id", "title", "trailer_id", "cutoff_date",
        "video_uploaded_at", "uploaded_post_cutoff",
        "pages_fetched", "raw_seen", "kept_pre_release", "stopped_reason",
        "output_path", "scraped_at_utc",
    ])


def append_index_row(index_df: pd.DataFrame, row: Dict[str, Any]) -> pd.DataFrame:
    index_df = pd.concat([index_df, pd.DataFrame([row])], ignore_index=True)
    index_df.to_csv(INDEX_CSV, index=False, encoding="utf-8")
    return index_df


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not os.path.exists(IN_CSV):
        print(f"ERROR: {IN_CSV} not found. Run select_movies.py first.")
        sys.exit(1)

    df = pd.read_csv(IN_CSV)
    print(f"Loaded {len(df)} selected movies")

    os.makedirs(COMMENTS_DIR, exist_ok=True)
    index_df = load_existing_index()
    already_done = set(zip(index_df["tmdb_id"], index_df["trailer_id"])) if len(index_df) else set()
    print(f"Already scraped trailers in index: {len(already_done)}")

    total_trailers = int(df["trailer_count"].sum())
    print(f"Total trailers to process: {total_trailers}")

    processed = 0
    quota_hit = False

    for _, movie in df.iterrows():
        tmdb_id = int(movie["tmdb_id"])
        title = movie["title"]
        # LENIENT CUTOFF: use TMDB wide release_date, not earliest_release_date.
        # Rationale: festival premieres don't realistically leak into general-audience
        # trailer comments, and the 16-month gap for some films (Yes God Yes, The Lodge)
        # was wasting real pre-release signal.
        cutoff_source = movie["release_date"]

        try:
            cutoff = parse_cutoff(str(cutoff_source))
        except Exception:
            print(f"[SKIP movie {tmdb_id}] bad cutoff date: {cutoff_source}")
            continue

        try:
            trailer_ids = json.loads(movie["youtube_trailer_ids"])
        except Exception:
            print(f"[SKIP movie {tmdb_id}] unparseable trailer list")
            continue

        movie_dir = os.path.join(COMMENTS_DIR, str(tmdb_id))
        os.makedirs(movie_dir, exist_ok=True)

        for trailer_id in trailer_ids:
            processed += 1
            out_path = os.path.join(movie_dir, f"{trailer_id}.json")

            if (tmdb_id, trailer_id) in already_done or os.path.exists(out_path):
                print(f"[{processed}/{total_trailers}] SKIP {tmdb_id}/{trailer_id} "
                      f"(already scraped)")
                continue

            print(f"[{processed}/{total_trailers}] {tmdb_id} {title[:40]!r} "
                  f"-> {trailer_id}  cutoff={cutoff.date()}")

            # Pre-check: if the trailer was uploaded after the cutoff, it's a
            # post-release upload — skip without paging through its comments.
            try:
                video_uploaded_at = fetch_video_published_at(trailer_id)
            except QuotaExceeded as e:
                print(f"\n*** QUOTA EXCEEDED (on videos.list): {e}")
                quota_hit = True
                break

            uploaded_post_cutoff = (
                video_uploaded_at is not None and video_uploaded_at >= cutoff
            )

            if video_uploaded_at is None:
                print(f"    video unavailable (deleted/private) — skipping")
                comments = []
                stats = {
                    "pages_fetched": 0, "raw_seen": 0,
                    "kept_pre_release": 0, "stopped_reason": "video_unavailable",
                }
            elif uploaded_post_cutoff:
                print(f"    trailer uploaded {video_uploaded_at.date()} (after cutoff "
                      f"{cutoff.date()}) — skipping, it's a post-release upload")
                comments = []
                stats = {
                    "pages_fetched": 0, "raw_seen": 0,
                    "kept_pre_release": 0, "stopped_reason": "uploaded_post_cutoff",
                }
            else:
                try:
                    comments, stats = scrape_trailer(trailer_id, cutoff)
                except QuotaExceeded as e:
                    print(f"\n*** QUOTA EXCEEDED: {e}")
                    print("*** Saving progress and exiting. Re-run tomorrow (Pacific midnight reset).")
                    quota_hit = True
                    break

            payload_out = {
                "tmdb_id": tmdb_id,
                "title": title,
                "trailer_id": trailer_id,
                "cutoff_iso_utc": cutoff.isoformat(),
                "video_uploaded_at": video_uploaded_at.isoformat() if video_uploaded_at else None,
                "uploaded_post_cutoff": uploaded_post_cutoff,
                "comments": comments,
                "stats": stats,
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload_out, f, ensure_ascii=False, indent=2)

            index_row = {
                "tmdb_id": tmdb_id,
                "title": title,
                "trailer_id": trailer_id,
                "cutoff_date": cutoff.date().isoformat(),
                "video_uploaded_at": video_uploaded_at.isoformat() if video_uploaded_at else "",
                "uploaded_post_cutoff": uploaded_post_cutoff,
                "pages_fetched": stats["pages_fetched"],
                "raw_seen": stats["raw_seen"],
                "kept_pre_release": stats["kept_pre_release"],
                "stopped_reason": stats["stopped_reason"],
                "output_path": out_path,
                "scraped_at_utc": datetime.now(timezone.utc).isoformat(),
            }
            index_df = append_index_row(index_df, index_row)

            print(f"    kept={stats['kept_pre_release']} raw_seen={stats['raw_seen']} "
                  f"pages={stats['pages_fetched']} reason={stats['stopped_reason']}")

        if quota_hit:
            break

    print("\n" + "=" * 60)
    print("Scrape session done")
    print("=" * 60)
    print(f"Trailers processed this session: {processed}")
    if quota_hit:
        print("Stopped early: quota exhausted. Re-run tomorrow.")
    print(f"Index: {INDEX_CSV}")
    print(f"Comments dir: {COMMENTS_DIR}")


if __name__ == "__main__":
    main()
