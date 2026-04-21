# Predicting Post-Release Letterboxd Ratings from Pre-Release YouTube Trailer Comments

**CS505 Final Project** — can you predict how a movie will be rated *after* it comes out, using only the YouTube trailer comments posted *before* it came out?

This repository contains the full data-collection and modeling pipeline used to answer that question on a dataset of 210 movies and ~236K pre-release comments.

---

## Table of contents

1. [Problem](#problem)
2. [Results summary](#results-summary)
3. [Repository layout](#repository-layout)
4. [Pipeline overview](#pipeline-overview)
5. [What each script does](#what-each-script-does)
6. [How to reproduce from scratch](#how-to-reproduce-from-scratch)
7. [How to rerun only the models](#how-to-rerun-only-the-models)
8. [Design decisions & rationale](#design-decisions--rationale)
9. [Known limitations](#known-limitations)

---

## Problem

- **Input**: for each movie, all YouTube comments posted on its official trailer(s) **before the movie was released**.
- **Label**: the movie's post-release aggregate Letterboxd rating, bucketed into 5 classes (bad / mediocre / average / good / great).
- **Task**: 5-class classification. We also report the coarser 3-class version (negative / neutral / positive) as a secondary analysis.
- **Critical constraint**: the train/val/test split is at the **movie level**, not the comment level, to prevent a movie's signal from leaking across splits.

---

## Results summary

Final dataset: **173 movies** (filtered from 210 after requiring ≥30 pre-release comments), **~236K comments**, **~18M characters of text**.

### 5-class (primary task)

| Model | Test Accuracy | Test Macro F1 |
|---|---|---|
| Majority-class baseline | 0.231 | 0.075 |
| **TF-IDF + Logistic Regression** 🏆 | **0.423** | **0.401** |
| Structured features + Logistic Regression | 0.385 | 0.364 |
| Structured features + Random Forest | 0.269 | 0.251 |
| DistilBERT embeddings + Logistic Regression | 0.192 | 0.188 |
| Ensemble (soft vote of top 3) | 0.269 | 0.274 |

### 3-class (secondary analysis, merges bad/mediocre and good/great)

| Model | Test Accuracy | Test Macro F1 |
|---|---|---|
| **TF-IDF + Logistic Regression** 🏆 | **0.538** | **0.522** |
| Structured features + Logistic Regression | 0.462 | 0.454 |
| DistilBERT embeddings + Logistic Regression | 0.308 | 0.291 |

**Headline**: TF-IDF + LogReg on raw concatenated comments is our strongest model — **5× the baseline macro F1** in 5-class, and **53.8% accuracy in 3-class**. Hand-engineered features (VADER sentiment + hype/spam/credibility/temporal heuristics) are a close second. Feature-extraction DistilBERT underperforms at this dataset size — likely needs end-to-end fine-tuning to be competitive.

All per-class metrics, confusion matrices, hyperparameter sweeps, and ensemble details are saved under `results/` (committed).

---

## Repository layout

```
nlp project/
├── README.md                          (this file)
├── .env.example                       ← template; copy to .env and fill in keys
├── .gitignore
├── requirements.txt
│
├── collect_tmdb_movies.py             Step 1  — pull candidate movies from TMDB
├── inspect_tmdb_movies.py             Step 1b — quick QC on the candidate list
├── fetch_earliest_release.py          Step 1c — earliest theatrical date per movie
├── select_movies.py                   Step 1d — stratified pick of 150 movies
├── expand_to_200.py                   Step 1e — add ~60 more to reach 200-movie target
├── scrape_youtube_comments.py         Step 2  — pull pre-release YouTube comments
├── scrape_letterboxd.py               Step 3  — scrape post-release Letterboxd ratings
├── build_final_dataset.py             Step 4  — filter, bucket labels, train/val/test split
│
├── models/
│   ├── baseline_majority.py           Model 0 — always-predict-majority floor
│   ├── tfidf_logreg.py                Model 1 — TF-IDF + LogReg (n-grams, balanced)
│   ├── structured_features.py         Model 2 — VADER + hype/spam/temporal feats + RF/LR
│   ├── distilbert_model.py            Model 3 — DistilBERT embeddings + LR
│   ├── tune_and_ensemble.py           Post-hoc: TF-IDF HP sweep, 3-class eval, soft-vote
│   └── compare_all.py                 Summary table across all model outputs
│
├── data/                              (not committed — regenerate with the scripts)
│   ├── raw/                           scraped inputs
│   │   ├── movies_master_tmdb.csv, movies_master_tmdb.json
│   │   ├── movies_selected.csv        the 210-movie working set
│   │   ├── letterboxd_scrape_log.csv
│   │   ├── comments_index.csv         per-trailer scrape stats
│   │   └── comments/{tmdb_id}/{trailer_id}.json   raw comment dumps
│   └── processed/
│       ├── movies_dataset.csv         the final modeling table (one row per movie)
│       └── movie_comments.jsonl       per-movie lists of raw comments
│
└── results/                           (not committed — regenerate by running the models)
    ├── baselines/majority_baseline.json
    ├── tfidf_logreg/results.json
    ├── structured_features/{results.json, features.csv}
    ├── distilbert/{results.json, movie_embeddings.npy, movie_embeddings_meta.csv}
    ├── improvements.json               tuned TF-IDF, 3-class metrics, ensemble
    └── comparison.json                 one-table summary
```

---

## Pipeline overview

```
  TMDB Discover API          YouTube Data API v3           letterboxd.com
       │                             │                            │
       ▼                             ▼                            ▼
collect_tmdb_movies.py       scrape_youtube_comments.py    scrape_letterboxd.py
fetch_earliest_release.py    (per-trailer JSONs)           (updates movies_selected.csv)
select_movies.py
expand_to_200.py
       │                             │                            │
       └────────────┬────────────────┴────────────┬───────────────┘
                    ▼                             ▼
              build_final_dataset.py (filter, bucket, stratified movie-level split)
                                      │
                                      ▼
                        data/processed/movies_dataset.csv
                                      │
                ┌─────────────────────┼─────────────────────┐
                ▼                     ▼                     ▼
       baseline_majority.py   tfidf_logreg.py     structured_features.py
                                                 distilbert_model.py
                                      │
                                      ▼
                        tune_and_ensemble.py + compare_all.py
```

---

## What each script does

### Data collection

**`collect_tmdb_movies.py`** — queries TMDB's `/discover/movie` for years 2020–2024 with filters (English-language, released, popularity ≥ 5, vote_count ≥ 200), then fetches each movie's details and videos. Keeps only movies with ≥ 2 official YouTube trailers. Caps at 4 trailers per movie. **Output**: `data/raw/movies_master_tmdb.csv` (388 candidate movies).

**`inspect_tmdb_movies.py`** — reads the master CSV and prints per-year counts, language sanity check, trailer-count histogram, popularity distribution, release-date range, and any suspicious rows. **No outputs written.**

**`fetch_earliest_release.py`** — for each movie, calls TMDB `/movie/{id}/release_dates` and computes the earliest global theatrical/premiere date across regions (type ∈ {premiere, limited theatrical, theatrical}). Adds `earliest_release_date`, `earliest_release_type`, `earliest_release_country` columns. **Overwrites** `movies_master_tmdb.csv` in place.

**`select_movies.py`** — stratified pick of 150 movies (30 per year × 5 years, 3 popularity tertiles within each year). Uses `RANDOM_SEED = 42` for reproducibility. **Output**: `data/raw/movies_selected.csv`.

**`expand_to_200.py`** — picks ~60 additional movies from the unused pool using stratified sampling (seed 43), fetches their earliest-release dates and Letterboxd ratings, and **appends** them to the existing `movies_selected.csv`. End state: ~210 rows.

**`scrape_youtube_comments.py`** — for each (movie, trailer), uses YouTube Data API v3 `commentThreads.list` with `order=time` (newest-first) to pull comments. Pre-checks the trailer's upload date; if uploaded after the movie's cutoff, skips that trailer. Filters comments to `published_at < release_date` (we use the wide-release cutoff, see [Design decisions](#design-decisions--rationale)). Caps at 1,000 kept pre-release comments per trailer. **Idempotent & resumable**: skips trailers whose JSON already exists; on quota exhaustion, saves progress and exits. **Outputs**: `data/raw/comments/{tmdb_id}/{trailer_id}.json`, `data/raw/comments_index.csv`.

**`scrape_letterboxd.py`** — for each movie in `movies_selected.csv`, guesses the Letterboxd URL from title+year (two slug variants), fetches the page, and extracts the aggregate rating from the JSON-LD block (primary) or twitter meta tag (fallback). Scores both slug variants and picks the one with more ratings to avoid matching obscure collisions. **Politeness**: 2s/request, browser User-Agent. **Updates** `movies_selected.csv` in place with `letterboxd_url`, `letterboxd_rating`, `letterboxd_rating_count`, `letterboxd_match_method`.

### Dataset assembly

**`build_final_dataset.py`** — loads `movies_selected.csv` and every `comments/*/*.json`. Drops movies with < 30 pre-release comments. Assigns each movie a 5-class `rating_bucket` (bad/mediocre/average/good/great — custom thresholds chosen to roughly balance classes). Stratified 70/15/15 movie-level train/val/test split, seed 42. Aggregates comments into a single `comments_text` blob per movie (for TF-IDF) and preserves the per-comment list (for structured-features and DistilBERT). **Outputs**: `data/processed/movies_dataset.csv` (one row per movie) and `data/processed/movie_comments.jsonl` (richer per-comment data).

### Modeling

**`models/baseline_majority.py`** — always predicts the train-set majority class (class 4, "great"). Establishes the floor every real model must beat.

**`models/tfidf_logreg.py`** — TF-IDF (word + bigrams, min_df=2, max_df=0.95, max_features=20k, English stopwords) on the concatenated comments_text. Multinomial LogisticRegression with `class_weight="balanced"` and L2 regularization (C=1.0). Also prints top-10 discriminative features per class. **Best-performing model.**

**`models/structured_features.py`** — extracts 25 per-movie interpretable features from the per-comment JSONL:
- **Sentiment** (VADER): mean, std, pos-frac, neg-frac
- **Hype / negative anticipation**: fractions of comments containing hand-curated hype or negative-hype phrases
- **Volume & engagement**: comment count, avg length, like counts (mean & median), reply rate
- **Spam-like**: uppercase fraction, emoji density, short-comment fraction
- **Temporal** (novel): sentiment & hype fraction in early/middle/late thirds of each movie's pre-release comment timeline
- **Credibility-weighted sentiment** (novel): VADER compound averaged with weights `log1p(like_count) + 1`
- **Hype-minus-neg** (novel): a signed expectation-reality proxy

Trains both a RandomForest (500 trees) and a scaled LogisticRegression. Prints top-10 feature importances from the RF.

**`models/distilbert_model.py`** — loads `distilbert-base-uncased`. For each movie, samples up to 200 comments (preferring longer ones), mean-pools token embeddings per comment, then mean-pools comment embeddings to a single 768-d movie vector. Trains a LogisticRegression on those vectors. This is **feature-extraction**, not fine-tuning — the BERT weights are frozen. On CPU this script takes ~20–30 min; with GPU/MPS much faster.

**`models/tune_and_ensemble.py`** — three post-hoc analyses in one script:
1. TF-IDF hyperparameter sweep (135 configs: C × ngram_range × min_df × max_features, selected on val macro F1).
2. 3-class re-evaluation: collapse the 5-class predictions to {neg, neu, pos} and recompute metrics.
3. Soft-vote ensemble: average the class-probability outputs of TF-IDF, structured-LR, and DistilBERT-LR, then argmax.

**`models/compare_all.py`** — walks each `results/*/results.json`, pulls train/val/test acc and macro F1, prints a single summary table, writes `results/comparison.json`.

---

## How to reproduce from scratch

The full end-to-end run takes ~4–6 hours of wall time (mostly YouTube comment scraping) and uses one day of YouTube Data API quota (~10k units).

### 1. Clone, create a venv, install deps

```bash
git clone https://github.com/jonahr4/CS505-Final-Project.git
cd CS505-Final-Project
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Get API keys

- **TMDB**: https://www.themoviedb.org/settings/api — free, instant.
- **YouTube Data API v3**: https://console.cloud.google.com/apis/credentials — requires a Google Cloud project with the API enabled. Free, 10k units/day.

### 3. Set environment variables

```bash
cp .env.example .env
# edit .env and paste your keys
export $(cat .env | xargs)
```

### 4. Run the pipeline

```bash
# Data collection (long pole: step 2, ~2–4 hours)
python collect_tmdb_movies.py                 # ~5 min
python inspect_tmdb_movies.py                 # <1 min (optional QC)
python fetch_earliest_release.py              # ~2 min
python select_movies.py                       # instant
python expand_to_200.py                       # ~10 min
python scrape_letterboxd.py                   # ~10 min (polite 2s/req)
python scrape_youtube_comments.py             # 2–4 hours, quota-bound, resumable

# Dataset build
python build_final_dataset.py                 # ~30 sec

# Models
python models/baseline_majority.py            # instant
python models/tfidf_logreg.py                 # ~10 sec
python models/structured_features.py          # ~30 sec
python models/distilbert_model.py             # ~20 min on CPU, much faster with GPU/MPS
python models/tune_and_ensemble.py            # ~20 sec
python models/compare_all.py                  # instant
```

If your YouTube quota runs out mid-scrape, the script exits cleanly and you can just rerun it after the daily reset (midnight Pacific). It skips trailers whose JSON already exists.

---

## How to rerun only the models

No data or results are committed — everything must be regenerated via the scripts. Once you have `data/processed/movies_dataset.csv` and `data/processed/movie_comments.jsonl` locally (by running the data-collection pipeline above), you can run the models in any order:

```bash
source venv/bin/activate
python models/baseline_majority.py        # instant
python models/tfidf_logreg.py             # ~10 sec
python models/structured_features.py      # ~30 sec
python models/distilbert_model.py         # ~20 min on CPU
python models/tune_and_ensemble.py        # ~20 sec
python models/compare_all.py              # instant
```

All models use seed=42 so numbers are reproducible.

---

## Design decisions & rationale

### Leakage boundary: wide release date, not festival premiere

A "pre-release" comment is one posted before the movie was publicly available. The naive choice is `release_date` (TMDB's wide-release date). A stricter alternative is `earliest_release_date` (includes film-festival premieres). We tried both:

- **Strict** (earliest premiere): lost ~20% of our movies to big festival-to-wide gaps (e.g. *Yes, God, Yes* premiered at Sundance 16 months before wide release; under strict, we'd throw away all trailer comments in that 16-month window).
- **Lenient** (wide release): recovered those movies. The leakage risk is minimal — general YouTube commenters aren't festival attendees and almost certainly haven't seen the film before its wide release.

We ship with **lenient**. `earliest_release_date` is still stored for optional sensitivity analysis.

### Rating bucket thresholds

Letterboxd ratings in our dataset cluster between 2.5 and 3.5 (mean 3.0, std 0.63). Equal-width 5-class bins would leave the extreme classes nearly empty. We use class-balancing thresholds that also have interpretable star-rating meanings:

| Bucket | Range | Label |
|---|---|---|
| 0 | r < 2.3 | bad |
| 1 | 2.3 ≤ r < 2.75 | mediocre |
| 2 | 2.75 ≤ r < 3.15 | average |
| 3 | 3.15 ≤ r < 3.5 | good |
| 4 | r ≥ 3.5 | great |

This gives class sizes of 22 / 21 / 30 / 25 / 29 after filtering — well-balanced.

### Movie-level split, not comment-level

Splitting comments would leak a movie's signal across train and test (same movie shows up in both). We split *movies* 70/15/15, stratified by rating bucket, seed=42. All comments for a movie live in the same split.

### Min-comments threshold of 30

Movies with <30 pre-release comments don't give us enough text for meaningful features. This cutoff drops ~20% of movies (mostly ones whose "official" TMDB trailers were actually post-release marketing uploads).

### Why TF-IDF beats DistilBERT here

Feature-extraction DistilBERT (frozen weights, mean-pooled embeddings averaged over ~200 comments/movie) washes out the task-specific signal. TF-IDF directly captures which **words and phrases** correlate with rating buckets (e.g. "kraven"/"bella thorne" → bad; "a24"/"pixar" → great). End-to-end fine-tuning of DistilBERT is the natural follow-up but needs larger n and careful regularization to avoid overfitting.

---

## Known limitations

1. **Small test set (26 movies)** — a single misclassification moves test accuracy by ~3.8 points. Reported numbers have real variance; we note them with whole-percentage-point skepticism.
2. **Hyperparameter tuning via val-set selection didn't help** — with n=26 val movies, the tuned TF-IDF configuration that won on val underperformed on test. We report both tuned and untuned numbers honestly.
3. **No DistilBERT fine-tuning** — a legitimate next step, but beyond the scope of this project.
4. **Popularity-biased sample** — our movies all satisfy TMDB's `vote_count ≥ 200` filter, so they're audience-aware films. True indie films with ~50 Letterboxd ratings are underrepresented.
5. **Single snapshot of Letterboxd ratings** — we scraped ratings at one point in time. Letterboxd averages drift slowly but this is worth noting.
