# Predicting Post-Release Letterboxd Ratings from Pre-Release YouTube Trailer Comments

> **CS505 Final Project** — Boston University
> Jigar Kanakhara · Bhavya Bavissi · Jonah Rothman

Can pre-release YouTube trailer comments predict how a movie is eventually rated on Letterboxd? We built an end-to-end pipeline (TMDB → YouTube → Letterboxd) that produced **173 movies** and **~236,000 pre-release comments**, then trained **six** different models to find out.

**TL;DR — TF-IDF + Logistic Regression wins.** A 1972-style bag-of-words classifier beats two end-to-end fine-tuned DistilBERT variants at this dataset scale. Honest finding, real story for the report.

---

## Table of contents

1. [Highlights](#highlights)
2. [Final results](#final-results)
3. [Pipeline](#pipeline)
4. [What every script does](#what-every-script-does)
5. [How to reproduce](#how-to-reproduce)
6. [Design decisions](#design-decisions)
7. [Limitations](#limitations)

---

## Highlights

- **6 model variants** spanning lexical, hand-engineered, frozen-transformer, and two end-to-end fine-tuning regimes.
- **Three evaluation metrics**: accuracy, macro F1, and *mean absolute label distance* (MAE — added per midterm reviewer feedback because the 5-class task is ordinal, not nominal).
- **All three "novel components"** from the proposal implemented and quantified:
  - **Temporal modeling** → per-window ablation shows the *late* window dominates.
  - **Expectation–reality gap** → a hand-crafted hype/negativity signal correlates positively with actual ratings (Pearson r = 0.28).
  - **Credibility-weighted aggregation** → like-weighted VADER sentiment.
- **Reproducible**: every stochastic step uses seed=42; the YouTube/Letterboxd scrapers are checkpointed and resume from quota exhaustion.
- **Honest negative result**: end-to-end DistilBERT fine-tuning (per-comment AND movie-level) does **not** beat TF-IDF at n=121 train movies — but movie-level fine-tuning achieves the best **MAE** of any model (predictions are closer to truth even when wrong).

---

## Final results

### 5-class test set (n=26 movies)

| Model | Acc ↑ | Macro F1 ↑ | MAE ↓ |
|---|---:|---:|---:|
| Majority baseline | 0.231 | 0.075 | 1.77 |
| **🏆 TF-IDF + LR** | **0.423** | **0.401** | 1.19 |
| Structured + LR | 0.385 | 0.364 | 1.31 |
| Structured + RF | 0.269 | 0.251 | 1.58 |
| DistilBERT (frozen) + LR | 0.192 | 0.188 | 1.77 |
| DistilBERT FT (per-comment, [CLS]) | 0.231 | 0.174 | 1.65 |
| DistilBERT FT (movie-level, [CLS]) | 0.269 | 0.161 | **1.27** |
| Ensemble (soft vote) | 0.269 | 0.274 | 1.46 |

### 3-class collapsed task ({bad,mediocre} / {average} / {good,great})

| Model | Acc ↑ | Macro F1 ↑ |
|---|---:|---:|
| **🏆 TF-IDF + LR** | **0.538** | **0.522** |
| Structured + LR | 0.462 | 0.454 |
| DistilBERT (frozen) + LR | 0.308 | 0.291 |

### Per-window temporal ablation (Structured + LR)

| Variant | Acc | Macro F1 | MAE |
|---|---:|---:|---:|
| Full (all windows)  | 0.385 | 0.364 | 1.31 |
| No temporal         | 0.346 | 0.318 | 1.35 |
| Early only          | 0.346 | 0.329 | 1.35 |
| Middle only         | 0.308 | 0.287 | 1.50 |
| **Late only**       | **0.385** | **0.368** | **1.23** |

The **late** pre-release window matches or exceeds the full feature set on every metric. Comments closer to release carry the most signal — consistent with the hypothesis that as release approaches, the audience pool shifts from core fans to the broader Letterboxd-representative crowd.

### Expectation–reality gap

`hype_minus_neg` (fraction of hype phrases minus fraction of negative-anticipation phrases) vs. actual rating bucket:

- **Pearson correlation: r = 0.28, Spearman ρ = 0.32** — positive but noisy.
- **Over-hyped flops** (high hype, low rating): *Disenchanted, The Prom, After Ever Happy, Halloween Ends*
- **Sleeper hits** (low hype, high rating): *Wicked Little Letters, Dungeons & Dragons: Honor Among Thieves, Challengers, The Fall Guy*

### Error analysis

8 of 26 test movies were misclassified by **all 4** primary models:
*The Invisible Man, Twisters, The Matrix Resurrections, The New Mutants, The Gray Man, A Good Person, After Ever Happy, What's Love Got to Do with It?*

Aggregate mean signed error across all `(model, movie)` pairs: **+0.39 buckets** — models systematically **over-predict**. Plausible: viral trailers generate enthusiasm regardless of eventual quality. Franchise reboots (*Matrix Resurrections*, *New Mutants*) likely fool models that learned a positive *matrix*/*x-men* prior.

---

## Pipeline

```
  TMDB Discover API          YouTube Data API v3           letterboxd.com
       │                             │                            │
       ▼                             ▼                            ▼
collect_tmdb_movies.py        scrape_youtube_comments.py   scrape_letterboxd.py
fetch_earliest_release.py     (per-trailer JSONs)
select_movies.py
expand_to_200.py
       └──────────────────┬───────────────────────────┬───────────┘
                          ▼                           ▼
                 build_final_dataset.py  (filter, bucket, movie-level split)
                                          │
                                          ▼
                          data/processed/movies_dataset.csv
                                          │
              ┌───────────┬───────────────┼───────────────────┬─────────────┐
              ▼           ▼               ▼                   ▼             ▼
        baseline   tfidf_logreg   structured_features   distilbert_*   compare_all
         (floor)     (lexical)     (VADER + heuristics) (5 transformer (final
                                                          variants)     summary)
                                          │
                                          ▼
                       analysis/{temporal_ablation, expectation_reality,
                                 error_analysis, add_mae_metric}.py
```

---

## What every script does

### Data collection (root)

| Script | Role |
|---|---|
| `collect_tmdb_movies.py` | TMDB Discover query → 388 candidate movies (English, 2020–2024, ≥200 votes, ≥2 official trailers) |
| `inspect_tmdb_movies.py` | Quality check: per-year counts, language sanity, trailer histogram |
| `fetch_earliest_release.py` | Calls `/movie/{id}/release_dates`, computes earliest global theatrical/premiere date |
| `select_movies.py` | Stratified pick of 150 movies (30/year × 5 years × 3 popularity tertiles, seed=42) |
| `expand_to_200.py` | Adds 60 more movies from the unused pool to land at ~210 (seed=43) |
| `scrape_youtube_comments.py` | YouTube Data API v3 `commentThreads.list` with `order=time`. Pre-checks each trailer's upload date and skips post-release uploads. Caps at 1,000 pre-release comments per trailer. **Resumable** on quota exhaustion. |
| `scrape_letterboxd.py` | Constructs candidate URLs from title + year, parses the JSON-LD `aggregateRating` block. Picks the slug with more logged ratings to avoid obscure short-film collisions. 2-second-per-request politeness. |

### Dataset assembly

| Script | Role |
|---|---|
| `build_final_dataset.py` | Filters to ≥30 pre-release comments per movie, assigns 5-class buckets, makes the 70/15/15 movie-level stratified split. Outputs `movies_dataset.csv` (one row per movie) and `movie_comments.jsonl` (per-comment data). |

### Models (under `models/`)

| Script | Role |
|---|---|
| `baseline_majority.py` | Predicts train majority class — establishes the floor every real model must beat |
| `tfidf_logreg.py` | TF-IDF (uni+bigrams, 20K features) → balanced multinomial L2 logistic regression. **Best model.** |
| `structured_features.py` | 25 hand-crafted per-movie features (VADER sentiment, hype/negativity phrases, engagement, spam-likeness, **temporal windows**, **credibility-weighted sentiment**). Trains both Logistic Regression and Random Forest. |
| `distilbert_model.py` | Frozen `distilbert-base-uncased`, mean-pool comment embeddings to one 768-d movie vector, LR on top. (Midterm-era setup.) |
| `distilbert_finetune.py` | **End-to-end fine-tune, per-comment.** Each comment inherits its movie's bucket (pseudo-labels). Train [CLS] head on 80K examples. Movie-level inference via mean-pooled softmax probabilities. |
| `distilbert_finetune_movie.py` | **End-to-end fine-tune, movie-level.** One input per movie (top-60 most-liked comments concatenated, 512 tokens). Heavier regularization for n=121. |
| `longformer_finetune.py` | Longformer (4096-token context) variant — written and tested. **Aborted in production** due to swap-thrashing on 16GB unified-memory M4 Air. Included for reference / GPU-machine reruns. |
| `tune_and_ensemble.py` | TF-IDF hyperparameter sweep (135 configs), 3-class re-evaluation of all 5-class predictions, soft-vote ensemble of TF-IDF + structured-LR + DistilBERT-LR. |
| `compare_all.py` | Walks every `results/*/results.json`, prints a unified summary table. |
| `add_mae_metric.py` | Reads each model's saved confusion matrix and computes mean absolute label distance — the ordinal-aware metric the grader recommended. |

### Analysis (under `analysis/`)

| Script | Role |
|---|---|
| `expectation_reality.py` | Computes Pearson/Spearman correlation between `hype_minus_neg` and rating bucket; identifies over-hyped flops and sleeper hits. Saves figure. |
| `error_analysis.py` | Cross-references test predictions across all primary models. Identifies movies misclassified by all 4 (the "universal misses"). Reports off-by-X distribution. |
| `temporal_ablation.py` | Trains Structured + LR with each temporal window (early / middle / late) in isolation. Quantifies which window carries most signal. |

### Report

| File | Role |
|---|---|
| `final_report.tex` | LaTeX source for the final 4–6 page paper. Drop into Overleaf with the figures. |

---

## How to reproduce

> **Note on the dataset.** The `data/` directory (raw scraped YouTube comments and Letterboxd ratings, ~236K comments across 173 movies) is **not committed to the repo**. This is intentional, for two reasons: (1) size and (2) YouTube's terms of service discourage redistributing scraped comment text. The full dataset can be regenerated end-to-end from the scripts below using a free TMDB key and a free YouTube Data API v3 key. All `results/*/results.json` confusion matrices that back the numbers in the report are similarly regeneratable. If a grader needs a frozen snapshot of the dataset for spot-checking, contact the authors and we'll share it directly.

### 1. Setup

```bash
git clone https://github.com/jonahr4/CS505-Final-Project.git
cd CS505-Final-Project
git checkout final-branch
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. API keys

Copy `.env.example` → `.env` and fill in:
- **TMDB key**: free at https://www.themoviedb.org/settings/api
- **YouTube Data API v3 key**: free at https://console.cloud.google.com/apis/credentials (10K quota units/day)

```bash
cp .env.example .env
# edit .env
export $(cat .env | xargs)
```

### 3. Run pipeline (once, ~4–6 hours wall time)

```bash
# Data collection
python collect_tmdb_movies.py            # ~5 min
python inspect_tmdb_movies.py            # optional QC
python fetch_earliest_release.py         # ~2 min
python select_movies.py                  # instant
python expand_to_200.py                  # ~10 min
python scrape_letterboxd.py              # ~10 min (polite 2s/req)
python scrape_youtube_comments.py        # 2–4 hours, quota-bound, resumable

# Dataset
python build_final_dataset.py            # ~30 sec

# Baseline + classical models
python models/baseline_majority.py       # instant
python models/tfidf_logreg.py            # ~10 sec
python models/structured_features.py     # ~30 sec

# Frozen DistilBERT (feature extraction)
python models/distilbert_model.py        # ~20 min on CPU, faster on MPS/GPU

# End-to-end fine-tuning (the grader's primary ask)
python models/distilbert_finetune.py        # ~3 hours on M4 Air MPS
python models/distilbert_finetune_movie.py  # ~5 min on M4 Air MPS

# Post-hoc analyses
python models/tune_and_ensemble.py       # ~20 sec
python models/compare_all.py             # instant
python models/add_mae_metric.py          # instant
python analysis/temporal_ablation.py     # ~5 sec
python analysis/expectation_reality.py   # ~5 sec
python analysis/error_analysis.py        # ~10 sec
```

If your YouTube quota runs out mid-scrape, the script exits cleanly. Wait until midnight Pacific (the daily reset) and rerun — it skips trailers whose JSON already exists.

### 4. Build the report PDF

```bash
# Generate figures
python make_final_figures.py             # creates report_figures/

# Then upload final_report.tex + report_figures/ to Overleaf and compile,
# or compile locally if you have MacTeX:
# pdflatex final_report.tex
```

---

## Design decisions

### Leakage boundary: wide release date, not festival premiere

Naive choice: TMDB's `release_date` (wide release). Stricter alternative: `earliest_release_date` (includes festival premieres). We initially used strict and lost ~15 movies whose festival-to-wide gap was huge (e.g. *Promising Young Woman*: Sundance January → wide December = 11 months of throw-away comments).

**We ship with lenient (wide release).** General YouTube commenters in the gap window haven't seen the film. The strict cutoff is still stored in the dataset for any future sensitivity analysis.

### Rating bucket thresholds

Letterboxd ratings cluster narrowly (mean 3.0, std 0.63). Equal-width 5 bins → empty extreme classes. We use class-balancing thresholds:

| Bucket | Range | Label | n |
|---|---|---|---:|
| 0 | r < 2.3 | bad | 26 |
| 1 | 2.3 ≤ r < 2.75 | mediocre | 28 |
| 2 | 2.75 ≤ r < 3.15 | average | 39 |
| 3 | 3.15 ≤ r < 3.5 | good | 36 |
| 4 | r ≥ 3.5 | great | 44 |

### Movie-level split, not comment-level

Splitting comments would leak a movie's signal across train and test. We split *movies* 70/15/15 (121/26/26), stratified by rating bucket, seed=42. All comments for a movie live in the same split.

### Per-comment vs movie-level fine-tuning

Two complementary failure modes for transformers at n=121:

- **Per-comment** turns 121 movies into 80K training examples by giving every comment its movie's label. Fixes the "tiny n" problem but introduces severe label noise (sarcastic comments under "great" movies still labeled "great"). **Result**: severe overfit, val F1 plateaued at 0.10.
- **Movie-level** keeps clean labels but only has 121 training examples for a 66M-parameter model. **Result**: best val F1 at epoch 1, never recovered. Achieves the best MAE of any model (1.27) — even when wrong, it's close.

Reporting both is the academically honest thing to do. Neither beats TF-IDF, which is itself a notable finding.

### Why TF-IDF wins

TF-IDF processes the **entire** ~100K characters of comment text per movie. DistilBERT sees ~2K. At n=121 with very long inputs and noisy text, lexical features beat compressed semantic representations — a pattern documented since Pang et al. (2002).

---

## Limitations

1. **Small test set (26 movies)** — single misclassification moves accuracy ~3.8 points. Reported numbers carry real variance.
2. **Popularity bias**: TMDB `vote_count ≥ 200` filter excludes truly obscure indie films.
3. **Single-snapshot Letterboxd scrape**: ratings drift; recently released films have not converged.
4. **Longformer aborted**: written and tested but not run to completion due to memory pressure on 16GB MacBook Air. Included as `models/longformer_finetune.py` for GPU/Linux reruns.
5. **No genre-aware modeling**: error analysis hints at franchise-reboot bias; future work could ablate by genre.

---

## Repo structure

```
.
├── README.md                          ← this file
├── final_report.tex                   ← LaTeX source for the final paper
├── requirements.txt
├── .env.example                       ← API key template
├── .gitignore
│
├── collect_tmdb_movies.py             Step 1
├── inspect_tmdb_movies.py             Step 1b
├── fetch_earliest_release.py          Step 1c
├── select_movies.py                   Step 1d
├── expand_to_200.py                   Step 1e
├── scrape_youtube_comments.py         Step 2
├── scrape_letterboxd.py               Step 3
├── build_final_dataset.py             Step 4
├── make_figures.py                    midterm figures
├── make_final_figures.py              final-report figures
│
├── models/
│   ├── baseline_majority.py
│   ├── tfidf_logreg.py
│   ├── structured_features.py
│   ├── distilbert_model.py
│   ├── distilbert_finetune.py            ← per-comment, [CLS] head, no pooling
│   ├── distilbert_finetune_movie.py      ← movie-level, [CLS] head
│   ├── longformer_finetune.py            ← long-context variant (untested at scale)
│   ├── tune_and_ensemble.py
│   ├── compare_all.py
│   └── add_mae_metric.py                 ← ordinal MAE metric
│
└── analysis/
    ├── temporal_ablation.py
    ├── expectation_reality.py
    └── error_analysis.py
```

`data/`, `results/`, and `report_figures/` are **not committed** — regenerate them via the scripts. This keeps the repo lean and forces reproducibility.

---

## Acknowledgments

This is the final-branch deliverable for **CS505 (Boston University)**. All transformer experiments use the [Hugging Face](https://huggingface.co/) `transformers` library; classical models use [scikit-learn](https://scikit-learn.org/); sentiment scoring uses [VADER](https://github.com/cjhutto/vaderSentiment).

API keys are read from environment variables — never commit your `.env` file. If you push to a public repo with keys hard-coded, rotate them immediately.
