#!/usr/bin/env python3
"""
Item 3 from midterm plan: expectation-reality writeup.

For each movie compute hype_minus_neg (proposal's expectation-reality proxy)
and plot it vs the actual Letterboxd rating bucket. Highlight outliers:
    - "over-hyped flops": high hype, low rating
    - "sleeper hits": low hype, high rating

Outputs:
    report_figures/fig_expectation_reality.png
    results/expectation_reality.json  (correlation + outlier list)
"""

import json
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FEATS = "results/structured_features/features.csv"
OUT_FIG = "report_figures/fig_expectation_reality.png"
OUT_JSON = "results/expectation_reality.json"

os.makedirs("report_figures", exist_ok=True)
os.makedirs("results", exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                     "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


def main():
    feats = pd.read_csv(FEATS)
    df = feats[["tmdb_id", "title", "rating_bucket", "rating_bucket_label",
                "hype_frac", "nega_frac", "hype_minus_neg"]].copy()

    # Pearson correlation between hype_minus_neg and rating_bucket
    r = df["hype_minus_neg"].corr(df["rating_bucket"], method="pearson")
    rho = df["hype_minus_neg"].corr(df["rating_bucket"], method="spearman")
    print(f"Pearson  corr(hype_minus_neg, rating_bucket) = {r:.3f}")
    print(f"Spearman corr(hype_minus_neg, rating_bucket) = {rho:.3f}")

    # Outlier definitions:
    #   over-hyped flop: hype_minus_neg in top quartile, rating_bucket in {0,1}
    #   sleeper hit:     hype_minus_neg in bottom quartile, rating_bucket in {3,4}
    h_top = df["hype_minus_neg"].quantile(0.75)
    h_bot = df["hype_minus_neg"].quantile(0.25)
    over_hyped = df[(df["hype_minus_neg"] >= h_top) &
                    (df["rating_bucket"].isin([0, 1]))].sort_values("hype_minus_neg", ascending=False)
    sleepers = df[(df["hype_minus_neg"] <= h_bot) &
                  (df["rating_bucket"].isin([3, 4]))].sort_values("hype_minus_neg")

    print("\nOver-hyped flops (top by hype, ended up bad/mediocre):")
    print(over_hyped[["title", "hype_minus_neg", "rating_bucket_label"]].head(10).to_string(index=False))
    print("\nSleeper hits (bottom by hype, ended up good/great):")
    print(sleepers[["title", "hype_minus_neg", "rating_bucket_label"]].head(10).to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    rng = np.random.default_rng(0)
    jitter = rng.uniform(-0.15, 0.15, size=len(df))
    colors = {0: "#c03030", 1: "#d97a4a", 2: "#888888", 3: "#5a9b5a", 4: "#2d6b2d"}
    for bucket in range(5):
        mask = df["rating_bucket"] == bucket
        ax.scatter(df.loc[mask, "hype_minus_neg"], df.loc[mask, "rating_bucket"] + jitter[mask],
                   alpha=0.6, s=20, color=colors[bucket],
                   label=df.loc[mask, "rating_bucket_label"].iloc[0])

    # Annotate top-3 outliers in each direction
    for _, row in over_hyped.head(3).iterrows():
        ax.annotate(row["title"][:18], (row["hype_minus_neg"], row["rating_bucket"]),
                    xytext=(5, -8), textcoords="offset points", fontsize=7,
                    color="#7a1010")
    for _, row in sleepers.head(3).iterrows():
        ax.annotate(row["title"][:18], (row["hype_minus_neg"], row["rating_bucket"]),
                    xytext=(5, 4), textcoords="offset points", fontsize=7,
                    color="#1a5a1a")

    # Trend line
    x = df["hype_minus_neg"].values
    y = df["rating_bucket"].values.astype(float)
    coef = np.polyfit(x, y, 1)
    xx = np.linspace(x.min(), x.max(), 50)
    ax.plot(xx, np.polyval(coef, xx), "k--", linewidth=0.8, alpha=0.5,
            label=f"linear fit (Pearson r = {r:.2f})")
    ax.set_xlabel("hype_minus_neg  (fraction hype phrases  −  fraction negative phrases)")
    ax.set_ylabel("Rating bucket")
    ax.set_yticks(range(5))
    ax.set_yticklabels(["bad", "mediocre", "average", "good", "great"], fontsize=8)
    ax.set_title("Expectation–reality gap: pre-release hype vs post-release rating")
    ax.legend(fontsize=7, loc="upper left", ncol=2, frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_FIG, bbox_inches="tight")
    plt.close()
    print(f"\nSaved figure: {OUT_FIG}")

    out = {
        "pearson_correlation": round(float(r), 4),
        "spearman_correlation": round(float(rho), 4),
        "n_movies": len(df),
        "over_hyped_flops_top10": over_hyped.head(10)[
            ["tmdb_id", "title", "hype_minus_neg", "rating_bucket", "rating_bucket_label"]
        ].to_dict("records"),
        "sleeper_hits_top10": sleepers.head(10)[
            ["tmdb_id", "title", "hype_minus_neg", "rating_bucket", "rating_bucket_label"]
        ].to_dict("records"),
    }
    with open(OUT_JSON, "w") as f:
        json.dump(out, f, indent=2)
    print(f"Saved: {OUT_JSON}")


if __name__ == "__main__":
    main()
