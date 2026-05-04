#!/usr/bin/env python3
"""Generate figures for the final report."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = "report_figures"
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 200, "savefig.dpi": 200,
                     "font.size": 9, "axes.spines.top": False,
                     "axes.spines.right": False})


# ---------------------------------------------------------------------------
# Fig A — Final 6-model comparison bar chart (test acc, F1, MAE)
# ---------------------------------------------------------------------------

def load_metrics():
    with open("results/baselines/majority_baseline.json") as f:
        maj = json.load(f)["results"]["test"]
    with open("results/tfidf_logreg/results.json") as f:
        tfidf = json.load(f)["results"]["test"]
    with open("results/structured_features/results.json") as f:
        s = json.load(f)["results"]
        struct_lr = s["logistic_regression"]["test"]
        struct_rf = s["random_forest"]["test"]
    with open("results/distilbert/results.json") as f:
        bert_frozen = json.load(f)["results"]["test"]
    with open("results/distilbert_finetune/results.json") as f:
        bert_pc = json.load(f)["results"]["test"]
    with open("results/distilbert_finetune_movie/results.json") as f:
        bert_ml = json.load(f)["results"]["test"]
    return [
        ("Majority", maj),
        ("TF-IDF + LR", tfidf),
        ("Struct + LR", struct_lr),
        ("Struct + RF", struct_rf),
        ("DistilBERT (frozen)", bert_frozen),
        ("DistilBERT FT\n(per-comment)", bert_pc),
        ("DistilBERT FT\n(movie-level)", bert_ml),
    ]


def mae_from_cm(cm):
    cm = np.array(cm, dtype=float)
    n = cm.shape[0]
    w = np.abs(np.arange(n)[:, None] - np.arange(n)[None, :])
    return float((cm * w).sum() / max(cm.sum(), 1))


metrics = load_metrics()

# Inject MAE if missing (for older runs)
for name, m in metrics:
    if "mae" not in m and "confusion_matrix" in m:
        m["mae"] = round(mae_from_cm(m["confusion_matrix"]), 4)

names = [m[0] for m in metrics]
acc = [m[1].get("accuracy", 0) for m in metrics]
f1 = [m[1].get("macro_f1", 0) for m in metrics]
mae = [m[1].get("mae", 0) for m in metrics]

fig, ax = plt.subplots(figsize=(7.5, 3.2))
x = np.arange(len(names))
w = 0.27
b1 = ax.bar(x - w, acc, w, label="Accuracy ↑", color="#4a6fa5", edgecolor="white")
b2 = ax.bar(x,     f1,  w, label="Macro F1 ↑", color="#c97b4a", edgecolor="white")
b3 = ax.bar(x + w, [m / 4 for m in mae], w, label="MAE / 4 ↓", color="#7a7a7a", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(names, rotation=0, fontsize=7.5)
ax.set_ylabel("Score")
ax.set_ylim(0, 0.55)
ax.set_title("Test-set performance across all model variants (5-class task)", fontsize=10)
ax.legend(frameon=False, fontsize=8, loc="upper right")
for bars in (b1, b2):
    for b in bars:
        if b.get_height() > 0:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.005,
                    f"{b.get_height():.2f}", ha="center", fontsize=6.5)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_final_comparison.png", bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Fig B — TF-IDF confusion matrix (5-class) — keep but restyle for ACL look
# ---------------------------------------------------------------------------
with open("results/tfidf_logreg/results.json") as f:
    tfidf = json.load(f)
cm = np.array(tfidf["results"]["test"]["confusion_matrix"])
labels = ["Bad", "Med.", "Avg.", "Good", "Great"]

fig, ax = plt.subplots(figsize=(3.5, 3.2))
im = ax.imshow(cm, cmap="Blues", vmin=0)
for i in range(5):
    for j in range(5):
        color = "white" if cm[i, j] > cm.max() / 2 else "#222"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=10, color=color, fontweight="bold")
ax.set_xticks(range(5)); ax.set_xticklabels(labels, fontsize=8)
ax.set_yticks(range(5)); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("Predicted", fontsize=9); ax.set_ylabel("True", fontsize=9)
ax.set_title("TF-IDF + LR (5-class)", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_confusion_tfidf.png", bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Fig C — Temporal ablation
# ---------------------------------------------------------------------------
with open("results/temporal_ablation.json") as f:
    ta = json.load(f)
labels_t = [v["label"].replace(" (all temporal windows)", "")
                       .replace(" features", "")
                       .replace("Only ", "")
                       .replace(" window", "")
            for v in ta["variants"]]
f1s = [v["test"]["macro_f1"] for v in ta["variants"]]
maes = [v["test"]["mae"] for v in ta["variants"]]
accs = [v["test"]["accuracy"] for v in ta["variants"]]

fig, ax = plt.subplots(figsize=(6.0, 2.6))
x = np.arange(len(labels_t))
w = 0.27
ax.bar(x - w, accs, w, label="Accuracy", color="#4a6fa5", edgecolor="white")
ax.bar(x,     f1s,  w, label="Macro F1", color="#c97b4a", edgecolor="white")
ax.bar(x + w, [m/4 for m in maes], w, label="MAE / 4 ↓", color="#7a7a7a", edgecolor="white")
ax.set_xticks(x); ax.set_xticklabels(labels_t, fontsize=7.5)
ax.set_ylabel("Score"); ax.set_ylim(0, 0.50)
ax.set_title("Per-window temporal ablation (Structured + LR)", fontsize=9)
ax.legend(frameon=False, fontsize=7.5, loc="upper right")
for i, v in enumerate(f1s):
    ax.text(i, v + 0.005, f"{v:.2f}", ha="center", fontsize=7)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_temporal_ablation.png", bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Fig D — Expectation-reality already generated; copy/regenerate to match style
# (Already exists at fig_expectation_reality.png — leave as is.)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Fig E — Rating distribution (kept from midterm but re-styled)
# ---------------------------------------------------------------------------
df = pd.read_csv("data/processed/movies_dataset.csv")
ratings = df["letterboxd_rating"].astype(float)
fig, ax = plt.subplots(figsize=(6.0, 2.4))
ax.hist(ratings, bins=np.arange(1.4, 4.6, 0.1), color="#4a6fa5",
        edgecolor="white", linewidth=0.5)
for x in [2.3, 2.75, 3.15, 3.5]:
    ax.axvline(x, color="#c03030", linestyle="--", linewidth=0.7)
labels = ["bad", "mediocre", "average", "good", "great"]
centers = [2.0, 2.52, 2.95, 3.33, 3.9]
for lab, c in zip(labels, centers):
    ax.text(c, ax.get_ylim()[1] * 0.92, lab, ha="center",
            fontsize=7, color="#333")
ax.set_xlabel("Letterboxd rating", fontsize=8.5)
ax.set_ylabel("# movies", fontsize=8.5)
ax.set_title("Rating distribution (n=173)", fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig_rating_distribution.png", bbox_inches="tight")
plt.close()

print("Saved figures:")
for f in sorted(os.listdir(OUT_DIR)):
    print(" ", f)
