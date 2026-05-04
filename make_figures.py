#!/usr/bin/env python3
"""Generate midterm-report figures from the saved results and processed dataset."""

import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT_DIR = "report_figures"
os.makedirs(OUT_DIR, exist_ok=True)
plt.rcParams.update({"figure.dpi": 150, "savefig.dpi": 150,
                     "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})


# ---------------------------------------------------------------------------
# Fig 1 — Letterboxd rating distribution + bucket overlay
# ---------------------------------------------------------------------------
df = pd.read_csv("data/processed/movies_dataset.csv")
ratings = df["letterboxd_rating"].astype(float)

fig, ax = plt.subplots(figsize=(6.2, 2.6))
ax.hist(ratings, bins=np.arange(1.4, 4.6, 0.1), color="#4a6fa5",
        edgecolor="white", linewidth=0.5)
for x in [2.3, 2.75, 3.15, 3.5]:
    ax.axvline(x, color="#c03030", linestyle="--", linewidth=0.8)
labels = ["bad", "mediocre", "average", "good", "great"]
centers = [2.0, 2.52, 2.95, 3.33, 3.9]
for lab, c in zip(labels, centers):
    ax.text(c, ax.get_ylim()[1] * 0.92, lab, ha="center",
            fontsize=8, color="#333")
ax.set_xlabel("Letterboxd rating")
ax.set_ylabel("# movies")
ax.set_title("Rating distribution across 173 final movies (bucket boundaries dashed)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig1_rating_distribution.png", bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Fig 2 — Test macro F1 across all models
# ---------------------------------------------------------------------------
with open("results/comparison.json") as f:
    comp = json.load(f)
with open("results/improvements.json") as f:
    imp = json.load(f)

# Pull the 3-class metrics from improvements.json summary
three_class = {row["model"]: row for row in imp["summary_test"]}

models_order = [
    ("Majority baseline", comp[0]),
    ("TF-IDF + LogReg", comp[1]),
    ("Structured + RF", comp[2]),
    ("Structured + LR", comp[3]),
    ("DistilBERT + LR", comp[4]),
]

names = [m[0] for m in models_order]
f1_5 = [m[1]["test_f1"] for m in models_order]
acc_5 = [m[1]["test_acc"] for m in models_order]

fig, ax = plt.subplots(figsize=(6.2, 3.2))
x = np.arange(len(names))
w = 0.38
bars1 = ax.bar(x - w / 2, acc_5, w, label="Accuracy",
               color="#4a6fa5", edgecolor="white")
bars2 = ax.bar(x + w / 2, f1_5, w, label="Macro F1",
               color="#c97b4a", edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(names, rotation=15, ha="right", fontsize=8)
ax.set_ylabel("Score")
ax.set_ylim(0, 0.55)
ax.set_title("5-class test performance")
ax.legend(frameon=False, fontsize=8)
for bars in (bars1, bars2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.01,
                f"{b.get_height():.2f}", ha="center", fontsize=7)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig2_5class_comparison.png", bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Fig 3 — TF-IDF confusion matrix on test (5-class)
# ---------------------------------------------------------------------------
with open("results/tfidf_logreg/results.json") as f:
    tfidf = json.load(f)
cm = np.array(tfidf["results"]["test"]["confusion_matrix"])
labels_short = ["bad", "med", "avg", "good", "great"]

fig, ax = plt.subplots(figsize=(4.0, 3.6))
im = ax.imshow(cm, cmap="Blues", vmin=0)
for i in range(5):
    for j in range(5):
        color = "white" if cm[i, j] > cm.max() / 2 else "#222"
        ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                fontsize=10, color=color)
ax.set_xticks(range(5)); ax.set_xticklabels(labels_short, fontsize=8)
ax.set_yticks(range(5)); ax.set_yticklabels(labels_short, fontsize=8)
ax.set_xlabel("Predicted"); ax.set_ylabel("True")
ax.set_title("TF-IDF + LogReg confusion matrix (test)")
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig3_tfidf_confusion.png", bbox_inches="tight")
plt.close()


# ---------------------------------------------------------------------------
# Fig 4 — Top-10 RF feature importances from structured model
# ---------------------------------------------------------------------------
with open("results/structured_features/results.json") as f:
    struct = json.load(f)
top = struct["top_feature_importances_rf"][:10]
feats = [t["feature"] for t in top][::-1]
imps = [t["importance"] for t in top][::-1]

fig, ax = plt.subplots(figsize=(6.2, 2.8))
ax.barh(feats, imps, color="#4a6fa5", edgecolor="white")
ax.set_xlabel("RF feature importance")
ax.set_title("Top-10 structured features (random forest)")
for i, v in enumerate(imps):
    ax.text(v + 0.001, i, f"{v:.3f}", va="center", fontsize=7)
plt.tight_layout()
plt.savefig(f"{OUT_DIR}/fig4_feature_importance.png", bbox_inches="tight")
plt.close()


print("Wrote figures to", OUT_DIR)
for f in sorted(os.listdir(OUT_DIR)):
    print(" ", f)
