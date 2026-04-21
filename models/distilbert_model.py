#!/usr/bin/env python3
"""
Model 3: DistilBERT embeddings, sampled comments, aggregated to movie level,
classifier on top.

Approach (feature-extraction, not fine-tuning — appropriate for n=127):
    1. Sample up to MAX_COMMENTS_PER_MOVIE comments per movie (too many would
       bias toward super-popular movies).
    2. Run each sampled comment through distilbert-base-uncased. Take the
       mean-pooled hidden state (768-d vector) as the comment embedding.
    3. Mean-pool comment embeddings to a single 768-d vector per movie.
    4. Train LogisticRegression on movie-level embeddings.

Fine-tuning would be the next step but isn't feasible at n=89 train movies
without severe overfitting and would need careful regularization.

On CPU this is slow but finite — expect ~10-20 min runtime depending on
hardware.

Outputs:
    results/distilbert/movie_embeddings.npy
    results/distilbert/results.json
"""

import json
import os
import random
import sys
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from transformers import AutoModel, AutoTokenizer

IN_JSONL = "data/processed/movie_comments.jsonl"
OUT_DIR = "results/distilbert"
OUT_EMB = os.path.join(OUT_DIR, "movie_embeddings.npy")
OUT_META = os.path.join(OUT_DIR, "movie_embeddings_meta.csv")
OUT_JSON = os.path.join(OUT_DIR, "results.json")

MODEL_NAME = "distilbert-base-uncased"
MAX_COMMENTS_PER_MOVIE = 200      # cap for cost — preserves signal, bounds runtime
BATCH_SIZE = 32
MAX_LENGTH = 96                   # trailer comments are short; 96 tokens covers most
RANDOM_SEED = 42


def mean_pool(last_hidden: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
    """Mean-pool token embeddings using attention mask (ignore padding)."""
    mask = attn_mask.unsqueeze(-1).float()
    summed = (last_hidden * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def embed_comments(
    texts: List[str], tokenizer, model, device: str
) -> np.ndarray:
    """Embed a list of comments; returns (N, 768) array."""
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i + BATCH_SIZE]
            enc = tokenizer(
                batch,
                padding=True, truncation=True, max_length=MAX_LENGTH,
                return_tensors="pt",
            ).to(device)
            out_h = model(**enc).last_hidden_state  # (B, T, H)
            pooled = mean_pool(out_h, enc["attention_mask"])  # (B, H)
            out.append(pooled.cpu().numpy())
    return np.vstack(out) if out else np.zeros((0, 768), dtype=np.float32)


def evaluate(y_true, y_pred, n_classes):
    n = len(y_true)
    acc = float(np.mean(y_true == y_pred))
    per_class, f1s = {}, []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": int(np.sum(y_true == c))}
        f1s.append(f1)
    return {"n": n, "accuracy": round(acc, 4),
            "macro_f1": round(float(np.mean(f1s)), 4), "per_class": per_class}


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def main() -> None:
    if not os.path.exists(IN_JSONL):
        print(f"ERROR: {IN_JSONL} not found.")
        sys.exit(1)

    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)

    device = "cuda" if torch.cuda.is_available() else (
        "mps" if torch.backends.mps.is_available() else "cpu"
    )
    print(f"Device: {device}")

    print(f"Loading {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).to(device)
    print("Model loaded.")

    # Load movies
    records = []
    with open(IN_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            records.append(json.loads(line))
    print(f"Loaded {len(records)} movies")

    movie_vectors = []
    movie_meta = []

    for i, rec in enumerate(records, start=1):
        comments = rec.get("comments") or []
        texts = [(c.get("text") or "").strip() for c in comments]
        texts = [t for t in texts if t]  # drop empties

        if not texts:
            print(f"  [{i}/{len(records)}] {rec['title'][:40]!r}: NO TEXTS, zero vector")
            vec = np.zeros(768, dtype=np.float32)
        else:
            if len(texts) > MAX_COMMENTS_PER_MOVIE:
                # Prefer long comments (more signal); fall back to random sample
                # Score = length; take top-K by length then shuffle for variety
                idx = np.argsort([-len(t) for t in texts])[:MAX_COMMENTS_PER_MOVIE * 2]
                pick = list(idx)
                random.shuffle(pick)
                texts = [texts[j] for j in pick[:MAX_COMMENTS_PER_MOVIE]]

            comment_embs = embed_comments(texts, tokenizer, model, device)
            vec = comment_embs.mean(axis=0).astype(np.float32)

        movie_vectors.append(vec)
        movie_meta.append({
            "tmdb_id": rec["tmdb_id"],
            "title": rec["title"],
            "rating_bucket": rec["rating_bucket"],
            "rating_bucket_label": rec["rating_bucket_label"],
            "split": rec["split"],
            "n_comments_used": len(texts),
        })
        if i % 5 == 0 or i == len(records):
            print(f"  [{i}/{len(records)}] {rec['title'][:40]!r}: used {len(texts)} comments")

    X = np.vstack(movie_vectors)
    meta = pd.DataFrame(movie_meta)

    os.makedirs(OUT_DIR, exist_ok=True)
    np.save(OUT_EMB, X)
    meta.to_csv(OUT_META, index=False)
    print(f"\nSaved embeddings: {OUT_EMB}  shape={X.shape}")
    print(f"Saved meta: {OUT_META}")

    # Classifier on movie embeddings
    train_mask = meta["split"] == "train"
    val_mask = meta["split"] == "val"
    test_mask = meta["split"] == "test"

    X_train, X_val, X_test = X[train_mask], X[val_mask], X[test_mask]
    y_train = meta.loc[train_mask, "rating_bucket"].to_numpy()
    y_val = meta.loc[val_mask, "rating_bucket"].to_numpy()
    y_test = meta.loc[test_mask, "rating_bucket"].to_numpy()
    n_classes = int(meta["rating_bucket"].max()) + 1

    scaler = StandardScaler().fit(X_train)
    X_train_s = scaler.transform(X_train)
    X_val_s = scaler.transform(X_val)
    X_test_s = scaler.transform(X_test)

    print("\nTraining LogReg on DistilBERT movie embeddings...")
    clf = LogisticRegression(
        C=1.0, max_iter=5000,
        class_weight="balanced", solver="lbfgs",
        random_state=RANDOM_SEED,
    )
    clf.fit(X_train_s, y_train)

    results = {}
    for name, Xs, y in [("train", X_train_s, y_train),
                        ("val", X_val_s, y_val),
                        ("test", X_test_s, y_test)]:
        y_pred = clf.predict(Xs)
        m = evaluate(y, y_pred, n_classes)
        m["confusion_matrix"] = confusion_matrix(y, y_pred, n_classes).tolist()
        results[name] = m
        print(f"  {name}: acc={m['accuracy']}  macroF1={m['macro_f1']}")

    out = {
        "model": "distilbert_features_logreg",
        "base_model": MODEL_NAME,
        "max_comments_per_movie": MAX_COMMENTS_PER_MOVIE,
        "max_length_tokens": MAX_LENGTH,
        "pooling": "mean",
        "classifier": "logistic_regression (C=1.0, class_weight=balanced)",
        "embedding_dim": int(X.shape[1]),
        "results": results,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
