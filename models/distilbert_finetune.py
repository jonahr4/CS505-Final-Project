#!/usr/bin/env python3
"""
Model 4: Fine-tune DistilBERT end-to-end (no pooling, [CLS] head).

Per midterm grader feedback: instead of using a frozen DistilBERT to embed
comments and then mean-pooling the embeddings, we train DistilBERT itself
on the classification task with a head on the [CLS] token.

Setup:
    Training:
        - Each comment from a movie inherits that movie's rating bucket as its
          label (per-comment pseudo-labels). This converts our 89-train-movie
          problem into a ~150K training-example problem.
        - Standard HF AutoModelForSequenceClassification on
          distilbert-base-uncased, [CLS] -> Linear(5).
        - AdamW, lr=2e-5, weight_decay=0.01, batch=32, max_length=128.
        - Up to MAX_EPOCHS, with early stopping on movie-level val macro F1.

    Inference (movie-level):
        For each test/val movie, run all its comments through the model, take
        the softmax probabilities, mean-pool the probabilities across the
        movie's comments, then argmax. This is "decision aggregation" — the
        decision is at the movie level (matching our split), but each comment
        gets a vote weighted by its softmax confidence.

Outputs:
    results/distilbert_finetune/
        results.json               (metrics for train, val, test, including MAE)
        model/                     (final HF model dir, only if SAVE_MODEL=True)
        per_movie_predictions.csv  (so we can do error analysis)
"""

import json
import os
import random
import sys
import time
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ---------------- Config ----------------

IN_JSONL = "data/processed/movie_comments.jsonl"
OUT_DIR = "results/distilbert_finetune"

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128
N_CLASSES = 5

# Training hyperparameters
BATCH_SIZE = 32
LR = 2e-5
WEIGHT_DECAY = 0.01
WARMUP_FRAC = 0.06
MAX_EPOCHS = 3
PATIENCE = 1                   # epochs of no val-F1 improvement before stopping
RANDOM_SEED = 42

# To control runtime: cap the number of comments per movie used for TRAINING
# (still use all val/test comments at inference). With ~150K total comments
# across 121 train movies, capping at 800/movie keeps train ≈100K examples
# and one epoch ≈ 30-40 min on MPS.
MAX_TRAIN_COMMENTS_PER_MOVIE = 800
MAX_INFER_COMMENTS_PER_MOVIE = 400  # bounds inference time per movie

SAVE_MODEL = False             # final model is large (~250MB); skip by default


# ---------------- Setup ----------------

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------- Data ----------------

@dataclass
class CommentExample:
    text: str
    label: int
    movie_idx: int    # index into the movie_meta list


def load_data(jsonl_path: str) -> Tuple[List[Dict], List[List[CommentExample]]]:
    """Returns (movie_meta, per_movie_comments).

    movie_meta[i] has tmdb_id, title, rating_bucket, rating_bucket_label, split.
    per_movie_comments[i] is a list of CommentExample for that movie.
    """
    movie_meta: List[Dict] = []
    per_movie_comments: List[List[CommentExample]] = []

    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            rec = json.loads(line)
            meta = {
                "tmdb_id": rec["tmdb_id"],
                "title": rec["title"],
                "rating_bucket": rec["rating_bucket"],
                "rating_bucket_label": rec["rating_bucket_label"],
                "split": rec["split"],
            }
            comments = []
            for c in rec.get("comments", []):
                t = (c.get("text") or "").strip()
                if t:
                    comments.append(CommentExample(
                        text=t, label=int(rec["rating_bucket"]), movie_idx=i,
                    ))
            movie_meta.append(meta)
            per_movie_comments.append(comments)

    return movie_meta, per_movie_comments


def build_train_examples(movie_meta, per_movie_comments,
                         cap_per_movie: int, rng: random.Random) -> List[CommentExample]:
    """Per-comment training set, capped per movie to control runtime."""
    out: List[CommentExample] = []
    for meta, comments in zip(movie_meta, per_movie_comments):
        if meta["split"] != "train":
            continue
        if len(comments) > cap_per_movie:
            comments = rng.sample(comments, cap_per_movie)
        out.extend(comments)
    rng.shuffle(out)
    return out


class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], tokenizer, max_length: int):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True, padding="max_length", max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


# ---------------- Metrics ----------------

def evaluate_movie_level(per_movie_probs: List[np.ndarray],
                         per_movie_labels: List[int]) -> Dict:
    """Aggregate comment-level softmax probs → movie-level prediction.

    For each movie: mean-pool comment-level probs, argmax.
    Compute movie-level accuracy, macro F1, MAE, and confusion matrix.
    """
    n = len(per_movie_labels)
    y_true = np.array(per_movie_labels)
    y_pred = np.array([int(np.argmax(p.mean(axis=0))) if p.shape[0] > 0
                       else 0 for p in per_movie_probs])

    acc = float(np.mean(y_true == y_pred))
    f1s = []
    per_class = {}
    for c in range(N_CLASSES):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": int(np.sum(y_true == c))}
        f1s.append(f1)
    macro_f1 = float(np.mean(f1s))

    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    weights = np.abs(np.arange(N_CLASSES)[:, None] - np.arange(N_CLASSES)[None, :])
    mae = float((cm * weights).sum() / max(cm.sum(), 1))

    return {
        "n": n,
        "accuracy": round(acc, 4),
        "macro_f1": round(macro_f1, 4),
        "mae": round(mae, 4),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
        "predictions": y_pred.tolist(),
    }


# ---------------- Training loop ----------------

def train_one_epoch(model, loader, optimizer, scheduler, device, log_every=50):
    model.train()
    total_loss = 0.0
    n_batches = 0
    t0 = time.time()
    for step, batch in enumerate(loader, start=1):
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        total_loss += float(loss.item())
        n_batches += 1
        if step % log_every == 0:
            elapsed = time.time() - t0
            print(f"    step {step}/{len(loader)}  loss={total_loss/n_batches:.4f}  "
                  f"elapsed={elapsed:.0f}s")
    return total_loss / max(n_batches, 1)


def predict_proba_for_movie(model, tokenizer, comments, device,
                            batch_size: int, max_length: int) -> np.ndarray:
    """Return (n_comments, N_CLASSES) softmax probabilities for one movie's comments."""
    if not comments:
        return np.zeros((0, N_CLASSES), dtype=np.float32)
    texts = [c.text for c in comments]
    out = []
    model.eval()
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, truncation=True, padding="max_length",
                            max_length=max_length, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            out.append(probs)
    return np.vstack(out)


def evaluate_split(model, tokenizer, movie_meta, per_movie_comments,
                   split_name: str, device, infer_batch: int,
                   max_infer_per_movie: int, rng: random.Random) -> Dict:
    """Run inference on every movie in this split, return movie-level metrics."""
    per_movie_probs = []
    per_movie_labels = []
    movie_keys = []
    for meta, comments in zip(movie_meta, per_movie_comments):
        if meta["split"] != split_name:
            continue
        if len(comments) > max_infer_per_movie:
            comments = rng.sample(comments, max_infer_per_movie)
        probs = predict_proba_for_movie(model, tokenizer, comments,
                                        device, infer_batch, MAX_LENGTH)
        per_movie_probs.append(probs)
        per_movie_labels.append(meta["rating_bucket"])
        movie_keys.append((meta["tmdb_id"], meta["title"],
                           meta["rating_bucket_label"]))
    metrics = evaluate_movie_level(per_movie_probs, per_movie_labels)
    metrics["movie_keys"] = movie_keys
    return metrics


# ---------------- Main ----------------

def main():
    if not os.path.exists(IN_JSONL):
        print(f"ERROR: {IN_JSONL} missing.")
        sys.exit(1)

    set_seed(RANDOM_SEED)
    rng = random.Random(RANDOM_SEED)
    device = get_device()
    print(f"Device: {device}")
    if device == "cpu":
        print("WARNING: training on CPU will be slow. MPS not available.")

    print("Loading data...")
    movie_meta, per_movie_comments = load_data(IN_JSONL)
    n_train_movies = sum(1 for m in movie_meta if m["split"] == "train")
    n_val_movies = sum(1 for m in movie_meta if m["split"] == "val")
    n_test_movies = sum(1 for m in movie_meta if m["split"] == "test")
    print(f"Movies: {len(movie_meta)} total | train={n_train_movies} "
          f"val={n_val_movies} test={n_test_movies}")

    train_examples = build_train_examples(
        movie_meta, per_movie_comments, MAX_TRAIN_COMMENTS_PER_MOVIE, rng
    )
    print(f"Training examples (per-comment): {len(train_examples)} "
          f"(cap {MAX_TRAIN_COMMENTS_PER_MOVIE}/movie)")

    print("Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=N_CLASSES
    ).to(device)

    train_dataset = TextDataset(
        [e.text for e in train_examples],
        [e.label for e in train_examples],
        tokenizer, MAX_LENGTH,
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE,
                              shuffle=True, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                  weight_decay=WEIGHT_DECAY)
    n_steps = len(train_loader) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(WARMUP_FRAC * n_steps),
        num_training_steps=n_steps,
    )

    best_val_f1 = -1.0
    best_state = None
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, MAX_EPOCHS + 1):
        print(f"\n=== Epoch {epoch}/{MAX_EPOCHS} ===")
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, device)
        train_time = time.time() - t0
        print(f"  train_loss={train_loss:.4f}  epoch_time={train_time:.0f}s")

        print("  evaluating val...")
        val_metrics = evaluate_split(
            model, tokenizer, movie_meta, per_movie_comments,
            "val", device, infer_batch=64,
            max_infer_per_movie=MAX_INFER_COMMENTS_PER_MOVIE, rng=rng,
        )
        print(f"  val: acc={val_metrics['accuracy']}  "
              f"macroF1={val_metrics['macro_f1']}  mae={val_metrics['mae']}")

        history.append({
            "epoch": epoch, "train_loss": round(train_loss, 4),
            "val_acc": val_metrics["accuracy"],
            "val_f1": val_metrics["macro_f1"],
            "val_mae": val_metrics["mae"],
        })

        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            epochs_without_improvement = 0
            print(f"  ↑ new best val F1: {best_val_f1}")
        else:
            epochs_without_improvement += 1
            print(f"  no improvement (patience {epochs_without_improvement}/{PATIENCE})")
            if epochs_without_improvement > PATIENCE:
                print("  early stopping")
                break

    print("\nLoading best model weights for final eval...")
    if best_state is not None:
        model.load_state_dict(best_state)

    final_results = {}
    for split in ("train", "val", "test"):
        print(f"  evaluating {split}...")
        metrics = evaluate_split(
            model, tokenizer, movie_meta, per_movie_comments,
            split, device, infer_batch=64,
            max_infer_per_movie=MAX_INFER_COMMENTS_PER_MOVIE, rng=rng,
        )
        final_results[split] = metrics
        print(f"  {split}: acc={metrics['accuracy']}  "
              f"macroF1={metrics['macro_f1']}  mae={metrics['mae']}")

    os.makedirs(OUT_DIR, exist_ok=True)

    # Per-movie predictions for downstream error analysis
    pred_rows = []
    for split in ("val", "test"):
        keys = final_results[split]["movie_keys"]
        preds = final_results[split]["predictions"]
        for (tmdb_id, title, label_str), pred in zip(keys, preds):
            pred_rows.append({
                "split": split, "tmdb_id": tmdb_id, "title": title,
                "true_label": label_str, "predicted_bucket": pred,
            })
    pd.DataFrame(pred_rows).to_csv(
        os.path.join(OUT_DIR, "per_movie_predictions.csv"), index=False
    )

    # Strip non-serializable bits before saving
    serializable_results = {}
    for split, m in final_results.items():
        m2 = {k: v for k, v in m.items() if k != "movie_keys"}
        serializable_results[split] = m2

    out = {
        "model": "distilbert_finetune_cls",
        "base_model": MODEL_NAME,
        "config": {
            "max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
            "lr": LR, "weight_decay": WEIGHT_DECAY, "warmup_frac": WARMUP_FRAC,
            "max_epochs": MAX_EPOCHS, "patience": PATIENCE,
            "max_train_comments_per_movie": MAX_TRAIN_COMMENTS_PER_MOVIE,
            "max_infer_comments_per_movie": MAX_INFER_COMMENTS_PER_MOVIE,
            "seed": RANDOM_SEED, "device": device,
        },
        "training_history": history,
        "best_val_f1": best_val_f1,
        "results": serializable_results,
    }
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {os.path.join(OUT_DIR, 'results.json')}")
    print(f"Saved: {os.path.join(OUT_DIR, 'per_movie_predictions.csv')}")

    if SAVE_MODEL:
        model_dir = os.path.join(OUT_DIR, "model")
        model.save_pretrained(model_dir)
        tokenizer.save_pretrained(model_dir)
        print(f"Saved model to: {model_dir}")


if __name__ == "__main__":
    main()
