#!/usr/bin/env python3
"""
Model 4b: Fine-tune DistilBERT at the MOVIE level (no per-comment pseudo-labels).

Motivation:
    Per-comment fine-tuning (distilbert_finetune.py) suffers from pseudo-label
    noise — every comment under a "great" movie inherits that label, even
    sarcastic, off-topic, or hostile comments. Movie-level fine-tuning trades
    that noise for fewer examples (n_train=121).

Setup:
    For each movie we build ONE input by sampling and concatenating its
    most-liked comments until we hit roughly the model's max_length budget.
    Each movie = one (input, label) pair. Train end-to-end with [CLS] head.

    Compared to the per-comment approach this is cleaner labels but ~660x
    fewer training examples. Aggressive regularization (high weight decay,
    early stopping) compensates.

Inference:
    For each val/test movie we sample its most-liked comments and concatenate
    the same way as training, run a single forward pass, take argmax of
    softmax. No multi-comment aggregation needed.

Outputs:
    results/distilbert_finetune_movie/results.json
    results/distilbert_finetune_movie/per_movie_predictions.csv
"""

import json
import os
import random
import sys
import time
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    get_linear_schedule_with_warmup,
)

IN_JSONL = "data/processed/movie_comments.jsonl"
OUT_DIR = "results/distilbert_finetune_movie"

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 512
N_CLASSES = 5

BATCH_SIZE = 8                  # smaller batch — sequences are long
LR = 2e-5
WEIGHT_DECAY = 0.05             # heavier than per-comment due to small n
WARMUP_FRAC = 0.1
MAX_EPOCHS = 8                  # small data, more epochs OK with early stop
PATIENCE = 2
RANDOM_SEED = 42

# How many comments per movie to concatenate. We pick top-K by like count to
# bias toward higher-signal comments, then truncate at MAX_LENGTH tokens.
TOP_K_COMMENTS_PER_MOVIE = 60


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


def load_data(jsonl_path: str):
    movie_meta, per_movie_comments = [], []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for rec in map(json.loads, f):
            movie_meta.append({
                "tmdb_id": rec["tmdb_id"], "title": rec["title"],
                "rating_bucket": int(rec["rating_bucket"]),
                "rating_bucket_label": rec["rating_bucket_label"],
                "split": rec["split"],
            })
            per_movie_comments.append(rec.get("comments", []))
    return movie_meta, per_movie_comments


def build_movie_text(comments: List[dict], top_k: int) -> str:
    """Pick top-K comments by like_count, then concatenate. Fallback to longest."""
    if not comments:
        return ""
    scored = [(int(c.get("like_count") or 0), len(c.get("text") or ""), c.get("text") or "")
              for c in comments]
    # Primary sort by likes (desc), tiebreak by length (desc)
    scored.sort(key=lambda x: (-x[0], -x[1]))
    picked = [t for _, _, t in scored[:top_k]]
    return " [SEP] ".join(t.strip().replace("\n", " ") for t in picked if t.strip())


class MovieDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts; self.labels = labels
        self.tokenizer = tokenizer; self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, i):
        enc = self.tokenizer(
            self.texts[i], truncation=True, padding="max_length",
            max_length=self.max_length, return_tensors="pt",
        )
        return {"input_ids": enc["input_ids"].squeeze(0),
                "attention_mask": enc["attention_mask"].squeeze(0),
                "labels": torch.tensor(self.labels[i], dtype=torch.long)}


def evaluate_set(model, loader, device, n_classes):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(input_ids=batch["input_ids"],
                           attention_mask=batch["attention_mask"]).logits
            preds = logits.argmax(dim=-1).cpu().numpy()
            all_pred.extend(preds.tolist())
            all_true.extend(batch["labels"].cpu().numpy().tolist())
    y_true = np.array(all_true); y_pred = np.array(all_pred)
    n = len(y_true); acc = float(np.mean(y_true == y_pred))
    f1s = []; per_class = {}
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp/(tp+fp) if (tp+fp) > 0 else 0.0
        rec = tp/(tp+fn) if (tp+fn) > 0 else 0.0
        f1 = 2*prec*rec/(prec+rec) if (prec+rec) > 0 else 0.0
        per_class[c] = {"precision": round(prec, 4), "recall": round(rec, 4),
                        "f1": round(f1, 4), "support": int(np.sum(y_true == c))}
        f1s.append(f1)
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    weights = np.abs(np.arange(n_classes)[:, None] - np.arange(n_classes)[None, :])
    mae = float((cm*weights).sum() / max(cm.sum(), 1))
    return {"n": n, "accuracy": round(acc, 4), "macro_f1": round(float(np.mean(f1s)), 4),
            "mae": round(mae, 4), "per_class": per_class,
            "confusion_matrix": cm.tolist(),
            "predictions": y_pred.tolist(), "labels": y_true.tolist()}


def main():
    if not os.path.exists(IN_JSONL):
        print(f"ERROR: {IN_JSONL} missing"); sys.exit(1)
    set_seed(RANDOM_SEED)
    device = get_device()
    print(f"Device: {device}")

    print("Loading data...")
    movie_meta, per_movie_comments = load_data(IN_JSONL)
    print(f"Movies: {len(movie_meta)}")

    print(f"Building movie-level texts (top-{TOP_K_COMMENTS_PER_MOVIE} by likes)...")
    movie_text = [build_movie_text(c, TOP_K_COMMENTS_PER_MOVIE) for c in per_movie_comments]
    char_lens = [len(t) for t in movie_text]
    print(f"  movie text chars — min={min(char_lens)} med={int(np.median(char_lens))} max={max(char_lens)}")

    by_split = {"train": [], "val": [], "test": []}
    for i, meta in enumerate(movie_meta):
        by_split[meta["split"]].append(i)
    print(f"Splits: train={len(by_split['train'])}  "
          f"val={len(by_split['val'])}  test={len(by_split['test'])}")

    print("Loading tokenizer & model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=N_CLASSES,
    ).to(device)

    def make_loader(idx_list, shuffle):
        return DataLoader(
            MovieDataset(
                [movie_text[i] for i in idx_list],
                [movie_meta[i]["rating_bucket"] for i in idx_list],
                tokenizer, MAX_LENGTH,
            ), batch_size=BATCH_SIZE, shuffle=shuffle, num_workers=0,
        )

    train_loader = make_loader(by_split["train"], shuffle=True)
    val_loader = make_loader(by_split["val"], shuffle=False)
    test_loader = make_loader(by_split["test"], shuffle=False)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    n_steps = len(train_loader) * MAX_EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=int(WARMUP_FRAC*n_steps),
        num_training_steps=n_steps,
    )

    best_f1 = -1.0; best_state = None; bad_epochs = 0
    history = []
    for epoch in range(1, MAX_EPOCHS+1):
        print(f"\n=== Epoch {epoch}/{MAX_EPOCHS} ===")
        model.train()
        t0 = time.time(); losses = []
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            outputs.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); scheduler.step(); optimizer.zero_grad()
            losses.append(float(outputs.loss.item()))
        train_loss = float(np.mean(losses))
        ep_t = time.time()-t0
        val_m = evaluate_set(model, val_loader, device, N_CLASSES)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 4),
                        "val_acc": val_m["accuracy"], "val_f1": val_m["macro_f1"],
                        "val_mae": val_m["mae"], "epoch_time_s": round(ep_t, 1)})
        print(f"  train_loss={train_loss:.4f}  ep_time={ep_t:.0f}s")
        print(f"  val: acc={val_m['accuracy']}  macroF1={val_m['macro_f1']}  mae={val_m['mae']}")
        if val_m["macro_f1"] > best_f1:
            best_f1 = val_m["macro_f1"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad_epochs = 0
            print(f"  ↑ new best val F1: {best_f1}")
        else:
            bad_epochs += 1
            print(f"  no improvement ({bad_epochs}/{PATIENCE})")
            if bad_epochs > PATIENCE:
                print("  early stopping"); break

    print("\nLoading best weights for final eval...")
    if best_state is not None:
        model.load_state_dict(best_state)

    final = {}
    for name, loader, idxs in [("train", make_loader(by_split["train"], False), by_split["train"]),
                               ("val", val_loader, by_split["val"]),
                               ("test", test_loader, by_split["test"])]:
        m = evaluate_set(model, loader, device, N_CLASSES)
        m["movie_keys"] = [(movie_meta[i]["tmdb_id"], movie_meta[i]["title"],
                            movie_meta[i]["rating_bucket_label"]) for i in idxs]
        final[name] = m
        print(f"  {name}: acc={m['accuracy']}  macroF1={m['macro_f1']}  mae={m['mae']}")

    os.makedirs(OUT_DIR, exist_ok=True)
    pred_rows = []
    for split in ("val", "test"):
        for (tmdb_id, title, lbl), pred, true in zip(
                final[split]["movie_keys"],
                final[split]["predictions"], final[split]["labels"]):
            pred_rows.append({"split": split, "tmdb_id": tmdb_id, "title": title,
                              "true_label": lbl, "true_bucket": int(true),
                              "predicted_bucket": int(pred)})
    pd.DataFrame(pred_rows).to_csv(os.path.join(OUT_DIR, "per_movie_predictions.csv"), index=False)

    serializable = {}
    for k, v in final.items():
        v2 = {k2: vv for k2, vv in v.items() if k2 != "movie_keys"}
        serializable[k] = v2

    out = {"model": "distilbert_finetune_movie_level",
           "base_model": MODEL_NAME,
           "config": {"max_length": MAX_LENGTH, "batch_size": BATCH_SIZE,
                      "lr": LR, "weight_decay": WEIGHT_DECAY,
                      "warmup_frac": WARMUP_FRAC, "max_epochs": MAX_EPOCHS,
                      "patience": PATIENCE,
                      "top_k_comments_per_movie": TOP_K_COMMENTS_PER_MOVIE,
                      "seed": RANDOM_SEED, "device": device},
           "training_history": history, "best_val_f1": best_f1,
           "results": serializable}
    with open(os.path.join(OUT_DIR, "results.json"), "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {os.path.join(OUT_DIR, 'results.json')}")


if __name__ == "__main__":
    main()
