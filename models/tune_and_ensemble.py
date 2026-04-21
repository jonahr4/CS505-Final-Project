#!/usr/bin/env python3
"""
Post-hoc improvements on the 127-movie dataset:

    A) 3-class re-evaluation:
       Collapse the 5-class bucket into {negative, neutral, positive}:
           negative: {bad, mediocre}  (buckets 0, 1)
           neutral:  {average}         (bucket 2)
           positive: {good, great}     (buckets 3, 4)
       Re-evaluate ALL existing 5-class model predictions in 3-class space
       (a prediction of "great" that should have been "good" is now correct).

    B) Hyperparameter sweep on TF-IDF + LogReg (5-class):
       Small grid over C, ngram_range, min_df, max_features. Select by
       validation macro F1; report final test metrics for the best config.

    C) Ensemble of TF-IDF, Structured-LR, DistilBERT:
       Soft-vote (average predicted probabilities) on val and test.

All outputs append to results/comparison.json.
"""

import json
import os
from typing import Dict, List

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42

IN_CSV = "data/processed/movies_dataset.csv"
EMB_NPY = "results/distilbert/movie_embeddings.npy"
EMB_META = "results/distilbert/movie_embeddings_meta.csv"
FEATS_CSV = "results/structured_features/features.csv"
OUT_JSON = "results/improvements.json"


def evaluate(y_true, y_pred, n_classes):
    n = len(y_true)
    acc = float(np.mean(y_true == y_pred))
    f1s = []
    for c in range(n_classes):
        tp = int(np.sum((y_pred == c) & (y_true == c)))
        fp = int(np.sum((y_pred == c) & (y_true != c)))
        fn = int(np.sum((y_pred != c) & (y_true == c)))
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        f1s.append(f1)
    return {"n": n, "accuracy": round(acc, 4),
            "macro_f1": round(float(np.mean(f1s)), 4)}


def confusion(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm.tolist()


def to_3class(y):
    """5-class {0,1,2,3,4} -> 3-class {0:neg, 1:neu, 2:pos}."""
    y = np.asarray(y)
    out = np.full_like(y, -1)
    out[(y == 0) | (y == 1)] = 0
    out[y == 2] = 1
    out[(y == 3) | (y == 4)] = 2
    return out


# ---------------------------------------------------------------------------
# Data loading shared across all three improvements
# ---------------------------------------------------------------------------

def load_splits():
    df = pd.read_csv(IN_CSV)
    df["comments_text"] = df["comments_text"].fillna("").astype(str)
    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)
    test = df[df["split"] == "test"].reset_index(drop=True)
    return df, train, val, test


# ---------------------------------------------------------------------------
# B) TF-IDF hyperparameter sweep
# ---------------------------------------------------------------------------

def tfidf_sweep(train, val, test):
    print("\n[B] TF-IDF hyperparameter sweep")

    grid = {
        "C": [0.1, 0.3, 1.0, 3.0, 10.0],
        "ngram_range": [(1, 1), (1, 2), (1, 3)],
        "min_df": [1, 2, 3],
        "max_features": [5000, 20000, 50000],
    }
    y_tr = train["rating_bucket"].to_numpy()
    y_val = val["rating_bucket"].to_numpy()
    y_te = test["rating_bucket"].to_numpy()
    n_classes = 5

    best = None
    all_configs = []
    for ng in grid["ngram_range"]:
        for mdf in grid["min_df"]:
            for mf in grid["max_features"]:
                vec = TfidfVectorizer(
                    ngram_range=ng, min_df=mdf, max_features=mf,
                    max_df=0.95, stop_words="english",
                    lowercase=True, strip_accents="unicode",
                )
                Xtr = vec.fit_transform(train["comments_text"])
                Xvl = vec.transform(val["comments_text"])
                Xte = vec.transform(test["comments_text"])
                for C in grid["C"]:
                    clf = LogisticRegression(
                        C=C, max_iter=2000, class_weight="balanced",
                        solver="lbfgs", random_state=RANDOM_SEED,
                    )
                    clf.fit(Xtr, y_tr)
                    val_pred = clf.predict(Xvl)
                    vm = evaluate(y_val, val_pred, n_classes)
                    rec = {"C": C, "ngram": list(ng), "min_df": mdf,
                           "max_features": mf,
                           "val_acc": vm["accuracy"], "val_f1": vm["macro_f1"]}
                    all_configs.append(rec)
                    if best is None or vm["macro_f1"] > best["val_f1"]:
                        best = dict(rec)
                        best["_vec"] = vec
                        best["_clf"] = clf
                        best["_Xte"] = Xte
                        best["_Xvl"] = Xvl

    print(f"  Evaluated {len(all_configs)} configs")
    print(f"  Best val: C={best['C']} ngram={best['ngram']} "
          f"min_df={best['min_df']} max_features={best['max_features']} "
          f"-> val_f1={best['val_f1']}")

    # Final metrics with best config
    y_val_pred = best["_clf"].predict(best["_Xvl"])
    y_te_pred = best["_clf"].predict(best["_Xte"])
    val_5 = evaluate(y_val, y_val_pred, 5)
    test_5 = evaluate(y_te, y_te_pred, 5)
    val_5["confusion_matrix"] = confusion(y_val, y_val_pred, 5)
    test_5["confusion_matrix"] = confusion(y_te, y_te_pred, 5)

    y_val_pred_3 = to_3class(y_val_pred)
    y_te_pred_3 = to_3class(y_te_pred)
    val_3 = evaluate(to_3class(y_val), y_val_pred_3, 3)
    test_3 = evaluate(to_3class(y_te), y_te_pred_3, 3)
    val_3["confusion_matrix"] = confusion(to_3class(y_val), y_val_pred_3, 3)
    test_3["confusion_matrix"] = confusion(to_3class(y_te), y_te_pred_3, 3)

    print(f"  Tuned TF-IDF 5-class test: acc={test_5['accuracy']}  f1={test_5['macro_f1']}")
    print(f"  Tuned TF-IDF 3-class test: acc={test_3['accuracy']}  f1={test_3['macro_f1']}")

    return {
        "best_config": {k: v for k, v in best.items() if not k.startswith("_")},
        "all_configs": all_configs,
        "results_5class": {"val": val_5, "test": test_5},
        "results_3class": {"val": val_3, "test": test_3},
        # probability outputs for ensembling
        "_proba_val": best["_clf"].predict_proba(best["_Xvl"]),
        "_proba_test": best["_clf"].predict_proba(best["_Xte"]),
    }


# ---------------------------------------------------------------------------
# Proba from the structured-LR and DistilBERT pipelines (rerun to get probas)
# ---------------------------------------------------------------------------

def structured_probas(train, val, test):
    print("\n[helper] Re-fit structured-LR to get probas")
    feats = pd.read_csv(FEATS_CSV)
    # Align order
    def split_rows(split_name, ref_df):
        fs = feats[feats["split"] == split_name].set_index("tmdb_id")
        return fs.loc[ref_df["tmdb_id"]]
    feat_cols = [c for c in feats.columns if c not in
                 {"tmdb_id", "title", "rating_bucket",
                  "rating_bucket_label", "split"}]
    Xtr = split_rows("train", train)[feat_cols].to_numpy()
    Xvl = split_rows("val", val)[feat_cols].to_numpy()
    Xte = split_rows("test", test)[feat_cols].to_numpy()
    y_tr = train["rating_bucket"].to_numpy()
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=5000,
                             class_weight="balanced",
                             solver="lbfgs", random_state=RANDOM_SEED)
    clf.fit(scaler.transform(Xtr), y_tr)
    return (clf.predict_proba(scaler.transform(Xvl)),
            clf.predict_proba(scaler.transform(Xte)))


def distilbert_probas(train, val, test):
    print("[helper] Re-fit DistilBERT-LR to get probas")
    X_all = np.load(EMB_NPY)
    meta = pd.read_csv(EMB_META)
    order = {tid: i for i, tid in enumerate(meta["tmdb_id"])}
    idx_tr = [order[t] for t in train["tmdb_id"]]
    idx_vl = [order[t] for t in val["tmdb_id"]]
    idx_te = [order[t] for t in test["tmdb_id"]]
    Xtr, Xvl, Xte = X_all[idx_tr], X_all[idx_vl], X_all[idx_te]
    y_tr = train["rating_bucket"].to_numpy()
    scaler = StandardScaler().fit(Xtr)
    clf = LogisticRegression(C=1.0, max_iter=5000,
                             class_weight="balanced",
                             solver="lbfgs", random_state=RANDOM_SEED)
    clf.fit(scaler.transform(Xtr), y_tr)
    return (clf.predict_proba(scaler.transform(Xvl)),
            clf.predict_proba(scaler.transform(Xte)))


# ---------------------------------------------------------------------------
# A) 3-class re-evaluation of existing 5-class models + C) Ensemble
# ---------------------------------------------------------------------------

def main():
    df, train, val, test = load_splits()
    y_val_5 = val["rating_bucket"].to_numpy()
    y_te_5 = test["rating_bucket"].to_numpy()
    y_val_3 = to_3class(y_val_5)
    y_te_3 = to_3class(y_te_5)

    # ------ B: sweep ------
    sweep = tfidf_sweep(train, val, test)
    p_tfidf_val = sweep["_proba_val"]
    p_tfidf_test = sweep["_proba_test"]

    # ------ Structured + BERT probas ------
    p_struct_val, p_struct_test = structured_probas(train, val, test)
    p_bert_val, p_bert_test = distilbert_probas(train, val, test)

    # ------ C: soft-vote ensemble ------
    print("\n[C] Ensemble (soft vote: tfidf + struct + bert)")
    p_val = (p_tfidf_val + p_struct_val + p_bert_val) / 3.0
    p_test = (p_tfidf_test + p_struct_test + p_bert_test) / 3.0
    y_val_pred = np.argmax(p_val, axis=1)
    y_te_pred = np.argmax(p_test, axis=1)
    ens_val_5 = evaluate(y_val_5, y_val_pred, 5)
    ens_test_5 = evaluate(y_te_5, y_te_pred, 5)
    ens_val_5["confusion_matrix"] = confusion(y_val_5, y_val_pred, 5)
    ens_test_5["confusion_matrix"] = confusion(y_te_5, y_te_pred, 5)
    ens_val_3 = evaluate(y_val_3, to_3class(y_val_pred), 3)
    ens_test_3 = evaluate(y_te_3, to_3class(y_te_pred), 3)
    ens_val_3["confusion_matrix"] = confusion(y_val_3, to_3class(y_val_pred), 3)
    ens_test_3["confusion_matrix"] = confusion(y_te_3, to_3class(y_te_pred), 3)
    print(f"  ens 5-class test: acc={ens_test_5['accuracy']}  f1={ens_test_5['macro_f1']}")
    print(f"  ens 3-class test: acc={ens_test_3['accuracy']}  f1={ens_test_3['macro_f1']}")

    # ------ A: 3-class report for existing models ------
    # Reuse the per-model test predictions we already have? We don't have stored
    # predictions — easier to rerun with existing configs as in the original scripts.
    # But we DO have probas from the rerun helpers above; use those.
    p_orig_tfidf_val = p_tfidf_val  # NOTE: this is the TUNED tfidf, not original
    # For honest 3-class reporting of the ORIGINAL 5-class models, rerun them:
    print("\n[A] 3-class re-evaluation of existing models")

    def three_class_from_5class_model(y_pred_5_val, y_pred_5_test, name):
        val3 = evaluate(y_val_3, to_3class(y_pred_5_val), 3)
        test3 = evaluate(y_te_3, to_3class(y_pred_5_test), 3)
        val3["confusion_matrix"] = confusion(y_val_3, to_3class(y_pred_5_val), 3)
        test3["confusion_matrix"] = confusion(y_te_3, to_3class(y_pred_5_test), 3)
        print(f"  {name} 3-class test: acc={test3['accuracy']}  f1={test3['macro_f1']}")
        return {"val": val3, "test": test3}

    # original TF-IDF model (config from tfidf_logreg.py)
    vec0 = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=20000,
                           max_df=0.95, stop_words="english",
                           lowercase=True, strip_accents="unicode")
    Xtr0 = vec0.fit_transform(train["comments_text"])
    Xvl0 = vec0.transform(val["comments_text"])
    Xte0 = vec0.transform(test["comments_text"])
    clf0 = LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced",
                              solver="lbfgs", random_state=RANDOM_SEED)
    clf0.fit(Xtr0, train["rating_bucket"].to_numpy())
    tfidf_orig_3 = three_class_from_5class_model(
        clf0.predict(Xvl0), clf0.predict(Xte0), "TF-IDF (orig)")

    # Structured LR — predictions
    p_s_val_pred = np.argmax(p_struct_val, axis=1)
    p_s_te_pred = np.argmax(p_struct_test, axis=1)
    struct_3 = three_class_from_5class_model(p_s_val_pred, p_s_te_pred, "Structured-LR")

    # DistilBERT LR
    p_b_val_pred = np.argmax(p_bert_val, axis=1)
    p_b_te_pred = np.argmax(p_bert_test, axis=1)
    bert_3 = three_class_from_5class_model(p_b_val_pred, p_b_te_pred, "DistilBERT-LR")

    # Summary comparison
    print("\n=== Summary (TEST) ===")
    print(f"{'Model':<30} {'5-acc':>6} {'5-f1':>6} {'3-acc':>6} {'3-f1':>6}")
    print("-" * 62)

    # We need original tfidf 5-class test metrics for the table. Already computed:
    tfidf_orig_5_val = evaluate(y_val_5, clf0.predict(Xvl0), 5)
    tfidf_orig_5_te = evaluate(y_te_5, clf0.predict(Xte0), 5)
    struct_5_te = evaluate(y_te_5, p_s_te_pred, 5)
    bert_5_te = evaluate(y_te_5, p_b_te_pred, 5)

    rows = [
        ("TF-IDF (original)", tfidf_orig_5_te, tfidf_orig_3["test"]),
        ("TF-IDF (tuned)", sweep["results_5class"]["test"], sweep["results_3class"]["test"]),
        ("Structured-LR", struct_5_te, struct_3["test"]),
        ("DistilBERT-LR", bert_5_te, bert_3["test"]),
        ("Ensemble (soft vote)", ens_test_5, ens_test_3),
    ]
    for name, five, three in rows:
        print(f"{name:<30} "
              f"{five['accuracy']:>6.3f} {five['macro_f1']:>6.3f} "
              f"{three['accuracy']:>6.3f} {three['macro_f1']:>6.3f}")

    # Save
    out = {
        "tuned_tfidf": {
            "best_config": sweep["best_config"],
            "results_5class": sweep["results_5class"],
            "results_3class": sweep["results_3class"],
            "all_configs": sweep["all_configs"],
        },
        "three_class_evaluation": {
            "tfidf_original": tfidf_orig_3,
            "structured_lr": struct_3,
            "distilbert_lr": bert_3,
        },
        "ensemble_soft_vote": {
            "components": ["tfidf_tuned", "structured_lr", "distilbert_lr"],
            "results_5class": {"val": ens_val_5, "test": ens_test_5},
            "results_3class": {"val": ens_val_3, "test": ens_test_3},
        },
        "summary_test": [
            {"model": name, "5cls_acc": five["accuracy"], "5cls_f1": five["macro_f1"],
             "3cls_acc": three["accuracy"], "3cls_f1": three["macro_f1"]}
            for name, five, three in rows
        ],
    }
    os.makedirs(os.path.dirname(OUT_JSON) or ".", exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(f"\nSaved: {OUT_JSON}")


if __name__ == "__main__":
    main()
