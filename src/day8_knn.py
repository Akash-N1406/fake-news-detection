"""
Day 8: K-Nearest Neighbors (Non-Parametric Model)
Fake News Detection Project

Uses sklearn's KNeighborsClassifier (matches the brief's provided code
skeleton). Everything AROUND the model -- feature set comparison, k
tuning via a proper validation split, and full evaluation -- is our own
pipeline logic.

Distance metric note: TF-IDF vectors were L2-normalized in Day 5. For unit
vectors, Euclidean distance and cosine distance are related by a monotonic
transform (||a-b||^2 = 2 - 2*cos_sim(a,b) when ||a||=||b||=1), so ranking
of nearest neighbors is IDENTICAL whether we call it "euclidean" or
"cosine" here. We use sklearn's default euclidean metric and get
cosine-like behavior on TF-IDF without extra work. Embeddings are NOT
L2-normalized (they're averaged raw Word2Vec vectors), so euclidean there
is genuinely just euclidean -- worth noting as a methodological detail in
the report.

k selection: tuned via an 80/20 split carved out of the TRAINING set only
(never the test set) to avoid contaminating final evaluation. Once k is
chosen, we retrain on the FULL training set and evaluate once on the
held-out test set.
"""

import os
import sys
import time
import json

import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from day7_consolidate_features import load_feature_set

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "reports")
K_CANDIDATES = [3, 5, 7, 9, 11]
VAL_SIZE = 0.2
RANDOM_STATE = 42


def tune_k(X_train, y_train, feature_name):
    """Find the best k using a validation split carved out of training data."""
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=RANDOM_STATE
    )

    print(f"  Tuning k on validation split ({X_tr.shape[0]} tr / {X_val.shape[0]} val)...")
    best_k, best_acc = None, -1
    for k in K_CANDIDATES:
        t0 = time.time()
        knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
        knn.fit(X_tr, y_tr)
        preds = knn.predict(X_val)
        acc = accuracy_score(y_val, preds)
        elapsed = time.time() - t0
        print(f"    k={k:2d}  val_accuracy={acc:.4f}  ({elapsed:.1f}s)")
        if acc > best_acc:
            best_k, best_acc = k, acc

    print(f"  -> Best k={best_k} (val_accuracy={best_acc:.4f})")
    return best_k


def evaluate_on_test(X_train, y_train, X_test, y_test, k, feature_name):
    print(f"  Training final KNN (k={k}) on full training set...")
    t0 = time.time()
    knn = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    knn.fit(X_train, y_train)
    preds = knn.predict(X_test)
    elapsed = time.time() - t0

    metrics = {
        "feature_set": feature_name,
        "k": k,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "train_predict_time_sec": round(elapsed, 2),
    }

    print(f"  Accuracy:  {metrics['accuracy']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1:        {metrics['f1']:.4f}")
    print(f"  Confusion Matrix (rows=actual, cols=predicted, [0=Real,1=Fake]):")
    print(f"    {metrics['confusion_matrix']}")
    print(f"\n{classification_report(y_test, preds, target_names=['Real', 'Fake'])}")

    return metrics


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []

    # BoW skipped by design: for KNN specifically, raw counts over a 5000-dim
    # vocabulary tend to be dominated by document length (longer docs have
    # larger-magnitude vectors), and TF-IDF already addresses this via
    # normalization -- running both would mostly duplicate the same story.
    # We compare TF-IDF (sparse, high-dim, count-based) against embeddings
    # (dense, low-dim, semantic) as the two meaningfully different cases.
    feature_sets_to_run = ["tfidf", "embeddings"]

    for feature_name in feature_sets_to_run:
        print(f"\n{'='*60}")
        print(f"Feature set: {feature_name}")
        print('='*60)

        X_train, X_test, y_train, y_test = load_feature_set(feature_name)
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        best_k = tune_k(X_train, y_train, feature_name)
        metrics = evaluate_on_test(X_train, y_train, X_test, y_test, best_k, feature_name)
        all_results.append(metrics)

    with open(f"{RESULTS_DIR}/day8_knn_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Summary across feature sets:")
    print('='*60)
    for r in all_results:
        print(f"  {r['feature_set']:12s} k={r['k']:2d}  acc={r['accuracy']:.4f}  f1={r['f1']:.4f}")

    print(f"\nSaved results to {RESULTS_DIR}/day8_knn_results.json")


if __name__ == "__main__":
    main()