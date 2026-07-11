"""
Day 9: Logistic Regression (Parametric Model)
Fake News Detection Project

Uses sklearn's LogisticRegression, matching the brief's skeleton.

Unlike KNN (Day 8), Logistic Regression is cheap to train even at 5000
dimensions -- it's a convex optimization problem, not a per-query distance
computation -- so we run it across all THREE feature sets (BoW, TF-IDF,
embeddings) for a complete comparison. This also sets up the key
parametric-vs-non-parametric contrast for the report: KNN struggled badly
on high-dimensional TF-IDF (curse of dimensionality) while Logistic
Regression is expected to do well there, since linear decision boundaries
tend to work well in high-dimensional sparse text spaces.

Regularization strength C is tuned via a validation split carved from the
TRAINING set only (same discipline as Day 8), then evaluated once on the
held-out test set.
"""

import os
import sys
import time
import json

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from day7_consolidate_features import load_feature_set

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "reports")
C_CANDIDATES = [0.01, 0.1, 1.0, 10.0]
VAL_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 1000


def tune_c(X_train, y_train, feature_name):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=RANDOM_STATE
    )

    print(f"  Tuning C on validation split ({X_tr.shape[0]} tr / {X_val.shape[0]} val)...")
    best_c, best_acc = None, -1
    for c in C_CANDIDATES:
        t0 = time.time()
        clf = LogisticRegression(C=c, max_iter=MAX_ITER, random_state=RANDOM_STATE)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        elapsed = time.time() - t0
        print(f"    C={c:<6} val_accuracy={acc:.4f}  ({elapsed:.1f}s)")
        if acc > best_acc:
            best_c, best_acc = c, acc

    print(f"  -> Best C={best_c} (val_accuracy={best_acc:.4f})")
    return best_c


def evaluate_on_test(X_train, y_train, X_test, y_test, c, feature_name):
    print(f"  Training final Logistic Regression (C={c}) on full training set...")
    t0 = time.time()
    clf = LogisticRegression(C=c, max_iter=MAX_ITER, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    elapsed = time.time() - t0

    metrics = {
        "feature_set": feature_name,
        "C": c,
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

    return metrics, clf


def show_top_features(clf, feature_name):
    """For BoW/TF-IDF, show which words most push toward Fake vs Real --
    a nice interpretability angle Logistic Regression offers that KNN and
    embeddings don't (coefficients map directly back to vocabulary words)."""
    if feature_name == "embeddings":
        return  # embedding dimensions aren't individually interpretable

    import json as json_lib
    vocab_path = os.path.join(SCRIPT_DIR, "..", "data", "bow_vocabulary.json")
    with open(vocab_path) as f:
        vocabulary = json_lib.load(f)
    idx_to_word = {v: k for k, v in vocabulary.items()}

    coefs = clf.coef_.flatten()
    top_fake_idxs = coefs.argsort()[::-1][:10]
    top_real_idxs = coefs.argsort()[:10]

    print(f"\n  Top words pushing toward FAKE: {[idx_to_word[i] for i in top_fake_idxs]}")
    print(f"  Top words pushing toward REAL: {[idx_to_word[i] for i in top_real_idxs]}")


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = []

    feature_sets_to_run = ["bow", "tfidf", "embeddings"]

    for feature_name in feature_sets_to_run:
        print(f"\n{'='*60}")
        print(f"Feature set: {feature_name}")
        print('='*60)

        X_train, X_test, y_train, y_test = load_feature_set(feature_name)
        print(f"Train: {X_train.shape}, Test: {X_test.shape}")

        best_c = tune_c(X_train, y_train, feature_name)
        metrics, clf = evaluate_on_test(X_train, y_train, X_test, y_test, best_c, feature_name)
        show_top_features(clf, feature_name)
        all_results.append(metrics)

    with open(f"{RESULTS_DIR}/day9_logreg_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Summary across feature sets:")
    print('='*60)
    for r in all_results:
        print(f"  {r['feature_set']:12s} C={r['C']:<6} acc={r['accuracy']:.4f}  f1={r['f1']:.4f}")

    print(f"\nSaved results to {RESULTS_DIR}/day9_logreg_results.json")


if __name__ == "__main__":
    main()