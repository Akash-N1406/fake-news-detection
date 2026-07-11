"""
Day 10: Random Forest (Ensemble Model)
Fake News Detection Project

Uses sklearn's RandomForestClassifier, matching the brief's skeleton.

Tuning grid is kept modest (n_estimators x max_depth, 4 combos) since tree
ensembles are far more expensive to fit than Logistic Regression --
each tree does repeated feature-subset evaluation at every split, which
adds up fast at 5000 dimensions x 30,883 rows. Runs all three feature sets,
same tuning discipline as Day 8/9 (validation split carved from training
data only, final evaluation once on held-out test set).

Bonus: Random Forest gives us feature_importances_ -- a second, independent
lens on "which words matter" alongside Day 9's Logistic Regression
coefficients. Comparing the two is a good Discussion-section point: do a
linear model and a tree ensemble agree on what's discriminative?
"""

import os
import sys
import time
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from day7_consolidate_features import load_feature_set

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "reports")
PARAM_GRID = [
    {"n_estimators": 100, "max_depth": None},
    {"n_estimators": 100, "max_depth": 30},
    {"n_estimators": 200, "max_depth": None},
    {"n_estimators": 200, "max_depth": 30},
]
VAL_SIZE = 0.2
RANDOM_STATE = 42


def tune_params(X_train, y_train, feature_name):
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=RANDOM_STATE
    )

    print(f"  Tuning on validation split ({X_tr.shape[0]} tr / {X_val.shape[0]} val)...")
    best_params, best_acc = None, -1
    for params in PARAM_GRID:
        t0 = time.time()
        clf = RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_val)
        acc = accuracy_score(y_val, preds)
        elapsed = time.time() - t0
        print(f"    n_estimators={params['n_estimators']:<4} max_depth={str(params['max_depth']):<5} "
              f"val_accuracy={acc:.4f}  ({elapsed:.1f}s)")
        if acc > best_acc:
            best_params, best_acc = params, acc

    print(f"  -> Best params={best_params} (val_accuracy={best_acc:.4f})")
    return best_params


def evaluate_on_test(X_train, y_train, X_test, y_test, params, feature_name):
    print(f"  Training final Random Forest ({params}) on full training set...")
    t0 = time.time()
    clf = RandomForestClassifier(
        n_estimators=params["n_estimators"],
        max_depth=params["max_depth"],
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    elapsed = time.time() - t0

    metrics = {
        "feature_set": feature_name,
        "params": params,
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


def show_feature_importance(clf, feature_name):
    if feature_name == "embeddings":
        return  # embedding dims aren't individually interpretable

    import json as json_lib
    vocab_path = os.path.join(SCRIPT_DIR, "..", "data", "bow_vocabulary.json")
    with open(vocab_path) as f:
        vocabulary = json_lib.load(f)
    idx_to_word = {v: k for k, v in vocabulary.items()}

    importances = clf.feature_importances_
    top_idxs = importances.argsort()[::-1][:15]
    print(f"\n  Top 15 most important words (Random Forest, {feature_name}):")
    for idx in top_idxs:
        print(f"    {idx_to_word[idx]}: {importances[idx]:.4f}")


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

        best_params = tune_params(X_train, y_train, feature_name)
        metrics, clf = evaluate_on_test(X_train, y_train, X_test, y_test, best_params, feature_name)
        show_feature_importance(clf, feature_name)
        all_results.append(metrics)

    with open(f"{RESULTS_DIR}/day10_random_forest_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Summary across feature sets:")
    print('='*60)
    for r in all_results:
        print(f"  {r['feature_set']:12s} params={r['params']}  acc={r['accuracy']:.4f}  f1={r['f1']:.4f}")

    print(f"\nSaved results to {RESULTS_DIR}/day10_random_forest_results.json")


if __name__ == "__main__":
    main()