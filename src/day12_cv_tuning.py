"""
Day 12: Hyperparameter Tuning via k-Fold Cross-Validation
Fake News Detection Project

Days 8-10 tuned KNN, Logistic Regression, and Random Forest against a
single validation split carved from the training set. That's fast but
noisy -- performance on one particular 80/20 split can be a bit lucky or
unlucky. This script re-validates each model's chosen hyperparameters
using 5-fold StratifiedKFold cross-validation on the TRAINING set only
(the test set stays untouched until Day 14's final consolidated
evaluation), giving a mean +/- std accuracy for each candidate
configuration -- a much more defensible number for the report.

Each model runs on the feature set that won in Days 8-10:
  - KNN                 -> embeddings (curse-of-dimensionality result from Day 8)
  - Logistic Regression -> tfidf (best performer from Day 9)
  - Random Forest       -> tfidf (best performer from Day 10)

The Neural Network is excluded here since its architecture was fixed
rather than grid-searched -- its robustness is checked separately in
Day 13 via repeated runs with different random seeds.
"""

import os
import sys
import time
import json

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from day7_consolidate_features import load_feature_set

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "reports")
N_FOLDS = 5
RANDOM_STATE = 42
SCORING = ["accuracy", "precision", "recall", "f1"]


def run_cv(estimator_fn, param_grid, X_train, y_train, model_name):
    """estimator_fn: function(params) -> sklearn estimator instance"""
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = []

    for params in param_grid:
        t0 = time.time()
        clf = estimator_fn(params)
        cv_results = cross_validate(
            clf, X_train, y_train, cv=skf, scoring=SCORING, n_jobs=-1
        )
        elapsed = time.time() - t0

        summary = {"params": params}
        for metric in SCORING:
            scores = cv_results[f"test_{metric}"]
            summary[f"{metric}_mean"] = float(scores.mean())
            summary[f"{metric}_std"] = float(scores.std())

        print(f"  {params}  acc={summary['accuracy_mean']:.4f}+/-{summary['accuracy_std']:.4f}  "
              f"f1={summary['f1_mean']:.4f}+/-{summary['f1_std']:.4f}  ({elapsed:.1f}s)")
        results.append(summary)

    best = max(results, key=lambda r: r["accuracy_mean"])
    print(f"  -> Best (by CV mean accuracy): {best['params']}")
    return results, best


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    all_results = {}

    print("="*60)
    print("KNN -- embeddings (5-fold CV)")
    print("="*60)
    X_train, _, y_train, _ = load_feature_set("embeddings")
    knn_grid = [{"n_neighbors": k} for k in [3, 5, 7, 9, 11]]
    results, best = run_cv(
        lambda p: KNeighborsClassifier(n_neighbors=p["n_neighbors"], n_jobs=-1),
        knn_grid, X_train, y_train, "knn"
    )
    all_results["knn"] = {"feature_set": "embeddings", "cv_results": results, "best": best}

    print("\n" + "="*60)
    print("Logistic Regression -- tfidf (5-fold CV)")
    print("="*60)
    X_train, _, y_train, _ = load_feature_set("tfidf")
    logreg_grid = [{"C": c} for c in [0.01, 0.1, 1.0, 10.0]]
    results, best = run_cv(
        lambda p: LogisticRegression(C=p["C"], max_iter=1000, random_state=RANDOM_STATE),
        logreg_grid, X_train, y_train, "logreg"
    )
    all_results["logreg"] = {"feature_set": "tfidf", "cv_results": results, "best": best}

    print("\n" + "="*60)
    print("Random Forest -- tfidf (5-fold CV)")
    print("="*60)
    X_train, _, y_train, _ = load_feature_set("tfidf")
    rf_grid = [
        {"n_estimators": 100, "max_depth": None},
        {"n_estimators": 100, "max_depth": 30},
        {"n_estimators": 200, "max_depth": None},
        {"n_estimators": 200, "max_depth": 30},
    ]
    results, best = run_cv(
        lambda p: RandomForestClassifier(
            n_estimators=p["n_estimators"], max_depth=p["max_depth"],
            n_jobs=-1, random_state=RANDOM_STATE
        ),
        rf_grid, X_train, y_train, "rf"
    )
    all_results["rf"] = {"feature_set": "tfidf", "cv_results": results, "best": best}

    with open(f"{RESULTS_DIR}/day12_cv_tuning_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "="*60)
    print("Summary: best hyperparameters per model (via 5-fold CV)")
    print("="*60)
    for model_name, data in all_results.items():
        b = data["best"]
        print(f"  {model_name:10s} ({data['feature_set']:10s}) {b['params']}  "
              f"acc={b['accuracy_mean']:.4f}+/-{b['accuracy_std']:.4f}")

    print(f"\nSaved results to {RESULTS_DIR}/day12_cv_tuning_results.json")


if __name__ == "__main__":
    main()