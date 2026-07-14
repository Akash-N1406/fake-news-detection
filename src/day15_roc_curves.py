"""
Day 15: ROC Curves & Additional Visualizations
Fake News Detection Project

Days 8-11 saved hard predictions (class labels) but not predicted
probabilities, which ROC/PR curves need. This script retrains each model
ONCE more using its already-determined best config (feature set +
hyperparameters, read dynamically from the Day 8-13 result files -- same
pattern as Day 12/13/14), extracts predicted probabilities on the test
set, and plots:

  1. ROC curves (all 4 models overlaid) + AUC per model
  2. Precision-Recall curves (all 4 models overlaid) + Average Precision
     per model -- more informative than ROC when classes aren't perfectly
     balanced (ours is 55/45, mild but non-trivial) since PR curves focus
     on the positive (Fake) class directly rather than being influenced by
     the large number of true negatives.

Retraining is necessary here (Days 8-11 didn't persist trained model
objects), but cheap since we already know the winning hyperparameters --
no tuning happens in this script, just one fit per model.
"""

import os
import sys
import json

import numpy as np
from scipy import sparse
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from day7_consolidate_features import load_feature_set
from day11_neural_network import build_model, to_dense_float32

import tensorflow as tf
from tensorflow import keras
tf.random.set_seed(42)
tf.get_logger().setLevel("ERROR")

REPORTS_DIR = os.path.join(SCRIPT_DIR, "..", "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")
RANDOM_STATE = 42

MODEL_COLORS = {
    "KNN": "#264653", "Logistic Regression": "#2a9d8f",
    "Random Forest": "#e9c46a", "Neural Network": "#e76f51"
}


def load_json(filename):
    with open(f"{REPORTS_DIR}/{filename}") as f:
        return json.load(f)


def best_single_split(results_list):
    return max(results_list, key=lambda r: r["accuracy"])


def get_knn_probs():
    best = best_single_split(load_json("day8_knn_results.json"))
    feature_set, k = best["feature_set"], best["k"]
    print(f"KNN: feature_set={feature_set}, k={k}")
    X_train, X_test, y_train, y_test = load_feature_set(feature_set)
    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    return probs, y_test


def get_logreg_probs():
    best = best_single_split(load_json("day9_logreg_results.json"))
    feature_set, C = best["feature_set"], best["C"]
    print(f"Logistic Regression: feature_set={feature_set}, C={C}")
    X_train, X_test, y_train, y_test = load_feature_set(feature_set)
    clf = LogisticRegression(C=C, max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    return probs, y_test


def get_rf_probs():
    best = best_single_split(load_json("day10_random_forest_results.json"))
    feature_set, params = best["feature_set"], best["params"]
    print(f"Random Forest: feature_set={feature_set}, params={params}")
    X_train, X_test, y_train, y_test = load_feature_set(feature_set)
    clf = RandomForestClassifier(
        n_estimators=params["n_estimators"], max_depth=params["max_depth"],
        n_jobs=-1, random_state=RANDOM_STATE
    )
    clf.fit(X_train, y_train)
    probs = clf.predict_proba(X_test)[:, 1]
    return probs, y_test


def get_nn_probs():
    best = best_single_split(load_json("day11_neural_net_results.json"))
    feature_set = best["feature_set"]
    print(f"Neural Network: feature_set={feature_set}")
    X_train, X_test, y_train, y_test = load_feature_set(feature_set)
    X_train = to_dense_float32(X_train)
    X_test = to_dense_float32(X_test)

    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, stratify=y_train, random_state=RANDOM_STATE
    )
    model = build_model(X_tr.shape[1])
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=3, restore_best_weights=True
    )
    model.fit(
        X_tr, y_tr, validation_data=(X_val, y_val),
        epochs=30, batch_size=64, callbacks=[early_stop], verbose=0
    )
    probs = model.predict(X_test, verbose=0).flatten()
    return probs, y_test


def plot_roc_curves(model_probs):
    plt.figure(figsize=(8, 7))
    auc_scores = {}

    for model_name, (probs, y_test) in model_probs.items():
        fpr, tpr, _ = roc_curve(y_test, probs)
        roc_auc = auc(fpr, tpr)
        auc_scores[model_name] = roc_auc
        plt.plot(fpr, tpr, label=f"{model_name} (AUC={roc_auc:.4f})",
                 color=MODEL_COLORS[model_name], linewidth=2)

    plt.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves: All Models Compared")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/07_roc_curves.png", dpi=150)
    plt.close()
    print("Saved 07_roc_curves.png")
    return auc_scores


def plot_pr_curves(model_probs):
    plt.figure(figsize=(8, 7))
    ap_scores = {}

    for model_name, (probs, y_test) in model_probs.items():
        precision, recall, _ = precision_recall_curve(y_test, probs)
        ap = average_precision_score(y_test, probs)
        ap_scores[model_name] = ap
        plt.plot(recall, precision, label=f"{model_name} (AP={ap:.4f})",
                 color=MODEL_COLORS[model_name], linewidth=2)

    baseline = y_test.mean()  # fraction of positive (Fake) class
    plt.axhline(baseline, linestyle="--", color="gray", label=f"Random Classifier (AP={baseline:.2f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves: All Models Compared")
    plt.legend(loc="lower left")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/08_precision_recall_curves.png", dpi=150)
    plt.close()
    print("Saved 08_precision_recall_curves.png")
    return ap_scores


def main():
    os.makedirs(FIG_DIR, exist_ok=True)
    model_probs = {}

    print("Retraining each model with its best known config to get probabilities...\n")
    model_probs["KNN"] = get_knn_probs()
    model_probs["Logistic Regression"] = get_logreg_probs()
    model_probs["Random Forest"] = get_rf_probs()
    model_probs["Neural Network"] = get_nn_probs()

    print("\nGenerating ROC curves...")
    auc_scores = plot_roc_curves(model_probs)

    print("Generating Precision-Recall curves...")
    ap_scores = plot_pr_curves(model_probs)

    summary = {
        model: {"roc_auc": auc_scores[model], "average_precision": ap_scores[model]}
        for model in model_probs
    }
    with open(f"{REPORTS_DIR}/day15_roc_pr_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print("ROC AUC / Average Precision Summary")
    print('='*60)
    for model, scores in sorted(summary.items(), key=lambda x: x[1]["roc_auc"], reverse=True):
        print(f"  {model:<22} AUC={scores['roc_auc']:.4f}  AP={scores['average_precision']:.4f}")

    print(f"\nSaved to {REPORTS_DIR}/day15_roc_pr_summary.json")


if __name__ == "__main__":
    main()