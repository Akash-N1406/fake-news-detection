"""
Day 13: Neural Network Robustness via Repeated Seeds
Fake News Detection Project

KNN, Logistic Regression, and Random Forest were validated via k-fold
cross-validation in Day 12. Neural networks don't fit that pattern as
cleanly -- their main source of run-to-run variance isn't which data
they're trained on, it's the RANDOM WEIGHT INITIALIZATION and the
stochasticity of mini-batch gradient descent itself. Two networks trained
on the exact same data can converge to meaningfully different solutions
just from a different random seed.

So instead of k-fold CV, this script trains the Day 11 architecture
(unchanged) N_RUNS times on the SAME train/val split, varying only the
random seed, and reports mean +/- std test accuracy across runs. A small
std here means the ~98% accuracy from Day 11 is a stable property of the
architecture+data, not a lucky initialization. A large std would mean the
single Day 11 run shouldn't be trusted as "the" result.

Feature set: uses whichever feature set won in Day 11 (BoW), read
dynamically from day11_neural_net_results.json.

The TEST set is used here (unlike Day 12's CV, which stayed within
training data) because we're not tuning anything -- we're checking how
stable a FIXED architecture's test performance is across random seeds.
This is a legitimate, distinct use of the test set: repeated evaluation
of one fixed, already-decided model, not model selection.
"""

import os
import sys
import json
import time

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from day7_consolidate_features import load_feature_set
from day11_neural_network import to_dense_float32

import tensorflow as tf
from tensorflow import keras
tf.get_logger().setLevel("ERROR")

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "reports")
N_RUNS = 5
SEEDS = [1, 7, 42, 123, 2024]
MAX_EPOCHS = 30
BATCH_SIZE = 64
PATIENCE = 3
VAL_SIZE = 0.2


def load_best_nn_feature_set():
    with open(f"{RESULTS_DIR}/day11_neural_net_results.json") as f:
        results = json.load(f)
    best = max(results, key=lambda r: r["accuracy"])
    return best["feature_set"]


def build_model(input_dim, seed):
    initializer = keras.initializers.GlorotUniform(seed=seed)
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu", kernel_initializer=initializer),
        keras.layers.Dropout(0.3, seed=seed),
        keras.layers.Dense(64, activation="relu", kernel_initializer=initializer),
        keras.layers.Dropout(0.3, seed=seed),
        keras.layers.Dense(1, activation="sigmoid", kernel_initializer=initializer),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def run_single_seed(X_train, y_train, X_test, y_test, seed):
    tf.random.set_seed(seed)
    np.random.seed(seed)

    from sklearn.model_selection import train_test_split
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=seed
    )

    model = build_model(X_tr.shape[1], seed)
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )

    t0 = time.time()
    history = model.fit(
        X_tr, y_tr, validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS, batch_size=BATCH_SIZE,
        callbacks=[early_stop], verbose=0
    )
    elapsed = time.time() - t0
    n_epochs = len(history.history["loss"])

    probs = model.predict(X_test, verbose=0).flatten()
    preds = (probs >= 0.5).astype(int)

    return {
        "seed": seed,
        "epochs_run": n_epochs,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "time_sec": round(elapsed, 1),
    }


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    feature_set = load_best_nn_feature_set()
    print(f"Using feature set from Day 11's best NN result: {feature_set}")

    X_train, X_test, y_train, y_test = load_feature_set(feature_set)
    X_train = to_dense_float32(X_train)
    X_test = to_dense_float32(X_test)
    print(f"Train: {X_train.shape}, Test: {X_test.shape}\n")

    run_results = []
    for i, seed in enumerate(SEEDS[:N_RUNS], 1):
        print(f"Run {i}/{N_RUNS} (seed={seed})...")
        result = run_single_seed(X_train, y_train, X_test, y_test, seed)
        run_results.append(result)
        print(f"  epochs={result['epochs_run']:2d}  accuracy={result['accuracy']:.4f}  "
              f"f1={result['f1']:.4f}  ({result['time_sec']}s)")

    accs = [r["accuracy"] for r in run_results]
    f1s = [r["f1"] for r in run_results]
    precisions = [r["precision"] for r in run_results]
    recalls = [r["recall"] for r in run_results]

    summary = {
        "feature_set": feature_set,
        "n_runs": N_RUNS,
        "seeds": SEEDS[:N_RUNS],
        "runs": run_results,
        "summary": {
            "accuracy": {"mean": float(np.mean(accs)), "std": float(np.std(accs)),
                         "min": float(np.min(accs)), "max": float(np.max(accs))},
            "precision": {"mean": float(np.mean(precisions)), "std": float(np.std(precisions))},
            "recall": {"mean": float(np.mean(recalls)), "std": float(np.std(recalls))},
            "f1": {"mean": float(np.mean(f1s)), "std": float(np.std(f1s))},
        },
    }

    with open(f"{RESULTS_DIR}/day13_nn_robustness_results.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Neural Network Robustness Summary ({N_RUNS} runs, {feature_set})")
    print('='*60)
    print(f"Accuracy: {summary['summary']['accuracy']['mean']:.4f} +/- {summary['summary']['accuracy']['std']:.4f} "
          f"(range: {summary['summary']['accuracy']['min']:.4f} - {summary['summary']['accuracy']['max']:.4f})")
    print(f"F1:       {summary['summary']['f1']['mean']:.4f} +/- {summary['summary']['f1']['std']:.4f}")

    print(f"\nSaved to {RESULTS_DIR}/day13_nn_robustness_results.json")


if __name__ == "__main__":
    main()