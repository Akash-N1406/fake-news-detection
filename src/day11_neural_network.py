"""
Day 11: Simple Neural Network (Deep Learning)
Fake News Detection Project

Uses TensorFlow/Keras, matching the brief's "Simple Neural Network"
requirement and the framework choice made back in the phishing-detection
project (kept consistent across both internship projects).

Architecture: a small feedforward (Dense) network -- Input -> Dense(128,
relu) -> Dropout(0.3) -> Dense(64, relu) -> Dropout(0.3) -> Dense(1,
sigmoid). This is deliberately simple ("Simple Neural Network" per spec,
not a deep/complex architecture), with dropout for regularization since
5000-dim sparse inputs can overfit fast with only 30k training examples.

Feature sets: BoW/TF-IDF matrices are converted from sparse to dense
float32 arrays -- Keras Dense layers expect dense input, and at 30,883 x
5000 this is ~600MB per matrix, which is manageable but the main reason
training here is slower than the sklearn models. Embeddings are already
dense at 100 dims, so that run is much faster.

Early stopping (on validation loss) prevents wasting epochs once the
model stops improving, and also acts as a form of regularization.
"""

import os
import sys
import time
import json

import numpy as np
from scipy import sparse
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)

tf.random.set_seed(42)
tf.get_logger().setLevel("ERROR")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)
from day7_consolidate_features import load_feature_set

RESULTS_DIR = os.path.join(SCRIPT_DIR, "..", "reports")
VAL_SIZE = 0.2
RANDOM_STATE = 42
MAX_EPOCHS = 30
BATCH_SIZE = 64
PATIENCE = 3


def to_dense_float32(X):
    if sparse.issparse(X):
        return X.toarray().astype(np.float32)
    return X.astype(np.float32)


def build_model(input_dim):
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(128, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(64, activation="relu"),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


def train_and_evaluate(X_train, y_train, X_test, y_test, feature_name):
    print(f"  Converting to dense float32...")
    X_train_d = to_dense_float32(X_train)
    X_test_d = to_dense_float32(X_test)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train_d, y_train, test_size=VAL_SIZE, stratify=y_train, random_state=RANDOM_STATE
    )
    print(f"  Train: {X_tr.shape}, Val: {X_val.shape}, Test: {X_test_d.shape}")

    model = build_model(X_tr.shape[1])
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=PATIENCE, restore_best_weights=True
    )

    print(f"  Training (max {MAX_EPOCHS} epochs, early stopping patience={PATIENCE})...")
    t0 = time.time()
    history = model.fit(
        X_tr, y_tr,
        validation_data=(X_val, y_val),
        epochs=MAX_EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[early_stop],
        verbose=2,
    )
    elapsed = time.time() - t0
    n_epochs_run = len(history.history["loss"])
    print(f"  Stopped after {n_epochs_run} epochs ({elapsed:.1f}s)")

    probs = model.predict(X_test_d, verbose=0).flatten()
    preds = (probs >= 0.5).astype(int)

    metrics = {
        "feature_set": feature_name,
        "epochs_run": n_epochs_run,
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1": f1_score(y_test, preds),
        "confusion_matrix": confusion_matrix(y_test, preds).tolist(),
        "train_time_sec": round(elapsed, 2),
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

    feature_sets_to_run = ["bow", "tfidf", "embeddings"]

    for feature_name in feature_sets_to_run:
        print(f"\n{'='*60}")
        print(f"Feature set: {feature_name}")
        print('='*60)

        X_train, X_test, y_train, y_test = load_feature_set(feature_name)
        metrics = train_and_evaluate(X_train, y_train, X_test, y_test, feature_name)
        all_results.append(metrics)

    with open(f"{RESULTS_DIR}/day11_neural_net_results.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\n{'='*60}")
    print("Summary across feature sets:")
    print('='*60)
    for r in all_results:
        print(f"  {r['feature_set']:12s} epochs={r['epochs_run']:2d}  acc={r['accuracy']:.4f}  f1={r['f1']:.4f}")

    print(f"\nSaved results to {RESULTS_DIR}/day11_neural_net_results.json")


if __name__ == "__main__":
    main()