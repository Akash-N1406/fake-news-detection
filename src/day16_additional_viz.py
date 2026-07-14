"""
Day 16: Additional Visualizations
Fake News Detection Project

Rounds out the evaluation section with three more visuals, using ALL
combinations from Days 8-11 (not just each model's single best config):

  1. Model x Feature-Set accuracy HEATMAP -- Days 8-11 each tried multiple
     feature sets per model, but the comparisons so far (Day 14) only
     showed each model's best. This heatmap shows the FULL picture: e.g.
     KNN's dramatic swing between embeddings (94.6%) and TF-IDF (82.2%)
     is invisible in a "best only" table but is one of the most
     interesting findings in the whole project.

  2. Grouped Precision/Recall/F1 bar chart -- one bar group per model
     (using each model's best config), so metric tradeoffs are visible
     side by side rather than buried in separate confusion matrices.

  3. Training time comparison -- a practical, often-overlooked axis: a
     model with near-identical accuracy but 100x faster training is a
     meaningful finding for a Discussion section on real-world tradeoffs.
"""

import os
import json

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPORTS_DIR = os.path.join(SCRIPT_DIR, "..", "reports")
FIG_DIR = os.path.join(REPORTS_DIR, "figures")

sns.set_style("whitegrid")
MODEL_COLORS = {
    "KNN": "#264653", "Logistic Regression": "#2a9d8f",
    "Random Forest": "#e9c46a", "Neural Network": "#e76f51"
}
MODEL_FILES = {
    "KNN": "day8_knn_results.json",
    "Logistic Regression": "day9_logreg_results.json",
    "Random Forest": "day10_random_forest_results.json",
    "Neural Network": "day11_neural_net_results.json",
}


def load_json(filename):
    with open(f"{REPORTS_DIR}/{filename}") as f:
        return json.load(f)


def get_time_field(entry):
    """NN results use 'train_time_sec', others use 'train_predict_time_sec'."""
    return entry.get("train_predict_time_sec", entry.get("train_time_sec"))


def build_full_results_table():
    rows = []
    for model_name, filename in MODEL_FILES.items():
        for entry in load_json(filename):
            rows.append({
                "Model": model_name,
                "Feature Set": entry["feature_set"],
                "Accuracy": entry["accuracy"],
                "Precision": entry["precision"],
                "Recall": entry["recall"],
                "F1": entry["f1"],
                "Time (s)": get_time_field(entry),
            })
    return pd.DataFrame(rows)


def plot_heatmap(df):
    pivot = df.pivot(index="Model", columns="Feature Set", values="Accuracy") * 100
    col_order = [c for c in ["bow", "tfidf", "embeddings"] if c in pivot.columns]
    pivot = pivot[col_order]
    pivot = pivot.reindex(["KNN", "Logistic Regression", "Random Forest", "Neural Network"])

    plt.figure(figsize=(8, 5))
    sns.heatmap(pivot, annot=True, fmt=".2f", cmap="RdYlGn", vmin=80, vmax=100,
                cbar_kws={"label": "Accuracy (%)"}, linewidths=0.5)
    plt.title("Accuracy by Model x Feature Set (Full Comparison)")
    plt.ylabel("")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/09_model_feature_heatmap.png", dpi=150)
    plt.close()
    print("Saved 09_model_feature_heatmap.png")
    print(pivot.round(2))


def plot_metric_comparison(df):
    """Grouped bar chart using each model's best (highest-accuracy) config."""
    best_rows = df.loc[df.groupby("Model")["Accuracy"].idxmax()]
    best_rows = best_rows.set_index("Model").reindex(
        ["KNN", "Logistic Regression", "Random Forest", "Neural Network"]
    )

    metrics = ["Precision", "Recall", "F1"]
    x = np.arange(len(best_rows))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, metric in enumerate(metrics):
        values = best_rows[metric] * 100
        ax.bar(x + (i - 1) * width, values, width, label=metric)

    ax.set_xticks(x)
    ax.set_xticklabels(best_rows.index)
    ax.set_ylabel("Score (%)")
    ax.set_ylim(85, 100)
    ax.set_title("Precision / Recall / F1 by Model (Best Configuration)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/10_metric_comparison.png", dpi=150)
    plt.close()
    print("\nSaved 10_metric_comparison.png")
    print(best_rows[["Feature Set"] + metrics].round(4))


def plot_training_time(df):
    """Training time for each model's best config -- log scale since KNN/LogReg
    are seconds while Random Forest/NN can be 10-100x slower."""
    best_rows = df.loc[df.groupby("Model")["Accuracy"].idxmax()]
    best_rows = best_rows.set_index("Model").reindex(
        ["KNN", "Logistic Regression", "Random Forest", "Neural Network"]
    )

    plt.figure(figsize=(8, 5))
    colors = [MODEL_COLORS[m] for m in best_rows.index]
    bars = plt.bar(best_rows.index, best_rows["Time (s)"], color=colors)
    plt.yscale("log")
    plt.ylabel("Training + Prediction Time (seconds, log scale)")
    plt.title("Training Time by Model (Best Configuration)")
    for bar, val in zip(bars, best_rows["Time (s)"]):
        plt.text(bar.get_x() + bar.get_width()/2, val, f"{val:.1f}s",
                  ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/11_training_time_comparison.png", dpi=150)
    plt.close()
    print("\nSaved 11_training_time_comparison.png")
    print(best_rows[["Feature Set", "Time (s)"]])


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Loading all model x feature-set combinations from Days 8-11...\n")
    df = build_full_results_table()

    print("="*60)
    print("Full results table (all combinations)")
    print("="*60)
    print(df.round(4).to_string(index=False))

    print("\n" + "="*60)
    print("Heatmap: Accuracy by Model x Feature Set")
    print("="*60)
    plot_heatmap(df)

    print("\n" + "="*60)
    print("Metric Comparison (best config per model)")
    print("="*60)
    plot_metric_comparison(df)

    print("\n" + "="*60)
    print("Training Time Comparison (best config per model)")
    print("="*60)
    plot_training_time(df)

    df.to_csv(f"{REPORTS_DIR}/day16_full_results_table.csv", index=False)
    print(f"\nSaved full results table to {REPORTS_DIR}/day16_full_results_table.csv")


if __name__ == "__main__":
    main()