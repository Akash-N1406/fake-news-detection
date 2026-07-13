"""
Day 14: Consolidate Results
Fake News Detection Project

Pulls together everything from Days 8-13 into the final artifacts the
report's Results section needs:
  1. One summary table: model, best feature set, single-split test metrics
     (Days 8-11), and the robustness check (Day 12 CV for KNN/LogReg/RF,
     Day 13 repeated-seeds for the Neural Network)
  2. A grouped bar chart comparing accuracy across all 4 models, with error
     bars from the robustness check
  3. A 2x2 grid of confusion matrices, one per model, for visual comparison
  4. CSV + Markdown versions of the table for direct use in the IEEE report

This script only reads existing results (no retraining), so it's fast.
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


def load_json(filename):
    with open(f"{REPORTS_DIR}/{filename}") as f:
        return json.load(f)


def best_single_split(results_list):
    return max(results_list, key=lambda r: r["accuracy"])


def build_summary_table():
    knn_best = best_single_split(load_json("day8_knn_results.json"))
    logreg_best = best_single_split(load_json("day9_logreg_results.json"))
    rf_best = best_single_split(load_json("day10_random_forest_results.json"))
    nn_best = best_single_split(load_json("day11_neural_net_results.json"))

    cv_tuning = load_json("day12_cv_tuning_results.json")
    nn_robustness = load_json("day13_nn_robustness_results.json")

    rows = [
        {
            "Model": "KNN",
            "Type": "Non-Parametric",
            "Feature Set": knn_best["feature_set"],
            "Test Accuracy": knn_best["accuracy"],
            "Test Precision": knn_best["precision"],
            "Test Recall": knn_best["recall"],
            "Test F1": knn_best["f1"],
            "CV/Robustness Acc Mean": cv_tuning["knn"]["best"]["accuracy_mean"],
            "CV/Robustness Acc Std": cv_tuning["knn"]["best"]["accuracy_std"],
            "Confusion Matrix": knn_best["confusion_matrix"],
        },
        {
            "Model": "Logistic Regression",
            "Type": "Parametric",
            "Feature Set": logreg_best["feature_set"],
            "Test Accuracy": logreg_best["accuracy"],
            "Test Precision": logreg_best["precision"],
            "Test Recall": logreg_best["recall"],
            "Test F1": logreg_best["f1"],
            "CV/Robustness Acc Mean": cv_tuning["logreg"]["best"]["accuracy_mean"],
            "CV/Robustness Acc Std": cv_tuning["logreg"]["best"]["accuracy_std"],
            "Confusion Matrix": logreg_best["confusion_matrix"],
        },
        {
            "Model": "Random Forest",
            "Type": "Ensemble",
            "Feature Set": rf_best["feature_set"],
            "Test Accuracy": rf_best["accuracy"],
            "Test Precision": rf_best["precision"],
            "Test Recall": rf_best["recall"],
            "Test F1": rf_best["f1"],
            "CV/Robustness Acc Mean": cv_tuning["rf"]["best"]["accuracy_mean"],
            "CV/Robustness Acc Std": cv_tuning["rf"]["best"]["accuracy_std"],
            "Confusion Matrix": rf_best["confusion_matrix"],
        },
        {
            "Model": "Neural Network",
            "Type": "Deep Learning",
            "Feature Set": nn_best["feature_set"],
            "Test Accuracy": nn_best["accuracy"],
            "Test Precision": nn_best["precision"],
            "Test Recall": nn_best["recall"],
            "Test F1": nn_best["f1"],
            "CV/Robustness Acc Mean": nn_robustness["summary"]["accuracy"]["mean"],
            "CV/Robustness Acc Std": nn_robustness["summary"]["accuracy"]["std"],
            "Confusion Matrix": nn_best["confusion_matrix"],
        },
    ]
    return rows


def save_tables(rows):
    df = pd.DataFrame(rows).drop(columns=["Confusion Matrix"])
    numeric_cols = ["Test Accuracy", "Test Precision", "Test Recall", "Test F1",
                     "CV/Robustness Acc Mean", "CV/Robustness Acc Std"]
    for col in numeric_cols:
        df[col] = (df[col] * 100).round(2)

    df.to_csv(f"{REPORTS_DIR}/final_model_comparison.csv", index=False)

    with open(f"{REPORTS_DIR}/final_model_comparison.md", "w") as f:
        f.write("# Final Model Comparison\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n*All values in %. CV/Robustness column: k-fold CV for KNN/LogReg/RF, "
                "repeated-seed runs for Neural Network.*\n")

    print(df.to_string(index=False))
    return df


def plot_accuracy_comparison(rows):
    models = [r["Model"] for r in rows]
    test_acc = [r["Test Accuracy"] * 100 for r in rows]
    cv_mean = [r["CV/Robustness Acc Mean"] * 100 for r in rows]
    cv_std = [r["CV/Robustness Acc Std"] * 100 for r in rows]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(x - width/2, test_acc, width, label="Single Test-Split Accuracy",
           color=[MODEL_COLORS[m] for m in models], alpha=0.6)
    ax.bar(x + width/2, cv_mean, width, yerr=cv_std, capsize=5,
           label="CV / Robustness Mean Accuracy (+/- std)",
           color=[MODEL_COLORS[m] for m in models])

    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Model Comparison: Test Accuracy vs Robustness-Validated Accuracy")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(85, 100)
    ax.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/05_final_model_comparison.png", dpi=150)
    plt.close()
    print(f"\nSaved 05_final_model_comparison.png")


def plot_confusion_matrices(rows):
    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes = axes.flatten()

    for ax, row in zip(axes, rows):
        cm = np.array(row["Confusion Matrix"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Real", "Fake"], yticklabels=["Real", "Fake"],
                    cbar=False)
        ax.set_title(f"{row['Model']} ({row['Feature Set']})")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

    plt.suptitle("Confusion Matrices: Best Configuration per Model", fontsize=14, y=1.00)
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/06_confusion_matrices_grid.png", dpi=150)
    plt.close()
    print(f"Saved 06_confusion_matrices_grid.png")


def print_leaderboard(rows):
    print(f"\n{'='*70}")
    print("FINAL LEADERBOARD (ranked by test accuracy)")
    print('='*70)
    ranked = sorted(rows, key=lambda r: r["Test Accuracy"], reverse=True)
    for i, r in enumerate(ranked, 1):
        print(f"  {i}. {r['Model']:<22} ({r['Type']:<15}) "
              f"acc={r['Test Accuracy']*100:.2f}%  f1={r['Test F1']*100:.2f}%  "
              f"[{r['Feature Set']}]")


def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Building consolidated summary table from Days 8-13...\n")
    rows = build_summary_table()

    print("="*70)
    print("FINAL MODEL COMPARISON TABLE")
    print("="*70)
    save_tables(rows)

    plot_accuracy_comparison(rows)
    plot_confusion_matrices(rows)
    print_leaderboard(rows)

    print(f"\nSaved: final_model_comparison.csv, final_model_comparison.md")
    print("Week 2 (modeling) fully consolidated. Ready for Week 3 (report + presentation).")


if __name__ == "__main__":
    main()