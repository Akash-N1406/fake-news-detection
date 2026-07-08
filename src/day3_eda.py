"""
Day 3: Exploratory Data Analysis
Fake News Detection Project

Generates the visuals and summary stats needed for the report's
"Results"/"Dataset Description" sections:
1. Class balance
2. Article length distribution (fake vs real)
3. Title length distribution (fake vs real)
4. Most frequent words, fake vs real (after Day 2 cleaning/stopword removal)
5. Average token counts, fake vs real

All figures are saved to reports/figures/ as PNGs for direct use in the
IEEE report and presentation.
"""

import os
from collections import Counter

import pandas as pd
import matplotlib
matplotlib.use("Agg")  # headless-safe backend
import matplotlib.pyplot as plt
import seaborn as sns

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
FIG_DIR = os.path.join(SCRIPT_DIR, "..", "reports", "figures")

sns.set_style("whitegrid")
LABEL_NAMES = {0: "Real", 1: "Fake"}
PALETTE = {"Real": "#2a9d8f", "Fake": "#e63946"}


def ensure_dirs():
    os.makedirs(FIG_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(f"{DATA_DIR}/combined_cleaned.csv")
    df["label_name"] = df["label"].map(LABEL_NAMES)
    df["title_token_count"] = df["title_clean"].fillna("").apply(lambda s: len(s.split()) if s else 0)
    return df


def plot_class_balance(df):
    counts = df["label_name"].value_counts()
    plt.figure(figsize=(6, 5))
    ax = sns.barplot(x=counts.index, y=counts.values, hue=counts.index, palette=PALETTE, legend=False)
    for i, v in enumerate(counts.values):
        ax.text(i, v + 200, f"{v}\n({v/len(df)*100:.1f}%)", ha="center", fontweight="bold")
    plt.title("Class Balance: Real vs Fake News Articles")
    plt.ylabel("Number of Articles")
    plt.xlabel("")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/01_class_balance.png", dpi=150)
    plt.close()
    print("Saved 01_class_balance.png")
    print(counts)
    print()


def plot_length_distribution(df, col, title, filename, xlabel):
    plt.figure(figsize=(8, 5))
    for label_name in ["Real", "Fake"]:
        subset = df[df["label_name"] == label_name][col]
        sns.kdeplot(subset, label=label_name, fill=True, alpha=0.3, color=PALETTE[label_name])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{filename}", dpi=150)
    plt.close()
    print(f"Saved {filename}")
    print(df.groupby("label_name")[col].describe())
    print()


def get_top_words(series, n=20):
    """Count word frequency across a column of pre-cleaned, space-joined text."""
    counter = Counter()
    for text in series.dropna():
        counter.update(text.split())
    return counter.most_common(n)


def plot_top_words(df, n=20):
    fig, axes = plt.subplots(1, 2, figsize=(14, 7))

    for ax, label_name in zip(axes, ["Real", "Fake"]):
        subset = df[df["label_name"] == label_name]["text_clean"]
        top = get_top_words(subset, n=n)
        words, counts = zip(*top)
        sns.barplot(x=list(counts), y=list(words), ax=ax, color=PALETTE[label_name])
        ax.set_title(f"Top {n} Words — {label_name} News")
        ax.set_xlabel("Frequency")

    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/04_top_words_by_label.png", dpi=150)
    plt.close()
    print("Saved 04_top_words_by_label.png\n")


def print_summary_stats(df):
    print("=== Dataset Summary ===")
    print(f"Total articles: {len(df)}")
    print(f"Real: {(df['label']==0).sum()}  Fake: {(df['label']==1).sum()}")
    print(f"\nAvg article length (tokens): Real={df[df.label==0].text_token_count.mean():.1f}, "
          f"Fake={df[df.label==1].text_token_count.mean():.1f}")
    print(f"Avg title length (tokens): Real={df[df.label==0].title_token_count.mean():.1f}, "
          f"Fake={df[df.label==1].title_token_count.mean():.1f}")
    print()


def main():
    ensure_dirs()
    df = load_data()

    print_summary_stats(df)
    plot_class_balance(df)
    plot_length_distribution(
        df, "text_token_count",
        "Article Body Length Distribution (Real vs Fake)",
        "02_article_length_distribution.png",
        "Token Count (after cleaning)"
    )
    plot_length_distribution(
        df, "title_token_count",
        "Title Length Distribution (Real vs Fake)",
        "03_title_length_distribution.png",
        "Token Count (after cleaning)"
    )
    plot_top_words(df, n=20)

    print("Day 3 EDA complete. Figures saved to reports/figures/")


if __name__ == "__main__":
    main()