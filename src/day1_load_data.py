"""
Day 1: Data Loading & Initial Inspection
Fake News Detection Project

Loads Fake.csv and True.csv, merges them with labels, removes corrupted rows
and exact duplicates, and saves a combined raw dataset for downstream cleaning.

Label convention: 1 = fake, 0 = real (matches Kaggle fake-news competition convention)
"""

import os
import pandas as pd

# Resolve data/ relative to this script's location, not the current working
# directory -- so this works whether you run it from src/ or project root.
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "..", "data")


def load_raw():
    fake = pd.read_csv(f"{RAW_DIR}/Fake.csv")
    true = pd.read_csv(f"{RAW_DIR}/True.csv")
    fake["label"] = 1
    true["label"] = 0
    return fake, true


def drop_corrupted_rows(df):
    """Drop rows where 'title' is actually a URL (row got shifted/corrupted at source)."""
    is_corrupted = df["title"].str.startswith("http", na=False)
    return df[~is_corrupted].copy()


def drop_empty_text(df):
    return df[df["text"].str.strip() != ""].copy()


def drop_exact_duplicates(df):
    return df.drop_duplicates(subset=["title", "text"]).copy()


def main():
    fake, true = load_raw()
    print(f"Raw Fake.csv: {fake.shape}, Raw True.csv: {true.shape}")

    combined = pd.concat([fake, true], ignore_index=True)
    print(f"Combined (before cleaning): {combined.shape}")

    combined = drop_corrupted_rows(combined)
    combined = drop_empty_text(combined)
    combined = drop_exact_duplicates(combined)

    # Shuffle so fake/real rows aren't in two contiguous blocks
    combined = combined.sample(frac=1, random_state=42).reset_index(drop=True)

    print(f"Combined (after cleaning): {combined.shape}")
    print(combined["label"].value_counts(normalize=True))

    # subject/date kept as metadata only -- NOT used as model features.
    # subject perfectly separates fake vs real in this dataset (leakage), so
    # using it as a feature would make the classification task trivial and
    # uninformative. See report Discussion section.
    combined.to_csv(f"{RAW_DIR}/combined_raw.csv", index=False)
    print(f"Saved to {RAW_DIR}/combined_raw.csv")


if __name__ == "__main__":
    main()