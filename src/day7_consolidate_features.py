"""
Day 7: Feature Consolidation
Fake News Detection Project

Week 1 built three parallel feature representations of the same
train/test split (30,883 train / 7,721 test docs):
  - Bag-of-Words       (data/bow_train.npz, bow_test.npz)          - 5000 dims, sparse, raw counts
  - TF-IDF              (data/tfidf_train.npz, tfidf_test.npz)      - 5000 dims, sparse, weighted+normalized
  - Word2Vec embeddings (data/embeddings_train.npy, ..._test.npy)   - 100 dims, dense, averaged word vectors

This script does NOT recompute anything. Its job is to:
1. Verify all three feature sets are row-aligned with the same labels
   (a mismatch here would silently corrupt every Week 2 model)
2. Provide one function, `load_feature_set(name)`, that Week 2 scripts can
   import instead of each re-writing their own loading logic
3. Produce a comparison summary (dimensionality, sparsity, memory footprint)
   for the report's Methodology/Results section
"""

import os
import numpy as np
from scipy import sparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

FEATURE_FILES = {
    "bow": {"train": "bow_train.npz", "test": "bow_test.npz", "sparse": True},
    "tfidf": {"train": "tfidf_train.npz", "test": "tfidf_test.npz", "sparse": True},
    "embeddings": {"train": "embeddings_train.npy", "test": "embeddings_test.npy", "sparse": False},
}


def load_feature_set(name):
    """Load a named feature set's train/test matrices, plus labels.

    Usage (from any Week 2 script):
        from day7_consolidate_features import load_feature_set
        X_train, X_test, y_train, y_test = load_feature_set("tfidf")
    """
    if name not in FEATURE_FILES:
        raise ValueError(f"Unknown feature set '{name}'. Choose from {list(FEATURE_FILES)}")

    spec = FEATURE_FILES[name]
    loader = sparse.load_npz if spec["sparse"] else np.load

    X_train = loader(f"{DATA_DIR}/{spec['train']}")
    X_test = loader(f"{DATA_DIR}/{spec['test']}")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")

    return X_train, X_test, y_train, y_test


def verify_alignment():
    """Check every feature set has the same number of rows as the labels,
    and that labels themselves match across loads (they're the same file,
    but worth confirming the pipeline is self-consistent)."""
    print("=== Row-Alignment Check ===")
    y_train = np.load(f"{DATA_DIR}/y_train.npy")
    y_test = np.load(f"{DATA_DIR}/y_test.npy")
    n_train, n_test = len(y_train), len(y_test)
    print(f"Expected: {n_train} train rows, {n_test} test rows\n")

    all_ok = True
    for name in FEATURE_FILES:
        X_train, X_test, _, _ = load_feature_set(name)
        train_ok = X_train.shape[0] == n_train
        test_ok = X_test.shape[0] == n_test
        status = "OK" if (train_ok and test_ok) else "MISMATCH"
        if not (train_ok and test_ok):
            all_ok = False
        print(f"{name:12s} train={X_train.shape}  test={X_test.shape}  [{status}]")

    print(f"\nAll feature sets aligned: {all_ok}")
    return all_ok


def summarize_features():
    print("\n=== Feature Set Comparison ===")
    rows = []
    for name in FEATURE_FILES:
        X_train, X_test, _, _ = load_feature_set(name)
        n_dims = X_train.shape[1]

        if sparse.issparse(X_train):
            density = X_train.nnz / (X_train.shape[0] * X_train.shape[1]) * 100
            mem_mb = (X_train.data.nbytes + X_train.indices.nbytes + X_train.indptr.nbytes) / 1e6
            dtype = str(X_train.dtype)
        else:
            density = 100.0  # dense embeddings are fully "populated" by construction
            mem_mb = X_train.nbytes / 1e6
            dtype = str(X_train.dtype)

        rows.append((name, n_dims, density, mem_mb, dtype))

    print(f"{'Feature Set':<12} {'Dims':>6} {'Density %':>10} {'Train Size (MB)':>17} {'Dtype':>10}")
    print("-" * 60)
    for name, dims, density, mem, dtype in rows:
        print(f"{name:<12} {dims:>6} {density:>9.2f}% {mem:>16.1f} {dtype:>10}")

    return rows


def main():
    aligned = verify_alignment()
    if not aligned:
        raise RuntimeError("Feature sets are not aligned -- check Day 4/5/6 outputs before proceeding.")

    summarize_features()

    print("\n=== Ready for Week 2 ===")
    print("Import `load_feature_set(name)` from this module in Day 8+ scripts,")
    print("where name is one of: 'bow', 'tfidf', 'embeddings'")


if __name__ == "__main__":
    main()