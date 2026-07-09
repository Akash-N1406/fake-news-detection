"""
Day 5: TF-IDF (from scratch)
Fake News Detection Project

Implements TF-IDF manually -- no TfidfVectorizer. Reuses the same
vocabulary and train/test split from Day 4 (top 5000 training-set words)
so BoW and TF-IDF are directly comparable as feature representations over
the same feature space.

Formulas used (documented explicitly since there are several common variants):

  TF(t, d)  = count(t, d) / total_tokens(d)
              -> normalized term frequency, so document length doesn't
                 dominate (recall from Day 3 EDA: Fake articles run longer
                 on average, so raw counts alone would bias toward Fake)

  IDF(t)    = ln((1 + N) / (1 + df(t))) + 1
              -> "smooth" IDF: the +1 in numerator/denominator avoids
                 division by zero for words absent from a doc, and the
                 trailing +1 ensures a word appearing in every document
                 still gets a small positive weight rather than exactly 0.
                 (This matches the well-established smooth-idf convention;
                 we implement it directly with our own df counts, not
                 sklearn's vectorizer.)

  TFIDF(t,d) = TF(t,d) * IDF(t)

  Row-wise L2 normalization is applied after, so document vectors are
  comparable regardless of length -- standard practice for text
  classification (this is also what makes cosine similarity meaningful
  downstream, e.g. for the KNN model in Week 2).
"""

import os
import json
from collections import Counter

import pandas as pd
import numpy as np
from scipy import sparse

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")


def load_split_tokens():
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    train_df["text_clean"] = train_df["text_clean"].fillna("")
    test_df["text_clean"] = test_df["text_clean"].fillna("")

    train_tokens = train_df["text_clean"].apply(str.split).tolist()
    test_tokens = test_df["text_clean"].apply(str.split).tolist()
    return train_tokens, test_tokens


def load_vocabulary():
    with open(f"{DATA_DIR}/bow_vocabulary.json") as f:
        vocabulary = json.load(f)
    return vocabulary


def compute_idf(token_lists, vocabulary):
    """Document frequency + smooth IDF, computed from the TRAINING set only."""
    n_docs = len(token_lists)
    n_vocab = len(vocabulary)
    doc_freq = np.zeros(n_vocab, dtype=np.int64)

    for tokens in token_lists:
        seen = set(tokens)  # only count each word once per document
        for word in seen:
            col_idx = vocabulary.get(word)
            if col_idx is not None:
                doc_freq[col_idx] += 1

    idf = np.log((1 + n_docs) / (1 + doc_freq)) + 1
    return idf, doc_freq


def compute_tfidf_matrix(token_lists, vocabulary, idf):
    """Build the TF-IDF sparse matrix: normalized TF * IDF, then row-wise L2 norm."""
    n_docs = len(token_lists)
    n_vocab = len(vocabulary)

    matrix = sparse.lil_matrix((n_docs, n_vocab), dtype=np.float64)

    for row_idx, tokens in enumerate(token_lists):
        doc_len = len(tokens)
        if doc_len == 0:
            continue
        counts = Counter(tokens)
        for word, count in counts.items():
            col_idx = vocabulary.get(word)
            if col_idx is not None:
                tf = count / doc_len
                matrix[row_idx, col_idx] = tf * idf[col_idx]

    matrix = matrix.tocsr()

    # Row-wise L2 normalization
    row_norms = np.sqrt(matrix.multiply(matrix).sum(axis=1)).A.flatten()
    row_norms[row_norms == 0] = 1  # avoid div-by-zero for empty rows
    matrix = matrix.multiply(1 / row_norms[:, None]).tocsr()

    return matrix


def main():
    print("Loading train/test split and vocabulary from Day 4...")
    train_tokens, test_tokens = load_split_tokens()
    vocabulary = load_vocabulary()
    print(f"Vocabulary size: {len(vocabulary)}")

    print("\nComputing document frequency + IDF from training set...")
    idf, doc_freq = compute_idf(train_tokens, vocabulary)

    idx_to_word = {v: k for k, v in vocabulary.items()}
    print("\nWords with lowest IDF (appear in almost every document -> least discriminative):")
    lowest_idf_idxs = idf.argsort()[:10]
    for idx in lowest_idf_idxs:
        print(f"  {idx_to_word[idx]}: idf={idf[idx]:.3f}, appears in {doc_freq[idx]}/{len(train_tokens)} docs")

    print("\nWords with highest IDF (rare -> most discriminative when present):")
    highest_idf_idxs = idf.argsort()[::-1][:10]
    for idx in highest_idf_idxs:
        print(f"  {idx_to_word[idx]}: idf={idf[idx]:.3f}, appears in {doc_freq[idx]}/{len(train_tokens)} docs")

    print("\nBuilding TF-IDF matrix for train set...")
    X_train = compute_tfidf_matrix(train_tokens, vocabulary, idf)
    print(f"Train TF-IDF matrix shape: {X_train.shape}, non-zero entries: {X_train.nnz}")

    print("Building TF-IDF matrix for test set (using train-derived IDF)...")
    X_test = compute_tfidf_matrix(test_tokens, vocabulary, idf)
    print(f"Test TF-IDF matrix shape: {X_test.shape}, non-zero entries: {X_test.nnz}")

    sparse.save_npz(f"{DATA_DIR}/tfidf_train.npz", X_train)
    sparse.save_npz(f"{DATA_DIR}/tfidf_test.npz", X_test)
    np.save(f"{DATA_DIR}/idf_values.npy", idf)

    print(f"\nSaved: tfidf_train.npz, tfidf_test.npz, idf_values.npy")

    # Sanity check: top TF-IDF words for the first training doc, vs raw BoW
    print("\n--- Example: first training document, top words by TF-IDF weight ---")
    doc_vec = X_train[0].toarray().flatten()
    top_idxs = doc_vec.argsort()[::-1][:10]
    for idx in top_idxs:
        if doc_vec[idx] > 0:
            print(f"  {idx_to_word[idx]}: tfidf={doc_vec[idx]:.4f}")


if __name__ == "__main__":
    main()