"""
Day 4: Train/Test Split + Bag-of-Words (from scratch)
Fake News Detection Project

Two things happen here, in order:

1. Train/test split (80/20, stratified by label). This is done ONCE and
   reused for every feature representation (BoW, TF-IDF, embeddings) and
   every model in Week 2-3, so results are comparable across methods.
   Critically, the split happens BEFORE vocabulary building -- the
   vocabulary is learned only from the training set, so no test-set
   information leaks into the features (a common mistake).

2. Bag-of-Words, implemented manually:
   - Build a vocabulary of the top-N most frequent words in the TRAINING
     set only
   - Represent each document as a vector of raw word counts over that
     vocabulary
   - No CountVectorizer -- vocabulary building and counting are our own code.
     scipy.sparse is used purely as a storage container (a document-term
     matrix at 5000 words x ~30k docs would be >1GB dense; sparse keeps it
     manageable), not as a shortcut for the counting logic itself.
"""

import os
import json
from collections import Counter

import pandas as pd
import numpy as np
from scipy import sparse
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

MAX_FEATURES = 5000
TEST_SIZE = 0.2
RANDOM_STATE = 42


def split_data():
    df = pd.read_csv(f"{DATA_DIR}/combined_cleaned.csv")
    df["text_clean"] = df["text_clean"].fillna("")

    train_df, test_df = train_test_split(
        df, test_size=TEST_SIZE, stratify=df["label"], random_state=RANDOM_STATE
    )

    print(f"Train: {train_df.shape[0]} rows  ({train_df.label.mean()*100:.1f}% fake)")
    print(f"Test:  {test_df.shape[0]} rows  ({test_df.label.mean()*100:.1f}% fake)")

    train_df.to_csv(f"{DATA_DIR}/train.csv", index=False)
    test_df.to_csv(f"{DATA_DIR}/test.csv", index=False)

    return train_df, test_df


def build_vocabulary(token_lists, max_features=MAX_FEATURES):
    """Count word frequency across the training corpus, keep the top-N words.

    token_lists: list of tokenized documents (each a list of word strings)
    Returns: dict mapping word -> column index in the document-term matrix
    """
    freq = Counter()
    for tokens in token_lists:
        freq.update(tokens)

    most_common = freq.most_common(max_features)
    vocabulary = {word: idx for idx, (word, _) in enumerate(most_common)}
    return vocabulary, freq


def vectorize(token_lists, vocabulary):
    """Build a sparse document-term matrix of raw word counts.

    Rows = documents, columns = vocabulary words, values = count of that
    word in that document. Words not in the vocabulary are simply ignored
    (this is expected -- it's how BoW naturally handles unseen test words).
    """
    n_docs = len(token_lists)
    n_vocab = len(vocabulary)

    matrix = sparse.lil_matrix((n_docs, n_vocab), dtype=np.int32)

    for row_idx, tokens in enumerate(token_lists):
        counts = Counter(tokens)
        for word, count in counts.items():
            col_idx = vocabulary.get(word)
            if col_idx is not None:
                matrix[row_idx, col_idx] = count

    return matrix.tocsr()


def main():
    train_df, test_df = split_data()

    train_tokens = train_df["text_clean"].apply(str.split).tolist()
    test_tokens = test_df["text_clean"].apply(str.split).tolist()

    print(f"\nBuilding vocabulary from training set (top {MAX_FEATURES} words)...")
    vocabulary, freq = build_vocabulary(train_tokens, MAX_FEATURES)
    print(f"Vocabulary size: {len(vocabulary)}")
    print(f"Total unique words seen in training set: {len(freq)}")
    print(f"Top 15 words: {[w for w, _ in freq.most_common(15)]}")

    print("\nVectorizing train set...")
    X_train = vectorize(train_tokens, vocabulary)
    print(f"Train BoW matrix shape: {X_train.shape}, non-zero entries: {X_train.nnz}")

    print("Vectorizing test set...")
    X_test = vectorize(test_tokens, vocabulary)
    print(f"Test BoW matrix shape: {X_test.shape}, non-zero entries: {X_test.nnz}")

    # Save everything needed for Week 2 modeling
    sparse.save_npz(f"{DATA_DIR}/bow_train.npz", X_train)
    sparse.save_npz(f"{DATA_DIR}/bow_test.npz", X_test)
    np.save(f"{DATA_DIR}/y_train.npy", train_df["label"].values)
    np.save(f"{DATA_DIR}/y_test.npy", test_df["label"].values)

    with open(f"{DATA_DIR}/bow_vocabulary.json", "w") as f:
        json.dump(vocabulary, f)

    print(f"\nSaved: bow_train.npz, bow_test.npz, y_train.npy, y_test.npy, bow_vocabulary.json")

    # Sanity check: what does one document's BoW vector look like?
    print("\n--- Example: first training document ---")
    doc_vec = X_train[0].toarray().flatten()
    top_word_idxs = doc_vec.argsort()[::-1][:10]
    idx_to_word = {v: k for k, v in vocabulary.items()}
    print("Top words in this document by count:")
    for idx in top_word_idxs:
        if doc_vec[idx] > 0:
            print(f"  {idx_to_word[idx]}: {doc_vec[idx]}")


if __name__ == "__main__":
    main()