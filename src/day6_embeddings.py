"""
Day 6: Word Embeddings (Word2Vec, trained on our own corpus)
Fake News Detection Project

Unlike BoW/TF-IDF (implemented by hand), embeddings are trained here using
gensim's Word2Vec -- implementing skip-gram with negative sampling from
raw backprop is a project in itself and outside this spec's scope. What
IS done manually is everything around it: training only on the training
split (no leakage), turning per-word vectors into a single per-document
vector, and handling documents with no in-vocabulary words.

Why train our own instead of using pretrained vectors (e.g. GloVe):
this sandboxed environment can't reach external hosts like Stanford's GloVe
download, and training our own also means the embeddings are shaped by the
actual vocabulary/style of this news corpus rather than generic web text.

Approach:
1. Train Word2Vec (skip-gram) on the TRAINING set's tokenized documents only
2. Represent each document as the average of its words' vectors
   (a standard, simple baseline for turning word embeddings into document
   features -- more sophisticated pooling exists, but averaging is the
   right complexity level here and is what most course-level treatments use)
3. Documents with zero in-vocabulary words get a zero vector fallback
"""

import os
import pandas as pd
import numpy as np
from gensim.models import Word2Vec

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

VECTOR_SIZE = 100
WINDOW = 5
MIN_COUNT = 5
EPOCHS = 10


def load_split_tokens():
    train_df = pd.read_csv(f"{DATA_DIR}/train.csv")
    test_df = pd.read_csv(f"{DATA_DIR}/test.csv")
    train_df["text_clean"] = train_df["text_clean"].fillna("")
    test_df["text_clean"] = test_df["text_clean"].fillna("")

    train_tokens = train_df["text_clean"].apply(str.split).tolist()
    test_tokens = test_df["text_clean"].apply(str.split).tolist()
    return train_tokens, test_tokens


def train_word2vec(train_tokens):
    model = Word2Vec(
        sentences=train_tokens,
        vector_size=VECTOR_SIZE,
        window=WINDOW,
        min_count=MIN_COUNT,
        sg=1,  # skip-gram (better for smaller/medium corpora than CBOW)
        workers=4,
        epochs=EPOCHS,
        seed=42,
    )
    return model


def document_vector(tokens, model):
    """Average the embedding vectors of all in-vocabulary words in a document."""
    vectors = [model.wv[word] for word in tokens if word in model.wv]
    if not vectors:
        return np.zeros(model.vector_size)
    return np.mean(vectors, axis=0)


def build_document_matrix(token_lists, model):
    return np.array([document_vector(tokens, model) for tokens in token_lists])


def main():
    print("Loading train/test split...")
    train_tokens, test_tokens = load_split_tokens()

    print(f"\nTraining Word2Vec (skip-gram, dim={VECTOR_SIZE}) on training set only...")
    model = train_word2vec(train_tokens)
    print(f"Word2Vec vocabulary size: {len(model.wv)}")

    # Sanity check: does the embedding space capture meaningful semantic
    # relationships? Good to include in the report as evidence embeddings
    # learned something real, not just noise.
    print("\n--- Nearest neighbors sanity check ---")
    for word in ["trump", "government", "president", "fake"]:
        if word in model.wv:
            similar = model.wv.most_similar(word, topn=5)
            print(f"  '{word}' -> {[w for w, _ in similar]}")
        else:
            print(f"  '{word}' not in vocabulary (appeared < {MIN_COUNT} times)")

    print("\nBuilding document vectors (averaged word embeddings)...")
    X_train = build_document_matrix(train_tokens, model)
    X_test = build_document_matrix(test_tokens, model)

    zero_vec_train = (X_train == 0).all(axis=1).sum()
    zero_vec_test = (X_test == 0).all(axis=1).sum()
    print(f"Train matrix shape: {X_train.shape} ({zero_vec_train} zero-vector docs)")
    print(f"Test matrix shape: {X_test.shape} ({zero_vec_test} zero-vector docs)")

    np.save(f"{DATA_DIR}/embeddings_train.npy", X_train)
    np.save(f"{DATA_DIR}/embeddings_test.npy", X_test)
    model.save(f"{DATA_DIR}/word2vec.model")

    print(f"\nSaved: embeddings_train.npy, embeddings_test.npy, word2vec.model")


if __name__ == "__main__":
    main()