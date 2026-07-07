"""
Day 2: Text Cleaning & Manual Tokenization
Fake News Detection Project

Everything here is written from scratch, per the spec's "from scratch" requirement:
- No nltk.word_tokenize / spacy tokenizers
- No sklearn text preprocessing shortcuts
- Custom stopword list (hand-picked, not imported from a library)

Steps:
1. Lowercase
2. Strip URLs, HTML artifacts, and non-alphabetic characters
3. Tokenize by splitting on whitespace (after cleaning, this is a fair manual
   tokenizer since punctuation has already been stripped)
4. Remove stopwords using our own list
5. Drop tokens that are too short to carry meaning (single letters, etc.)
"""

import os
import re
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")

# Hand-curated stopword list (common English function words that carry
# little discriminative signal for this classification task). Built manually
# rather than imported, per spec.
STOPWORDS = set("""
a an the this that these those
is am are was were be been being
i you he she it we they me him her us them
my your his its our their mine yours hers ours theirs
myself yourself himself herself itself ourselves yourselves themselves
and or but if while although because as until since
of in on at by for with about against between into through
during before after above below to from up down out off over under
again further then once here there when where why how
all any both each few more most other some such
no nor not only own same so than too very
can will just don should now
do does did doing have has had having
will would shall should may might must
who which what whom whose
s t d ll m o re ve y
""".split())

# Extremely short tokens (1-2 chars) after cleaning are almost never
# informative words in English news text.
MIN_TOKEN_LEN = 3


def clean_text(text: str) -> str:
    """Lowercase and strip everything except letters and spaces."""
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)          # URLs
    text = re.sub(r"<.*?>", " ", text)                      # HTML tags
    text = re.sub(r"[^a-z\s]", " ", text)                   # punctuation/digits
    text = re.sub(r"\s+", " ", text).strip()                # collapse whitespace
    return text


def tokenize(clean_str: str) -> list:
    """Manual whitespace tokenizer (safe post-cleaning since punctuation is gone)."""
    return clean_str.split(" ") if clean_str else []


def remove_stopwords(tokens: list) -> list:
    return [t for t in tokens if t not in STOPWORDS and len(t) >= MIN_TOKEN_LEN]


def process_row(text: str) -> list:
    cleaned = clean_text(text)
    tokens = tokenize(cleaned)
    tokens = remove_stopwords(tokens)
    return tokens


def main():
    df = pd.read_csv(f"{DATA_DIR}/combined_raw.csv")
    print(f"Loaded {df.shape[0]} rows")

    print("Cleaning + tokenizing title...")
    df["title_tokens"] = df["title"].astype(str).apply(process_row)

    print("Cleaning + tokenizing text (this takes a bit on ~38k articles)...")
    df["text_tokens"] = df["text"].astype(str).apply(process_row)

    # Store both the token list (as a space-joined string, CSV-friendly) and
    # a fully cleaned string version -- vectorizers in Week 2 will consume
    # the cleaned string directly.
    df["title_clean"] = df["title_tokens"].apply(lambda toks: " ".join(toks))
    df["text_clean"] = df["text_tokens"].apply(lambda toks: " ".join(toks))

    # Basic sanity stats
    df["text_token_count"] = df["text_tokens"].apply(len)
    print("\nToken count stats (article body, after cleaning):")
    print(df["text_token_count"].describe())

    empty_after_cleaning = (df["text_token_count"] == 0).sum()
    print(f"\nArticles with zero tokens after cleaning: {empty_after_cleaning}")
    if empty_after_cleaning > 0:
        df = df[df["text_token_count"] > 0].copy()
        print(f"Dropped -> {df.shape[0]} rows remain")

    # Save the columns we actually need going forward (drop raw token-list
    # columns to keep the CSV lean; title_clean/text_clean already encode them)
    out_cols = ["title", "text", "subject", "date", "label",
                "title_clean", "text_clean", "text_token_count"]
    df[out_cols].to_csv(f"{DATA_DIR}/combined_cleaned.csv", index=False)
    print(f"\nSaved to {DATA_DIR}/combined_cleaned.csv")

    # Show a before/after example
    print("\n--- Example ---")
    print("RAW:", df["text"].iloc[0][:200])
    print("CLEAN:", df["text_clean"].iloc[0][:200])


if __name__ == "__main__":
    main()