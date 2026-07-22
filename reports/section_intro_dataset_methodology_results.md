# 1. Introduction

## 1.1 Problem Statement

The proliferation of online news has been accompanied by a corresponding
rise in deliberately fabricated or misleading articles -- commonly termed
"fake news" -- that can influence public opinion, election outcomes, and
individual decision-making at scale. Manual fact-checking cannot keep pace
with the volume of content published daily, motivating automated approaches
to flag likely-fake articles for review.

This project builds a machine learning pipeline, implemented from first
principles rather than relying on pre-built end-to-end solutions, to
classify news articles as **real** or **fake** based on their textual
content. The pipeline covers data acquisition and cleaning, manual text
preprocessing and tokenization, three distinct feature engineering
approaches (Bag-of-Words, TF-IDF, and Word2Vec embeddings), and four
classification algorithms spanning non-parametric, parametric, ensemble,
and deep learning paradigms.

## 1.2 Importance of Fake News Detection

Automated fake news detection matters for three practical reasons: (1)
**scale** -- the volume of published content vastly exceeds human
fact-checking capacity; (2) **speed** -- misinformation often spreads
fastest in the hours immediately after publication, before manual
fact-checks can be completed; and (3) **triage** -- even an imperfect
classifier is useful as a first-pass filter that directs limited human
fact-checking resources toward the content most likely to need it, rather
than replacing human judgment entirely.

## 1.3 Project Scope and Approach

Per the assignment brief, this project deliberately avoids black-box,
pre-built pipelines. Text cleaning and tokenization were implemented as
custom functions (not `nltk.word_tokenize` or similar library calls);
Bag-of-Words and TF-IDF were implemented directly from their underlying
mathematical definitions rather than via `CountVectorizer` /
`TfidfVectorizer`; and all four classification algorithms were evaluated,
tuned, and cross-validated using a consistent, leakage-aware methodology
described in Section 3.

# 2. Dataset Description

## 2.1 Source

This project uses the **Fake and Real News Dataset** (Kaggle, published by
user clmentbisaillon), comprising two CSV files:

- `Fake.csv` -- 23,481 articles labeled as fake news
- `True.csv` -- 21,417 articles labeled as real news

Each record contains four fields: `title`, `text`, `subject`, and `date`.

## 2.2 Data Cleaning

Raw data required several cleaning steps before use:

- **10 corrupted rows** removed, where the `title` field contained a
  malformed URL rather than an article title (a source-data artifact).
- **631 rows with empty article text** removed.
- **5,793 exact duplicate articles** (identical title and text) removed to
  prevent train/test leakage, where the same article could otherwise
  appear in both splits and artificially inflate evaluation metrics.

After cleaning, **38,653 articles** remained (54.8% real, 45.2% fake),
subsequently reduced to **38,604** after Section 2.4's dateline-related
cleaning removed a further 49 articles that became empty once a
non-content prefix was stripped.

## 2.3 Feature Leakage Investigation

Two significant feature leakage issues were identified and addressed
during exploratory data analysis, both illustrating that dataset metadata
can trivially "solve" a classification task without reflecting genuine
content understanding:

**Subject category.** The `subject` field perfectly separates the two
classes in this dataset: Fake articles are exclusively labeled `News`,
`politics`, `left-news`, `Government News`, `US_News`, or `Middle-east`,
while Real articles are exclusively labeled `politicsNews` or `worldnews`,
with zero overlap. A model using `subject` as a feature would classify
articles with near-perfect accuracy without learning anything about
fake-versus-real language. `subject` was therefore excluded from all
feature representations and used only as descriptive metadata.

**Reuters wire-service dateline.** During EDA, word-frequency analysis
revealed that 99.8% of Real articles (21,159 of 21,196) contained the
word "Reuters," typically as a dateline prefix (e.g., `WASHINGTON
(Reuters) - `), compared to only 1.3% of Fake articles. Left unaddressed,
a model could reach near-99% accuracy simply by detecting this token,
rather than learning distinguishing content patterns. A regex-based
dateline stripper was implemented to remove this prefix from article text
(covering 98.5% of affected articles), with "reuters" additionally added
to the stopword list to neutralize residual mid-article mentions.

## 2.4 Class Balance

The final cleaned dataset (38,604 articles) is reasonably balanced: **54.9%
real, 45.1% fake** -- see Figure 1 (`01_class_balance.png`). This mild
imbalance was accounted for via stratified train/test splitting and
stratified k-fold cross-validation throughout, ensuring the class ratio
was preserved in every train/validation/test partition.

## 2.5 Article and Title Length

Fake articles run slightly longer on average (232.5 tokens vs. 220.1 for
Real, post-cleaning) and have substantially longer titles (10.1 tokens vs.
7.7 for Real) -- see Figures 2-3. Longer, more elaborate headlines are
consistent with the sensationalized framing often associated with fake
news content.

# 3. Methodology

## 3.1 Preprocessing and Tokenization

All text cleaning and tokenization was implemented manually:

1. **Lowercasing** and removal of URLs, HTML artifacts, and
   non-alphabetic characters via regular expressions.
2. **Whitespace tokenization**, valid here since punctuation was already
   stripped in step 1.
3. **Stopword removal** using a hand-curated list of ~150 common English
   function words (articles, pronouns, auxiliary verbs, conjunctions),
   plus the corpus-specific addition of "reuters" (Section 2.3).
4. **Minimum token length filtering** (tokens under 3 characters dropped).

## 3.2 Train/Test Split

An 80/20 stratified split (30,883 train / 7,721 test articles, `random_state=42`)
was performed **before** any vocabulary or feature construction, ensuring no
test-set information could leak into feature engineering -- a common
methodological error when vocabulary is built from the full dataset prior
to splitting.

## 3.3 Feature Engineering

Three feature representations were built, all implemented from their
underlying definitions rather than via library vectorizers:

**Bag-of-Words (BoW).** A 5,000-word vocabulary was built from the
training set's most frequent words. Each document was represented as a
sparse vector of raw word counts over this vocabulary.

**TF-IDF.** Using the same 5,000-word vocabulary, term frequency was
computed as `count(word, doc) / total_tokens(doc)` (length-normalized),
and inverse document frequency as the smoothed formula
`ln((1+N)/(1+df(word))) + 1`, with document frequency computed from the
training set only. The final TF-IDF vectors were L2-normalized row-wise.

**Word2Vec Embeddings.** A skip-gram Word2Vec model (100 dimensions,
window size 5, minimum word count 5) was trained on the training corpus
using `gensim`. Document-level vectors were computed as the mean of their
constituent words' embeddings.

## 3.4 Models

Four classification algorithms were implemented via `scikit-learn` (and
`TensorFlow/Keras` for the neural network), consistent with the assignment's
provided code skeleton:

- **K-Nearest Neighbors (KNN)** -- non-parametric, evaluated across k = 3,
  5, 7, 9, 11.
- **Logistic Regression** -- parametric, regularization strength C tuned
  across 0.01, 0.1, 1.0, 10.0.
- **Random Forest** -- ensemble, tuned across n_estimators (100, 200) and
  max_depth (None, 30).
- **Neural Network** -- deep learning, a feedforward architecture
  (Dense(128) -> Dropout(0.3) -> Dense(64) -> Dropout(0.3) -> Dense(1,
  sigmoid)) trained with early stopping on validation loss.

All hyperparameter tuning used a validation split carved from the
**training set only**; the held-out test set was touched exactly once per
model for final evaluation, preserving an unbiased performance estimate.

## 3.5 Robustness Validation

Beyond a single train/test evaluation, robustness was assessed via:

- **5-fold stratified cross-validation** for KNN, Logistic Regression, and
  Random Forest, reporting mean ± standard deviation across folds.
- **Repeated training across 5 random seeds** for the Neural Network,
  since its dominant source of run-to-run variance is weight
  initialization and stochastic gradient descent rather than data
  partitioning.

# 4. Results

## 4.1 Overall Performance

| Model | Type | Best Feature Set | Test Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---|---|---|---|---|---|---|
| Logistic Regression | Parametric | TF-IDF | 98.36% | 98.89% | 97.44% | 98.16% | 0.9985 |
| Neural Network | Deep Learning | TF-IDF | ~98.2% | ~98.1% | ~97.7% | ~98.0% | 0.9969-0.9978 |
| Random Forest | Ensemble | TF-IDF | 98.01% | 99.29% | 96.27% | 97.75% | 0.9980 |
| KNN | Non-Parametric | Embeddings | 94.56% | 96.28% | 91.47% | 93.84% | 0.9867 |

*(Neural Network figures reflect a modest range observed across repeated
runs and feature sets rather than a single fixed value; see Section 3.5.)*

## 4.2 Robustness (Cross-Validation / Repeated Seeds)

| Model | Metric Validated | Mean Accuracy | Std Dev |
|---|---|---|---|
| KNN (embeddings) | 5-fold CV | 94.26% | ±0.29% |
| Logistic Regression (TF-IDF) | 5-fold CV | 98.10% | ±0.11% |
| Random Forest (TF-IDF) | 5-fold CV | 97.55% | ±0.18% |
| Neural Network (TF-IDF/BoW) | 5 seeds | ~98.2% | ±0.04-0.08% |

All standard deviations are small (under 0.3 percentage points),
indicating each model's reported performance is a stable property of its
configuration rather than a result of a favorable single split or
initialization.

## 4.3 Feature Set Comparison

Accuracy varied substantially by feature representation, most dramatically
for KNN (82.24% on TF-IDF vs. 94.56% on embeddings -- see Figure 9,
`09_model_feature_heatmap.png`). For the other three models, TF-IDF was
consistently the strongest or near-strongest feature set, with BoW close
behind and embeddings trailing by 1-4 percentage points -- suggesting that,
for this task, sparse weighted word-count representations retain more
directly useful signal than a 100-dimensional averaged embedding.

## 4.4 Confusion Matrices and Error Patterns

Confusion matrices for each model's best configuration are shown in Figure
6 (`06_confusion_matrices_grid.png`). Random Forest showed the highest
precision (99.29%) but comparatively lower recall (96.27%), indicating a
more conservative model that under-flags some fake articles in exchange
for very few false alarms on real news. Logistic Regression offered the
most balanced precision/recall trade-off among the top performers.

## 4.5 ROC and Precision-Recall Analysis

All four models achieved ROC AUC above 0.98 (Figure 7,
`07_roc_curves.png`), with Logistic Regression, Random Forest, and the
Neural Network clustering tightly near-perfect discrimination (AUC
0.997-0.999) and KNN trailing modestly (AUC 0.987). Precision-Recall
curves (Figure 8, `08_precision_recall_curves.png`) show the same
ordering, with Average Precision scores closely tracking ROC AUC for each
model.

## 4.6 Training Time

Training time varied by roughly three orders of magnitude across models
(Figure 11, `11_training_time_comparison.png`): Logistic Regression
trained in a fraction of a second, while Random Forest required up to two
minutes for its largest configuration (200 trees) on a single-core
environment. This cost/benefit trade-off is discussed further in Section 5.