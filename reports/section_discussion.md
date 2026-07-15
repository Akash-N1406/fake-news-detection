# Discussion: Parametric vs. Non-Parametric Models

## Overview

This project evaluated four classifiers spanning two fundamentally different
modeling philosophies: **non-parametric** (K-Nearest Neighbors), **parametric**
(Logistic Regression), and two additional paradigms included for completeness
-- an **ensemble** method (Random Forest) and a **deep learning** method
(a feedforward Neural Network). All four were evaluated across three feature
representations (Bag-of-Words, TF-IDF, and averaged Word2Vec embeddings) on
the same 30,883/7,721 train/test split, with results further validated via
5-fold cross-validation (KNN, Logistic Regression, Random Forest) and
repeated-seed training (Neural Network).

## Final Results Summary

| Model | Type | Best Feature Set | Test Accuracy | F1 | ROC AUC | Training Time |
|---|---|---|---|---|---|---|
| Logistic Regression | Parametric | TF-IDF | 98.36% | 98.16% | 0.9985 | 0.17-0.79s |
| Neural Network | Deep Learning | TF-IDF / BoW | 98.0-98.3% | 97.8-98.1% | 0.9969-0.9978 | 17-23s |
| Random Forest | Ensemble | TF-IDF | 98.01% | 97.75% | 0.9980 | 15-120s |
| KNN | Non-Parametric | Embeddings | 94.56% | 93.84% | 0.9867 | 3-6s |

*(Training-time ranges reflect variation observed across hardware -- a single-core
sandboxed environment versus a multi-core development machine -- rather than
differences in the algorithms themselves; relative ordering between models was
consistent across both.)*

## Why the Non-Parametric Model (KNN) Struggled

KNN's performance depended dramatically on which feature representation it was
given -- far more than any other model in this study. On TF-IDF (5,000
dimensions, sparse), KNN reached only **82.24% accuracy**. On Word2Vec
embeddings (100 dimensions, dense), the *same algorithm* reached **94.56%
accuracy** -- a 12-point swing from a feature representation change alone.

This is a direct illustration of the **curse of dimensionality**. KNN's
entire decision rule rests on the assumption that "nearby" points in feature
space are similar. In high-dimensional sparse spaces, however, distances
between points become increasingly uniform -- the ratio between the nearest
and farthest neighbor's distance approaches 1 as dimensionality grows, which
strips the neighbor ranking of much of its meaning. TF-IDF's 5,000-word
vocabulary, though sparse, is exactly the kind of high-dimensional space
where this breaks down. Word2Vec's 100-dimensional dense space, by contrast,
compresses semantic information into far fewer dimensions, preserving
meaningful distance relationships and letting KNN's neighbor-voting logic
actually work.

Cross-validation confirmed this was not a fluke of one train/test split:
KNN on embeddings held at **94.26% +/- 0.29%** accuracy across 5 folds, a
small standard deviation indicating a stable (if modest, relative to the
other models) result.

## Why the Parametric Model (Logistic Regression) Won

Logistic Regression -- the simplest and most "parametric" model in this
study, with a fixed number of learned coefficients regardless of dataset
size -- was the **best-performing model overall**, at 98.36% accuracy and a
0.9985 ROC AUC, edging out both the ensemble and deep learning approaches.

This result makes sense given the structure of the task. Text classification
via TF-IDF or Bag-of-Words features is very often close to *linearly
separable*: individual words (or weighted words) push the decision toward
"Fake" or "Real" largely independently of one another, which is precisely
the assumption a linear model like Logistic Regression encodes. Where the
data genuinely rewards a linear decision boundary, adding model complexity
(more trees, more layers) buys little and costs a great deal:

- Logistic Regression trained in well under **1 second**.
- Random Forest took **15-120 seconds** for a comparable (slightly lower)
  accuracy.
- The Neural Network took **17-23 seconds** and, despite its added capacity,
  did not exceed Logistic Regression's accuracy on any feature set.

Cross-validation again confirmed stability: **98.10% +/- 0.11%** across 5
folds, the tightest margin of error among the three CV-validated models.

Logistic Regression's coefficients also offered direct interpretability
(Day 9) -- something KNN and the Neural Network cannot provide without
additional tooling. The top coefficients pushing toward "Fake"
(`via`, `read`, `pic`, `watch`, `com`, `featured`) and toward "Real"
(weekday names, `reporters`, `citing`) were corroborated independently by
Random Forest's feature importances (Day 10), giving two structurally
different models converging on the same signal -- a meaningfully stronger
form of evidence than either result alone.

## An Honest Limitation: What Is the Model Actually Learning?

The words driving classification lean heavily toward **formatting and
sourcing artifacts** rather than semantic content about truthfulness itself.
Terms like `via`, `pic`, `featured`, `getty`, and `image` most plausibly
reflect embedded social-media captions, image credits, and clickbait framing
common in the Fake articles of this dataset, while `reporters`, `citing`,
and weekday names reflect wire-service journalistic convention in the Real
articles. This was reinforced by an earlier, more direct leakage discovery
during EDA: the Reuters dateline (`CITY (Reuters) - `) appeared in 99.8% of
Real articles and had to be explicitly stripped from the corpus (see Dataset
Description) to prevent the model from trivially keying off source
attribution rather than content.

This suggests the ~98% accuracy achieved here reflects strong performance at
distinguishing *this dataset's* Real and Fake articles by their
**stylistic and sourcing conventions**, which is a genuinely useful and
demonstrable signal, but is a narrower claim than "detecting fake news"
in general. A model trained here may generalize less well to fake news
that mimics wire-service formatting, or real news published without it.
This is a legitimate scope limitation to state plainly in the report's
Conclusion, not a flaw to hide.

## Ensemble and Deep Learning in Context

Random Forest and the Neural Network both performed competitively
(97.75-98.1% F1) without matching Logistic Regression's top result. Random
Forest's feature importances provided useful interpretability similar to
Logistic Regression's coefficients, at a substantial multiple of the
training cost. The Neural Network, despite having the most representational
capacity of the four models, converged quickly (typically within 4-14
epochs before early stopping) and did not translate that capacity into a
measurable accuracy advantage on this task -- itself a useful finding:
additional model capacity is not automatically beneficial when the
underlying decision boundary is close to linear.

## Conclusion of Discussion

Across single-split evaluation, 5-fold cross-validation, and repeated-seed
robustness checks, the ranking was consistent: **Logistic Regression >=
Neural Network >= Random Forest >> KNN**, with the gap between the top three
being small and the gap to KNN being substantial and feature-set-dependent.
The central lesson is not that simpler models are always better, but that
**model choice should follow from the structure of the problem and its
features** -- a linearly-separable, high-dimensional sparse text problem
favored a linear model, while KNN's non-parametric, distance-based approach
only became competitive once given a lower-dimensional, denser feature
space suited to its assumptions.