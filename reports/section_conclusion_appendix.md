# 6. Conclusion

## 6.1 Summary of Findings

This project built a complete, from-scratch text classification pipeline
for fake news detection, evaluating four algorithms (KNN, Logistic
Regression, Random Forest, and a Neural Network) across three feature
representations (Bag-of-Words, TF-IDF, and Word2Vec embeddings). Logistic
Regression, the simplest and most parametric model tested, achieved the
best overall performance (98.36% accuracy, 0.9985 ROC AUC on TF-IDF
features), closely followed by the Neural Network and Random Forest.
KNN performed substantially worse on high-dimensional sparse features but
recovered to a respectable 94.56% accuracy once given a lower-dimensional
dense embedding space -- a direct, empirically demonstrated illustration
of the curse of dimensionality affecting distance-based classifiers.

All results were validated beyond a single train/test split via 5-fold
cross-validation (KNN, Logistic Regression, Random Forest) and
repeated-seed training (Neural Network), with all four models showing
low variance (standard deviations under 0.3 percentage points),
indicating the reported performance figures are stable and reproducible
rather than artifacts of a favorable data split or initialization.

## 6.2 Key Insights

1. **Feature representation matters more than model choice for
   distance-based methods.** KNN's accuracy swung by 12 percentage points
   depending solely on which feature set it was given, more than the gap
   between the best and worst-performing models overall.
2. **Model complexity did not translate to better accuracy on this
   task.** Logistic Regression, with a fixed and comparatively small
   number of parameters, matched or exceeded the ensemble and deep
   learning approaches while training in a fraction of a second versus
   up to two minutes for Random Forest.
3. **Dataset metadata can silently solve a classification task without
   genuine content understanding.** Two independent leakage sources
   (the `subject` category and the Reuters wire-service dateline) were
   identified and corrected during this project; had they gone unnoticed,
   reported accuracy would have been inflated by an artifact of the
   dataset's collection process rather than a reflection of the model's
   ability to detect fake news.
4. **What the model learns is narrower than "fake news detection" in
   general.** The most predictive words across models were dominated by
   formatting and sourcing conventions (`via`, `pic`, `featured`,
   `reporters`, `citing`) rather than semantic claims about factual
   accuracy. The models in this project are better described as
   detecting *this dataset's* stylistic signature of fake vs. real
   articles than as general-purpose misinformation detectors.

## 6.3 Limitations

- **Dataset scope.** The dataset consists primarily of U.S. political
  news from 2016-2017; performance may not generalize to other domains,
  languages, time periods, or more sophisticated modern fake news that
  deliberately mimics wire-service formatting.
- **Word2Vec corpus size.** The Word2Vec embeddings were trained on
  ~30,883 documents, a modest corpus by typical standards for this
  technique (which often benefits from millions of documents), likely
  contributing to some noisy nearest-neighbor relationships observed
  during embedding validation.
- **Style vs. substance.** As discussed in Section 5, the strongest
  predictive signals reflect formatting/sourcing artifacts rather than
  claims-level fact-checking, meaning this pipeline is better understood
  as a stylistic-pattern classifier than a truth-verification system.
- **Static evaluation.** All evaluation used a single fixed dataset
  snapshot; no testing was performed against adversarially-constructed
  examples designed to evade detection, nor against out-of-distribution
  news sources.

## 6.4 Future Scope

- Evaluate generalization on an independent, more recent fake-news
  dataset (e.g., via the NewsAPI-based scraping approach mentioned in
  the assignment brief) to test whether performance holds outside this
  dataset's specific stylistic conventions.
- Train Word2Vec (or substitute a larger pretrained embedding set, where
  available) on a substantially larger corpus to assess whether semantic
  embeddings can close the gap with TF-IDF given more training data.
- Incorporate metadata features known NOT to leak the label directly
  (e.g., publication recency, article length) as a secondary experiment
  once stylistic-artifact features are controlled for or removed.
- Extend the Neural Network to a more sophisticated architecture (e.g.,
  an LSTM or attention-based model operating on token sequences directly,
  rather than averaged/aggregated features) to test whether sequential
  context provides an accuracy advantage the current feedforward
  architecture cannot capture.

# 7. Appendix

## 7.1 Project Repository Structure

```
fake-news-detection/
|-- data/                   # Fake.csv, True.csv, and derived feature files
|-- notebooks/              # Consolidated walkthrough notebook
|-- src/                    # Pipeline scripts (day1 through day16)
|-- reports/                # Result JSONs, figures, and report sections
|-- requirements.txt
|-- README.md
```

## 7.2 Pipeline Script Index

| Script | Purpose |
|---|---|
| `day1_load_data.py` | Load, merge, and clean raw Fake.csv/True.csv |
| `day2_clean_tokenize.py` | Manual text cleaning, dateline stripping, tokenization |
| `day3_eda.py` | Exploratory data analysis and visualization |
| `day4_bag_of_words.py` | Train/test split; Bag-of-Words feature construction |
| `day5_tfidf.py` | TF-IDF feature construction |
| `day6_embeddings.py` | Word2Vec embedding training and document vectorization |
| `day7_consolidate_features.py` | Feature set validation and unified loader |
| `day8_knn.py` | K-Nearest Neighbors training and evaluation |
| `day9_logistic_regression.py` | Logistic Regression training and evaluation |
| `day10_random_forest.py` | Random Forest training and evaluation |
| `day11_neural_network.py` | Neural Network (Keras) training and evaluation |
| `day12_cv_tuning.py` | Cross-validation-based hyperparameter tuning |
| `day13_nn_robustness.py` | Neural Network robustness via repeated seeds |
| `day14_consolidate_results.py` | Final model comparison table and charts |
| `day15_roc_curves.py` | ROC and Precision-Recall curve generation |
| `day16_additional_viz.py` | Heatmap, metric comparison, training time charts |

## 7.3 Hyperparameter Search Spaces

| Model | Hyperparameter | Values Tested |
|---|---|---|
| KNN | k (n_neighbors) | 3, 5, 7, 9, 11 |
| Logistic Regression | C (inverse regularization) | 0.01, 0.1, 1.0, 10.0 |
| Random Forest | n_estimators | 100, 200 |
| Random Forest | max_depth | None, 30 |
| Neural Network | Architecture | Dense(128)-Dropout(0.3)-Dense(64)-Dropout(0.3)-Dense(1) |
| Neural Network | Early stopping patience | 3 epochs (monitor: validation loss) |

## 7.4 Software Environment

- Python 3.12
- scikit-learn (KNN, Logistic Regression, Random Forest, evaluation metrics)
- TensorFlow/Keras (Neural Network)
- gensim (Word2Vec)
- pandas, numpy, scipy.sparse (data handling)
- matplotlib, seaborn (visualization)

## 7.5 Sample Test Data

A representative sample of test-set predictions (article title, true
label, and each model's prediction) is included in
`reports/day16_full_results_table.csv` and the individual `dayN_*_results.json`
files, which contain full confusion matrices and per-model metrics for
independent verification.