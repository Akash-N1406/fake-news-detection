# Fake News Detection using NLP & Machine Learning

A from-scratch machine learning pipeline that classifies news articles as **real** or **fake**, built for the IICT AI & ML Summer Internship Program 2026. Preprocessing, feature extraction, model training, and evaluation are implemented manually rather than relying on pre-built end-to-end solutions.

**Full IEEE-format report and presentation deck are in `reports/`.**

## Problem Statement

Build a text classification pipeline covering:
- Manual text cleaning and tokenization
- Feature engineering (Bag-of-Words, TF-IDF, Word2Vec embeddings) — all implemented from their mathematical definitions, not via library vectorizers
- Four model families: KNN (non-parametric), Logistic Regression (parametric), Random Forest (ensemble), Neural Network (deep learning)
- Full evaluation: accuracy, precision, recall, F1, ROC/PR curves, confusion matrices, cross-validation, and repeated-seed robustness checks

## Key Results

| Model | Type | Best Feature Set | Test Accuracy | F1 | ROC AUC |
|---|---|---|---|---|---|
| **Logistic Regression** | Parametric | TF-IDF | **98.36%** | **98.16%** | **0.9985** |
| Neural Network | Deep Learning | TF-IDF | ~98.2% | ~98.0% | 0.997-0.998 |
| Random Forest | Ensemble | TF-IDF | 98.01% | 97.75% | 0.9980 |
| KNN | Non-Parametric | Embeddings | 94.56% | 93.84% | 0.9867 |

Logistic Regression — the simplest model tested — achieved the best overall performance, training in a fraction of a second versus up to two minutes for Random Forest. Full analysis in `reports/section_discussion.md` and the final report.

## Two Feature-Leakage Findings

Both discovered during EDA (Day 3) and corrected before modeling:

1. **`subject` category** perfectly separates Fake vs. Real articles with zero overlap — excluded from all feature sets, used only as metadata.
2. **Reuters wire-service dateline** (`CITY (Reuters) - `) appeared in 99.8% of Real articles vs. 1.3% of Fake — stripped via regex, with "reuters" added to the stopword list to catch residual mentions.

Left unaddressed, either would have let models "cheat" via source formatting rather than learning genuine content patterns.

## Dataset

[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle, by clmentbisaillon)

- `Fake.csv` — 23,481 fake news articles
- `True.csv` — 21,417 real news articles
- Combined and cleaned to **38,604 rows** after removing corrupted rows, empty text, exact duplicates, and applying the Reuters dateline fix above

## Project Structure

```
fake-news-detection/
├── data/                   # Fake.csv, True.csv (download from Kaggle) + derived feature files
├── notebooks/              # Jupyter notebooks for viewing and model checking
├── src/                    # Pipeline scripts, day1 through day16 (run in order)
├── reports/                # Result JSONs, figures, report sections, final report + presentation
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

```bash
git clone https://github.com/Akash-N1406/fake-news-detection.git
cd fake-news-detection

python3 -m venv venv
source venv/bin/activate       # WSL2/Ubuntu

pip install -r requirements.txt
```

## Getting the Data

1. Download `Fake.csv` and `True.csv` from [Kaggle](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Place both files in `data/`
3. Run the pipeline scripts in order (see below)

## Running the Pipeline

```bash
cd src
python day1_load_data.py              # merge + clean raw data
python day2_clean_tokenize.py         # manual cleaning, dateline stripping, tokenization
python day3_eda.py                    # exploratory analysis + figures
python day4_bag_of_words.py           # train/test split + Bag-of-Words
python day5_tfidf.py                  # TF-IDF features
python day6_embeddings.py             # Word2Vec embeddings (slow: several minutes)
python day7_consolidate_features.py   # validate feature sets, unified loader
python day8_knn.py                    # KNN
python day9_logistic_regression.py    # Logistic Regression
python day10_random_forest.py         # Random Forest (slow: several minutes)
python day11_neural_network.py        # Neural Network (Keras)
python day12_cv_tuning.py             # CV-based hyperparameter tuning
python day13_nn_robustness.py         # NN robustness via repeated seeds
python day14_consolidate_results.py   # final comparison table + charts
python day15_roc_curves.py            # ROC / Precision-Recall curves
python day16_additional_viz.py        # heatmap, metric comparison, training time
```

Each script reads its inputs from `data/` or `reports/` and writes its outputs there — safe to re-run individually once earlier steps have completed at least once.

## Progress

| Day | Task | Status |
|-----|------|--------|
| 1 | Data loading, merging, initial cleaning | ✅ |
| 2 | Manual text cleaning + tokenization + Reuters leakage fix | ✅ |
| 3 | Exploratory Data Analysis | ✅ |
| 4 | Train/test split + Bag-of-Words | ✅ |
| 5 | TF-IDF | ✅ |
| 6 | Word2Vec embeddings | ✅ |
| 7 | Feature consolidation + validation | ✅ |
| 8 | KNN | ✅ |
| 9 | Logistic Regression | ✅ |
| 10 | Random Forest | ✅ |
| 11 | Neural Network (Keras) | ✅ |
| 12-13 | CV-based hyperparameter tuning + NN robustness | ✅ |
| 14 | Consolidate results | ✅ |
| 15-16 | ROC/PR curves + additional visualizations | ✅ |
| 17-19 | IEEE report (Intro–Appendix), 20 pages | ✅ |
| 20-21 | Presentation deck + final polish | ✅ |

## Algorithms

- **KNN** (non-parametric) — best on Word2Vec embeddings; illustrates the curse of dimensionality (82.2% on TF-IDF vs. 94.6% on embeddings)
- **Logistic Regression** (parametric) — best overall model, fastest to train
- **Random Forest** (ensemble) — strong precision, more conservative recall
- **Neural Network** (deep learning, TensorFlow/Keras) — competitive but did not exceed Logistic Regression

## Notebooks

- **`notebooks/01_project_walkthrough.ipynb`** — narrative tour of the dataset, EDA, feature engineering, and final results. Reads already-generated data/figures rather than re-running the slow steps (Word2Vec training, Random Forest tuning).
- **`notebooks/02_model_inference_and_checking.ipynb`** — retrains the best model (Logistic Regression on TF-IDF, trains in under a second) and lets you classify your own text, spot-check test-set predictions, and inspect misclassified examples directly.

Both notebooks import directly from `src/` (e.g., `day2_clean_tokenize.process_row`, `day7_consolidate_features.load_feature_set`) rather than duplicating logic, so they always stay consistent with the actual pipeline.

## Deliverables

- **`reports/Fake_News_Detection_IEEE_Report.docx`** — 20-page IEEE-format report (Introduction, Dataset Description, Methodology with full algorithm mathematics, Results, Discussion, Conclusion, Appendix with code excerpts)
- **`reports/Fake_News_Detection_Presentation.pptx`** — 15-slide presentation deck
- **`reports/figures/`** — all 11 generated figures (EDA, model comparisons, ROC/PR curves, heatmaps)
- **`reports/*_results.json`** — raw results for every model/feature-set combination, for independent verification

## Known Limitations

The strongest predictive words across models (`via`, `pic`, `featured`, `getty`, `reporters`, `citing`) reflect **formatting and sourcing conventions** rather than semantic claims about factual accuracy. This project's ~98% accuracy reflects strong performance at detecting *this dataset's* stylistic signature of Real vs. Fake articles — a narrower, more honest claim than general-purpose fake news detection. See the report's Discussion and Conclusion sections for full treatment.