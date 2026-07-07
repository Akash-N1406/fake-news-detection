# Fake News Detection using NLP & Machine Learning

A from-scratch machine learning pipeline that classifies news articles as **real** or **fake**, built as a portfolio / interview-prep project. Preprocessing, feature extraction, model training, and evaluation are implemented manually rather than relying on pre-built end-to-end solutions.

## Problem Statement

Build a text classification pipeline covering:
- Manual text cleaning and tokenization
- Feature engineering (Bag-of-Words, TF-IDF, word embeddings)
- Multiple model families: KNN, Logistic Regression, Random Forest, Neural Network
- Full evaluation (accuracy, precision, recall, F1, confusion matrices)

## Dataset

[Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset) (Kaggle, by clmentbisaillon)

- `Fake.csv` — 23,481 fake news articles
- `True.csv` — 21,417 real news articles
- Combined and cleaned to **38,604 rows** after removing corrupted rows, empty text, and exact duplicates (see `src/day1_load_data.py`, `src/day2_clean_tokenize.py`)

**Note on `subject`:** the `subject` column perfectly separates fake vs. real articles in this dataset (Fake only uses `News`/`politics`/`left-news` etc., True only uses `politicsNews`/`worldnews`). It is **not** used as a model feature — only the article `title`/`text` are, to keep the task a genuine language-classification problem rather than a trivial category lookup.

## Project Structure

```
fake-news-detection/
├── data/                   # Fake.csv, True.csv (download from Kaggle, see below)
├── notebooks/              # Final consolidated walkthrough notebook (Week 3)
├── src/                    # Pipeline scripts, run in day order
├── reports/                # IEEE-format report, presentation
├── requirements.txt
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
python day1_load_data.py         # merge + clean raw data -> data/combined_raw.csv
python day2_clean_tokenize.py    # manual text cleaning + tokenization -> data/combined_cleaned.csv
```

(More scripts added as the project progresses — see Progress below.)

## Progress

| Day | Task | Status |
|-----|------|--------|
| 1 | Data loading, merging, initial cleaning | ✅ |
| 2 | Manual text cleaning + tokenization | ✅ |
| 3 | Exploratory Data Analysis | ⬜ |
| 4 | Bag-of-Words | ⬜ |
| 5 | TF-IDF | ⬜ |
| 6 | Word embeddings | ⬜ |
| 7 | Feature consolidation | ⬜ |
| 8 | KNN | ⬜ |
| 9 | Logistic Regression | ⬜ |
| 10 | Random Forest | ⬜ |
| 11 | Neural Network (Keras) | ⬜ |
| 12-13 | Hyperparameter tuning + cross-validation | ⬜ |
| 14 | Consolidate results | ⬜ |
| 15-16 | Evaluation + visualizations | ⬜ |
| 17-19 | IEEE report (Intro–Appendix) | ⬜ |
| 20-21 | Presentation + final polish | ⬜ |

## Algorithms

- **KNN** (non-parametric)
- **Logistic Regression** (parametric)
- **Random Forest** (ensemble)
- **Neural Network** (deep learning, TensorFlow/Keras)

## Report Format

IEEE standard — see `reports/` for the final submission covering Introduction, Dataset Description, Methodology, Results, Discussion (parametric vs. non-parametric comparison), Conclusion, and Appendix.