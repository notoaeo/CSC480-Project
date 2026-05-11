# Sentiment Analysis of Movie Reviews — CSC 480 Final Project

**Author:** Ethan Olla
**Course:** CSC 480, Spring 2026 (Instructor: Chicheng Zhang)
**Status:** Mid-project (experiments complete, ablations and writeup in progress)

This repository contains the code, data pipeline, and results for a systematic
comparison of classical machine learning classifiers on the IMDb Large Movie
Review Dataset. It accompanies the project proposal and mid-project
presentation for CSC 480.

## TL;DR

| Model               | BoW   | TF-IDF | TF-IDF + bigrams |
|---------------------|-------|--------|------------------|
| Naive Bayes         | 0.842 | 0.857  | 0.884            |
| Logistic Regression | 0.881 | 0.893  | 0.901            |
| Linear SVM          | 0.860 | 0.891  | **0.911**        |

Best model: **Linear SVM + TF-IDF (unigrams + bigrams)** at **91.12%** test
accuracy on the 25k-review IMDb test split, beating the 88.9% baseline reported
by Maas et al. (2011) on the same data.

## Project structure

```
.
├── README.md                  ← you are here
├── run_experiments.py         ← main pipeline (data → features → models → plots)
├── requirements.txt           ← minimal python dependencies
├── results.csv                ← per-(model, vectorizer) metrics
├── summary.json               ← compact run summary
├── top_features.json          ← top positive/negative LogReg coefficients
├── sample_errors.json         ← misclassified review examples for error analysis
├── plot_accuracy.png          ← bar chart of all nine model/feature combinations
├── plot_confusion.png         ← confusion matrix for the best model
├── plot_features.png          ← top 15 positive and negative LogReg features
└── CSC480_MidProject_Olla.pptx ← mid-project presentation slides
```

## Reproducing the results

### 1. Environment

Tested on Python 3.12. Install dependencies:

```bash
pip install -r requirements.txt
```

`requirements.txt` contains:

```
numpy
pandas
scikit-learn
matplotlib
```

### 2. Dataset

The experiments use the IMDb Large Movie Review Dataset (Maas et al., 2011).
The script expects a single CSV at `imdb_repo/IMDB-Dataset.csv` with two
columns: `review` and `sentiment` (values `positive` or `negative`).

You can get the CSV two ways:

```bash
# Option A: github mirror (used during development)
git clone --depth 1 https://github.com/Ankit152/IMDB-sentiment-analysis.git imdb_repo

# Option B: original Stanford tarball, then convert to CSV
curl -LO https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
tar -xzf aclImdb_v1.tar.gz
# ...and write a short script to flatten pos/neg folders into review,sentiment columns
```

Both produce the same 50,000 labeled reviews with a 50/50 positive/negative split.

### 3. Run the pipeline

```bash
python run_experiments.py
```

This single command will:

1. Load and clean the 50k reviews (strip `<br />` tags, lowercase).
2. Create a stratified 25k/25k train/test split (random_state=42).
3. Fit three vectorizers: BoW, TF-IDF unigrams, TF-IDF unigrams+bigrams.
4. Train Multinomial NB, Logistic Regression, and Linear SVM on each.
5. Report held-out test accuracy, precision, recall, F1, and 5-fold CV mean.
6. Run a short error analysis on the best model.
7. Extract top-15 positive and negative features from the LogReg model.
8. Save `results.csv`, `summary.json`, and three PNG plots to the working directory.

Expected runtime: about 90 seconds on a laptop.

### 4. Regenerating the slide deck

The mid-project deck (`CSC480_MidProject_Olla.pptx`) is built from
`plot_accuracy.png`, `plot_confusion.png`, and `plot_features.png`, so rerun
the pipeline first and then rebuild if needed. The deck itself was generated
with `pptxgenjs`; the generator script is not required to reproduce the
numbers.

## Method notes

- **Cleaning:** only HTML `<br />` stripping and lowercasing. No stemming or
  lemmatization.
- **Vectorizer hyperparameters:** `min_df=5`, `max_df=0.9`, `sublinear_tf=True`
  for TF-IDF. English stop words are kept (they encode negation and intensity).
- **Models:** default scikit-learn hyperparameters except
  `LogisticRegression(solver="liblinear", max_iter=1000)` and `LinearSVC(C=1.0)`.
  All three are tuned with 5-fold cross-validation on the training set; CV
  accuracies agree with held-out test accuracies to within 0.005.
- **Random seed:** 42 everywhere, so the train/test split and reported numbers
  are deterministic.

## Known limitations and next steps

1. **Bigrams still miss long-range negation.** An error analysis on misclassified
   reviews shows they skew longer (mean 235 words vs. 229 for correct), which
   suggests mixed-tone and sarcastic reviews are the dominant failure mode.
2. **LR and SVM are within a point.** The final writeup will add paired
   statistical comparisons and an ablation over vocabulary size cutoffs,
   stop words, and sublinear TF to separate them.
3. **No neural baseline.** This is intentional for the mid-project (classical
   pipelines only), but the final report will add a short discussion of where
   word-vector and transformer models would pick up the remaining headroom.

## References

1. Maas, A. L., Daly, R. E., Pham, P. T., Huang, D., Ng, A. Y., & Potts, C. (2011).
   *Learning Word Vectors for Sentiment Analysis.* ACL 2011.
2. Pang, B., Lee, L., & Vaithyanathan, S. (2002).
   *Thumbs up? Sentiment Classification using Machine Learning Techniques.* EMNLP 2002.
3. Wang, S., & Manning, C. D. (2012).
   *Baselines and Bigrams: Simple, Good Sentiment and Topic Classification.* ACL 2012.
