# 04 — Feature Engineering

Feature engineering is where a raw, imperfect dataset becomes something a model can actually
learn from. This section covers the four preprocessing problems every real dataset runs into:
gaps in the data, extreme values, non-numeric categories, and unevenly represented classes — each
treated from first principles (why the problem breaks naive approaches, a from-scratch
implementation checked against the library equivalent, and a real measured failure mode) rather
than as a list of function calls.

## Topics

| # | Topic | What it covers |
|---|-------|-----------------|
| 01 | [Missing Values](01-missing-values/) | MCAR/MAR/MNAR taxonomy, manual mean/median imputation vs. `SimpleImputer`, KNN imputation, measured imputation-before-split leakage |
| 02 | [Handling Outliers](02-handling-outliers/) | Manual IQR/Z-score detection vs. `scipy`/`sklearn`, measured bias from removing genuine extreme values |
| 03 | [Data Encoding](03-data-encoding/) | Manual one-hot encoding vs. `pd.get_dummies`/`OneHotEncoder`, high-cardinality blow-up, unseen-category failures |
| 04 | [Imbalanced Datasets](04-handling-imbalanced-dataset/) | Why accuracy misleads, from-scratch SMOTE synthesis, measured SMOTE-before-split leakage, class weighting vs. resampling |

## Why feature engineering matters

Clean, well-engineered features have more impact on model accuracy than the choice of algorithm —
a model trained on data with unaddressed gaps, uncontrolled outliers, unencoded categories, or
unaddressed class imbalance will underperform regardless of how sophisticated the model itself is.
Every topic in this section pairs a from-scratch implementation with the equivalent library
tooling, and — where the failure mode is a leakage bug — a real, measured comparison between the
leaky and correct pipeline, not just an assertion that leakage matters.

## Prerequisites

- `01-python-foundation` (control flow, functions, basic OOP)
- `02-statistics` (mean, median, standard deviation, percentiles)
- `03-data-analysis` (NumPy, Pandas — recommended, not strictly required)

## What's next

`05-machine-learning` — the engineered features from this section become model inputs, starting
with `01-introduction` through the first supervised learning algorithms. `01-missing-values` and
`04-handling-imbalanced-dataset` both bridge explicitly to `05-machine-learning/09-knn`'s
distance-based reasoning.
