# Descriptive Statistics

## What you'll learn

How to compress a dataset into a small set of honest summary numbers — central tendency (mean,
median, mode), dispersion (range, variance, standard deviation, IQR), shape (skewness, kurtosis),
and how two variables move together (covariance, Pearson correlation) — and, just as importantly,
when each of those summaries is misleading.

## Why it matters

Every downstream step in a data science workflow (visualization, feature engineering, modeling)
starts from `.describe()`-style summaries and correlation checks. Trusting a mean and standard
deviation on a skewed or multimodal distribution, or trusting an aggregate correlation without
checking subgroups, produces conclusions that are wrong in specific, predictable ways (see this
topic's Failure modes: skewed/multimodal distributions, Simpson's paradox).

## Prerequisites

- Comfort with NumPy/Pandas basics (`01-python-foundation`)
- No prior statistics required — this topic derives every formula from its definition

## What you'll build

- Mean, variance, standard deviation, and Pearson correlation implemented directly from their
  mathematical definitions in pure Python — verified against NumPy on the real Iris dataset (exact
  match to six decimal places)
- A real, executed simulation (200,000 trials) measuring the actual bias of the `/n` vs `/(n-1)`
  variance-denominator choice, quantifying *why* Bessel's correction exists

See [`notes.md`](notes.md) for the full write-up including real captured output, and
[`descriptive_statistics.ipynb`](descriptive_statistics.ipynb) (all cells executed) for the
practical tour on the Iris dataset — central tendency, dispersion, box plots, skewness/kurtosis,
correlation heatmaps, IQR-based outlier detection, and grouped statistics.

## Where it shows up in real systems

`.describe()`, histograms, and box plots are the first step of virtually every EDA workflow —
catching data-entry errors, motivating transforms (e.g. log-transforming skewed data), and flagging
outliers before they corrupt a mean-based feature. Correlation matrices are a standard first pass at
feature selection and multicollinearity checks before linear regression.

## What's next

`02-probability` — the mathematical framework for reasoning about the uncertainty that these
summary statistics are estimates of.
