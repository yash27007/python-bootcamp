# Handling Outliers

## What you'll learn

The IQR (interquartile range) and Z-score outlier-detection rules, derived from their definitions
and implemented from scratch with NumPy, checked against `scipy.stats.zscore` and
`sklearn.preprocessing.RobustScaler`. Why the two rules can disagree on the same data — the IQR
rule is robust to the very outliers it's detecting; a naive mean/std-based Z-score isn't, because
the outliers being tested for also inflate the statistics used to test for them.

## Why it matters

A handful of extreme values can dominate a mean, a standard deviation, or a squared-error model
fit — but a statistical outlier rule cannot tell a data-entry error apart from a rare, genuine
observation. This topic measures that distinction directly: removing genuinely extreme (not
erroneous) values from a dataset and watching a regression's slope estimate get measurably worse,
not better.

## Prerequisites

- `02-statistics` (percentiles, standard deviation, distribution shape)
- `03-data-analysis/05-matplotlib` or `06-seaborn` (reading a box plot)

## What you'll build

- Manual IQR and Z-score outlier detectors implemented directly from their definitions, matched
  exactly against `scipy.stats.zscore` and `sklearn.preprocessing.RobustScaler`'s
  median/IQR-based centering and scaling
- A real experiment: a linear relationship that holds genuinely into the tail of the data, with
  the IQR rule applied blindly and the resulting slope-estimate bias measured against the known
  true slope

See [`notes.md`](notes.md) for the full write-up including the derivation and captured experiment
output, and [`02-handling-outliers.ipynb`](02-handling-outliers.ipynb) (all cells executed) for
the practical tour.

## Where it shows up in real systems

Outlier detection is a standard first-pass data-quality check before any aggregate or model fit —
flagging sensor glitches, entry typos, and payment anomalies. Fraud and intrusion detection are,
at their core, outlier-detection problems at larger scale (Isolation Forest, One-Class SVM, Local
Outlier Factor generalize the same idea to many features at once).

## What's next

`03-data-encoding` — categorical features need a different kind of transformation before a model
can use them: converting category labels into numbers without inventing a false ordering.
