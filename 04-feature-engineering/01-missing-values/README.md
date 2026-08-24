# Missing Values

## What you'll learn

Why missing data breaks most algorithms outright, and the MCAR/MAR/MNAR taxonomy that determines
*which* fix is statistically valid for a given gap — not just a list of imputation method names.
Mean/median imputation built from scratch with NumPy (`np.nanmean`/`np.nanmedian`), checked
element-for-element against `sklearn.impute.SimpleImputer`, plus `KNNImputer` as the
locally-aware alternative once a single global statistic isn't enough.

## Why it matters

Real data always has gaps — sensor dropouts, skipped survey questions, failed joins — and picking
the wrong fix for the actual missingness mechanism produces a confidently wrong model rather than
an obviously broken one. Imputing before a train/test split is one of the most common — and easy
to miss — data leakage bugs in applied ML, measured concretely in this topic rather than just
asserted.

## Prerequisites

- `02-statistics` (mean, median, and what "distribution" means)
- `03-data-analysis/02-pandas` (`Series`, `fillna`, boolean masking)

## What you'll build

- A manual mean/median imputer implemented in plain NumPy, verified against `SimpleImputer` on
  the identical Titanic `age` column
- A `KNNImputer` comparison against global-mean imputation, with an explicit bridge to
  [`../../05-machine-learning/09-knn`](../../05-machine-learning/09-knn) for the distance-based
  reasoning underneath it
- A real, measured data-leakage demonstration: imputing before vs. after a train/test split,
  compared on both the raw imputed values and downstream model accuracy, amplified on a smaller
  synthetic dataset to make the effect clearly visible

See [`notes.md`](notes.md) for the full write-up including the MCAR/MAR/MNAR derivation and
captured experiment output, and [`01-missing-values.ipynb`](01-missing-values.ipynb) (all cells
executed) for the practical tour.

## Where it shows up in real systems

Every production pipeline ingesting real-world data hits missing values as routine, not an edge
case — the MCAR/MAR/MNAR triage question comes before any fix is chosen. The leakage failure mode
here is a direct instance of the single most common data-leakage bug in applied ML: fitting any
preprocessing step on data that includes information from the evaluation set.

## What's next

`02-handling-outliers` — the next preprocessing failure mode: extreme values (rather than absent
ones) distorting statistics and model fits.
