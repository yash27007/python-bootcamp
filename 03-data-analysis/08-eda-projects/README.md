# EDA Projects

## What you'll learn

This is the section's capstone, not a new concept — three full exploratory-data-analysis passes
on real, messy datasets, each composing the tools built up across `01-numpy` through `07-sqlite`
(array statistics, `DataFrame` cleaning/`groupby`, Matplotlib/Seaborn distribution and correlation
plots) into one end-to-end workflow: missing-value diagnosis → dtype/format cleaning →
distribution/correlation visualization → written interpretation. See
[`notes.md`](notes.md) for why this topic deliberately uses a reduced structure instead of the
full 12-section template, and exactly what each project ties back to.

## Why it matters

A model trained on unexamined data silently inherits every problem in that data — missing values,
wrong dtypes, duplicates, outliers. EDA is the step that catches these before they become modeling
artifacts. All three projects here hit a real version of this: a string-typed `Installs` column
(`"10,000+"`) that isn't numeric until cleaned, a `to_csv` call that failed on a fresh checkout
because its output directory didn't exist, a 300k-row dataset too large to `pairplot` in full
without sampling first.

## Prerequisites

- `01-numpy`, `02-pandas`, `03-data-manipulation`, `05-matplotlib`, `06-seaborn` (every technique
  used here was built and derived in those topics)
- `07-sqlite` (the relational-querying mindset — filter, aggregate, join — carries over to
  `DataFrame` operations used throughout)

## What you'll find here

- **`wine-quality/`** — UCI red/white wine physicochemical data vs. quality score;
  correlation-heatmap-driven analysis (`redwine.ipynb`, `whitewinequality.ipynb`)
- **`flight-price-prediction/`** — ~300,000-row flight booking dataset; one-hot encoding,
  sampled `pairplot` analysis (`flight_price_prediction_v1.ipynb` complete, `_v2.ipynb` kept as
  an earlier exploratory pass)
- **`google-playstore-dataset/`** — Play Store app listings; missing-value/duplicate diagnosis,
  string-to-numeric cleaning, category popularity analysis (`googleplaystore.ipynb`)

All six notebooks execute end to end with no unexecuted cells.

## Where it shows up in real systems

This is what EDA looks like industry-wide, before any model is ever trained: inspect
missingness, dtypes, duplicates, distributions, and correlations, and write down what each finding
implies. Skipping this step is one of the most common causes of a model that performs well in a
notebook and fails in production on data nobody verified was clean.

## What's next

This is the last topic in `03-data-analysis`. See the section-level [`../README.md`](../README.md)
for what comes next — `04-feature-engineering` builds directly on the cleaning and transformation
patterns exercised here.
