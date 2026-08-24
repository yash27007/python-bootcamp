# Inferential Statistics

## What you'll learn

How to draw conclusions about an entire population from a sample, with a quantified level of
certainty: sampling distributions and standard error, confidence intervals, the full hypothesis
testing framework (Type I/II errors, p-values), the standard test family (one-sample/two-sample/
paired t-tests, ANOVA, chi-square), effect size, and — critically — what a confidence interval and
a p-value do and do not actually mean.

## Why it matters

Every A/B test, clinical trial, and "is this change actually better" decision in engineering and
product work runs through this topic's tools. Misreading a confidence interval's meaning, or
running enough tests that one comes back significant by chance (p-hacking, the multiple-comparisons
problem), produces false confidence in decisions — this topic covers both the correct machinery and
the specific ways it gets misused.

## Prerequisites

- `02-probability` (the sampling distribution and Central Limit Theorem are used directly to
  justify confidence intervals and t-tests here)

## What you'll build

- A from-scratch bootstrap confidence interval (100,000 resamples) for the mean, compared directly
  against the parametric t-based interval on the same data
- A from-scratch permutation test (100,000 shuffles) for a two-group mean difference, compared
  directly against the independent t-test's p-value on the same data

See [`notes.md`](notes.md) for the full write-up including real captured output, and
[`inferential_statistics.ipynb`](inferential_statistics.ipynb) (all cells executed) for the
practical tour — confidence intervals (with a 20-interval capture-rate visualization), one-sample/
two-sample/paired t-tests, ANOVA, chi-square, normality testing, and a p-value/rejection-region
visualization.

## Where it shows up in real systems

A/B testing in product and marketing is applied hypothesis testing end to end. Clinical trials use
paired and independent t-tests with pre-registered significance levels specifically to guard against
p-hacking. Bootstrap methods are the default whenever a metric's sampling distribution has no clean
closed form (a ratio of two random quantities, a cross-validated model score) — resample the data,
recompute the metric, and read the interval off the resulting distribution.

## What's next

`03-data-analysis` — applying these statistical tools during real exploratory data analysis with
NumPy, Pandas, Matplotlib, and Seaborn.
