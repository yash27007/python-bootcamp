# Handling Imbalanced Datasets

## What you'll learn

Why accuracy is a misleading metric when one class vastly outnumbers another — made concrete by
actually computing a naive "always predict majority class" model's accuracy on a real imbalanced
dataset, then confirming it with a fitted `DummyClassifier`. SMOTE's actual synthesis mechanism —
find $k$ nearest minority neighbors, interpolate — implemented from scratch and visualized, with
an explicit bridge to [`../../05-machine-learning/09-knn`](../../05-machine-learning/09-knn)'s
distance-based reasoning.

## Why it matters

A model can reach 90%+ accuracy on imbalanced data while identifying zero cases of the class that
actually matters — this topic measures that gap directly, then measures the real tradeoff between
class weighting and resampling on minority-class recall. Applying SMOTE before a train/test split
is the same category of data-leakage bug as imputing before a split, and it's measured with real
numbers here, not just asserted.

## Prerequisites

- `01-missing-values` (the leakage pattern — fit on train, apply to test — recurs here for SMOTE)
- `05-machine-learning/09-knn` (SMOTE's neighbor-search step is exactly KNN's distance computation)

## What you'll build

- A real, computed naive majority-class-baseline accuracy (0.90 on a 900/100-imbalanced dataset),
  confirmed against `sklearn.dummy.DummyClassifier`
- A from-scratch SMOTE-style synthetic-sample generator on a toy 2D dataset — real minority points
  vs. generated synthetic points, plotted together
- A real, measured SMOTE-before-split leakage experiment (+0.0213 mean accuracy inflation over 300
  trials) and a real class-weighting-vs-resampling comparison on minority-class recall

See [`notes.md`](notes.md) for the full write-up including captured experiment output, and
[`04-handling-imbalanced-dataset.ipynb`](04-handling-imbalanced-dataset.ipynb) (all cells
executed) for the practical tour.

## Where it shows up in real systems

Fraud detection, medical diagnosis, defect detection, and churn prediction are naturally
imbalanced classification problems where precision/recall/F1 replace accuracy as the metrics that
actually matter. SMOTE and class weighting are both standard production techniques — always
applied strictly inside the training-fold boundary of cross-validation, exactly as demonstrated
in this topic's leakage experiment.

## What's next

This is the last topic in `04-feature-engineering`. From here: `05-machine-learning`, where these
engineered features become model inputs — starting with `01-introduction` through the first
supervised algorithms.
