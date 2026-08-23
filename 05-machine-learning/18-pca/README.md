# 18 – Principal Component Analysis (PCA)

## What you'll learn

How to reduce a high-dimensional dataset to a small number of new, uncorrelated directions that capture most of its variance — derived from covariance, eigenvectors, and eigenvalues rather than presented as a black-box `fit_transform`. Covers why PCA finds the *variance-maximizing* directions specifically (not just any rotation), how much information is lost when dimensions are dropped, and where a linear method like PCA breaks down (kernel PCA for nonlinear structure).

## Why it matters

High-dimensional data is hard to visualize, expensive to model, and often redundant — correlated features force a model to spend capacity re-deriving correlations the raw data already encodes. PCA is the standard first tool for compressing that redundancy away, both as a visualization aid (project to 2D/3D) and as a preprocessing step before modeling.

## Prerequisites

- `02-statistics` — variance, covariance, and basic linear algebra (eigenvectors/eigenvalues).
- `09-knn`'s Failure modes section — the curse-of-dimensionality argument PCA is one answer to.

## What you'll build

A from-scratch PCA (covariance matrix → eigendecomposition → projection) validated against `sklearn.decomposition.PCA` on real datasets — breast cancer and digits — including explained-variance-ratio analysis, 2D visualization, reconstruction error at different numbers of components, and a kernel PCA comparison for nonlinear structure (`PCA-Principal-Component-Analysis.ipynb`, `PCA-2.ipynb`).

## Where it appears in real systems

- Preprocessing before clustering or supervised learning on high-dimensional tabular or image data.
- Visualizing high-dimensional embeddings and datasets in 2D/3D.
- Noise reduction — dropping low-variance components often removes noise along with redundancy.
- Compression — a precursor to the idea behind autoencoders (`16-anomaly-detection`'s Practical implementation section touches this connection).

## What's next

This is the last topic in the current `05-machine-learning` sequence; from here the curriculum moves into `06-deep-learning`, where dimensionality reduction reappears as a *learned* (nonlinear) transformation rather than a fixed linear projection.
