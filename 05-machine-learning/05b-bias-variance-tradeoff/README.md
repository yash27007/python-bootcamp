# 05b – Bias-Variance Tradeoff

## What you'll learn

Why a model that fits its training data almost perfectly can still perform worse on new data than a model that fits training data worse. This topic formalizes "overfitting" and "underfitting" into a single decomposition of test error — bias² + variance + irreducible noise — so that "the model is overfitting" stops being a vague diagnosis and becomes a specific, measurable statement.

## Why it matters

Every model-selection decision in ML — how complex a model to use, how much to regularize, how much data to collect, whether to add more features — is really a decision about where to sit on the bias-variance tradeoff. Without this framework, "add more parameters" and "just minimize training error" look like unambiguously good ideas; with it, you can predict *in advance* that they aren't.

## Prerequisites

- `05-machine-learning/01-introduction` — supervised learning setup, the idea of a model $f(x;\theta)$ fit from data.
- `05-machine-learning/05-cross-validation` — estimating generalization error from data you already have; this topic uses the same idea (multiple resampled training sets) to estimate bias and variance empirically.
- Basic expectation/variance algebra (statistics fundamentals).

## What you'll build

A from-scratch NumPy experiment: fit polynomials of increasing degree to many independently resampled noisy training sets drawn from the same fixed-seed synthetic function, then empirically compute bias², variance, and test error at each degree — reproducing the classic U-shaped test-error-vs-model-complexity curve from first principles, not from a textbook picture.

## Where it appears in real systems

- Choosing model complexity/capacity (tree depth, polynomial degree, neural network size, number of boosting rounds) is always implicitly a bias-variance decision.
- Regularization (`04-regularization`) is a direct tool for trading variance for bias.
- Cross-validation (`05-cross-validation`) is the practical tool used to *estimate* where a model sits on this tradeoff, since you can't compute bias/variance directly on real data (you don't know the true function).
- "More data helps" and "more data doesn't help" are both true statements depending on whether a model is in a high-variance or high-bias regime — this topic explains why.

## What's next

`04-regularization` (managing the tradeoff by penalizing model complexity) and later model families (`10-decision-tree`, `11-random-forest`, `13-gradient-boosting`) that make bias-variance tradeoffs explicit through hyperparameters like tree depth and ensemble size.
