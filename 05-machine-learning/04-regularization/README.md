# 04 – Regularization (Ridge, Lasso, ElasticNet)

## What you'll learn

Why a flexible model (many features, a high-degree polynomial) needs more than "minimize training error" to generalize well, and how adding a penalty on coefficient size — Ridge's $L_2$ penalty, Lasso's $L_1$ penalty, and ElasticNet's blend of both — controls overfitting continuously instead of through a discrete, combinatorial search over feature subsets. Includes *why* $L_1$ drives coefficients exactly to zero (automatic feature selection) while $L_2$ only shrinks them, derived from the geometry of each penalty's constraint region.

## Why it matters

Regularization is the direct, practical lever for managing the bias-variance tradeoff (`05b-bias-variance-tradeoff`) once a model is already chosen — it's usually cheaper and more reliable than manually picking a smaller feature set or a lower polynomial degree, and it's the standard first response to "training error is low but validation error is high."

## Prerequisites

- `02-linear-regression` — the unregularized least-squares objective this topic adds a penalty term to.
- `05b-bias-variance-tradeoff` — the problem regularization exists to manage.
- `05-cross-validation` — used here to select the penalty strength $\lambda$.

## What you'll build

Ridge, Lasso, and ElasticNet fit from first principles and compared against `sklearn`'s implementations on the Algerian Forest Fires dataset (`algerian-forest.ipynb`, `model-training.ipynb`) — a real, correlated multi-feature dataset where unregularized OLS coefficients become unstable, made concrete rather than illustrated on synthetic data alone.

## Where it appears in real systems

- Nearly every production linear/logistic model ships with $L_1$/$L_2$ regularization on by default (`sklearn`'s `Ridge`, `Lasso`, `ElasticNet`, and the `penalty` parameter on `LogisticRegression`).
- The same $L_2$ idea reappears as **weight decay** in neural network training.
- Feature selection via Lasso's sparsity is a common alternative to manual feature engineering when the true informative subset of features is unknown.

## What's next

`05-cross-validation` for choosing $\lambda$ rigorously (if not already covered), then `06-logistic-regression`, where the same regularization machinery reappears on a classification loss instead of squared error.
