# 01 – Introduction to ML

## What you'll learn

What "machine learning" actually means as a formal problem — given examples $(x_i, y_i)$, find a function $f$ that fits them *and* generalizes to inputs never seen before — and the two fundamentally different strategies for finding that function: **instance-based learning** (compare a new point to stored examples, e.g. k-nearest-neighbors) and **model-based learning** (fit a compact set of parameters up front, e.g. a line through the data). Also covers the mathematical language a model-based rule is expressed in — a hyperplane $\mathbf{w}^T\mathbf{x} + b = 0$ — which every supervised method in this section builds on.

## Why it matters

This topic sets the vocabulary and mental model for the rest of `05-machine-learning`. Every algorithm that follows is one more way of finding $f$ — a different assumption about its form, a different search procedure, a different tradeoff — and "instance-based vs. model-based" is the first fork in that tree (KNN sits on one side, linear/logistic regression and everything downstream of a weight vector sits on the other).

## Prerequisites

- Section `01-python-foundation` and `02-statistics` — comfort with NumPy arrays and basic vector/linear algebra.
- No prior ML topic required — this is the entry point.

## What you'll build

A from-scratch geometric comparison of instance-based and model-based prediction on the same toy dataset (`instance-vs-model-based-and-geometry.ipynb`), visualizing what a hyperplane decision boundary actually looks like and how it differs from a nearest-neighbor vote.

## Where it appears in real systems

Every supervised model in production is, underneath its library API, either doing a live comparison against stored data (recommendation systems, similarity search, KNN-based fraud checks) or evaluating a fixed set of learned parameters (essentially everything else — regression, logistic regression, trees, boosted ensembles, neural networks). Recognizing which category a new algorithm falls into is the fastest way to guess its strengths, weaknesses, and computational cost before reading a single line of its documentation.

## What's next

`02-linear-regression` — the simplest model-based method, and the first place the hyperplane from this topic gets a concrete learning algorithm (gradient descent / closed-form least squares) attached to it.
