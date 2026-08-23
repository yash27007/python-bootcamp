"""
Tiny, self-contained "training pipeline" that test_pipeline.py exercises.

Deliberately small (Iris, logistic regression) so the whole point — testing
the *shape* of a pipeline, not the sophistication of a model — stays visible.
Nothing here is meant to be a good model; it's meant to be a testable one.
"""

from __future__ import annotations

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


def preprocess(X: np.ndarray) -> np.ndarray:
    """Standardize each feature column to zero mean, unit variance.

    This is the kind of small, pure data-transform function unit tests are
    best at: no randomness, no I/O, a clear input/output contract.
    """
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0  # guard against a constant column
    return (X - mean) / std


def load_fixture_split(random_state: int = 0):
    """Load Iris and return a fixed, seeded train/test split.

    Using a fixed random_state is what makes the regression test in
    test_pipeline.py reproducible instead of flaky.
    """
    data = load_iris()
    X_train, X_test, y_train, y_test = train_test_split(
        data.data, data.target, test_size=0.3, random_state=random_state, stratify=data.target
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: np.ndarray, y_train: np.ndarray, random_state: int = 0) -> LogisticRegression:
    """Train a logistic regression classifier on already-preprocessed features."""
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(preprocess(X_train), y_train)
    return model


def evaluate(model: LogisticRegression, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """Return accuracy of `model` on a preprocessed test set."""
    preds = model.predict(preprocess(X_test))
    return float((preds == y_test).mean())
