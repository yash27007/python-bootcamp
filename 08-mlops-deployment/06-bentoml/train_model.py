"""
Trains a small, real sklearn model and pickles it to disk — the artifact
both the from-scratch server (server.py) and the BentoML service
(service.py) load and serve. Mirrors the same joblib-pickle pattern used
in 01-docker/train_and_pickle.py and 04-model-packaging-versioning, kept
self-contained here so this topic's demos don't depend on paths outside
their own directory.

Run with:  .venv/bin/python train_model.py
"""

from __future__ import annotations

import joblib
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0, stratify=data.target
)

model = LogisticRegression(max_iter=1000, random_state=0).fit(X_train, y_train)
acc = (model.predict(X_test) == y_test).mean()
print(f"trained LogisticRegression on Iris, test accuracy = {acc:.4f}")

joblib.dump(model, "model.pkl")
print("saved model.pkl")
print(f"target names = {list(data.target_names)}")
print(f"feature names = {list(data.feature_names)}")
