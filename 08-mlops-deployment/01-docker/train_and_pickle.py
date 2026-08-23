"""
train_and_pickle.py -- trains a tiny sklearn model and pickles it to
model.pkl, so the Dockerfile in this folder has a real artifact to COPY
and app.py has a real model to load and serve.

Run once, outside the container, before building the image:
    .venv/bin/python train_and_pickle.py
"""

import pickle

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression

X, y = load_iris(return_X_y=True)
model = LogisticRegression(max_iter=200).fit(X, y)

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Trained LogisticRegression on Iris, saved to model.pkl")
print(f"Train accuracy: {model.score(X, y):.4f}")
