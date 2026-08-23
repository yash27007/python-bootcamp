"""
Trains two versions of a toy sklearn model, registers both through
registry.py under the SAME requested filename ("model.pkl") — deliberately
recreating the `model_v1.pkl`, `model_v2_final.pkl` naming problem from
notes.md, except the registry never relies on the filename — and shows the
registry log correctly distinguishing the two versions by content hash.

Run with:  .venv/bin/python train_and_register.py
"""

from __future__ import annotations

import json

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from registry import load_log, load_model_by_hash, register

data = load_iris()
X_train, X_test, y_train, y_test = train_test_split(
    data.data, data.target, test_size=0.3, random_state=0, stratify=data.target
)


def accuracy(model) -> float:
    preds = model.predict(X_test)
    return float((preds == y_test).mean())


print("=== Training version A: LogisticRegression(C=1.0) ===")
model_a = LogisticRegression(C=1.0, max_iter=1000, random_state=0).fit(X_train, y_train)
acc_a = accuracy(model_a)
print(f"accuracy = {acc_a:.4f}")

meta_a = register(
    model_a,
    requested_name="model.pkl",  # <-- same requested filename as version B
    metric_name="accuracy",
    metric_value=acc_a,
    extra_metadata={"hyperparameters": {"C": 1.0}, "training_data": "sklearn.datasets.load_iris (random_state=0 split)"},
)
print(f"registered as {meta_a['model_file']} (hash {meta_a['short_hash']})\n")


print("=== Training version B: LogisticRegression(C=0.01) ===")
model_b = LogisticRegression(C=0.01, max_iter=1000, random_state=0).fit(X_train, y_train)
acc_b = accuracy(model_b)
print(f"accuracy = {acc_b:.4f}")

meta_b = register(
    model_b,
    requested_name="model.pkl",  # <-- filename collides with version A on purpose
    metric_name="accuracy",
    metric_value=acc_b,
    extra_metadata={"hyperparameters": {"C": 0.01}, "training_data": "sklearn.datasets.load_iris (random_state=0 split)"},
)
print(f"registered as {meta_b['model_file']} (hash {meta_b['short_hash']})\n")


print("=== Registry log (both entries, same requested_name, different hash) ===")
log = load_log()
for entry in log:
    print(json.dumps({
        "requested_name": entry["requested_name"],
        "short_hash": entry["short_hash"],
        "model_file": entry["model_file"],
        "metric_value": entry["metric_value"],
        "hyperparameters": entry.get("hyperparameters"),
    }, indent=2))

assert meta_a["hash"] != meta_b["hash"], "expected different content hashes for different models"
assert meta_a["model_file"] != meta_b["model_file"], "expected different storage files despite same requested_name"
print("\nCONFIRMED: identical requested_name ('model.pkl') for both versions did NOT cause a collision —")
print(f"they are stored as distinct files ({meta_a['model_file']} vs {meta_b['model_file']}) because their content hashes differ.")


print("\n=== Loading version A back by hash and verifying integrity ===")
reloaded_a = load_model_by_hash(meta_a["short_hash"])
reloaded_preds_match = (reloaded_a.predict(X_test) == model_a.predict(X_test)).all()
print(f"reloaded model predictions match original: {reloaded_preds_match}")
