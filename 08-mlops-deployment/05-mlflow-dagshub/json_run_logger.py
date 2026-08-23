"""From-scratch experiment tracker: append one JSON record per run to a
JSONL (JSON Lines) file. This is the core idea underneath MLflow's tracking
store -- a structured, queryable log of params/metrics/timestamp keyed by a
run ID -- with none of MLflow's server, UI, or artifact storage.

Run directly: .venv/bin/python json_run_logger.py
"""

import json
import time
import uuid
from pathlib import Path

from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

LOG_PATH = Path(__file__).parent / "runs.jsonl"


def log_run(params: dict, metrics: dict) -> str:
    """Append one run record and return its run_id. This single function
    is the entire "tracking store": open the file in append mode, write one
    JSON object per line, close it. No index, no query engine, no server."""
    run_id = uuid.uuid4().hex[:12]
    record = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": params,
        "metrics": metrics,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return run_id


def query_runs(min_accuracy: float | None = None) -> list[dict]:
    """The one "query" this format supports without an external tool:
    read every line, parse it, filter in Python. Demonstrates why this is
    fine for a handful of runs and unworkable for thousands (see notes.md
    Failure modes)."""
    if not LOG_PATH.exists():
        return []
    runs = []
    with open(LOG_PATH) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if min_accuracy is None or record["metrics"].get("accuracy", 0) >= min_accuracy:
                runs.append(record)
    return runs


if __name__ == "__main__":
    # Start from a clean log each time this demo script runs, so the
    # printed output below is reproducible and not appended to stale runs
    # from a previous execution.
    if LOG_PATH.exists():
        LOG_PATH.unlink()

    X, y = load_breast_cancer(return_X_y=True)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

    print("=== Logging 3 real training runs to a plain JSONL file ===")
    for C in (0.01, 1.0, 100.0):
        model = LogisticRegression(C=C, max_iter=2000)
        model.fit(X_train, y_train)
        acc = accuracy_score(y_test, model.predict(X_test))
        run_id = log_run(params={"C": C, "model": "LogisticRegression"}, metrics={"accuracy": acc})
        print(f"logged run {run_id}: C={C} accuracy={acc:.4f}")

    print(f"\n=== Raw contents of {LOG_PATH.name} ===")
    print(LOG_PATH.read_text())

    print("=== Querying runs with accuracy >= 0.97 ===")
    for run in query_runs(min_accuracy=0.97):
        print(run)
