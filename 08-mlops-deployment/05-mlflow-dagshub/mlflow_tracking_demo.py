"""Practical implementation: the same experiment-tracking idea as
json_run_logger.py, using MLflow's real tracking client against a local
SQLite tracking store (mlflow.db). Logs several real LogisticRegression
training runs on the breast-cancer dataset, varying `C`, matching
json_run_logger.py's from-scratch demo so the two can be compared directly.

Run directly: .venv/bin/python mlflow_tracking_demo.py
Inspect afterwards with: .venv/bin/mlflow ui --backend-store-uri sqlite:///mlflow.db
"""

from pathlib import Path

import mlflow
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

HERE = Path(__file__).parent
mlflow.set_tracking_uri(f"sqlite:///{HERE / 'mlflow.db'}")
mlflow.set_experiment("breast-cancer-logistic-regression")

X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler().fit(X_train)
X_train, X_test = scaler.transform(X_train), scaler.transform(X_test)

print("=== Logging real training runs to MLflow (sqlite:///mlflow.db) ===")
results = []
for C in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
    with mlflow.start_run(run_name=f"logreg-C={C}") as run:
        model = LogisticRegression(C=C, max_iter=2000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        mlflow.log_param("C", C)
        mlflow.log_param("model_type", "LogisticRegression")
        mlflow.log_param("max_iter", 2000)
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)

        results.append((run.info.run_id, C, acc, f1))
        print(f"run_id={run.info.run_id} C={C:<8} accuracy={acc:.4f} f1={f1:.4f}")

print("\n=== Querying logged runs back via MlflowClient ===")
client = mlflow.tracking.MlflowClient()
experiment = client.get_experiment_by_name("breast-cancer-logistic-regression")
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
)
print(f"{'run_id':<34}{'C':<10}{'accuracy':<12}{'f1_score':<10}")
for r in runs:
    print(
        f"{r.info.run_id:<34}"
        f"{r.data.params.get('C'):<10}"
        f"{float(r.data.metrics.get('accuracy')):<12.4f}"
        f"{float(r.data.metrics.get('f1_score')):<10.4f}"
    )

best = runs[0]
print(f"\nBest run by accuracy: C={best.data.params.get('C')} accuracy={best.data.metrics.get('accuracy'):.4f}")
