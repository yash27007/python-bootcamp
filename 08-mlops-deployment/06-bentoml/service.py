"""
The practical/production version of server.py, using BentoML instead of a
hand-rolled http.server. Same model, same request/response shape, same
preprocess -> predict -> postprocess pipeline — just expressed through a
framework that adds what the from-scratch version doesn't have out of the
box: input schema validation via typed signatures, automatic OpenAPI docs,
adaptive request batching, and a standard packaging format (a "Bento") for
deployment.

Run with:  .venv/bin/bentoml serve service:IrisClassifier --reload
Then:      curl -X POST http://127.0.0.1:3000/predict \
             -H "Content-Type: application/json" \
             -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

See notes.md's "Practical implementation" section for whether this file
was actually executed in this environment.
"""

from __future__ import annotations

import joblib
import numpy as np

import bentoml

TARGET_NAMES = ["setosa", "versicolor", "virginica"]
N_FEATURES = 4


@bentoml.service(name="iris_classifier")
class IrisClassifier:
    """Wraps the same joblib-pickled LogisticRegression trained by
    train_model.py behind a BentoML service — the direct practical
    counterpart to PredictHandler in server.py."""

    def __init__(self) -> None:
        # Loaded once per worker at startup, exactly like MODEL in
        # server.py — this is the "load the model once, not per request"
        # idea BentoML shares with the from-scratch version.
        self.model = joblib.load("model.pkl")

    @bentoml.api
    def predict(self, features: list[float]) -> dict:
        """POST /predict — body: {"features": [f1, f2, f3, f4]}."""
        if len(features) != N_FEATURES:
            raise ValueError(f"'features' must have exactly {N_FEATURES} elements")

        x = np.array([features], dtype=float)
        pred_index = int(self.model.predict(x)[0])
        probabilities = self.model.predict_proba(x)[0].tolist()

        return {
            "prediction": TARGET_NAMES[pred_index],
            "prediction_index": pred_index,
            "probabilities": dict(zip(TARGET_NAMES, probabilities)),
        }
