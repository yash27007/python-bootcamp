"""
Minimal model-serving HTTP endpoint, built from Python's stdlib
`http.server` only — no Flask, no BentoML, no framework. Demonstrates the
actual mechanism a serving framework automates: an HTTP server that stays
running, loads the model once at startup, and on each request runs
preprocess -> predict -> postprocess -> JSON response.

Endpoint:
  POST /predict
  body: {"features": [5.1, 3.5, 1.4, 0.2]}
  response: {"prediction": "setosa", "prediction_index": 0,
             "probabilities": {...}}

Run with:  .venv/bin/python server.py
Then in another process:  .venv/bin/python client_demo.py
"""

from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import joblib

MODEL = joblib.load("model.pkl")
TARGET_NAMES = ["setosa", "versicolor", "virginica"]
N_FEATURES = 4


def preprocess(payload: dict) -> list[float]:
    """Validate and coerce the raw JSON body into the shape .predict() needs.

    This is the exact step that goes silently missing when someone "just
    calls .predict() from a script" — see notes.md's Why-simpler-fails and
    Failure-modes sections.
    """
    if "features" not in payload:
        raise ValueError("missing required field 'features'")
    features = payload["features"]
    if not isinstance(features, list) or len(features) != N_FEATURES:
        raise ValueError(f"'features' must be a list of {N_FEATURES} numbers")
    try:
        return [float(x) for x in features]
    except (TypeError, ValueError) as exc:
        raise ValueError("all elements of 'features' must be numeric") from exc


def postprocess(pred_index: int, probabilities: list[float]) -> dict:
    return {
        "prediction": TARGET_NAMES[pred_index],
        "prediction_index": pred_index,
        "probabilities": dict(zip(TARGET_NAMES, probabilities)),
    }


class PredictHandler(BaseHTTPRequestHandler):
    def log_message(self, format, *args):  # quieter test output
        pass

    def _send_json(self, status: int, body: dict) -> None:
        payload = json.dumps(body).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_POST(self):
        if self.path != "/predict":
            self._send_json(404, {"error": "not found"})
            return

        length = int(self.headers.get("Content-Length", 0))
        raw = self.rfile.read(length)

        try:
            payload = json.loads(raw)
        except json.JSONDecodeError:
            self._send_json(400, {"error": "malformed JSON body"})
            return

        try:
            features = preprocess(payload)
        except ValueError as exc:
            self._send_json(400, {"error": str(exc)})
            return

        pred_index = int(MODEL.predict([features])[0])
        probabilities = MODEL.predict_proba([features])[0].tolist()
        self._send_json(200, postprocess(pred_index, probabilities))


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8000), PredictHandler)
    print("serving on http://127.0.0.1:8000 (POST /predict) ... Ctrl+C to stop")
    server.serve_forever()
