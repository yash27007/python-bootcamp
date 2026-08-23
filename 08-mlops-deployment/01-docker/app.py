"""
app.py -- minimal Flask model-serving app.

Loads the pickled sklearn model produced by train_and_pickle.py and serves
one prediction per POST request to /predict. This is the "code" half of
the containerized system that the Dockerfile in this folder packages up;
it deliberately does nothing fancy so the Dockerfile stays the focus.
"""

import pickle

from flask import Flask, jsonify, request

app = Flask(__name__)

with open("model.pkl", "rb") as f:
    MODEL = pickle.load(f)

IRIS_CLASSES = ["setosa", "versicolor", "virginica"]


@app.get("/health")
def health():
    return jsonify(status="ok")


@app.post("/predict")
def predict():
    """Expects JSON: {"features": [sepal_length, sepal_width, petal_length, petal_width]}"""
    payload = request.get_json(force=True)
    features = payload["features"]
    prediction = MODEL.predict([features])[0]
    return jsonify(
        prediction=int(prediction),
        class_name=IRIS_CLASSES[int(prediction)],
    )


if __name__ == "__main__":
    # 0.0.0.0 so the port is reachable from outside the container, not just
    # from inside it.
    app.run(host="0.0.0.0", port=5000)
