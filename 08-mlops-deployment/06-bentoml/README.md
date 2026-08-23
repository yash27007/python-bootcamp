# 06 – BentoML

## What you'll learn

Why a trained, versioned, tracked model (`04`/`05`) still isn't something
another program or person can *use* — it's a file that only means
something inside the exact Python process that trained it. Model serving
solves this by wrapping `.predict()` behind a network API: a small,
always-running program implementing request -> preprocess -> predict ->
postprocess -> response, built from scratch first (Python's stdlib
`http.server`, no framework) and then with the real `bentoml` client.

| Topic | Status |
|-------|--------|
| Building Data Science Projects With BentoML | ✅ Complete |

## Why it matters

"Just import the model and call `.predict()`" works for a single script
run by the person who trained it. It breaks the moment anything else needs
a prediction: a different language/process can't `import joblib`, there's
no concurrency handling for multiple simultaneous callers, no validation
of untrusted input, and no versioned API contract callers can depend on.
Serving is the layer that turns a model artifact into something callable
from anywhere, over HTTP.

## Prerequisites

- `04-model-packaging-versioning` — this topic serves the exact kind of
  artifact that section produces (a pickled model plus its metadata); the
  from-scratch server here loads a model saved the same way.
- `05-mlflow-dagshub` — a tracked, registered model is what typically gets
  promoted to serving in a real pipeline; this topic picks up where that
  one leaves off.
- Basic Flask/HTTP familiarity is useful but not required — the
  from-scratch section uses only Python's stdlib `http.server`.

## What you'll build

- `train_model.py` — trains and pickles a real `LogisticRegression` on
  Iris, the artifact both demos below serve.
- `server.py` — a from-scratch HTTP model-serving endpoint using only
  Python's stdlib `http.server`/`ThreadingHTTPServer`: loads the pickled
  model once at startup, validates and parses each `POST /predict` body,
  runs `.predict()`/`.predict_proba()`, and returns JSON. Actually run as
  a background process and actually hit with real requests (via
  `requests` and `curl`), including deliberately malformed input to show
  the validation working — real captured output in `notes.md`.
- `service.py` — the practical counterpart: a real BentoML
  `bentoml.Service` (`@bentoml.service` + `@bentoml.api`) wrapping the
  same model. `uv add bentoml` installed cleanly in this environment, so
  this was **actually run** with `bentoml serve` and actually hit with a
  real request — captured output (including a real `500` from an
  unvalidated malformed request, contrasted with the from-scratch
  server's clean `400`) is in `notes.md`.
- A real load-test (`client_demo.py`): 20 sequential requests against the
  from-scratch endpoint with measured latency, hypothesis stated first.

## Where it appears in real systems

- Every production ML system that answers requests in real time (fraud
  checks, recommendations, an image classifier behind an upload button)
  is this same pipeline underneath, usually via a framework like BentoML,
  TensorFlow Serving, or a cloud provider's managed endpoint instead of
  hand-rolled `http.server` code.
- Training/serving skew — preprocessing done differently at train time
  versus serve time — is one of the most consequential, hardest-to-detect
  failure modes in real ML systems, precisely because it degrades
  predictions silently with no crash and no obviously wrong offline
  metric. See `notes.md`'s Failure modes section.
- BentoML's packaging format (a "Bento") and its integration with model
  registries connects directly back to `04`'s and `05`'s registries — the
  serving step this topic builds is the natural next stage after a model
  is trained, packaged, and tracked.

## What's next

This is the last topic in `08-mlops-deployment`'s current build-out —
serving closes the loop from raw training code to a model another system
can actually call.
