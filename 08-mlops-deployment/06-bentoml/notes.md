# Model serving with BentoML

## Problem

By the end of `04-model-packaging-versioning` and `05-mlflow-dagshub`, a
trained model is a well-identified, versioned, tracked artifact: a
content-addressed `.pkl` (or an MLflow-registered model) with metadata
recording exactly what produced it. That solves *finding and trusting* the
right model. It does not solve a completely different problem: **that
artifact still isn't something another program, another team's service, or
a person outside the training environment can actually use.** It's a file
that requires a Python process with the exact right libraries installed,
sitting on the same machine, to do anything with. A recommendation engine
written in Go, a mobile app, a teammate's notebook on a different laptop,
or a nightly batch job written by someone who has never seen this repo all
need to get a prediction out of this model — and right now none of them
can, because the only interface to it is "import this pickle file in
Python." This topic answers: **how do you turn a trained model into
something other software can call, reliably, regardless of what language
or process it's calling from?**

## Intuition

Think about how every other reusable piece of software gets consumed once
it's built. A weather forecasting model doesn't hand out its internal
`.pkl` file to every app that wants tomorrow's forecast — it sits behind
`api.weather.example.com/forecast?city=...`, and any client that can make
an HTTP request, in any language, gets a JSON answer back. The model itself
never left its own server; only the *inputs and outputs* crossed the
network. That's the entire idea of model serving: wrap `model.predict()`
in a small program that (1) stays running, (2) listens on a network port,
(3) turns each incoming request into the exact input shape `.predict()`
expects, (4) turns the raw prediction back into a response the caller can
use, and (5) sends that response back over the same connection. Nothing
about the *model* changes — `.predict()` is still `.predict()`. What
changes is that it's now reachable by anything that speaks HTTP, not just
by whatever Python process happens to have that pickle file on disk.

## Why simpler approaches fail

The instinctive simplest approach is: "just `import joblib; model =
joblib.load('model.pkl')` and call `model.predict(x)` in whatever script
needs a prediction." This works for the exact demos in
`04-model-packaging-versioning` — a single Python script, run once, on one
machine, by one person who wrote it. It breaks down the moment "whatever
needs a prediction" is anything else:

- **Cross-language, cross-process boundaries.** A `.pkl` file is a
  Python-and-library-version-specific serialization. A Go backend, a
  browser, or a mobile app cannot `import joblib` and load it — there is
  no code path from "arbitrary caller" to "prediction" that doesn't
  involve some kind of network boundary, because that's the only interface
  every language and platform shares.
- **No concurrency handling.** A bare `model.predict(x)` call inside a
  request-handling function that's invoked from multiple threads or
  processes at once has no defined behavior for "what happens when two
  callers ask at the same time" — a real serving layer has to decide
  whether requests share a model instance, whether predictions are batched
  together, and how many can run concurrently, none of which "import and
  call" addresses at all.
- **No input validation.** A script written by the person who trained the
  model can be trusted to pass it well-formed input, because they wrote
  both sides. An external caller sending a raw JSON body cannot be
  trusted the same way — wrong number of features, wrong types, missing
  fields, or malicious payloads are all now on the table, and
  `model.predict()` itself does nothing to protect against any of it (see
  Failure modes and the from-scratch demo below).
- **No versioned API contract.** "Call `.predict()` on whatever object
  happens to be named `model`" has no stable, documented shape — if the
  model's expected input changes (a feature added, reordered, or dropped),
  every caller silently breaks with no warning, because there was never a
  contract to break in the first place. A real API endpoint has a schema
  (implicit or explicit) that callers can depend on and that changes are
  visible against.

## Conceptual foundation

This section is titled "Conceptual foundation" rather than "Mathematical
foundation," following the same substitution made in
`04-model-packaging-versioning`: model *serving* has no equation to
derive — the model's own math was already covered wherever it was trained
(`05-machine-learning`, `06-deep-learning`). What's foundational here is a
conceptual pipeline and a set of systems tradeoffs, not a formula.

### The serving pipeline

Every model-serving endpoint, from the from-scratch version below to a
production BentoML/TensorFlow-Serving/TorchServe deployment, implements
the same four-stage pipeline for each request:

```
request  ->  preprocess  ->  predict  ->  postprocess  ->  response
```

- **Request**: raw bytes arrive over the network (typically a JSON body
  over HTTP). Nothing about this stage knows or cares what language the
  caller was written in.
- **Preprocess**: the raw request is validated and transformed into
  *exactly* the numeric shape the model's `.predict()` expects (a NumPy
  array of the right shape and dtype, features in the right order). This
  is the stage "just call `.predict()`" skips entirely, and skipping it is
  what causes training/serving skew (Failure modes).
- **Predict**: the actual model inference — the one line of code that was
  the entire simple approach above. Everything else in this topic exists
  to make this one line safely and reliably reachable from outside the
  training environment.
- **Postprocess**: the raw model output (a class index, a probability
  vector, a regression float) is turned into something meaningful to the
  caller — a class label instead of `2`, a rounded confidence, a
  structured JSON object instead of a bare NumPy array (which isn't even
  JSON-serializable without conversion).
- **Response**: the postprocessed result is serialized back over the same
  connection the request arrived on.

### Batching and concurrency: what's different from a script call

A one-off script calls `.predict()` once, gets an answer, exits. A serving
endpoint stays running and may receive many requests concurrently, which
introduces two considerations a script never faces:

- **Concurrency**: can multiple requests be handled at the same time, and
  does the underlying model object tolerate being called from multiple
  threads simultaneously? The from-scratch demo below uses
  `ThreadingHTTPServer` specifically so it can accept a next connection
  while a previous one is still being handled — a plain `HTTPServer`
  would serialize every request, one at a time, which behaves fine for a
  demo but not for real concurrent load.
- **Batching**: most ML inference (especially anything running on a GPU)
  is far more efficient per-example when many inputs are predicted in one
  call than one at a time — a model call with a batch of 32 inputs is
  typically much faster than 32 separate calls with a batch of 1. A
  serving framework can exploit this by briefly collecting several
  requests that arrive close together and running them through
  `.predict()` as a single batch before splitting the results back out to
  each caller — a "one-off script call" never needs this because it only
  ever has one input in hand at a time. The from-scratch server below does
  *not* implement batching (each request is predicted individually,
  batch size 1) precisely to keep the from-scratch version honest about
  what it doesn't do; BentoML implements adaptive batching automatically
  for services that opt in.

## Algorithm

The serving pipeline as concrete steps, matching `server.py`'s
`PredictHandler.do_POST`:

1. Load the model once, at process startup, into a module-level variable —
   never inside the per-request handler (loading a pickle file is
   comparatively slow; doing it on every request would make cold, wasteful
   work out of every single call).
2. On each incoming `POST /predict`, read the request body and parse it as
   JSON. If parsing fails, respond `400` immediately — never let a
   downstream stage receive un-parseable input.
3. **Preprocess**: validate the parsed JSON has the required `features`
   key, that it's a list of the expected length, and that every element
   coerces to `float`. Any violation responds `400` with a specific error
   message and stops — the model is never called with malformed input.
4. **Predict**: call `model.predict([features])` (and, here,
   `model.predict_proba` for the confidence scores) on the validated,
   correctly-shaped input.
5. **Postprocess**: map the numeric prediction index back to a human-
   readable class name, and pair each class name with its probability.
6. **Response**: serialize the result dict to JSON, set
   `Content-Type: application/json`, and write it back to the client with
   a `200` status.

## From-scratch implementation

`train_model.py` trains and pickles a real `LogisticRegression` on Iris
(test accuracy 1.0000 on this split — see the executed output below), the
same artifact both this section's server and the BentoML service in
Practical implementation load.

`server.py` is the actual from-scratch endpoint — Python's stdlib
`http.server` only, no Flask, no BentoML:

```python
MODEL = joblib.load("model.pkl")          # loaded once, at startup

def preprocess(payload: dict) -> list[float]:
    if "features" not in payload:
        raise ValueError("missing required field 'features'")
    features = payload["features"]
    if not isinstance(features, list) or len(features) != N_FEATURES:
        raise ValueError(f"'features' must be a list of {N_FEATURES} numbers")
    return [float(x) for x in features]

class PredictHandler(BaseHTTPRequestHandler):
    def do_POST(self):
        raw = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        payload = json.loads(raw)                     # 400 on failure
        features = preprocess(payload)                 # 400 on failure
        pred_index = int(MODEL.predict([features])[0])
        probabilities = MODEL.predict_proba([features])[0].tolist()
        self._send_json(200, postprocess(pred_index, probabilities))
```

**Actually run.** The server was started as a background process
(`.venv/bin/python server.py`, listening on `127.0.0.1:8000`) and hit with
real requests from `client_demo.py` (using the `requests` library) and
`curl` in a separate process. Real captured output:

```
=== Single real request/response ===
request:  POST /predict {'features': [5.1, 3.5, 1.4, 0.2]}
status:   200
response: {'prediction': 'setosa', 'prediction_index': 0, 'probabilities':
  {'setosa': 0.9755857838256975, 'versicolor': 0.024413990613541763,
   'virginica': 2.2556076087795102e-07}}

=== Malformed request (Failure modes: input validation) ===
request:  POST /predict {'features': [1, 2]}
status:   400
response: {'error': "'features' must be a list of 4 numbers"}
```

Two additional real `curl` requests confirmed the same validation holds
for completely non-numeric input and for a non-JSON body:

```
$ curl -X POST http://127.0.0.1:8000/predict -d 'not json'
{"error": "malformed JSON body"}                                   # 400

$ curl -X POST http://127.0.0.1:8000/predict \
       -d '{"features": ["a","b","c","d"]}'
{"error": "all elements of 'features' must be numeric"}            # 400
```

In every case the server stayed up and responded with a clear `400` —
never a `500` or a hung connection — because `preprocess()` runs *before*
`.predict()` is ever called.

## Practical implementation

`service.py` is the direct BentoML counterpart to `server.py` — same
model, same request/response shape, expressed through the framework
instead of stdlib `http.server`:

```python
@bentoml.service(name="iris_classifier")
class IrisClassifier:
    def __init__(self) -> None:
        self.model = joblib.load("model.pkl")      # loaded once, like MODEL above

    @bentoml.api
    def predict(self, features: list[float]) -> dict:
        if len(features) != N_FEATURES:
            raise ValueError(f"'features' must have exactly {N_FEATURES} elements")
        x = np.array([features], dtype=float)
        pred_index = int(self.model.predict(x)[0])
        probabilities = self.model.predict_proba(x)[0].tolist()
        return {"prediction": TARGET_NAMES[pred_index], ...}
```

The mapping back to the from-scratch version is direct: `@bentoml.service`
replaces the manually-constructed `ThreadingHTTPServer`; `__init__` is the
same "load once at startup" step as the module-level `MODEL = ...`;
`@bentoml.api` replaces the hand-written `do_POST` routing and JSON
parsing/serialization (BentoML infers the request/response schema from the
method's Python type hints and generates an OpenAPI spec from it — the
from-scratch version has no such schema beyond what `preprocess` checks by
hand); the function body is, deliberately, almost identical, because the
*serving pipeline* (preprocess -> predict -> postprocess) doesn't change —
only who's responsible for the HTTP plumbing around it does.

**Execution status: this WAS actually run in this environment.**
`uv add bentoml` installed cleanly (`bentoml==1.4.39` plus its
dependencies — no build failures, no incompatible pins with the rest of
this repo's `pyproject.toml`). The service was started with:

```
.venv/bin/bentoml serve service:IrisClassifier
```

and produced real, captured log output:

```
[INFO] [cli] Starting production HTTP BentoServer from "service:IrisClassifier"
      listening on http://localhost:3000 (Press CTRL+C to quit)
[INFO] [entry_service:iris_classifier:1] Service iris_classifier initialized
```

A real request from a separate process:

```
$ curl -X POST http://127.0.0.1:3000/predict \
       -H "Content-Type: application/json" \
       -d '{"features": [5.1, 3.5, 1.4, 0.2]}'

{"prediction": "setosa", "prediction_index": 0,
 "probabilities": {"setosa": 0.9755857838256975,
 "versicolor": 0.024413990613541763,
 "virginica": 2.2556076087795102e-07}}
```

Identical prediction and probabilities to the from-scratch server, as
expected — same model, same input, same math; only the serving layer
differs. See Failure modes below for what happened when this same
malformed-input request that `server.py` handled gracefully (`400`) was
sent to this BentoML service instead.

## Experiment

**Hypothesis:** a from-scratch HTTP endpoint running locally, serving a
tiny sklearn model, should have low, fairly consistent per-request latency
dominated by fixed request/response overhead (HTTP parsing, JSON
encode/decode) rather than by the model computation itself — `predict()`
on a 4-feature logistic regression is a handful of floating-point
multiplications, essentially free compared to the surrounding HTTP
round-trip.

**Setup**: with `server.py` running, `client_demo.py` sent 20 sequential
real POST requests (cycling through three different feature vectors),
timing each with `time.perf_counter()` around the `requests.post()` call
(so the measurement includes real network stack + HTTP + JSON overhead,
not just server-side compute).

**Actual result:**

```
requests sent: 20
min latency:    0.767 ms
median latency: 1.023 ms
max latency:    1.353 ms
mean latency:   1.041 ms
```

**Interpretation:** consistent with the hypothesis — latency is small
(sub-2ms) and tightly clustered (min to max spans well under a
millisecond), with no long-tail outliers across 20 sequential requests.
This is the expected shape for a trivial model served locally with no
network hop: the cost is essentially fixed per-request overhead (loopback
TCP, HTTP parsing, `json.dumps`/`json.loads`), not model computation.

**Limitations:** this measures a *local loopback* server with sequential
(not concurrent) requests and a single, tiny model — it says nothing about
latency under real concurrent load, over a real network with non-trivial
round-trip time, or for a model whose `.predict()` call is itself
expensive (a large neural network, for instance, where inference compute
would dominate over HTTP overhead instead of the reverse). It also doesn't
measure cold-start latency (see Failure modes) since the server was
already warm for all 20 requests.

## Failure modes

- **Training/serving skew — one of the most important ML-specific failure
  modes.** If the preprocessing applied at training time (feature scaling,
  categorical encoding, feature ordering, handling of missing values)
  isn't applied *identically* at serving time, the model receives inputs
  that don't match the distribution it was trained on — and it will still
  return a confident-looking prediction, because nothing about calling
  `.predict()` on wrong-but-numerically-valid input raises an error. This
  is silent and often goes undetected for a long time: accuracy just
  quietly degrades in production while every offline metric from training
  still looks fine, because the offline evaluation used the same
  (correct) preprocessing the model was trained on. Concretely, in this
  topic's demo: if a caller sent Iris features in centimeters while the
  training data (hypothetically) had been standardized to
  zero-mean/unit-variance before fitting, `MODEL.predict()` would still
  return *a* class with *a* probability — it has no way to know the input
  distribution is wrong. The only real defenses are (a) making
  preprocessing a single shared piece of code/artifact used by both the
  training pipeline and the serving endpoint, never reimplemented twice
  (this repo's `preprocess()` in `server.py` and the equivalent step in
  `service.py` are simple enough to eyeball for consistency with training,
  but a real pipeline should package a `Pipeline` or transformer object
  alongside the model, exactly to prevent training and serving code from
  drifting apart independently), and (b) monitoring the *distribution* of
  incoming production features against the training distribution, not
  just the model's own accuracy.
- **No input validation lets malformed requests crash the service.**
  `server.py`'s `preprocess()` exists specifically to prevent this — every
  malformed request tested above returned a clean `400`, never a crash.
  The BentoML service demonstrates the *opposite* case concretely: sending
  it `{"features": [1, 2]}` (wrong length) raised a Python `ValueError`
  inside `predict()`, which BentoML caught and turned into a generic
  `500 {"error": "An unexpected error has occurred, please check the
  server log."}` — a real, captured failure from this environment. The
  process itself didn't crash (BentoML's runtime caught the exception
  around the handler), but the caller got an opaque `500` with no
  actionable message, and the real cause is buried in a full Python
  traceback in the server log rather than a specific, structured error
  response. This is exactly what happens when validation is left to
  "whatever a Python exception happens to do" instead of an explicit
  precondition check with a clear error message returned to the caller —
  contrast with `server.py`'s explicit `400 {"error": "'features' must be
  a list of 4 numbers"}`.
- **Cold-start latency.** The first request after a serving process
  starts (or after it's been scaled to zero and a new instance spins up,
  as in serverless/autoscaled deployments) pays a one-time cost this
  topic's Experiment section deliberately excluded: loading the model
  file from disk, initializing the framework's runtime, and (for larger
  models) moving weights onto a GPU. For the tiny Iris model here that
  cost is small, but for a large deep learning model it can be seconds to
  tens of seconds — long enough that a caller hitting a freshly-started
  instance sees dramatically higher latency than the steady-state numbers
  measured above. Production serving systems address this by keeping a
  minimum number of "warm" instances running rather than scaling fully to
  zero, at the direct cost of paying for idle compute between requests.

## Real-world usage

- Every production ML system that isn't purely batch/offline needs some
  form of this: a recommendation service, a fraud-detection check on a
  transaction, an image classifier behind a mobile app upload button — all
  of them are, underneath, this same request -> preprocess -> predict ->
  postprocess -> response pipeline, usually wrapped by a framework like
  BentoML, TensorFlow Serving, TorchServe, or a cloud provider's managed
  endpoint rather than hand-rolled `http.server` code.
- BentoML specifically adds, on top of the pipeline every serving system
  needs: adaptive batching (grouping concurrent requests into one model
  call automatically), a standard packaging format (a "Bento" bundles the
  service code, model artifact, and dependencies together for consistent
  deployment), and integration with model registries like the one built in
  `04-model-packaging-versioning` and MLflow's registry in
  `05-mlflow-dagshub`.
- The training/serving skew failure mode above is precisely why real
  systems invest in *feature stores* and shared preprocessing pipelines —
  infrastructure whose entire purpose is making sure the transformation
  applied to a feature at training time is, by construction, the exact
  same code path applied at serving time, not two independent
  reimplementations that can silently drift apart.

## Mental model

**A trained model is a function nobody outside its own process can call.
Serving is the thin, always-running program that gives that function a
network address — wrapping preprocess -> predict -> postprocess behind a
request/response contract so that "can this model make a prediction" stops
depending on which language, process, or machine is asking.**

## Questions to think about

1. `server.py`'s `preprocess()` checks that `features` is a list of
   exactly 4 numeric values, but does *not* check that those values fall
   within any sensible range (e.g. a negative petal length). Is that a
   gap in "input validation," or is it a different problem entirely (and
   if so, which failure mode does it belong to)?
2. The BentoML service returned an opaque `500` for the same malformed
   input `server.py` handled with a specific `400`. Sketch what you'd add
   to `service.py`'s `predict()` to give BentoML callers the same clear,
   structured error message `server.py` already gives — where exactly
   would that check need to go?
3. This topic's Experiment measured latency for *sequential* requests
   only. If you sent 20 requests *concurrently* instead (all fired at
   once) against `server.py`'s `ThreadingHTTPServer`, what would you
   expect to happen to total wall-clock time compared to 20x the median
   sequential latency — and what would have to be true about the model
   and the server for that expectation to hold?
4. Training/serving skew was described here as "silent" — no error, no
   crash, just quietly wrong predictions. Propose one concrete monitoring
   signal (not a fix to the pipeline itself) that could catch this in
   production *before* someone notices accuracy has degraded. What would
   that signal need to compare against what?
