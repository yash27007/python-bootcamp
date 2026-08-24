# 04 – Model Packaging & Versioning

## What you'll learn

Why a trained model — a large binary artifact plus the metadata needed to
use it correctly — is a problem neither Git nor an experiment log alone
solves, and how content-addressed storage (the same idea underneath Git's
own object model, covered in `02-git`) fixes it: serialization format
tradeoffs (pickle/joblib vs framework-native vs ONNX), and a real, tiny
"model registry" built from scratch — hash-based identity, JSON metadata
sidecars, and an append-only registration log.

| Topic | Status |
|-------|--------|
| Serialization formats and their tradeoffs | ✅ Complete |
| Model registries as "Git for models" | ✅ Complete |
| Building a from-scratch content-addressed registry | ✅ Complete |
| Distinguishing colliding filenames by content hash | ✅ Complete |

## Why it matters

Every model this course has trained so far (Phase 1's `05-machine-learning`
notebooks, `01-docker`'s pickled Iris classifier) ends its life as a
`.pkl` file with no built-in answer to "which version is this, what
produced it, and can I trust it hasn't changed." This topic is the
mechanism that answer relies on, before a production tool like MLflow's
Model Registry (`05-mlflow-dagshub`) automates it.

## Prerequisites

- `02-git` — this topic's registry reuses Git's content-addressing idea
  (identity = hash of content) directly, applied to model binaries instead
  of source blobs.
- `03-testing-ci` — a registry entry is most useful when it's tied to a
  pipeline run that already passed its test suite.
- Basic scikit-learn familiarity (`05-machine-learning`).

## What you'll build

- `registry.py` — a real, from-scratch content-addressed model registry:
  `register()` serializes a model with `joblib`, computes its SHA-256
  content hash, stores it under a hash-derived filename (never the
  caller-requested name), writes a JSON metadata sidecar (hash, timestamp,
  metric, library versions), and appends to an append-only
  `registry_log.json`; `load_model_by_hash()` reloads a model and
  re-verifies its integrity.
- `train_and_register.py` — trains two real `LogisticRegression` models on
  Iris (different `C`) and registers both under the *same* requested
  filename (`"model.pkl"`) on purpose — actually run with
  `.venv/bin/python`, with the real output (both models registered under
  distinct hashes, log correctly distinguishing them, integrity-verified
  reload) captured in `notes.md`.

The `registry/` directory `register()` writes into is gitignored, not
committed — run `.venv/bin/python train_and_register.py` to regenerate it
locally; the real output from that run is already captured in `notes.md`.

## Where it appears in real systems

- Any production system that retrains models needs a reliable answer to
  "which artifact is currently serving, and can I reproduce it" — this is
  the mechanism, at small scale, that answers that.
- `05-mlflow-dagshub` builds the production-grade version of this
  exact idea: MLflow's Model Registry, with stage transitions and full
  run/data lineage.
- `07-cicd` pipelines naturally register a model automatically
  once it passes the test suite from `03-testing-ci`, tying a registry
  entry to the commit and test results that produced it.

## What's next

`05-mlflow-dagshub` — the production experiment-tracking and model-registry
tool that automates everything this topic did by hand.
