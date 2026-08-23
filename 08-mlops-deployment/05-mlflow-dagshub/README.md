# 05 – MLflow, DagsHub & DVC

## What you'll learn

Why `04`'s model registry only ever sees the *winning* artifact, and how
experiment tracking closes that gap: a structured, run-ID-keyed log of
params/metrics/artifacts/code-version, built from scratch first (a plain
JSONL file) and then with the real `mlflow` client against a local SQLite
tracking store. Alongside it, DVC's data-versioning model — the same
content-addressing idea from `02-git` and `04`, applied to large training
data files via a small `.dvc` pointer file committed to Git instead of the
data itself.

| Topic | Status |
|-------|--------|
| MLOps with MLflow & DagsHub | ✅ Complete |
| Data Version Control (DVC) | ✅ Complete |

## Why it matters

A trained model's provenance isn't just "which artifact" (`04`'s
question) — it's "which of the dozens of experiments that led there used
which hyperparameters, on which data, and how do the results compare."
A manually-updated spreadsheet of runs doesn't scale, can't be queried,
and has no verifiable link back to the code/data version that produced
each row. This topic replaces that spreadsheet with a real, queryable
tracking store, and closes the matching gap on the data side: without
data versioning, "which dataset produced this number" is a matter of
memory, not verification.

## Prerequisites

- `02-git` — DVC's `.dvc` pointer files reuse Git's content-addressed
  object model directly, just applied to large data files instead of
  source blobs.
- `04-model-packaging-versioning` — this topic's experiment-tracking store
  is the production-grade companion to `04`'s from-scratch registry; the
  "structured log beats manual bookkeeping" argument is the same one,
  applied one level up (experiments, not just the final artifact).
- Basic scikit-learn familiarity (`05-machine-learning`).

## What you'll build

- `json_run_logger.py` — a from-scratch experiment tracker: `log_run()`
  appends one JSON record (run ID, timestamp, params, metrics) per run to
  a JSONL file; `query_runs()` filters it back out. Actually run against
  three real `LogisticRegression` fits (varying `C`) on the breast-cancer
  dataset, with real output captured in `notes.md`.
- `mlflow_tracking_demo.py` — the same experiment, logged with the real
  `mlflow` client to a local SQLite tracking store (`mlflow.db`): six real
  runs varying `C`, logged params/metrics, queried back and ranked by
  accuracy via `MlflowClient().search_runs()`. Actually executed, with
  real logged metrics captured in `notes.md`.
- A documented (not executed — `dvc` isn't installed in this environment)
  walkthrough of `dvc init` / `dvc add` against a small toy dataset
  (`data/toy_dataset.csv`), showing the exact generated `.dvc` pointer
  file and what actually gets committed to Git versus what stays in DVC's
  own cache.

## Where it appears in real systems

- Any team iterating on models needs a systematic answer to "which run
  produced this number, with what configuration, on what data" at the
  scale of hundreds or thousands of runs — not six.
- MLflow's Model Registry is the production version of `04`'s from-scratch
  registry, built directly on top of this topic's tracking store — a
  registered model links back to the exact run (and transitively, the
  exact params/data) that produced it.
- DagsHub (or a self-hosted MLflow server + DVC remote) is what lets a
  team share one tracking store and one data-version cache instead of
  everyone's experiments and datasets living only on their own machine.

## What's next

`06-bentoml` — packaging a chosen model (now traceable back to the exact
run, params, and data that produced it) for serving.
