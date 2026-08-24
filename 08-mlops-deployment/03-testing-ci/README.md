# 03 – Testing & CI

## What you'll learn

Why "I ran it once and the numbers looked fine" isn't a reliable way to
know a change to a training pipeline didn't break something, and how a
`pytest` test suite fixes that: the ML-adapted test pyramid (unit tests for
data-transform functions, model input/output invariant tests, an
end-to-end smoke test on a tiny fixture), and what's genuinely different
about testing ML code versus regular software — non-determinism, and
testing *properties* of a model's output rather than exact values.

| Topic | Status |
|-------|--------|
| The ML test pyramid (unit / invariant / smoke) | ✅ Complete |
| What's different about testing ML vs. regular software | ✅ Complete |
| Writing and running a real `pytest` suite | ✅ Complete |
| Deliberately breaking a function and watching tests catch it | ✅ Complete |

## Why it matters

Every topic that comes before this one in `08-mlops-deployment` produces
something (a Docker image, a Git-committed pipeline) that will be *changed*
again later — by a teammate, by a future refactor, by an upgraded library.
Without automated tests, every one of those changes is a gamble on whether
someone remembers to manually re-check the results. Tests are what turn
"probably still works" into a fast, repeatable, automatic answer.

## Prerequisites

- `02-git` — tests are what a CI pipeline runs on every commit/PR; this
  topic assumes you already have a Git history of changes worth gating.
- Basic Python and NumPy/pandas familiarity (`01-python-foundation`,
  `03-data-analysis`).
- No prior testing-framework knowledge assumed.

## What you'll build

- `pipeline.py` — a tiny, real training pipeline (Iris, logistic
  regression): a pure `preprocess()` transform, a seeded fixture loader,
  `train_model()`, and `evaluate()`.
- `test_pipeline.py` — a real `pytest` suite with one test per pyramid
  layer: a unit test on `preprocess()`'s shape/dtype/statistical output, an
  invariant test on trained-model probability outputs, and a regression
  test pinning accuracy to a threshold on a fixed fixture — actually run
  with `.venv/bin/pytest`, with the real pass output captured in `notes.md`.
- A real, deliberately-introduced bug (dividing by variance instead of
  standard deviation in `preprocess()`) with a stated hypothesis, the real
  failing `pytest` output it produced, and the fix reverted and reconfirmed
  passing — all captured in `notes.md`'s Experiment section.

## Where it appears in real systems

- CI pipelines (`08-mlops-deployment/07-cicd`) run exactly this
  kind of suite automatically on every pull request and block merging on
  failure — this topic builds the tests; that one builds the automation
  around them.
- Any team maintaining a production training pipeline relies on a suite
  like this to catch the "someone refactored a data-transform function and
  it silently changed behavior" class of bug before it reaches a served
  model.

## What's next

`04-model-packaging-versioning` — once a pipeline change is verified not to
break anything, the model it produces still needs to be saved, hashed, and
tracked as a versioned artifact distinct from the code that produced it.
