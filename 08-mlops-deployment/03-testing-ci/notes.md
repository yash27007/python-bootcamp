# Testing & CI for ML pipelines

## Problem

A training pipeline is code, and code changes over time: a preprocessing
function gets refactored, a hyperparameter gets tuned, a new feature gets
added, a library gets upgraded. Every one of those changes can silently
change what the pipeline produces — a shape bug in a transform, a swapped
train/test split, a metric computed on the wrong column. The question this
topic answers is mechanical and unglamorous but load-bearing: **how do you
know a change to a training pipeline or model didn't break something,
before you ship it** — not after a stakeholder notices predictions look
wrong in production?

## Intuition

Think of the pipeline as a chain: raw data → preprocessing → training →
evaluation → a served model. A bug can enter at any link, and the failure
mode is rarely a crash — it's usually a *plausible-looking wrong number*.
If `preprocess()` accidentally divides by variance instead of standard
deviation (a one-character bug: forgetting `np.sqrt`), the pipeline still
runs to completion, still produces a model, still produces predictions.
Nothing crashes. The only way to know something is wrong is to check a
*property* you expect to hold — "each column should have unit variance
after standardizing" — and have something automatically re-check it every
time the code changes.

That's what a test suite is: a codified set of "this should always be true"
statements, checked automatically, cheaply, and repeatedly, so a human
doesn't have to remember to eyeball the numbers every single time.

## Why simpler approaches fail

The default without a test suite is: "I ran the training script once,
looked at the accuracy number, it looked reasonable, shipped it." This
fails in three specific ways:

1. **It doesn't catch regressions.** A one-time manual check only verifies
   the pipeline works *today, with today's code*. It says nothing about
   whether tomorrow's refactor of `preprocess()` silently changes what it
   does. There's no automatic re-check.
2. **It doesn't run automatically.** A human has to remember to do it, has
   to remember to do it for *every* change (not just the ones that "seem
   risky" — the variance/std bug above is exactly the kind of change that
   looks harmless), and has to actually run it before merging, not after.
3. **It doesn't scale past one person's memory.** On a team, or even
   working alone across months, "I recall it worked" is not a check
   anyone else (including future-you) can rely on or reproduce. There's no
   artifact — no log, no pass/fail signal — that a reviewer or a CI system
   can act on.

A single manual run also only ever tests the *happy path with today's
data*. It doesn't test edge cases (an empty batch, a constant-valued
column, a class that's missing from a fold) because a human eyeballing
"the numbers look fine" isn't systematically probing those.

## Conceptual foundation

This section is titled "Conceptual foundation" rather than "Mathematical
foundation": testing strategy has no closed-form math to derive — the
foundational idea here is conceptual (how to structure a test suite around
an ML pipeline, and what a "test" even means when there's no single
correct output), not mathematical.

### The test pyramid, adapted to ML

The classic software test pyramid (many fast unit tests at the bottom, a
handful of slow integration/end-to-end tests at the top) maps onto an ML
pipeline like this:

```
        /  end-to-end smoke test  \      <- pipeline runs on a tiny fixture,
       /   (1 test, slow-ish)      \        produces *a* model, doesn't crash
      /----------------------------\
     /  model I/O invariant tests   \    <- properties of predictions:
    /   (few tests, medium speed)    \      shapes, value ranges, probabilities
   /------------------------------- \
  /   unit tests on data transforms  \   <- pure functions: preprocess(),
 /    (many tests, fast)              \     feature engineering, no randomness
/--------------------------------------\
```

- **Unit tests for data-transform functions** — `preprocess()` is a pure
  function (no randomness, no I/O): given an input array, it always
  produces the same output array. These are the cheapest, fastest,
  highest-value tests because they isolate exactly one thing.
- **Model input/output invariant tests** — you cannot assert "the model
  predicts class 2 for this input" as a hard requirement (see below), but
  you *can* assert invariants that any correctly-functioning model must
  satisfy: `predict_proba` rows sum to 1, all probabilities are in
  `[0, 1]`, output shape matches `(n_samples, n_classes)`.
- **An end-to-end smoke test on a tiny fixture** — load a small, fixed
  dataset, run the *entire* pipeline (preprocess → train → evaluate), and
  assert it completes and produces a model with the expected interface.
  This is the test that catches "someone changed a function signature and
  the pipeline no longer even runs" — a shockingly common real failure
  that unit tests alone miss because each unit test mocks its inputs.

### What's genuinely different about testing ML vs. regular software

Regular software testing usually asserts *exact* expected outputs:
`add(2, 3) == 5`, always, every time. ML testing can't do that, for two
structural reasons:

1. **Non-determinism.** Training involves randomness — weight
   initialization, data shuffling, stochastic optimizers. Even with a
   fixed random seed, results can differ slightly across library versions,
   hardware, or parallelism settings. Asserting an exact accuracy value
   like `assert acc == 0.9666666` is asserting something that isn't
   actually guaranteed to reproduce.
2. **No single "correct" output.** For `add(2, 3)`, there is exactly one
   right answer. For "what should this trained model predict on this
   input," there usually isn't — a model is a statistical approximation,
   not a deterministic function with a known ground truth for arbitrary
   inputs. What *is* knowable is a set of **properties** the output must
   satisfy (valid probability distribution, correct shape, accuracy above
   some floor on a fixed fixture) — testing ML means testing those
   properties, not pinning exact values.

This is why the three tests below use `np.allclose(..., atol=...)` and
threshold checks (`acc >= ACCURACY_FLOOR`) rather than `==` on floats.

## Algorithm

The workflow this topic's `test_pipeline.py` follows:

1. Define a **fixed fixture**: a small, seeded dataset and train/test split
   (`load_fixture_split(random_state=0)`), so every test run sees the same
   data — reproducibility is a precondition for any of this to be
   meaningful.
2. **Unit-test** a pure data-transform function in isolation
   (`preprocess()`): check shape, dtype, and the statistical property it's
   supposed to produce (zero mean, unit variance).
3. **Invariant-test** the trained model's output: train once (fixture-scoped
   so all tests share one trained model, keeping the suite fast), then
   assert properties of `predict_proba` output that must hold regardless of
   how good the model is.
4. **Regression-test** a scalar quality metric: compute accuracy on the
   fixed test split and assert it stays above a floor with headroom under
   the real observed value — not pinned to the exact float.
5. Run the suite with `pytest`; a non-zero exit code means "do not merge/
   ship this change" — this is the signal a CI pipeline gates on (see
   "Real-world usage").

## From-scratch implementation

`pipeline.py` — the tiny pipeline under test:

```python
def preprocess(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=np.float64)
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std == 0] = 1.0
    return (X - mean) / std

def load_fixture_split(random_state: int = 0):
    data = load_iris()
    return train_test_split(data.data, data.target, test_size=0.3,
                             random_state=random_state, stratify=data.target)

def train_model(X_train, y_train, random_state: int = 0):
    model = LogisticRegression(max_iter=1000, random_state=random_state)
    model.fit(preprocess(X_train), y_train)
    return model

def evaluate(model, X_test, y_test) -> float:
    preds = model.predict(preprocess(X_test))
    return float((preds == y_test).mean())
```

`test_pipeline.py` — the three real tests (full file in this directory),
one per pyramid layer:

```python
def test_preprocess_output_shape_and_dtype():
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=np.int64)
    out = preprocess(X)
    assert out.shape == X.shape
    assert out.dtype == np.float64
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(out.std(axis=0), 1.0, atol=1e-8)

def test_prediction_probabilities_are_valid(trained_model, fixture_split):
    _, X_test, _, _ = fixture_split
    probs = trained_model.predict_proba(preprocess(X_test))
    assert probs.shape == (X_test.shape[0], 3)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-6)

def test_accuracy_does_not_regress(trained_model, fixture_split):
    _, X_test, _, y_test = fixture_split
    acc = evaluate(trained_model, X_test, y_test)
    assert acc >= ACCURACY_FLOOR
```

**Real, actually-executed pass output** (`.venv/bin/pytest test_pipeline.py -v`,
run from `08-mlops-deployment/03-testing-ci/`):

```
============================= test session starts ==============================
platform linux -- Python 3.13.9, pytest-9.1.1, pluggy-1.6.0 -- .../.venv/bin/python3
collecting ... collected 3 items

test_pipeline.py::test_preprocess_output_shape_and_dtype PASSED          [ 33%]
test_pipeline.py::test_prediction_probabilities_are_valid PASSED         [ 66%]
test_pipeline.py::test_accuracy_does_not_regress PASSED                  [100%]

============================== 3 passed in 0.82s ===============================
```

## Practical implementation

This *is* the practical implementation — `pytest` is the industry-standard
Python test runner, and this small suite is structured the way a real
pipeline's test suite is: fixtures for shared setup (`@pytest.fixture`,
here scoped `module` so the model is trained once and reused, keeping the
suite fast), one test function per property, plain `assert` statements
(pytest rewrites them to give the rich failure output seen below — no
special assertion-library syntax needed). The only thing a production
suite adds on top is scale (hundreds of such tests) and a CI runner
executing it automatically (`07-cicd`, forward-pointer below).

## Experiment

**Hypothesis:** if `preprocess()`'s standardization formula is broken —
specifically, dividing by *variance* instead of *standard deviation* (a
one-character bug: dropping the square root) — the test suite will catch
it. Concretely: `test_preprocess_output_shape_and_dtype` should fail
(output columns will no longer have unit variance, since variance ≠ std
unless std happens to equal 1), while `test_prediction_probabilities_are_valid`
should still pass (probabilities are still a valid distribution regardless
of *how* the inputs were scaled) and `test_accuracy_does_not_regress` might
or might not fail depending on whether the distorted feature scale still
lets logistic regression separate Iris classes.

**Setup:** in `pipeline.py`, changed

```python
std = X.std(axis=0)
...
return (X - mean) / std
```

to

```python
variance = X.var(axis=0)
...
return (X - mean) / variance   # BUG: divides by variance instead of std
```

and reran the exact same suite, no other changes.

**Actual result** (`.venv/bin/pytest test_pipeline.py -v`):

```
============================= test session starts ==============================
collecting ... collected 3 items

test_pipeline.py::test_preprocess_output_shape_and_dtype FAILED          [ 33%]
test_pipeline.py::test_prediction_probabilities_are_valid PASSED         [ 66%]
test_pipeline.py::test_accuracy_does_not_regress PASSED                  [100%]

=================================== FAILURES ===================================
____________________ test_preprocess_output_shape_and_dtype ____________________

    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-8)
>   assert np.allclose(out.std(axis=0), 1.0, atol=1e-8)
E   assert False
E    +  where False = <function allclose at 0x7085ef78d530>(array([0.40824829, 0.40824829, 0.34874292]), 1.0, atol=1e-08)

test_pipeline.py:50: AssertionError
=========================== short test summary info ============================
FAILED test_pipeline.py::test_preprocess_output_shape_and_dtype - assert False
========================= 1 failed, 2 passed in 0.77s ==========================
```

**Interpretation:** exactly as hypothesized — the unit test on the pure
transform caught the bug immediately and precisely (the assertion message
even shows the wrong std values). The probability-invariant test passed
because it tests a property that's robust to *how* inputs were scaled, not
whether scaling was correct — a reminder that no single test layer catches
everything; that's the point of having a pyramid, not one giant test. The
regression test also happened to still pass on this particular toy
dataset/model combination — variance-scaling still left Iris linearly
separable enough for logistic regression at `ACCURACY_FLOOR = 0.90`. That
is itself informative: on a harder dataset the same bug could easily have
tanked accuracy below the floor too, but this run shows the two test
layers are *not redundant* — the unit test is what reliably caught this
specific class of bug, and the regression test alone would have missed it
here.

**Limitations:** this is one bug on one tiny dataset with one seed —
demonstrates the mechanism, not a general claim that unit tests always
catch what regression tests miss (or vice versa; the reverse is equally
possible, e.g. a bug in `evaluate()`'s comparison logic that the accuracy
test would catch but no property test would).

The pipeline was then reverted to the correct `std` formula and the full
suite reconfirmed passing (`3 passed in 0.73s`) before moving on.

## Failure modes

- **Flaky tests from unseeded randomness.** If `train_model` or the
  train/test split doesn't fix a random seed, the same test can pass one
  run and fail the next purely from noise — not from a real regression.
  This erodes trust in the suite (people start re-running failed CI jobs
  "hoping it passes" instead of investigating), which is worse than having
  no tests at all. Always seed anything stochastic in a test fixture.
- **Pinning exact floating-point values instead of tolerances.** Writing
  `assert acc == 0.9666666666` instead of `assert acc >= 0.90` makes tests
  fail on harmless changes — a library version bump that changes float
  rounding in the 10th decimal place, a different BLAS backend — for
  reasons that have nothing to do with correctness. This trains people to
  ignore failing tests, defeating the whole point.
- **No coverage of "it trains" path.** It's easy to write unit tests for
  small pure functions and skip testing that the full pipeline
  (data → preprocess → fit → predict) actually runs end-to-end without
  raising. That's exactly the class of bug — an API signature change, a
  missing argument, an incompatible dtype passed between stages — that
  only an end-to-end smoke test on a tiny fixture catches, and it's the
  failure mode unit tests structurally cannot see because each one only
  exercises one function in isolation.

## Real-world usage

In production ML systems, this test suite is what a **CI pipeline** runs
automatically on every pull request, before a change is allowed to merge —
"gating" the merge on the suite passing. That's the subject of a later
topic, `07-cicd`: this topic builds the tests; that one builds the
automation that runs them on every push and blocks merges on failure. The
tests themselves don't change between running them by hand and running
them in CI — what changes is *who* runs them (a server, not a person) and
*when* (automatically, on every change, not "when someone remembers to").

## Mental model

**A test is a codified property, checked automatically, so a human doesn't
have to re-verify it by hand every time the code changes — and for ML,
those properties are usually "this statistic holds" or "this shape/range
holds," not "this exact number matches," because non-determinism and the
lack of one true output make exact-value checks either wrong or
meaningless.**

## Questions to think about

1. `test_prediction_probabilities_are_valid` would still pass even for a
   completely untrained, randomly-initialized model. What does that tell
   you about what this test is — and isn't — protecting against, and why
   is `test_accuracy_does_not_regress` still needed alongside it?
2. Suppose `ACCURACY_FLOOR` were set to `0.999` instead of `0.90`. What
   failure mode from this topic would that risk creating, and what would
   you observe in CI over time if it happened?
3. The end-to-end smoke test is described as catching bugs unit tests
   structurally cannot see. Construct a concrete code change to
   `pipeline.py` that would pass all three existing tests here but should
   still be caught by a smoke test that also checks, say, that
   `train_model` returns an object with a `.predict` method. What would
   that change be?
4. If `load_fixture_split` used `random_state=None` instead of a fixed
   seed, which of the three tests would become flaky, and which (if any)
   would stay reliable regardless? Why the difference?
