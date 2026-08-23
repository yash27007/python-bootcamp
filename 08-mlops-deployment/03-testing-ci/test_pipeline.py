"""
Real pytest tests for pipeline.py — run with:

    .venv/bin/pytest 08-mlops-deployment/03-testing-ci/test_pipeline.py -v

Three tests, matching the ML test pyramid described in notes.md:

1. test_preprocess_output_shape_and_dtype  — unit test on a pure data-transform.
2. test_prediction_probabilities_are_valid — model I/O invariant test (a
   *property* of the output, not an exact value).
3. test_accuracy_does_not_regress          — regression test pinning a metric
   to a tolerance/threshold on a fixed fixture, not an exact float.
"""

import numpy as np
import pytest

from pipeline import evaluate, load_fixture_split, preprocess, train_model

# Below this accuracy on the fixed fixture split, something has regressed.
# Chosen with headroom under the real observed accuracy (~0.98), not pinned
# to the exact value — see notes.md "Failure modes" for why.
ACCURACY_FLOOR = 0.90


@pytest.fixture(scope="module")
def fixture_split():
    """A fixed, seeded train/test split — the tiny fixture dataset all three
    tests share, so "the whole pipeline runs end-to-end" is exercised once."""
    return load_fixture_split(random_state=0)


@pytest.fixture(scope="module")
def trained_model(fixture_split):
    X_train, _, y_train, _ = fixture_split
    return train_model(X_train, y_train, random_state=0)


def test_preprocess_output_shape_and_dtype():
    """Unit test: preprocess() must preserve shape and always return float64,
    regardless of the input dtype it was given."""
    X = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 10]], dtype=np.int64)

    out = preprocess(X)

    assert out.shape == X.shape
    assert out.dtype == np.float64
    # Each column should now have ~zero mean and ~unit variance.
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(out.std(axis=0), 1.0, atol=1e-8)


def test_prediction_probabilities_are_valid(trained_model, fixture_split):
    """Model I/O invariant test: this doesn't check *which* class the model
    predicts (that's what the regression test below is for) — it checks a
    property every valid probability output must have, regardless of model
    quality: each row of predict_proba sums to 1 and every value is in [0, 1].
    """
    _, X_test, _, _ = fixture_split

    probs = trained_model.predict_proba(preprocess(X_test))

    assert probs.shape == (X_test.shape[0], 3)  # 3 Iris classes
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    row_sums = probs.sum(axis=1)
    assert np.allclose(row_sums, 1.0, atol=1e-6)


def test_accuracy_does_not_regress(trained_model, fixture_split):
    """Regression test: pins accuracy to a *threshold*, not an exact float.
    A future change to preprocess() or the model that silently breaks
    training should fail this test even though it "ran without crashing"."""
    _, X_test, _, y_test = fixture_split

    acc = evaluate(trained_model, X_test, y_test)

    assert acc >= ACCURACY_FLOOR, f"accuracy {acc:.4f} fell below floor {ACCURACY_FLOOR}"
