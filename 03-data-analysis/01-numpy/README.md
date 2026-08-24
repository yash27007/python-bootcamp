# NumPy

## What you'll learn

Why NumPy's `ndarray` exists (per-element Python object overhead in plain lists), broadcasting
rules derived from actual shape arithmetic (not just stated), and the full practical surface —
array creation, indexing/slicing, vectorized operations, aggregation, reshaping, linear algebra,
and random number generation.

## Why it matters

Every numeric Python library — Pandas, scikit-learn, PyTorch, TensorFlow — is built on NumPy
arrays and copies its broadcasting semantics. Getting broadcasting wrong (see this topic's Failure
modes: silent wrong-shape bugs, views vs. copies) silently corrupts computations without raising an
exception, which makes understanding the underlying shape-arithmetic rule, not just the API,
essential.

## Prerequisites

- Comfort with Python lists and loops (`01-python-foundation`)
- No prior linear algebra required — broadcasting is derived from scratch in this topic

## What you'll build

- A manual, explicit `for`-loop element-wise computation timed against the equivalent vectorized
  NumPy expression on the same 2,000,000-element random data — a real measured ~28x speedup, with
  results verified to match exactly (`np.allclose`)
- A worked derivation of why shapes `(3,1)` and `(1,4)` broadcast to `(3,4)`, and a real example of
  an incompatible-shape `ValueError`
- A real, executed demonstration of the "accidental broadcasting" bug (subtracting a per-row mean
  from a square matrix without reshaping — no error, silently wrong values) and of the
  view-vs-copy mutation gotcha

See [`notes.md`](notes.md) for the full write-up including real captured output, and
[`numpy_basics.ipynb`](numpy_basics.ipynb) (all cells executed) for the practical tour — array
creation, indexing, vectorized ops, broadcasting, aggregation, reshaping/stacking, linear algebra,
random generation, and the from-scratch/failure-mode sections above.

## Where it shows up in real systems

Pandas columns, scikit-learn's `X`/`y` inputs, and PyTorch/TensorFlow tensors are all NumPy arrays
(or share its broadcasting rules) underneath. Feature normalization, matrix multiplication in a
neural network's forward pass, and image processing (pixel arrays as `(H, W, C)` NumPy arrays) all
depend on broadcasting and vectorization behaving as derived here.

## What's next

`02-pandas` — a labeled, columnar extension of the NumPy array for heterogeneous tabular data.
