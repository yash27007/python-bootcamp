# NumPy

## Problem

Numeric computation on large collections of numbers — millions of sensor readings, pixel values,
feature vectors — needs to run fast and be expressed concisely. Python's built-in `list` was never
designed for this: it's a general-purpose, heterogeneous container, and using it for numeric work
is both slow and awkward (a manual `for` loop to add two lists element-wise, another to multiply,
another to filter). Nearly every ML/DS library (Pandas, scikit-learn, PyTorch, TensorFlow) exists
on top of a solution to exactly this problem — NumPy's `ndarray`.

## Intuition

Take two lists of a million numbers each, and add them element-wise. In pure Python that's a `for`
loop (or a list comprehension) executing a million individual additions, each one going through
Python's general-purpose object machinery. NumPy stores both lists as a single contiguous block of
raw machine numbers (e.g. 8-byte floats back to back in memory) and hands the whole operation to a
compiled C loop — `a + b` becomes one call that does not re-enter the Python interpreter per
element. Same idea, same math — radically different execution path.

**Broadcasting** extends this further: instead of only allowing operations between arrays of
identical shape, NumPy lets a smaller array "stretch" to match a larger one's shape whenever the
stretch is unambiguous — e.g. adding a single per-column mean (shape `(3,)`) to every row of a
`(100, 3)` matrix, without writing a loop or manually replicating the mean 100 times.

## Why simpler approaches fail

The simplest approach — a plain Python `list` plus a `for` loop — fails for a specific, structural
reason: **a Python list stores boxed objects, not raw numbers.** Every element of `[1, 2, 3]` is
actually a full Python `int` object living somewhere on the heap, and the list itself is an array of
*pointers* to those objects, each carrying reference-count and type metadata. Every arithmetic
operation on a list element means: dereference the pointer, check the object's type at runtime,
dispatch to the right method, allocate a new boxed result object. That per-element overhead is
constant no matter how simple the arithmetic is, and it repeats for every single element — there is
no way to "batch" it away while still using plain lists and Python-level loops. NumPy's `ndarray`
sidesteps the entire problem by not boxing at all: it is a single typed, contiguous memory buffer,
so the type check happens once (when the operation is dispatched) rather than once per element, and
the actual arithmetic runs as a tight, compiled loop over raw memory.

## Mathematical foundation

### Broadcasting rules, derived from shape arithmetic

NumPy compares two shapes **from the trailing (rightmost) dimension leftward**. Missing leading
dimensions on the shorter shape are treated as `1`. For each aligned pair of dimensions:

- if they're **equal** → keep that size
- if **one of them is 1** → stretch it to match the other
- **otherwise** → incompatible, raise a `ValueError`

Walking through `(3,1)` and `(1,4)`:

```
shape a: (3, 1)
shape b:    (1, 4)      <- align from the right
```

- Rightmost pair: `1` vs `4` → one of them is 1 → stretch to `4`
- Next pair: `3` vs `1` → one of them is 1 → stretch to `3`
- Result shape: **`(3, 4)`**

Every `(i, j)` cell of the result is `a[i, 0] + b[0, j]` — `a`'s single column is conceptually
repeated across 4 columns, `b`'s single row is conceptually repeated across 3 rows (no memory is
actually duplicated; NumPy just re-reads the same value). This is *why* a column vector plus a row
vector produces a full outer-sum table rather than an error or an element-count mismatch.

**When shapes are NOT compatible:** `(2,3)` and `(2,)`. Align from the right: `(2,)`'s only
dimension (`2`) is compared against `(2,3)`'s trailing dimension (`3`). `2 != 3` and neither is
`1` → incompatible. NumPy raises `ValueError: operands could not be broadcast together with shapes
(2,3) (2,)`. (If the second shape had instead been `(3,)`, it *would* broadcast — against the
trailing dimension, matching column-wise.)

## Algorithm

To combine two arrays of shapes `S1` and `S2`:
1. Right-pad the shorter shape's tuple with leading `1`s until both have equal length.
2. Compare dimension pairs from right to left; at each position, the aligned sizes must be equal
   or one of them must be `1`.
3. If any pair fails that test, raise a shape-mismatch error.
4. Otherwise the result shape takes, at each position, whichever of the two sizes is not `1` (or
   either, if both are equal).
5. Conceptually replicate each array along any dimension where it had size `1`, then apply the
   operation element-wise.

## From-scratch implementation

A literal, explicit `for` loop over indices (`numpy_basics.ipynb`, "12. From-Scratch") computing
`c[i] = a[i]*b[i] + a[i]**2` on two arrays of 2,000,000 random floats, timed against the equivalent
vectorized NumPy expression on the *same* data:

```python
n = 2_000_000
rng = np.random.default_rng(0)
a1 = rng.random(n)
b1 = rng.random(n)

start = time.perf_counter()
result_manual = [0.0] * n
for i in range(n):
    result_manual[i] = a1[i] * b1[i] + a1[i] ** 2
t_manual = time.perf_counter() - start

start = time.perf_counter()
result_vec = a1 * b1 + a1 ** 2
t_vec = time.perf_counter() - start
```

Actual measured output:

```
Manual Python loop:  0.4148s
Vectorized NumPy:    0.014581s
NumPy is 28.4x faster
Results match: True
```

`np.allclose` confirms the manual loop and the vectorized expression compute the same values — the
~28x gap is purely execution-path overhead (boxed-object dispatch vs. a compiled loop over raw
memory), not a difference in what's being computed.

## Practical implementation

`numpy_basics.ipynb` covers the full practical surface: array creation (`np.array`, `np.zeros`,
`np.arange`, `np.linspace`), attributes (`.shape`, `.dtype`, `.ndim`), indexing/slicing (basic,
boolean-mask, fancy), vectorized arithmetic and ufuncs, broadcasting (scalar/row/column, plus the
derivation above and its failure mode below), aggregation functions, reshaping/stacking/splitting,
linear algebra (`@`, `np.linalg.inv/eig/solve`), and random number generation via the modern
`np.random.default_rng` API. Section 10 times `arr**2` (vectorized) against a list comprehension —
Section 12 above extends that with an explicit `for` loop and a two-array expression, since a list
comprehension still hides its loop inside a single bytecode op.

## Experiment

**Hypothesis:** the from-scratch manual loop and the vectorized NumPy expression compute identical
results, and the vectorized version is at least an order of magnitude faster on real (random,
non-trivial) data.

**Setup:** two arrays of 2,000,000 `float64` values from `np.random.default_rng(0)`, timed with
`time.perf_counter()`, computing `a*b + a**2` both ways.

**Actual result:** manual loop 0.4148s, vectorized 0.014581s, speedup ≈ 28.4x, `np.allclose` result
match = `True`.

**Interpretation:** confirms both the correctness (same computation, same answer) and the
performance claim (order-of-magnitude speedup) rather than asserting either.

**Limitations:** the exact speedup factor depends on machine, Python version, and array size/dtype
— it is not a universal constant (NumPy's own docs and folklore cite "10-100x" as a typical range,
which this measurement falls within). For very small arrays, the fixed overhead of dispatching a
NumPy call can make the two comparable or even favor a plain loop — vectorization wins by
amortizing per-call overhead over many elements, so the *win grows* with array size, it isn't fixed.

## Failure modes

- **Accidental broadcasting — wrong shape, no error, silently wrong answer.** Subtracting a
  per-row mean from a matrix requires reshaping the 1-D mean vector into a column `(n, 1)` first.
  On a **square** matrix, forgetting that reshape doesn't raise an error — `grades - row_means`
  broadcasts `row_means` (shape `(4,)`) against the trailing dimension (columns) instead of rows,
  because the shapes happen to be "compatible" under the broadcasting rule even though that's not
  what was intended. Measured: `grades - row_means` (no reshape) leaves row 0 summing to `15.0`
  instead of the `~0.0` a correct row-centering must produce; `grades - row_means.reshape(-1, 1)`
  correctly gives row 0 sum `0.0`. The bug produces a result of the *right shape* and no exception
  — it is only wrong if you check the actual values, which is what makes it dangerous.
- **Views vs. copies — mutating a view unexpectedly changes the original array.** Basic slicing
  (`arr[a:b]`) returns a **view** that shares memory with the original array; fancy indexing (a
  list/array of indices, or a boolean mask) always returns a **copy**. Measured:
  `view = arr[2:5]; view[0] = 99` mutates `arr` itself (`arr.base is arr` for the original,
  `view.base is arr` for the slice — the view doesn't own its data). The equivalent fancy-indexed
  `arr3[[2,3,4]]` does *not* mutate `arr3` when modified, because it already copied. Code that
  assumes "slicing an array gives me an independent array" silently corrupts data other code still
  holds a reference to; `.copy()` makes the independence explicit and intentional.

## Real-world usage

Every numeric Python library sits on NumPy arrays underneath — Pandas columns are backed by NumPy
arrays, scikit-learn's `fit(X, y)` expects `X`/`y` as (or convertible to) NumPy arrays, and
PyTorch/TensorFlow tensors share the same broadcasting semantics because NumPy's rules became the
de facto standard the whole ecosystem copied. Feature normalization (subtract mean, divide by std,
Section 5's Z-score example), matrix multiplication in the forward/backward pass of a neural
network, and image processing (pixel arrays are literally NumPy arrays with shape `(H, W, C)`) all
depend on broadcasting and vectorization working correctly — silently wrong broadcasting in a
training pipeline can corrupt every downstream computation without ever raising an exception.

## Mental model

An `ndarray` is a typed, contiguous memory buffer plus a shape — arithmetic on it is a compiled
loop over raw memory, not a Python loop over boxed objects, and broadcasting is just a *rule for
inferring how to line up two shapes before that loop runs*, not magic: if you can derive the
result shape by hand (right-align, stretch the 1s), NumPy will do exactly that and nothing more.

## Questions to think about

1. Two arrays have shapes `(5, 1, 3)` and `(4, 3)`. Do they broadcast? If so, to what shape? Derive
   it using the right-to-left alignment rule before checking with NumPy.
2. Why does the from-scratch vs. vectorized speedup in the Experiment section grow (roughly) with
   array size rather than staying constant? What would you expect at `n = 100` instead of `n =
   2,000,000`?
3. You compute `matrix - matrix.mean(axis=1)` on a square matrix and get no error. What check would
   immediately tell you whether the subtraction happened along the axis you intended?
4. If `view = arr[::2]` (a strided view) is modified in place, does the original `arr` change? What
   about `view = arr[arr > 0]` (boolean-mask indexing)? Explain the difference using view-vs-copy
   semantics.
5. Broadcasting lets you avoid writing an explicit loop to replicate a smaller array. What is the
   memory cost of that replication — does NumPy actually allocate the stretched-out data, or not?
   How would you check?
