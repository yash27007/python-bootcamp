# Pandas

## What you'll learn

Why `DataFrame` exists on top of NumPy (heterogeneous, labeled, missing-value-containing tabular
data vs. NumPy's homogeneous, unlabeled arrays), `groupby` as the split-apply-combine pattern
implemented by hand before shown as `.agg()`, and the full practical surface — selection,
filtering, missing values, string/datetime accessors.

## Why it matters

`DataFrame`/`groupby` are the default interface for tabular data across the ML/DS stack — every
CSV/Parquet read, every feature-engineering filter, every per-segment metric starts here. This
topic's Failure modes (chained-indexing silently failing to write back, an int column silently
upcasting to float on a missing value) are two of the most common real causes of silently corrupted
features in production pipelines.

## Prerequisites

- `01-numpy` (a `DataFrame` column is a NumPy array underneath — its dtype/copy rules still apply)
- No prior Pandas experience required

## What you'll build

- Split-apply-combine implemented by hand with a plain Python `dict` of lists on a toy orders
  dataset, then the same dataset aggregated with `.groupby().agg()` — verified to match exactly
  for both `sum` and `mean`
- A real, executed reproduction of chained-indexing's `ChainedAssignmentError`
  (`SettingWithCopyWarning`'s modern equivalent) — showing the original DataFrame is silently left
  unmodified — and its `.loc`-based fix
- A real, executed demonstration of an `int64` column upcasting to `float64` the moment a single
  `NaN` is written into it

See [`notes.md`](notes.md) for the full write-up including real captured output, and
[`pandas_basics.ipynb`](pandas_basics.ipynb) (all cells executed) for the practical tour — Series/
DataFrame basics, `.loc`/`.iloc`, filtering, missing values, `.apply`/`.map`, `.str`/`.dt`
accessors, and the from-scratch/failure-mode sections above.

## Where it shows up in real systems

Reading any tabular dataset returns a `DataFrame`; feature pipelines filter and transform columns
with `.loc`/`.apply`; per-segment metrics (revenue by region, error rate by model version) are
computed with `.groupby(...).agg(...)` before any modeling happens. The chained-indexing and
dtype-coercion failure modes covered here are recurring, genuinely subtle sources of silently
corrupted features in real pipelines.

## What's next

`03-data-manipulation` — `groupby` extended to merges/joins, pivoting, and rolling windows.
