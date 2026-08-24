# Pandas

## Problem

Real-world tabular data is **heterogeneous, labeled, and full of missing values** — a customer
table has a string name column, an integer age column, a float salary column, and some cells are
simply unknown. NumPy arrays are none of those things by design: an `ndarray` is homogeneous (one
dtype for the whole array), positionally indexed (no column names, no row labels), and has no
native representation for "this value is missing." Pandas exists to solve exactly the gap between
"NumPy's fast, uniform numeric arrays" and "the messy, mixed-type, labeled tables data actually
comes in."

## Intuition

Think of a spreadsheet: each column has its own type (names are text, ages are whole numbers,
scores are decimals), rows and columns both have labels you refer to by name, and some cells are
just blank. A `DataFrame` models exactly that — a dict of named, individually-typed columns, each
one internally a NumPy array (or NumPy-backed extension array), sharing a common row index. So
`df['age']` isn't a new concept; it's "the NumPy array living in the 'age' slot of this dict-like
structure, wrapped so you can also filter/align/label it by row." Pandas' value-add on top of raw
NumPy is exactly: per-column dtypes, row/column labels, aligned operations that respect those
labels, and first-class `NaN` handling.

`groupby` extends this same idea to aggregation: **split** the rows into groups sharing a key
value, **apply** a function to each group independently, **combine** the per-group results back
into one structure — the split-apply-combine pattern. `df.groupby('region')['amount'].sum()` reads
as "split by region, apply sum, combine into one Series indexed by region" — not as an opaque
one-liner.

## Why simpler approaches fail

A raw NumPy array can't represent the "orders" table above cleanly: forcing `region` (text) and
`amount` (numeric) into one 2-D array means either upcasting everything to a single `object` dtype
(losing all of NumPy's speed and type safety — back to boxed Python objects) or maintaining
*separate* parallel NumPy arrays yourself (`regions = np.array([...])`, `amounts = np.array([...])`)
and manually keeping them aligned by index on every filter/sort/join — exactly the bookkeeping
Pandas exists to automate. Grouped aggregation without Pandas means manually maintaining a
dict-of-lists per key (the From-scratch section below does this explicitly) and re-deriving that
logic — and its edge cases (empty groups, multiple aggregation functions, multi-column keys) — by
hand every time. `DataFrame`/`groupby` exist so this bookkeeping is written once, correctly, and
reused.

## Conceptual foundation

*(Substituting for "Mathematical foundation" — this topic's foundation is a conceptual model, not
a derivation; documented per the template's substitution allowance.)*

**`DataFrame` as a labeled, columnar extension of NumPy arrays.** A `DataFrame` is conceptually a
dict `{column_name: 1-D NumPy array}`, all sharing one common row `Index`. Each column keeps its
own dtype independently — that's the "heterogeneous" part NumPy alone can't offer. `.loc[]` indexes
by label (row/column *names*), `.iloc[]` indexes by position (integer offsets) — this distinction
exists because once rows have labels, "the 3rd row" and "the row labeled 3" are no longer
guaranteed to be the same row (e.g. after sorting, or dropping rows).

**`groupby` as split-apply-combine.** `df.groupby(key)` doesn't itself compute anything — it builds
an internal mapping from each distinct key value to the row indices sharing that value (the
*split*). Calling `.agg(func)` (or `.sum()`, `.mean()`, etc.) on the result runs `func` on each
group independently (the *apply*) and stitches the per-group results back into one Series/DataFrame
indexed by the group keys (the *combine*). This is the same three-step shape as a manual
`for key, rows in groups.items(): result[key] = func(rows)` loop — `groupby` just handles the
grouping, missing-key, and multi-column-key bookkeeping for you.

## Algorithm

Split-apply-combine, generically:
1. **Split** — scan the rows once, bucket each row's relevant value(s) by its group-key value(s)
   into a mapping `key -> [values...]`.
2. **Apply** — for each key, run the aggregation function on that key's bucket of values,
   independently of every other key's bucket.
3. **Combine** — collect the per-key results into one output structure, indexed by the group keys.

## From-scratch implementation

Split-apply-combine implemented by hand with a plain Python `dict` of lists — no Pandas — on a toy
orders dataset, then the *same* dataset aggregated with `.groupby().agg()`, verified for an exact
match (`pandas_basics.ipynb`, "11. From-Scratch"):

```python
orders = {
    'region': ['east', 'west', 'east', 'west', 'east', 'north', 'north', 'west'],
    'amount': [100, 200, 150, 300, 50, 400, 250, 100],
}

# SPLIT
groups = {}
for region, amount in zip(orders['region'], orders['amount']):
    groups.setdefault(region, []).append(amount)

# APPLY + COMBINE
manual_sum  = {region: sum(vals)           for region, vals in groups.items()}
manual_mean = {region: sum(vals)/len(vals) for region, vals in groups.items()}

# Same data, pandas
orders_df   = pd.DataFrame(orders)
pandas_sum  = orders_df.groupby('region')['amount'].agg('sum')
pandas_mean = orders_df.groupby('region')['amount'].agg('mean')
```

Actual output:

```
split: {'east': [100, 150, 50], 'west': [200, 300, 100], 'north': [400, 250]}
manual sum : {'east': 300, 'west': 600, 'north': 650}
manual mean: {'east': 100.0, 'west': 200.0, 'north': 325.0}

pandas sum:
 region
east     300
north    650
west     600
Name: amount, dtype: int64

pandas mean:
 region
east     100.0
north    325.0
west     200.0
Name: amount, dtype: float64

sum matches exactly:  True
mean matches exactly: True
```

The manual dict-based split-apply-combine and `.groupby().agg()` produce identical results
(compared as dicts, sorted by key to remove ordering as a confound) — confirming `groupby` is doing
exactly the split/apply/combine steps described above, not something categorically different.

## Practical implementation

`pandas_basics.ipynb` covers the practical surface: `Series`/`DataFrame` creation, inspection
(`.info()`, `.describe()`, `.head()`/`.tail()`), selection (`[]`, `.loc`, `.iloc`), boolean
filtering and `.query()`, adding/renaming/dropping columns, missing-value handling
(`.isna()`, `.dropna()`, `.fillna()`), sorting/`.value_counts()`, `.apply()`/`.map()`, string
(`.str`) and datetime (`.dt`) accessors, and — added in this pass — the split-apply-combine
`.groupby().agg()` mapping above plus the two failure-mode demonstrations below. Section 3's
`.io`/`.iloc`
distinction is the direct practical consequence of the "`DataFrame` has labels, not just positions"
conceptual foundation above.

## Experiment

**Hypothesis:** a hand-rolled split-apply-combine implementation and `.groupby().agg()` compute
identical aggregates on the same input data, for more than one aggregation function (sum and mean),
confirming `groupby` is a direct implementation of the pattern rather than a different algorithm
that happens to look similar.

**Setup:** the 8-row toy `orders` dataset above (3 distinct region keys, uneven group sizes: east=3,
west=3, north=2 rows), aggregated by `sum` and by `mean`, both manually and via `.groupby().agg()`.

**Actual result:** exact match on both aggregations (see output above; `sum matches exactly: True`,
`mean matches exactly: True`).

**Interpretation:** confirms the conceptual claim ("`groupby` is split-apply-combine") with a real,
checked comparison rather than an assertion.

**Limitations:** this toy dataset has no missing group keys, no multi-column grouping, and no
`NaN`s inside the aggregated column — `groupby`'s real implementation handles those cases (e.g.
`NaN` keys dropped by default, multi-key grouping via a list of columns) that this from-scratch
version doesn't attempt to replicate.

## Failure modes

- **Chained indexing → `SettingWithCopyWarning` / `ChainedAssignmentError`.** `df[mask]['col'] = x`
  is *two* separate indexing operations: `df[mask]` first (which may return a copy of the data, not
  a view into the original), then `['col'] = x` assigns into *that* intermediate, about-to-be-
  discarded object. Measured: `city_df[city_df['city'] == 'NYC']['salary'] = 999999` raised
  `ChainedAssignmentError: A value is being set on a copy of a DataFrame or Series through chained
  assignment...` and `city_df.equals(before)` was `True` — the original was silently **not**
  modified, despite no exception stopping execution. (Older Pandas versions raise the same warning
  under the name `SettingWithCopyWarning`; this environment's Pandas 3.0.2 has copy-on-write always
  enabled and reports it as `ChainedAssignmentError`, but the underlying bug and the fix are the
  same.) The fix is a single non-chained call: `city_df.loc[mask, 'salary'] = 999999`, which
  correctly updated both matching rows in the same run.
- **Unexpected dtype coercion — an int column silently becomes float when a NaN is introduced.**
  NumPy has no representation for a missing *integer* (`NaN` is inherently a float/`double`
  concept), and Pandas columns are NumPy-array-backed, so writing a `NaN` into an `int64` column
  forces the **entire column** to upcast to `float64` — not just the one cell. Measured:
  `pd.Series([1,2,3,4])` has `dtype: int64`; after `counts.iloc[2] = np.nan`, `dtype` becomes
  `float64` and every value prints with a trailing `.0`. The same series with a non-`NaN` update
  (`iloc[2] = 999`) keeps `dtype: int64`. Silent dtype coercion breaks code downstream that assumed
  integer semantics — e.g. using the column to index into another array, or an equality check
  after an implicit `int()` cast elsewhere in a pipeline.

## Real-world usage

`DataFrame`/`groupby` are the default interface for tabular data across the ML/DS stack: reading a
CSV/Parquet file returns a `DataFrame`, feature engineering pipelines filter/transform columns with
`.loc`/`.apply`, and per-segment metrics (revenue by region, error rate by model version, click-
through by cohort) are computed with `.groupby(...).agg(...)` before any modeling happens.
`SettingWithCopyWarning`/dtype-coercion bugs are common causes of silently corrupted features in
real pipelines — a filter-then-modify step that looks correct but never actually wrote back to the
DataFrame, or an integer ID column that quietly became float and no longer matches on `==` after a
join with another integer-typed table.

## Mental model

A `DataFrame` is a dict of independently-typed NumPy arrays sharing one row index — `groupby` is
just "build a mapping from key to row-indices, then run your aggregation function once per bucket
and stitch the answers back together by key." Every gotcha in this topic (chained-indexing silently
not writing back, dtype quietly upcasting) traces back to that same underlying fact: a `DataFrame`
column is a real NumPy array with real NumPy constraints, wrapped in a labeling and alignment layer
— the wrapper is convenient, but the underlying array's rules (one dtype per column, ambiguous
copy-vs-view semantics under multi-step indexing) still apply.

## Questions to think about

1. `df.groupby('key')['value'].agg(['sum', 'mean', 'count'])` in one call vs. three separate
   `.groupby('key')['value'].sum()` / `.mean()` / `.count()` calls — do these do the same amount of
   "splitting" work? What would change your answer if `key` had a million distinct values?
2. In the From-scratch experiment, why was it necessary to sort both the manual and pandas results
   by key before comparing them for equality? What would happen if group order differed?
3. You write `df[df['status'] == 'active']['score'] = 0` and get no error, but the values are
   unchanged when you check `df` afterward. Walk through exactly which object got modified instead.
4. A pipeline computes `df['user_id'].astype(int)` after a step that introduced some missing
   `user_id` values. What error or behavior would you expect, and why does the answer trace back to
   the same underlying fact as the dtype-coercion failure mode above?
5. If you only had NumPy (no Pandas) and needed to group a 2-D array's rows by the value in column
   0 and sum column 1 per group, what would your from-scratch approach look like — and which parts
   of Pandas's `groupby` would you be re-deriving by hand?
