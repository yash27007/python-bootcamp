# Data Encoding

## Problem

Most machine learning models are, underneath, arithmetic on numbers: dot products, distance
calculations, gradient updates. A column of category strings — `"red"`, `"blue"`, `"green"` — has
no arithmetic meaning; you cannot compute `"red" + "blue"` or a gradient with respect to a string.
Before a model can use a categorical feature at all, it has to be converted into some numeric
representation — and *which* representation is chosen has real consequences for what the model can
learn and how many features it ends up with.

## Intuition

Imagine describing a T-shirt's color to someone who only understands numbers. You could assign
red=1, blue=2, green=3 — but now "blue is between red and green" and "green minus red equals two"
are both nonsense statements that a model naively fed these numbers would still learn to
optimize against, treating an arbitrary numbering as if it meant something ordinal. The
alternative: describe the shirt with three yes/no questions — "is it red? is it blue? is it
green?" — each answered 1 or 0. No ordering is implied, no arithmetic relationship is invented; the
model sees only "this category is present" or "it isn't." That's one-hot encoding. When the
categories genuinely *do* have an order (small/medium/large), the single-integer approach becomes
correct again — the encoding choice should follow the actual structure of the category, not be
picked by default.

## Why simpler approaches fail

- **Passing raw strings to a model directly** fails outright for the vast majority of ML
  algorithms — linear models, SVMs, neural networks, distance-based methods (KNN, k-means) require
  numeric input and cannot operate on strings at all.
- **A single arbitrary integer per category (naive label encoding) for a *nominal* feature**
  (colors, cities, product categories — no inherent order) silently invents a false ordinal
  relationship and a false magnitude relationship that a model trained on this encoding will pick
  up on, degrading its actual understanding of the feature.
- **One-hot encoding applied without checking cardinality** — the "safe" choice for nominal data —
  creates one new column per distinct category. This works fine for a handful of categories and
  becomes a serious problem once a column has hundreds or thousands of distinct values (a ZIP
  code, a user ID) — measured concretely below.

## Algorithm

**One-hot encoding:**
1. Find every distinct category in the column.
2. Create one new binary column per distinct category.
3. For each row, set the column matching that row's category to 1 and all others to 0.

**Ordinal / label encoding:**
1. Establish (or assume) an order over the distinct categories.
2. Assign each category the integer corresponding to its rank in that order.

**Target-guided ordinal encoding:**
1. Group by category and compute the mean of the target variable within each group.
2. Replace each category with its group's target mean.

## From-scratch implementation

`03-data-encoding.ipynb` builds one-hot encoding by hand: sort the distinct categories, then for
each one build a binary column with `(series == category).astype(float)` — a plain Python loop
over categories, no encoding library call — and compares the result against both
`pd.get_dummies` and `sklearn.preprocessing.OneHotEncoder` on the identical `color` column:

```
   color_blue  color_green  color_purple  color_red  color_yellow
0         0.0          0.0           0.0        1.0           0.0
1         1.0          0.0           0.0        0.0           0.0
2         0.0          1.0           0.0        0.0           0.0
3         0.0          0.0           0.0        0.0           1.0
4         0.0          0.0           1.0        0.0           0.0
5         0.0          0.0           0.0        1.0           0.0
6         0.0          1.0           0.0        0.0           0.0

manual vs pd.get_dummies match: True
manual vs sklearn OneHotEncoder match: True
```

The manual loop and both library functions produce identical binary matrices — `pd.get_dummies`
and `OneHotEncoder` are exactly this loop, with column-naming and sparse-matrix bookkeeping added.

## Practical implementation

The notebook covers `OneHotEncoder` (including transforming *new* unseen-at-fit data),
`LabelEncoder`, `OrdinalEncoder` (with an explicit `categories=` order for genuinely ordinal data
like shirt sizes), and target-guided ordinal encoding (mapping each category to its group's target
mean via `groupby(...).mean().to_dict()` + `.map(...)`) on the `tips` dataset.

## Experiment

**Hypothesis:** one-hot encoding a high-cardinality column produces a dimensionality blow-up
proportional to the number of distinct categories, and a fitted encoder cannot represent a
category it never saw during fitting.

**Setup 1 — cardinality:** 5,000 synthetic rows of random 5-digit ZIP codes, one-hot encoded with
`pd.get_dummies`.

**Actual result 1:**
```
Rows: 5000   Distinct ZIP codes: 4887
One-hot encoded shape: (5000, 4887)  (one column per distinct ZIP code)
Memory footprint of the one-hot matrix: 24.44 MB
Memory footprint of the original single column: 0.0650 MB
```

**Setup 2 — unseen categories:** an `OneHotEncoder` fit only on `{red, blue, green, yellow,
purple}`, then asked to transform a row containing `"black"`, both with the default
`handle_unknown='error'` and with `handle_unknown='ignore'`.

**Actual result 2:**
```
Fitted encoder rejects an unseen category:
  Found unknown categories ['black'] in column 0 during transform

With handle_unknown='ignore', an unseen category encodes to all zeros:
  [[0. 0. 0. 0. 0.]]  columns: ['color_blue', 'color_green', 'color_purple', 'color_red', 'color_yellow']
```

**Interpretation:** cardinality directly and near-linearly drives the output width (4,887 distinct
values → 4,887 columns), a **376x expansion in memory** for this single feature (24.44 MB vs.
0.065 MB) — the practical reason high-cardinality columns need a different treatment (frequency
encoding, target encoding, hashing, or grouping rare categories). The unseen-category result
confirms a fitted encoder is a strict lookup table over training-time categories, and shows the
concrete difference between the default "fail loudly" behavior and the opt-in "silently encode as
all-zero" fallback.

**Limitations:** synthetic uniform-random ZIP codes (real ZIP codes are not uniformly distributed
and have far more structure/repetition than this worst-case simulation); a single toy 5-category
encoder for the unseen-category demo.

## Failure modes

- **High-cardinality columns exploding dimensionality** — demonstrated concretely above: 4,887
  distinct ZIP codes become 4,887 columns and a 376x memory increase for one feature. Beyond the
  memory cost, this also worsens the curse of dimensionality for any distance-based or
  regularized model and makes most of the resulting columns nearly always zero (extremely sparse,
  low-signal-per-column).
- **Unseen categories at inference time breaking a fitted encoder** — demonstrated concretely
  above: the default `OneHotEncoder` raises `ValueError` on any category not seen during `.fit()`,
  which is exactly what happens in production when a new category appears after training (a new
  product SKU, a new city) unless `handle_unknown='ignore'` (or an explicit "unknown" bucket
  strategy) is chosen deliberately, in advance, as part of the pipeline design.
- **Naive label/ordinal encoding on genuinely nominal data** invents a false numeric ordering and
  false magnitude relationships between categories that have none, silently degrading any model
  sensitive to the values' magnitude or order (linear models, distance-based methods) — one-hot
  encoding exists specifically to avoid this for nominal features.

## Real-world usage

Every model-serving pipeline that accepts categorical input in production faces the unseen-category
problem eventually — new users, new product categories, new geographic regions appear after the
encoder was fit, and a pipeline that hasn't decided what to do about it fails at inference time
instead of training time, which is a worse place for a bug to surface. High-cardinality categorical
features (user IDs, product IDs, ZIP/postal codes) are common in recommendation systems and
ad-tech, where target encoding, feature hashing, or learned embeddings (mapping each category to a
dense low-dimensional vector, as in `06-deep-learning`'s embedding layers) replace naive one-hot
encoding specifically to avoid the dimensionality blow-up measured above.

## Mental model

One-hot encoding turns "which category" into "a set of yes/no questions," deliberately discarding
any notion of order or magnitude — the right choice exactly when none exists. Ordinal encoding
does the opposite: it deliberately imposes order, the right choice exactly when that order is
real. Choosing between them is a question about the *data's actual structure*, not a matter of
convention — and either encoder, once fit, is only ever as complete as the categories it was shown
during fitting.

## Questions to think about

1. The from-scratch one-hot loop and `pd.get_dummies` produced bit-for-bit identical output. What
   is `pd.get_dummies` actually providing beyond that loop — what would you have to add to the
   manual version to make it a drop-in production replacement?
2. `OrdinalEncoder(categories=[['extra-small','small','medium','large','extra-large']])` requires
   the order to be specified explicitly. What would happen — silently, without an error — if you
   used plain `LabelEncoder` on the same ordinal column instead, and why is that dangerous
   specifically because it wouldn't crash?
3. Given the measured 376x memory blow-up for 4,887 ZIP codes, sketch (in words) how target
   encoding would produce a single numeric column instead — and what new failure mode (a form of
   leakage) that introduces if the target means are computed before a train/test split.
4. `handle_unknown='ignore'` encodes an unseen category as all-zeros — indistinguishable from "this
   row is missing every category." Why might that be a worse silent failure, in some applications,
   than the default `ValueError`?
5. Target-guided ordinal encoding assigns each category a single number (its target mean). Under
   what circumstance would you expect this to outperform one-hot encoding for a high-cardinality
   column, and what information does it necessarily discard that one-hot encoding preserves?
