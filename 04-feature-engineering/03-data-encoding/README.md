# Data Encoding

## What you'll learn

Why most ML models need numeric input, not raw category strings, and how one-hot, ordinal, and
target-guided encoding each represent categorical structure differently. One-hot encoding built
by hand with a plain Python/Pandas loop — one binary column per distinct category — checked
element-for-element against both `pd.get_dummies` and `sklearn.preprocessing.OneHotEncoder`.

## Why it matters

Encoding choice is a question about the data's actual structure, not a default: treating a
nominal category as ordinal invents a false numeric relationship a model will still learn to
exploit. High-cardinality categorical columns (ZIP codes, user IDs) blow up one-hot encoding's
output width, and a fitted encoder is a strict lookup table — a category unseen during training
breaks it at inference time unless that's handled deliberately.

## Prerequisites

- `03-data-analysis/02-pandas` (`DataFrame`, `groupby`, `.map`)
- `01-missing-values` (fitted-transformer discipline — fit on train, apply to test — is the same
  pattern used for encoders)

## What you'll build

- A manual one-hot encoder (Python loop over distinct categories) verified bit-for-bit against
  `pd.get_dummies` and `OneHotEncoder` on the same data
- A real measured dimensionality/memory blow-up from one-hot-encoding a high-cardinality synthetic
  ZIP-code column (4,887 distinct values → 4,887 columns, a 376x memory increase)
- A concrete unseen-category failure: a fitted `OneHotEncoder` raising `ValueError` on an unseen
  category, then fixed with `handle_unknown='ignore'`

See [`notes.md`](notes.md) for the full write-up including captured output, and
[`03-data-encoding.ipynb`](03-data-encoding.ipynb) (all cells executed) for the practical tour.

## Where it shows up in real systems

Every model-serving pipeline that accepts categorical input eventually meets a category it never
saw during training — new users, new product SKUs, new regions — and a pipeline without an
explicit decision for that case fails at inference instead of training. High-cardinality features
in recommendation systems and ad-tech typically move past naive one-hot encoding to target
encoding, hashing, or learned embeddings for exactly the reason measured here.

## What's next

`04-handling-imbalanced-dataset` — a different kind of feature-engineering problem: when the
*target* classes, not the features, are unevenly represented.
