# Reading Data

## What you'll learn

Why getting external data into a usable `DataFrame` needs answers to questions raw bytes don't
answer for you — encoding, delimiter/quoting, and per-column type inference — via a minimal
hand-rolled quote-aware CSV parser compared to `pd.read_csv`, plus the full practical surface:
CSV, JSON, Excel, SQLite, Parquet, and HTML tables.

## Why it matters

Every ML/DS pipeline begins with a read step, and ID columns (customer IDs, SKUs, zip codes) are
one of the most common sources of the silent-type-coercion bug in production pipelines — they
look numeric but aren't meant to be treated as numbers. This topic's Failure modes (an
encoding mismatch, a numeric-looking ID column silently losing its leading zeros) are exactly
those recurring, silent bugs, reproduced and fixed here.

## Prerequisites

- `02-pandas` (reading returns a `DataFrame`; this topic is about getting one correctly, not
  what to do with it once you have it)

## What you'll build

- A minimal quote-aware CSV parser (handles an embedded comma inside a quoted field) compared
  against a naive `split(',')` parser and against `pd.read_csv`, on the same text
- A real, executed reproduction of a UTF-8/Latin-1 encoding mismatch (`UnicodeDecodeError`) and
  its fix
- A real, executed demonstration of a numeric-looking ID column (`'00123'`) silently losing its
  leading zeros under default type inference, and the `dtype=str` fix

See [`notes.md`](notes.md) for the full write-up including real captured output, and
[`data_reading.ipynb`](data_reading.ipynb) (all cells executed) for the practical tour — CSV,
JSON, Excel, SQLite, Parquet, HTML tables, a real CSV-vs-Parquet performance comparison, and the
from-scratch/failure-mode sections above.

## Where it shows up in real systems

Training data usually starts as CSV/Parquet on disk or query results from a warehouse;
production features are frequently read from sources with unpredictable encodings. Parquet's
popularity over CSV in big-data pipelines is a direct consequence of the dtype-preservation
issue covered here — Parquet stores the dtype alongside the data, so it can't silently
reinterpret a string ID column as an integer the way a CSV round-trip can.

## What's next

`05-matplotlib` — turning the DataFrames this topic reads into plots, starting from the
Figure/Axes hierarchy every other plotting tool (including Seaborn) builds on.