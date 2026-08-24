# Data Manipulation

## What you'll learn

A merge/join as a build-index-then-probe lookup — implemented by hand with plain Python `dict`s
on two toy tables, then compared to `pd.merge`'s output on the same data — plus the full
practical reshaping surface: `groupby` aggregation, joins/concatenation, pivot tables/crosstabs,
melt/pivot/stack/unstack, rolling/expanding windows, and duplicate handling.

## Why it matters

Almost no real analysis starts with data already in the shape it needs — tables need to be
joined, summarised, and reshaped first. This topic's Failure modes (a many-to-many join silently
multiplying row counts, an inner join silently dropping unmatched rows) are two of the most
common silent correctness bugs in real data pipelines — a metric that's quietly too high or too
low with no error anywhere in the run.

## Prerequisites

- `02-pandas` (this topic extends `groupby`/split-apply-combine to joins, pivoting, and reshaping)

## What you'll build

- A manual inner join implemented with plain Python `dict`s (build an index, then probe) on two
  toy tables, verified byte-identical to `pd.merge(..., how='inner')` on the same data
- A real, executed demonstration of many-to-many join row-count explosion (3+3 input rows → 5
  output rows from one repeated key) and inner-join's silent unmatched-row drop

See [`notes.md`](notes.md) for the full write-up including real captured output, and
[`data_manipulation.ipynb`](data_manipulation.ipynb) (all cells executed) for the practical tour
— groupby, merging/joining, concatenation, pivot tables, reshaping, rolling windows, duplicates,
and the from-scratch/failure-mode sections above.

## Where it shows up in real systems

Every multi-table analysis starts with a join (orders + customers, events + users, features +
labels); pivot tables and grouped aggregation are the standard shape for dashboards and reports;
reshape operations (`melt`) are frequently required before plotting (see `06-seaborn`, which
prefers long-format data). The row-count-explosion and silent-row-loss failure modes covered here
are recurring, genuinely subtle sources of silently wrong metrics in real pipelines.

## What's next

`04-data-reading` — getting external data (CSV/Excel/JSON/SQL) into the DataFrames this topic
manipulates, correctly handling encoding, delimiters, and types.