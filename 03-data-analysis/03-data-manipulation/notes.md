# Data Manipulation

## Problem

Raw tabular data almost never arrives in the shape an analysis or model needs it in. A sales
table might need to be **combined** with a customer table (joining on a shared key), **summarised**
by region and product (aggregation), **reshaped** from one row per transaction to one row per
day (pivoting), or **checked** for accidentally duplicated records. Data manipulation is the set
of operations that turn "data that exists" into "data in the shape the next step requires" —
without this step, most real analyses can't even start.

## Intuition

Think of three physical operations on paper spreadsheets: **stapling two sheets together by a
shared ID column** (a join/merge), **turning a long list of transactions into a summary table**
(a pivot table), and **checking whether any row got copied twice** (deduplication). Pandas
mechanises exactly these three operations, plus a fourth — **grouped aggregation** (see
`../02-pandas/notes.md` for split-apply-combine) — as the core data-manipulation toolkit. None of
these are new concepts; they're what a careful analyst already does by hand, done at scale and
without transcription errors.

## Why simpler approaches fail

Joining two tables by hand (writing a nested loop over both tables' rows) is exactly what
`pd.merge` does internally, except a hand-written nested loop over `n` and `m` rows costs
`O(n*m)` comparisons unless you build an index first — and even after building an index by hand,
you still have to decide by hand what happens to **unmatched rows** on each side (keep them?
drop them? fill with `NaN`?) for every single join you write, repeatedly re-deriving the same
inner/left/right/outer semantics. Reshaping data (wide ↔ long) by hand means manually tracking
which original column each reshaped value came from — easy to get subtly wrong once there is
more than one "value" column. `pd.merge`/`.pivot_table`/`.melt` exist so this bookkeeping, and
its edge cases, are implemented once, correctly, and reused.

## Conceptual foundation

*(Substituting for "Mathematical foundation" per the template's documented substitution
allowance — this topic's foundation is the relational-join model, not a numeric derivation.)*

**A merge/join is a lookup, not a loop over both tables at once.** The efficient way to join two
tables on a key is: build an index (a `dict`: key → list of matching rows) from one table (the
"build" side, typically the smaller table), then scan the other table once, using the index for
O(1) average-case lookups instead of rescanning the build side per row. This is exactly the
same "build a lookup structure, then look things up in it" idea behind `groupby`'s split step
(see `../02-pandas/notes.md`) — a join and a group-by are both, underneath, "index the data by a
key, then do something per key."

**Join types differ only in what happens to *unmatched* rows.** An inner join keeps only rows
whose key exists on both sides. A left join keeps every left row (filling right-side columns
with `NaN` where there's no match). A right join is the mirror image. An outer join keeps every
row from both sides. The join *algorithm* (build an index, then look up) is identical across all
four — only the handling of "no match found" differs.

**Reshaping is a relabeling of which axis holds "variable identity."** In *wide* format, each
distinct variable (e.g. `math`, `english`, `science` scores) is its own column. In *long* format,
one column holds the variable's *name* and another holds its *value* — the same information,
re-labeled along a different axis. `melt` goes wide→long; `pivot`/`pivot_table` goes long→wide.

## Algorithm

Hash-join (inner join), generically:
1. **Build** — scan the smaller/right table once, bucket its rows into a `dict`: `key -> [rows...]`.
2. **Probe** — scan the larger/left table once; for each row, look up its key in the index and
   emit one output row per match found (zero matches → the row is dropped for an inner join).
3. Left/right/outer joins add one extra step: for rows on the "kept" side(s) with no match, emit
   one output row with the other side's columns filled `NaN` instead of dropping the row.

## From-scratch implementation

A manual inner join implemented with plain Python `dict`s on two toy tables, compared to
`pd.merge`'s result on the same data (`data_manipulation.ipynb`, "2b. From-Scratch"):

```python
left_table = {'id': [1, 2, 3, 4], 'name': ['Alice', 'Bob', 'Carol', 'Dave']}
right_table = {'id': [2, 3, 3, 5], 'dept': ['Eng', 'Mkt', 'Eng', 'HR']}

# Build an index: right-table id -> list of matching row-dicts (the "hash join" build step)
right_index = {}
for rid, dept in zip(right_table['id'], right_table['dept']):
    right_index.setdefault(rid, []).append({'id': rid, 'dept': dept})

# INNER JOIN by hand: for every left row, emit one output row per match found in the index
manual_rows = []
for lid, name in zip(left_table['id'], left_table['name']):
    for match in right_index.get(lid, []):
        manual_rows.append({'id': lid, 'name': name, 'dept': match['dept']})

manual_df = pd.DataFrame(manual_rows)
pandas_df = pd.merge(pd.DataFrame(left_table), pd.DataFrame(right_table), on='id', how='inner')
```

Actual output:

```
Manual inner join:
   id   name dept
0   2    Bob  Eng
1   3  Carol  Mkt
2   3  Carol  Eng

pd.merge inner join:
   id   name dept
0   2    Bob  Eng
1   3  Carol  Mkt
2   3  Carol  Eng

Manual join matches pd.merge exactly: True
```

The manual dict-based join and `pd.merge` produce byte-identical output on the same two toy
tables — confirming `pd.merge` is doing exactly the build-index-then-probe steps described
above. This same toy dataset also demonstrates both failure modes below: `id=3` appears twice
on the right side (a many-to-many key → two output rows for one input row), and `id=1`/`id=4`
(left) and `id=5` (right) have no match and are silently dropped by the inner join.

## Practical implementation

`data_manipulation.ipynb` covers the practical surface, extended in this pass: `groupby`
(split-apply-combine — `.agg()`, `.transform()`, `.filter()`, multi-level grouping),
`pd.merge`/`.merge()` (all four join types, joining on differently-named columns, joining on an
index), `pd.concat` (vertical and horizontal), `.pivot_table`/`pd.crosstab`, reshaping
(`.melt`/`.pivot`/`.stack`/`.unstack`), rolling/expanding windows, and duplicate handling
(`.duplicated()`/`.drop_duplicates()`) — plus, added in this pass, the from-scratch manual join
above and the two failure-mode demonstrations below.

## Experiment

**Hypothesis:** a hand-rolled inner join (build a dict index, then probe) produces byte-identical
output to `pd.merge(..., how='inner')` on the same two tables, including reproducing a
many-to-many row-count explosion and a silent unmatched-row drop, without any special-casing.

**Setup:** the toy `left_table`/`right_table` above — 4 left rows, 4 right rows, one right-side
key (`id=3`) duplicated, one left-side key (`id=1`, `id=4`) with no right match, one right-side
key (`id=5`) with no left match.

**Actual result:** exact match (`Manual join matches pd.merge exactly: True`); the manual join
naturally produced 3 output rows from 4+4 input rows (not 4), automatically reproducing both the
row-multiplication and the row-dropping without being coded specifically to do either.

**Interpretation:** confirms the conceptual claim ("a join is build-index-then-probe, with join
type controlling only what happens to unmatched rows") with a checked comparison, not an
assertion.

**Limitations:** this toy dataset has no `NaN` keys and only one duplicated key on one side —
`pd.merge`'s real implementation also handles multi-column join keys, `NaN`-key behavior, and
join-order-preserving semantics that this from-scratch version doesn't attempt to replicate.

## Failure modes

- **Row-count explosion in a many-to-many join.** When a key repeats on *both* sides, `merge`
  emits the Cartesian product of matching rows for that key. Measured: two 3-row tables sharing
  key `'a'` twice on each side and `'b'` once each produced **5** output rows from **3+3** input
  rows — `'a'` alone contributed `2×2=4` rows. In a real pipeline (e.g. a dimension table with
  an accidental duplicate row), this silently multiplies downstream row counts and any sums
  computed after the join.
- **Inner join silently drops unmatched rows — no error, no warning.** Measured on the toy
  `left_table`/`right_table`: `pd.merge(..., how='inner')` returned 3 rows from 4 left + 4 right
  input rows; `left ids dropped: {1, 4}`, `right ids dropped: {5}` — every one of those rows is
  simply gone from the output, with nothing in the successful, no-exception run to indicate it
  happened. A pipeline that inner-joins a transactions table against a slightly-stale customer
  table will silently lose every transaction from a customer not yet in that table.

## Real-world usage

Joins/merges are how every multi-table analysis starts (orders + customers, events + users,
features + labels); pivot tables and `groupby` aggregation are the standard shape for reporting
and dashboards (revenue by region × month); reshape operations (`melt`) are frequently required
before plotting (most plotting libraries, including Seaborn, prefer long-format data — see
`../06-seaborn/notes.md`); duplicate detection catches double-counted records from retried
ingestion jobs or accidental double-joins. The row-count-explosion and silent-row-loss failure
modes above are two of the most common silent correctness bugs in real data pipelines — a metric
that's quietly too high (explosion) or too low (silent drop) with no error anywhere in the run.

## Mental model

A join is "build an index from one table's key, then look each row of the other table up in it
— join *type* only decides what to do when nothing is found." Every join surprise in this topic
(row-count explosion, silently vanished rows) is a direct, mechanical consequence of that one
sentence: duplicate keys on both sides multiply matches per lookup, and an inner join's "keep
only matches" rule means "no match" is invisible in the output, not flagged.

## Questions to think about

1. In the many-to-many failure-mode demo, why did key `'a'` (appearing twice on each side)
   produce 4 rows while key `'b'` (appearing once on each side) produced only 1? What general
   rule connects the count of matching rows on each side to the count of output rows for that
   key?
2. If you inner-joined the transactions/customers scenario above and got a *plausible-looking*
   but wrong total revenue, what single diagnostic (from this notebook) would you run first to
   check whether rows were silently dropped?
3. `pd.merge(a, b, on='id', how='left')` versus `how='outer'` — for a case where every `id` in
   `a` also exists in `b` (no unmatched rows on the left), are these two calls guaranteed to
   produce the same output? What would make them differ?
4. The from-scratch join builds its index from the *right* table. What would change,
   performance-wise, if the right table were 10 rows and the left table were 10 million rows,
   versus building the index from the left table instead?
5. `wide.melt(id_vars='student', var_name='subject', value_name='score')` followed by
   `.pivot(index='student', columns='subject', values='score')` should round-trip back to
   something close to the original `wide` table. What could make this round-trip fail to recover
   the exact original (consider duplicate `(student, subject)` pairs)?
