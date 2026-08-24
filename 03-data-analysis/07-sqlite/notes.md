# SQLite

## Problem

Every prior topic in this section (`01-numpy` through `06-seaborn`) works on data that already
fits comfortably in memory as an array or a `DataFrame`. Real applications don't get that luxury:
data is written continuously by more than one process, is too large to reload in full for every
question, has relationships between entities (an author has many books, a book has many sales)
that a single flat table can't express without duplication, and must survive being queried by
a *filter* — "find the rows where X" — without re-scanning everything every time. SQLite exists
to answer exactly this: efficient, safe, relational access to data that lives on disk, addressed
by declarative queries instead of manual file parsing.

## Intuition

A CSV file is a phone book with no index — to find one name you start at page 1 and read every
line until you find it (or reach the end and conclude it isn't there). A relational database with
an index on the right column is the same phone book, but with tabs cut into the page edges
alphabetically — you jump straight to "M" and scan a handful of nearby entries. SQLite is that
indexed phone book, built into Python's standard library (`sqlite3`), stored as a single file, with
no separate server process to run.

Concretely: `03-data-manipulation`'s `merge`/`join` operations on Pandas `DataFrame`s are the same
relational idea (combine `authors` and `books` on a shared key) — SQLite's `JOIN` is that same
operation, expressed declaratively and evaluated by a query planner that can choose to use an
index instead of comparing every row pair.

## Why simpler approaches fail

- **A flat CSV re-read every query doesn't scale.** `pd.read_csv(...)` followed by
  `df[df['col'] == value]` re-parses the entire file from disk and re-scans every row, every time,
  for every query — there is no way to "remember" that the file is sorted or indexed across calls
  unless the whole file is kept in memory and re-sorted by hand.
- **No concurrent-write safety.** Two processes appending to the same CSV at the same time can
  interleave writes and corrupt the file — there is no locking protocol. SQLite provides
  transactional writes (`conn.commit()`) with file-level locking so concurrent access doesn't
  silently corrupt data.
- **No relational integrity.** A CSV has no concept of "this `author_id` in `books.csv` must exist
  in `authors.csv`" — a typo'd foreign key just silently produces an orphaned row. SQLite supports
  `REFERENCES` constraints (used in this topic's `books.author_id INTEGER REFERENCES authors(id)`)
  that make the relationship an enforceable property of the schema, not a convention the analyst
  has to remember.
- **No query optimizer.** Pandas boolean filtering (`df[df['col'] == value]`) is *always* a full
  column scan — there is no way to tell Pandas "this column is indexed, jump straight to the
  matching rows" the way `CREATE INDEX` + `WHERE` lets SQLite's query planner do (measured below).

## Mathematical foundation

**B-tree index intuition — why an indexed lookup beats a full scan.**

An unindexed `WHERE column = value` query has no choice but to check every row: with $n$ rows,
that's $n$ comparisons in the worst case — $O(n)$. This is exactly what the from-scratch
`linear_scan` function below does over a plain Python list, and exactly what SQLite itself falls
back to on a table with no index (confirmed below: `SCAN` in the query plan).

A B-tree index keeps the indexed column's values in **sorted order**, arranged as a balanced tree
where each node holds a small, bounded number of keys and pointers to child subtrees. Finding a
value means:

1. Compare the target against the keys in the root node — this determines which one of the
   root's children could possibly contain the target (everything in that subtree is bounded
   between two adjacent keys in the root).
2. Descend into that one child. Repeat: compare, pick the one subtree that could contain the
   target, descend.
3. Stop when a leaf is reached — the value is either there or it doesn't exist in the table.

Each step throws away all but one branch of a bounded-size split, so the tree's height — the
number of steps needed — is $\log_b(n)$, where $b$ is the branching factor (typically large,
because a database page holds many keys). This is the same divide-and-eliminate structure as
binary search over a sorted array (branching factor 2), just with a much larger branching factor
because each tree node is sized to fill one disk page efficiently. The practical consequence: the
number of comparisons needed to find one row grows *logarithmically* with the table size instead
of linearly — doubling the table adds one more level to the tree, not twice the work.

$$
T_{\text{scan}}(n) = O(n) \qquad T_{\text{indexed}}(n) = O(\log_b n)
$$

For $n = 100{,}000$ and a realistic B-tree branching factor of a few hundred (SQLite pages are
4 KB by default, holding many index entries per page), $\log_b n$ is on the order of 2–3 node
visits versus up to 100,000 row comparisons for a full scan — a gap that only widens as $n$ grows,
which is exactly what's measured below.

## Algorithm

What `CREATE INDEX idx_value ON lookup(value)` followed by `SELECT id FROM lookup WHERE value = ?`
does, generically:

1. `CREATE INDEX` builds a separate B-tree structure keyed on `value`, each leaf entry pointing
   back to the corresponding row in the base table — a one-time $O(n \log n)$ cost paid at index
   creation, not at query time.
2. On `SELECT ... WHERE value = ?`, SQLite's query planner checks whether an index exists on the
   filtered column. If it does, it uses the index's B-tree to descend directly to the matching
   leaf (`SEARCH` in `EXPLAIN QUERY PLAN`) instead of reading every row in table order (`SCAN`).
3. Without a usable index, the planner has no choice but to scan the whole table — this is not a
   SQLite limitation, it is the same fundamental fact the from-scratch demo shows: there is no way
   to "jump" to a matching row without some sorted/indexed structure telling you where to jump to.

## From-scratch implementation

Section 8 of `sqlite_basics.ipynb` builds a genuine linear scan over a plain Python list —
`linear_scan(records, target_value)`, checking each `(id, value)` tuple in order — and times it
against an SQLite query on the *identical* 100,000-row dataset, both without and with a B-tree
index (`CREATE INDEX idx_value ON lookup(value)`). This is a real, measured comparison, not a
theoretical one:

```
Rows: 100,000   target value: 824501

Python linear scan (list):        0.845 ms  -> id=50000
SQLite, NO index (table scan):    2.403 ms  -> id=50000
SQLite, WITH index (B-tree):      0.082 ms  -> id=50000

Speedup, indexed vs Python linear scan: 10x
Speedup, indexed vs SQLite table scan:   29x

EXPLAIN QUERY PLAN with the index present: [(3, 0, 62, 'SEARCH lookup USING INDEX idx_value (value=?)')]
```

`EXPLAIN QUERY PLAN` is the direct, mechanical confirmation of the Mathematical foundation
section above: once the index exists, SQLite's own plan literally says `SEARCH ... USING INDEX`
instead of `SCAN` — the query planner made the $O(\log n)$ choice, not just a faster constant
factor on the same $O(n)$ work.

## Practical implementation

`sqlite_basics.ipynb` covers the full practical surface, all cells executed:

1. **Creating tables** — schema with `PRIMARY KEY`, `REFERENCES` foreign keys.
2. **INSERT** — `executemany` with parameterised `?` placeholders.
3. **SELECT** — `WHERE`, `ORDER BY`, `GROUP BY`/`HAVING` aggregation.
4. **JOIN** — inner join across two tables, three-table join with aggregation.
5. **UPDATE / DELETE** — conditional row mutation and deletion, `conn.commit()`.
6. **Pandas ↔ SQLite integration** — `pd.read_sql(query, conn)` to load a query result directly
   into a `DataFrame` (mapping straight back to `02-pandas`), and `df.to_sql(...)` to write a
   `DataFrame` back to a table.
7. **SQL injection — vulnerable vs. fixed** (Failure modes below).
8. **Index vs. linear scan, measured** (From-scratch above).

## Experiment

**Hypothesis:** an SQL query on an indexed column will be measurably faster than an equivalent
manual Python linear scan and an unindexed SQL query over the same data, and the gap will show up
as a change in SQLite's own query plan (`SEARCH` vs `SCAN`), not just a wall-clock difference.

**Setup:** 100,000 `(id, value)` rows generated once (`random.seed(0)`), used identically for all
three methods: (1) a hand-written `linear_scan` over the Python list, (2) an SQLite table with no
index, (3) the same SQLite table after `CREATE INDEX idx_value ON lookup(value)`. All three search
for the same target value, guaranteed present at the midpoint of the generation order.

**Actual result:** indexed SQL query = 0.082 ms, Python linear scan = 0.845 ms (10x slower),
unindexed SQL table scan = 2.403 ms (29x slower than indexed) — full output in From-scratch above.
`EXPLAIN QUERY PLAN` confirmed the mechanism: `SEARCH lookup USING INDEX idx_value (value=?)`.

**Interpretation:** confirms both halves of the hypothesis — the indexed query is fastest, and the
*reason* is visible directly in the query plan, not inferred from timing alone. The unindexed
SQLite scan was slower than the raw Python linear scan despite both being $O(n)$ — SQLite pays
overhead per row (cursor/type handling) that a tight Python loop over native tuples doesn't, which
is itself a useful, honestly-reported observation: indexing is what changes the *complexity class*,
not "SQL is inherently faster than Python."

**Limitations:** this is a single-column equality lookup on an in-memory (`:memory:`) database — a
disk-backed database under concurrent load, a range query, or a query on a low-cardinality column
(where the index barely narrows the search) would show a smaller gap. The measured numbers are one
run on one machine and will vary with hardware, but the qualitative order (indexed ≪ scan) is
structural, not incidental, given the $O(\log n)$ vs $O(n)$ argument above.

## Failure modes

- **SQL injection from unparameterized queries.** Building a query by string interpolation lets
  attacker-controlled input change the query's *structure*, not just its data. Demonstrated
  directly: `f"SELECT title, price FROM books WHERE title = '{malicious_input}'"` with
  `malicious_input = "x' OR '1'='1"` produced the query
  `SELECT title, price FROM books WHERE title = 'x' OR '1'='1'` — the injected `OR '1'='1'` is
  always true, so the filter was neutralised entirely and **all 12 rows** were returned instead of
  zero. The fix — `cursor.execute('SELECT title, price FROM books WHERE title = ?', (user_input,))`
  — treats the same payload as a single literal string value, never as SQL syntax: the identical
  payload through the parameterised form returned **0 rows**, and a destructive
  `"'; DROP TABLE books; --"` payload through the same parameterised query left the table's row
  count unchanged. The rule this demonstrates: string-format a query and user input can rewrite
  the query; pass it as a bound parameter and it cannot, structurally, regardless of its content.
- **Missing an index leaves a query slow without an obvious symptom.** A query with no index still
  *returns the correct answer* — nothing errors, nothing warns — it is only slow, and only exactly
  as slow as the table is large. On a small development dataset (tens or hundreds of rows) an
  unindexed `WHERE` clause is imperceptible; the same query against a production table with
  millions of rows becomes a multi-second scan with no code change, no error message, and no
  obvious cause unless someone thinks to run `EXPLAIN QUERY PLAN` and notices `SCAN` where a
  `SEARCH` was expected.

## Real-world usage

Every production system that serves queries against structured data faster than "read the whole
table" — a web app's user lookup by ID or email, an analytics dashboard filtering millions of
event rows by date range, an ORM's `.filter(...)` call — relies on exactly this index mechanism,
usually in a more capable engine (Postgres, MySQL) with the same underlying B-tree (or similar
tree/hash structure) idea. SQLite itself is embedded directly in browsers, mobile apps (iOS/Android
local storage), and is frequently the first real database a data-analysis pipeline reaches for
once a CSV genuinely stops scaling — `pd.read_sql`/`df.to_sql` (used in this topic) is the standard
bridge between exploratory Pandas work and a persistent, queryable store. Parameterised queries are
the universal defense against SQL injection across every SQL-backed system, not a SQLite-specific
practice — the exact same `?`/`%s`-placeholder discipline applies to Postgres, MySQL, and every ORM
built on top of them.

## Mental model

An index doesn't make one query faster by a constant factor — it changes *which class of problem*
the lookup is: from "check everything" ($O(n)$) to "eliminate most of the search space at each
step" ($O(\log n)$), and `EXPLAIN QUERY PLAN`'s `SEARCH`/`SCAN` distinction is the mechanical proof
of which class you're in. A parameterised `?` isn't a style preference — it's the only syntactic
guarantee that user input can change a query's *data* and never its *structure*.

## Questions to think about

1. The unindexed SQLite table scan (2.403 ms) was slower than the raw Python `linear_scan`
   (0.845 ms) over the same 100,000 rows, even though both are $O(n)$. What per-row overhead does
   SQLite likely pay that a tight Python loop over native tuples doesn't?
2. `CREATE INDEX` itself costs $O(n \log n)$ to build. Under what workload (read-heavy vs.
   write-heavy) does that one-time cost stop being worth paying, and why?
3. The SQL injection demo used `OR '1'='1'` to bypass a filter, not `DROP TABLE`. Why is a
   read-oriented bypass often the more dangerous real-world injection — think about what a `WHERE`
   clause is usually protecting in a multi-user application.
4. `EXPLAIN QUERY PLAN` showed `SEARCH` after indexing `value`. If a query instead filtered on a
   column with only two distinct values (e.g. a boolean `in_stock` flag) across a million rows,
   would you expect SQLite's planner to use an index on that column? Why might the planner
   reasonably choose a scan instead, even with an index available?
5. `df.to_sql(...)` and `pd.read_sql(...)` bridge Pandas and SQLite directly. Given the measured
   gap between an indexed and unindexed lookup, what would change about how you design a table's
   indexes if you knew the table would primarily be *read* via `pd.read_sql` filters rather than
   scanned in full into a `DataFrame`?
