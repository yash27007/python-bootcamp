# SQLite

## What you'll learn

Relational data access with Python's built-in `sqlite3` module — schema creation, parameterised
CRUD, `GROUP BY`/`HAVING` aggregation, multi-table `JOIN`, and the Pandas↔SQLite bridge
(`pd.read_sql`/`df.to_sql`). Underneath the practical surface: the B-tree index intuition for why
an indexed lookup beats a full table scan — derived as an $O(\log n)$ vs $O(n)$ argument and then
*measured*, not just asserted, against a real 100,000-row from-scratch Python linear scan.

## Why it matters

A flat CSV re-read into memory every query doesn't scale, has no concurrent-write safety, and no
way to enforce that a foreign key actually points somewhere real. SQLite is the file-based,
server-free relational database that solves all three — and it's where the query-optimizer
concept (an index changes *how* a query is answered, not just how fast) first becomes concrete and
directly measurable in this curriculum.

## Prerequisites

- `02-pandas` (the Pandas↔SQLite bridge assumes `DataFrame` fluency)
- `03-data-manipulation` (`JOIN` is the same relational idea as Pandas `merge`, expressed
  declaratively)

## What you'll build

- A small bookstore schema (`authors`, `books`, `sales`) with CRUD, joins, and aggregation, all
  cells executed
- A real vulnerable-vs-parameterised SQL injection demonstration: an `OR '1'='1'` payload
  leaking all 12 rows through string interpolation, neutralised (0 rows, table intact) through a
  `?`-parameterised query
- A genuinely timed comparison — Python linear scan vs. unindexed SQL vs. indexed SQL — on the
  same 100,000-row dataset, confirmed mechanically via `EXPLAIN QUERY PLAN`'s `SEARCH`/`SCAN`
  distinction

See [`notes.md`](notes.md) for the full write-up, including the B-tree derivation and real
captured timing/injection output, and [`sqlite_basics.ipynb`](sqlite_basics.ipynb) (all cells
executed) for the practical tour.

## Where it shows up in real systems

Every application that queries structured data faster than "scan everything" — a web app's
user lookup, an analytics dashboard's date-range filter, an ORM's `.filter(...)` — relies on the
same B-tree (or similar) index mechanism measured here, usually in a more capable engine
(Postgres, MySQL) built on the identical idea. The parameterised-query defense against SQL
injection demonstrated here is universal across every SQL-backed system, not a SQLite-specific
practice.

## What's next

`08-eda-projects` — the section's capstone: three full exploratory-data-analysis projects that
draw on every tool in this section (NumPy, Pandas, Matplotlib, Seaborn, and the relational access
patterns from this topic) end to end on real datasets.
