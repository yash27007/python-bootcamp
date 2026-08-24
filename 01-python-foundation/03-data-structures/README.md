# 03 – Data Structures

Detailed notes (Big-O complexity of list/dict/set/tuple, hashing derived step-by-step, why dict/set
lookup is O(1) amortized): [notes.md](notes.md)

Real, actually-executed notebook covering `list`/`tuple`/`set`/`dict` creation and operations, a
from-scratch hash table verified against a real `dict`, a timed hash-table-vs-list-scan comparison,
a degenerate-hash-function collision demo, and an unhashable/mutable-key failure demo, all with real
pasted output: [data_structures.ipynb](data_structures.ipynb)

## What you'll learn

Why choosing the right data structure for an access pattern matters — not as a rule of thumb, but
derived from what each structure actually costs. Big-O complexity for the core operations on
`list`, `tuple`, `set`, and `dict`; the hashing mechanism (hash function → bucket → collision
resolution) that makes `dict`/`set` lookup O(1) on average, built by hand and measured against a
list's O(n) linear search on identical data.

| Topic | Status |
|-------|--------|
| Lists: creation, indexing/slicing, mutation, sorting | ✅ Complete |
| Tuples: creation, unpacking, hashability, `namedtuple` | ✅ Complete |
| Sets: creation, mutation, set algebra | ✅ Complete |
| Dicts: creation, access, iteration, merging, `defaultdict`/`Counter` | ✅ Complete |
| Mathematical foundation: Big-O derivation, hashing derived from first principles | ✅ Complete |
| From-scratch: separate-chaining hash table, verified against a real `dict` | ✅ Complete |
| Experiment: real timed hash-table vs. list-linear-search comparison | ✅ Complete |
| Failure modes: degenerate hashing, unhashable/mutable keys (loud and silent) | ✅ Complete |

## Why it matters

Using a `list` for membership testing or key-based lookup is one of the most common real
performance bugs in Python code — it silently works at small scale and silently becomes the
bottleneck as data grows, because the cost is O(n) per lookup rather than the O(1) a `set`/`dict`
would give. Understanding *why* dict/set lookup is fast (not just that it is) is also what makes
their failure modes — hash collisions, unhashable/mutable keys — predictable instead of mysterious.

## Prerequisites

`01-basics` (names as references, mutable vs. immutable — directly relevant here: hashability
requires immutability) and `02-control-flow` (loops, used throughout the from-scratch hash table).

## What you'll build

- A separate-chaining hash table implemented in plain Python, verified key-for-key against a real
  `dict` on 2,000 entries.
- A real, measured timing comparison: the from-scratch hash table's lookup vs. a Python list's
  linear search on the same 20,000-element dataset (~219x faster, measured).
- A reproduced collision-degradation failure: an intentionally bad hash function that collapses
  the table's O(1) average case toward the O(n) worst case, measured (~132x slower than the
  well-spread version on the same data).
- A reproduced unhashable-key `TypeError`, and a reproduced *silent* bug from a custom object whose
  hash-relevant field is mutated after being used as a dict key.

## Where it appears in real systems

Deduplication and membership testing at scale (`if x in seen_set`), caching/memoization
(`functools.lru_cache` and hand-rolled caches are hash tables), word/token counting and
groupby-style aggregation (`Counter`, `defaultdict`), and database hash indexes (the same
hash-to-bucket mechanism at the storage-engine level, contrasted with B-tree indexes later in
`03-data-analysis/07-sqlite`). CPython's per-process hash-seed randomization for strings exists
specifically because the collision-degradation failure mode demonstrated here is a real,
exploitable denial-of-service vector if an attacker can predict and control input keys.

## What's next

`04-functions` — code reuse and abstraction; closures (which rely on Python's scope rules, not
this topic's hashing) and first-class functions.
