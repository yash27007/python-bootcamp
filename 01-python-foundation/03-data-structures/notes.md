# Data Structures — Lists, Tuples, Sets, Dicts

## Problem

A program almost always needs to hold more than one value at a time, and different tasks put
different pressure on how that collection is used: appending items as they arrive, looking
something up by position, checking whether a value is already present, or looking a value up by a
meaningful key instead of a position. Python could get away with a single collection type — a
list — and technically support all of these. The question this topic answers is: what does that
choice cost, and what structure should be reached for instead, for a given access pattern?

## Intuition

Think of the four built-in collections as four different physical organizations of the same pile
of items:

- **`list`** — a numbered row of boxes. Finding "the 3rd box" is instant (jump straight to it).
  Finding "the box with the red ball in it" means checking boxes one at a time from the start
  until it's found — there is no shortcut.
- **`tuple`** — the same numbered row, but nailed shut once built. Nothing can be added, removed,
  or swapped in place. Useful whenever "this data should not change after creation" is itself part
  of the contract (a coordinate, a database row, a dict key).
- **`set`** — no numbering at all. Instead, every item is dropped into one of many labeled bins,
  and the bin an item belongs to is computed directly from the item's own content. Checking
  "is the red ball in here" means computing which bin it would be in and looking in only that bin
  — no scanning the whole pile.
- **`dict`** — the same bin system as a `set`, except each bin holds a `(key, value)` pair instead
  of just a key. "Look up Alice's age" computes Alice's bin directly and reads the value stored
  there.

The list's cost model is "no setup, but every search walks the whole pile." The set/dict's cost
model is "some setup (computing which bin), but every search jumps straight to one bin." That
trade — no setup vs. some setup, in exchange for search speed — is the entire content of this
topic.

## Why simpler approaches fail

**"Just use a list for everything."** It works, and for small collections the constant-factor
difference is invisible. It stops working the moment two things happen together: the collection
gets large, and the operation performed is a *lookup by value* (membership testing, deduplication,
or key-based access) rather than a lookup by position. `x in some_list` must, in the worst case,
compare `x` against every element — checking membership in a list of `n` items costs O(n) time,
and doing that check repeatedly (e.g. once per row while deduplicating a larger dataset) costs
O(n²) total. A `set`/`dict` does not eliminate work by magic — it *relocates* the work: instead of
paying a linear scan on every lookup, it pays a small, roughly constant amount of work per lookup
by doing the equivalent of "pre-sorting into bins" once, at insertion time. The Experiment section
below measures this difference directly rather than asserting it.

## Mathematical foundation

**Big-O complexity of the four built-in structures**, `n` = number of elements:

| Operation | `list` | `tuple` | `set` | `dict` |
|---|---|---|---|---|
| Index access (`x[i]`) | O(1) | O(1) | n/a | n/a (key lookup, below) |
| Append / add | O(1) amortized | n/a (immutable) | O(1) amortized | O(1) amortized |
| Insert at front / arbitrary position | O(n) | n/a | n/a | n/a |
| Delete at front / arbitrary position | O(n) | n/a | O(1) amortized | O(1) amortized |
| Membership test (`x in ...`) / lookup by key | O(n) | O(n) | O(1) amortized | O(1) amortized |
| Worst case for set/dict lookup | — | — | O(n) | O(n) |

**Why list append is O(1) amortized, not O(1) exactly.** A Python list is backed by a contiguous
array with some spare capacity. Appending when there's spare capacity is a genuine O(1) write.
When the array is full, Python allocates a new, larger array (roughly 1.125× the old size) and
copies every existing element into it — an O(n) operation — then the append proceeds. That
expensive copy happens rarely enough (each one buys room for many more O(1) appends before the
next one) that the *average* cost per append, over a long sequence of appends, works out to O(1).
This is "amortized" cost: not every individual operation is cheap, but the average is.

**Why insert-at-front is O(n).** `list.insert(0, x)` must shift every existing element one slot to
the right to make room at index 0 — there is no way around touching all `n` elements, because a
Python list's index *is* its physical array position.

**Deriving why dict/set lookup is O(1) amortized — from hashing.** The mechanism, step by step:

1. Every Python object that can be a dict key/set member has a `__hash__` method producing an
   integer, `hash(key)`, computed from the key's *content* (its value), not its memory address, in
   O(1) time for the built-in immutable types (a bounded number of arithmetic operations on the
   value's bits — genuinely constant for a fixed-size type like `int`, `float`, or a short `str`).
2. A dict/set maintains an internal array of `m` **buckets** (`m` is a power of two, resized as the
   structure grows, similarly amortized to list's resizing above). Inserting or looking up `key`
   computes `bucket_index = hash(key) % m` (a bitmask in CPython, equivalently cheap) — again O(1),
   independent of how many keys are already stored.
3. Two different keys can map to the same `bucket_index` — a **collision**. Python (this repo's
   from-scratch implementation, and conceptually CPython's real implementation) resolves this by
   letting a bucket hold more than one entry — chaining, in the from-scratch version below — and
   checking `==` against every entry in that bucket to find the exact match.
4. **This is the crux of the O(1) *average* claim**: step 3's cost is proportional to the number of
   keys sharing that one bucket, not to `n`, the total number of keys in the whole structure. If
   the hash function spreads keys roughly evenly across `m` buckets, and `m` is kept proportional
   to `n` (which is exactly what dict/set resizing maintains — CPython keeps the table at most
   ~2/3 full), then the *expected* number of keys per bucket is a small constant
   (`n / m ≈` a small constant, independent of `n`'s absolute size) — so steps 1–3 combined cost
   O(1) on average, regardless of how large the structure grows.
5. **This derivation has an explicit assumption baked in**: it requires the hash function to
   distribute keys roughly uniformly across buckets. If it doesn't (a poorly written `__hash__`,
   or an adversarially chosen set of keys), many keys collide into the same bucket, step 3's
   per-lookup cost grows with the number of colliding keys, and the *worst case* for any hash
   table — including Python's real `dict`/`set` — is O(n): every key in one bucket, checked one
   at a time, exactly like a list. Demonstrated for real below.

## Algorithm

Building a chaining hash table (what `dict`/`set` are conceptually doing underneath):

1. Allocate an array of `m` empty buckets (each bucket a small list).
2. **Insert(key, value):** compute `index = hash(key) % m`. Scan `buckets[index]` for an existing
   entry with this `key` (equality check); if found, overwrite its value. If not found, append
   `(key, value)` to `buckets[index]`.
3. **Lookup(key):** compute `index = hash(key) % m`. Scan `buckets[index]` for an entry with this
   `key`; return its value if found, else raise a not-found error.
4. **(Not implemented below, but part of the real picture):** resize `m` upward and rehash every
   existing entry once the table gets too full — this is what keeps step 4 of the Mathematical
   foundation's derivation (`n / m ≈ constant`) true as the structure grows, the same amortization
   argument as list append.

## From-scratch implementation

A separate-chaining hash table, built in plain Python, verified for correctness against a real
`dict` on 2,000 keys, then stress-tested for its bucket-fill distribution. See
[`data_structures.ipynb`](data_structures.ipynb), section "6. From-scratch: what `dict`/`set` do
underneath."

Real executed output:

```
size: 2000  matches dict on every key: True
missing key raises KeyError: KeyError('does_not_exist')
buckets used: 881/1024, max chain length: 8, avg (non-empty): 2.27
```

Every one of 2,000 keys round-trips correctly through the hand-rolled table, matching a real
`dict` value-for-value, and a missing key raises `KeyError` exactly as `dict.get`'s stricter
cousin (`__getitem__`/`.get` with no default) would. The bucket-fill numbers directly confirm the
Mathematical foundation's claim: with 2,000 keys spread across 1,024 buckets, the average
non-empty bucket holds ~2.27 entries — a small constant, not something that grows linearly with
the 2,000 total keys.

## Practical implementation

The full practical notebook — [`data_structures.ipynb`](data_structures.ipynb) — covers, with real
executed examples: `list` creation, indexing/slicing, mutation (`append`, `insert`, `extend`,
`remove`, `pop`, `del`), sorting (`.sort()` in place vs. `sorted()` returning a new list, custom
`key=`), and other list methods; `tuple` creation, unpacking (including starred unpacking),
hashability (tuples as dict keys), and `namedtuple`; `set` creation, mutation, and the full set-
algebra operator set (union, intersection, difference, symmetric difference, subset/superset);
`dict` creation, access (`[]` vs. `.get()` with defaults), deletion, iteration (keys/values/items),
merging with `|`, `setdefault`, and `collections.defaultdict`/`Counter`. This maps directly back
to the Mathematical foundation: `list` operations above are exactly the O(n)/O(1) operations
derived there, and `set`/`dict` are CPython's real, production-grade version of the from-scratch
hash table above (open addressing internally rather than chaining, but the same hash → bucket →
collision-resolution shape).

## Experiment

**Hypothesis (stated before running):** on the same 20,000 keys, the from-scratch hash table's
lookup (hash + short chain scan) will be roughly constant-time regardless of dataset size, while a
Python list's linear search (`in`) will scale with the list's length — so the hash table should
win by orders of magnitude at 20,000 elements, with the worst case (last element, forcing the list
scan to walk the entire list) chosen deliberately to make the comparison fair-to-unfavorable for
the hash table.

**Setup:** build a 20,000-element list of unique strings and load the identical strings into the
from-scratch hash table; look up the *last* element (worst case for list linear search) 200 times
via `timeit`, once through `in data_list` and once through `in our_ht` (`__contains__`).

**Actual result (real executed output):**

```
list linear search:    0.019503s for 200 lookups
our hash table lookup:  0.000089s for 200 lookups
our hash table is ~219x faster than list linear search on 20000 items
```

**Interpretation:** confirms the O(n) vs. O(1)-average prediction directly — roughly 220× faster
for a lookup pattern chosen to be the worst case for the list. This gap grows, not shrinks, as the
dataset grows further, because the list's cost is proportional to `n` while the hash table's stays
roughly flat (bucket count is grown to keep the average chain length small, per the Mathematical
foundation's derivation).

**Limitations:** this measures one Python process, one machine, one data shape (short unique
strings) — absolute numbers will vary by hardware and Python version, but the *qualitative*
O(n)-vs-O(1) gap and its direction are not implementation artifacts; they follow directly from the
algorithmic difference. The built-in `dict`/`set` are also meaningfully faster in absolute terms
than this from-scratch table (implemented in C, with open addressing rather than Python-level
chaining) — the point of the from-scratch version is the mechanism, not beating CPython's own
implementation.

## Failure modes

- **Mutable objects as dict/set keys — loud failure.** A `list` cannot be a dict key or set member:
  `dict`/`set` require `__hash__`, and mutable built-in types deliberately don't define one (their
  value, and therefore what they'd hash to, can change after insertion, which would silently break
  the bucket invariant). Real executed proof: `{[1, 2]: "coordinates"}` raises
  `TypeError: unhashable type: 'list'` — immediately, not silently. The fix is a `tuple` of the
  same values, which is immutable and hashable: `{(1, 2): "coordinates"}` works.
- **Mutable objects as dict/set keys — silent failure, worse than the crash above.** A custom class
  that defines `__hash__` based on a field that can later change creates a key whose hash bucket
  becomes wrong the moment that field is mutated — the object is still in the dict, just
  unreachable from its own current value. Real executed proof: a `BadKey` object with
  `__hash__`/`__eq__` based on `self.value` is inserted, found successfully
  (`d.get(k) == "found me"`), then `k.value` is mutated *after* insertion; `d.get(k)` afterward
  returns `None` even though `len(d) == 1` — the entry is still there, silently unfindable. This
  is strictly worse than the `TypeError` above because nothing crashes; the bug just produces wrong
  answers.
- **Hash collisions degrading to O(n) worst case.** The Mathematical foundation's O(1) derivation
  assumes roughly uniform bucket distribution. Reproduced for real: a `DegenerateHashTable` whose
  `_index` only ever uses 4 buckets (regardless of table size) collapses every one of 8,000 keys
  into 4 long chains (`max chain length 2067`, vs. `10` for the well-spread version on the same
  data), and lookup measured **132× slower** on the same operation
  (`lookup time — degenerate hashing: 0.008126s, well-spread hashing: 0.000061s`). Real `dict`/`set`
  are not immune to this in principle — an adversarial or poorly distributed set of keys can
  degrade any hash table — CPython mitigates it with a strong general-purpose hash and (for
  strings) randomized hash seeding per process specifically to make deliberate collision attacks
  impractical, but the underlying worst case is real.

## Real-world usage

- **Deduplication and membership testing at scale**: `if x in seen: continue` — using a `set` for
  `seen` instead of a `list` is one of the single most common Python performance fixes, and the
  Experiment above is exactly why.
- **Caching / memoization**: function-result caches (`functools.lru_cache`, or a hand-rolled
  `dict`) are hash tables keyed on function arguments — the whole point is O(1) "have I computed
  this before" lookup instead of scanning a list of past calls.
- **Database indexes**: a hash index (as opposed to a B-tree index, covered later in
  `03-data-analysis/07-sqlite`) is this exact mechanism at the storage-engine level — trading
  index-maintenance cost for O(1)-ish point lookups instead of a full table scan.
- **Word/token counting, groupby-style aggregation**: `collections.Counter` and
  `collections.defaultdict(list)` are both dicts used specifically because "give me the bucket for
  this key" is the operation being repeated, over and over, across a large dataset.
- **Hash-seed randomization as a security measure**: CPython randomizes string hashing per process
  by default specifically because the collision-degradation failure mode above is exploitable — an
  attacker who can predict hash values and control input keys (e.g. HTTP form field names) could
  otherwise force a server's dict-based data structures into their O(n) worst case as a denial-of-
  service attack.

## Mental model

**A `list` trades zero setup for a linear search; a `set`/`dict` trades a small, roughly-constant
amount of setup (computing a bucket from the key's own content) for a lookup that jumps straight
to one small bucket instead of scanning everything — and that trade only holds as long as the
hash function keeps buckets small, which is why a bad hash function (or a mutated key) undoes the
whole benefit.** Reach for `list` when order and position matter and lookups are rare or by index;
reach for `set`/`dict` the moment "is this here" or "look this up by key" happens more than a
handful of times against a collection that might grow.

## Questions to think about

1. Why is list `insert(0, x)` O(n) but `append(x)` O(1) amortized, when both are "add one
   element to a list"? What's the structural difference between the two positions?
2. The from-scratch hash table's `get` scans a bucket linearly once it's found the right one. Why
   doesn't that make the whole structure O(n) — walk through the distinction between "scanning one
   bucket" and "scanning the whole table," using the measured `max chain length: 8` /
   `avg (non-empty): 2.27` numbers above.
3. A colleague argues "hash tables are always faster than lists, so just use dicts/sets
   everywhere." Construct a case where a `list` is the right choice specifically *because* order
   or position matters, not despite the O(n) membership-test cost.
4. The `BadKey` demo shows a dict silently losing the ability to find an entry after its key is
   mutated, without raising any error. Why is this failure mode more dangerous in a real system
   than the `TypeError` from using an unhashable list — what kind of bug does it produce instead of
   a crash, and how would you notice it in practice?
5. `DegenerateHashTable` forces 8,000 keys into 4 buckets, and lookup was measured ~132x slower
   than the well-spread version, but still nowhere near as slow as a full list scan across the
   whole dataset would be if it had to check every one of 8,000 elements one-by-one for a
   non-existent key. Reconcile this: is a severely collision-degraded hash table meaningfully
   different from a list at that point, or effectively the same thing?
