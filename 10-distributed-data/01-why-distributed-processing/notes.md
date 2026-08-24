# Why Distributed Processing

> **Note on template substitution:** this topic is a distributed-systems topic, not a numerical-
> modeling one — there is no closed-form loss function or derivative to derive. Per
> `AGENTS.md`'s content standards, the
> "Mathematical foundation" section below is replaced with a **Conceptual foundation** section
> that plays the same structural role: it precisely defines the mechanisms (partitioning, the
> map/filter/aggregate primitives, shuffle, fault tolerance) that the Algorithm and
> From-scratch sections then implement. This substitution is documented here inline, as required.

## Problem

Every topic so far in this curriculum (`01`–`09`) has assumed the dataset and the computation
both fit comfortably in one machine's RAM and finish in a reasonable time on one machine's CPU/GPU.
That assumption breaks in two independent ways as data grows:

1. **Memory**: a dataset that doesn't fit in RAM can't be loaded with `pandas.read_csv()` at all —
   the process gets OOM-killed or swaps and grinds to a halt.
2. **Compute time**: even a dataset that *does* fit in RAM can take hours to process with a
   single CPU core if the per-row computation is expensive and there are enough rows.

Real-world scale hits both. A single day of clickstream logs, sensor telemetry, or transaction
records for a mid-sized company routinely runs into hundreds of millions to billions of rows —
far past what one machine's memory or a single core's throughput can handle. The question this
topic answers: what has to change, structurally, to process data too large for one machine?

## Intuition

Imagine sorting a deck of 10,000 playing cards by hand. Alone, it takes a long time. Hand out
100 cards to each of 100 friends, have each friend sort their own small pile (this part is
trivial to parallelize — no friend needs to talk to any other friend to sort their own pile), then
merge the 100 sorted piles back together. Two things happened:

- **Splitting the deck** into 100 piles is *partitioning*.
- **Each friend sorting independently** is an *embarrassingly parallel* operation — no
  coordination needed between workers.

Now imagine instead you ask everyone to hand you all the cards of a particular suit — say, every
spade in the room. Each friend has to look through their own pile, pull out their spades, and
physically hand them across the room to a "spade collector." That handoff — data moving between
workers because of *what it is* (its key), not because of the order it started in — is a
*shuffle*, and it is why this second task is fundamentally more expensive than the first: it
requires communication between workers, not just independent local work.

## Why simpler approaches fail

**"Just buy a bigger machine" (vertical scaling).** This works, for a while. But it hits real
limits:

- **Cost is non-linear.** A machine with 2x the RAM or cores of a commodity machine does not cost
  2x as much past a certain point — high-memory, high-core-count servers carry a steep price
  premium, and the biggest single machines available (even from a major cloud provider) cap out
  in the terabytes of RAM, not petabytes.
- **There is a physical ceiling.** No single machine, at any price, holds a petabyte of RAM or
  processes data at unbounded throughput. Real datasets at real companies exceed what any single
  machine — however expensive — can hold or process in reasonable time.
- **A single machine is a single point of failure.** If it crashes mid-job, the entire job is
  lost and must restart from the beginning.

**Hand-rolled `multiprocessing.Pool`.** This is the natural next step: instead of buying a bigger
machine, use all the cores the current machine already has. It genuinely helps — the measurement
below shows a real, substantial speedup for CPU-bound work. But it does not solve the underlying
problem this topic is about:

- It still holds **all data in one machine's RAM**. If the dataset doesn't fit, `multiprocessing`
  doesn't change that — there is no partition-to-disk, partition-to-another-machine mechanism.
- It has **no built-in fault tolerance**. If one worker process crashes, `Pool.map` raises and the
  whole batch is lost — there's no lineage graph, no automatic recomputation of just the failed
  chunk (Task 1's fault-tolerance demo below builds that mechanism by hand to show what a real
  engine automates).
- It has **no shuffle mechanism**. `Pool.map` can *only* express "apply this function to every
  item independently" — it has no primitive for redistributing data across workers by key for a
  `groupBy`/`join`. The shuffle demo below is deliberately implemented as plain Python, not
  `multiprocessing`, because `multiprocessing.Pool` genuinely cannot express it.

This is exactly the gap distributed engines like PySpark (`02-pyspark-local-mode`) fill: they
add partitioning across machines, an automatic shuffle mechanism, and fault tolerance via lineage
— none of which `multiprocessing.Pool` provides, however many cores it uses.

## Conceptual foundation (substituting for Mathematical foundation)

Four mechanisms, precisely defined, that every distributed data-processing engine (Spark, Flink,
Dask, MapReduce, ...) is built from:

**1. Partitioning.** The dataset is split into $P$ disjoint chunks (partitions),
$D = D_1 \cup D_2 \cup \dots \cup D_P$, each small enough to fit in one worker's memory and be
processed independently. How rows are assigned to partitions matters — by arrival order, by a
hash of some key, by a range of a sort key — and that choice determines how expensive later
operations (especially shuffles) will be.

**2. Map / filter / aggregate as embarrassingly-parallel primitives.**
- **Map**: apply a function $f$ to every row independently, $f(x_i)$ for each $x_i \in D$. No row
  needs any other row's value, so this can be computed on all $P$ partitions fully in parallel,
  with zero communication between workers until the results are collected.
- **Filter**: keep rows where a predicate holds, $\{x_i \in D : p(x_i)\}$. Same property — each
  row's fate depends only on itself.
- **Aggregate** (e.g. `sum`, `count`, `max`) *without* a `groupBy` key: each partition computes a
  local partial aggregate, and the partials are combined with the same associative operator
  (e.g. add up 8 partial sums into 1 final sum) — still requires no shuffle, because there's only
  one output group.

These three are why "distributed processing" doesn't automatically mean "hard to implement" — a
huge fraction of real data pipelines are pure map/filter/aggregate chains, and those parallelize
almost for free.

**3. Shuffle — why it's expensive.** A `groupBy(key)` or `join(other, on=key)` needs every row
that shares a key to end up on the *same* worker so that worker can compute the group's result.
But rows with the same key are not, in general, already on the same partition (partitioning was
by arrival order or a different key). So the engine must:
1. Look at every row on every partition and determine which partition *should* own it (based on
   its key, e.g. `hash(key) % num_partitions`).
2. Physically move rows that aren't already on their owning partition — across process boundaries
   at minimum, across the network in a real cluster.
3. Do this for potentially every row in the dataset, all at once.

That is an **all-to-all data movement** step, bounded only by total data volume and network/disk
bandwidth — categorically more expensive than a map, which touches each row exactly once, locally,
with no data movement at all. The measured demo below shows this is not a hand-wave: moving raw
rows to their owning partition ("naive shuffle") moves ~300x more data than **pre-aggregating
locally first and only shipping the small partial aggregates** (a "map-side combine") — the
optimization every real groupBy engine performs.

**4. Fault tolerance: lineage vs. replication.** With hundreds of machines running for hours, a
worker failure during the job is not a rare edge case — it is expected. Two strategies handle it:
- **Replication**: keep $k$ copies of each partition's data or result on different machines. If
  one is lost, read from another copy. Costs $k\times$ the storage/memory.
- **Lineage (recomputation)**: instead of storing redundant copies, record the *recipe* that
  produced each partition — which source data plus which deterministic transform — and, if a
  partition's result is lost, recompute *only that partition* from its recipe. Costs extra compute
  time on failure, but no steady-state storage overhead. This is the model Spark's RDDs use.

## Algorithm

For the from-scratch comparison below, the "algorithm" for each approach is:

**Single-threaded loop:**
```
for each row in data:
    result.append(f(row))
```

**`multiprocessing.Pool`:**
```
split data into chunks
spawn N worker processes
each worker: for each row in its chunk: result_chunk.append(f(row))
pickle each result_chunk back to the parent process
concatenate result_chunks in original order
```

**Naive shuffle vs. map-side combine (for the shuffle-cost demo):**
```
naive:      for each row in each partition: if row's key doesn't belong to this partition, move it
combine:    for each partition: locally aggregate rows by key -> (key, partial_sum, partial_count)
            then: for each local key not owned by this partition, move only the small partial record
```

**Fault tolerance by lineage (for the recomputation demo):**
```
lineage[partition_id] = (transform_function, source_partition_data)
on failure of partition_id:
    result[partition_id] = lineage[partition_id].transform_function(lineage[partition_id].source_partition_data)
```

## From-scratch implementation

All code and real, actually-executed output lives in
[`01-why-distributed-processing.ipynb`](01-why-distributed-processing.ipynb). Environment: 16
CPU cores available (`multiprocessing.cpu_count() == 16`), 300,000 synthetic rows generated
locally with `random.seed(42)` (no download).

**The toy computation** (`cpu_bound_transform`): a per-row transform with no shared state — a
stand-in for an expensive, independent feature computation applied to every row — summing
`sin(x*i)**2 + log(x+i)` for `i` in `1..149`.

**Real measured timings (single-threaded vs. `multiprocessing.Pool` at varying worker counts):**

| Approach                       | Time (s) | Speedup vs. single-threaded |
|---------------------------------|---------:|-----------------------------:|
| Single-threaded loop            |    4.331 |                        1.00x |
| `multiprocessing.Pool(1)`       |    4.181 |                        1.04x |
| `multiprocessing.Pool(2)`       |    1.893 |                        2.29x |
| `multiprocessing.Pool(4)`       |    0.963 |                        4.50x |
| `multiprocessing.Pool(8)`       |    0.626 |                        6.92x |
| `multiprocessing.Pool(16)`      |    0.495 |                        8.74x |

(Checksum of the 300,000 results matched exactly across every configuration — `assert
abs(sum(result_mp) - sum(result_single)) < 1e-6` passed for all five pool sizes, confirming every
approach computed the same answer.)

**Overhead demo — cheap per-row work (a stand-in for "multiprocessing helping I/O-bound or
trivial-per-row work"):** same 300,000 rows, `trivial_transform(x) = x + 1.0` instead of the
CPU-heavy transform:

| Approach                              | Time (s) |
|-----------------------------------------|---------:|
| Single-threaded loop (trivial op)       |   0.0092 |
| `multiprocessing.Pool(8)` (trivial op)  |   0.0561 |

`multiprocessing.Pool(8)` was **6.09x slower** than the plain loop here — the fixed cost of
spawning 8 processes and pickling 300,000 floats to and from them dwarfs the ~9ms of actual work.

**Shuffle-cost demo — naive shuffle vs. map-side combine**, 300,000 `(key, value)` rows, 100
distinct groupBy keys, arriving distributed across 8 partitions by row index (not by key):

| Strategy            | Data moved                                    |
|----------------------|------------------------------------------------|
| Naive shuffle        | 262,607 rows moved = 4.20 MB                    |
| Map-side combine     | 700 partial records moved = 0.0140 MB           |

Pre-aggregating locally before shuffling moved **300x less data** than shuffling raw rows.

**Fault-tolerance-by-lineage demo:** 4 partitions of 75,000 rows each, each with a recorded
lineage `(transform_function, source_data)`. Partition 2's *result* was deliberately deleted to
simulate a worker crash; it was recovered by recomputing **only that partition** from its lineage
in 0.944s (not the other three, and not the whole job) — the recomputed result verified identical
to a fresh direct computation.

## Practical implementation

`multiprocessing.Pool` above **is** the practical, standard-library implementation of "use all of
one machine's cores" — there's no separate from-scratch/practical split for it the way earlier
topics split a NumPy derivation from a PyTorch call, because `Pool` already *is* the practical
tool at this scale. The practical tool for the next scale up — beyond one machine's RAM, with
automatic shuffle and lineage-based fault tolerance — is PySpark, covered starting in
`02-pyspark-local-mode`, which is not yet installed at this point in the curriculum
(`uv add pyspark` happens in that topic). This topic's shuffle-cost and fault-tolerance-by-lineage
demos are deliberately hand-rolled here so that when PySpark's `.explain()` shows an `Exchange`
(shuffle) step, or a Spark UI shows a recomputed stage after a worker loss, both map back to a
mechanism already built and measured by hand.

## Experiment

**Hypothesis (stated before measuring, see the notebook's first cells):**
1. `multiprocessing.Pool` beats the single-threaded loop for the CPU-bound transform, with
   speedup growing with worker count but flattening well below a full 16x on 16 cores (process
   overhead, plus this machine's real characteristics).
2. `multiprocessing.Pool` is *slower* than the single-threaded loop for cheap per-row work —
   overhead dominates.
3. Neither single-threaded nor `multiprocessing` solves the shuffle problem — that needs a
   distributed engine.

**Actual result:**
1. Confirmed. Speedup climbed from 1.04x (1 process — pure overhead, no parallelism gained) to
   8.74x at 16 processes — a real, useful speedup, but well short of a theoretical 16x, consistent
   with process-spawn and IPC (pickling data across process boundaries) overhead. The scaling was
   also sublinear even between smaller pool sizes (2 procs → 2.29x, not 2.00x exactly, but 4 procs
   → 4.50x rather than a clean 4x) — a mix of `chunksize` batching efficiency and pool-startup
   amortization, not pure linear scaling.
2. Confirmed. 6.09x *slower* with `Pool(8)` than the plain loop for the trivial per-row op.
3. Confirmed by construction: the shuffle-cost and fault-tolerance demos were implemented in plain
   Python because `multiprocessing.Pool`'s single primitive (`.map`) cannot express "move this
   subset of rows to a different worker based on a key" or "hold a lineage graph and recompute one
   lost partition."

**Interpretation:** the crossover point for "is multiprocessing worth it" is set by per-row work
size, not row count — 300,000 rows of cheap work lost to overhead; 300,000 rows of the heavier
transform won decisively. This is the practical rule of thumb: parallelize when the per-unit work
is large enough that (spawn + serialize + deserialize) overhead is a small fraction of total time.

**Limitations:** these are wall-clock timings on one specific machine (16 cores) under whatever
else was running at measurement time — the exact numbers are not universal constants, and
`multiprocessing.Pool`'s overhead profile varies by OS (fork vs. spawn) and Python version. The
qualitative conclusions (multiprocessing wins for heavy per-row work, loses for cheap per-row
work, and can't express shuffling) are the durable takeaway, not the exact multipliers.

## Failure modes

- **Shuffle blowing up memory/network.** A `groupBy`/`join` with a bad partition key (or a key
  with very high cardinality) forces moving a large fraction of the dataset across workers —
  exactly what the naive-shuffle demo measured (4.20 MB moved for a modest 300K-row toy dataset;
  at real scale this is terabytes crossing a real network, and it can exceed available memory or
  network bandwidth entirely, causing spills to disk or job failure).
- **Data skew.** If one key holds a disproportionate share of the rows (e.g. one customer ID with
  10x the transactions of any other), the partition that owns that key becomes a bottleneck — one
  worker doing far more work than the rest, while other workers sit idle waiting for it. The
  shuffle-cost demo above used uniformly distributed keys specifically to isolate shuffle-volume
  cost from skew; a skewed key distribution makes the imbalance an additional, separate problem.
- **`multiprocessing` not helping I/O-bound work.** The overhead demo above is the same failure
  mode by a different cause: work too small per unit relative to fixed overhead. The same effect
  shows up for I/O-bound work (e.g. a network call per row) — spawning processes doesn't remove
  time spent waiting on I/O, and the process-management overhead can still dominate if not
  batched properly; the standard-library fix for I/O-bound concurrency is threading or `asyncio`,
  not `multiprocessing`, precisely because `multiprocessing`'s overhead is aimed at getting around
  the GIL for CPU-bound work, a cost that isn't worth paying when the bottleneck is waiting, not
  computing.

## Real-world usage

- Any company processing clickstream, telemetry, log, or transaction data at a scale of hundreds
  of millions to billions of rows/day runs this exact map → shuffle → fault-tolerant-execution
  pipeline, usually via Spark, Flink, or a managed equivalent (e.g. AWS EMR, Databricks, Google
  Dataflow).
  - **Ad tech / recommendation**: joining a billion-row event log against a user feature table —
    a `join`, i.e. a shuffle by user ID — happens continuously.
  - **Fraud detection**: `groupBy(account_id)` aggregations over transaction streams at bank
    scale, where skew (a small number of very active accounts) is a constant operational concern.
  - **ETL pipelines**: nightly batch jobs that read raw logs, filter and map to a clean schema,
    and aggregate into reporting tables — pure map/filter/aggregate at petabyte scale.
- Even single-machine tools reach for `multiprocessing`-style parallelism constantly:
  `scikit-learn`'s `n_jobs=-1`, `pandas`/`polars` internal parallel operations, and data-loading
  pipelines (e.g. PyTorch `DataLoader(num_workers=N)`) all apply the exact map-primitive
  parallelism measured above, just packaged behind a higher-level API.

## Mental model

**A single machine gives you speed; distribution gives you scale and survival.**
`multiprocessing.Pool` makes one machine's cores work in parallel — genuinely faster for
CPU-bound, sufficiently-chunky work (measured: up to 8.74x here), genuinely worse for
too-cheap work (measured: 6.09x slower). But it is still one machine: one RAM ceiling, one
point of failure, and no primitive for moving data by key. A shuffle is expensive because it
is the one operation that *requires* workers to talk to each other instead of working alone —
minimize shuffles, or pre-aggregate before shuffling, whenever possible. And fault tolerance
is a choice between paying for extra copies up front (replication) or paying to recompute only
what's lost, when it's lost (lineage) — distributed engines default to lineage because storage
for every partition's history is far cheaper than storage for every partition's copy.

## Questions to think about

1. In the shuffle-cost demo, map-side combine moved 300x less data than the naive shuffle. Under
   what circumstances would that advantage shrink or disappear — i.e. when would pre-aggregating
   locally *not* help much? (Hint: think about what happens as the number of distinct keys
   approaches the number of rows.)
2. The overhead demo showed `multiprocessing.Pool(8)` losing to a single-threaded loop by 6.09x on
   cheap per-row work. If the per-row work were 10x more expensive than `trivial_transform` but
   still much cheaper than `cpu_bound_transform`, would you expect `Pool` to win, lose, or be close
   to break-even? What would you need to measure to find out, rather than guess?
3. The fault-tolerance-by-lineage demo recomputed exactly one lost partition, not the whole job.
   What has to be true about a transform function for recomputing from lineage to reproduce
   *exactly* the same result as the original run (consider: what if `cpu_bound_transform` had
   depended on a global mutable counter, or on wall-clock time)?
4. `multiprocessing.Pool` speedup climbed to 8.74x on 16 cores, not 16x. Name at least two
   distinct sources of that gap based on what the code actually does (think about what happens
   before any worker starts computing, and what happens after every worker finishes).
5. Data skew was explicitly excluded from the shuffle-cost demo (uniform keys were used on
   purpose). Sketch how you would modify that demo to actually measure the cost of skew — what
   would you change, and what would you expect the "naive shuffle" and "map-side combine" numbers
   to do differently under a skewed key distribution versus a uniform one?
