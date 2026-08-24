# PySpark Local Mode

> **Note on template substitution:** like `01-why-distributed-processing`, this topic is a
> tooling/systems topic, not a numerical-modeling one — there is no loss function or derivative to
> derive for "what is a DataFrame API." Per
> `AGENTS.md`'s content standards, the
> "Mathematical foundation" section below is replaced with a **Conceptual foundation** section
> covering Spark's DataFrame API and lazy execution model (transformations vs. actions, the
> execution plan) — this substitution is documented here inline, as required.

## Problem

`01-why-distributed-processing` built, by hand, every mechanism a distributed engine needs:
partitioning, the map/filter/aggregate primitives, a shuffle, and fault tolerance by lineage. That
topic's from-scratch tools were a plain Python loop and `multiprocessing.Pool` — and they worked,
measurably (up to 8.74x speedup on a CPU-bound transform). But hand-rolling all four mechanisms
correctly, every time, for every new pipeline, does not scale as an *engineering practice*, even
though `multiprocessing.Pool` scales reasonably well as *raw compute*:

- Every new `groupBy`-shaped job would need its own hand-written map-side-combine logic (Task 1's
  shuffle-cost demo was ~40 lines of custom partition/combine code for one specific operation).
- Every new job would need its own lineage-tracking and recovery logic (Task 1's fault-tolerance
  demo was hand-built for one specific 4-partition scenario) — nobody rewrites that correctly for
  every pipeline in a real codebase.
- None of it survives past one machine's RAM. `multiprocessing.Pool` cannot spill to disk, cannot
  span multiple machines, and has no query optimizer deciding *how* to run an operation.

The problem this topic answers: how do you get partitioning, shuffle, and fault tolerance
*automatically*, correctly, for arbitrary map/filter/groupBy/join pipelines, without hand-writing
the mechanism every time? Answer: a real distributed processing engine. PySpark, run in local mode
on this machine (`local[*]`), is that engine — with no cluster required to learn how it works.

## Intuition

Task 1's shuffle-cost demo hand-wrote: split data into partitions, decide each row's owning
partition by key, move only what's needed, pre-aggregate first. That is *exactly* what Spark's
`groupBy().agg()` does — the difference is Spark decides the partition count, the shuffle
strategy, and the pre-aggregation automatically, from a single declarative line of code, and
verifiably shows its decision via `.explain()`.

Task 1's fault-tolerance demo hand-wrote: record `(transform_function, source_data)` per
partition, recompute only the lost one on failure. Spark's RDD/DataFrame lineage graph (the
"Catalyst" query plan, visible in every `.explain()` below) *is* that same recipe, tracked
automatically for every operation in the pipeline, not just the one demo scenario built by hand.

The mental shift this topic makes: Task 1 asked "what does a distributed engine have to do?" and
built tiny versions of each piece. This topic asks "what does it look like to just use one?" —
and then measures whether that convenience costs anything at the modest scale this course can run
locally.

## Why simpler approaches fail

`01-why-distributed-processing`'s hand-rolled `multiprocessing.Pool` approach specifically breaks
down in ways this topic's practical work makes concrete:

- **No query optimizer.** Task 1's shuffle-cost demo required knowing, by hand, that pre-aggregating
  before shuffling saves ~300x data volume, and writing that optimization explicitly. Below,
  Spark's `.explain()` on the join shows it *automatically* chose a `BroadcastHashJoin` over a
  shuffle — the same class of optimization Task 1's map-side-combine was, chosen by a planner
  rather than a human, without being asked.
- **No abstraction over "how," only "what."** `multiprocessing.Pool.map()` only expresses
  "apply this function to every item." It cannot express "group by this key and sum this column" —
  Task 1's groupBy demo had to hand-write partition assignment, local aggregation, and combining.
  Spark's DataFrame API expresses `.groupBy("category").agg(F.sum("amount"))` directly; the engine
  decides how.
- **No persistent, reusable execution plan.** Every `multiprocessing.Pool` call in Task 1 was a
  one-off script. Spark's lazy execution model (below) builds an actual query plan object that can
  be inspected (`.explain()`), optimized before running, and reused — this is infrastructure Task
  1's approach has no equivalent of.
- **Still one machine's RAM, still no real fault tolerance for a crash mid-job.** Task 1 was
  explicit about this limitation; PySpark's local mode doesn't change that either (it's still one
  process here) — but the *same code* written against Spark's DataFrame API in this notebook would
  run unmodified on a real multi-machine cluster with genuinely more RAM and genuine
  worker-failure recovery. The `multiprocessing.Pool` code from Task 1 would not.

## Conceptual foundation (substituting for Mathematical foundation)

**The DataFrame API.** A Spark DataFrame is a table-like, distributed collection of rows with a
known schema (column names and types) — conceptually close to a `pandas.DataFrame`, but the
data is (potentially) split across partitions on (potentially) different machines, and operations
on it don't run immediately.

**Transformations vs. actions — Spark's lazy execution model.** Every DataFrame operation in
PySpark falls into one of two categories:

- **Transformations** (`.filter()`, `.groupBy()`, `.join()`, `.select()`, ...) build up a
  **logical plan** describing *what* should happen, but do not touch any data. Calling
  `transactions_df.filter(F.col("amount") > 400.0)` returns instantly (measured below: 0.03s) —
  not because the filter is fast, but because it hasn't run at all yet.
- **Actions** (`.count()`, `.collect()`, `.show()`, `.write()`, ...) trigger actual execution: Spark
  takes the accumulated logical plan, optimizes it (Catalyst — Spark's query optimizer), converts
  it into a physical plan of concrete operations across partitions, and runs it. This is why the
  `.count()` action after the same filter took 0.28s — that's when the filter genuinely ran, over
  all 800,000 rows.

This maps directly onto Task 1's vocabulary: a transformation like `.groupBy().agg()` is
"map/filter/aggregate plus shuffle," expressed declaratively; laziness means Spark can look at the
*entire* chain of transformations before running anything and optimize across all of them at once
(e.g. push a filter down before a join, or pick a broadcast join over a shuffle join) — something
Task 1's eager, step-by-step Python code has no way to do, because each step runs the instant it's
called.

**The execution plan, made visible with `.explain()`.** `.explain()` prints the plan Spark would
run (or has planned to run), without running it, in three or four layers from logical to physical.
The key thing to watch for, learned directly from Task 1's shuffle vocabulary: an **`Exchange`**
node in the plan means data moves between partitions — a shuffle. A plan with no `Exchange` (a
pure filter, or a broadcast join) has no shuffle; a plan with `Exchange hashpartitioning(key, N)`
does. The practical work below shows both: a `groupBy` (always shuffles, since grouping needs
same-key rows collocated) and a `join` that Spark chose to run *without* a shuffle by broadcasting
the small side (`BroadcastExchange` — a different, cheaper kind of data movement: one small table
copied everywhere, not the whole large table exchanged).

## Algorithm

Not a numerical algorithm — a description of what actually happens when a Spark job runs, in the
order used throughout this notebook:

```
1. spark.read.parquet(path)               -> registers a source, reads only file metadata (lazy)
2. df.filter(...) / .groupBy(...).agg(...) / .join(...)
                                           -> each builds up the logical plan (lazy, no data moved)
3. an action is called (.count(), .collect(), .show())
   -> Catalyst optimizes the accumulated logical plan
   -> the plan is converted to a physical plan (concrete operators: FileScan, HashAggregate,
      Exchange/shuffle, BroadcastHashJoin or SortMergeJoin, ...)
   -> the physical plan actually executes across `local[*]`'s partitions/cores
4. results return to the driver (small, aggregated results) or are written out (large results)
```

## From-scratch implementation

N/A for this topic, by design — `01-why-distributed-processing` already built the from-scratch
groupBy (map-side combine) and broadcast-join primitives this topic's engine automates. Rather
than reimplementing PySpark's internals (which would defeat the point — PySpark exists precisely
so nobody has to hand-write a query optimizer or a shuffle engine per pipeline), this topic cites
that from-scratch work directly and re-runs the *same two operations* — a `groupBy` and a broadcast
join, on the same 800,000-row scale — through both the from-scratch `multiprocessing` approach and
real PySpark, timed side by side in the Experiment section below. That comparison **is** this
topic's from-scratch bridge: it makes concrete exactly what PySpark's engine costs and buys,
relative to the hand-rolled version already built.

## Practical implementation

All code and real, actually-executed output lives in
[`02-pyspark-local-mode.ipynb`](02-pyspark-local-mode.ipynb). Environment: 16 CPU cores, Java
`openjdk 21.0.11` (confirmed present beforehand), PySpark `4.2.0` (`uv add pyspark`), a
`SparkSession` in `local[*]` mode (uses all 16 cores as local "executors"), started for real in
3.29s. Dataset: 800,000 synthetic `transactions` rows (`customer_id`, `amount`, `category`) and
100,000 synthetic `customers` rows (`customer_id`, `region`, `signup_year`), generated locally with
fixed seeds (`random.seed(42)` / `random.seed(7)`), written to and read back from real Parquet
files (7.86 MB and 0.69 MB on disk) — no download.

**`.read`:** `spark.read.parquet(path)` for both tables — the call itself took ~1.16s (reading
schema/footer metadata, still lazy w.r.t. row data), and the first `.count()` action (100,000 +
800,000 rows) actually read all the data in 1.11s.

**`.filter`:** `transactions_df.filter(F.col("amount") > 400.0)` — the filter call returned in
0.03s (lazy). `.explain()` on the unexecuted plan:
```
*(1) Filter (isnotnull(amount#5) AND (amount#5 > 400.0))
+- *(1) ColumnarToRow
   +- FileScan parquet [...] PushedFilters: [IsNotNull(amount), GreaterThan(amount,400.0)]
```
Note `PushedFilters` — Catalyst pushed the filter predicate down into the Parquet reader itself,
so unneeded rows are skipped during the scan rather than read then discarded. The `.count()`
action then actually ran the filter: **160,337 of 800,000 rows** matched, in 0.28s.

**`.groupBy().agg()`:** grouping 800,000 transactions by `category` (10 keys), computing
sum/count/avg of `amount`. `.explain()`:
```
AdaptiveSparkPlan isFinalPlan=false
+- Sort [total_amount#31 DESC NULLS LAST], true, 0
   +- Exchange rangepartitioning(total_amount#31 DESC NULLS LAST, 16), ENSURE_REQUIREMENTS, [...]
      +- HashAggregate(keys=[category#6], functions=[sum(amount#5), count(1), avg(amount#5)])
         +- Exchange hashpartitioning(category#6, 16), ENSURE_REQUIREMENTS, [...]
            +- HashAggregate(keys=[category#6], functions=[partial_sum(amount#5), partial_count(1), partial_avg(amount#5)])
               +- FileScan parquet [...]
```
Two `Exchange` nodes: the first `hashpartitioning(category, 16)` is the actual groupBy shuffle
(after a **partial** `HashAggregate` — Spark doing Task 1's map-side combine automatically, before
shuffling); the second `rangepartitioning` is a separate shuffle for the `.orderBy()`. The action
(`.show(10)`) ran in 0.52s and produced a real, near-uniform per-category breakdown (categories
range from `other` at $20.24M/80,382 txns down to `home` at $19.95M/79,671 txns — synthetic data
generated with uniform random category assignment, so near-equal totals are expected).

**`.join()` — the automatic optimization:** joining `transactions_df` (800K rows) with
`customers_df` (100K rows, 0.69 MB) on `customer_id`. `.explain()` showed **no shuffle `Exchange`**
at all — Catalyst detected `customers_df` is under the default 10 MB auto-broadcast threshold and
chose a `BroadcastHashJoin`, copying the small table to every partition instead of shuffling the
large one:
```
+- BroadcastHashJoin [customer_id#4L], [customer_id#0L], Inner, BuildRight, false, false
   :- Filter isnotnull(customer_id#4L) +- FileScan parquet [transactions...]
   +- BroadcastExchange HashedRelationBroadcastMode(...)
      +- Filter isnotnull(customer_id#0L) +- FileScan parquet [customers...]
```
This *is* the from-scratch broadcast-join optimization from Task 1's vocabulary, chosen
automatically. The join action (`.count()`) ran in 0.31s, correctly producing 800,000 rows (every
`customer_id` exists in `customers_df` by construction).

**Forcing a real shuffle join (to see the shuffle case explicitly):** setting
`spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)` disables auto-broadcast, forcing
Spark to shuffle *both* sides on `customer_id` and merge-sort join them:
```
+- SortMergeJoin [customer_id#4L], [customer_id#0L], Inner
   :- Sort [...] +- Exchange hashpartitioning(customer_id#4L, 16), [...] +- Filter [...] +- FileScan [transactions]
   +- Sort [...] +- Exchange hashpartitioning(customer_id#0L, 16), [...] +- Filter [...] +- FileScan [customers]
```
Now both `Exchange hashpartitioning` nodes are the real shuffle this topic's brief asked to see —
and the forced-shuffle join measurably cost more: **0.67s vs. 0.31s** for the auto-broadcast join,
confirming Catalyst's default choice was the right one for this data shape. The threshold was
restored to Spark's default (10 MB) immediately after this measurement.

## Experiment

**Hypothesis (stated before measuring, see the notebook):**
1. At this modest, single-machine-friendly scale (800K rows), a hand-rolled
   `multiprocessing.Pool` map-side-combine groupBy would be **at least competitive with, and
   likely faster than**, PySpark's `groupBy().agg()` — PySpark pays real fixed costs (Python↔JVM
   serialization via Py4J, Catalyst query planning) that shouldn't pay for themselves yet at this
   scale.
2. Same expectation for the join: a from-scratch broadcast hash join (dict lookup, parallelized
   over `multiprocessing.Pool`) should be competitive with or faster than Spark's `.join()`.
3. PySpark's real advantage isn't raw speed at this toy scale — it's that the exact same code
   would keep working unmodified if the data were 1000x larger, where the from-scratch version
   needs a rewrite (or doesn't fit in RAM at all).

**Actual result — hypotheses 1 and 2 were both refuted:**

| Operation                  | `multiprocessing` (8 workers) | PySpark `local[*]` | PySpark speedup |
|-----------------------------|------------------------------:|--------------------:|-----------------:|
| `groupBy(category).agg()`   |                        0.452s |               0.174s |           2.60x |
| `join(customers)`           |                        0.531s |               0.164s |           3.23x |

PySpark was faster on **both** operations at this scale, by a clear margin — not just
"competitive." Both engines produced identical results (checksums/row counts matched exactly).

**Interpretation:** the hypothesis underestimated two things. First, these timings were taken
*after* the `SparkSession` was already warm (JVM startup, 3.29s, was paid once at the top of the
notebook and amortized across every subsequent job — a realistic assumption for any long-running
pipeline, but worth stating explicitly since it's not "free"). Second, and more importantly,
Spark's `HashAggregate`/`BroadcastHashJoin` operators run as JIT-compiled, whole-stage-codegenerated
JVM bytecode operating on Spark's internal columnar row format — while the `multiprocessing`
version pays real, repeated costs Task 1 didn't have to isolate for a single-machine `.map()`: a
`Pool` is re-spawned per operation here (as a fair fresh-call comparison), and every row crosses a
Python-object dict-lookup/tuple-unpacking boundary, both slower per-row than Spark's optimized
internal execution once the JVM is already running. In short: for `groupBy`/`join`-shaped
work specifically, a mature, JIT-optimized declarative engine can beat a straightforward hand-rolled
parallel implementation even at a scale small enough to fit trivially on one machine — the
"hand-rolled is faster for `.map()`-shaped work at small scale" lesson from Task 1's CPU-bound
transform did **not** generalize to `groupBy`/`join`-shaped work, because those aren't a single
`.map()` call; they involve grouping/hashing/joining logic Spark has spent years optimizing at the
bytecode level, that the from-scratch version reimplements naively in pure Python.

**Limitations:** single-machine timings on one specific machine (16 cores, PySpark 4.2.0, one run
per configuration — not averaged over repeated trials, so some of these numbers carry ordinary
measurement noise, particularly the 400-partition overhead demo below). `Pool` was re-spawned per
timed call rather than reused across calls, which is a fair "cold `Pool`" comparison but not the
only valid one — a long-lived `Pool` reused across many jobs would close part of this gap. The
qualitative result (PySpark's optimized engine beats a naive from-scratch groupBy/join even at
modest scale) is the durable takeaway; the exact multipliers are specific to this run.

## Failure modes

- **`.collect()`ing too much to the driver.** `.collect()` pulls every row of a distributed
  DataFrame back into the driver process as plain Python objects — measured above: the full
  (unaggregated) `joined` DataFrame is 800,000 rows, an estimated ~38 MB of Python objects if
  `.collect()`ed directly (deliberately *not* run in the notebook — the point of this failure mode
  is that you often only discover the size is a problem when the driver OOMs). The correct pattern,
  used throughout this notebook: aggregate or filter *first* (`region_spend`, 5 rows), and only
  `.collect()`/`.show()` the small, already-reduced result. At real production scale (millions to
  billions of rows), collecting an unaggregated DataFrame is one of the most common ways to crash
  a Spark driver — it defeats the entire purpose of staying distributed.
- **Skewed joins/groupBys.** The main `groupBy(category)` demo used 10 near-uniform keys (each
  ~80,000 rows, ~0.1% relative spread). A deliberately skewed key column — one key holding 30% of
  all rows (240,000 of 800,000, measured above) vs. the next-largest key holding 154 rows — sends a
  wildly disproportionate share of shuffled data to whichever task owns that key, so that task runs
  far longer than the rest while other cores finish and sit idle. This is the exact same underlying
  problem Task 1's shuffle-cost demo deliberately excluded (it used uniform keys to isolate
  shuffle *volume* cost from *skew* cost) — real-world keys are rarely uniform (one very active
  customer, one popular product, a NULL/default bucket absorbing bad data), and skew is a leading
  cause of "one Spark job stage taking 100x longer than every other stage."
- **Too many small partitions.** `local[*]` doesn't hide the fundamental partition-count tradeoff:
  measured above, the identical 50,000-row aggregation took 0.164s at 1 partition, 0.249s at 16
  partitions, and 0.701s at 400 partitions — *slower* with more partitions, for this trivial job
  size, because each partition carries fixed per-task scheduling overhead that dwarfs the
  sub-millisecond of real work available per partition once there are hundreds of them for a small
  dataset. The lesson generalizes in both directions: too few partitions leaves cores idle (can't
  parallelize below the partition count); too many partitions pays scheduling overhead far in
  excess of any parallelism gained. Partition count should track data volume and core count
  together, not be set to an arbitrarily large or small fixed number.

## Real-world usage

- PySpark local mode is a genuinely common **development and testing** setup — teams write and
  unit-test Spark pipelines against `local[*]` on a laptop or CI runner before deploying the exact
  same code, unmodified, to a real YARN/Kubernetes/Databricks cluster with hundreds of executors
  and terabytes of data. Everything measured in this notebook — `.read`, `.filter`,
  `.groupBy().agg()`, `.join()`, `.explain()` — is the identical API surface used in production.
- The broadcast-vs-shuffle join decision measured above (`BroadcastHashJoin` auto-chosen, 0.31s,
  vs. a forced `SortMergeJoin`, 0.67s) is the single most common join-performance lever
  in real Spark tuning: joining a large fact table against a small dimension table (customers,
  products, a lookup table) should broadcast the small side — Spark does this automatically below
  a size threshold, but engineers routinely tune that threshold, or force a broadcast with a hint,
  when Catalog statistics are stale or the table's true size is known to be small.
- Data-skew mitigation (salting hot keys, isolating a known-skewed key into a separate code path,
  adaptive query execution splitting large partitions) is one of the most common real-world Spark
  performance-tuning tasks — directly downstream of the skew failure mode measured above.

## Mental model

**PySpark automates exactly what Task 1 built by hand — and does it faster, not just more
conveniently, once the engine is warm.** Lazy execution means nothing runs until an action is
called, which lets Catalyst see the *whole* pipeline and choose the cheapest physical plan (a
broadcast instead of a shuffle, a pushed-down filter) — a decision Task 1's eager, step-by-step
Python code has no way to make, because each line runs the instant it's called with no view of
what comes next. `.explain()` is how you check that decision instead of guessing: an `Exchange`
node is a shuffle, a `BroadcastExchange` is the cheaper alternative, and a plan with neither means
no data moved between partitions at all. None of this makes the underlying mechanisms (shuffle
cost, skew, partition-count tradeoffs) go away — it automates handling them well by default and
makes them inspectable, which is a categorically different, more scalable engineering position
than re-deriving the same optimizations by hand in every new pipeline.

## Questions to think about

1. `.explain()` on the un-forced join showed a `BroadcastHashJoin` with no shuffle `Exchange`, but
   the groupBy always showed a shuffle `Exchange`. Why can a join sometimes avoid a shuffle
   entirely while a `groupBy` on a table with real duplicate keys fundamentally cannot? (Hint:
   think about what property `customers_df` had that made broadcasting possible.)
2. The experiment refuted the hypothesis that hand-rolled `multiprocessing` would be competitive
   with PySpark for `groupBy`/`join` work, even though Task 1 showed hand-rolled
   `multiprocessing` winning decisively for a `.map()`-shaped CPU-bound transform. What is
   structurally different between a pure `.map()` operation and a `groupBy`/`join` operation that
   would explain why a mature engine's advantage shows up in one case and not (as clearly) in the
   other?
3. The forced-shuffle join (0.67s) was slower than the auto-broadcast join (0.31s) here because
   `customers_df` was small. Sketch the scenario where forcing a shuffle join would instead be the
   *right* choice over broadcasting — what would have to be true about the size of both tables?
4. The 400-partition demo took 0.701s for 50,000 rows — more than 4x the 1-partition time. If the
   dataset were instead 500 million rows on a real multi-machine cluster, would you expect 400
   partitions to still be worse than 1? What does that tell you about how "the right partition
   count" should scale with data size, not stay fixed?
5. `spark.conf.set("spark.sql.autoBroadcastJoinThreshold", -1)` was used to force the shuffle join
   for this demo, then restored immediately after. In a real pipeline, what risk would leaving that
   setting permanently disabled introduce, given what you now know about broadcast joins being the
   faster choice for small-table joins?
