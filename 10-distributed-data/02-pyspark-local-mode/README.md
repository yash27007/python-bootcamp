# 02 – PySpark Local Mode

Detailed notes (the DataFrame API, lazy execution — transformations vs. actions, reading
`.explain()` plans, Conceptual-foundation substitution documented inline): [notes.md](notes.md)

Real PySpark jobs run in local mode (`local[*]`) on a locally-generated 800,000-row synthetic
dataset — `.read`, `.filter`, `.groupBy().agg()`, a `.join()` (both auto-broadcast and a
forced-shuffle variant), `.explain()` plans, a timed comparison against
`01-why-distributed-processing`'s from-scratch `multiprocessing` groupBy/join, and real
failure-mode demonstrations (collect-too-much, skewed keys, too many small partitions), all
actually executed with real pasted output:
[02-pyspark-local-mode.ipynb](02-pyspark-local-mode.ipynb)

## What you'll learn

Why `01-why-distributed-processing`'s hand-rolled `multiprocessing.Pool` approach doesn't scale as
an *engineering practice*, even though it scales reasonably well as raw compute — and how PySpark
automates the exact four mechanisms that topic built by hand (partitioning, map/filter/aggregate,
shuffle, fault tolerance) behind a declarative DataFrame API. Spark's lazy execution model
(transformations build a plan; only actions run it) and how to read `.explain()` output to see
whether an operation shuffles (`Exchange`) or not (`BroadcastExchange`, or neither).

| Topic | Status |
|-------|--------|
| Problem: hand-rolled distributed mechanisms don't scale as practice, only as compute | ✅ Complete |
| Why `multiprocessing.Pool` fails: no optimizer, no groupBy/join primitive, no reusable plan | ✅ Complete |
| DataFrame API, lazy execution (transformations vs. actions, execution plans) | ✅ Complete |
| Real `SparkSession` in `local[*]` mode, confirmed running (Spark 4.2.0, 16 cores) | ✅ Complete |
| Real `.read`/`.filter`/`.groupBy().agg()` on 800K locally-generated rows | ✅ Complete |
| Real `.join()`: auto-broadcast (`.explain()` shows `BroadcastHashJoin`) and forced shuffle (`.explain()` shows `Exchange`/`SortMergeJoin`) | ✅ Complete |
| Real timed comparison: PySpark vs. Task 1's from-scratch `multiprocessing` groupBy/join | ✅ Complete |
| Real failure-mode demos: collect-too-much estimate, skewed key, 1 vs. 16 vs. 400 partitions | ✅ Complete |

## Why it matters

Distributed engines look like magic if you haven't first hand-built what they automate — Task 1
did that. This topic closes the loop: it shows the *same* mechanisms (partitioning,
map/filter/aggregate, shuffle, fault tolerance via lineage) running behind a real engine's
declarative API, inspectable via `.explain()`, and — measured directly, not assumed — genuinely
faster than the hand-rolled version even at a scale small enough to fit trivially on one machine.
PySpark local mode is also how real Spark pipelines are developed and tested before deployment:
the exact API surface used here (`.read`, `.filter`, `.groupBy().agg()`, `.join()`) is unchanged
on a production cluster with hundreds of executors and terabytes of data.

## Prerequisites

- `01-why-distributed-processing` — this topic cites its from-scratch `multiprocessing` groupBy
  and broadcast-join work directly, and assumes its vocabulary (partitioning, shuffle, map-side
  combine, lineage) without re-deriving it.
- Java (confirmed present: `openjdk 21.0.11`) — required by PySpark's JVM-based execution engine.
  `uv add pyspark` installs the Python package; no separate Spark cluster or download needed for
  local mode.

## What you'll build

- A real `SparkSession` started in `local[*]` mode (Spark 4.2.0, all 16 local cores), confirmed
  running end to end, `SparkSession.stop()`ed cleanly at the end.
- 800,000 synthetic `transactions` rows and 100,000 synthetic `customers` rows, generated locally
  with fixed seeds, written to and read back from real Parquet files.
- Real `.filter`, `.groupBy().agg()`, and `.join()` jobs against that data, each with its
  `.explain()` physical plan captured — including forcing a genuine shuffle join
  (`spark.sql.autoBroadcastJoinThreshold = -1`) to see an `Exchange`/`SortMergeJoin` plan alongside
  the auto-chosen `BroadcastHashJoin`.
- A real, timed head-to-head: PySpark's `groupBy`/`join` vs. a from-scratch
  `multiprocessing.Pool` map-side-combine groupBy and broadcast hash join on the same 800K-row
  data — PySpark won both (2.60x on groupBy, 3.23x on join), refuting the topic's own
  stated hypothesis, with the refutation explained in `notes.md`'s Experiment section.
- Three real failure-mode demonstrations: an unexecuted-but-estimated `.collect()` footprint
  (~38 MB for the full join), a measured 30%-of-rows-in-one-key skew, and a measured 1-vs-16-vs-400
  partition timing comparison (400 partitions ran 4.3x slower than 1, for the same trivial job).

## Where it appears in real systems

Any team building a Spark pipeline develops and tests it in `local[*]` mode first, exactly as
done here, before running unmodified against a real cluster (YARN, Kubernetes, Databricks, AWS
EMR). The broadcast-vs-shuffle join decision measured in this topic is one of the most common
real-world Spark performance levers (joining a large fact table against a small dimension table);
skewed-key mitigation and partition-count tuning are standard, recurring Spark operations work.

## What's next

`03-streaming-fundamentals` — Tasks 1–2 processed a fixed, already-collected batch dataset. This
next topic asks what changes when data arrives continuously instead: producer/consumer
decoupling via a queue, backpressure, partitioning a stream while preserving per-key ordering, and
offsets — built from scratch in the standard library, since no message broker is assumed available
in this environment, ahead of `04-kafka`'s practical treatment.
