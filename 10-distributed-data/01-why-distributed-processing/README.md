# 01 – Why Distributed Processing

Detailed notes (partitioning, map/filter/aggregate as embarrassingly-parallel primitives, why
shuffle is expensive, lineage vs. replication for fault tolerance — Conceptual-foundation
substitution documented inline): [notes.md](notes.md)

Real, timed, locally-run 3-way comparison (single-threaded loop vs. `multiprocessing.Pool` vs. a
forward reference to PySpark), plus a shuffle-cost measurement and a fault-tolerance-by-lineage
demo, all actually executed with real pasted output:
[01-why-distributed-processing.ipynb](01-why-distributed-processing.ipynb)

## What you'll learn

Why every topic through `09-pytorch` implicitly assumed "the data and computation fit on one
machine" — and what specifically breaks that assumption at real scale. Why "buy a bigger machine"
(vertical scaling) and a hand-rolled `multiprocessing.Pool` both hit real limits (RAM ceiling, no
shuffle primitive, no fault tolerance), even though `multiprocessing.Pool` measurably helps for
CPU-bound work on one machine. The four mechanisms every distributed engine is built from:
partitioning, the map/filter/aggregate primitives, shuffle, and fault tolerance via lineage or
replication.

| Topic | Status |
|-------|--------|
| Problem: data/computation too large for one machine | ✅ Complete |
| Why vertical scaling and hand-rolled multiprocessing both fail | ✅ Complete |
| Partitioning, map/filter/aggregate, shuffle, fault tolerance (conceptual foundation) | ✅ Complete |
| Real timed comparison: single-threaded vs. `multiprocessing.Pool` (1–16 workers) | ✅ Complete |
| Real overhead demo: multiprocessing losing to cheap per-row work | ✅ Complete |
| Real shuffle-cost demo: naive shuffle vs. map-side combine | ✅ Complete |
| Real fault-tolerance-by-lineage demo: recompute one lost partition | ✅ Complete |

## Why it matters

Distributed engines like PySpark (next topic) look like magic if you haven't first felt, by hand,
what they automate: splitting data so work can run in parallel, the real cost of moving data
between workers for a `groupBy`/`join`, and what happens when a worker dies mid-job. This topic
builds and measures each of those by hand — in plain Python and the standard library only — so
that when PySpark's `.explain()` shows an `Exchange` step or a job survives a simulated worker
failure, both map back to a mechanism already understood and measured here.

## Prerequisites

- Comfort with Python's standard library (`multiprocessing`, `time`) — no new external
  dependencies for this topic.
- No prior distributed-systems background assumed; this topic builds the vocabulary from
  scratch.

## What you'll build

- A CPU-bound, embarrassingly-parallel toy computation applied to 300,000 locally-generated
  synthetic rows (fixed seed, no download).
- A real, timed comparison of that computation run single-threaded vs. across `multiprocessing.Pool`
  at 1/2/4/8/16 workers, with measured speedups up to 8.74x — plus a deliberate second measurement
  showing `multiprocessing.Pool` losing to a plain loop by 6.09x when the per-row work is too cheap
  to justify process overhead.
- A real shuffle-cost demo: naive row-by-row shuffle vs. map-side combine on the same 300,000
  keyed rows, showing a measured 300x reduction in data moved by pre-aggregating before shuffling.
- A real fault-tolerance-by-lineage demo: 4 partitions with recorded lineage, a simulated worker
  crash (one partition's result deleted), and recovery by recomputing only that partition.

## Where it appears in real systems

Any pipeline processing clickstream, telemetry, transaction, or log data at hundreds of millions
to billions of rows/day (ad tech, fraud detection, ETL) runs this exact map → shuffle →
fault-tolerant-execution pattern, typically via Spark or Flink. Even single-machine tools
(`scikit-learn`'s `n_jobs=-1`, `pandas`/`polars` internals, PyTorch `DataLoader(num_workers=N)`)
apply the same map-primitive parallelism measured here, packaged behind a higher-level API.

## What's next

`02-pyspark-local-mode` — installs PySpark (`uv add pyspark`) and runs the same kind of
map/filter/aggregate/shuffle operations from this topic through a real distributed engine in
local mode, comparing its `groupBy`/`join` timing directly against this topic's from-scratch
`multiprocessing` version.
