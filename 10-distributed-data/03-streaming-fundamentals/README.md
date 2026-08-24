# 03 – Streaming Fundamentals

Detailed notes (producer/consumer decoupling, backpressure, stream partitioning, consumer groups,
offsets — Conceptual-foundation substitution documented inline): [notes.md](notes.md)

Real, actually-run standard-library Python demos — a bounded `queue.Queue` showing measured
producer blocking under backpressure, a toy partitioned log with independent consumer offsets
(including a real hash-skew observation), and a genuine consumer crash-and-resume-from-offset
scenario alongside its failure-mode counterpart (no external offset persistence) — all with real
pasted output:
[03-streaming-fundamentals.ipynb](03-streaming-fundamentals.ipynb)

## What you'll learn

Why `01`/`02`'s batch assumption — a fixed, already-collected dataset — breaks for data that
arrives continuously and indefinitely, and why "just batch-process every N minutes" doesn't
actually solve the underlying problem (it adds latency and still doesn't handle a producer faster
than the consumer). The four mechanisms a streaming system needs: a bounded buffer for
backpressure, partitioning a stream by key for parallelism while preserving per-key order,
consumer groups for independent parallel readers of the same log, and durable offsets for
crash recovery.

| Topic | Status |
|-------|--------|
| Problem: continuous, indefinite arrival vs. `01`/`02`'s fixed batch | ✅ Complete |
| Why "batch every N minutes" fails: added latency, doesn't solve the rate mismatch | ✅ Complete |
| Conceptual foundation: producer/consumer decoupling, backpressure, partitioning, consumer groups, offsets | ✅ Complete |
| Real bounded-queue backpressure demo, measured `put()` blocking timings | ✅ Complete |
| Real toy partitioned log (SHA-256 keyed), independent consumer offsets, real skew observed | ✅ Complete |
| Real crash-and-resume from a committed offset, hypothesis-first, asserts passing | ✅ Complete |
| Real failure-mode counterpart: offset loss without external persistence, measured duplicates | ✅ Complete |

## Why it matters

Batch processing (`01`, `02`) assumes the data already exists in full before you start. Most
production data doesn't arrive that way — events, transactions, sensor readings, and logs keep
coming, and a system has to make real-time decisions about backpressure, ordering, and recovery
without ever seeing "all the data" at once. This topic builds every one of those decisions by hand,
in plain Python, so that when `04-kafka` introduces a real broker that automates and hardens all of
it, none of it looks like magic — it's the same mechanisms, demonstrated here first, that a broker
just does more durably and at scale.

## Prerequisites

- `01-why-distributed-processing` — this topic explicitly reuses its partitioning and shuffle-skew
  vocabulary, applied to streams instead of batches.
- `02-pyspark-local-mode` — same lineage of ideas (partitioning, parallelism), now for
  continuously-arriving data instead of a fixed DataFrame.
- Python's `queue`, `threading`, and `hashlib` standard-library modules — no new installs required.

## What you'll build

- A bounded `queue.Queue(maxsize=3)` producer/consumer pair (two real threads), with every
  `put()` call individually timed to show genuine blocking once the queue fills — not simulated,
  measured.
- A 4-partition toy log (`hashlib.sha256`-keyed), 10 events across 4 keys, with two independent
  consumer objects tracking separate per-partition offsets over the same log — plus an
  unplanned-but-kept real skew result (7 of 10 events landing in one partition) that doubles as a
  Failure-modes example.
- A genuine crash-and-resume scenario: a consumer processes 4 of 10 messages, "crashes," and a
  brand-new consumer object — constructed independently, with no reference to the crashed one —
  resumes from the externally-stored committed offset and finishes correctly (verified by
  assertion against the original log).
- The matching failure-mode counterpart: the identical crash scenario with the offset stored only
  in-process, showing real measured duplicate reprocessing (4 of 10 messages processed twice) when
  there's no external, durable offset store.

## Where it appears in real systems

Apache Kafka (and Kinesis, Pub/Sub, Redpanda) automate exactly these four mechanisms in
production: partitions become replicated, durable, append-only logs; the bounded-queue intuition
becomes client-side and broker-side flow control; consumer groups get automatic rebalancing when
members join or leave; and offsets are committed to a durably replicated internal topic instead of
a plain Python `dict`. Any system reading a stream of events in real time — fraud detection,
live dashboards, event-driven microservices, log aggregation — relies on this exact vocabulary.

## What's next

`04-kafka` (not yet built) — the practical treatment: real Kafka client code for topics,
partitions, consumer groups, and offset commits, checking first whether a broker is genuinely
available in this environment to run it for real, and if not, written and reviewed but explicitly
marked as not executed — same honesty discipline used throughout this course. It picks up directly
from this topic's from-scratch mechanisms and shows what a real broker does with them.
