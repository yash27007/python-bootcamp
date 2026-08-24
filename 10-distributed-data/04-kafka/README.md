# 04 – Kafka

Detailed notes (log-based storage model vs. a traditional queue, consumer groups and rebalancing,
delivery semantics and the dual-write problem behind why exactly-once is hard — Conceptual-
foundation substitution documented inline): [notes.md](notes.md)

Real, carefully-reviewed `kafka-python` producer/consumer code, honestly marked as **not executed
in this environment** (no Kafka broker available — verified, same finding as
`08-mlops-deployment/01-docker/notes.md`'s Docker daemon check):
[kafka_producer_consumer.py](kafka_producer_consumer.py). Text-only, also-unexecuted
`kafka-topics.sh`/`kafka-consumer-groups.sh` reference commands:
[kafka_topics_commands.sh](kafka_topics_commands.sh).

## What you'll learn

Why `03-streaming-fundamentals`'s from-scratch in-memory partitioned log — genuinely working, but
living entirely inside one Python process's memory — doesn't survive a process crash or scale
across machines, and what Kafka adds to close that gap: partitions as replicated, durable,
append-only logs (explicitly **not** a traditional queue that deletes on consume), consumer groups
with automatic rebalancing, and the precise reasoning behind why exactly-once delivery is a genuinely
hard problem (the dual-write problem) rather than just a missing config flag.

| Topic | Status |
|-------|--------|
| Problem: `03`'s from-scratch log doesn't survive a crash or scale across machines | ✅ Complete |
| Why simpler fixes (a file instead of a list) don't close the gap | ✅ Complete |
| Conceptual foundation: log-vs-queue, replication/ISR, consumer groups, rebalancing | ✅ Complete |
| Delivery semantics derived precisely: at-most-once, at-least-once, why exactly-once is hard | ✅ Complete |
| Real, reviewed producer/consumer code (`kafka-python`, manual offset commits) | ✅ Complete — **written and reviewed, not executed here** (no broker) |
| Topic-creation commands (`kafka-topics.sh`-style), documented as text only | ✅ Complete — not run |
| Expected-behavior walkthrough (explicitly not a fabricated "Result") | ✅ Complete |
| Failure modes: rebalancing storms, under-replicated partitions, commit-order tradeoff | ✅ Complete |

**Honesty note (stated plainly, not just in notes.md):** this environment has no Kafka broker —
confirmed directly (`nc -z localhost 9092` fails; no Docker daemon reachable from this WSL2
environment, the same fact already established for `08-mlops-deployment/01-docker`). The
producer/consumer code in this folder is real, syntactically correct, and reviewed against the
actual `kafka-python` v3.0.11 API, but it has not been run here, and this topic's "Experiment"
section is deliberately written as an expected-behavior walkthrough rather than a claimed result.

## Why it matters

`03-streaming-fundamentals` proved the mechanisms (bounded buffers, partitioned logs, consumer
offsets) work — but proved them inside a single process that can vanish along with everything it's
holding. Kafka is the production answer to "make all of that durable and distributed without every
application reimplementing replication and network coordination by hand" — this is the same
progression `01`→`02` already established for batch (hand-rolled multiprocessing → an engine that
automates partitioning/shuffle/fault-tolerance), now applied to streaming.

## Prerequisites

- `03-streaming-fundamentals` — this topic is explicitly the practical/production half of that
  topic's from-scratch mechanisms; every section here cites it directly rather than re-deriving.
- `kafka-python` (added via `uv add kafka-python`, pinned in this repo's `pyproject.toml`).
- No running Kafka broker is required to read this topic — the code is real but unexecuted here by
  design; a reader with `docker compose` and a Kafka image (or a local Kafka install) can run it.

## What you'll build (conceptually — written, not run here)

- A producer (`make_producer`/`produce_events`) using `acks="all"`, idempotent retries, and
  key-based partitioning, sending JSON-serialized clickstream events.
- A consumer (`make_consumer`/`consume_events`) in a named consumer group, with
  `enable_auto_commit=False` and a manual `consumer.commit(...)` called deliberately *after*
  processing each message — the concrete implementation of the at-least-once choice this topic's
  notes.md derives and argues for.
- Topic-creation, describe, and consumer-group-lag inspection commands, documented as
  copy-pasteable text against a real cluster.

## Where it appears in real systems

Kafka (and Kinesis/Pub/Sub/Redpanda/Confluent Cloud) underlies most production event-driven
architectures: clickstream/analytics pipelines feeding both real-time and batch consumers off the
same topic, service-to-service messaging that survives a downstream outage, change-data-capture
streaming database changes as events, and log/metric aggregation for observability. The
at-least-once-plus-idempotent-downstream-write pattern this topic derives is the default real-world
choice behind nearly all of them.

## What's next

`10-distributed-data`'s four topics are now complete — `01` (why distribute), `02` (PySpark local
mode), `03` (streaming fundamentals, from scratch), and `04` (Kafka, practical). The section README
(`10-distributed-data/README.md`) ties all four together.
