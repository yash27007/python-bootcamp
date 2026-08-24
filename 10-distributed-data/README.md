# 10 – Distributed Data

Why one machine stops being enough, first-principles: partitioning, parallelism, and shuffle
before PySpark syntax; then the streaming equivalent — producer/consumer, backpressure, and
partitioned logs — before Kafka. Each practical topic is bridged from a from-scratch
implementation built earlier in this same section.

| # | Topic | Status | Description |
|---|-------|--------|--------------|
| 01 | [Why Distributed Processing](./01-why-distributed-processing/) | ✅ Complete | Single-machine bottleneck, partitioning, map/filter/aggregate, shuffle cost, fault tolerance — a real timed single-thread vs. multiprocessing vs. PySpark comparison |
| 02 | [PySpark Local Mode](./02-pyspark-local-mode/) | ✅ Complete | DataFrame API, lazy execution, real jobs on an 800K-row dataset (`.filter`/`.groupBy().agg()`/`.join()`/`.explain()`), timed against Topic 01's from-scratch version |
| 03 | [Streaming Fundamentals](./03-streaming-fundamentals/) | ✅ Complete | Producer/consumer decoupling, backpressure, partitioned logs, consumer groups, offsets — a from-scratch bounded queue and a real crash-and-resume demo |
| 04 | [Kafka](./04-kafka/) | ✅ Complete | Log-based storage vs. a traditional queue, consumer groups/rebalancing, delivery semantics — real reviewed producer/consumer code, honestly marked as not executed here (no broker available) |

## Prerequisites

- `08-mlops-deployment/07-cicd` — the automate-a-manual-process idea Topic 01 and Topic 03
  both echo.
- Comfort with plain Python (`multiprocessing`, `queue.Queue`) — the from-scratch steps in
  Topics 01 and 03 use only the standard library.

## Environment note

Java is present (`openjdk 21.0.11`), so Topic 02's PySpark jobs run for real in local mode.
No Kafka broker is available in this environment (no daemon, confirmed via a direct port
check) — Topic 04's producer/consumer code is real and reviewed but explicitly marked as not
executed here, the same honesty discipline `08-mlops-deployment/01-docker` uses for its
un-built Dockerfile.

## What's next

`11-generative-ai` onward build specific architectures on top of `09-pytorch`'s foundation;
this section's distributed/streaming vocabulary (partitioning, shuffle, backpressure) reappears
whenever those later sections discuss training at real scale, even though their own code stays
toy-scale per their no-heavy-training constraint.
