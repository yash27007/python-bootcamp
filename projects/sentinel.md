# Sentinel

**Status:** 🗓 Planned — no repository yet. This card exists so the connection to the curriculum
is explicit before the project starts, not retrofitted after.

## Problem

Detect anomalous machine/sensor behavior from a continuous stream of telemetry, in something
closer to real time than a batch job — catching a failure mode as it develops, not hours later in
a scheduled report.

## Why it matters

Every anomaly-detection technique in `05-machine-learning/16-anomaly-detection` (Isolation
Forest, DBSCAN, LOF) is demonstrated on a static, already-collected dataset. Real machine-health
monitoring means the data arrives continuously, unbounded, and the model has to keep up —
exactly the batch-vs-streaming distinction `10-distributed-data/03-streaming-fundamentals` and
`04-kafka` build the vocabulary for.

## Concepts learned (curriculum cross-references, planned)

- `05-machine-learning/16-anomaly-detection` — the detection algorithms themselves
- `09-pytorch` — if the eventual model is a learned one (e.g. an autoencoder) rather than a
  classical method
- `10-distributed-data` — streaming ingestion, backpressure, partitioning by sensor/machine ID
- `08-mlops-deployment/08-monitoring` — this project's core is architecturally close to what
  `08-monitoring`'s own drift-detection topic teaches, applied to raw telemetry instead of model
  predictions

## Technologies (anticipated, not yet built)

Kafka (or a similar log), a time-series-capable store (ClickHouse is the current plan), MLflow for
tracking whichever detection model gets trained, edge inference for on-device detection.

## Prerequisites

`05-machine-learning/16-anomaly-detection`, `10-distributed-data`, and `08-mlops-deployment`
should all be complete before starting.

## Link to project repository

None yet — will be added here once the project has its own repo.

## Expected learning outcomes

Apply streaming and anomaly-detection concepts to genuinely time-ordered data, and confront the
production concerns (consumer lag, partition skew, alert fatigue) that a static-dataset demo
can't surface.
