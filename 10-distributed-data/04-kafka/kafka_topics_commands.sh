#!/usr/bin/env bash
# Kafka topic-management commands — 04-kafka.
#
# *** NOT RUN IN THIS ENVIRONMENT ***
# These are documented as TEXT/REFERENCE only. There is no Kafka broker
# available here (see kafka_producer_consumer.py's header and notes.md for
# the verified environment fact: no broker on localhost:9092, no Docker
# daemon). Nothing in this file has been executed; it is included so the
# topic-creation step referenced throughout notes.md is concrete and
# copy-pasteable against a real cluster, not left as a vague mention.
#
# Assumes a Kafka install's `bin/` directory is on PATH (or run with the
# full path, e.g. `$KAFKA_HOME/bin/kafka-topics.sh`).

set -euo pipefail

BOOTSTRAP="localhost:9092"
TOPIC="clickstream-events"

# --- Create the topic used by kafka_producer_consumer.py ---
# --partitions 4:   matches this topic's notes.md discussion of parallelism —
#                    up to 4 consumers in one group can each own one partition.
# --replication-factor 1: single-broker toy setup. A real deployment uses 3
#                    (see notes.md's "under-replicated partitions" failure
#                    mode for why 1 offers zero durability against broker loss).
kafka-topics.sh --create \
  --bootstrap-server "$BOOTSTRAP" \
  --topic "$TOPIC" \
  --partitions 4 \
  --replication-factor 1 \
  --config retention.ms=604800000    # 7-day retention, explicit rather than default

# --- Inspect the topic: partition count, leader/replica/ISR assignment ---
kafka-topics.sh --describe \
  --bootstrap-server "$BOOTSTRAP" \
  --topic "$TOPIC"

# --- List all topics on the cluster ---
kafka-topics.sh --list \
  --bootstrap-server "$BOOTSTRAP"

# --- Inspect consumer group state: current offsets, log-end offsets, LAG ---
# LAG per partition is the single most useful number for diagnosing a slow
# or stuck consumer — see notes.md's "consumer lag" / rebalancing-storm
# discussion in Failure modes.
kafka-consumer-groups.sh --describe \
  --bootstrap-server "$BOOTSTRAP" \
  --group clickstream-aggregator

# --- Delete the topic (irreversible; only ever used for cleanup) ---
kafka-topics.sh --delete \
  --bootstrap-server "$BOOTSTRAP" \
  --topic "$TOPIC"
