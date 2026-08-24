"""
Kafka producer/consumer reference implementation — 04-kafka.

*** IMPORTANT — READ BEFORE RUNNING ***
This script was WRITTEN AND CAREFULLY REVIEWED for correctness against the real
`kafka-python` API (v3.0.11, pinned via `uv add kafka-python` in this repo's
pyproject.toml), but it has NOT BEEN EXECUTED in this environment. There is no
Kafka broker available here (`nc -z localhost 9092` fails, no Docker daemon —
see `08-mlops-deployment/01-docker/notes.md` for the identical Docker-daemon
finding). No output below or in notes.md is fabricated. This follows the same
honesty discipline as that Dockerfile: real, reviewed code; explicitly marked
as unexecuted.

To actually run this against a real broker:
    1. Start Kafka locally (e.g. `docker compose up -d` with a kafka+zookeeper
       or KRaft-mode single-node compose file, or a local binary install).
    2. Create the topic first — see `kafka_topics_commands.sh` in this folder
       for the exact `kafka-topics.sh` invocation (also unexecuted, documented
       as text only).
    3. `python kafka_producer_consumer.py produce`   (in one terminal)
    4. `python kafka_producer_consumer.py consume`   (in another terminal)
"""

from __future__ import annotations

import json
import sys
import time

from kafka import KafkaProducer, KafkaConsumer, TopicPartition
from kafka.errors import KafkaError

BOOTSTRAP_SERVERS = ["localhost:9092"]
TOPIC = "clickstream-events"
CONSUMER_GROUP = "clickstream-aggregator"


def make_producer() -> KafkaProducer:
    """Build a producer with explicit, deliberate delivery-safety settings.

    acks="all": the leader waits for every in-sync replica to acknowledge the
    write before the produce call is considered successful — this is what
    makes a produced message durable against a single broker failing right
    after the write (see notes.md's "log-based model" section for why this
    matters: a partition is a replicated log, not a single in-memory list).

    retries + a bounded idempotence setting reduce (but, as notes.md explains,
    do not by themselves eliminate) duplicate-send risk on retry.
    """
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        acks="all",
        retries=5,
        enable_idempotence=True,  # producer-side dedup on retry, per-partition
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k is not None else None,
        linger_ms=10,  # small batching window — real throughput/latency tradeoff
    )


def produce_events(n: int = 20) -> None:
    """Produce `n` toy clickstream events, keyed by user_id.

    The key determines partition assignment (see notes.md's "Conceptual
    foundation" section for the hash-partitioning explanation) — every event
    for the same user_id lands in the same partition, preserving per-user
    ordering, exactly the property `03-streaming-fundamentals`'s from-scratch
    partitioned log demonstrated with `hashlib.sha256`-based routing.
    """
    producer = make_producer()
    try:
        for i in range(n):
            user_id = f"user-{i % 4}"
            event = {
                "user_id": user_id,
                "action": ["click", "view", "purchase"][i % 3],
                "seq": i,
                "ts": time.time(),
            }
            # send() is async and returns a FutureRecordMetadata; get() blocks
            # until the broker acknowledges (or raises on failure/timeout).
            future = producer.send(TOPIC, key=user_id, value=event)
            try:
                record_metadata = future.get(timeout=10)
                print(
                    f"produced seq={i} key={user_id} -> "
                    f"partition={record_metadata.partition} "
                    f"offset={record_metadata.offset}"
                )
            except KafkaError as exc:
                # A real producer must decide here: retry, dead-letter, or
                # crash loudly. Silently swallowing this is how messages go
                # missing without anyone noticing.
                print(f"FAILED to produce seq={i}: {exc}", file=sys.stderr)
        producer.flush()  # block until all buffered/batched sends complete
    finally:
        producer.close()


def make_consumer() -> KafkaConsumer:
    """Build a consumer in a named consumer group with MANUAL offset commits.

    enable_auto_commit=False is the deliberate choice this topic's notes.md
    hinges on: with auto-commit, the client library commits offsets on a
    timer regardless of whether processing actually finished, which can
    silently produce at-most-once loss on a crash between commit and
    completion. Manual commit lets us choose to commit strictly AFTER
    processing succeeds — at-least-once — and make that choice visible in
    code rather than hidden in a background timer.
    """
    return KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=CONSUMER_GROUP,
        enable_auto_commit=False,
        auto_offset_reset="earliest",  # first read of a group: start from log start
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k is not None else None,
        max_poll_records=10,
    )


def consume_events(max_messages: int = 20) -> None:
    """Consume events as part of CONSUMER_GROUP, committing offsets manually
    AFTER each message is fully processed (at-least-once semantics).

    Every consumer process started with the same group_id shares the topic's
    partitions between them (a "consumer group" — see notes.md); if this
    process crashes and restarts, `auto_offset_reset`/the group's last
    committed offset (not this process's memory) determines where it resumes
    — the same idea `03-streaming-fundamentals`'s from-scratch offset store
    demonstrated with a plain dict, now durably persisted by the broker
    itself in an internal `__consumer_offsets` topic.
    """
    consumer = make_consumer()
    processed = 0
    try:
        for message in consumer:
            # message.partition / message.offset / message.key / message.value
            print(
                f"received partition={message.partition} offset={message.offset} "
                f"key={message.key} value={message.value}"
            )

            # --- "processing" happens here ---
            # In a real system this is where business logic runs: write to a
            # database, update an aggregate, trigger a downstream call, etc.
            # If this step raises, we fall through to `finally`/loop-exit
            # WITHOUT committing — the next poll (this consumer or another
            # in the group, after a rebalance) will re-deliver this message.
            # That is exactly at-least-once: safe against message loss,
            # at the cost of needing idempotent processing downstream.

            # Commit only THIS message's partition/offset, and only after
            # processing succeeded — committing offset+1 (the next position
            # to read), which is the API's convention.
            tp = TopicPartition(message.topic, message.partition)
            consumer.commit(offsets={tp: (message.offset + 1, None)})

            processed += 1
            if processed >= max_messages:
                break
    finally:
        consumer.close()


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else ""
    if mode == "produce":
        produce_events()
    elif mode == "consume":
        consume_events()
    else:
        print("Usage: python kafka_producer_consumer.py [produce|consume]")
        sys.exit(1)
