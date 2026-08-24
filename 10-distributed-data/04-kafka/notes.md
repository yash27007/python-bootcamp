# Kafka

> **Note on template substitution:** like `01-why-distributed-processing`, `02-pyspark-local-mode`,
> and `03-streaming-fundamentals`, this is a systems/tooling topic, not a numerical-modeling one —
> there is no loss function or derivative underlying "what is a replicated log" or "what is a
> consumer-group rebalance." Per `docs/superpowers/plans/2026-08-24-phase5-distributed-data.md`'s
> Global Constraints, the "Mathematical foundation" section below is replaced with a **Conceptual
> foundation** section covering Kafka's log-based storage model, consumer groups and rebalancing,
> and delivery semantics — this substitution is documented here inline, as required.

> **Honesty notice, stated once up front and repeated at the practical-code section below:** this
> environment has **no Kafka broker available** (`nc -z localhost 9092` fails; no Docker daemon —
> the identical finding `08-mlops-deployment/01-docker/notes.md` already documents for this
> environment). The producer/consumer code in this topic was **written and carefully reviewed**
> against the real `kafka-python` API (v3.0.11, added via `uv add kafka-python`), but it has **not
> been executed here**. No output anywhere in this file is fabricated — the Experiment section
> below is explicitly a walkthrough of *expected* behavior, not a claimed result.

## Problem

`03-streaming-fundamentals` built a real, working producer/consumer system from scratch: a bounded
`queue.Queue` for backpressure, a toy partitioned log keyed by `hashlib.sha256`, and a
crash-and-resume demo where a fresh consumer object read a committed offset from an external
store and resumed correctly. That from-scratch system genuinely works — the demo's asserts pass.
But it has two limitations baked into how it was built, both demonstrated directly in that topic's
own code:

1. **The log lives in one process's memory.** `03`'s partitioned log is `list[list[tuple]]` —
   ordinary Python lists, living on the heap of one Python process. If that process crashes (not
   the toy "crash" simulated by just not calling a method further, but a real `kill -9` or an
   OOM-killed process), the log itself — every message ever produced — is gone, not just the
   consumer's position in it. The offset-persistence demo saved consumers from losing their
   *place*; nothing in `03` saves the *data* from a process death.
2. **It runs on one machine, in one process.** Producing and consuming both happen inside the same
   Python interpreter, sharing the same `committed_offsets` dict directly. There is no path from
   that code to "the producer runs on server A, three consumer processes run on servers B, C, D,
   and all four coordinate correctly" — nothing in `03`'s code sends anything over a network, and a
   plain Python dict isn't reachable from a different machine.

The problem this topic answers: what does it take to make the producer/consumer/partition/offset
model from `03` **durable across a process (or machine) crash** and **horizontally scalable across
machines**, without the application code that produces and consumes messages having to reimplement
replication and network coordination itself?

## Intuition

Think back to `03`'s checkout-line analogy: a door where customers enter (producer), a line
(queue/log), a cashier working through it (consumer). `03`'s from-scratch version is that whole
scene happening inside a single room that can vanish without warning — if the room burns down, the
line, the customers waiting in it, and the register's record of who's been served all disappear
together. Kafka's answer is to move the line into a building with three separate, independently
maintained copies of the register (replication), staffed by a shift system where any number of
cashiers can each work through *their own copy* of the same line at their own pace without getting
in each other's way or losing their place if one of them goes home sick (consumer groups + durable
offsets) — and critically, the line itself doesn't erase a customer's entry once they're served; it
just keeps a running tally of how far each cashier has gotten (the log-vs-queue distinction below).

## Why simpler approaches fail

`03-streaming-fundamentals`'s own Failure modes and Real-world sections already name exactly this
gap and defer to this topic to close it — worth citing precisely rather than re-deriving:

- *"Losing offset tracking on crash without external persistence... store the offset somewhere
  that outlives the consumer process — a file, a database row, or, in a real streaming system, the
  broker itself."* `03` solved this for the *consumer's position*, using a plain Python dict as the
  "external" store — external only relative to the consumer object, not external to the process or
  machine both live in. A dict living in the same Python process as the code reading it offers zero
  protection against that process's own crash taking the dict down with it.
- *"[Kafka and equivalents] automate and harden [these mechanisms] for production use: partitions
  replicated across brokers for durability, offsets committed to a durably replicated internal
  topic rather than a plain Python `dict`, consumer-group rebalancing handled automatically..."*

A hand-rolled fix — write the log to a file instead of a list, write offsets to a file instead of a
dict — narrows but doesn't close the gap: a single file on a single disk on a single machine is
still one hardware failure away from total loss, still has no story for multiple producer/consumer
processes across machines discovering each other and coordinating partition ownership, and still
requires the application to hand-implement replication (write to N copies, agree on which copy is
authoritative if they disagree) from scratch — precisely the kind of infrastructure-layer problem
`01-why-distributed-processing` argued shouldn't be reimplemented per-application. Kafka is that
infrastructure layer, built once and reused.

## Conceptual foundation (substituting for Mathematical foundation)

### The log-based model — not a queue

This is the single most important distinction in this topic, so it's stated precisely: **a Kafka
partition is an append-only, replicated, durable log — not a traditional message queue.**

A traditional queue (think RabbitMQ's default mode, or `03`'s own `queue.Queue`) treats "consume"
as destructive: a message is delivered to (at most) one consumer, and once acknowledged, it is
*removed* from the queue. This has a direct consequence: if a second consumer wants to read the
same messages independently (compute a different aggregate, archive them, replay them for
debugging), it cannot — the first consumer already deleted them.

A Kafka partition instead behaves like an append-only array on disk, indexed by an integer offset
(`0, 1, 2, 3, ...`), that is **never mutated by reads**. Producing a message appends it at the next
offset. Consuming a message means a consumer reads the record *at* some offset and then advances
its *own, independently-tracked* position — the log itself does not change and nothing is deleted
as a direct result of a read. Two completely different consumer groups can read the exact same
partition from offset 0 to the current end, totally independently, at completely different
speeds, with neither one's reading affecting what the other sees. This is the exact property
`03`'s notes.md flagged directly: *"the log doesn't 'empty out' as it's read, unlike a traditional
queue where consuming a message removes it."*

What does eventually remove data from a Kafka log is **retention** — a configured policy (time-based,
e.g. 7 days, or size-based) that deletes the *oldest* segments of the log regardless of whether
every consumer has read them, freeing disk space. This is an operational/capacity concern, not a
per-consumer "I'm done with this" signal — it's the same distinction as a video platform deleting a
recording after a year vs. marking it "watched" for one particular viewer.

**Replication** is what makes the log durable against a single machine's failure. Each partition
has one **leader** broker (accepts all reads/writes for that partition) and some number of
**follower** brokers that continuously copy the leader's log. A follower that is fully caught up is
called **in-sync (ISR — in-sync replica)**. `acks="all"` on the producer (used in this topic's code
below) means the leader waits for every current ISR to confirm the write before telling the
producer the send succeeded — so a message acknowledged as "produced" survives the leader itself
dying immediately afterward, as long as at least one ISR received it before the leader went down.
A partition with replication factor 1 (no followers) has no such protection — it is exactly as
durable as a single Python list, just running on a server instead of in your own process; this is
directly relevant to the "under-replicated partitions" failure mode below.

### Consumer groups and rebalancing

`03`'s "consumer groups" description carries over unchanged: a **consumer group** is a named set of
consumer processes that, between them, collectively read every partition of a topic, with each
partition assigned to exactly one consumer *within* a given group at a time (so within a group,
work is load-balanced across partitions; across different groups, each group gets its own
independent full copy of the stream). What Kafka adds beyond `03`'s toy version is the mechanism
by which "assigned to exactly one consumer" is decided and kept correct as consumers join, leave,
or crash — **rebalancing**.

A rebalance happens when group membership changes (a new consumer joins, an existing one leaves
cleanly, or the broker stops receiving heartbeats from one and presumes it dead). The group
coordinator (a designated broker) recomputes the partition-to-consumer assignment from scratch and
tells every consumer in the group its new assignment. During a rebalance, **the group as a whole
stops making progress** — every consumer must pause, receive its new assignment, and (depending on
whether offsets were committed before the rebalance began) potentially reprocess or skip messages
around the boundary — this is exactly why rebalancing storms (Failure modes, below) are damaging:
they are not "a partition moves," they are "the entire group's throughput drops to zero, repeatedly,
in quick succession."

### Delivery semantics: at-least-once, at-most-once, and why exactly-once is hard

`03`'s Experiment section already surfaced this precisely, in its own from-scratch terms — worth
deriving fully here because it is the central design decision any consumer of a Kafka-backed system
has to make explicitly:

- **At-most-once**: commit the offset *before* processing the message. If the consumer crashes
  between the commit and finishing processing, the message is never processed, but the offset says
  it was — the message is silently lost. Nothing about this requires Kafka specifically; it's a
  direct consequence of the ordering of "record progress" vs. "do the work."
- **At-least-once**: commit the offset *after* processing completes successfully (this topic's code
  below does this deliberately). If the consumer crashes after processing but before the commit
  finishes, the next consumer to own that partition re-reads from the last *committed* offset and
  reprocesses the same message — safe against loss, at the cost of possible duplicate processing.
  This is what `03`'s Experiment measured directly for its own in-memory version: reprocessing
  `event-0` through `event-3` when the offset store didn't survive a crash is exactly this failure
  mode, just triggered by "no external store at all" rather than "external store, but a crash in
  the narrow gap between processing and committing."
- **Exactly-once**: neither lost nor duplicated. This is not simply "pick a better commit order" —
  it runs into a genuinely hard problem: **the dual-write problem.** "Process the message" and
  "record that it was processed" are, in the general case, two *separate* writes to two *separate*
  systems (e.g. write a row to a database, and separately commit a Kafka offset). There is no way
  to make two independent writes to two independent systems atomic without a coordinating protocol
  between them — either the database write succeeds and the offset commit fails (reprocessing, a
  duplicate database write, if that write wasn't idempotent), or the offset commit succeeds and the
  database write is later found to have failed (the message is now lost, despite being marked
  processed). Kafka's own transactional producer API (`transactional.id`, atomic multi-partition
  writes) solves this *only when the entire pipeline is Kafka-to-Kafka* — read from one topic,
  transform, write to another topic, and commit the *consumer offset itself* as part of the same
  Kafka transaction, so a crash mid-transaction rolls back the write and the offset together,
  atomically, because both live inside the one system that can make that atomicity guarantee. The
  moment processing touches an external system Kafka doesn't control (a database, an HTTP call, a
  file), the dual-write problem reappears and true exactly-once requires the external system to
  cooperate too — typically via idempotent writes keyed on message identity (e.g. an upsert keyed
  on `(partition, offset)`, so reprocessing the same message a second time is a harmless no-op
  rather than a duplicate). **In practice, most real systems choose at-least-once delivery plus
  idempotent downstream processing** — this achieves an effectively-exactly-once *outcome* without
  requiring a hard distributed-transaction guarantee, which is exactly why this topic's consumer
  code below commits manually and after processing, and calls out where idempotency would need to
  be added for a real downstream write.

## Algorithm

**Producing a message:**
1. Application calls `producer.send(topic, key=k, value=v)`.
2. The client library computes `partition = hash(k) % num_partitions` (or uses a custom
   partitioner) — same idea as `03`'s `stable_hash(key) % P`, now computed client-side against the
   real partition count fetched from the broker's metadata.
3. The message is appended to an internal batch for that partition (batching controlled by
   `linger_ms`/`batch.size`) and sent to the partition's leader broker once the batch flushes.
4. The leader appends the message to its local log at the next offset, replicates to followers,
   and — if `acks="all"` — waits for all current ISRs to acknowledge before responding success to
   the producer.

**Consuming a message (in a consumer group):**
1. Consumer calls `poll()` (iterating a `KafkaConsumer` object does this internally); the broker
   returns a batch of records from the partition(s) currently assigned to this consumer.
2. Consumer processes each record.
3. Consumer commits its offset for that partition — either automatically on a timer
   (`enable_auto_commit=True`, the default) or manually (`consumer.commit(...)`, this topic's
   choice, made explicitly to control *when* relative to processing the commit happens).
4. On restart or rebalance, a consumer resumes each newly-assigned partition from that partition's
   last committed offset for its group (or `auto_offset_reset`'s policy — `earliest`/`latest` — if
   no committed offset exists yet for that group).

## From-scratch implementation

N/A for this topic specifically — `03-streaming-fundamentals` already *is* the from-scratch half of
this exact idea (partitioned log, consumer offsets, crash-and-resume), executed for real with real
captured output. This topic is deliberately the practical/production half of that same bridge,
mirroring how `02-pyspark-local-mode` was the practical half of `01-why-distributed-processing`'s
from-scratch multiprocessing comparison. Re-implementing a miniature broker with real network
replication here would not add insight beyond what `03` already demonstrated about the underlying
mechanisms — it would just be a smaller, buggier Kafka.

## Practical implementation

**Honesty notice (repeated from the top of this file):** no Kafka broker is available in this
environment. The code below is **real and has been reviewed carefully against the actual
`kafka-python` v3.0.11 API** (the version pinned via `uv add kafka-python` in this repo's
`pyproject.toml`/`uv.lock`) — correct imports, correct constructor arguments, correct method
signatures — but it has **not been executed**, and nothing below is a claimed or fabricated run
result.

Topic creation (documented as text only, matching this file's honesty discipline —
[`kafka_topics_commands.sh`](kafka_topics_commands.sh) has the full, commented version):

```bash
# NOT RUN — reference only, requires a real Kafka install's bin/ on PATH
kafka-topics.sh --create \
  --bootstrap-server localhost:9092 \
  --topic clickstream-events \
  --partitions 4 \
  --replication-factor 1
```

Producer and consumer, in full, in [`kafka_producer_consumer.py`](kafka_producer_consumer.py) in
this folder. Key excerpts:

```python
def make_producer() -> KafkaProducer:
    return KafkaProducer(
        bootstrap_servers=BOOTSTRAP_SERVERS,
        acks="all",              # durability: wait for all in-sync replicas
        retries=5,
        enable_idempotence=True,  # producer-side dedup on retry
        value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        key_serializer=lambda k: k.encode("utf-8") if k is not None else None,
        linger_ms=10,
    )
```

```python
def make_consumer() -> KafkaConsumer:
    return KafkaConsumer(
        TOPIC,
        bootstrap_servers=BOOTSTRAP_SERVERS,
        group_id=CONSUMER_GROUP,
        enable_auto_commit=False,       # deliberate: commit manually, after processing
        auto_offset_reset="earliest",
        value_deserializer=lambda v: json.loads(v.decode("utf-8")),
        key_deserializer=lambda k: k.decode("utf-8") if k is not None else None,
    )
```

```python
    for message in consumer:
        # ... process message.value here ...
        tp = TopicPartition(message.topic, message.partition)
        consumer.commit(offsets={tp: (message.offset + 1, None)})  # commit AFTER processing
```

This maps directly back to `03`'s from-scratch pieces: `producer.send(key=..., value=...)` is the
real version of `03`'s `stable_hash(key) % P` append; `group_id=` is the real version of `03`'s
manually-constructed `Consumer(group="group-1", ...)`; `consumer.commit(offsets={...})` is the real,
network-durable version of `03`'s `offset_store[group] = position` dict write — same idea, now
persisted to the broker's internal `__consumer_offsets` topic instead of process memory.

## Experiment

Since there is no broker to run a real produce/consume round-trip against, this section is written
as an **expected-behavior walkthrough**, not a fabricated "Result" — every step below follows
directly from the Algorithm and Conceptual-foundation sections above and the client library's
documented behavior, but none of it was observed running.

**Setup (as it would run):** the topic `clickstream-events` created with 4 partitions and
replication factor 1 (single-broker toy cluster). `python kafka_producer_consumer.py produce` run
in one terminal, `python kafka_producer_consumer.py consume` in another, with `group_id=
"clickstream-aggregator"`.

**Expected walkthrough:**

1. `produce_events()` sends 20 events, keyed `user-0` through `user-3` round-robin. For each key,
   the client library hashes the key to pick a partition (0–3) — every `user-2` event, for
   instance, is expected to land in the same partition as every other `user-2` event, for the same
   reason `03`'s `hashlib.sha256`-keyed partitioning was stable per key.
2. Each `producer.send(...).get(timeout=10)` call is expected to block briefly and then return a
   `RecordMetadata` with the assigned `partition` and the `offset` the broker assigned — offsets
   start at 0 for a fresh topic and increase by 1 per message *within a partition* (not
   topic-wide — partition 0's offsets and partition 1's offsets are independent counters).
3. With `acks="all"` and replication factor 1, "all in-sync replicas acknowledged" trivially means
   "the single leader broker wrote it" — there are no followers to wait for, so this setting adds
   no actual durability at replication factor 1 (this is called out explicitly, not glossed over,
   in Failure modes below).
4. `consume_events()`, started in `clickstream-aggregator`, is expected to receive all 20 messages
   (assuming it starts before or joins while messages are still in the log — `auto_offset_reset=
   "earliest"` means a *brand-new* group with no prior committed offset starts from the beginning
   of the log, not just newly-arriving messages).
5. For each message, the consumer is expected to print `partition=... offset=... key=... value=...`,
   then call `consumer.commit(offsets={tp: (message.offset + 1, None)})` — committing offset+1
   (the next position to read) is the API's documented convention, so that a resumed consumer's
   next read is the message *after* the one just committed, not a re-read of the same one.
6. If a second process were started with the *same* `group_id` while the first is still running, a
   rebalance would be triggered: the group coordinator would reassign some of the 4 partitions to
   the new consumer, and each consumer would resume its newly-assigned partitions from that
   partition's last committed offset for the group — this is the real, broker-mediated version of
   `03`'s crash-and-resume demo, except triggered by *scaling up* the group rather than a crash.

**Why this is presented this way rather than as a "Result":** claiming any of the above was
observed would violate this repository's explicit constraint (`AGENTS.md`, and this plan's Global
Constraints) against fabricating execution output. Every claim above is instead traceable either to
the `kafka-python` library's documented API contract or to `03`'s already-executed, structurally
identical from-scratch behavior — not to a run that happened here.

## Failure modes

- **Consumer group rebalancing storms.** If group members join and leave in quick succession (e.g.
  a consumer that's flapping — crashing on startup, or failing health checks and being restarted
  repeatedly by an orchestrator), every membership change triggers a new rebalance, and — as the
  Conceptual foundation section states precisely — the *entire group* pauses processing during
  each one. A group stuck in repeated rebalances can end up spending more wall-clock time
  rebalancing than actually consuming, degrading throughput to near zero even though no single
  consumer is permanently down. Mitigations: increase `session.timeout.ms`/heartbeat tolerance so a
  transient blip doesn't trigger a full rebalance, fix whatever is actually causing consumers to
  restart repeatedly, and prefer incremental/cooperative rebalancing protocols (supported by newer
  Kafka versions) over the "stop-the-world" eager rebalancing described above.
- **Under-replicated partitions.** A partition is under-replicated when one or more of its follower
  replicas has fallen out of sync with the leader (or is offline) — the ISR set is smaller than the
  configured replication factor. This directly weakens the durability guarantee `acks="all"`
  provides: `acks="all"` only waits for the *current* ISR set, so if that set has shrunk to just
  the leader (as it always trivially is at replication factor 1, per the Experiment section above),
  a leader crash immediately after acknowledging a write loses that write, with no follower having
  a copy to promote. This is the exact same class of failure `03`'s "offset loss without external
  persistence" demo showed for a single Python dict, one level up the stack — durability that
  looks solid until the specific component holding the only copy fails.
- **Offset-commit-before-vs-after-processing: the at-least-once/at-most-once tradeoff, precisely.**
  Already derived above, worth restating as the failure-mode framing: committing *before*
  processing (at-most-once) means a crash in the gap loses the message silently — the offset says
  "done," the work never happened, and there is no signal anywhere that anything is wrong.
  Committing *after* processing (at-least-once, this topic's choice) means a crash in the gap
  reprocesses the message — visible, bounded (at most the in-flight batch is duplicated, not
  unboundedly many messages), and safe as long as downstream processing is idempotent. Between the
  two, at-least-once is almost always the better default specifically *because* silent loss has no
  recovery path once it's happened, while a duplicate can be de-duplicated after the fact if the
  downstream write is idempotent (or tolerated if it isn't catastrophic, e.g. a duplicate log line)
  — this is precisely why this topic's `consume_events()` commits after processing, not before.

## Real-world usage

Kafka (and its close relatives — Amazon MSK/Kinesis, Confluent Cloud, Redpanda, Google Pub/Sub with
some semantic differences) is the standard substrate for event-driven architectures at scale:
clickstream and analytics pipelines (feeding both real-time dashboards and batch warehouses off the
same topic — the same "two independent consumer groups reading the same log" property from the
Conceptual foundation section), microservice-to-microservice communication that needs to survive a
downstream service being temporarily unavailable (the durable log absorbs the outage; the consumer
catches up from its last committed offset once the service returns), change-data-capture (streaming
every row-level database change as an event, e.g. via Debezium, so other systems can react without
polling the database), and log/metric aggregation feeding observability systems. The delivery
semantics derived in this topic — and the deliberate choice of at-least-once plus idempotent
downstream writes — are the actual, everyday design decision behind essentially all of these,
not a theoretical corner case.

## Mental model

A traditional queue is a todo list that erases each item once someone claims it — good for
"exactly one worker does exactly this," bad for "let three different systems read the same
history at their own pace." A Kafka partition is a replicated, append-only ledger that never
erases on read — consumers don't take items out, they just move a bookmark forward, and different
readers can hold completely independent bookmarks into the exact same ledger. Durability comes
from replicating the ledger itself across machines; scalability comes from splitting the ledger
into partitions consumers can split among themselves; and the hardest part of the whole system —
exactly-once delivery — is hard specifically because "did the work happen" and "does the bookmark
say it happened" are, in general, two separate facts recorded in two separate places, and nothing
free makes two separate writes atomic.

## Questions to think about

1. `03-streaming-fundamentals`'s from-scratch partitioned log used `hashlib.sha256`-based stable
   hashing for partition assignment specifically because Python's built-in `hash()` is salted
   per-process. `kafka-python`'s default partitioner has the same requirement — why would using a
   *non-stable* hash for partition assignment break a Kafka-based system in a way that a
   *single-process* stable hash would not, if the producer and consumer are different OS processes?
2. This topic's `make_producer()` sets `acks="all"` but the Experiment section notes this adds
   *no* actual durability improvement at replication factor 1. What replication factor would make
   `acks="all"` meaningfully different from `acks="1"` (leader-only), and what does that difference
   actually protect against?
3. The dual-write problem was described in terms of "process the message" and "commit the offset"
   being separate writes. If a consumer's processing step is itself an idempotent database upsert
   keyed on `(topic, partition, offset)`, does that fully eliminate the practical impact of
   at-least-once delivery, or does it just move the remaining risk somewhere else? Where?
4. A rebalancing storm was described as being triggered by a flapping consumer. If a consumer's
   processing logic is slow enough that it can't call `poll()` again before `max.poll.interval.ms`
   elapses, the broker presumes it dead and triggers a rebalance even though the process never
   actually crashed. What does this imply about the relationship between per-message processing
   time and `max_poll_records`/`max.poll.interval.ms` configuration?
5. `03`'s consumer groups let multiple consumers split a topic's partitions; this topic adds that
   two *different* consumer groups reading the same topic don't interfere with each other at all.
   Given that, why can't a single consumer group with more consumers than partitions ever increase
   throughput further — what happens to the extra consumers?
