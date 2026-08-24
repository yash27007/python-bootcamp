# Streaming Fundamentals

> **Note on template substitution:** like `01-why-distributed-processing` and
> `02-pyspark-local-mode`, this is a systems/tooling topic, not a numerical-modeling one — there is
> no loss function or derivative underlying "what is a queue" or "what is a partition offset." Per
> `docs/superpowers/plans/2026-08-24-phase5-distributed-data.md`'s Global Constraints, the
> "Mathematical foundation" section below is replaced with a **Conceptual foundation** section
> covering producer/consumer decoupling, backpressure, stream partitioning, consumer groups, and
> offsets — this substitution is documented here inline, as required.

## Problem

`01-why-distributed-processing` and `02-pyspark-local-mode` both start from a **fixed, already
collected** dataset sitting on disk: 800,000 rows, generated once, read once, processed once, done.
Real systems frequently don't have that luxury. Clickstream events, sensor readings, payment
transactions, log lines — these arrive continuously, indefinitely, one at a time or in small
bursts, with no natural "end" to wait for before starting to process them. The problem this topic
answers: how do you process data that never finishes arriving, when the rate it arrives at is not
under your control?

## Intuition

Picture a single checkout line at a store (the consumer) and a door where customers keep entering
(the producer). If customers enter faster than the cashier can ring them up, one of three things
happens: customers pile up in an ever-growing line (unbounded queue — eventually the store runs out
of floor space), the store stops letting new customers in until there's room (backpressure), or the
store hires more cashiers and splits customers into separate lines by, say, last name (partitioning
— each line stays internally ordered, and lines run in parallel). Every mechanism in this topic is
one of those three responses, formalized.

A concrete number: if events arrive at 1,000/second and take 2ms each to process (500/second
consumer throughput), the gap is 500 unconsumed events *every second*. After one minute that's
30,000 events sitting somewhere. Sitting where, and what happens when "somewhere" runs out of room,
is exactly the question this topic's from-scratch code answers with real, measured behavior.

## Why simpler approaches fail

The obvious first idea: "just batch-process every N minutes" — collect whatever arrived, run
`01`/`02`'s batch pipeline on it, repeat. This looks like it sidesteps the whole streaming problem
by turning it back into a batch problem. It fails on two independent axes:

- **Latency.** Every event now waits up to N minutes before being processed at all, even if the
  system is otherwise idle. For anything that needs to react to an event (fraud detection, an
  alert, a live dashboard), that delay defeats the purpose of having the data arrive continuously
  in the first place.
- **It doesn't solve the rate mismatch, it just delays discovering it.** If the producer is faster
  than the consumer *can ever be*, batching every N minutes just means each batch is bigger than the
  last — the backlog still grows without bound, it's now measured in batches instead of individual
  events. The underlying problem — what do you do when arrivals outpace processing — is untouched.
  Part 1's demo below shows what an **unbounded** buffer does under exactly this condition: it grows
  without limit, silently, until memory runs out. A **bounded** buffer is the only thing that turns
  that silent failure into a visible, handleable one (backpressure) — see the Failure modes section
  for the OOM this specifically prevents.

## Conceptual foundation (substituting for Mathematical foundation)

**Producer/consumer decoupling via a queue.** A *producer* generates events; a *consumer* processes
them. Coupling them directly (producer calls consumer's function inline) means the producer's
throughput is capped at the consumer's speed, and a slow or crashed consumer stalls the producer
entirely. A *queue* between them decouples the two: the producer's job is only "put items in the
queue," the consumer's job is only "take items out and process them." Each can run at its own pace,
in its own thread/process/machine, as long as something exists to hold items in between.

**Backpressure.** A queue with unlimited capacity (`queue.Queue()` with no `maxsize`) removes the
coupling entirely — the producer never waits, and the queue grows exactly as fast as the rate
mismatch. A **bounded** queue (`queue.Queue(maxsize=N)`) reintroduces coupling on purpose: once the
queue holds `N` items, `put()` blocks until the consumer frees a slot. This is backpressure — the
consumer's slowness propagates *backward* to the producer, forcing it to slow down rather than
letting the buffer between them grow forever. It converts an invisible, delayed failure (memory
exhaustion, eventually) into an immediate, visible one (the producer visibly stalls) — which is a
strictly better failure to have, because it's a signal something needs attention now, not a debt
that compounds silently.

**Partitioning a stream.** A single queue is a single point of serialization: even if multiple
consumer threads read from it, order is only guaranteed queue-wide, and one hot key's traffic
competes with every other key's traffic for the same buffer. Partitioning splits the stream into
`P` independent sub-streams by a **stable hash of a key** (e.g. `hash(user_id) % P`), so:

- every event for a given key always routes to the same partition — **order is preserved within a
  partition**, which is exactly (and only) as strong a guarantee as most stream-processing use
  cases actually need (e.g. "this user's events arrive in order," not "all users' events arrive in
  one global order").
- different partitions can be consumed **in parallel**, by different consumer threads/processes/
  machines, with no coordination needed between them.

The tradeoff: partitioning only balances load if the key distribution is itself balanced. A
hot key (or too few distinct keys relative to `P`) routes disproportionate traffic to one
partition — this is the identical skew failure mode `01-why-distributed-processing` and
`02-pyspark-local-mode` demonstrated for batch shuffles and joins, now showing up in a stream's
partition assignment instead (see Part 2's real output and Failure modes below).

**Consumer groups.** Multiple consumers can read the *same* partitioned log independently, each
tracking its own position — this is a "consumer group": a set of consumers that between them cover
every partition, without any one consumer's progress affecting another's. Two different consumer
groups can read the same log completely independently (e.g. one group computing real-time metrics,
another archiving raw events) — the log doesn't "empty out" as it's read, unlike a traditional
queue where consuming a message removes it. (`04-kafka` builds directly on this: a Kafka partition
*is* an append-only log that consumers read via position, not a message queue that deletes on
consume.)

**Offsets.** A consumer's position into a partition — "I have processed the first `k` messages of
this partition" — is its *offset*. The offset is the entire durable state a consumer needs to
resume correctly after a crash, **provided the offset itself is stored somewhere that survives the
crash** — i.e. not only as an attribute on the (crashable) consumer object itself. This is the
single idea Part 3 below demonstrates end to end with real, executed code: a consumer crashes,
its in-process state is gone, but a fresh consumer object reads the *externally stored* offset and
resumes exactly where the crashed one left off.

## Algorithm

**Bounded-queue backpressure:**
1. Producer calls `queue.put(item)`.
2. If the queue is at `maxsize`, `put()` blocks (does not return) until the consumer calls
   `queue.get()` and frees a slot.
3. Consumer calls `queue.get()`, processes the item, optionally calls `queue.task_done()`.
4. Repeat. The producer's effective rate is capped at the consumer's processing rate once the
   buffer is full.

**Partitioned log with independent offsets:**
1. Maintain `P` lists (partitions), each an append-only sequence of messages.
2. On producing a message with key `k`: compute `partition = stable_hash(k) % P`; append to
   `log[partition]`.
3. Each consumer maintains its own `offsets: dict[partition -> int]` (or list indexed by
   partition).
4. To consume: read `log[partition][offset]`, process it, then increment `offsets[partition]`.
5. Different consumers reading the same partitions do not interfere — each has its own offset
   state.

**Crash-and-resume:**
1. Consumer starts by reading its offset from a store *external to itself* (e.g. a shared dict, or
   in a real system, the broker) — `position = offset_store.get(group, default=0)`.
2. Process `log[position]`; on success, advance `position` and immediately write it back to the
   external offset store (commit *after* processing — see Failure modes for the alternative).
3. On crash: whatever was in the consumer object's memory is lost. The last value written to the
   external offset store is not.
4. A fresh consumer object, started with a new `Consumer(...)` call, reads the same external
   offset store, finds the last committed value, and resumes from there — with no coordination
   with the crashed instance needed, because it never existed as far as the fresh instance is
   concerned.

## From-scratch implementation

All three demos below are standard-library Python (`queue`, `threading`, `hashlib`), actually run,
with real captured output — see
[03-streaming-fundamentals.ipynb](03-streaming-fundamentals.ipynb) for the full executed notebook.
Excerpts:

**Backpressure** (bounded `queue.Queue(maxsize=3)`, a producer trying to push 6 items instantly
against a consumer that takes 0.3s per item):

```
  producer: put(0) took 0.000s (queue size now 1)
  producer: put(1) took 0.000s (queue size now 2)
  producer: put(2) took 0.000s (queue size now 3)
  producer: put(3) took 0.000s (queue size now 3)
  consumer: processed 0
  producer: put(4) took 0.300s (queue size now 3)
  consumer: processed 1
  producer: put(5) took 0.300s (queue size now 3)
  consumer: processed 2
  consumer: processed 3
  consumer: processed 4
  consumer: processed 5

total wall time: 1.803s
put() timings (item, seconds): [(0, 0.0), (1, 0.0), (2, 0.0), (3, 0.0), (4, 0.3), (5, 0.3)]
```

Once the queue is genuinely full, `put(4)` and `put(5)` each measurably block for ~0.3s — the
consumer's exact per-item processing time. That is backpressure, timed directly, not asserted.

**Partitioned log** (4 partitions, 10 events across 4 keys, SHA-256-based stable hashing):

```
  partition 0: []
  partition 1: [('user-3', 'click'), ('user-3', 'purchase')]
  partition 2: [('user-4', 'click')]
  partition 3: [('user-1', 'click'), ('user-2', 'click'), ('user-1', 'purchase'),
                ('user-1', 'logout'), ('user-2', 'purchase'), ('user-1', 'click'),
                ('user-2', 'logout')]
```

Every key's events land in exactly one partition (verified programmatically), and partition 3
happens to receive 7 of the 10 events for only 2 of the 4 keys — an unplanned but real
demonstration of hash-partition skew, kept rather than rebalanced, and picked up again in Failure
modes.

**Crash-and-resume** — see Experiment below for the full hypothesis-first walkthrough with real
output.

## Practical implementation

Nothing in this topic has a "practical/production library" step of its own — `queue.Queue` *is*
the standard, practical single-process tool for exactly this producer/consumer pattern in Python
(used, for example, inside `multiprocessing` and `concurrent.futures` internally). The
production-grade version of the *distributed, multi-process, durable* mechanisms this topic builds
by hand — a real partitioned log, real consumer groups, real durably-persisted offsets — is Kafka,
covered practically in `04-kafka`. This topic is the from-scratch half of that same bridge
`01`→`02` already built once for batch processing.

## Experiment

**Hypothesis (stated first):** if a consumer commits its offset *after* each successfully
processed message, a fresh consumer object created after a crash can read that committed offset
and resume from exactly the next unprocessed message — zero duplicate processing, zero skipped
messages — **provided the offset store itself survives the crash**.

**Setup:** a 10-message partition log. `consumer_1` processes 4 messages (committing its offset
after each), then simulates a crash before a 5th. A brand-new `consumer_2` object is constructed
afterward, reading its starting offset from the same external `committed_offsets` dict
`consumer_1` wrote to.

**Actual result (real output):**

```
-- run 1: consumer_1 processes 4 messages, then crashes before committing a 5th --
  [consumer_1] starting at offset 0 (from offset store: 'group-1' -> None)
  [consumer_1] processed 'event-0' at offset 0
  [consumer_1] processed 'event-1' at offset 1
  [consumer_1] processed 'event-2' at offset 2
  [consumer_1] processed 'event-3' at offset 3
  [consumer_1] *** CRASH *** after processing 4 messages this run (last committed offset: 4)

offset store after crash: {'group-1': 4}

-- run 2: a FRESH consumer object picks up where the offset store says to resume --
  [consumer_2] starting at offset 4 (from offset store: 'group-1' -> 4)
  [consumer_2] processed 'event-4' at offset 4
  [consumer_2] processed 'event-5' at offset 5
  [consumer_2] processed 'event-6' at offset 6
  [consumer_2] processed 'event-7' at offset 7
  [consumer_2] processed 'event-8' at offset 8
  [consumer_2] processed 'event-9' at offset 9

combined:        ['event-0', ..., 'event-9']  (matches original log exactly)
```

Both `assert`s in the notebook pass: the combined output of both runs equals the original log
exactly, with no duplicate and no missing messages.

**Interpretation:** the hypothesis is confirmed, under its stated condition. `consumer_2` never
communicated with `consumer_1` — it is a genuinely separate object — and still resumed correctly,
because the offset (not the consumer) carried the durable state across the crash.

**Limitation, demonstrated directly:** the condition "provided the offset store survives" is not
free. The notebook's Part 4 repeats the exact same crash scenario with the offset stored *only* as
an attribute on the (crashable) consumer object — `InMemoryOnlyConsumer`. Real result:

```
messages processed twice: ['event-0', 'event-1', 'event-2', 'event-3']
(4 of 10 messages were reprocessed)
```

Every message the crashed run had already processed gets reprocessed by the fresh consumer,
because there was nowhere outside the crashed process for the fresh one to learn what already
happened. This is the exact mechanism behind the "losing offset tracking on crash" failure mode
below.

## Failure modes

- **Unbounded queues cause OOM.** An unbounded `queue.Queue()` (no `maxsize`) never applies
  backpressure — `put()` always succeeds immediately. If the producer is durably faster than the
  consumer, the queue grows without limit for as long as the mismatch persists. This is not a
  hypothetical: at 1,000 events/second in vs. 500/second out, the queue grows by 500 items every
  second, and — being in-memory — eventually exhausts available RAM and the process is killed (or
  the whole machine thrashes). A bounded queue converts this into a visible, immediate producer
  stall (Part 1) instead of a silent, delayed crash.
- **Consumer lag when one partition is hot.** Part 2's real output showed partition 3 receiving 7
  of 10 events (from only 2 of 4 keys) while partition 0 received none. In a real system, if each
  partition is consumed by one worker, the worker on the hot partition falls further and further
  behind (its queue/log backlog grows) while the worker on the cold partition sits idle — the exact
  same skew failure mode `01`/`02` measured for batch shuffles and joins, now showing up as
  uneven consumer lag across partitions instead of uneven task duration. Mitigations mirror the
  batch case: choose a higher-cardinality/better-distributed partition key, or split a hot key
  across sub-partitions.
- **Losing offset tracking on crash without external persistence.** Demonstrated directly in the
  Experiment above (`InMemoryOnlyConsumer`): if the offset lives only inside the process that can
  crash, a restart cannot distinguish "already processed" from "not yet processed," and the only
  safe default is to reprocess from the beginning — producing exactly the duplicate-processing
  result measured (`event-0` through `event-3` processed twice). The fix is always the same shape:
  store the offset somewhere that outlives the consumer process — a file, a database row, or, in
  a real streaming system, the broker itself.

## Real-world usage

Every mechanism built from scratch in this topic — a bounded buffer with backpressure, a
partitioned append-only log, consumer groups with independent offsets, and durable offset
tracking across restarts — is exactly what Apache Kafka (and equivalents: AWS Kinesis, Google
Pub/Sub, Redpanda) automate and harden for production use: partitions replicated across brokers
for durability, offsets committed to a durably replicated internal topic rather than a plain
Python `dict`, consumer-group rebalancing handled automatically when a consumer joins or leaves,
and configurable backpressure/flow-control at the client level. `04-kafka` (not yet built) takes
this exact vocabulary — partitions, offsets, consumer groups, backpressure — and shows the real,
production client code for it, checking first whether a broker is available in this environment to
run it for real.

## Mental model

A stream is a batch that never finishes arriving. Everything in this topic is a way to keep
"never finishes" from meaning "eventually breaks": a bounded buffer turns an invisible overflow
into a visible, controllable slowdown (backpressure); partitioning turns "one ordered thing" into
"many independently-ordered, independently-parallel things," at the cost of only guaranteeing
order within a key; and an offset stored outside the consumer is the one piece of state that lets
a process die and a *different* process correctly pick up exactly where it left off — durability
of that one number is what "resume" actually means.

## Questions to think about

1. In Part 1's backpressure demo, what would change about the `put()` timings if `maxsize` were 1
   instead of 3? What would change if the consumer's `time.sleep(0.3)` were instead a
   *variable* delay depending on the item? Would backpressure still work correctly?
2. Part 2 used `hashlib.sha256` for partition assignment instead of Python's built-in `hash()`.
   Python's `hash()` for strings is salted per-process by default (for security reasons) — what
   would break if partition assignment used `hash(key) % P` instead of a stable hash, in a system
   where the producer and a consumer are different processes?
3. The Experiment committed the offset *after* processing each message (at-least-once: a crash
   between processing and committing means that message gets reprocessed on resume, as the
   Failure-modes duplicate-processing demo showed for the *no persistence* case — but even *with*
   persistence, a crash in that exact narrow window reprocesses one message). What would change if
   the offset were committed *before* processing instead? What failure mode does that introduce,
   and is it better or worse than reprocessing a message?
4. If a hot partition (like partition 3 in Part 2, with 7 of 10 events) is causing one consumer to
   fall behind, is increasing the total number of partitions (`P`) guaranteed to fix the skew? Under
   what condition would it not help at all?
5. A consumer group lets multiple consumers split the work of reading a partitioned log. What
   determines the *maximum* useful number of consumers in a single group reading one topic, given
   what you now know about how partitions and consumers relate?
