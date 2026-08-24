# Phase 5: Distributed Data First-Principles Build-Out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** New `10-distributed-data/` section teaching why distributed processing is needed at all (single-machine bottleneck) before PySpark syntax, then streaming's equivalent progression (producer/consumer → queue → backpressure → partitioning → Kafka), staying local/free throughout.

**Architecture:** 4 topics: (1) why distribute — single-machine bottleneck, partitioning, parallelism, map/filter/aggregate, shuffle, fault tolerance (conceptual, builds the mental model before touching PySpark); (2) PySpark local-mode practical (DataFrame API, real jobs on a real, if modest, dataset); (3) streaming fundamentals — producer/consumer, queue, backpressure, partitioning, consumer groups, offsets (from-scratch, since no Kafka broker is available in this environment); (4) Kafka practical — real, correct code, honestly marked unexecuted if no broker is available (verify first). Java is confirmed present (`openjdk 21.0.11`), so PySpark local mode should run for real.

**Tech Stack:** PySpark (local mode, `local[*]`), no cluster; Kafka conceptually only unless a broker is genuinely available.

**Spec:** `docs/superpowers/specs/2026-08-23-first-principles-curriculum-design.md`, `AGENTS.md`.

## Global Constraints

- Check `.venv/bin/python -c "import pyspark"` (currently fails, not installed) — `uv add pyspark`. Java is confirmed present.
- Check for a running/available Kafka broker before Task 4 (`nc -z localhost 9092` or similar) — very likely NOT available in this environment. If unavailable, Task 4's Kafka code is written/reviewed carefully but explicitly marked "not executed in this environment," same honesty discipline as Phase 3's Dockerfile.
- 12-section notes.md template throughout. Conceptual-foundation substitution allowed where genuinely no closed-form math (Task 1's distributed-systems concepts, Task 4's Kafka) — MUST be documented inline (this exact requirement has needed fix rounds twice already in this initiative, don't repeat).
- Real PySpark jobs must actually run in local mode and produce real output — never fabricate.
- Datasets: reuse existing repo datasets or a modest synthetic/public one generated locally (no new large downloads) — the "why distribute" story is about the *mechanism*, not needing genuinely big data; a few hundred thousand rows generated locally is enough to make partitioning/shuffle real without bloating the repo.
- Compare explicitly: single Python process vs `multiprocessing` vs PySpark local, per the design spec's §12 guidance — at least one real timed comparison.
- Every topic gets an orientation-format README + updated section/root README (last task's job).
- Review level: light (narrower reviewer pass or direct controller verification).
- Commit granularity: one commit per task.

---

### Task 1: Why Distribute — Single-Machine Bottleneck, Partitioning, Fault Tolerance

**Files:** Create `10-distributed-data/01-why-distributed-processing/` (README.md, notes.md, notebook)

**Content:** Problem = a dataset or computation too large/slow for one machine's memory or CPU. Why-simpler-fails = "just buy a bigger machine" (vertical scaling) hits real limits (cost, physical ceiling); a hand-rolled `multiprocessing.Pool` solves parallelism on ONE machine but not memory limits or fault tolerance. Conceptual foundation = partitioning (splitting data into independent chunks), the map/filter/aggregate primitives as embarrassingly-parallel operations, shuffle (the expensive step: redistributing data across partitions for a groupby/join — explain why it's expensive), fault tolerance (lineage/recomputation vs replication). From-scratch = a real, timed comparison: single-threaded Python loop vs `multiprocessing.Pool` vs (forward-reference) PySpark, on the same embarrassingly-parallel toy computation (e.g. a CPU-bound map over a few hundred thousand synthetic rows) — actually run all variants, real timings. Experiment = hypothesis about where the crossover point is (multiprocessing wins for CPU-bound single-machine work at this scale, doesn't solve the shuffle/memory problem PySpark's distributed model does) — stated first, then measured. Failure modes = the shuffle blowing up memory/network for a bad partition key, data skew (one partition way bigger than others), naive multiprocessing not helping I/O-bound work. Real-world, Mental model, Questions.

- [ ] Write notes.md + notebook (real timed 3-way comparison), README, commit: `git commit -m "Phase 5 Task 1: first-principles build-out — why distributed processing"`.

---

### Task 2: PySpark Local Mode — DataFrame API, Real Jobs

**Files:** Create `10-distributed-data/02-pyspark-local-mode/` (README.md, notes.md, notebook)

**Content:** `uv add pyspark`, confirm `SparkSession.builder.master("local[*]")` starts. Problem = writing distributed map/filter/aggregate/shuffle logic by hand (Task 1's from-scratch versions) doesn't scale past toy examples — need an engine that does partitioning/shuffle/fault-tolerance automatically. Why-simpler-fails = cite Task 1's hand-rolled multiprocessing version's limits. Conceptual foundation = Spark's DataFrame API and lazy execution (transformations vs actions, the execution plan) — connects directly to Task 1's map/filter/aggregate/shuffle vocabulary, now automated. From-scratch = N/A (Task 1 already built this) — cite it, bridge explicitly. Practical = REAL PySpark jobs in local mode on a real (locally generated, moderate-size — e.g. 500k-1M synthetic rows, or an existing repo dataset if a suitable one exists) dataset: `.read`, `.filter`, `.groupBy().agg()`, a `.join()` demonstrating shuffle, `.explain()` to show the execution plan — ACTUALLY RUN, real output. Experiment = compare PySpark's groupBy/join timing against Task 1's from-scratch multiprocessing version on the same data/operation, hypothesis-first. Failure modes = `.collect()`ing too much data back to the driver (defeats the purpose, OOMs), skewed joins, too many small partitions (`local[*]` doesn't hide fundamental partition-count tradeoffs). Real-world, Mental model, Questions.

- [ ] `uv add pyspark`; write notes.md + notebook (real jobs, real output); README; commit: `git commit -m "Phase 5 Task 2: first-principles build-out — PySpark local mode"`.

---

### Task 3: Streaming Fundamentals — Producer/Consumer, Queue, Backpressure, Partitioning

**Files:** Create `10-distributed-data/03-streaming-fundamentals/` (README.md, notes.md, notebook)

**Content:** Problem = Tasks 1-2 process a fixed, already-collected dataset (batch); real systems often need to process events as they arrive, indefinitely. Why-simpler-fails = "just batch-process every N minutes" adds latency and doesn't handle a producer that's faster than the consumer (unbounded queue growth = memory exhaustion). Conceptual foundation = producer/consumer decoupling via a queue, backpressure (what happens/should happen when the producer is faster), partitioning a stream for parallel consumption while preserving per-key ordering, consumer groups, offsets (a durable pointer into the log, letting a consumer resume after a crash — connects to Task 4's Phase-3-style checkpointing idea). From-scratch = a REAL Python implementation: a bounded in-memory queue (`queue.Queue` with `maxsize`) demonstrating backpressure (producer blocks when full), a toy partitioned log (a list of lists, one per partition, keyed by hash) with multiple consumer "offsets" tracking position independently — ACTUALLY RUN, showing a consumer crash-and-resume-from-offset scenario with real output. Experiment = the crash-and-resume scenario, hypothesis-first. Failure modes = unbounded queues causing OOM, consumer lag when one partition is hot (skew, same failure mode as Task 1/2's shuffle skew but for streams), losing offset tracking on crash without external persistence. Real-world = this is exactly what Kafka automates and hardens — bridge explicitly to Task 4. Mental model, Questions.

- [ ] Write notes.md + notebook (real from-scratch queue/partitioned-log/offset demo); README; commit: `git commit -m "Phase 5 Task 3: first-principles build-out — streaming fundamentals"`.

---

### Task 4: Kafka — Practical

**Files:** Create `10-distributed-data/04-kafka/` (README.md, notes.md, script or notebook)

**Content:** First, check for a real broker: try `kafka-python` or `confluent-kafka` against `localhost:9092` with a short timeout, or check if a broker process can be started locally (unlikely in this environment without Docker, which Phase 3 already established has no working daemon here). If genuinely unavailable (expected), write REAL, CORRECT producer/consumer code (topics, partitions, consumer groups, offset commits) using a real Kafka client library, reviewed carefully for correctness, but EXPLICITLY MARKED "written and reviewed, not executed in this environment — no Kafka broker available," same honesty discipline as Phase 3. Content: Problem = Task 3's from-scratch in-memory queue doesn't survive a process crash or scale across machines. Why-simpler-fails = cite Task 3's from-scratch limitations explicitly. Conceptual foundation = Kafka's log-based model (a partition is an append-only, replicated, durable log — not a traditional queue that deletes on consume), consumer groups and rebalancing, exactly-once vs at-least-once delivery semantics. Practical = the real (possibly unexecuted) producer/consumer code, `bin/kafka-topics.sh`-style topic creation commands documented. Experiment = if genuinely executable, a real produce/consume round-trip; if not, a clearly-labeled "expected behavior" walkthrough instead of a fabricated "Result." Failure modes = consumer group rebalancing storms, under-replicated partitions, offset commit before vs after processing (at-least-once vs at-most-once tradeoff). Real-world, Mental model, Questions.

- [ ] Check broker availability first (don't assume). Write notes.md + code per whichever path applies; README; commit: `git commit -m "Phase 5 Task 4: first-principles build-out — Kafka"`.

---

### Task 5: Section and Root README Finalization

- [ ] Create `10-distributed-data/README.md` (all 4 topics, ✅ Complete). Update root `README.md` (Curriculum table row 10, roadmap, section blurb) per the Phase 3/4 precedent. `git commit -m "Phase 5 Task 5: mark 10-distributed-data complete in section and root README"`.

## Verification

```bash
cd /home/yashwanth-aravind/ml-course/python-bootcamp
.venv/bin/python -c "
import pathlib
for t in sorted(pathlib.Path('10-distributed-data').iterdir()):
    if t.is_dir(): print(t.name, (t/'notes.md').exists(), (t/'README.md').exists())
"
```
