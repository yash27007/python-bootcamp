# 02 — Orchestration Patterns

## Problem

Topic 1 built the plumbing for agents to exchange messages, then used it for a
**peer-to-peer negotiation**: a buyer and two sellers, none of whom had any special
authority over the others, reached an outcome purely by talking directly to each other.
That shape works when every agent already understands the same small protocol and the
task is small enough that "let the agents work it out among themselves" is a coherent
strategy. Most real multi-agent tasks don't look like that. A research assistant that has
to gather facts from three different sources, summarize each, and produce one final
report is not a negotiation between equals — it's a task that needs to be **broken into
pieces**, those pieces need to be **handed to whichever agent is suited to do them**, and
the pieces' results need to be **combined back into one answer**. Nobody in that picture
is negotiating; someone is coordinating. This topic is about that different shape:
**orchestration** — a central component that decomposes a task, delegates the pieces, and
aggregates the results — and about the alternative shapes (sequential pipeline,
fan-out/fan-in) that also arise once a task has to be split across specialized workers.

## Intuition

Picture the same GPU-buying scenario from Topic 1, but now imagine the buyer doesn't
personally know how to evaluate whether a listing photo is legitimate, whether a stated
price is market-fair, or whether a seller's account is trustworthy — three different
skills. Instead of one buyer agent doing all three checks itself, or three peer agents
negotiating amongst themselves about who should do what, a **manager** agent takes the
listing, hands "check the photo" to a vision-checking worker, "check the price" to a
market-data worker, and "check the seller" to a reputation-checking worker — three
*specialists*, none of whom needs to know the other two exist — then combines their three
verdicts into one buy/don't-buy decision. That's the whole shape of manager/worker
orchestration: **one component decides how to split the work and how to combine the
answers; the workers just do their one job and report back.**

This topic's actual toy task strips the domain-specific skills away and keeps the
*structure*: sum a list of 997 numbers by splitting it into chunks, handing each chunk to
a worker, and adding the workers' partial sums back together — checkable exactly against
Python's own `sum()`.

## Why simpler approaches fail

Topic 1's peer-to-peer negotiation is a real, working pattern — but it depends on
assumptions that manager/worker orchestration does not need to make:

1. **It assumes a small number of roughly-equal, homogeneous agents that already share a
   protocol.** Topic 1's buyer and sellers all understood the same four performatives
   (`request`/`propose`/`accept`/`reject`) and all played symmetric roles (any seller
   could have played any other seller's part). A task with genuinely *heterogeneous*
   workers — one that fetches data, one that summarizes text, one that checks facts, each
   with a completely different interface — has no natural "everyone speaks the same
   protocol" story, because there is no shared thing for them to negotiate *about*; they
   are not competing for the same outcome, they are contributing different pieces of one.
2. **It doesn't scale past a handful of agents.** Topic 1's buyer personally collected
   every `propose` message and personally ran `min()` over them. That is fine for two or
   three sellers; it does not generalize to "decompose this task into 50 heterogeneous
   subtasks," because nothing in the peer-to-peer picture designates who is responsible
   for the decomposition step in the first place — decomposition was implicit in how the
   buyer's code happened to be written, not a role any agent was explicitly assigned.
3. **There is no single place enforcing the overall task's correctness.** In a
   negotiation, "correctness" is just "the buyer picked the cheapest offer it received" —
   trivially checkable from the buyer's own state. In a decomposed task (sum a list,
   summarize three articles, answer a multi-part question) correctness means "the pieces
   were combined *correctly*," which requires some component to know what the whole task
   was in the first place and to validate the combination against it. No peer in a
   negotiation has that global view by design — that is exactly the role this topic gives
   the orchestrator, and exactly the role this topic's "Failure modes" section shows
   breaking down when the orchestrator's own aggregation logic is wrong or unvalidated.

## Conceptual foundation

*(Systems/architecture topic, same documented substitution for "Mathematical foundation"
that Topic 1's notes.md uses — see its "Conceptual foundation" section for the general
justification.)*

### Manager/worker (hierarchical) orchestration, formalized

A manager/worker system consists of one **orchestrator** $O$ and a set of **workers**
$W = \{w_1, \ldots, w_n\}$, and proceeds in four explicit steps:

1. **Task decomposition.** $O$ takes one task $T$ and produces $n$ subtasks
   $T_1, \ldots, T_n$ such that $T = \text{combine}(T_1, \ldots, T_n)$ for some known
   combining function. In this topic's implementation, $T$ is a list of numbers and
   `split_into_chunks` produces $n$ contiguous sublists whose concatenation is $T$.
2. **Delegation.** $O$ sends each $T_i$ to exactly one worker $w_i$ via a direct message
   (`assign`, reusing Topic 1's direct topology). Delegation is **1-to-1 per subtask**: no
   worker receives another worker's subtask, and no worker needs to know $n$, or any
   other worker's identity, to do its job.
3. **Worker execution.** Each $w_i$ computes $R_i = f(T_i)$ for some function $f$ that is
   the *same* for every worker in a homogeneous pool (as in this topic — every `Worker`
   runs `sum(chunk)`) or *different* per worker in a heterogeneous pool (as in the
   GPU-listing example above — one worker checks photos, another checks prices). Crucially,
   $w_i$'s computation of $R_i$ does not depend on any $R_j$ for $j \neq i$ — this
   independence is what "Experiment" below measures directly.
4. **Result aggregation.** $O$ collects every $R_i$ and computes
   $R = \text{aggregate}(R_1, \ldots, R_n)$ — in this topic, `aggregate()` computes
   $\sum_i R_i$, and correctness means $R = \text{sum}(T)$ exactly.

### Contrast: sequential pipeline

A pipeline has the same $n$ subtasks $T_1, \ldots, T_n$, but instead of independent
execution, worker $i$'s computation *consumes* worker $i-1$'s output:

$$
R_0 = \text{init}, \qquad R_i = f(T_i, R_{i-1}) \text{ for } i = 1, \ldots, n, \qquad R = R_n
$$

There is no separate "aggregation" step — the last worker's output *is* the final
answer, because each stage already folded the previous stage's result into its own.
The defining structural difference from manager/worker is that $w_i$ **cannot start**
until $w_{i-1}$ has produced $R_{i-1}$ — a genuine data dependency, not just a scheduling
choice. This topic's `PipelineWorker` implements exactly this: each worker adds its own
fixed chunk's sum to a running `accumulator` field carried inside the message and forwards
it to the next worker in the chain.

### Contrast: fan-out/fan-in (parallel)

Fan-out/fan-in is the *parallel-execution reading* of manager/worker: the orchestrator
"fans out" $T_1, \ldots, T_n$ to $n$ workers that execute with no dependency between them
(exactly step 3 above), then "fans in" by collecting all $R_i$ before aggregating. The
manager/worker pattern in this topic's `Orchestrator`/`Worker` classes *is* a fan-out/fan-in
pattern — they are the same structure described from two angles: "who has authority to
decompose and combine" (manager/worker) and "what does the dependency graph between
workers look like" (fan-out/fan-in, contrasted with a pipeline's dependency chain).

### When each pattern is appropriate

| Pattern | Worker dependency | Best fit | Costs |
|---|---|---|---|
| Manager/worker (fan-out/fan-in) | none — workers are mutually independent | task cleanly splits into independent pieces (map-style); heterogeneous specialist workers; result order doesn't matter | orchestrator is a single point of failure/bottleneck; per-worker coordination overhead that doesn't shrink as workers grow (see "Failure modes") |
| Sequential pipeline | each stage needs the previous stage's output | task is inherently stateful/order-dependent (e.g. draft → critique → revise, or a running total that must reflect a specific processing order) | strictly can't be sped up by adding more workers — depth grows with $n$ regardless of available parallelism, as "Experiment" measures directly |

The deciding question is not "how many workers do I have" but **"does subtask $i$'s
correct execution require subtask $i-1$'s output?"** If yes, a pipeline is the honest
model of the task and forcing it into fan-out/fan-in either produces a wrong answer or
secretly reintroduces the pipeline's ordering as an out-of-band dependency. If no,
fan-out/fan-in exposes real independence a pipeline would hide.

## Algorithm

**Manager/worker (fan-out/fan-in), as implemented:**

1. `Orchestrator.decompose_and_delegate(numbers, workers)` splits `numbers` into
   `len(workers)` contiguous chunks (`split_into_chunks`) and sends each worker one direct
   `assign` message carrying only its chunk, at logical `round=0`.
2. Each `Worker.receive()` computes `sum(chunk)` and replies directly to the orchestrator
   with an `inform` message carrying `partial_sum`, `n_items`, and `round = 1` (one more
   than the round it received).
3. `Orchestrator.aggregate()` sums every collected `partial_sum`, optionally type-checking
   each one first, and records the aggregation's own logical round as
   `1 + max(round of every reply)`.

**Sequential pipeline, as implemented:**

1. `PipelineOrchestrator.run()` builds one `PipelineWorker` per chunk, each holding its own
   fixed chunk and the name of the *next* agent in the chain (the next worker, or the
   orchestrator for the last one).
2. It sends one `accumulate` message to the first worker with `accumulator=0`, `round=0`.
3. Each `PipelineWorker.receive()` adds its own chunk's sum to the incoming `accumulator`
   and forwards the updated value to `next_name` at `round = incoming.round + 1`.
4. The last worker's forward lands on the `PipelineOrchestrator` itself, which records the
   final `accumulator` as the answer and the message's `round` as the total logical depth.

## From-scratch implementation

Implemented in `001_orchestrator_patterns.ipynb`, actually executed, real output. Reuses
Topic 1's `Message` and `MessageBus` classes **unchanged** (copied with an explicit citation
comment, per this task's instruction to reuse-by-copy) — see
`01-communication-protocols/001_message_bus_negotiation.ipynb` for their original
definition and derivation.

- **`Worker`** / **`Orchestrator`** — the manager/worker pattern above.
- **`PipelineWorker`** / **`PipelineOrchestrator`** — the sequential-pipeline contrast.
- **`split_into_chunks`** — one shared, deterministic contiguous-chunking function used by
  both patterns so the comparison is fair (same task, same chunk boundaries).
- The task: `NUMBERS = list(range(1, 998))` — 997 numbers, deliberately not a round count,
  true sum `497503`.

As stated at the top of this section's notebook and required by this section's binding
constraints: **every agent here is a deterministic, scripted Python object — no live LLM
API call is made anywhere.** `Worker.receive()` runs plain `sum()`; nothing samples, calls
an external model, or varies run to run.

**Real captured output, 4 workers, manager/worker pattern:**

```
[00] orchestrator -> worker_0     assign     {'chunk': [1..250], 'round': 0}
[01]     worker_0 -> orchestrator inform     {'partial_sum': 31375, 'n_items': 250, 'round': 1}
[02] orchestrator -> worker_1     assign     {'chunk': [251..499], 'round': 0}
[03]     worker_1 -> orchestrator inform     {'partial_sum': 93375, 'n_items': 249, 'round': 1}
[04] orchestrator -> worker_2     assign     {'chunk': [500..748], 'round': 0}
[05]     worker_2 -> orchestrator inform     {'partial_sum': 155376, 'n_items': 249, 'round': 1}
[06] orchestrator -> worker_3     assign     {'chunk': [749..997], 'round': 0}
[07]     worker_3 -> orchestrator inform     {'partial_sum': 217377, 'n_items': 249, 'round': 1}

orchestrator aggregate: 497503
python sum() ground truth: 497503
aggregate happened at logical round: 2
CORRECT: orchestrator total matches sum() exactly
```

**Real captured output, 4 workers, sequential-pipeline pattern:**

```
[08] orchestrator -> pworker_0    accumulate {'accumulator': 0, 'round': 0}
[09]    pworker_0 -> pworker_1    accumulate {'accumulator': 31375, 'round': 1}
[10]    pworker_1 -> pworker_2    accumulate {'accumulator': 124750, 'round': 2}
[11]    pworker_2 -> pworker_3    accumulate {'accumulator': 280126, 'round': 3}
[12]    pworker_3 -> orchestrator accumulate {'accumulator': 497503, 'round': 4}

pipeline final result: 497503
python sum() ground truth: 497503
final result arrived at logical round: 4
CORRECT: pipeline total matches sum() exactly
```

Both patterns produce the exact same, independently-verifiable correct total — they differ
only in *how* they get there, which is exactly what "Experiment" measures.

## Practical implementation

As in Topic 1, this section has **no separate practical/library step** distinct from the
from-scratch implementation — the real practical frameworks in this space (LangGraph,
CrewAI, AutoGen) are all designed around orchestrating real LLM calls, which this section
is barred from making. See "Real-world usage" below for how their designs map back onto
this notebook's `Orchestrator`/`Worker`/`PipelineWorker` abstractions by name, without
running them — the same substitution Topic 1's notes.md documents.

## Experiment

**Hypothesis.** For the same decomposable sum task, fan-out/fan-in orchestration reaches
its final answer in a number of logical rounds that stays **constant** as worker count $n$
grows, while a sequential pipeline over the same task needs a number of logical rounds that
grows **linearly** with $n$ — because fan-out/fan-in workers are mutually independent
(Conceptual foundation, above) while pipeline stages are each dependent on the previous
stage's output.

**Honesty about what "logical round" means, and why wall-clock time was not used.** This
bus is synchronous, single-threaded, and single-process (unchanged from Topic 1) — every
`bus.send()` blocks until the receiver's `receive()` returns. A naive
`time.perf_counter()` comparison would measure Python function-call overhead and chunk
size, not the structural difference between the two patterns. Wrapping workers in
`threading` would not fix this either: CPython's GIL serializes pure-Python `sum()` work
regardless of thread count, so a thread-based run would show no real speedup for this
CPU-bound task; a `multiprocessing`/`ProcessPoolExecutor` version would mostly measure
process-spawn overhead for a task this small. Instead, every message in both patterns
already carries an explicit **logical clock** — a `round` field that a worker computes as
`1 + (the round of the message that caused this one)`. A round only advances when a
message's content causally depends on a previous message's content. This measures "how
many sequential dependency steps would this pattern need if every independent worker
actually executed at the same time" — the real structural claim fan-out/fan-in makes —
without fabricating a wall-clock number this single-process notebook cannot honestly
produce.

**Setup.** `logical_rounds_fan_out(n)` and `logical_rounds_pipeline(n)` each run the full
task (997 numbers) with $n$ workers, assert the result equals `sum(NUMBERS)`, and return
the logical round at which the final answer was produced. Tested at
$n \in \{2, 4, 8, 16, 32\}$.

**Actual measured result:**

| n (workers) | fan-out/fan-in rounds | pipeline rounds |
|---:|---:|---:|
| 2  | 2 | 2  |
| 4  | 2 | 4  |
| 8  | 2 | 8  |
| 16 | 2 | 16 |
| 32 | 2 | 32 |

Both `fan_out_rounds == 2` and `pipeline_rounds == n` held exactly for every tested $n$
(asserted in the notebook, not eyeballed).

A secondary, weaker metric — raw message count — was also measured for completeness:
fan-out/fan-in sends $2n$ messages ($n$ assigns + $n$ informs); the pipeline sends $n+1$
(one `accumulate` hop per stage, including the kickoff). Both counts grow **linearly** in
$n$ — message count alone does *not* show the O(1)-vs-O(n) structural gap that logical
rounds do; it only shows fan-out/fan-in has a larger constant factor for the same worker
count. This mirrors Topic 1's own experiment, where message count and "real" structural
cost (deliveries) were shown to be two different things worth measuring separately.

**Interpretation.** The hypothesis is confirmed by exact, asserted measurement: as worker
count grows, fan-out/fan-in's dependency depth stays flat while the pipeline's grows one
step per additional worker. This is the honest, structural version of "fan-out/fan-in is
more efficient as worker count grows" for this task — it is a real property of the
dependency graph, not of wall-clock time in this particular execution environment. Whether
that structural advantage becomes an actual wall-clock speedup depends entirely on whether
the deployment environment can really run $n$ workers concurrently (real threads/processes/
machines) — a claim this notebook does not make because it cannot honestly measure it here.

**Limitations.** This measures logical dependency depth only, not latency, CPU time, or
memory. It assumes a fixed decomposition shape (contiguous chunks, one round of delegation,
one round of collection) and a homogeneous, always-correctly-responding worker pool — see
"Failure modes" for what breaks when a worker does not behave as assumed. It also does not
model real per-worker overhead (process startup, network latency, LLM API round-trip time
in a real deployment) — see "Failure modes" → over-decomposition for how that overhead is
measured indirectly through message count instead.

## Failure modes

**1. A worker returns a malformed (wrong-type) result.** `Orchestrator.aggregate()` has an
optional type check. With it off, a single `MalformedWorker` that replies with a string
instead of a number crashes `sum()` deep inside Python's own summation:
`crashed with TypeError: unsupported operand type(s) for +: 'int' and 'str'` — a real
failure, but a confusing one that does not point at which worker caused it. With the type
check on, the same scenario fails cleanly at aggregation time:
`worker returned a non-numeric partial_sum: 'sorry, I could not compute that'` — a much
more diagnosable failure, and a fair model of an LLM-backed worker that returned prose
instead of a number.

**2. A worker returns a wrong-but-well-typed result — silently poisons the total.** A type
check cannot catch a value that is the *right type* and simply *wrong*. `OffByBugWorker`
always reports `sum(chunk) - 100` — a plausible bug shape (an off-by-a-constant error, or an
LLM-backed worker that miscounted). `validate=True` raised no error because the value
genuinely is an `int` — only the number itself is wrong. Concretely:

```
orchestrator total (one buggy worker): 497403
python sum() ground truth:             497503
silently wrong by: 100
type validation raised no error: True
```

This is the dangerous case highlighted honestly rather than glossed over: no validation
this orchestrator runs catches it. Catching it requires a *redundant* check against
something the buggy worker's output cannot corrupt on its own.

**3. Not every redundant check catches every bug.** The orchestrator already tracks
`n_items_seen` (total items reported summed across all workers) as a candidate redundant
check. A `HonestDroppingWorker` that silently drops the last element of its chunk *does*
get caught by it — measured: `n_items_seen: 996` vs `len(NUMBERS): 997`, so the check
correctly fails. But this same check would have passed right over `OffByBugWorker` above
(it reports the correct item count, just the wrong sum) — no single sanity check catches
every class of wrong-worker failure; what gets caught depends entirely on which invariant a
given bug happens to violate.

**4. Orchestrator as single point of failure / bottleneck.** Every pattern in this notebook
routes all delegation and all aggregation through one `Orchestrator` object. If its own
logic is wrong (exactly what failure modes 1–3 demonstrate) or it becomes unavailable, the
entire task stalls or silently corrupts — no worker has any way to detect or route around a
broken orchestrator, because no worker talks to any other worker (fan-out/fan-in) or to
anyone but its immediate chain neighbor (pipeline). Centralizing coordination makes
reasoning about the system easy (exactly one place aggregation logic lives) at the direct
cost of exactly one place where everything can break — the same tradeoff Topic 1 identified
for the blackboard topology, but here applied to a single always-involved coordinator
instead of an optional shared store.

**5. Over-decomposition: coordination overhead exceeds the parallelism benefit.** The
logical-rounds experiment showed fan-out/fan-in staying at a constant 2 rounds regardless
of `n_workers` — but that metric is *causal dependency depth*, and it does not capture
per-worker overhead (constructing a chunk, building and dispatching a `Message`,
registering a worker). The message-count experiment measured the cost that *does* scale:
$2n$ messages for fan-out/fan-in, growing without bound as $n$ grows. If the 997-number
task were split across, say, 997 workers (one number per worker), there would be 1,994
coordination messages moving a grand total of one number each — the real parallelism
benefit (rounds staying flat) is unchanged, but the coordination cost this notebook can
actually count has grown to dominate the tiny amount of real work being delegated. This is
the measured, not merely asserted, version of "orchestration overhead exceeds the benefit
of decomposition" once the pieces get too small.

## Real-world usage

- **LangGraph** (LangChain) models multi-agent workflows as an explicit graph of nodes and
  edges, where a node can be an LLM call, a tool call, or a sub-graph — this notebook's
  manager/worker pattern (`decompose_and_delegate` → parallel `Worker` nodes → `aggregate`)
  maps directly onto a LangGraph "map" over parallel branches joined back into one
  aggregation node; the sequential pipeline maps onto a simple linear chain of nodes.
- **CrewAI's hierarchical process** puts a dedicated manager agent in charge of breaking a
  goal into tasks, assigning each task to the crew member best suited for it, and reviewing
  results before finishing — structurally the same decompose → delegate → aggregate loop as
  this notebook's `Orchestrator`, except CrewAI's manager and workers are real LLM calls
  making those decisions dynamically, rather than this notebook's fixed
  `split_into_chunks`/`sum()`.
- **AutoGen's group chat manager** coordinates multiple conversable agents in a shared
  conversation, deciding at each turn which agent should speak next — closer to Topic 1's
  broadcast topology plus a scheduling policy than to direct manager/worker delegation, but
  it plays the same central-coordinator role this notebook's `Orchestrator` plays.

None of these three frameworks are called, imported, or executed anywhere in this section —
named only to connect this notebook's toy abstractions to real, production orchestration
designs by name, the same substitution Topic 1's notes.md documents and this section's
binding constraints require.

## Mental model

Peer-to-peer negotiation (Topic 1) is *"we all talk it out"* — fine when everyone shares a
protocol and no one needs a global view of correctness. Orchestration is *"one component
decides how to split the work, hands out the pieces, and is responsible for putting them
back together correctly."* Within orchestration, the only question that matters for
choosing a pattern is: **does piece $i$ need piece $i-1$'s answer to even start?** No →
fan-out/fan-in, and the dependency depth stays flat no matter how many workers you add.
Yes → a pipeline, and the depth grows one step per stage no matter how many workers you
add — more workers cannot shorten a chain that is inherently sequential.

## Questions to think about

1. This notebook's `split_into_chunks` always produces contiguous chunks of roughly equal
   size. Design a decomposition where some workers get much larger chunks than others (a
   deliberately unbalanced split). What happens to the "logical rounds" metric in that
   case, and what does that reveal about a limitation of using dependency depth alone as
   an efficiency metric?
2. `OffByBugWorker`'s bug (subtracting a constant) was chosen specifically because it
   passes both the type check and the `n_items_seen` check. Design one more redundant
   check the `Orchestrator` could run that *would* catch this specific bug, without simply
   recomputing `sum(NUMBERS)` directly (which defeats the point of delegating the work at
   all). What does your check assume about the task that a fully general orchestrator
   could not assume?
3. The pipeline pattern's logical-round count grows linearly with worker count, which
   sounds like a pure downside — but the experiment fixed the *chunk-per-worker* task
   structure. Describe a real task where a pipeline's ordering (stage $i$ depending on
   stage $i-1$'s actual output, not just running after it) is not just tolerable but
   *required for correctness*, such that forcing it into fan-out/fan-in would produce a
   wrong answer, not just a slower one.
4. Failure mode 4 (orchestrator as single point of failure) was stated but not
   demonstrated with running code, unlike failure modes 1–3. Sketch, in words or
   pseudocode, how you would demonstrate it concretely using this notebook's own
   `MessageBus` (hint: what specific method on `Orchestrator` or `MessageBus` would need
   to raise an exception, and what would happen to every worker's already-sent `inform`
   reply if it did?).
5. AutoGen's group chat manager, CrewAI's hierarchical process, and LangGraph's parallel
   map/join were all named as running over real LLM calls in production. If a worker in
   one of those real systems is itself another orchestrator (a sub-orchestrator managing
   its own sub-workers), how does the "logical rounds" metric from this notebook compose
   across that nesting — is the round count of the outer orchestrator's fan-out step just
   "however many rounds the slowest inner orchestrator needs," or does something else have
   to be tracked?
