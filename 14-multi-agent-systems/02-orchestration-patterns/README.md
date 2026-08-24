# 02 – Orchestration Patterns

Detailed notes (manager/worker orchestration formalized as decompose → delegate → execute →
aggregate, contrasted precisely with a sequential-pipeline dependency chain and with
fan-out/fan-in's independent-worker structure, a real measured logical-round experiment, and
an honest statement of why wall-clock timing was not used and what the no-live-LLM-calls
substitution means for this topic):
[notes.md](notes.md)

Real, actually-executed, from-scratch `Orchestrator`/`Worker`/`PipelineWorker`
implementation (plain Python, reusing Topic 1's `Message`/`MessageBus` unchanged) — a
manager/worker fan-out/fan-in solution and a sequential-pipeline solution to the same
checkable toy task (sum 997 numbers, verified exactly against Python's own `sum()`), a real
measured logical-round experiment across worker counts, and concrete demonstrations of three
worker-failure scenarios (malformed type, wrong-but-well-typed value, dropped items), all
with real pasted output:
[001_orchestrator_patterns.ipynb](001_orchestrator_patterns.ipynb)

## What you'll learn

Why Topic 1's peer-to-peer negotiation doesn't scale to heterogeneous specialist workers or
tasks that need a single place enforcing overall correctness — and what a central
**orchestrator** adds instead: an explicit task-decomposition step, one-to-one delegation to
workers, independent worker execution, and result aggregation. Then, precisely: the
difference between **manager/worker (fan-out/fan-in)** orchestration, where workers are
mutually independent and dependency depth stays constant as worker count grows, and a
**sequential pipeline**, where each stage depends on the previous one's output and
dependency depth grows linearly with worker count — measured directly via an explicit
logical clock carried on every message, not asserted.

## Why it matters

Most real multi-agent systems are not negotiations between equals — they're a coordinator
splitting a task across specialists and combining their results. This topic is where that
shape gets built and stress-tested at toy scale, on top of the exact communication
primitives Topic 1 established, before any later topic tackles a harder coordination
problem (e.g. what happens when a worker itself needs to be an orchestrator over its own
sub-workers).

## Prerequisites

- `14-multi-agent-systems/01-communication-protocols` — this topic reuses its `Message` and
  `MessageBus` classes directly and assumes familiarity with the direct/broadcast/blackboard
  topology discussion.
- Comfort with plain Python classes and dataclasses — no ML framework knowledge is required.

## What you'll build

- `Orchestrator` and `Worker` classes implementing manager/worker (fan-out/fan-in)
  orchestration: task decomposition into contiguous chunks, one-to-one delegation, fully
  independent worker execution, and aggregation with an optional type-check validation gate.
- `PipelineOrchestrator` and `PipelineWorker` classes implementing the same task as a
  sequential pipeline, where each stage folds its own fixed chunk into a running accumulator
  and forwards it to the next stage — a genuine data dependency, not just an execution order.
- A shared, deterministic `split_into_chunks` decomposition used by both patterns so the
  comparison is fair, applied to a 997-number list checked exactly against Python's built-in
  `sum()`.
- A real measured logical-round experiment across `n_workers ∈ {2, 4, 8, 16, 32}`: fan-out/
  fan-in stays at a constant 2 rounds; the pipeline needs exactly `n` rounds — asserted, not
  eyeballed — plus a secondary message-count comparison showing why message count alone does
  not reveal the same structural gap.
- Three concrete, executed worker-failure demonstrations: a malformed (wrong-type) result
  caught cleanly by a type check vs. an uncaught crash without one; a wrong-but-well-typed
  result that a type check cannot catch and that silently poisons the aggregate total by a
  measured amount; and a dropped-item bug caught by a redundant item-count check that the
  previous bug slipped past.

## Where it appears in real systems

LangGraph models this as an explicit graph — a parallel "map" over branches joined back at
an aggregation node for fan-out/fan-in, or a linear chain of nodes for a pipeline. CrewAI's
hierarchical process puts a manager agent in charge of decomposing a goal, assigning tasks
to specialist crew members, and reviewing their results — the same decompose → delegate →
aggregate loop this topic's `Orchestrator` implements, but driven by real LLM calls at each
step instead of this topic's fixed, scripted logic. AutoGen's group chat manager plays the
same central-coordinator role while deciding, turn by turn, which conversable agent should
act next. None of these three are called live in this section (no API keys are available or
authorized in this environment) — they're named in notes.md's "Real-world usage" to connect
this topic's toy abstractions to real, production orchestration designs by name.

## What's next

Later `14-multi-agent-systems` topics build on both this topic's orchestration patterns and
Topic 1's communication primitives to tackle harder coordination problems — e.g. what
happens when workers themselves disagree about a result, or when a worker needs to be an
orchestrator over its own sub-workers (nested orchestration) — reusing this topic's decompose
→ delegate → aggregate vocabulary rather than inventing new plumbing per topic.
