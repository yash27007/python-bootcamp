# 01 – Agent Communication Protocols

Detailed notes (a message formalized as `(sender, receiver, performative, content)`, the
real FIPA-ACL performative vocabulary this pattern descends from, three communication
topologies contrasted precisely — direct, broadcast, blackboard — and an honest, explicit
statement of the no-live-LLM-calls substitution this whole section makes):
[notes.md](notes.md)

Real, actually-executed, from-scratch `Message`/`MessageBus` implementation (plain Python,
no network, no LLM calls) — three scripted toy agents (one buyer, two sellers) that
actually negotiate a price over the bus, a real captured message transcript, a real
measured experiment on message-count scaling under direct vs. broadcast topology, and a
concrete demonstration of a protocol-mismatch failure, all with real pasted output:
[001_message_bus_negotiation.ipynb](001_message_bus_negotiation.ipynb)

## What you'll learn

Why "call function A, then call function B" is not multi-agent communication, and what a
deliberately designed communication protocol adds that a sequential function-call chain
quietly loses: separate persistent state per agent, an enforced message-shape contract
instead of ad hoc argument passing, and an explicit (rather than accidental)
synchronization model. Then: the real FIPA-ACL performative vocabulary (`inform`,
`request`, `propose`, `accept`/`reject`) as the standard this topic's message format
descends from, and three concrete communication topologies — direct point-to-point,
broadcast, and shared blackboard — built from scratch and actually run.

## Why it matters

Every later topic in `14-multi-agent-systems` assumes agents can exchange information
correctly; this topic is where that mechanism is built and stress-tested first, before any
topic tries to compose multiple agents around a harder task. It is also the first topic in
the curriculum to work with genuinely *multiple*, independently-stateful agents at once —
`13-llms-from-scratch` built and trained a single model; this topic is about what changes
architecturally once there is more than one.

## Prerequisites

- Comfort with plain Python classes and dataclasses — no ML framework knowledge is
  required for this topic.
- `08-mlops-deployment/01-docker` or `08-mlops-deployment/02-git` (optional but useful) —
  for the shape of a "systems/architecture topic with a documented math substitution," the
  same convention this topic's notes.md follows for its "Conceptual foundation" section.

## What you'll build

- A minimal `Message` dataclass: `(sender, receiver, performative, content)`, plus an
  auto-incrementing id for transcript ordering.
- A plain-Python `MessageBus` (no network, no threads) supporting all three topologies:
  `send()` (direct), `broadcast()`, and `publish()`/`read_blackboard()` (blackboard).
- Three **scripted, non-LLM** toy agents — `BuyerAgent` and two `SellerAgent`s — that
  actually exchange messages to complete a real toy negotiation: a real captured
  transcript of `request` → `propose` (x2) → `accept`/`reject`, with the buyer correctly
  identifying and accepting the cheaper of two real offers purely by reading incoming
  messages.
- A real measured experiment: message-count overhead at $n \in \{3, 6, 10\}$ sellers under
  direct topology ($3n$ log entries — 9, 18, 30, exactly as predicted) vs. broadcast
  topology ($2n+1$ — 7, 13, 21, exactly as predicted), with the important nuance that
  actual message *deliveries* are $3n$ under both.
- A concrete, executed demonstration of a protocol mismatch: a `cancel` performative sent
  to a `SellerAgent` whose `receive()` has no branch for it — silently ignored, no
  exception, no state change, no warning.

## Where it appears in real systems

Real multi-agent LLM frameworks solve exactly this problem at production scale: AutoGen's
conversable agents exchange chat-style messages through a conversation loop; CrewAI
coordinates specialist agents through explicit task delegation; MCP (Model Context
Protocol — covered in depth by the not-yet-built `15-agent-skills-and-mcp`) standardizes a
request/response pattern between a model client and a tool server. None of these are
called live in this section (no API keys are available or authorized in this
environment) — they're named in notes.md's "Real-world usage" to connect this topic's toy
`Message`/`MessageBus` abstractions to real, production message-passing designs by name.

## What's next

Later `14-multi-agent-systems` topics build on this topic's `Message`/`MessageBus`
primitives to tackle harder coordination problems — e.g. task decomposition across
specialist agents, and conflict resolution when agents' proposals disagree — using the
same three topologies established here rather than inventing new plumbing per topic.
