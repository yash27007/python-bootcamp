# 14 – Multi-Agent Systems

Communication protocols → orchestration patterns → swarm coordination, first-principles,
in increasing decentralization: point-to-point/broadcast messaging, then a central
orchestrator delegating to workers, then fully decentralized coordination with no central
controller at all. Builds on `13-llms-from-scratch` (agents here are the kind of thing an
LLM could drive) — no live external LLM API calls are made anywhere in this section
(no API keys are available/authorized in this environment); agents are deterministic,
scripted Python objects standing in for what an LLM-backed agent would do, stated
explicitly rather than glossed over.

| # | Topic | Status | Description |
|---|-------|--------|--------------|
| 01 | [Agent Communication Protocols](./01-communication-protocols/) | ✅ Complete | `Message`/`MessageBus` built from scratch (direct/broadcast/blackboard), a real scripted 3-agent negotiation transcript, real measured message-count scaling, a real protocol-mismatch demo |
| 02 | [Orchestration Patterns](./02-orchestration-patterns/) | ✅ Complete | Manager/worker fan-out/fan-in vs. sequential-pipeline orchestration on top of Topic 01's message bus, a real logical-round experiment, three concrete worker-failure demonstrations |
| 03 | [Swarm Coordination (PSO)](./03-swarm-coordination/) | ✅ Complete | Particle Swarm Optimization derived and implemented from scratch in NumPy, a real convergence run, a real swarm-size experiment, a real measured premature-convergence failure |

## Prerequisites

- `13-llms-from-scratch` — the section this one builds on conceptually (agents as the kind
  of thing an LLM-backed system would drive), though no LLM is actually invoked here.
- `08-mlops-deployment/01-docker` or `02-git` (optional) — for the "systems/architecture
  topic with a documented math substitution" convention Topics 01-02 follow.
- `12-reinforcement-learning/03-policy-gradients` (useful, not required) — Topic 03
  contrasts PSO's gradient-free swarm search against it directly.

## Environment note

No live external LLM API calls are made anywhere in this section — no API keys are
available or authorized in this environment, and it would break offline reproducibility.
Topics 01-02's agents are deterministic, scripted Python logic; Topic 03's PSO agents are
plain numerical particles, not LLM-backed at all. Every notebook runs fast and offline.

## What's next

`15-agent-skills-and-mcp` continues the toy-scale, no-live-API discipline, covering Agent
Skills and the Model Context Protocol — the standardized tool-use/request-response pattern
this section's Topic 01 named but did not build.
