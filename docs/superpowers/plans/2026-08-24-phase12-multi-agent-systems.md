# Phase 12: Multi-Agent Systems First-Principles Build-Out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** New `14-multi-agent-systems/` section: communication protocols → orchestration patterns → swarm coordination, first-principles, toy scale throughout. Builds on `13-llms-from-scratch` (agents in this section are the kind of thing an LLM could drive, but no live LLM API calls are made anywhere in this section — deterministic/scripted agent logic stands in, so every run is fast, offline, and reproducible; this is stated explicitly in notes.md, not glossed over).

**Architecture:** 3 topics, 1 task each, in increasing decentralization: point-to-point/broadcast messaging → a central orchestrator delegating to workers → fully decentralized swarm coordination with no central controller at all.

**Tech Stack:** Plain Python (classes, queues) for Topics 1-2 — this is a systems/software-architecture topic, not a deep-learning one, same category as `08-mlops-deployment`. NumPy for Topic 3's Particle Swarm Optimization, which has real derivable math (velocity/position update equations) and is a legitimate, textbook multi-agent swarm algorithm — chosen specifically so "swarm" content has real math and real measured convergence instead of hand-waved multi-LLM-agent chat transcripts.

**Spec:** `docs/superpowers/specs/2026-08-23-first-principles-curriculum-design.md`, `AGENTS.md`.

## Global Constraints

- **Binding: no heavy/long-running training.** Nothing in this section trains a neural network for more than a few seconds if it trains one at all; Topic 3's PSO is iterative optimization, not gradient-based training, and converges in well under a minute.
- **Binding: no live external LLM API calls.** This section does not call any hosted LLM service (no API keys are available/authorized in this environment, and it would make notebooks non-reproducible offline). Agents are implemented as deterministic Python objects (rule-based logic, small heuristics, or classical algorithms) that stand in for "what an LLM-backed agent would do" — state this substitution honestly in each topic's notes.md rather than pretending these are LLM agents.
- 12-section notes.md template. Real math where it exists (Topic 3); a documented "Conceptual foundation" substitution is fine for Topics 1-2 (systems/architecture topics, same pattern as `08-mlops-deployment`'s Docker/Git topics) — but still include what real formalism there is (message-passing semantics, delegation/aggregation logic).
- Review level: light.

---

### Task 1: Agent Communication Protocols

**Files:** Create `14-multi-agent-systems/01-communication-protocols/` (README.md, notes.md, notebook)

**Content:** Problem = a single agent (like `13-llms-from-scratch`'s TinyGPT, or any LLM-backed assistant) can only do one thing at a time in one context — some tasks are naturally split across multiple specialized agents that need to exchange information. Why-simpler-fails = just calling multiple functions in sequence in one process isn't "multi-agent" — the point of agent communication protocols is that each agent has its own state/goals and the *interface between them* has to be designed (what messages look like, who can send what to whom, sync vs async). Conceptual foundation = formalize a message as (sender, receiver, performative/intent, content) — cite the real FIPA ACL performative vocabulary (`inform`, `request`, `propose`, `accept`/`reject`) by name as the standard this pattern descends from; contrast three concrete communication topologies: direct point-to-point request-response, broadcast, and a shared blackboard (publish to a common store, others read). From-scratch = implement a minimal in-process `Message` class and `MessageBus` (plain Python, no network) supporting all three topologies; build 3 toy scripted agents (e.g. a "buyer" and two "seller" agents) that actually exchange messages to complete a real toy negotiation (buyer broadcasts a `request`, sellers `propose` prices, buyer `accept`s the best one) — run it, print/capture the real message transcript. Experiment = hypothesis about direct vs broadcast message-count overhead as agent count scales (e.g. 3 vs 6 vs 10 agents) — actually measured by counting messages sent under each topology for the same task. Failure modes = message ordering/race conditions when agents run concurrently (discuss even if this toy implementation is synchronous), a receiver that doesn't understand a performative it's sent (protocol mismatch), broadcast storms as agent count grows. Real-world usage = cite real multi-agent frameworks' message-passing designs (e.g. AutoGen's conversable agents, CrewAI's task delegation, MCP's request/response — forward-reference `15-agent-skills-and-mcp`) by name, no deep dive. Mental model, Questions.

- [ ] Write notes.md + notebook (real message-passing negotiation demo across 3 topologies, honest that agents are scripted not LLM-backed). README. `git commit -m "Phase 12 Task 1: first-principles build-out — agent communication protocols"`.

### Task 2: Orchestration Patterns

**Files:** Create `14-multi-agent-systems/02-orchestration-patterns/` (README.md, notes.md, notebook)

**Content:** Problem = Task 1's agents talked to each other directly (peer-to-peer negotiation) — many real multi-agent systems instead have a central coordinator that decomposes a task and assigns pieces to specialized workers, then combines their results. Why-simpler-fails = cite Task 1's peer-to-peer negotiation explicitly: it works for a handful of agents with a shared simple protocol, but doesn't scale to heterogeneous workers with very different capabilities, and there's no single place enforcing the overall task's correctness. Conceptual foundation = formalize the manager/worker (hierarchical) orchestration pattern: task decomposition, delegation, worker execution, result aggregation — and contrast it with the sequential-pipeline pattern (each agent's output is the next agent's input) and the fan-out/fan-in parallel pattern; state precisely when each is appropriate (this is the actual conceptual content, treat it with real rigor even though there's no closed-form derivation). From-scratch = build a real `Orchestrator` class and 3+ `Worker` agents (plain Python, reuse Task 1's `Message`/`MessageBus`) that decompose a genuinely non-trivial toy task into subtasks, delegate them, and aggregate results — pick something with a checkable correct answer, e.g. a manager splitting a list of numbers into chunks, delegating "find the sum of this chunk" to N workers in parallel (fan-out/fan-in), aggregating, and comparing against `sum()` directly as a correctness check; then also implement the SAME task as a sequential pipeline and measure the wall-clock difference — actually run both, real numbers. Experiment = hypothesis that fan-out/fan-in orchestration is faster than sequential for this decomposable task as worker count grows — actually measured (Python's GIL means "parallel" here should honestly use `multiprocessing` or `concurrent.futures` if you want a REAL wall-clock speedup, or state clearly if you're using simple sequential simulation of parallelism and reporting message-count/step-count instead of wall-clock — be honest about which). Failure modes = a worker failing or returning a wrong/malformed result and no aggregation-time validation catching it (show a concrete broken-worker example), an orchestrator becoming a single point of failure/bottleneck as task complexity grows, over-decomposition where coordination overhead exceeds the parallelism benefit. Real-world usage = cite real orchestration frameworks (LangGraph, CrewAI's hierarchical process, AutoGen's group chat manager) by name. Mental model, Questions.

- [ ] Write notes.md + notebook (real fan-out/fan-in vs sequential orchestration, real measured comparison). README. `git commit -m "Phase 12 Task 2: first-principles build-out — orchestration patterns"`.

### Task 3: Swarm Coordination — Particle Swarm Optimization

**Files:** Create `14-multi-agent-systems/03-swarm-coordination/` (README.md, notes.md, notebook)

**Content:** Problem = Task 2's orchestrator is a single point of control coordinating every worker — what happens when there's no central coordinator at all, and coordination has to emerge from purely local interactions between many simple agents? Why-simpler-fails = cite Task 2's orchestrator-as-bottleneck/single-point-of-failure explicitly — some problems (large-scale search/optimization, robotics swarms) need robustness to no single agent failing the whole system, which a central orchestrator can't give you. Mathematical foundation = derive Particle Swarm Optimization precisely: each particle (agent) has a position and velocity in the search space; the velocity update rule $v_{i}^{t+1} = w v_i^t + c_1 r_1 (p_i^{best} - x_i^t) + c_2 r_2 (g^{best} - x_i^t)$ combines inertia, the particle's own best-known position, and the swarm's best-known position — derive/explain each term's role (inertia weight $w$ for exploration vs. exploitation, cognitive term $c_1$, social term $c_2$), position update $x_i^{t+1} = x_i^t + v_i^{t+1}$. This IS genuinely a multi-agent swarm algorithm — each particle is an independent agent, coordination emerges purely from each agent broadcasting its own best-known position (a legitimate, minimal communication protocol — connect explicitly back to Task 1's broadcast topology). From-scratch = a REAL PSO implementation in NumPy (no library), actually run on a real toy optimization problem (e.g. minimizing a 2D function with a known global minimum, like the Sphere or Rastrigin function restricted to 2D for visualizability) with 20-30 particles for a modest number of iterations — show real convergence (best-known value decreasing over iterations, converging near the true minimum), plot particle positions at a few snapshots if feasible. Experiment = hypothesis about swarm size (e.g. 5 vs 20 vs 50 particles) vs. convergence speed/quality — actually measured. Failure modes = premature convergence to a local minimum when the social term dominates too early (a real concrete demo: bias $c_2 \gg c_1$ and show it converges to a worse optimum on a multi-modal function), swarm diversity collapse. Real-world usage = cite real applications (engineering design optimization, robotics swarm search) and briefly connect back to `12-reinforcement-learning`'s policy optimization as a contrast (gradient-based vs. gradient-free/swarm-based search). Mental model, Questions.

- [ ] Write notes.md + notebook (real PSO convergence, real premature-convergence failure demo). README. `git commit -m "Phase 12 Task 3: first-principles build-out — swarm coordination (particle swarm optimization)"`.

### Task 4: Section/root README

- [ ] Create `14-multi-agent-systems/README.md` (all 3 topics, ✅ Complete). Update root `README.md` (roadmap diagram, curriculum table row 🚧→✅, "What's Inside" prose). `git commit -m "Phase 12 Task 4: mark 14-multi-agent-systems complete in section and root README"`.

## Verification

```bash
cd /home/yashwanth-aravind/ml-course/python-bootcamp
.venv/bin/python -c "
import pathlib
for t in sorted(pathlib.Path('14-multi-agent-systems').iterdir()):
    if t.is_dir(): print(t.name, (t/'notes.md').exists(), (t/'README.md').exists())
"
```
