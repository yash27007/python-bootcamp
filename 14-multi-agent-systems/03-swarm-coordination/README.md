# 03 – Swarm Coordination (Particle Swarm Optimization)

Detailed notes (the orchestrator-as-single-point-of-failure problem stated precisely, why
large-scale search and robotics swarms need robustness a central coordinator can't give, a full
derivation of the PSO velocity/position update with every term justified, an explicit connection
back to Topic 1's broadcast topology as PSO's entire communication protocol, a real measured
swarm-size experiment, a real measured premature-convergence failure demonstration, and a
precise contrast with `12-reinforcement-learning/03-policy-gradients`'s gradient-based
optimization):
[notes.md](notes.md)

Real, actually-executed, from-scratch NumPy Particle Swarm Optimization implementation (no PSO
library) — a real convergence run on the 2D Sphere function with position-snapshot plots, a real
measured swarm-size experiment (5 vs. 20 vs. 50 particles), and a real measured
premature-convergence failure demonstration on the 2D Rastrigin function comparing a balanced
parameter setting against a social-heavy one, all with real pasted output and real plots:
[001_particle_swarm_optimization.ipynb](001_particle_swarm_optimization.ipynb)

## What you'll learn

Why a central orchestrator (Topic 2) is the wrong shape for problems that either can't be
decomposed up front (large, poorly-structured search and optimization) or can't tolerate a
single point of failure (robotics swarms, physically distributed search) — and what replaces it:
**swarm coordination**, where many simple independent agents ("particles") each pull toward
their own best-known position and the swarm's best-known position, with no agent ever
distinguished as "in charge." Then, precisely: the full derivation of **Particle Swarm
Optimization (PSO)** — the inertia, cognitive, and social terms of the velocity update, why
$r_1, r_2$ are randomized, and why PSO is a genuine multi-agent algorithm whose only
communication is each particle broadcasting its own best-known position, reusing Topic 1's
broadcast topology in its simplest possible form.

## Why it matters

Not every coordination problem has a "whole" that a manager can decompose and reassemble.
Large-scale continuous optimization (engineering design search, hyperparameter search over
non-differentiable objectives) and physically distributed multi-agent systems (robot swarms)
both need coordination that emerges from purely local rules and survives any single agent
failing — properties a central orchestrator structurally cannot provide. This topic is where the
curriculum's multi-agent track builds that alternative shape and demonstrates, with real
measured numbers, both why it works and precisely how it can fail.

## Prerequisites

- `14-multi-agent-systems/01-communication-protocols` — this topic's entire particle-to-particle
  coordination mechanism is a direct instance of this topic's broadcast topology, reused in its
  simplest possible form (one position, one scalar value, no addressing, no reply).
- `14-multi-agent-systems/02-orchestration-patterns` — this topic's "Why simpler approaches
  fail" section builds directly on that topic's orchestrator-as-single-point-of-failure argument.
- `12-reinforcement-learning/03-policy-gradients` (useful, not required) — this topic's
  "Real-world usage" section draws an explicit gradient-free vs. gradient-based contrast against
  it.
- Comfort with NumPy vectorized array operations — no ML framework knowledge is required.

## What you'll build

- A fully vectorized, from-scratch `pso()` function in plain NumPy: velocity update
  ($v_i^{t+1} = w v_i^t + c_1 r_1 (p_i^{best}-x_i^t) + c_2 r_2 (g^{best}-x_i^t)$), position
  update, per-particle personal-best tracking, and swarm-wide global-best tracking — no PSO
  library used.
- A real convergence run on the 2D Sphere function ($f(x,y)=x^2+y^2$), 25 particles, 40
  iterations: global best drops from $\approx 2.68$ to $\approx 2\times10^{-6}$, landing
  $0.0014$ from the true optimum — with a log-scale convergence-curve plot and three real
  scatter-plot position snapshots (start, mid, end).
- A real measured swarm-size experiment (5 vs. 20 vs. 50 particles, 10 seeds each) showing
  diminishing-returns convergence speed: mean iterations-to-threshold drops from 21.2 (5
  particles) to 14.0 (20) to 11.3 (50).
- A real measured premature-convergence demonstration on the 2D Rastrigin function: a
  social-heavy parameter setting ($w=0.9, c_1=0.2, c_2=3.5$) lands on a final value $6.80\times$
  worse than a balanced setting ($w=0.7, c_1=1.5, c_2=1.5$), stalling its last improvement at
  iteration 35.7 of 60 versus balanced's 56.3 — plus a genuine finding that spatial diversity
  collapse and premature convergence are related but distinct symptoms, not the same
  measurement.

## Where it appears in real systems

PSO and related swarm/evolutionary optimizers are used directly in engineering design
optimization — tuning continuous design parameters (antenna geometry, aerodynamic shapes,
structural dimensions) against expensive, non-differentiable simulated-performance objectives —
and in robotics swarm search and exploration, where multi-robot search-and-rescue and mapping
systems need coordination that survives any single robot's failure, which is exactly the
property this topic's "Why simpler approaches fail" section argues a central orchestrator can't
provide. No live external service is called anywhere in this topic — PSO particles are numerical
search agents operating on a plain mathematical objective function, not LLM-backed agents, so
there is no live-API substitution to make in the first place.

## What's next

Later `14-multi-agent-systems` topics return to agents with genuine individual behavior and
disagreement (rather than PSO's identical, purely numerical particles) — building on Topic 1's
communication primitives, Topic 2's orchestration patterns, and this topic's fully decentralized
coordination as the three basic shapes multi-agent coordination can take, before tackling harder
problems like conflict resolution and nested orchestration.
