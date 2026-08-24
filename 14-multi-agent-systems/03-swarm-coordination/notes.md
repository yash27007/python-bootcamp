# 03 — Swarm Coordination (Particle Swarm Optimization)

## Problem

Topic 2's orchestrator sits in the middle of everything: it decomposes the task, decides who
gets which piece, and combines the results. That single component is also the whole system's
single point of control — every worker's output flows back through it, and nothing in the
system happens without it deciding it should. That works well when the task genuinely has one
"whole" that needs to be assembled from parts (Topic 2's 997-number sum), and when it's
acceptable for the system to depend on one component staying alive and correct. But some
problems don't have that shape at all. Searching a huge, poorly-understood space for a good
solution — tuning dozens of continuous engineering parameters at once, for instance — has no
natural "decompose the search into independent chunks and hand them out" structure, because no
one knows in advance which region of the space is worth searching until agents have already
looked there. And some systems — a fleet of search-and-rescue robots exploring a collapsed
building, sensors mapping a hazardous area — cannot afford to depend on one coordinator at all,
because a coordinator that goes down (loses power, loses signal, gets physically destroyed)
takes the whole system down with it. **This topic asks: can a group of many simple agents solve
a hard search problem with no central coordinator whatsoever — where coordination emerges
purely from local interactions, and the system keeps working even if any single agent drops
out?**

## Intuition

Picture a flock of birds looking for the best feeding ground in a valley, with no leader bird
directing the others. Each bird doesn't know where the best spot is, but it does know two
things: the best spot *it personally* has found so far, and — by watching the flock — roughly
where the best spot *anyone* in the flock has found so far seems to be. Each bird nudges its own
flight path a little toward both of those pulls, while still carrying some of its own momentum
and some randomness in exactly how hard it nudges. No bird is in charge. No bird has the full
picture. Yet the flock as a whole reliably drifts toward good feeding grounds, because every
bird is quietly pooling its own tiny bit of information with everyone else's.

**Particle Swarm Optimization (PSO)** is exactly this idea turned into an optimization
algorithm: a population ("swarm") of independent search agents ("particles"), each with a
position (a candidate solution) and a velocity, each remembering its own best find and paying
attention to the swarm's best find, with no particle ever directing another.

## Why simpler approaches fail

Topic 2 named this precisely, and it applies directly here: a manager/worker orchestrator gives
you exactly one place enforcing correctness and coordinating the whole system — and exactly one
place whose failure takes the whole system down. Topic 2's own words on why that matters:

> There is no single place enforcing the overall task's correctness... [the orchestrator]
> requires some component to know what the whole task was in the first place.

That centralization is a *feature* for a task like "sum this list correctly," where the
orchestrator's global view is exactly what makes correctness checkable. It becomes a *liability*
for two different kinds of problems this topic is aimed at:

1. **Large-scale, poorly-structured search and optimization.** There is no way to decompose "find
   the global minimum of an unknown, possibly non-convex, high-dimensional function" into
   independent chunks up front, the way Topic 2 could cleanly split a list of numbers into
   contiguous ranges. Nobody knows which region of the search space is worth exploring until
   agents have actually explored it — the decomposition itself would have to be adaptive,
   informed by results as they arrive, which is precisely what a swarm does implicitly and a
   fixed up-front decomposition cannot.
2. **Robotics swarms and physically distributed search.** A team of robots searching a
   disaster site, or drones mapping a wildfire's perimeter, cannot route every decision through
   one central coordinator: the coordinator may lose connectivity, run out of power, or be
   physically destroyed, and Topic 2's structure gives no answer for what happens to the other
   workers when that one component fails — there is no "worker" role left standing in that
   design once the orchestrator is gone. These systems need **robustness to any single agent's
   failure**, which requires a system with no single point that everything else depends on.

Both problems need the same fix: replace "one component decides and combines" with
"coordination that emerges from purely local rules, run by every agent identically, with no
agent playing a distinguished role." PSO is the concrete instance of that fix this topic builds.

## Mathematical foundation

### Setup

The search space is $\mathbb{R}^d$ (a $d$-dimensional continuous space), and the goal is to find
$x^* = \arg\min_x f(x)$ for some objective function $f: \mathbb{R}^d \to \mathbb{R}$ — no
assumption is made that $f$ is differentiable, convex, or even continuous; PSO only ever needs
to *evaluate* $f$ at points, never differentiate it.

A swarm consists of $n$ **particles**, indexed $i = 1, \ldots, n$. Particle $i$ at iteration $t$
has:

- a **position** $x_i^t \in \mathbb{R}^d$ — a candidate solution,
- a **velocity** $v_i^t \in \mathbb{R}^d$ — the direction and speed the particle is currently
  moving through the search space,
- a **personal best** $p_i^{best}$ — the best (lowest-$f$) position particle $i$ has visited
  across all iterations so far, and the value $f(p_i^{best})$,
- access to the swarm's **global best** $g^{best}$ — the best position *any* particle in the
  swarm has visited so far, and $f(g^{best})$.

### Velocity update — derived term by term

The core update rule is:

$$
v_i^{t+1} = \underbrace{w\, v_i^t}_{\text{inertia}}
  + \underbrace{c_1 r_1 (p_i^{best} - x_i^t)}_{\text{cognitive term}}
  + \underbrace{c_2 r_2 (g^{best} - x_i^t)}_{\text{social term}}
$$

Each term is a separate pull on the particle's next velocity, and each has a distinct
justification:

**Inertia term, $w\, v_i^t$.** Without it, a particle's velocity would be recomputed from
scratch every iteration, purely as a function of where the bests currently are — the particle
would have no "momentum" carrying it through a region it was already moving through. The
inertia weight $w \in [0, 1]$ (typically close to $1$ early and decayed, or held at a fixed
moderate value like $0.7$–$0.9$ for a toy run) controls how much of the previous velocity
carries forward. A high $w$ favors **exploration**: the particle keeps moving broadly through
the space, resisting being immediately snapped toward a best position. A low $w$ favors
**exploitation**: the particle's velocity is dominated each step by the pulls below, so it
converges quickly onto wherever those pulls point. This is the same exploration/exploitation
trade-off named explicitly elsewhere in this curriculum's reinforcement-learning topics
(`12-reinforcement-learning`), applied here to search instead of sequential decision-making.

**Cognitive term, $c_1 r_1 (p_i^{best} - x_i^t)$.** The vector $p_i^{best} - x_i^t$ points from
the particle's current position toward the best position *it personally* has ever found — scaled
by the cognitive coefficient $c_1$ (how strongly the particle trusts its own history) and by
$r_1$, a fresh random number drawn independently every iteration for every particle (and, in the
vectorized implementation below, every dimension) uniformly from $[0, 1]$. This term encodes
individual memory: even with no communication from the rest of the swarm at all, a lone particle
with only this term would perform a noisy local search around the best point it has personally
discovered — analogous to a hill-climber that remembers its best step.

**Social term, $c_2 r_2 (g^{best} - x_i^t)$.** The vector $g^{best} - x_i^t$ points from the
particle's current position toward the best position *the whole swarm* has found — scaled by the
social coefficient $c_2$ and an independent random draw $r_2 \sim U[0,1]$. This is the only term
that lets information cross between particles at all: it is what turns $n$ independent
hill-climbers into a coordinated swarm. Crucially, this term does not require any particle to
know *which other particle* found $g^{best}$, or how, or why — it only needs to know the value
$g^{best}$ itself.

**Why $r_1, r_2$ are randomized.** If $r_1$ and $r_2$ were fixed constants, every particle's
velocity update would be a deterministic function of $(x_i^t, v_i^t, p_i^{best}, g^{best})$, and
the swarm would converge onto whatever direction those two pulls average to with no ability to
explore off that exact line. Drawing $r_1, r_2$ fresh each iteration injects the stochastic
exploration that lets different particles investigate different points along (and around) the
cognitive/social pull directions, rather than collapsing onto one predictable trajectory.

### Position update

$$
x_i^{t+1} = x_i^t + v_i^{t+1}
$$

A plain Euler integration step: the particle simply moves by its newly computed velocity. After
moving, $f(x_i^{t+1})$ is evaluated; if it improves on $f(p_i^{best})$, $p_i^{best}$ is updated
to $x_i^{t+1}$; if it also improves on $f(g^{best})$ across the whole swarm, $g^{best}$ is
updated too. That's the entire algorithm — repeated for a fixed number of iterations or until
some convergence criterion is met.

### This is a genuine multi-agent swarm algorithm

It is tempting to read PSO as "just a numerical optimizer with a colorful name," but it fits
this section's multi-agent framing exactly, not just by analogy:

- **Each particle is an independent agent** with its own persistent state ($x_i$, $v_i$,
  $p_i^{best}$) — exactly the property Topic 1 identified as what separates real multi-agent
  systems from a sequential function-call chain: "separate persistent state per agent."
- **Coordination emerges purely from each agent broadcasting its own best-known position.**
  Every particle's only outgoing "message" to the rest of the swarm is implicit: "here is the
  best value I've found and where I found it." No particle addresses another particle directly,
  no particle waits for a reply, and any particle can be removed from the swarm at any iteration
  without requiring any change to how the remaining particles compute their updates — dropping
  particle $j$ just means $j$'s report no longer contributes to next iteration's $g^{best}$
  computation.
- **This is Topic 1's broadcast topology, reused as the swarm's entire protocol.** Topic 1
  defined broadcast as one agent's message being visible to every other agent in the system with
  a single `broadcast()` call, contrasted with expensive pairwise `send()`s. PSO's global best
  is exactly that: every particle effectively "publishes" its personal best each iteration, and
  the single shared quantity $g^{best}$ (the minimum over everyone's publication) is what every
  other particle reads back — the same one-to-many communication shape, minimized to its
  simplest possible payload (one position, one scalar value), with no orchestrator reading,
  routing, or validating any of it.

## Algorithm

1. Initialize $n$ particles with random positions $x_i^0$ (uniform over the search domain) and
   random velocities $v_i^0$.
2. Evaluate $f(x_i^0)$ for every particle; set each particle's $p_i^{best} \leftarrow x_i^0$ and
   compute the swarm's initial $g^{best} \leftarrow \arg\min_i f(x_i^0)$.
3. For each iteration $t = 0, \ldots, T-1$:
   a. For every particle $i$, draw $r_1, r_2 \sim U[0,1]$ and update velocity:
      $v_i^{t+1} = w v_i^t + c_1 r_1 (p_i^{best} - x_i^t) + c_2 r_2 (g^{best} - x_i^t)$.
   b. (Optional but standard) Clip $v_i^{t+1}$ to $[-v_{max}, v_{max}]$ to prevent particles
      from overshooting the search domain in one step.
   c. Update position: $x_i^{t+1} = x_i^t + v_i^{t+1}$, clipped to the search domain's bounds.
   d. Evaluate $f(x_i^{t+1})$; update $p_i^{best}$ if improved.
   e. Update $g^{best}$ if any particle's new $p_i^{best}$ beats the current $g^{best}$.
4. Return $g^{best}$ as the swarm's solution.

## From-scratch implementation

Implemented in NumPy only (no PSO library) in
[001_particle_swarm_optimization.ipynb](001_particle_swarm_optimization.ipynb), fully vectorized
over particles: a `pso()` function taking any vectorized objective, particle count, dimension,
bounds, and hyperparameters $w, c_1, c_2$, and returning the swarm's best position, best value,
and the per-iteration history of the global-best value.

**Main convergence demo — 2D Sphere function**, $f(x,y) = x^2+y^2$ (global minimum $0$ at the
origin), 25 particles, domain $[-5,5]^2$, 40 iterations, $w=0.7,\ c_1=1.5,\ c_2=1.5$. Real
executed output:

```
Final gbest position: [0.00047  0.001344]
Final gbest value:    0.00000203
Distance from true optimum (0,0): 0.00142381
History (every 5 iters): [2.684464 0.016602 0.008643 0.001717 0.000012 0.000002 0.000002 0.000002 0.000002]
```

The swarm's global-best value drops from an initial $\approx 2.68$ (iteration 0) to
$\approx 2 \times 10^{-6}$ by iteration 20 and stays flat from there — the swarm found and
settled on a point $0.0014$ away from the true minimum, well within numerical noise for a toy
problem at this scale. The notebook also plots the convergence curve on a log scale and three
scatter-plot snapshots of every particle's position at iterations 0, 20, and 40 — the swarm
visibly starts as a uniform random scatter across $[-5,5]^2$ and ends as a tight cluster at the
origin.

## Practical implementation

Production PSO usage largely mirrors this from-scratch version directly — unlike gradient-based
optimizers, there is no meaningfully different "framework version" with a fundamentally
different algorithm underneath (no autodiff, no backward pass to hand off to a compiled
kernel). Libraries like `pyswarms` and `scipy.optimize.differential_evolution` (a related
population-based, gradient-free optimizer) add engineering conveniences on top of the same core
loop this notebook implements directly: parallel/vectorized objective evaluation across workers,
adaptive inertia-weight schedules (decaying $w$ from a higher exploration value toward a lower
exploitation value over the run, rather than this topic's fixed $w$), boundary-handling
strategies more sophisticated than simple clipping, and multiple named topologies for which
particles a given particle treats as its "neighborhood" for the social term (this topic uses the
simplest: a single global $g^{best}$ visible to the whole swarm, i.e. Topic 1's broadcast
topology exactly; a "local best" ring topology, common in production PSO, is Topic 1's direct
topology applied to a fixed neighbor set instead). No practical implementation of PSO changes the
velocity/position update equations themselves — the mathematics above *is* the practical
algorithm, engineered for scale rather than reinvented.

## Experiment

**Hypothesis:** a larger swarm should reach a lower best-known value within the same iteration
budget, and should cross any fixed quality threshold in fewer iterations, because more particles
sample more of the search space each iteration — with diminishing returns per added particle.

**Setup:** same Sphere problem and hyperparameters as the main demo, 40-iteration budget, swarm
sizes $\{5, 20, 50\}$, each averaged over 10 random seeds, threshold $f < 10^{-3}$.

**Actual result** (real executed output):

```
n_particles= 5  final_best mean=4.927e-05  std=9.602e-05  reached 0.001 in 100% of runs, mean iter-to-threshold=21.2
n_particles=20  final_best mean=8.476e-07  std=1.303e-06  reached 0.001 in 100% of runs, mean iter-to-threshold=14.0
n_particles=50  final_best mean=3.841e-07  std=5.035e-07  reached 0.001 in 100% of runs, mean iter-to-threshold=11.3
```

**Interpretation:** the hypothesis holds directionally on every measure. Going from 5 to 20
particles nearly halves the mean iterations-to-threshold (21.2 → 14.0, a 34% reduction) and
improves the mean final value by almost two orders of magnitude ($4.9\times10^{-5}$ →
$8.5\times10^{-7}$); going from 20 to 50 gives a further but visibly smaller improvement
(14.0 → 11.3 iterations, a 19% reduction; final value roughly halves again). This is exactly the
predicted diminishing-returns shape: doubling particle count does not double search quality,
because particles increasingly cover overlapping territory once there are enough of them to
already sample the (here, low-dimensional) space densely. All three swarm sizes reached the
threshold in 100% of the 10 seeds on this easy, convex, 2D problem — the *quality* difference
between swarm sizes shows up in speed and final precision, not in whether the swarm succeeds at
all. A harder, higher-dimensional, multi-modal problem would be expected to show swarm size
affecting success rate too, not just speed — outside this experiment's scope.

**Limitations:** only one (easy, convex, 2D) objective was tested; only one hyperparameter
setting per swarm size was tried (no joint tuning of $w, c_1, c_2$ per swarm size); 10 seeds is
enough to see a clear trend but not enough to bound the estimates tightly.

## Failure modes

**Premature convergence when the social term dominates too early.** PSO's social term is what
makes it a swarm instead of $n$ independent hill-climbers — but if it dominates the cognitive
term too heavily, particles stop doing meaningful individual search and simply rush toward
whatever `gbest` was found first, with no mechanism left to refine it further or escape it if
it's a poor local optimum. Demonstrated concretely on the 2D **Rastrigin function**,
$f(x,y) = 20 + x^2+y^2 - 10\cos(2\pi x) - 10\cos(2\pi y)$ (global minimum $0$ at the origin, but
densely covered with local minima at every neighboring integer lattice point) — same swarm size
(25 particles) and iteration budget (60), 15 seeds, two parameter settings differing only in
$w, c_1, c_2$:

- **Balanced**: $w=0.7,\ c_1=1.5,\ c_2=1.5$.
- **Social-heavy**: $w=0.9,\ c_1=0.2,\ c_2=3.5$ — high inertia keeps particles moving, and an
  almost-absent cognitive term leaves almost nothing pulling a particle back toward its own
  discoveries.

Real measured output:

```
Balanced     (w=0.7, c1=1.5, c2=1.5): mean final best = 0.0664  (std=0.2482, best=0.0000, worst=0.9951)
Social-heavy (w=0.9, c1=0.2, c2=3.5): mean final best = 0.4515  (std=0.3584, best=0.0292, worst=1.1516)
Social-heavy is worse by a factor of 6.80x
Mean last-improvement iteration (out of 60): balanced=56.3, social-heavy=35.7
Final mean swarm diversity (distance to centroid): balanced=0.158, social-heavy=1.220
```

The social-heavy swarm's mean final value is $6.80\times$ worse than balanced's, and its worst
seed (1.15) is worse than balanced's worst seed (0.995) too. The clearest direct evidence of
*premature* convergence is the "last-improvement iteration": on average, the balanced swarm keeps
finding small improvements all the way to iteration 56.3 out of 60, while the social-heavy swarm
stops improving at iteration 35.7 — nearly 25 iterations of its budget are spent making zero
progress. That is premature convergence exactly as defined: the search has effectively
terminated in an inferior region while the loop mechanically keeps running.

**Diversity collapse — a more subtle finding than expected.** The naive expectation is that a
social-heavy swarm "collapses" spatially onto one point faster than a balanced one. The measured
result is the opposite: final spatial diversity (mean particle distance to the swarm centroid)
is *higher* for social-heavy (1.220) than for balanced (0.158). This is because social-heavy's
high inertia ($w=0.9$) keeps particles carrying momentum and oscillating around the frozen
`gbest` rather than settling onto it, while balanced's lower inertia and stronger cognitive term
let particles do the fine local search that both refines `gbest` further *and* lets the swarm
physically tighten around it. The lesson: diversity collapse and premature convergence are
related failure symptoms, not the same measurement — a swarm can stop improving (premature
convergence, the more consequential failure) without its particles ever physically clustering
together, if high inertia keeps them orbiting the same stuck point instead. Always check whether
`gbest` is still improving, not just whether particles look spread out.

## Real-world usage

**Engineering design optimization.** PSO and related swarm/evolutionary methods are used to tune
continuous design parameters — antenna geometries, aerodynamic surface shapes, structural truss
dimensions — where the objective (simulated performance) is expensive to evaluate and not
differentiable with respect to the design parameters in any convenient closed form, ruling out
gradient-based optimization outright.

**Robotics swarm search and exploration.** Multi-robot search-and-rescue and environmental
mapping systems use PSO-inspired coordination (each robot as a "particle," sharing local
readings the way particles share personal bests) precisely because it needs no central
coordinator — directly the property this topic's "Why simpler approaches fail" argued a
manager/worker orchestrator cannot give: robustness to any single robot's failure, since no
robot's role is distinguished from any other's.

**Contrast with `12-reinforcement-learning/03-policy-gradients`.** Both are optimization methods
over a parameter space, but they solve fundamentally different problem shapes. Policy gradients
optimizes a neural network's weights $\theta$ by computing $\nabla_\theta J(\theta)$ and taking a
gradient step — it *requires* the objective to be differentiable (or at least amenable to a
differentiable estimator, as REINFORCE constructs via the log-probability trick) and uses that
gradient's exact direction of steepest ascent at every step. PSO needs none of that: it only ever
*evaluates* $f$ at points and never differentiates it, which is exactly why it can optimize
objectives policy gradients cannot touch — a black-box physics simulator, a non-differentiable
engineering cost function, a discrete/combinatorial-flavored search space forced into a
continuous relaxation. The trade is real, not free: gradient information, when available, is a
far more efficient search signal per function evaluation than PSO's population-based scatter, so
policy gradients converges faster on problems where a gradient exists and is informative. PSO
gives up efficiency to gain applicability — a genuinely different tool for a genuinely different
problem shape, not a strictly better or worse version of the same idea.

## Mental model

**A swarm is a population of independent agents, each pulled by only two things — its own best
memory and the group's best memory — with no agent ever in charge; good solutions emerge from
that pull-and-broadcast loop, not from anyone deciding where to look.**

## Questions to think about

1. If $c_1 = 0$ (no cognitive term at all, only social), what does each particle's trajectory
   look like, and why would the swarm likely converge faster but find worse solutions than a
   balanced setting — connect your answer to this topic's Rastrigin failure-mode result.
2. If $c_2 = 0$ (no social term at all, only cognitive), the particles never communicate with
   each other. What does the swarm reduce to, and would you still call it "multi-agent
   coordination" under Topic 1's definition of what separates multi-agent systems from
   independent sequential processes?
3. This topic's swarm-size experiment found diminishing returns going from 20 to 50 particles on
   an easy 2D problem. Would you expect the same diminishing-returns curve to hold on a
   50-dimensional problem — why or why not, and what does that imply about how swarm size should
   scale with search-space dimensionality?
4. The global-best topology used here (every particle sees the single swarm-wide best) is
   Topic 1's broadcast topology. Sketch what a "local best" swarm using Topic 1's direct topology
   instead (each particle only sees a fixed set of neighbors' bests, not the whole swarm's)
   would need to change in the velocity update — and predict whether it would make premature
   convergence more or less likely than the global-best version, and why.
5. A central orchestrator (Topic 2) can enforce a global correctness check that no individual
   worker can see on its own. What is the PSO swarm's equivalent, if any — is there anything
   validating that $g^{best}$ is genuinely correct at the point it's reported, or does the swarm
   trust every particle's self-reported value unconditionally? What failure would an incorrect
   `gbest` report (e.g. from a corrupted particle) cause, given this topic's design?
