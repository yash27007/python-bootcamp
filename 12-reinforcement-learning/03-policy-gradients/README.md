# 03 – Policy Gradients (REINFORCE)

Detailed notes (deriving the policy gradient theorem from the log-derivative trick, REINFORCE's
per-timestep return and update rule, why the environment's transition dynamics cancel out of the
gradient the same way they never entered Q-learning's update): [notes.md](notes.md)

Real, actually-executed, small-scale REINFORCE training run: a parameterized `nn.Module` policy
trained on a continuous 2D navigation task where a tabular Q-table cannot be built at all, plus real
measured demonstrations of REINFORCE's two core failure modes (high gradient variance, reward
sparsity), all with real pasted output and a real training-curve plot:
[03-policy-gradients.ipynb](03-policy-gradients.ipynb)

## What you'll learn

Why `02-q-learning`'s tabular approach — a table with one row per state — is not just expensive but
literally impossible to build once the state space is continuous, and how to derive a genuinely
different kind of algorithm: one that trains a parameterized policy $\pi_\theta(a\mid s)$ directly,
using the **log-derivative trick** to get a gradient of the expected return that never needs the
environment's transition model:

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}
\Big[ \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(a_t \mid s_t) \cdot G_t \Big]
$$

— the **policy gradient theorem** — and REINFORCE, the Monte-Carlo algorithm that turns this into a
practical per-episode training loop built on the exact `nn.Module` + `optimizer.step()` pattern from
`09-pytorch/02-nn-module-and-training-loop`.

| Topic | Status |
|-------|--------|
| Problem: tabular Q-tables don't scale to continuous/high-dimensional state spaces | ✅ Complete |
| Why simpler fixes (discretize the state space, then tabulate) fall short | ✅ Complete |
| Policy gradient theorem derived via the log-derivative trick | ✅ Complete |
| REINFORCE's per-timestep return and update rule derived | ✅ Complete |
| Real `nn.Module` policy trained with REINFORCE on a continuous-state task | ✅ Complete |
| Real, measured training curve: reward and success rate rising over training | ✅ Complete |
| Real failure-mode demos: high gradient variance, reward sparsity | ✅ Complete |
| Baseline subtraction named as the standard variance-reduction fix | ✅ Complete |

## Why it matters

`02-q-learning` proved model-free learning works by recovering value iteration's exact answer on a
small, discrete grid world — but flagged its own scaling limit as a table that grows multiplicatively
with every added state variable, and cannot represent a continuous state at all. This topic builds
the environment that limit actually bites (a continuous 2D position, not a discrete cell) and shows a
structurally different algorithm — direct policy optimization via a sampled gradient estimator —
succeeding where tabular methods have no path forward. It's also the first topic in this section
where the "function" being learned generalizes across states it has never exactly visited, which is
the property every deep-RL system beyond toy grid worlds depends on.

## Prerequisites

- `02-q-learning` — this topic's Problem section is built directly on that topic's own named failure
  mode (curse of dimensionality for tabular methods); its Mathematical foundation reuses the same
  discounted-return construction ($G_t$) introduced there and in `01-mdps-and-bellman-equation`.
- `09-pytorch/02-nn-module-and-training-loop` — the policy network is a small `nn.Module`, and the
  training loop is the same forward → loss → `.backward()` → `optimizer.step()` pattern from that
  topic, applied to a different (REINFORCE) loss.
- `09-pytorch/01-tensors-and-autograd` — `loss.backward()` relies on the same reverse-mode automatic
  differentiation covered there; no new autograd concepts are introduced here.

## What you'll build

- `ContinuousGridEnv`: a hand-rolled `reset()`/`step()` environment with a genuinely continuous 2D
  state (position in $[0,1]^2$), four discrete actions, and potential-based reward shaping — designed
  specifically so a tabular $Q$-table cannot be built for it at all (a real, printed comparison shows
  even a coarse $100\times100$ discretization needing $10{,}000$ rows, and still losing precision the
  environment's own movement noise would need).
- A small `PolicyNet` (`nn.Module`, `2 -> 32 -> 4` with a softmax output) trained with REINFORCE for
  600 episodes in under 4 seconds — a real training curve showing average reward rising from `9.72`
  to `13.41` and goal-reaching success rate rising from `57%` to `87%`.
- A real, measured demonstration of REINFORCE's **high-variance** failure mode: return
  standard-deviation relative to mean (`std/|mean|`) measured at `1.16` for an untrained policy,
  dropping to `0.16` after training — and traced directly to a real non-monotonic dip in the training
  curve.
- A real, measured demonstration of the **reward-sparsity** failure mode: the identical setup with
  reward shaping removed gets stuck at exactly `0%` success for `500` of `800` episodes before
  learning suddenly kicks in once a first success occurs by chance.

## Where it appears in real systems

The policy gradient theorem derived here (log-derivative trick, transition dynamics canceling out of
the gradient) is the mathematical foundation under **PPO**, the algorithm used to train **RLHF**
pipelines that align large language models to human preferences — REINFORCE with a learned value
baseline and additional stabilization, but the same core gradient. It's also the standard approach
for **robotics control** (continuous joint/actuator commands, where a $Q$-table over a continuous
action space is intractable) and for **game-playing agents** with large or continuous state/action
spaces, for the same curse-of-dimensionality reason this topic's toy environment was built to
demonstrate.

## What's next

The 12-reinforcement-learning section's three topics form a complete arc: `01-mdps-and-bellman-equation`
(model-based, exact — full transition model required), `02-q-learning` (model-free, tabular — learns
from samples but still needs one table row per state), `03-policy-gradients` (model-free, function
approximation — generalizes across states via a parameterized policy, the approach that scales to
real-world state and action spaces). See the section root README for how these three fit together and
what deep-RL topics (actor-critic methods, PPO, DQN's function-approximation analogue) build on next.
