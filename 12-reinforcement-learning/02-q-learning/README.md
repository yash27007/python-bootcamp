# 02 – Q-Learning

Detailed notes (deriving the Q-learning update from the Bellman optimality equation adapted to
sampled transitions, temporal-difference learning, the exploration-exploitation tradeoff formalized
as $\epsilon$-greedy): [notes.md](notes.md)

Real, actually-executed, toy-scale tabular Q-learning trained model-free on the exact same 5x5
stochastic grid-world MDP from `01-mdps-and-bellman-equation` — a genuine, measured correctness check
against value iteration's answer, plus concrete low-exploration and high-learning-rate failure-mode
demonstrations, all with real pasted output:
[02-q-learning.ipynb](02-q-learning.ipynb)

## What you'll learn

Why value iteration's requirement — the full transition function $P(s'\mid s,a)$ and reward function
$R$, known in advance — is a real limitation that most environments don't satisfy, and how to derive
a **model-free** alternative from the same Bellman optimality equation. The derivation chain: the
Bellman equation for $Q^*$, rewritten as an expectation over a single next state; a sampled transition
turned into an unbiased estimate of that expectation (the TD target); the Q-learning update rule that
nudges a table toward that target one sample at a time

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \Big]
$$

and the exploration-exploitation tradeoff formalized as $\epsilon$-greedy action selection.

| Topic | Status |
|-------|--------|
| Problem: value iteration needs the full MDP model, most environments don't hand you one | ✅ Complete |
| Why simpler fixes (build-a-model-then-value-iterate) fall short | ✅ Complete |
| Q-learning update derived from Bellman optimality equation for $Q^*$ | ✅ Complete |
| Exploration-exploitation tradeoff formalized as $\epsilon$-greedy | ✅ Complete |
| Real tabular Q-learning agent trained model-free on Task 1's exact grid world | ✅ Complete |
| Real, measured correctness check against Task 1's value-iteration solution | ✅ Complete |
| Real failure-mode demos: insufficient exploration, learning rate too high | ✅ Complete |
| Curse of dimensionality named and bridged forward to `03-policy-gradients` | ✅ Complete |

## Why it matters

`01-mdps-and-bellman-equation` proved that repeatedly applying the Bellman equation converges to the
optimal value function — but only *if* you already know the environment's transition probabilities
and reward function. Q-learning is the first topic in this section that learns from experience alone,
which is the setting almost every real RL application actually faces. It's also a genuine correctness
test of the theory: if a completely different algorithm (sampling-based, model-free) converges to the
same answer a model-based algorithm computed exactly, that's real evidence the underlying math (the
Bellman optimality equation) is the right object being approximated by both.

## Prerequisites

- `01-mdps-and-bellman-equation` — this topic reuses its exact grid-world MDP, its $V^*$/$\pi^*$ as
  the ground truth to check against, and its Bellman-equation derivation as the starting point for
  deriving the Q-learning update.
- Basic NumPy (`01-python-foundation`) — the Q-table is a plain `(states, actions)` NumPy array
  updated in place; no additional libraries needed.

## What you'll build

- A model-free `GridWorldEnv` simulator exposing only `reset()`/`step()` — internally samples from
  the same transition model Task 1 used, but never exposes the probability distribution to the agent.
- A tabular Q-learning agent (plain NumPy `Q` array, $\epsilon$-greedy with decay) trained for 3000
  episodes on the identical 5x5 stochastic grid world, with a real training curve (average reward
  climbing from about `-3` to about `+5.2` per episode).
- A genuine correctness check: the learned value function and greedy policy compared directly against
  Task 1's value-iteration solution — measured value-function MAE (`0.17`) and exact greedy-policy
  match (`18/21 = 85.71%`), with the handful of mismatches traced to states where value iteration's
  own true action-values are nearly tied.
- Two concrete, measured failure-mode demonstrations: a seed-averaged low-exploration comparison
  showing pure-greedy Q-learning underperforming $\epsilon$-greedy, and a learning-rate sweep from
  `alpha=0.1` to `alpha=1.0` showing growing error and reward oscillation as the step size grows.

## Where it appears in real systems

Q-learning's update rule is the core of **DQN** and its successors (the algorithm family behind
early superhuman Atari-playing agents) — DQN is this exact TD-target/TD-error mechanism with the
table replaced by a neural network. It's also the conceptual basis for value-based methods in
robotics and control (where the true transition dynamics are too complex or costly to model exactly)
and appears, in spirit, in any system that estimates the value of an action from logged interaction
data rather than a hand-built simulator of the environment.

## What's next

`03-policy-gradients` (not yet built) — motivated directly by this topic's third failure mode: a
tabular Q-table needs one row per state, and the number of states grows multiplicatively with every
extra state variable, making it infeasible for large or continuous state spaces. Policy gradients
replace the table with a parameterized function (a neural network policy) that generalizes across
similar states instead of memorizing each one — a genuinely different paradigm (value-based vs.
policy-based) that this section will contrast directly against the Q-learning approach built here.
