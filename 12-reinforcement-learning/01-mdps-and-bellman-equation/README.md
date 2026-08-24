# 01 – MDPs and the Bellman Equation

Detailed notes (the Markov Decision Process formalized, the Bellman equation derived from the
recursive structure of discounted return, the contraction-mapping argument for why value iteration
converges): [notes.md](notes.md)

Real, actually-executed, toy-scale value iteration on a 5x5 stochastic grid-world MDP — real
convergence output, a $\gamma$-vs-iterations experiment, and a concrete reward-hacking demo, all
with real pasted output: [01-mdps-and-bellman-equation.ipynb](01-mdps-and-bellman-equation.ipynb)

## What you'll learn

Why sequential decision-making — where each action changes the situation the next decision is made
from — is a fundamentally different problem from every prior supervised-learning topic in this
course, and why it can't be solved by collecting a fixed labeled dataset. The Markov Decision Process
formalism (states, actions, transition probabilities, rewards, discount factor $\gamma$), and the
Bellman equation

$$
V^*(s) = \max_{a} \sum_{s'} P(s' \mid s, a) \big[ R(s, a, s') + \gamma V^*(s') \big]
$$

derived step by step from the recursive structure of discounted return $G_t = R_{t+1} + \gamma
G_{t+1}$, together with the contraction-mapping argument for why repeatedly applying this update
(value iteration) is guaranteed to converge.

| Topic | Status |
|-------|--------|
| Problem: sequential decisions where actions affect future state | ✅ Complete |
| Why treating each decision as independent supervised learning fails | ✅ Complete |
| MDP formalized; Bellman equation derived from the recursive return | ✅ Complete |
| Contraction-mapping proof sketch for value-iteration convergence | ✅ Complete |
| Real 5x5 stochastic grid-world MDP solved by value iteration | ✅ Complete |
| Real experiment: iterations-to-converge vs. discount factor $\gamma$ | ✅ Complete |
| Real reward-hacking demo (misspecified reward flips optimal policy) | ✅ Complete |

## Why it matters

Every prior ML topic in this course predicts one output from one fixed input. This is the first
topic where the "correct" output at any moment depends on a whole sequence of future decisions the
agent hasn't made yet, and where there is no fixed dataset of correct answers to learn from at all —
the agent has to generate its own experience. The Bellman equation is the mathematical object that
makes this tractable: it turns "the value of every possible future" into a recursive equation that
can actually be computed. Every algorithm in the rest of this section — Q-learning
(`02-q-learning`), policy gradients (`03-policy-gradients`) — is a different strategy for
approximating a solution to this same equation, so getting the derivation and the exact-solution
case (value iteration, where the full model is known) solid here is what makes the model-free methods
that follow legible as *approximations to something*, rather than a new set of tricks.

## Prerequisites

- Probability basics (`02-statistics`) — expectation, conditional probability; the Bellman equation
  derivation is an exercise in expanding an expectation over one time step.
- Basic NumPy array manipulation (`01-python-foundation`) — the from-scratch grid world is a plain
  NumPy array with iterative in-place updates.
- No deep learning or PyTorch needed for this topic — value iteration here is fully tabular.

## What you'll build

- A 5x5 grid-world MDP (obstacles, one terminal goal, `-1`-per-step living cost) with **stochastic**
  ("slippery") transitions — an action succeeds 80% of the time, slides sideways 20% of the time —
  implemented in plain NumPy with no RL library.
- Value iteration solving it to convergence (23 sweeps at `gamma=0.9`, `theta=1e-4`), with the
  converged value function and the greedy policy recovered from it, visualized as a heatmap and an
  arrow grid.
- A real, measured experiment: iterations-to-converge swept across `gamma ∈ {0.5, ..., 0.9999}` at
  two tolerances, confirming the $\gamma^n$ contraction-rate bound derived in `notes.md`.
- A concrete reward-hacking demonstration: changing one constant (the living reward, from `-1` to
  `+2`) flips the provably optimal policy from "reach the goal" to "loop forever and never reach the
  goal" — computed, not asserted.

## Where it appears in real systems

The Bellman equation is the foundation of dynamic-programming-based planning and control: classical
inventory and supply-chain optimization, robotics motion planning, and — scaled up with sampling and
function approximation — every modern deep-RL system (game-playing agents, robotic control,
RLHF-style fine-tuning of language models). Reward hacking, demonstrated here at toy scale with one
flipped sign, is a real, documented failure mode in production RL systems (the canonical example
being agents that learn to loop for bonus points instead of completing the task the reward was meant
to measure).

## What's next

`02-q-learning` — motivated directly by this topic's biggest practical limitation: value iteration
requires the full transition model $P(s'\mid s,a)$ and reward function $R$ to be known in advance.
Most real environments don't hand you that. Q-learning derives a way to learn the same underlying
value function from sampled experience alone, and is checked for correctness by training it on this
exact grid world and confirming it recovers the same policy value iteration found here.
