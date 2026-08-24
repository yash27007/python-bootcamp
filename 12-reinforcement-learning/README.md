# 12 – Reinforcement Learning

MDPs → Q-learning → policy gradients, first-principles: value-based and policy-based RL as two
genuinely different paradigms, both solved on the same grid-world where possible so results are
directly comparable, at toy scale per this repo's no-heavy-training constraint for sections 11-15.

| # | Topic | Status | Description |
|---|-------|--------|--------------|
| 01 | [MDPs and the Bellman Equation](./01-mdps-and-bellman-equation/) | ✅ Complete | Bellman equation derived from recursive discounted return, a real 5x5 stochastic grid-world solved by value iteration, a concrete reward-hacking demo |
| 02 | [Q-Learning](./02-q-learning/) | ✅ Complete | Model-free TD learning on the exact same grid-world, real measured 85.71% policy match against Topic 01's value iteration |
| 03 | [Policy Gradients](./03-policy-gradients/) | ✅ Complete | REINFORCE derived via the log-derivative trick, a real trained policy on a continuous-state environment a Q-table can't represent |

## Prerequisites

- `09-pytorch/02-nn-module-and-training-loop` — Topic 03's parameterized policy cites this
  training-loop pattern.

## Environment note

Every training run in this section is toy-scale (seconds, not minutes). Topic 03's PyTorch
training uses `torch.set_num_threads(1)`, the same fix Phase 9 (Generative AI) needed for tiny
matrix ops on this environment's CPU.

## What's next

`13-llms-from-scratch` onward continue the toy-scale discipline and, for the multi-agent and
agent-tooling sections, build on ideas from both this section (sequential decision-making) and
earlier ones.
