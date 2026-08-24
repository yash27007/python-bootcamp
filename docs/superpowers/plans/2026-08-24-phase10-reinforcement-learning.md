# Phase 10: Reinforcement Learning First-Principles Build-Out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** New `12-reinforcement-learning/` section: MDPs → Q-learning → policy gradients, first-principles, toy-scale throughout (binding "no heavy training" constraint, per AGENTS.md).

**Architecture:** 3 topics, 1 task each. Value-based (Q-learning) and policy-based (policy gradients) are genuinely different paradigms — teach both, contrasted directly.

**Tech Stack:** NumPy for the MDP/Q-learning topics (tabular, no deep learning needed — the point is the RL mechanism, not function approximation); PyTorch (cite `09-pytorch`) for the policy-gradient topic, which needs a parameterized policy.

**Spec:** `docs/superpowers/specs/2026-08-23-first-principles-curriculum-design.md`, `AGENTS.md`.

## Global Constraints

- **Binding: no heavy/long-running training.** Tiny grid-world/toy environments, few episodes, well under a few minutes per notebook. `torch.set_num_threads(1)` for any PyTorch topic (Phase 9's lesson — avoids severe CPU thread-oversubscription slowdown on tiny models).
- 12-section notes.md template. Real math throughout (Bellman equation, policy gradient theorem) — no substitution needed.
- Review level: light.

---

### Task 1: MDPs and the Bellman Equation

**Files:** Create `12-reinforcement-learning/01-mdps-and-bellman-equation/` (README.md, notes.md, notebook)

**Content:** Problem = an agent making a sequence of decisions where each choice affects future options and rewards, not just an immediate outcome (unlike every supervised-learning topic so far, which predicts one output from one input). Why-simpler-fails = treating each decision as an independent supervised-learning problem ignores that today's action changes tomorrow's state — there's no fixed labeled dataset to learn from, the agent has to generate its own experience. Mathematical foundation = formalize a Markov Decision Process (states, actions, transition probabilities, rewards, discount factor $\gamma$), derive the Bellman equation for the state-value function $V(s) = \max_a \mathbb{E}[r + \gamma V(s')]$ from the recursive structure of expected discounted return — this derivation is the mathematical core of the whole section, give it real weight. From-scratch = a REAL small grid-world MDP (e.g. a 4x4 or 5x5 grid with a goal state and a few obstacles) implemented in plain NumPy, solved by value iteration (repeatedly applying the Bellman update until convergence) — actually run, show the value function converging, real output. Experiment = hypothesis about how many iterations until convergence for a given grid size/discount factor, actually measured. Failure modes = choosing $\gamma$ too close to 1 causing slow convergence, an incorrectly specified reward function producing unintended optimal behavior (reward hacking — a real concrete toy example). Real-world, Mental model, Questions.

- [ ] Write notes.md + notebook (real value-iteration convergence). README. `git commit -m "Phase 10 Task 1: first-principles build-out — MDPs and the Bellman equation"`.

### Task 2: Q-Learning

**Files:** Create `12-reinforcement-learning/02-q-learning/` (README.md, notes.md, notebook)

**Content:** Problem = Task 1's value iteration needs the full MDP (transition probabilities, reward function) known in advance — real environments usually don't hand you that, the agent has to learn from experience alone. Why-simpler-fails = cite Task 1's value-iteration requirement explicitly (needs the model). Mathematical foundation = derive the Q-learning update rule from the Bellman equation adapted to action-values learned from sampled transitions (temporal-difference learning), the exploration-exploitation tradeoff formalized ($\epsilon$-greedy). From-scratch = a REAL tabular Q-learning agent (plain NumPy Q-table) trained on the SAME grid-world from Task 1 but WITHOUT access to the transition model — actually run for a modest number of episodes, show the learned Q-table converging to match Task 1's value-iteration solution (a genuine correctness check: model-free learning should recover the model-based answer). Experiment = hypothesis that Q-learning converges to the same policy as Task 1's value iteration, actually measured and compared. Failure modes = insufficient exploration getting stuck in a suboptimal policy, learning rate too high causing oscillation, the curse of dimensionality for tabular Q-learning (motivates function approximation — bridge forward to Task 3). Real-world, Mental model, Questions.

- [ ] Write notes.md + notebook (real Q-learning vs value-iteration convergence check). README. `git commit -m "Phase 10 Task 2: first-principles build-out — Q-learning"`.

### Task 3: Policy Gradients

**Files:** Create `12-reinforcement-learning/03-policy-gradients/` (README.md, notes.md, notebook)

**Content:** Problem = Task 2's tabular Q-table doesn't scale to large/continuous state spaces (cite the curse-of-dimensionality failure mode from Task 2 explicitly). Why-simpler-fails = a Q-table with one entry per state is infeasible once states are continuous or high-dimensional. Mathematical foundation = derive the policy gradient theorem (why $\nabla_\theta J(\theta) = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot R]$ — the log-derivative trick) at least at a sketch level, explain REINFORCE's update rule. From-scratch = a REAL small parameterized policy (`nn.Module`, cite `09-pytorch/02-nn-module-and-training-loop`) trained with REINFORCE on a small toy environment (a simplified continuous or larger discrete grid-world, or a tiny custom environment — your call, keep it fast) — actually train for a small number of episodes, show the policy improving (increasing average reward over training), real output. Experiment = hypothesis that average episode reward increases over training, actually measured and plotted. Failure modes = high variance in the gradient estimate (a real, fundamental REINFORCE weakness — mention baseline subtraction as the standard fix, don't necessarily implement it), reward sparsity making learning very slow. Real-world, Mental model, Questions.

- [ ] Write notes.md + notebook (real REINFORCE training curve, `torch.set_num_threads(1)`). README. `git commit -m "Phase 10 Task 3: first-principles build-out — policy gradients"`.

### Task 4: Section/root README

- [ ] Create `12-reinforcement-learning/README.md` (all 3 topics, ✅ Complete). Update root `README.md`. `git commit -m "Phase 10 Task 4: mark 12-reinforcement-learning complete in section and root README"`.

## Verification

```bash
cd /home/yashwanth-aravind/ml-course/python-bootcamp
.venv/bin/python -c "
import pathlib
for t in sorted(pathlib.Path('12-reinforcement-learning').iterdir()):
    if t.is_dir(): print(t.name, (t/'notes.md').exists(), (t/'README.md').exists())
"
```
