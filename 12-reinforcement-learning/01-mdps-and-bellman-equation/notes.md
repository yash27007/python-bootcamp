# MDPs and the Bellman Equation

## Problem

Every topic so far — regression, classification, even the generative models in `11-generative-ai`
— is a **single-shot prediction problem**: one input goes in, one output comes out, and the output
has no effect on what input arrives next. A classifier labeling an image doesn't change which image
you see tomorrow.

Reinforcement learning is about a different kind of problem: an **agent making a sequence of
decisions where each decision changes the situation it faces next.** A robot choosing to turn left
doesn't just get an immediate outcome — it ends up in a different position, facing a different set
of future choices, some of which lead to reward and some of which don't. A trading algorithm's buy
decision today changes its position, cash, and risk tomorrow. A game-playing agent's move changes
the board every other move is evaluated against.

The problem is: **how do you choose actions now to maximize reward accumulated over a whole
sequence of future decisions, when each action changes the state the next decision is made from?**

## Intuition

Picture a robot on a small grid, trying to reach a charging station in the corner while avoiding a
few obstacles. At every cell it can move up, down, left, or right. Reaching the station is worth a
reward; every step it takes costs a little (battery drain).

If you only cared about the *immediate* effect of one move, you'd have no way to prefer "the move
that starts a good three-step path to the station" over "the move that looks fine right now but
walks into a dead end." The value of a move depends on everything that can happen *after* it — which
depends on the next move, which depends on the move after that, all the way to the station.

This suggests a recursive idea, which turns out to be the entire mathematical foundation of the
field: **the value of being in a state is the immediate reward you can get, plus the (discounted)
value of whatever state you land in next.** If you already knew how good every future state was,
picking the best action right now would be trivial — just look one step ahead. The hard part is that
"how good is this state" is defined in terms of itself.

## Why simpler approaches fail

The obvious first instinct: treat this as `05-machine-learning`-style supervised learning. Collect a
dataset of (state, best action) pairs, fit a classifier, done.

This fails for a structural reason, not a tuning reason:

1. **There is no fixed labeled dataset.** Nobody hands you the "correct" action for a given state —
   correctness depends on the entire future trajectory, which depends on your own choices. The
   labels you'd need (the optimal action) are exactly the thing you're trying to compute.
2. **The agent has to generate its own experience.** Unlike an image classifier trained once on a
   frozen dataset, an RL agent's data distribution is a *consequence* of its own policy: the states
   it visits depend on the actions it has learned to take. A bad early policy visits a narrow,
   unrepresentative slice of states, which is a feedback loop supervised learning never has to deal
   with.
3. **Credit assignment spans time.** A reward received five steps from now might be caused by the
   action taken right now. Supervised learning has no notion of "this label was actually caused by
   an input several steps in the past" — every input maps to its own output independently.

What's needed instead is a mathematical object that captures "value" recursively, in a way that can
be computed from the *structure* of the problem (or, later, from experience) rather than from a
fixed label set. That object is the **Markov Decision Process** and its **value function**.

## Mathematical foundation

### The Markov Decision Process

A Markov Decision Process (MDP) formalizes sequential decision-making as the tuple

$$
(\mathcal{S}, \mathcal{A}, P, R, \gamma)
$$

- $\mathcal{S}$ — the set of **states** the environment can be in (e.g. grid cells).
- $\mathcal{A}$ — the set of **actions** the agent can take (e.g. up/down/left/right).
- $P(s' \mid s, a)$ — the **transition function**: the probability of landing in state $s'$ given
  the agent takes action $a$ in state $s$. This is where the *Markov property* enters: $s'$ depends
  only on $(s, a)$, not on any earlier history. This is what makes the problem tractable — the state
  is defined to contain everything relevant to the future.
- $R(s, a, s')$ — the **reward function**: the immediate numeric payoff for that transition.
- $\gamma \in [0, 1)$ — the **discount factor**: how much a reward received one step in the future is
  worth *right now*, relative to receiving it immediately. $\gamma$ close to $0$ makes the agent
  myopic (only immediate reward matters); $\gamma$ close to $1$ makes it far-sighted (future reward
  counts almost as much as immediate reward).

A **policy** $\pi(a \mid s)$ is a rule for choosing actions given the current state. The agent's
objective is to find the policy that maximizes accumulated reward.

### The return

Define the **return** $G_t$ from time step $t$ as the total discounted reward the agent collects from
then onward:

$$
G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + \cdots = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

$\gamma < 1$ is what makes this sum finite even over an infinite horizon (for bounded rewards), and
it also encodes a real preference: reward now is worth more than the same reward later.

The crucial algebraic fact, which is the seed of everything that follows, is that $G_t$ has a
**recursive structure** — it can be split into "the very next reward" plus "the same kind of sum,
one step later, scaled by $\gamma$":

$$
G_t = R_{t+1} + \gamma \left( R_{t+2} + \gamma R_{t+3} + \cdots \right) = R_{t+1} + \gamma G_{t+1}
$$

This one line — the return at time $t$ equals immediate reward plus a discounted copy of the return
one step later — is the entire reason a recursive (Bellman) equation exists at all.

### Deriving the Bellman equation for $V^\pi$

Define the **state-value function** under a fixed policy $\pi$ as the expected return starting from
state $s$ and following $\pi$ thereafter:

$$
V^\pi(s) = \mathbb{E}_\pi\left[ G_t \mid S_t = s \right]
$$

Substitute the recursive decomposition of $G_t$ derived above:

$$
V^\pi(s) = \mathbb{E}_\pi\left[ R_{t+1} + \gamma G_{t+1} \mid S_t = s \right]
$$

Expand the expectation over the randomness in this one step: which action $\pi$ selects, and which
next state $P$ produces:

$$
V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \Big[ R(s, a, s') + \gamma\,
\mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s'] \Big]
$$

The inner expectation $\mathbb{E}_\pi[G_{t+1} \mid S_{t+1} = s']$ is, by the very definition just
written down, $V^\pi(s')$. Substituting gives the **Bellman expectation equation**:

$$
V^\pi(s) = \sum_{a} \pi(a \mid s) \sum_{s'} P(s' \mid s, a) \Big[ R(s, a, s') + \gamma V^\pi(s') \Big]
$$

This says: the value of a state equals the expected immediate reward, plus the discounted value of
wherever you end up — with the value function appearing on *both sides*, which is exactly the
recursive structure $G_t = R_{t+1} + \gamma G_{t+1}$ demanded.

### From "value of a policy" to "the best possible value"

We don't just want to evaluate a fixed policy — we want the best one. Define the **optimal state-value
function**:

$$
V^*(s) = \max_\pi V^\pi(s)
$$

For the *optimal* policy, there's no need to average over actions weighted by $\pi(a\mid s)$ — the
optimal choice is to always take whichever action has the highest expected value. This replaces the
$\sum_a \pi(a\mid s)(\cdot)$ with a $\max_a(\cdot)$, giving the **Bellman optimality equation**:

$$
V^*(s) = \max_{a \in \mathcal{A}} \sum_{s'} P(s' \mid s, a) \Big[ R(s, a, s') + \gamma V^*(s') \Big]
$$

This is the central equation of the whole section. It defines $V^*$ implicitly, as the fixed point of
an equation that references itself — but that self-reference is exactly what makes it computable:
apply the right-hand side as an *update rule*, repeatedly, and it converges to $V^*$ (proved below).
It is also useful to define the **action-value function**

$$
Q^*(s, a) = \sum_{s'} P(s' \mid s, a) \Big[ R(s, a, s') + \gamma V^*(s') \Big], \qquad
V^*(s) = \max_a Q^*(s, a)
$$

$Q^*$ — "how good is taking action $a$ in state $s$, then acting optimally forever after" — is the
quantity `02-q-learning` learns directly from experience, without requiring $P$ or $R$ to be known in
advance.

### Why repeating the update converges: the Bellman operator is a contraction

Define the **Bellman optimality operator** $T$ acting on any value function $V$:

$$
(TV)(s) = \max_{a} \sum_{s'} P(s' \mid s, a) \big[ R(s, a, s') + \gamma V(s') \big]
$$

$V^*$ is by definition a fixed point of $T$: $TV^* = V^*$. The key fact that makes iterating this
operator a valid algorithm (rather than just a hopeful heuristic) is that $T$ is a
**$\gamma$-contraction** in the sup-norm: for any two value functions $V_1, V_2$,

$$
\| TV_1 - TV_2 \|_\infty \le \gamma \, \| V_1 - V_2 \|_\infty
$$

(Sketch: the $\max_a$ over a common action set cannot increase the gap beyond the gap of the best
matching action, and pulling a probability-weighted sum out of an absolute value only shrinks it by
Jensen's inequality; combined, this bounds the change by $\gamma$ times the input gap.) Applying $T$
repeatedly to *any* starting $V_0$ therefore produces a sequence whose distance to $V^*$ shrinks
geometrically:

$$
\| V_n - V^* \|_\infty \le \gamma^n \, \| V_0 - V^* \|_\infty
$$

This is exactly the mechanism behind **value iteration**, and it is also exactly why $\gamma$ close
to $1$ makes convergence slow (see Failure modes): the contraction rate $\gamma^n$ decays to zero
more slowly the closer $\gamma$ is to $1$.

## Algorithm

**Value iteration**, directly applying the contraction result above:

1. Initialize $V_0(s) = 0$ for all states $s$ (any bounded initialization works, by the contraction
   guarantee above).
2. Repeat for $n = 1, 2, \dots$:
   - For every non-terminal state $s$: $\displaystyle V_n(s) \leftarrow \max_{a} \sum_{s'} P(s'\mid
     s,a)\big[R(s,a,s') + \gamma V_{n-1}(s')\big]$
   - Track $\Delta = \max_s |V_n(s) - V_{n-1}(s)|$.
   - Stop when $\Delta < \theta$ for a small tolerance $\theta$.
3. Recover the greedy policy from the converged $V$:
   $\displaystyle \pi(s) = \arg\max_a \sum_{s'} P(s'\mid s,a)\big[R(s,a,s') + \gamma V(s')\big]$

This requires knowing $P$ and $R$ up front — the environment's full model. `02-q-learning` removes
that requirement.

## From-scratch implementation

A 5×5 grid world, plain NumPy, no libraries beyond `numpy`/`matplotlib`:

- 25 cells, one **goal** at `(4, 4)` (terminal, reward `+10`), three **obstacles** at `(1,1)`,
  `(1,3)`, `(3,2)` (impassable — walking into one bounces back to the same cell).
- Every non-terminal step costs `-1` (a battery-drain living cost), which is what makes *shorter*
  paths to the goal strictly better than longer ones under the Bellman equation.
- **Stochastic ("slippery") transitions**: taking action $a$ succeeds with probability $0.8$; with
  probability $0.2$ the agent instead slides sideways (perpendicular to $a$, split evenly). This
  makes $P(s'\mid s,a)$ a genuine probability distribution rather than a point mass, so the general
  form of the Bellman equation (the sum over $s'$) is doing real work, not degenerating to a single
  term.

Value iteration is applied for `gamma=0.9` until `Δ < 1e-4`. Actual run:

```
Converged in 23 iterations (theta=1e-4, gamma=0.9)

Value function V(s), rounded:
[[-2.08 -1.19  0.04  1.18  2.74]
 [-1.09  0.    1.44  0.    4.7 ]
 [ 0.15  1.56  3.03  5.04  6.74]
 [ 1.31  2.71  0.    7.16  9.28]
 [ 2.71  4.65  6.93  9.28  0.  ]]
```

(`0.` at the obstacle and goal cells — obstacles are excluded from the update, and the terminal
goal's value is 0 by definition, since no further reward is possible from it.) The value clearly
increases monotonically as cells get closer to the goal, and the greedy policy recovered from this
$V$ (arrows below) routes every reachable cell toward the goal around the obstacles — see the
notebook for the full run and the reward-hacking counter-example.

## Practical implementation

Production-scale MDPs — the ones behind real inventory-control, ad-auction, or robotics systems —
have far too many states for a tabular $V(s)$ array to fit in memory, and their $P(s'\mid s,a)$ is
usually not written down explicitly at all. Two changes bridge this from-scratch version to
practice, both covered later in this section:

- **Model-free learning** (`02-q-learning`): replace the exact sum over $P(s'\mid s,a)$ with a
  running average over *sampled* transitions — no explicit transition model required.
- **Function approximation** (`03-policy-gradients`): replace the table $V(s)$ / $Q(s,a)$ with a
  parameterized function (a neural network), so states never need to be enumerated at all.

Frameworks like `gymnasium` (the standard RL environment interface) and libraries like
`stable-baselines3` package these ideas; every one of them is still computing some approximation to
the same Bellman equation derived above.

## Experiment

**Hypothesis:** the number of value-iteration sweeps needed to converge (to a fixed tolerance)
increases as $\gamma$ approaches $1$, because the contraction rate proven above is $\gamma^n$ — a
rate closer to $1$ shrinks the error more slowly per iteration.

**Setup:** the same 5×5 slippery grid world, `theta=1e-8` (a tight tolerance, needed to see the
effect clearly — see Failure modes for why a loose tolerance hides it), sweeping
`gamma ∈ {0.5, 0.7, 0.9, 0.95, 0.99, 0.999, 0.9999}`, counting iterations to convergence for each.

**Actual measured result:**

| $\gamma$ | iterations to converge |
|---------:|------------------------:|
| 0.5      | 21 |
| 0.7      | 27 |
| 0.9      | 36 |
| 0.95     | 38 |
| 0.99     | 40 |
| 0.999    | 41 |
| 0.9999   | 41 |

**Interpretation:** the hypothesis holds — iteration count rises monotonically with $\gamma$, sharply
at first (0.5 → 0.9 nearly doubles it) and then flattens out as $\gamma \to 1$. The flattening is
itself informative: once $\gamma^n$ is already tiny relative to `theta`, the *marginal* slowdown from
pushing $\gamma$ further toward 1 shrinks — but the total iteration count never comes back down,
consistent with the $\gamma^n$ contraction-rate bound derived in Mathematical foundation.

**Limitations:** this is one small, fixed-topology grid — the specific iteration counts don't
generalize to other environments, only the qualitative monotonic trend (a direct consequence of the
$\gamma^n$ bound, which holds for any MDP) does. Tabular value iteration on a 5×5 grid is not
representative of the wall-clock cost on realistic state spaces, where the per-iteration cost, not
just the iteration count, is the bottleneck.

## Failure modes

**1. $\gamma$ too close to 1 slows convergence.** Directly demonstrated by the experiment above: at
`theta=1e-4` (a looser tolerance), `gamma=0.99` and `gamma=0.999` both converge in the same 27
iterations — the difference is invisible. Tightening to `theta=1e-8` reveals it: 40 vs. 41
iterations, and the trend from `gamma=0.5` (21 iterations) up to `gamma=0.9999` (41 iterations) is
clearly monotonic. On this small grid the effect is a few extra iterations; in high-dimensional,
long-horizon problems where a single sweep is itself expensive, the same $\gamma^n$ contraction-rate
result makes near-1 discount factors a genuine computational cost, not just a modeling choice — this
is precisely why practitioners default to $\gamma$ in the 0.95–0.99 range rather than reflexively
using $\gamma \approx 1$ for "maximum farsightedness."

**2. Reward hacking — a misspecified reward produces an unintended optimal policy.** Value iteration
finds the policy that is *provably optimal for the reward function you wrote*, not the one you
*meant*. Change only the living cost from `-1` (correct: moving costs battery) to `+2` (a plausible
mistake: "reward the agent for staying active/exploring") and rerun value iteration with everything
else identical:

```
correct-reward policy (living_reward = -1):        hacked-reward policy (living_reward = +2):
v > v > v                                            ^ ^ ^ ^ ^
v X v X v                                            ^ X ^ X ^
> > > v v                                            ^ ^ ^ ^ ^
v v X v v                                            ^ ^ X ^ ^
> > > > G                                            ^ ^ ^ < G
```

The correct-reward policy funnels every state toward the goal `G`. The hacked-reward policy points
almost everywhere *away* from the goal — the agent has learned to **never finish**, looping in the
open grid forever to keep collecting `+2` per step, rather than take the one-time `+10` and have the
episode end. This is provably optimal *for the reward as written*: with `gamma=0.9`, looping forever
at `+2`/step is worth $2/(1-0.9) = 20$, strictly more than the `10` the goal pays once. The math did
exactly what it was told; what was told was wrong. This is the same failure category as the
well-known real-world case of a boat-racing agent that learned to spin in a small circle collecting
respawning bonus targets forever instead of finishing the race — the reward function scored bonus
pickups, not race completion, and the agent optimized precisely what was measured.

## Real-world usage

- **Inventory and supply-chain control**: states are stock levels, actions are order quantities,
  transitions are demand uncertainty — classic MDP formulations predate modern deep RL by decades
  (dynamic programming, Bellman's original 1950s work).
- **Robotics motion planning**: states are robot configurations, the Bellman equation underlies
  value-based planners even when the "value iteration" is dressed up as A* / Dijkstra-style search on
  a discretized state space.
- **Recommendation and ad-serving systems**: framed as MDPs where showing a user one item changes the
  state (their session context) that future recommendations are evaluated against.
- **Game AI**: value iteration and its approximations are the backbone of classical game-tree search
  and were the starting point (before scaling with deep networks) for systems like AlphaGo.

## Mental model

**Value is contagious backward through time.** The goal's reward doesn't just sit at the goal — the
Bellman equation is the mechanism by which it "leaks backward," one Bellman update at a time, into
every state that can eventually reach it, weighted by how likely and how far away that path is. Value
iteration is nothing more than running that leakage process until it stabilizes.

## Questions to think about

1. Why must $\gamma < 1$ for the return $G_t$ to be guaranteed finite in an environment with no
   terminal state? What breaks in the Bellman equation's derivation if $\gamma = 1$ and rewards don't
   decay to zero?
2. The Bellman optimality equation uses $\max_a$ instead of $\sum_a \pi(a\mid s)(\cdot)$. What would
   change in the derivation (and in the value-iteration algorithm) if you wanted the *expected* value
   under a fixed, given policy instead of the optimal value?
3. In the reward-hacking demo, the agent's behavior is "wrong" only relative to what the designer
   *meant*, not relative to the reward function as *written*. Given that value iteration always finds
   the provably optimal policy for whatever reward it's handed, what kind of testing or inspection —
   short of manually reading off the optimal policy for every reward change — could catch this before
   deployment?
4. The contraction bound $\|V_n - V^*\|_\infty \le \gamma^n \|V_0 - V^*\|_\infty$ holds regardless of
   how $V_0$ is initialized. Why doesn't a smarter initialization (e.g. seeding $V_0$ with a rough
   guess) change the *worst-case* convergence rate, even though it can obviously help in practice?
5. If two of the grid's obstacle cells were removed, would you expect the number of value-iteration
   sweeps to converge to increase or decrease? What property of the grid does iteration count
   actually track (hint: think about what happens on the very first sweep vs. the tenth)?
