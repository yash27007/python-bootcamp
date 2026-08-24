# Q-Learning

## Problem

`01-mdps-and-bellman-equation` solved a 5x5 grid-world MDP by **value iteration** — repeatedly
applying the Bellman optimality update

$$
V^*(s) \leftarrow \max_{a} \sum_{s'} P(s' \mid s, a) \Big[ R(s, a, s') + \gamma V^*(s') \Big]
$$

until it converged. Look closely at what that update needs on the right-hand side: the full
transition function $P(s' \mid s, a)$ — the exact probability of landing in every possible next
state for every state-action pair — and the full reward function $R$. Both were handed to the
algorithm as plain Python dictionaries before a single episode of "experience" ever happened.

Most real environments don't hand you that. A robot doesn't know in advance the exact probability
distribution over where its wheels will actually take it for a given motor command (that depends on
friction, terrain, calibration drift). A trading algorithm doesn't know the market's transition
probabilities. A game-playing agent isn't given the opponent's exact strategy as a lookup table. In
every one of these cases, the only way to find out what happens after taking an action is to
**actually take it** and observe the result.

The problem is: **can an agent learn the same kind of optimal value function and policy that value
iteration computes, using only sampled experience — (state, action, reward, next state) tuples
collected by interacting with the environment — without ever being told $P$ or $R$ directly?**

## Intuition

Imagine learning to navigate a new building with the lights off, versus being handed the floor plan
first. With the floor plan (the model — value iteration's situation), you can compute the shortest
path to any room by reasoning about the whole layout before taking a single step. In the dark (the
model-free situation — this topic), you can only learn "room A tends to lead to room B when I turn
left here" by actually walking around, bumping into some walls, and gradually building up a mental
map of a very specific quantity: **not the full layout, but "how good is it to be in room X and take
action A"** — an action-value estimate — refined a little after every step you take.

That refinement has a simple, greedy trick to it: after taking action $a$ in state $s$, landing in
$s'$, and collecting reward $r$, you don't need to know the *true* value of $s'$ — you just use your
current, possibly-wrong *estimate* of it, and nudge your estimate for $(s,a)$ a little bit toward
"$r$ plus the discounted value of wherever I ended up." Do this enough times, at every state-action
pair, and the estimates converge to the true values — even though at every individual step, the
target you're nudging toward was itself only an estimate. This is the seed of **temporal-difference
(TD) learning**.

## Why simpler approaches fail

The obvious thing to try first: just run value iteration, but replace the exact sum
$\sum_{s'} P(s'\mid s,a)[\dots]$ with... nothing, because there's no $P$ to sum over. That's not a
tuning problem — the algorithm's very first line requires an object (the transition function) that
doesn't exist in the model-free setting. Value iteration is *not a special case that degrades
gracefully* when $P$ is unavailable; it simply cannot run at all.

A second instinct: build a model. Watch the environment for a while, estimate $\hat P(s'\mid s,a)$
and $\hat R(s,a,s')$ from observed frequencies, then hand those estimates to value iteration
("model-based RL"). This can work, but it adds real cost: estimating a full transition distribution
for every state-action pair requires visiting each one enough times to get a reliable *distribution*
estimate (not just a value estimate), and for large or continuous state spaces, storing and updating
that estimated model is itself expensive. It also couples correctness to how good the estimated model
is — errors in $\hat P$ propagate into whatever value iteration computes from it.

What's needed is a way to update value estimates **directly from samples**, without ever
constructing an explicit model of the transition dynamics as an intermediate step. That's exactly
what Q-learning does — it is a genuinely different kind of algorithm, not value iteration with a
patch bolted on.

## Mathematical foundation

### From $V^*$ to $Q^*$

Recall from `01-mdps-and-bellman-equation` that the action-value function is defined as

$$
Q^*(s, a) = \sum_{s'} P(s' \mid s, a) \Big[ R(s, a, s') + \gamma V^*(s') \Big], \qquad
V^*(s) = \max_a Q^*(s, a)
$$

Substituting $V^*(s') = \max_{a'} Q^*(s', a')$ into the first equation gives the Bellman optimality
equation written **entirely in terms of $Q^*$**, with no separate $V$:

$$
Q^*(s, a) = \sum_{s'} P(s' \mid s, a) \Big[ R(s, a, s') + \gamma \max_{a'} Q^*(s', a') \Big]
$$

This form is the key move that makes model-free learning possible: the right-hand side is now an
**expectation over a single next state $s'$**, given a *fixed* $(s, a)$. An expectation over
samples can be estimated by sampling — this is the mathematical bridge from "requires the full
distribution $P$" to "can be estimated one transition at a time."

### From an expectation to a sample: the TD target

Write the right-hand side as an expectation explicitly:

$$
Q^*(s, a) = \mathbb{E}_{s' \sim P(\cdot \mid s,a)} \Big[ R(s, a, s') + \gamma \max_{a'} Q^*(s', a') \Big]
$$

Given one sampled transition $(s, a, r, s')$ from actually acting in the environment, the bracketed
quantity

$$
y = r + \gamma \max_{a'} Q(s', a')
$$

is a single unbiased sample of the thing $Q^*(s,a)$ is defined as the *expectation* of (using the
current table $Q$ as a stand-in for the unknown $Q^*$ inside the max — this is what makes the
algorithm work despite never knowing $P$). $y$ is called the **TD target**, and
$\delta = y - Q(s,a)$ is called the **TD error** — the gap between what the current table predicted
for $(s,a)$ and what this one sample of experience suggests it should be.

### The Q-learning update rule

Rather than solving for $Q^*(s,a)$ in one shot (impossible without $P$), nudge the table's current
estimate a small step $\alpha$ (the **learning rate**) toward the sampled target:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ \underbrace{r + \gamma \max_{a'} Q(s',a')}_{\text{TD target } y} - Q(s,a) \Big]
$$

This is the **Q-learning update**. Two things about it are worth making explicit:

1. **Averaging away sampling noise.** A single transition $(s,a,r,s')$ is a noisy sample — the same
   $(s,a)$ pair, sampled again, might land in a different $s'$ (exactly this happens in the
   stochastic grid world below: taking `right` sometimes lands you where you intended, sometimes
   slides you sideways). $\alpha < 1$ means each update only partially trusts the latest sample,
   so repeated visits to the same $(s,a)$ average out the noise — the table converges to the true
   *expectation* over $s'$, exactly the quantity $Q^*$ is defined as, without ever computing that
   expectation directly via $P$.
2. **Off-policy.** The target uses $\max_{a'} Q(s',a')$ — the value of the *best* action available
   from $s'$ — regardless of which action the agent actually takes next. This means Q-learning
   learns about the optimal policy's values even while *behaving* according to a different
   (exploring) policy, which is exactly what's needed next.

Convergence relies on the same $\gamma$-contraction property of the Bellman optimality operator
proven in `01-mdps-and-bellman-equation`, plus a standard stochastic-approximation condition on
$\alpha$ (roughly: every $(s,a)$ pair visited infinitely often, with $\alpha$ decaying appropriately)
— formal proof is out of scope here, but the practical takeaway (see Failure modes) is that
$\alpha$ too large breaks the "averaging out noise" property that makes convergence possible.

### The exploration-exploitation tradeoff and $\epsilon$-greedy

Point 2 above raises a problem: if the agent always takes the action with the highest current
$Q(s,a)$ estimate (**exploit**), it will never try actions it currently underestimates, and its
estimate of them can never be corrected — including the possibility that an unfamiliar action is
actually better. But if it always picks actions at random (**explore**) it never uses what it has
learned to actually reach good states efficiently, and the reward it collects along the way (which
is fine to sacrifice while *learning*, but matters if the environment is also being used for real)
would be poor.

**$\epsilon$-greedy** is the standard resolution: with probability $\epsilon$, take a uniformly
random action (explore); otherwise take $\arg\max_a Q(s,a)$ (exploit):

$$
a = \begin{cases} \text{uniform random action} & \text{with probability } \epsilon \\
\arg\max_{a'} Q(s, a') & \text{with probability } 1-\epsilon \end{cases}
$$

$\epsilon$ is typically started high (e.g. $0.2$–$1.0$, favoring exploration when the table is still
mostly wrong and there's nothing good to exploit yet) and decayed over training toward a small floor
(e.g. $0.01$), shifting the balance toward exploitation as the table becomes trustworthy. Section 7a
below shows concretely what happens when $\epsilon$ is set too low too early.

## Algorithm

**Tabular Q-learning:**

1. Initialize $Q(s,a) = 0$ for all state-action pairs (any bounded initialization works).
2. Repeat for each episode:
   - Initialize $s$ (reset the environment).
   - Repeat for each step of the episode, until $s$ is terminal:
     - Choose $a$ from $s$ using the current $Q$-table and an $\epsilon$-greedy policy.
     - Take action $a$; observe reward $r$ and next state $s'$ from the environment.
     - Update: $Q(s,a) \leftarrow Q(s,a) + \alpha\big[r + \gamma \max_{a'} Q(s',a') - Q(s,a)\big]$
       (using $0$ in place of $\gamma \max_{a'} Q(s',a')$ if $s'$ is terminal).
     - $s \leftarrow s'$.
3. After training, the learned greedy policy is $\pi(s) = \arg\max_a Q(s,a)$, and the implied value
   function is $V(s) = \max_a Q(s,a)$ — directly comparable to value iteration's $V^*$ and $\pi^*$.

Note what this algorithm never references: $P(s'\mid s,a)$. Every quantity it touches comes from
`(s, a, r, s')` tuples the environment itself produces.

## From-scratch implementation

The exact same 5x5 stochastic ("slippery") grid-world MDP from `01-mdps-and-bellman-equation` is
reused deliberately — same 25-cell grid, goal at `(4,4)` (reward `+10`, terminal), obstacles at
`(1,1)`, `(1,3)`, `(3,2)` (impassable), `-1` living cost per non-terminal step, actions succeed with
probability `0.8` and slide sideways with probability `0.2` split evenly. Reusing the identical MDP
is what turns this into a genuine correctness check rather than a demo on a different problem: if
model-free Q-learning recovers the same $V^*$/$\pi^*$ that model-based value iteration found for
*this specific MDP*, that's real evidence the update rule derived above is doing what the math says
it should.

The critical implementation detail: a `GridWorldEnv` class exposes only `reset()` and `step(action)`
to the agent. `step` internally samples from `transition_probs` (the same function value iteration
uses to compute exact sums), but returns only the sampled `(next_state, reward, done)` — the agent's
training loop never calls `transition_probs` itself.

Trained for 3000 episodes, `alpha=0.1`, `gamma=0.9` (identical $\gamma$ to Task 1, for a fair
comparison), $\epsilon$ decaying geometrically from `0.2` to a floor of `0.01`, random start state
each episode. Actual run:

```
Trained for 3000 episodes
avg reward, first 100 episodes: -3.02
avg reward, last 100 episodes:  5.24
```

**Correctness check — the real comparison:**

```
Value-function MAE over 21 non-terminal, non-obstacle states: 0.1731
Value-function max abs error: 0.4057
Greedy-policy match with value iteration: 18/21 = 85.71%
```

```
Value-iteration policy pi* (model-based, Task 1):    Q-learning policy pi_Q (model-free):
v > v > v                                             > > v > v
v X v X v                                             v X v X v
> > > v v                                              > v > v v
v v X v v                                              > v X v v
> > > > G                                              > > > > G
```

Three states disagree, and inspecting the *true* action-values (computed by value iteration) at
those states shows why:

```
(0, 0): VI='down'  Q-learning='right'   (true action-values from VI: {'up': -2.79, 'down': -2.08, 'left': -2.78, 'right': -2.14})
(2, 1): VI='right'  Q-learning='down'   (true action-values from VI: {'up': 0.41, 'down': 1.24, 'left': -0.51, 'right': 1.56})
(3, 0): VI='down'  Q-learning='right'   (true action-values from VI: {'up': -0.53, 'down': 1.31, 'left': 0.2, 'right': 1.21})
```

At `(0,0)`, `down` (-2.08) and `right` (-2.14) are within `0.06` of each other — a difference smaller
than the noise a few thousand sampled episodes leave in a finite Q-table estimate. `(2,1)` and
`(3,0)` show larger true gaps (up to ~0.3), meaning the mismatch there is a real, if small,
estimation error rather than a coin flip on an exact tie — consistent with 3000 episodes being
"enough to get close, not infinite."

## Practical implementation

Every production RL system's model-free core still is this update rule, generalized:

- **DQN (Deep Q-Networks)** and its descendants replace the table $Q(s,a)$ with a neural network
  $Q_\theta(s,a)$, and replace the tabular update with a gradient step that pushes
  $Q_\theta(s,a)$ toward the same TD target $y = r + \gamma\max_{a'}Q_\theta(s',a')$ derived above —
  the math is identical, only the function class representing $Q$ changes.
- **`gymnasium`** (the standard RL environment interface) formalizes exactly the `reset()`/`step()`
  interface built from scratch here — this notebook's `GridWorldEnv` is a miniature, hand-rolled
  version of a `gymnasium.Env`.
- Libraries like `stable-baselines3` ship production-grade Q-learning variants (DQN, and value-based
  methods more broadly) with replay buffers, target networks, and other stabilization tricks — all of
  which exist to fix specific failure modes of the plain update derived here (see Failure modes and
  `03-policy-gradients` for the next-order limitation this doesn't fix: tabular/table-like methods
  not scaling to large state spaces at all).

## Experiment

**Hypothesis:** a tabular Q-learning agent, trained model-free on the exact same MDP that Task 1
solved by value iteration, converges to a value function and greedy policy that closely match value
iteration's — because Q-learning's update rule is derived from the same Bellman optimality equation,
just applied to sampled transitions instead of the full model.

**Setup:** 3000 episodes, `alpha=0.1`, `gamma=0.9` (matching Task 1's $\gamma$), $\epsilon$ decayed
from `0.2` to a floor of `0.01`, random start state each episode, max 200 steps/episode. Compare the
resulting $V_Q$/$\pi_Q$ directly against $V^*$/$\pi^*$ from value iteration on the identical MDP.

**Actual measured result:**

| Metric | Value |
|---|---|
| Value-function MAE (21 non-terminal, non-obstacle states) | 0.1731 |
| Value-function max abs error | 0.4057 |
| Greedy-policy match with value iteration | 18/21 = 85.71% |

**Interpretation:** the hypothesis holds at a practically meaningful level. `0.17` average error is
small relative to the reward scale (`-1` per step, `+10` at goal, values ranging roughly `-2` to
`+9.3`), and all three policy mismatches occur at states where value iteration's own true
action-values are close together (see the from-scratch section above) — exactly where sampling-based
learning is expected to occasionally settle on the second-best action rather than a sign the
algorithm failed to converge.

**Limitations:** one grid, one seed, one hyperparameter configuration. Section 7 below shows this
same comparison degrading measurably under worse hyperparameters — the 85.71% match above is not a
free property of "running Q-learning," it depends on reasonable exploration and a reasonable learning
rate.

## Failure modes

**1. Insufficient exploration ($\epsilon$ too low) degrades the learned policy.** A controlled,
seed-averaged comparison (20 seeds, identical `alpha=0.1`/`gamma=0.9`, fixed start state `(0,0)`,
short 80-episode budget so the exploration difference actually matters) between pure-greedy
($\epsilon=0$) and $\epsilon$-greedy ($\epsilon=0.3$, no decay):

```
epsilon=0.0 (pure greedy, no exploration):
  mean value-MAE over 20 seeds:    2.435 (+/- 0.206)
  mean policy match over 20 seeds: 80.5%

epsilon=0.3 (epsilon-greedy):
  mean value-MAE over 20 seeds:    2.259 (+/- 0.132)
  mean policy match over 20 seeds: 82.4%
```

Pure-greedy is measurably worse, on average, under an identical short training budget — real,
seed-averaged numbers. The gap is softened here by this particular grid's own stochastic
("slippery") transitions, which inject a small amount of *accidental* exploration even under a
nominally greedy policy (a sideways slip occasionally forces a visit to a state the agent wouldn't
have chosen). In a **fully deterministic** environment that accidental exploration vanishes: a
pure-greedy agent can lock onto the very first path it happens to find to the goal — even a strictly
longer one — because states off that path are never sampled again, so their $Q$ values never get the
chance to update past their zero initialization and be recognized as better. This is the textbook
"stuck in a suboptimal policy" failure of insufficient exploration.

**2. Learning rate too high causes oscillation instead of convergence.** Sweeping `alpha` from `0.1`
to `1.0` (everything else identical — 3000 episodes, `gamma=0.9`, decaying `epsilon`):

```
 alpha |  value-MAE |   reward std (last 200 ep) |  reward mean (last 200 ep)
----------------------------------------------------------------------------
   0.1 |      0.173 |                      3.058 |                      5.485
   0.5 |      0.849 |                      3.111 |                      6.000
   0.9 |      1.123 |                      4.672 |                      4.890
   1.0 |      2.918 |                     10.293 |                      1.850
```

Value-MAE against value iteration's answer grows sharply as `alpha` rises, and reward variance in
the last 200 episodes (a proxy for whether the policy has actually stabilized late in training) more
than triples from `alpha=0.1` to `alpha=1.0`, while mean reward collapses. At `alpha=1.0` the update
$Q(s,a) \leftarrow Q(s,a) + 1.0[y - Q(s,a)] = y$ **completely discards** the old estimate on every
visit and replaces it with whatever the single, noisy sample happened to say — the table chases
sampling noise instead of averaging it out, and never settles. This is the same
update-magnitude-vs-noise tradeoff that motivates learning-rate schedules throughout deep learning
(`06-deep-learning`), visible here in miniature.

**3. The curse of dimensionality for tabular Q-learning.** This entire topic works because the state
space has exactly 25 cells, so `Q` is a `25 x 4` array and every state gets visited enough times, in
a few thousand episodes, for a reliable estimate. A tabular $Q$-table needs one row *per state*, and
every extra state variable multiplies the table size: a robot tracking $(x, y)$ position, a
discretized heading, and a battery level might already need $50 \times 50 \times 8 \times 10 =
200{,}000$ rows, and a genuinely continuous state (real-valued position, sensor readings) has
infinitely many states — a table cannot represent it at all, no matter how much memory or training
time is available. `03-policy-gradients` (not yet built) addresses exactly this: replacing the table
with a parameterized function that *generalizes* across similar states instead of needing every one
visited individually.

## Real-world usage

- **DQN and successors** (the algorithm behind early superhuman Atari-playing agents) are this exact
  update rule with the table replaced by a neural network — the core TD-target/TD-error mechanism is
  unchanged.
- **Recommendation systems** sometimes frame item selection as a bandit/MDP problem solved with
  Q-learning-style value estimation from logged user-interaction data (a model-free setting almost by
  necessity — nobody has an explicit model of user behavior).
- **Robotics and control**, where the true dynamics (friction, actuator noise, wear) are expensive or
  impossible to model exactly, making model-free learning from real or simulated rollouts the
  practical default over hand-derived transition models.
- **Classical RL benchmarks** (`gymnasium`'s FrozenLake, CliffWalking, Taxi environments) are exactly
  this kind of small, tabular, stochastic-grid problem — the toy environment here is deliberately in
  the same family.

## Mental model

**Value iteration reasons about the whole map before taking a step; Q-learning builds the map one
footstep at a time.** Both are computing the same underlying object (an optimal action-value
function satisfying the Bellman optimality equation) — the difference is entirely in what information
is available to work from: the full transition model up front, versus only the transitions the agent
personally experiences, one sampled $(s,a,r,s')$ at a time, nudged into a running estimate that
converges to the same answer given enough of them.

## Questions to think about

1. Q-learning's target uses $\max_{a'} Q(s',a')$ regardless of which action is actually taken next
   (off-policy). What would change about the update rule — and about what the agent ends up learning
   — if the target instead used $Q(s', a')$ for whatever action $a'$ the $\epsilon$-greedy policy
   actually selects at $s'$? (This alternative algorithm is called SARSA — you don't need to know the
   name to reason about the consequence.)
2. The Q-learning update only ever touches $Q(s,a)$ for the specific $(s,a)$ pair just visited. Given
   that, why does training on 3000 episodes with *random* start states each episode matter for how
   well the final table matches value iteration's answer, compared to always starting from the same
   fixed cell?
3. In the $\alpha=1.0$ failure-mode result, the reward *mean* over the last 200 episodes is much
   lower than at $\alpha=0.1$, even though $\alpha=1.0$ episodes still occasionally reach the goal
   using a table that "should" have seen just as much experience. What does this say about the
   relationship between a $Q$-table's *accuracy* and the *greedy policy* derived from it — can a
   noisy, oscillating table still occasionally produce a good policy by chance, and why would that be
   unreliable?
4. This topic's environment is small enough that random exploration alone (no clever strategy)
   reliably visits every state within a few thousand episodes. Sketch, in plain terms, why a maze ten
   times larger with a single narrow corridor to the goal would make pure random exploration take
   dramatically longer to find any reward signal at all — and connect this to why reward *shaping* or
   smarter exploration strategies become necessary as environments scale up.
5. Q-learning is off-policy and model-free; value iteration is (implicitly) model-based and computes
   values for every state simultaneously via full sweeps. Given a fixed compute budget, under what
   circumstances would you still prefer to build an estimated model $\hat P$ from sampled data and run
   value iteration on it, rather than running Q-learning directly on the same samples?
