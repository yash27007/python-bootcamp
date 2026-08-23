# AdaBoost

## Problem

`11-random-forest` solved the variance problem: many independent trees, averaged, cancel out each
other's idiosyncratic errors. But averaging only helps when the individual models are *unbiased* —
each one is right on average, just noisy. What if the individual models are instead **weak** —
systematically bad, only slightly better than a coin flip, because they are deliberately too simple
to capture the real pattern (e.g. a decision stump that only asks one yes/no question)? Averaging a
crowd of weak, biased models doesn't fix bias; it just gives you a stable, confident, *wrong* answer.
The problem AdaBoost solves: **how do you combine many weak learners — each barely better than
random guessing — into one strong learner, when averaging alone won't do it?**

## Intuition

Imagine training new employees to detect fraudulent invoices. Instead of hiring one expert, you hire
200 entry-level workers. Employee #1 looks at every invoice and makes many mistakes. Instead of
discarding employee #1's mistakes, you hand employee #2 **the exact same invoices, but tell them to
pay extra attention to the ones #1 got wrong**. Employee #3 then focuses on what #1 *and* #2 still
get wrong, and so on. After 200 rounds, you combine everyone's opinion — weighted by how good each
employee turned out to be — into one final verdict. No single employee is an expert, but the
sequence, each specializing in the previous ones' blind spots, becomes one.

This is **AdaBoost (Adaptive Boosting)**: train weak learners *sequentially*, with each new learner
paying extra attention — via reweighted training data — to the examples the previous ensemble got
wrong, then combine all of them in a weighted vote.

## Why simpler approaches fail

`11-random-forest`'s bagging approach — train many models independently (on different bootstrap
samples) and average — targets **variance**: it works because averaging cancels out the part of each
model's error that is idiosyncratic to its particular training sample, while leaving the shared bias
untouched (`11-random-forest`'s Mathematical foundation derives this: averaging unbiased estimators
stays unbiased, and variance shrinks like $\sigma^2/B$ toward a correlation floor).

That's precisely the problem here: a **weak learner** (e.g. a decision stump — a tree with a single
split) is not unbiased-but-noisy, it is **systematically wrong** — its whole error surface is bias,
not variance, because it's too simple to represent the true decision boundary no matter which
training sample it sees. Bagging a thousand stumps trained on a thousand different bootstrap samples
just gives a thousand *slightly different but equally biased* stumps; averaging them converges to
the *average bias*, not to the truth — the systematic error doesn't cancel out because it isn't
idiosyncratic, it's shared by every stump regardless of which data it saw. Bagging reduces variance
by averaging independent models; it does nothing for bias. A fundamentally different mechanism is
needed — one that directly targets what each model gets *wrong*, not just what it's *inconsistent*
about. That mechanism is boosting: instead of training independently and averaging, train
*sequentially*, and make each new model specialize in correcting the previous ensemble's mistakes.

| Aspect | Bagging (Random Forest, `11-random-forest`) | Boosting (AdaBoost) |
|--------|------------------------|---------------------|
| Trees built | Independently, in parallel | Sequentially (each depends on the previous) |
| Goal | Reduce **variance** | Reduce **bias** |
| Training data | Bootstrap samples (different rows) | All data, but with changing **weights** |
| Tree depth | Deep (low-bias, high-variance trees) | Shallow (usually depth-1 stumps — high bias) |
| Prediction | Average / majority vote | **Weighted** vote (better classifiers vote more) |
| Overfitting | Less prone | More prone (sensitive to noise/outliers) |

## Mathematical foundation

### The weak learner

AdaBoost typically uses **decision stumps** — decision trees with depth 1 (a single split):

```
Is income > $50,000?
       |
   Yes → Predict: "Will buy" (Class 1)
   No  → Predict: "Won't buy" (Class 0)
```

**Formal definition of a weak learner:** an algorithm that produces a classifier with weighted
accuracy only slightly better than 50% — error rate $\epsilon < 0.5$. A single stump is weak on its
own; AdaBoost's job is to combine many of them into a strong learner.

### Setup

Labels use $\pm1$ encoding: $y_i \in \{-1, +1\}$, and each weak learner outputs
$h_t(x) \in \{-1, +1\}$. Start with uniform sample weights:
$$w_i^{(1)} = \frac{1}{N}, \quad i = 1, \dots, N$$
so every example is equally important at round 1.

### Weighted error and the weak learner's vote weight

At round $t$, the weak learner is trained to minimize the **weighted error rate** — the fraction of
total *weight* (not count) that is misclassified:
$$\epsilon_t = \sum_{i=1}^N w_i^{(t)} \cdot \mathbb{1}[h_t(x_i) \neq y_i]$$

Once trained, that stump gets a vote weight $\alpha_t$. Deriving it starts from asking: what should
$\alpha_t$ be to most reduce the ensemble's overall loss? AdaBoost minimizes the **exponential loss**
of the running ensemble score $F(x) = \sum_t \alpha_t h_t(x)$:
$$L(y, F(x)) = e^{-y \cdot F(x)}, \qquad J = \sum_{i=1}^N e^{-y_i F(x_i)}$$

Exponential loss heavily penalizes confident *wrong* predictions: when $F(x)$ has the wrong sign and
is large in magnitude, $e^{-yF(x)} = e^{|F(x)|}$ grows unboundedly. Holding the ensemble-so-far fixed
and adding one new term $\alpha_t h_t(x)$, the total loss after adding stump $t$ splits into the part
already correctly classified and the part misclassified by $h_t$:
$$J(\alpha_t) = \sum_{i: \text{correct}} w_i^{(t)} e^{-\alpha_t} + \sum_{i: \text{wrong}} w_i^{(t)} e^{\alpha_t}
= (1-\epsilon_t) e^{-\alpha_t} + \epsilon_t\, e^{\alpha_t}$$

(using $w_i^{(t)}$ as the un-normalized accumulated exponential-loss weight from previous rounds).
Minimizing $J(\alpha_t)$ over $\alpha_t$ — take the derivative, set to zero —
$$\frac{dJ}{d\alpha_t} = -(1-\epsilon_t)e^{-\alpha_t} + \epsilon_t e^{\alpha_t} = 0
\;\;\Longrightarrow\;\; \frac{\epsilon_t}{1-\epsilon_t} = e^{-2\alpha_t}$$
$$\boxed{\alpha_t = \frac{1}{2}\ln\left(\frac{1-\epsilon_t}{\epsilon_t}\right)}$$

**Reading $\alpha_t$:**
- $\epsilon_t = 0$ (perfect classifier) → $\alpha_t \to +\infty$ (huge positive vote)
- $\epsilon_t = 0.5$ (random guessing) → $\alpha_t = 0$ (no vote — contributes nothing)
- $\epsilon_t > 0.5$ (worse than random) → $\alpha_t < 0$ (its vote is *inverted*)

| Error rate ($\epsilon_t$) | Alpha ($\alpha_t$) | Interpretation |
|----|----|----|
| 0.01 | 2.30 | Very strong learner — large vote |
| 0.10 | 1.10 | Good learner |
| 0.30 | 0.42 | Mediocre learner — small vote |
| 0.45 | 0.10 | Barely better than random — tiny vote |
| 0.50 | 0.00 | Useless — zero vote |
| 0.60 | -0.20 | Worse than random — vote gets flipped! |

### Weight update

The same exponential-loss minimization gives the weight update for the *next* round. Since
$w_i^{(t+1)} \propto w_i^{(t)} \cdot e^{-\alpha_t y_i h_t(x_i)}$ (the un-normalized weight is exactly
the accumulated per-example exponential loss so far):
$$w_i^{(t+1)} = w_i^{(t)} \cdot e^{-\alpha_t \cdot y_i \cdot h_t(x_i)}$$

Breaking it down, since $y_i, h_t(x_i) \in \{-1, +1\}$:
- **Correctly classified**: $y_i h_t(x_i) = +1$ → multiply by $e^{-\alpha_t}$ → **weight decreases**
- **Misclassified**: $y_i h_t(x_i) = -1$ → multiply by $e^{+\alpha_t}$ → **weight increases**

Then **renormalize** so weights remain a valid probability distribution:
$$w_i^{(t+1)} = \frac{w_i^{(t+1)}}{\sum_{j=1}^N w_j^{(t+1)}}$$

The next weak learner is trained on a distribution where misclassified examples carry more weight —
exactly the "pay extra attention to what the previous employee got wrong" behavior from Intuition,
now derived from minimizing exponential loss rather than asserted by analogy.

### Final prediction

After $T$ rounds, the strong classifier is a **weighted majority vote**:
$$F(x) = \text{sign}\left(\sum_{t=1}^T \alpha_t \cdot h_t(x)\right)$$

For regression (AdaBoost.R2), it's a weighted average instead of a signed vote:
$$F(x) = \frac{\sum_{t=1}^T \alpha_t \cdot h_t(x)}{\sum_{t=1}^T \alpha_t}$$

### Worked example

10 training examples, stumps on a single feature $x$:

| Example | x | y | $w_1$ (initial) |
|---|---|---|---|
| 1 | 1.0 | +1 | 0.1 |
| 2 | 2.0 | +1 | 0.1 |
| 3 | 3.0 | +1 | 0.1 |
| 4 | 4.0 | -1 | 0.1 |
| 5 | 5.0 | -1 | 0.1 |
| 6 | 6.0 | +1 | 0.1 |
| 7 | 7.0 | +1 | 0.1 |
| 8 | 8.0 | -1 | 0.1 |
| 9 | 9.0 | -1 | 0.1 |
| 10 | 10.0 | +1 | 0.1 |

Best stump: "$x < 2.5 \to +1$, else $-1$." This misclassifies examples 6, 7, 10 (their true label is
+1 but the stump predicts $-1$). Weighted error: $\epsilon_1 = 0.1+0.1+0.1 = 0.3$.
$$\alpha_1 = \frac12\ln\left(\frac{1-0.3}{0.3}\right) = \frac12\ln(2.33) = 0.424$$

Misclassified examples (6, 7, 10) get multiplied by $e^{+0.424}\approx1.528$; correctly classified
examples get multiplied by $e^{-0.424}\approx0.655$. After renormalizing, misclassified examples end
up with ~16.7% weight each (up from 10%), while others drop to ~7.2% each. Round 2's stump now sees
a distribution where examples 6, 7, 10 dominate, and will specialize in getting *them* right — this
continues for $T$ rounds, the distribution shifting each round to focus on the ensemble's current
blind spots.

## Algorithm

```
INPUT: Training data (X, y) with y in {-1, +1}, number of rounds T

w_i = 1/N for all i                                   # initialize uniform weights

FOR t = 1 to T:
    1. Fit weak learner h_t (stump) on (X, y) using sample weights w
    2. eps_t = sum_i w_i * 1[h_t(x_i) != y_i]           # weighted error
    3. alpha_t = 0.5 * ln((1 - eps_t) / eps_t)          # vote weight
    4. w_i <- w_i * exp(-alpha_t * y_i * h_t(x_i))      # reweight
    5. w_i <- w_i / sum(w)                              # renormalize

PREDICTION:
    F(x) = sign( sum_t alpha_t * h_t(x) )
```

## From-scratch implementation

`05-machine-learning/12-adaboost/adaboost-from-scratch.ipynb` implements this loop by hand — no
`sklearn.ensemble.AdaBoostClassifier` — using `sklearn.tree.DecisionTreeClassifier(max_depth=1)` as
the weak-learner stump (the point being demonstrated is the boosting *mechanism* — reweighting and
weighted voting — not re-deriving tree splitting, already covered from scratch in `10-decision-tree`).

```python
def adaboost_fit(X, y, n_rounds, rng):
    N = len(y)
    w = np.full(N, 1.0 / N)
    stumps, alphas = [], []
    ensemble_score = np.zeros(N)
    train_errors = []

    for t in range(n_rounds):
        stump = DecisionTreeClassifier(max_depth=1, random_state=rng.randint(1_000_000))
        stump.fit(X, y, sample_weight=w)
        pred = stump.predict(X)

        eps = np.clip(np.sum(w[pred != y]), 1e-10, 1 - 1e-10)
        alpha = 0.5 * np.log((1 - eps) / eps)

        w = w * np.exp(-alpha * y * pred)
        w = w / w.sum()

        stumps.append(stump); alphas.append(alpha)
        ensemble_score += alpha * pred
        train_errors.append(np.mean(np.sign(ensemble_score) != y))

    return stumps, np.array(alphas), np.array(train_errors)
```

Run on a toy `make_moons` dataset (300 points, $\pm1$-encoded labels) for 50 rounds, the training
error drops from round to round as the ensemble accumulates stumps (see Experiment below for the
measured numbers).

## Practical implementation

`sklearn.ensemble.AdaBoostClassifier`/`AdaBoostRegressor` implement this loop in optimized code,
identical in mechanism to the from-scratch step above — same reweighting, same $\alpha_t$ formula —
plus a `learning_rate` shrinkage parameter and support for weak learners other than stumps.

```python
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=1),
    n_estimators=200,
    learning_rate=0.5,
    random_state=42
)
ada.fit(X_train, y_train)

y_pred = ada.predict(X_test)
y_prob = ada.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_test, y_prob):.4f}")

# Staged predictions: accuracy as a function of number of estimators
staged_scores = list(ada.staged_score(X_test, y_test))
print(f"Accuracy at 10 trees: {staged_scores[9]:.4f}")
print(f"Accuracy at 100 trees: {staged_scores[99]:.4f}")

param_grid = {
    'n_estimators': [50, 100, 200, 500],
    'learning_rate': [0.01, 0.1, 0.5, 1.0],
    'estimator__max_depth': [1, 2, 3]
}
grid = GridSearchCV(AdaBoostClassifier(random_state=42), param_grid, cv=5, scoring='roc_auc', n_jobs=-1)
grid.fit(X_train, y_train)
```

`05-machine-learning/12-adaboost/adaboost-classification.ipynb` (travel-package purchase prediction,
comparing `AdaBoostClassifier` against Logistic Regression, Decision Tree, Random Forest, and
Gradient Boosting, plus `RandomizedSearchCV` tuning and an ROC-AUC plot) and
`adaboost-regression.ipynb` (used-car price prediction, comparing `AdaBoostRegressor` against
several baselines) apply this to two real end-to-end projects.

**The shrinkage trick (learning rate).** Instead of $F(x) = \sum_t \alpha_t h_t(x)$, scale each
term: $F(x) = \sum_t \eta \cdot \alpha_t h_t(x)$. Smaller $\eta$ means each stump contributes less,
requiring more estimators but usually generalizing better — the same `learning_rate × n_estimators
≈ constant` tradeoff seen throughout boosting methods (see `13-gradient-boosting`).

| Hyperparameter | Controls | Default | Effect |
|---|---|---|---|
| `n_estimators` | Number of weak learners | 50 | More = potentially better, can overfit |
| `learning_rate` | Shrinks each stump's contribution | 1.0 | Lower = slower learning, needs more estimators |
| `estimator` | Type of weak learner | `DecisionTreeClassifier(max_depth=1)` | Can change depth/type |

### Loss function behind AdaBoost — exponential loss (recap)

As derived above, AdaBoost is provably minimizing $J = \sum_i e^{-y_i F(x_i)}$ round by round. This
is *why* AdaBoost is sensitive to outliers: the loss grows unboundedly for confidently-wrong
predictions, so a mislabeled or noisy example keeps getting up-weighted every round it's
misclassified, receiving astronomically high weight and dominating later rounds' training.

### AdaBoost vs. Gradient Boosting, at a glance

(`13-gradient-boosting` builds directly on this comparison.)

| Aspect | AdaBoost | Gradient Boosting |
|---|---|---|
| Strategy | Reweight examples | Fit residuals / pseudo-residuals |
| Loss function | Fixed (exponential loss) | Flexible (any differentiable loss) |
| Weak learner | Usually stumps (depth = 1) | Usually shallow trees (depth 2–8) |
| Sensitivity to outliers | Very high | Lower (with robust loss functions) |
| First paper | Freund & Schapire, 1995 | Friedman, 2001 |

## Experiment

`adaboost-from-scratch.ipynb`'s training-error-vs-rounds curve.

**Hypothesis (stated before running):** training error should decrease monotonically-ish as the
number of boosting rounds increases — each new stump is chosen specifically to correct the current
ensemble's residual mistakes — though with enough rounds on noisy data it may eventually start
overfitting (falling training error, rising test error).

**Setup:** `make_moons(n_samples=300, noise=0.3)`, $y\in\{-1,+1\}$, 50 rounds of the manual AdaBoost
loop above, `DecisionTreeClassifier(max_depth=1)` stumps, training error recomputed after every round
using the running weighted-vote ensemble.

**Result:**

| Round | Training error |
|---|---|
| 1 | 0.1933 |
| 5 | 0.1033 |
| 10 | 0.1033 |
| 50 | 0.1033 |

**Interpretation:** partially confirms the hypothesis — training error drops sharply over the first
few rounds (0.193 → 0.103) as the ensemble picks up the dominant structure of the moons boundary,
then **plateaus** rather than continuing toward zero. This is an honest, useful result: "training
error decreases with more rounds" does not mean "goes to zero" — with `noise=0.3`, some points from
the two moons genuinely overlap in feature space, and no stump-based ensemble can separate points
that are inherently ambiguous. The plateau is a noise floor, not a sign the algorithm stopped
working.

**Limitations:** only training error was tracked (no held-out test error), so this experiment cannot
show the "eventually may overfit" half of the hypothesis directly — see Failure modes below for that
risk, and `13-gradient-boosting`'s Experiment for a learning-rate/overfitting study on the successor
method. Single toy dataset, single noise level, single random seed.

## Failure modes

- **Sensitive to noisy labels and outliers.** The exponential-loss weight update keeps up-weighting
  points the ensemble gets wrong; a mislabeled point that no reasonable stump can ever classify
  correctly gets exponentially increasing weight every round, eventually dominating training and
  distorting later stumps.
- **Can overfit with too many rounds.** Especially without shrinkage (`learning_rate` near 1.0) or on
  small/noisy datasets, driving training error to zero can mean fitting noise, not signal.
- **Sequential training — cannot be parallelized** across rounds (unlike Random Forest's independent
  trees), so it's inherently slower to train.
- **No native loss flexibility.** Fixed to exponential loss; not something you can swap out for a
  more outlier-robust loss the way `13-gradient-boosting` allows.

## Real-world usage

- **Good fit:** clean, low-noise, moderate-sized tabular datasets where a well-understood, explainable
  ensemble method is wanted, and stumps as weak learners are enough.
- **Consider alternatives:** noisy data → gradient boosting with a robust loss (Huber, MAE); need
  speed/parallelism → Random Forest; very large datasets → XGBoost/LightGBM.
- **Strengths that make it a reasonable default in the right setting:** the algorithmic steps are
  simple to follow end-to-end (no hidden machinery beyond reweighting and a weighted vote); it's often
  competitive with Random Forest in accuracy on clean tabular data; feature importance is available
  (which features stumps split on most, and how much each contributed via $\alpha_t$), giving some
  interpretability despite the ensemble having hundreds of learners; and — unlike bagging, which only
  targets variance — boosting's sequential correction targets bias directly, while the weighted-vote
  combination of many stumps also damps down variance somewhat, so a well-regularized AdaBoost
  ensemble improves on *both* fronts relative to a single stump, even though bias-reduction is its
  primary mechanism.

## Mental model

AdaBoost is a class focusing more and more on the questions it keeps getting wrong: each new weak
learner is trained on a reweighted version of the data where the previous ensemble's mistakes carry
more weight, and the final answer is a weighted vote where the more accurate learners get more say.

## Questions to think about

1. In the alpha formula $\alpha_t = \frac12\ln\frac{1-\epsilon_t}{\epsilon_t}$, why does $\epsilon_t
   > 0.5$ produce a *negative* $\alpha_t$, and what does it mean, mechanically, for a weak learner's
   vote to be "inverted" in the final weighted sum?
2. The weight-update rule multiplies correctly-classified examples' weights by $e^{-\alpha_t}$ and
   misclassified examples' weights by $e^{+\alpha_t}$. If $\alpha_t$ is very small (a nearly-useless
   stump, $\epsilon_t \approx 0.5$), how much does the weight distribution actually change round to
   round — and what does that predict about how many rounds a *weak* weak-learner needs to matter?
3. `Why simpler approaches fail` argues bagging can't fix bias because biased errors are shared across
   bootstrap samples, not idiosyncratic to them. Contrast this with a *deep, overfit* tree (the base
   learner in `11-random-forest`) — why is a deep tree's error idiosyncratic (variance) while a
   stump's error is systematic (bias), even though both are trained on the same kind of data?
4. AdaBoost minimizes exponential loss, and Failure modes says this makes it very sensitive to
   outliers. Sketch, using the loss formula $e^{-yF(x)}$, why a single badly mislabeled point can
   contribute more to the total loss $J$ than dozens of correctly classified points combined.
5. If you ran the from-scratch experiment for many more rounds (say 1000) on a noisy version of the
   moons dataset, what would you expect to see happen to training error versus held-out test error,
   and why does that risk motivate the `learning_rate` shrinkage parameter in the Practical
   implementation section?
