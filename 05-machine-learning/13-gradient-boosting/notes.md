# Gradient Boosting

## Problem

`12-adaboost` showed how to combine weak learners sequentially, each one correcting the previous
ensemble's mistakes, by reweighting examples and deriving everything from minimizing exponential
loss. That derivation was tightly coupled to exponential loss and $\pm1$-labeled classification —
what if the target is continuous (regression), or you want to optimize a different loss (one that's
more robust to outliers, say), where AdaBoost's exponential-loss weight-update math simply doesn't
apply? The problem: **generalize AdaBoost's "sequentially focus on mistakes" idea to arbitrary
differentiable loss functions**, not just exponential loss on classification.

## Intuition

You're estimating the value of a house. Your first guess is $300,000. It's actually $350,000, so
you're off by $50,000. Instead of throwing that guess away and starting over, you train a *second*
model whose only job is to predict the gap — the $50,000 error. Add its prediction to your first
guess, and your combined estimate is now $350,000, essentially exact. In practice the second model
won't be perfect either, so there's a new, smaller gap; train a third model to close *that* one.
Repeat.

This is **Gradient Boosting**: build models sequentially, where each new model predicts the
**residual error** left by the current ensemble, gradually closing the gap between prediction and
target.

## Why simpler approaches fail

AdaBoost's whole mechanism — up-weight misclassified points, derive $\alpha_t$ by minimizing
$\sum_i e^{-y_i F(x_i)}$ — is a clever trick specific to exponential loss and $\pm1$ classification
labels. It doesn't generalize cleanly:

- **Regression has no natural "misclassified" indicator to reweight.** AdaBoost's whole update
  hinges on $\mathbb{1}[h_t(x_i)\neq y_i]$ and $y_i h_t(x_i) \in \{-1,+1\}$; there's no equivalent
  notion of a binary right/wrong to reweight by when predicting a continuous value.
- **Other loss functions (robust regression, multi-class log loss) don't have a matching closed-form
  $\alpha_t$ derivation.** AdaBoost's $\alpha_t = \frac12\ln\frac{1-\epsilon_t}{\epsilon_t}$ came
  from differentiating exponential loss specifically — differentiating a different loss (say
  absolute error, or log loss) gives a *different* optimal step, and reworking the whole derivation
  by hand for every loss function you might want is not scalable.

What's needed is a *general recipe* — one that works for any differentiable loss, and derives
"what should the next weak learner fit?" directly from calculus on that loss, rather than a
loss-specific reweighting trick. That recipe is functional gradient descent: instead of reweighting
examples, fit the next model directly to the **gradient of the loss** with respect to the current
predictions.

## Mathematical foundation

### Fitting residuals — the MSE special case

Consider regression with targets $y_i$ and ensemble prediction $F(x_i)$. Round 1 trains $h_1$
(predictions $\hat y_1$); the residual is $r_i = y_i - \hat y_1(x_i)$. Round 2 trains $h_2$ *directly
on the residuals* $r_i$; the updated prediction is $\hat y_1 + h_2(x)$, with new residuals
$r_i' = y_i - \hat y_1(x_i) - h_2(x_i)$, and so on. After $T$ rounds:
$$F(x) = h_1(x) + h_2(x) + \dots + h_T(x)$$

### Why "gradient"

The connection to gradients: for squared-error loss $L(y,F) = \frac12(y-F)^2$, the **negative
gradient** of the loss with respect to the prediction $F$ is exactly the ordinary residual:
$$-\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} = -\frac{\partial \frac12(y_i-F(x_i))^2}{\partial F(x_i)} = y_i - F(x_i) = r_i$$

Fitting the residuals is therefore **gradient descent in function space**: each new tree is a step in
the direction that most reduces the loss, where the "direction" lives in the space of functions
(predictions at every training point) rather than a finite parameter vector. This is the
generalization `Why simpler approaches fail` asked for: swap in a different loss function, get a
different formula for the "residual" (a **pseudo-residual**), and the *same algorithm* still applies.

### The general algorithm

For any differentiable loss $L(y, F(x))$, the pseudo-residual at round $t$ is:
$$r_i^{(t)} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F=F^{(t-1)}}$$

```
1. Initialize: F_0(x) = argmin_gamma sum_i L(y_i, gamma)
   (for MSE, this is just mean(y))

2. For t = 1, ..., T:
   a. Compute pseudo-residuals: r_i^(t) = -dL(y_i, F(x_i))/dF(x_i), evaluated at F = F_{t-1}
   b. Fit a regression tree h_t to {(x_i, r_i^(t))}
   c. Find optimal step size (leaf values): gamma_t = argmin_gamma sum_i L(y_i, F_{t-1}(x_i) + gamma * h_t(x_i))
   d. Update: F_t(x) = F_{t-1}(x) + eta * gamma_t * h_t(x)

3. Return F_T(x)
```

where $\eta$ (eta) is the **learning rate** (shrinkage), typically 0.01–0.3.

### Pseudo-residuals for different loss functions

| Loss function | Use case | Pseudo-residual $r_i$ |
|---|---|---|
| $\frac12(y_i-F)^2$ (MSE) | Regression | $y_i - F(x_i)$ |
| $\lvert y_i-F\rvert$ (MAE) | Robust regression | $\text{sign}(y_i - F(x_i))$ |
| Huber loss | Regression, outlier-robust | MSE for small errors, MAE for large errors |
| Log loss / binary cross-entropy | Classification | $y_i - \sigma(F(x_i))$ |
| Deviance | Multi-class | $(y_k - P(y=k \mid x))$ per class |

**Key insight:** the flexibility to plug in any loss function is what makes gradient boosting a
strict generalization of AdaBoost (exponential loss) and of plain MSE-residual boosting.

### Worked example (MSE loss)

| x | y (actual) |
|---|---|
| 1 | 2.5 |
| 2 | 3.8 |
| 3 | 6.1 |
| 4 | 8.0 |
| 5 | 9.5 |

Initial prediction (mean): $F_0 = (2.5+3.8+6.1+8.0+9.5)/5 = 5.98$.

**Round 1 residuals** $r_1 = y - F_0$: $-3.48,\ -2.18,\ +0.12,\ +2.02,\ +3.52$.

Fit a stump to the residuals; best split at $x=2.5$: left ($x\le2$) mean residual $=(-3.48-2.18)/2=
-2.83$; right ($x>2$) mean residual $=(0.12+2.02+3.52)/3=1.89$.

With learning rate $\eta=0.1$: $F_1(x) = F_0(x) + 0.1 \times h_1(x)$. For $x=1$:
$F_1(1) = 5.98 + 0.1\times(-2.83) = 5.70$ (actual: 2.5 — still far, but closer than $F_0=5.98$).
Repeating for $T=200$ rounds gradually closes the remaining gap.

### Classification: log-odds and pseudo-residuals

For binary classification, the ensemble outputs a log-odds score:
$$F(x) = \log\left(\frac{P(y=1\mid x)}{P(y=0\mid x)}\right), \qquad P(y=1\mid x) = \sigma(F(x)) = \frac{1}{1+e^{-F(x)}}$$

Pseudo-residuals for binary cross-entropy loss: $r_i = y_i - \sigma(F(x_i))$ — the gap between the
actual class (0 or 1) and the model's current predicted probability. Initial prediction:
$F_0 = \log\left(\frac{p}{1-p}\right)$ where $p$ is the fraction of class-1 examples.

For $K$-class classification, train $K$ trees per round (one per class); predictions combine via
softmax: $P(y=k\mid x) = \dfrac{e^{F_k(x)}}{\sum_{j=1}^K e^{F_j(x)}}$.

### Learning rate and the bias-variance tradeoff

$$F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x), \quad \eta \in (0,1]$$

| Learning rate | `n_estimators` needed | Training speed | Generalization |
|---|---|---|---|
| 0.3 | ~100 | Fast | OK |
| 0.1 | ~300 | Medium | Good |
| 0.05 | ~600 | Slow | Better |
| 0.01 | ~3000 | Very slow | Often best |

**Rule of thumb:** lower learning rate + more trees almost always outperforms higher learning rate +
fewer trees, given enough computation budget — the same shrinkage tradeoff introduced in
`12-adaboost`'s Practical implementation, now central rather than an add-on.

### Stochastic gradient boosting (subsampling)

Friedman (2002): at each round, train on a random subset (without replacement) of the training rows
instead of all $N$. Benefits: (1) faster training (fewer examples per tree), (2) additional
regularization (reduces overfitting), and (3) the excluded-per-round examples can be used to estimate
generalization error, similar in spirit to `11-random-forest`'s OOB error — though here the mechanism
(`subsample < 1.0`) decorrelates *sequential* trees rather than parallel ones. Typical setting:
`subsample=0.5`–`0.8`.

## Algorithm

```
INPUT: Training data (X, y), differentiable loss L, number of rounds T, learning rate eta

F_0(x) = argmin_gamma sum_i L(y_i, gamma)     # e.g. mean(y) for MSE

FOR t = 1 to T:
    1. Compute pseudo-residuals r_i = -dL(y_i, F_{t-1}(x_i)) / dF_{t-1}(x_i)
    2. Fit regression tree h_t to {(x_i, r_i)}
    3. gamma_t = argmin_gamma sum_i L(y_i, F_{t-1}(x_i) + gamma * h_t(x_i))
    4. F_t(x) = F_{t-1}(x) + eta * gamma_t * h_t(x)

RETURN F_T
```

## From-scratch implementation

`05-machine-learning/13-gradient-boosting/gradient-boosting-from-scratch.ipynb` implements this by
hand for squared-error loss, where — per the derivation above — the pseudo-residual is just the
ordinary residual. Each round fits a shallow `sklearn.tree.DecisionTreeRegressor` to the current
residuals and adds a shrunk copy to the running prediction:

```python
def gb_fit(X, y, n_rounds, learning_rate, max_depth=2):
    F0 = np.mean(y)
    current_pred = np.full(len(y), F0)
    trees, train_mse = [], []
    for t in range(1, n_rounds + 1):
        residuals = y - current_pred                 # pseudo-residual for MSE loss
        tree = DecisionTreeRegressor(max_depth=max_depth, random_state=0)
        tree.fit(X, residuals)
        current_pred = current_pred + learning_rate * tree.predict(X)
        trees.append(tree)
        train_mse.append(np.mean((y - current_pred) ** 2))
    return F0, trees, np.array(train_mse)
```

On a toy 1D target $y = x\sin(x) + \varepsilon$, plotting the ensemble's prediction on a grid at
checkpoints (round 1, 5, 20, 100) against the true function shows it visibly converging toward the
target curve round by round — training MSE dropped from 7.394 after round 1 to 0.047 after 100
rounds in the executed run.

## Practical implementation

`sklearn.ensemble.GradientBoostingClassifier`/`GradientBoostingRegressor` implement exactly this
loop — same pseudo-residual fitting, same shrinkage — in optimized code, with regularization controls
the from-scratch step deliberately left out (tree-complexity limits, row/feature subsampling, and
early stopping).

```python
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, log_loss

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

gb = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    max_features=0.8,
    min_samples_leaf=10,
    random_state=42,
    validation_fraction=0.2,
    n_iter_no_change=20,      # early stopping
    tol=1e-4
)
gb.fit(X_train, y_train)
print(f"Optimal trees: {gb.n_estimators_}")

y_pred = gb.predict(X_val)
y_prob = gb.predict_proba(X_val)[:, 1]
print(classification_report(y_val, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_val, y_prob):.4f}")

# Early stopping via staged predictions, done manually
val_losses = [log_loss(y_val, pred) for pred in gb.staged_predict_proba(X_val)]
optimal_n = val_losses.index(min(val_losses)) + 1
print(f"Optimal trees (manual scan): {optimal_n}")
```

`05-machine-learning/13-gradient-boosting/GradientBoost-Classification.ipynb` (travel-package
purchase prediction) and `Gradientboost-Regression.ipynb` (used-car price prediction) apply this to
two real end-to-end projects, mirroring `12-adaboost`'s and `11-random-forest`'s companion notebooks.

### Regularization in gradient boosting

Without regularization, gradient boosting overfits badly. Techniques, roughly in order of impact:

1. **Learning rate ($\eta$) — shrinkage.** The most important regularizer (see above).
2. **Tree complexity constraints:** `max_depth` (typically 3–8, vs. unlimited for Random Forest),
   `min_samples_leaf`, `max_leaf_nodes`.
3. **Subsampling (row sampling):** `subsample < 1.0`.
4. **Feature subsampling (column sampling):** `max_features < 1.0`, borrowed from Random Forest.
5. **Early stopping:** monitor validation loss via `staged_predict`/`staged_predict_proba` and stop
   once it stops improving (shown in the code above).

| Hyperparameter | Controls | Typical range |
|---|---|---|
| `n_estimators` | Number of rounds | 100–3000 |
| `learning_rate` | Shrinkage ($\eta$) | 0.01–0.3 |
| `max_depth` | Tree depth | 3–8 |
| `subsample` | Row sampling fraction | 0.5–0.9 |
| `max_features` | Feature sampling fraction | 0.5–1.0 |
| `min_samples_leaf` | Min samples per leaf | 1–50 |
| `min_impurity_decrease` | Minimum split gain | 0–0.01 |

**Tuning strategy:** fix `learning_rate=0.1`, tune `n_estimators` with early stopping; then tune
`max_depth` (most impactful after `n_estimators`); add row/column subsampling; once the best tree
structure is found, halve `learning_rate` and double `n_estimators`.

### Comparing Random Forest, AdaBoost, and Gradient Boosting

| Feature | Random Forest (`11-random-forest`) | AdaBoost (`12-adaboost`) | Gradient Boosting |
|---|---|---|---|
| Trees built | Parallel | Sequential | Sequential |
| Depth per tree | Deep | Shallow (stumps) | Shallow (3–8) |
| Learning method | Bagging | Reweighted samples | Gradient descent |
| Loss function | N/A | Exponential | Any differentiable |
| Outlier sensitivity | Low | Very high | Medium (depends on loss) |
| Interpretability | Low | Medium | Low |
| Speed | Fast (parallel) | Slow (sequential) | Slow (sequential) |
| Tuning effort | Low | Low | High |
| Typical accuracy | Good | Good | Better |

### Beyond sklearn: XGBoost, LightGBM, CatBoost

| Library | Key improvement over vanilla GBM |
|---|---|
| XGBoost | Regularized objective (L1/L2 on leaves), exact split finding, parallelism |
| LightGBM | Leaf-wise tree growth (faster), GOSS, EFB — much faster on large data |
| CatBoost | Native categorical feature handling, ordered boosting, symmetric trees |

In practice, use XGBoost or LightGBM instead of sklearn's `GradientBoostingClassifier` for large
datasets — they're much faster, and one of them, XGBoost, is `14-xgboost`'s subject: it adds
explicit regularization and a second-order optimization step on top of exactly this algorithm.

## Experiment

`gradient-boosting-from-scratch.ipynb`'s training-MSE-vs-round curve and learning-rate comparison.

**Hypothesis (stated before running):** training loss should decrease monotonically as rounds
accumulate (each new tree is fit specifically to reduce the remaining residual). Separately, a larger
learning rate should reduce *validation* loss fastest per round but start overfitting (validation
loss turning back up) sooner than a smaller learning rate, which needs more rounds to reach a
comparable or better result.

**Setup:** toy 1D target $y = x\sin(x) + \varepsilon$ ($\varepsilon\sim\mathcal N(0,0.5^2)$), 150
training points. Part 1: `learning_rate=0.3`, `max_depth=2`, 100 rounds, training MSE recorded each
round. Part 2: 70/30 train/validation split, three learning rates $\{0.5, 0.1, 0.02\}$, 150 rounds
each, validation MSE recorded each round.

**Result:**

- Training MSE: 7.394 after round 1 → 0.047 after round 100 — monotonically decreasing, confirming
  the first half of the hypothesis.
- Learning-rate comparison (validation MSE):

| Learning rate | Best val MSE | Round of best | Val MSE at round 150 |
|---|---|---|---|
| 0.5 | 0.489 | 58 | 0.532 (rising — overfitting) |
| 0.1 | 0.417 | 139 | 0.417 (still near-best) |
| 0.02 | 0.891 | 150 | 0.891 (still decreasing — not converged yet) |

**Interpretation:** `learning_rate=0.5` bottoms out earliest and then rises — it overshoots and
starts fitting noise, exactly as hypothesized. `learning_rate=0.1` keeps improving well past round 58
and reaches the best overall validation MSE in this budget. `learning_rate=0.02` is still decreasing
at round 150 — it hasn't overfit, but it also hasn't had enough rounds yet to catch up, illustrating
the "needs more trees" half of the shrinkage tradeoff directly.

**Limitations:** single toy dataset, single train/val split, only 150 rounds tested (the smallest
learning rate likely needs several times that to fully converge), and `max_depth=2` fixed rather than
swept jointly with learning rate.

## Failure modes

- **Slow to train** — sequential by construction, cannot be parallelized across rounds the way
  Random Forest's independent trees can (only within-tree split-finding parallelizes).
- **Sensitive to learning rate and number of estimators** — too high a learning rate or too many
  rounds without early stopping overfits (see Experiment above); too low a learning rate without
  enough rounds underfits (hasn't converged).
- **Easy to overfit without early stopping** — unlike bagging, more rounds is not free; validation
  loss must be monitored.
- **Many hyperparameters, high tuning effort** relative to Random Forest or AdaBoost.
- **Memory intensive** — stores all $T$ trees.
- **Not very interpretable** — hundreds of shallow trees, though feature importance (built-in or
  permutation) is still available.
- **Sensitive to outliers with MSE loss** — use Huber or MAE loss (see Pseudo-residuals table) for
  robustness, unlike plain squared-error gradient boosting.

## Real-world usage

- **State-of-the-art on tabular data** — often the best out-of-the-box performance among classical ML
  methods, ahead of both a single tree and Random Forest, at the cost of much more tuning effort.
- **Flexible loss functions** let it target the actual business metric more directly: MSE, MAE, Huber
  for regression; log loss or deviance for classification.
- **Robust to different feature scales, handles mixed data types** — tree-based, like its Random
  Forest and AdaBoost predecessors, no standardization required.
- **Good, built-in feature importance** — same mechanisms as Random Forest (mean decrease in impurity,
  permutation importance).
- In practice, prefer optimized implementations (XGBoost, LightGBM, CatBoost — see `14-xgboost`) over
  sklearn's `GradientBoostingClassifier`/`Regressor` for large datasets, for both speed and the extra
  regularization those libraries add.

## Mental model

Gradient boosting fits a sequence of models to the errors of the previous ensemble, using the
gradient of the loss to know which direction "fixing the error" means for any loss function — not
just squared error. AdaBoost is the special case where that loss is fixed to exponential loss and the
"error direction" collapses into reweighting examples; gradient boosting generalizes the same
sequential-correction idea to any differentiable loss by fitting the negative gradient directly.

## Questions to think about

1. For squared-error loss, the pseudo-residual is exactly the ordinary residual $y_i - F(x_i)$. Work
   out the pseudo-residual for absolute-error loss ($L = \lvert y_i - F\rvert$) from the Pseudo-
   residuals table — why does it only encode *direction* (sign) and not *magnitude* of the error, and
   what effect would that have on how aggressively the ensemble corrects large versus small errors
   compared to MSE loss?
2. `Why simpler approaches fail` argues AdaBoost's exponential-loss weight-update doesn't generalize.
   Given the general pseudo-residual formula $r_i^{(t)} = -[\partial L/\partial F]_{F=F^{(t-1)}}$,
   what pseudo-residual would you get by differentiating exponential loss $L = e^{-yF}$ itself — and
   in what sense is AdaBoost a special case of this framework rather than a genuinely different
   algorithm?
3. The learning-rate experiment showed `learning_rate=0.02` was still improving at round 150 while
   `learning_rate=0.5` had already started overfitting. If compute budget were unlimited, would you
   expect the 0.02 run to eventually reach a *lower* validation MSE than the 0.1 run's best (0.417),
   the same, or worse — and what would you need to check to find out?
4. Stochastic gradient boosting's `subsample` parameter and Random Forest's bootstrap sampling both
   involve training on a random subset of rows. Why does `11-random-forest`'s Mathematical foundation
   frame bootstrap sampling as *decorrelating independent trees*, while here subsampling is framed as
   *regularization* for a sequential ensemble — are these actually the same underlying effect, or two
   different mechanisms that happen to use similar-looking randomness?
5. Why does `max_depth` for gradient boosting trees stay shallow (3–8) while Random Forest trees are
   grown deep (often unlimited)? Connect your answer to which family (bagging vs. boosting) is
   fighting bias versus variance.
