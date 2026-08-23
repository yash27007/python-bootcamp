# XGBoost

## Problem

`13-gradient-boosting` gave a general recipe — fit each new tree to the negative gradient of any
differentiable loss — but nothing in that recipe stops the ensemble from growing arbitrarily complex
trees, and nothing in it uses more than first-order (gradient) information about the loss surface
when deciding how to fit each tree. At production scale that's a real problem: plain gradient
boosting is **slow to train** on large datasets (recomputing full residuals and searching all
possible splits, with no way to skip work) and **prone to overfitting** (no explicit penalty on tree
complexity — regularization is entirely implicit, via `max_depth`/`min_samples_leaf`/learning rate).
Production systems need speed, principled regularization, and — since real tabular data is rarely
complete — automatic handling of missing values. The problem: **make gradient boosting fast enough
and regularized enough for production use at scale.**

## Intuition

If sklearn's `GradientBoostingClassifier` is a regular car, XGBoost (eXtreme Gradient Boosting) is a
sports car built for the same road: same underlying idea (sequential trees fit to correct the
ensemble's errors), but engineered for speed and control. It adds an explicit "complexity budget" —
every extra leaf in a tree costs something, and every leaf's prediction is penalized for being too
large — so the algorithm is discouraged from growing needlessly complex trees even before you tune
`max_depth` by hand. And instead of only asking "which direction reduces the loss" (the gradient), it
also asks "how sharply curved is the loss right here" (the Hessian, the second derivative), so each
step is sized more intelligently instead of using one fixed learning rate everywhere.

## Why simpler approaches fail

Plain gradient boosting (`13-gradient-boosting`) has two structural gaps that matter at production
scale:

1. **No built-in regularization on tree complexity.** The objective it minimizes is just
   $\sum_i L(y_i, \hat y_i)$ — nothing in the objective itself penalizes a tree for having many
   leaves or large leaf values. All complexity control comes from *external* hyperparameters
   (`max_depth`, `min_samples_leaf`, learning rate) tuned by trial and error, with no principled
   connection between "how complex is this tree" and "how much does the objective actually charge for
   that complexity."
2. **First-order-only optimization and no principled handling of missing values or parallelism.**
   Fitting each tree to the negative gradient (a first-order quantity) ignores how the loss curves —
   two points with the same gradient but very different local curvature get treated identically, even
   though a Newton-style method would take very different step sizes for each. Sklearn's plain
   implementation also has no built-in strategy for missing values (they need imputation) and limited
   opportunities for parallelism beyond within-tree split search.

XGBoost is exactly gradient boosting with both gaps closed: an explicit complexity penalty added
*into* the objective, and a second-order (gradient + Hessian) approximation used for faster, more
principled split-finding — plus the systems engineering (sparsity-aware splits, approximate
quantile sketches, parallelized split search) that makes both of those practical at scale.

## Mathematical foundation

### The regularized objective

Vanilla gradient boosting minimizes $\sum_i L(y_i, \hat y_i)$. XGBoost adds a **complexity penalty**
directly into the objective when fitting the $t$-th tree $f_t$:
$$\mathcal{L}^{(t)} = \sum_{i=1}^n L\!\left(y_i,\ \hat y_i^{(t-1)} + f_t(x_i)\right) + \Omega(f_t)$$
$$\Omega(f_t) = \gamma T + \frac12 \lambda \sum_{j=1}^T w_j^2$$

where $T$ is the number of leaves in the new tree, $w_j$ is the score (output value) at leaf $j$,
$\gamma$ is a penalty per leaf (discourages bushy trees — every extra leaf must earn its keep), and
$\lambda$ is an L2 penalty on leaf scores (shrinks leaf weights, discourages overconfident
predictions). This is the explicit "complexity budget" from Intuition: $\gamma T$ charges for the
*number* of leaves, $\frac12\lambda\sum w_j^2$ charges for how *large* each leaf's prediction is.

### Second-order (Newton) approximation

To optimize this efficiently, XGBoost approximates the loss with a **second-order Taylor expansion**
around the current prediction $\hat y^{(t-1)}$:
$$L\!\left(y_i, \hat y^{(t-1)}+f_t(x_i)\right) \approx L\!\left(y_i,\hat y^{(t-1)}\right) + g_i f_t(x_i) + \frac12 h_i f_t(x_i)^2$$
where $g_i = \dfrac{\partial L(y_i,\hat y^{(t-1)})}{\partial \hat y^{(t-1)}}$ (gradient, first
derivative) and $h_i = \dfrac{\partial^2 L(y_i,\hat y^{(t-1)})}{\partial(\hat y^{(t-1)})^2}$ (Hessian,
second derivative).

**Why second-order beats first-order.** A first-order (gradient-only) method treats the loss as
locally *linear*: it knows which direction reduces the loss but has no information about how far to
go before that direction stops helping, so it must rely on an externally chosen, fixed learning rate
— too small and convergence is slow, too large and it overshoots in regions of high curvature. A
second-order method fits a local *quadratic* approximation (a parabola matching the loss's value,
slope, *and* curvature at the current point) and jumps straight to that parabola's minimum — in
regions of high curvature (steep, narrow loss) it automatically takes a small step; in regions of low
curvature (flat loss) it automatically takes a large step. This is a strictly better *local*
approximation of the true loss than a straight line, which is why using $h_i$ alongside $g_i$ gives
faster, more accurate convergence than gradient information alone.

For MSE loss: $g_i = \hat y^{(t-1)} - y_i,\ h_i = 1$. For log loss (binary classification):
$g_i = p_i - y_i,\ h_i = p_i(1-p_i)$, where $p_i = \sigma(\hat y^{(t-1)})$.

### Optimal leaf weights and the optimal tree score

Minimizing the (regularized, second-order-approximated) objective for a fixed tree structure, the
optimal weight for leaf $j$ is:
$$w_j^* = -\frac{\sum_{i\in I_j} g_i}{\sum_{i\in I_j} h_i + \lambda}$$
where $I_j$ is the set of training examples landing in leaf $j$. This is the Newton step from the
Why-second-order argument above, applied per leaf: the leaf weight is the (regularized) ratio of
accumulated gradient to accumulated Hessian, with $\lambda$ in the denominator shrinking the leaf
value toward zero (stronger regularization, more shrinkage).

Plugging the optimal leaf weights back in, the minimum loss for a given tree structure is:
$$\mathcal{L}^* = -\frac12 \sum_{j=1}^T \frac{\left(\sum_{i\in I_j} g_i\right)^2}{\sum_{i\in I_j} h_i + \lambda} + \gamma T$$

### The split gain formula

Deciding whether to split a node uses:
$$\text{Gain} = \frac12\left[\frac{\left(\sum_{i\in I_L} g_i\right)^2}{\sum_{i\in I_L}h_i+\lambda} + \frac{\left(\sum_{i\in I_R}g_i\right)^2}{\sum_{i\in I_R}h_i+\lambda} - \frac{\left(\sum_{i\in I}g_i\right)^2}{\sum_{i\in I}h_i+\lambda}\right] - \gamma$$
where $I_L,I_R$ are the left/right child example sets and $I$ is the parent. The first two terms
score the children; the third scores the (unsplit) parent; $\gamma$ is the cost of creating a new
leaf. **A split is only made if `Gain > 0`** — if no split produces positive gain, the tree stops
growing there. A higher $\gamma$ requires a larger gain to justify any split, producing simpler trees
(this is what the `gamma`/`min_split_loss` hyperparameter controls).

## Algorithm

```
INPUT: Training data (X, y), loss L, regularization params (gamma, lambda), T rounds

F_0(x) = initial prediction (e.g. mean(y) for MSE)

FOR t = 1 to T:
    1. Compute g_i = dL(y_i, F_{t-1}(x_i))/dF, h_i = d^2L(y_i, F_{t-1}(x_i))/dF^2  for all i
    2. Greedily grow a tree f_t, choosing splits that maximize the Gain formula
       (a split with Gain <= 0 is not made; the node becomes a leaf)
    3. Assign each leaf j the optimal weight w_j* = -sum(g_i in I_j) / (sum(h_i in I_j) + lambda)
    4. Update: F_t(x) = F_{t-1}(x) + eta * f_t(x)

RETURN F_T
```

### Tree building methods

- **Exact greedy:** for each feature, sort all examples by value, scan every possible split point.
  Guaranteed optimal but $O(n \cdot d \cdot \log n)$ — too slow for millions of examples.
- **Approximate (weighted quantile sketch):** instead of $N-1$ split candidates, use an
  $\epsilon$-approximate quantile sketch to generate $O(1/\epsilon)$ candidates (typically
  $\epsilon\approx0.01$, ~100 candidates per feature). This is what lets XGBoost scale to millions of
  rows.
- **Sparsity-aware split finding:** for missing values / sparse features, XGBoost learns a **default
  direction** (left or right) per split — computing gain for both possible default directions and
  keeping the better one — avoiding explicit imputation and sometimes improving performance when
  missingness itself is informative.

### Column and row subsampling

Borrowed from Random Forest (`11-random-forest`) to decorrelate trees and speed up training:
`colsample_bytree` (sample features once per tree), `colsample_bylevel` (per depth level),
`colsample_bynode` (per split — most like Random Forest's `max_features`), `subsample` (sample rows
per tree). These add stochasticity that both speeds up training and reduces overfitting.

### System optimizations

| Optimization | Description |
|---|---|
| Cache-aware access | Feature values stored in blocks sorted by value → CPU cache hits during scanning |
| Out-of-core computation | Reads from disk in blocks when the dataset doesn't fit in RAM |
| Parallelized tree building | Split-finding across features parallelized (not between trees) |
| Column block | Pre-sorted feature columns stored in compressed blocks → fast column access |

These engineering details, on top of the second-order math above, make XGBoost 10–100x faster than
sklearn's GBM on large datasets — see Experiment below for a smaller-scale, directly measured
comparison.

## From-scratch implementation

Not a full reimplementation — reimplementing XGBoost's tree-building machinery is out of scope (see
the repo's from-scratch guidance: don't reimplement mature production systems for their own sake).
Instead, `05-machine-learning/14-xgboost/newton-step-and-xgboost-vs-gb.ipynb` isolates the one idea
that most distinguishes XGBoost's optimization from plain gradient boosting: **first-order vs.
second-order steps**, on a simple 1D loss curve $L(x) = 0.05(x-4)^4 + (x-4)^2$ (single minimum at
$x=4$, curvature varying with $x$):

```python
def first_order_steps(x0, eta, n_steps):
    xs, x = [x0], x0
    for _ in range(n_steps):
        x = x - eta * grad(x)          # fixed learning rate — analogous to plain gradient boosting
        xs.append(x)
    return np.array(xs)

def second_order_steps(x0, n_steps):
    xs, x = [x0], x0
    for _ in range(n_steps):
        x = x - grad(x) / hess(x)      # Newton step — analogous to XGBoost's w_j* = -sum(g)/(sum(h)+lambda)
        xs.append(x)
    return np.array(xs)
```

Starting both at $x_0=0$ for 6 steps: the first-order trajectory (learning rate 0.05) reaches
$x\approx2.67$, still 1.342 away from the true minimum at $x\approx4.02$; the second-order (Newton)
trajectory reaches $x=4.00$, only 0.015 away — landing almost exactly on the minimum in the same
number of steps, because each Newton step automatically rescales by local curvature instead of using
one fixed step size everywhere. This is a direct, minimal illustration of why using the Hessian
($h_i$) alongside the gradient ($g_i$) in the split-finding math above gives XGBoost better local
approximations and faster convergence than a gradient-only approach.

## Practical implementation

```python
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = xgb.XGBClassifier(
    n_estimators=2000,          # will early-stop before this
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=3,
    gamma=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_lambda=1.0,
    reg_alpha=0.1,
    scale_pos_weight=1,         # adjust for imbalanced data
    eval_metric='auc',
    early_stopping_rounds=50,
    random_state=42,
    n_jobs=-1
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)

print(f"Best iteration: {model.best_iteration}")
print(f"Best val AUC: {model.best_score:.4f}")
y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]
print(classification_report(y_val, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_val, y_prob):.4f}")
```

For maximum performance, XGBoost's native `DMatrix` API:

```python
dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_val, label=y_val)

params = {
    'objective': 'binary:logistic', 'eval_metric': 'auc',
    'eta': 0.05, 'max_depth': 6, 'min_child_weight': 3,
    'subsample': 0.8, 'colsample_bytree': 0.8,
    'lambda': 1.0, 'alpha': 0.1, 'nthread': -1, 'seed': 42
}
evals_result = {}
booster = xgb.train(
    params, dtrain, num_boost_round=2000,
    evals=[(dtrain, 'train'), (dval, 'val')],
    early_stopping_rounds=50, evals_result=evals_result, verbose_eval=100
)
```

`05-machine-learning/14-xgboost/XgboostBoost-Classification-Implementation.ipynb` (travel-package
purchase prediction) and `Xgboost-Regression-Implementation.ipynb` (used-car price prediction) apply
this to the same two real end-to-end projects used throughout `11-random-forest`–`13-gradient-boosting`,
directly connecting back to the from-scratch step above: every split those notebooks' trees make
internally uses the same $g_i$/$h_i$-based gain formula and Newton-style leaf weights derived in
Mathematical foundation, not a from-scratch reimplementation but the identical mathematical mechanism
at production speed.

### Understanding `min_child_weight`

One of the most impactful hyperparameters: the **minimum sum of instance weights (Hessian)** required
in a child node. For MSE loss ($h_i=1$), this is just the minimum number of samples in a leaf. For
log loss, $h_i=p_i(1-p_i)\le0.25$, so `min_child_weight=1` requires the sum of $p_i(1-p_i)$ to exceed
1 — roughly 4+ examples with mid-range probabilities. Higher `min_child_weight` → fewer splits → less
overfitting → simpler model.

### Hyperparameters — reference

| Parameter | Description | Default | Typical range |
|---|---|---|---|
| `n_estimators` | Number of boosting rounds | 100 | 100–5000 |
| `learning_rate` (eta) | Step size shrinkage | 0.3 | 0.01–0.3 |
| `max_depth` | Maximum tree depth | 6 | 3–10 |
| `min_child_weight` | Min sum of Hessian in a child | 1 | 1–10 |
| `gamma` (min_split_loss) | Min gain to split a node | 0 | 0–5 |
| `subsample` | Row sampling fraction | 1.0 | 0.5–0.9 |
| `colsample_bytree` | Feature sampling fraction per tree | 1.0 | 0.5–0.9 |
| `colsample_bylevel` | Feature sampling per level | 1.0 | 0.5–0.9 |
| `lambda` (reg_lambda) | L2 regularization on leaf weights | 1 | 1–10 |
| `alpha` (reg_alpha) | L1 regularization on leaf weights | 0 | 0–1 |
| `scale_pos_weight` | Balance for imbalanced classes | 1 | sum(neg)/sum(pos) |

**Sequential tuning strategy (recommended for production):** (1) fix `learning_rate=0.1`, find
optimal `n_estimators` via early stopping; (2) tune `max_depth` and `min_child_weight`; (3) tune
`gamma`; (4) tune `subsample`/`colsample_bytree`; (5) tune `reg_lambda`/`reg_alpha`; (6) reduce
`learning_rate`, increase `n_estimators`, early-stop again. Alternatively, automate the search with
Optuna (`suggest_int`/`suggest_float` over the ranges above, `cross_val_score` as the objective).

### Feature importance

| Importance type | What it measures | Use case |
|---|---|---|
| `weight` | How many times a feature is used to split | Quick scan |
| `gain` | Average gain (loss reduction) when feature is used | Most informative |
| `cover` | Average coverage (sum of Hessians) of feature splits | Data coverage |
| Permutation | Drop in score when feature is shuffled | Most reliable |

```python
xgb.plot_importance(model, importance_type='gain', max_num_features=20)
importance_dict = model.get_booster().get_score(importance_type='gain')

import shap
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_val)
shap.summary_plot(shap_values, X_val)
```

### XGBoost vs. LightGBM vs. CatBoost

| Aspect | XGBoost | LightGBM | CatBoost |
|---|---|---|---|
| Tree growth | Level-wise (BFS) | Leaf-wise | Symmetric (oblivious) |
| Speed | Fast | Fastest on large data | Medium |
| Memory | Medium | Low | Medium |
| Categorical features | Manual encoding needed | Label encoding, no need to one-hot | Native, no encoding |
| Overfitting | Good | More prone (leaf-wise) | Best regularization |
| Best at | General purpose | Large datasets | Data with many categoricals |
| GPU support | Yes | Yes | Yes |

**Leaf-wise vs. level-wise tree growth:** level-wise (XGBoost) grows all leaves at the same depth —
more uniform, less prone to overfitting; leaf-wise (LightGBM) always splits the leaf with the highest
gain — faster convergence but can overfit more easily on small data.

### XGBoost for regression

`XGBRegressor` takes the same core machinery with a regression objective, e.g.
`objective='reg:squarederror'` (MSE), `'reg:absoluteerror'` (MAE, outlier-robust), or
`'reg:squaredlogerror'` (RMSLE, for strictly positive targets):

```python
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000, learning_rate=0.05, max_depth=6,
    early_stopping_rounds=50
)
model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=100)
```

### Common mistakes

| Mistake | Why it's bad | Fix |
|---|---|---|
| Not using early stopping | Overfits, wastes time | Always set `early_stopping_rounds` |
| Too-high learning rate (0.3+) | Unstable, overfits | Start with 0.05–0.1 |
| Not handling class imbalance | Poor recall on minority class | Set `scale_pos_weight = sum(neg)/sum(pos)` |
| Not encoding categoricals | XGBoost can't use strings | Label encode or use CatBoost |
| Tuning all hyperparams at once | Slow, confusing | Tune sequentially (depth → regularization → learning rate) |
| Large `max_depth` (>8) | Severe overfitting | Keep at 4–6 for most datasets |

## Experiment

`newton-step-and-xgboost-vs-gb.ipynb`'s `GradientBoostingClassifier` vs. `XGBClassifier` comparison.

**Hypothesis (stated before running):** on the same dataset/split, `XGBClassifier` should train
*faster* than sklearn's `GradientBoostingClassifier` (optimized split-finding, `tree_method='hist'`)
and generalize *at least as well*, thanks to the regularization term $\Omega(f)$ that plain gradient
boosting has no equivalent of.

**Setup:** `make_classification` (4000 samples, 20 features, 12 informative), 80/20 train/test split,
identical `n_estimators=300, learning_rate=0.1, max_depth=3` for both models. First run used default
XGBoost settings (`n_jobs=-1`, default `tree_method`); a follow-up run explicitly set
`tree_method='hist'` (XGBoost's histogram-based fast split finder) and `n_jobs=1`.

**Result:**

| Configuration | Training time | Test accuracy |
|---|---|---|
| `GradientBoostingClassifier` | 3.59s | 0.9387 |
| `XGBClassifier`, default settings, `n_jobs=-1` | 38.15s | 0.9425 |
| `XGBClassifier`, `tree_method='hist'`, `n_jobs=1` | 0.52s | 0.9425 |

**Interpretation:** with default settings, XGBoost was *slower* than plain gradient boosting on this
run — the opposite of the hypothesis — because `n_jobs=-1` spawns worker threads whose overhead
exceeds the benefit on a dataset this small (4000 rows × 20 features), and the default tree-building
method wasn't the fast histogram-based one. With `tree_method='hist'` and `n_jobs=1` explicitly set,
XGBoost was ~7x faster than `GradientBoostingClassifier` (0.52s vs. 3.59s) at slightly *better*
accuracy (0.9425 vs. 0.9387) — confirming the hypothesis once XGBoost is configured the way it's
actually meant to be used at scale. This is itself a useful, honest finding: XGBoost's speed
advantage is not automatic from the default constructor call, it depends on using the histogram
split method it was built around, and on not over-parallelizing a small workload.

**Limitations:** single synthetic dataset/split; identical `n_estimators`/`learning_rate`/`max_depth`
for both models rather than each independently tuned; timing is hardware- and thread-count-dependent
and will vary run to run; only one dataset size tested, so the crossover point where `n_jobs=-1`
overhead stops dominating was not characterized.

## Failure modes

- **Many hyperparameters to tune** — `max_depth`, `min_child_weight`, `gamma`, `subsample`,
  `colsample_*`, `reg_lambda`, `reg_alpha`, `learning_rate`, `n_estimators` all interact; sequential
  tuning (see Practical implementation) is needed to make this tractable.
- **Can still overfit** with too many rounds and too little regularization (`gamma`, `reg_lambda`,
  `reg_alpha` all near 0) — the regularized objective raises the bar but doesn't eliminate the risk,
  and `early_stopping_rounds` is still needed in practice.
- **Less interpretable than a single tree** — hundreds of trees, though `gain`/`cover`/permutation
  importance and SHAP values partially recover interpretability.
- **The speed advantage isn't automatic** — as the Experiment above shows directly, default settings
  (`n_jobs=-1` on a small dataset, non-histogram tree method) can be *slower* than plain gradient
  boosting; getting XGBoost's real speed requires configuring it for the workload's actual scale.

## Real-world usage

- **The default choice for tabular data competitions** and many production tabular pipelines — the
  combination of regularization, second-order optimization, and engineering (sparsity-aware splits,
  approximate quantile sketches, cache-aware access, parallelized split-finding) has made it a
  long-standing standard for structured/tabular problems.
- **Native missing-value handling** (sparsity-aware split finding) is a genuine practical advantage
  over Random Forest and plain sklearn gradient boosting, which need imputation.
- **Early stopping with a validation set** is the standard way to pick `n_estimators` automatically
  rather than tuning it by hand.
- In practice often compared against LightGBM (faster on very large data, leaf-wise growth) and
  CatBoost (best native categorical handling) — see the comparison table above for when each wins.

## Mental model

XGBoost is gradient boosting with a built-in complexity budget and a smarter (second-order) way of
deciding each step: the same "fit each new tree to correct the ensemble's errors" idea from
`13-gradient-boosting`, but the objective itself charges for tree complexity ($\Omega(f) = \gamma T +
\frac12\lambda\|w\|^2$), and each step uses both the gradient and the curvature (Hessian) of the loss
instead of a single fixed learning rate — landing closer to the true minimum in fewer, better-chosen
steps.

## Questions to think about

1. In the split gain formula, what happens to `Gain` as $\lambda \to \infty$ for a fixed candidate
   split, and why does that match the intuition that $\lambda$ "shrinks leaf weights toward zero"?
2. The from-scratch Newton-step illustration showed the second-order step landing almost exactly on
   the true minimum in 6 steps, while the first-order step was still 1.34 away. If the loss curve had
   *constant* curvature everywhere (a pure parabola, not the quartic used here), would you expect the
   gap between first-order and second-order convergence to be larger, smaller, or the same — why?
3. `min_child_weight` is described as "the minimum sum of Hessian in a child." For log loss, why does
   a batch of examples with predicted probabilities all near 0 or all near 1 (very confident
   predictions) contribute *less* total Hessian than the same number of examples with predictions
   near 0.5 — and what does that imply about which examples `min_child_weight` effectively protects
   against being split on further?
4. The Experiment section found XGBoost's default settings were *slower* than plain gradient boosting
   on a 4000-row dataset, but ~7x faster with `tree_method='hist'` and `n_jobs=1`. At what rough
   dataset size (rows × features) would you expect `n_jobs=-1`'s thread-spawn overhead to become
   worth paying — and how would you design an experiment to find that crossover point?
5. `13-gradient-boosting`'s Mental model frames gradient boosting as using "the gradient of the loss
   to know which direction fixing the error means." How does this Mental model's "second-order...
   smarter way of deciding each step" change that story — is XGBoost fixing a *different* problem
   than gradient boosting, or the *same* problem with a better local approximation? What would have to
   be true about a loss function for the first-order and second-order approaches to converge to
   exactly the same place?
