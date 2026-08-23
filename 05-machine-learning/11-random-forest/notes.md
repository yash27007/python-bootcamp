# Random Forest

## Problem

A single decision tree is unstable: it has **high variance**. Train it on the full dataset, then
train an identical tree (same algorithm, same hyperparameters) on the same dataset with just a
handful of rows changed, and the resulting tree can look completely different — a different feature
wins at the root, and that single change cascades into a different structure below it. A model whose
predictions swing that much based on which exact rows happened to be in the training set is not a
model you can trust in production, even if its *average* behavior over many possible training sets
would be fine. The problem: **how do we keep a decision tree's low bias (it can fit complex,
non-linear, non-parametric relationships) while getting rid of its high variance?**

## Intuition

Imagine you need to decide whether to invest in a stock. Instead of asking one expert (who might
give a biased or noisy answer), you ask 500 different experts — each with a slightly different
background, each having read a random subset of the available news articles — and take a vote. Any
one expert might be wrong in an idiosyncratic way, but idiosyncratic errors tend to cancel out when
you average across many independent opinions, leaving the shared signal. This is exactly how
**Random Forest** works: build many decision trees, each trained on a randomly different view of the
data, and combine their predictions. The result is almost always more stable than any single tree.

## Why simpler approaches fail

The obvious first idea — "if one tree overfits, just train many trees and average them" — fails in
the most naive form: **training multiple trees on the exact same data just reproduces the same
tree.** Decision tree growing is deterministic given the data and hyperparameters (mod tie-breaking):
if you fit `DecisionTreeClassifier` on identical `(X, y)` ten times with the same settings, you get
ten identical trees. Averaging ten copies of the same model doesn't reduce variance at all — it's
still exactly as unstable as the one tree, because there is no diversity between the "many" models to
average away. Something has to make the trees see *different* data or *different* features, or the
whole ensembling idea does nothing.

## Mathematical foundation

### Bias-variance recap

A model's expected test error decomposes as:
$$\text{Total Error} = \text{Bias}^2 + \text{Variance} + \text{Irreducible Noise}$$

| Model | Bias | Variance | Problem |
|-------|------|----------|---------|
| Deep decision tree | Low | High | Overfits |
| Shallow decision tree | High | Low | Underfits |
| Random forest | Low | **Low** | Best of both worlds |

Random Forest's goal is exactly the rightmost row: keep a deep tree's low bias while pulling its
variance down toward that of a much simpler model — see `05-machine-learning/05b-bias-variance-tradeoff`
for the full derivation of this decomposition; here we take it as given and focus on *why averaging
trees reduces the variance term*.

### Why averaging many estimators reduces variance

Suppose we have $B$ estimators, each an unbiased predictor of the true value with variance
$\sigma^2$. If the estimators were **independent and identically distributed (i.i.d.)**, the
variance of their average is:
$$\text{Var}\left(\frac{1}{B}\sum_{b=1}^B \hat{y}_b\right) = \frac{1}{B^2}\sum_{b=1}^B \text{Var}(\hat{y}_b) = \frac{\sigma^2}{B}$$

With $B=500$ independent trees, variance would drop to $\sigma^2/500$ — a huge reduction — while
**bias is unchanged**: the average of unbiased estimators is still unbiased, since
$\mathbb{E}\left[\frac{1}{B}\sum \hat{y}_b\right] = \mathbb{E}[\hat{y}]$ regardless of $B$. This is
the whole appeal: averaging trades nothing in bias for a large reduction in variance, *provided the
independence assumption holds*.

It doesn't, fully — trees trained on the same underlying dataset share information and are
correlated. For $B$ estimators with pairwise correlation $\rho$ and individual variance $\sigma^2$,
the exact variance of the average is:
$$\text{Var}(\text{ensemble}) = \rho \sigma^2 + \frac{1-\rho}{B}\sigma^2$$

As $B \to \infty$, the second term vanishes, but the first term $\rho \sigma^2$ is **irreducible** —
no amount of averaging removes the part of the variance that all the trees share. This is why the
"why simpler approaches fail" story matters mathematically, not just intuitively: at $\rho=1$ (all
trees identical), $\text{Var}(\text{ensemble}) = \sigma^2$ — zero improvement, matching the "same
data, same tree" failure above. The entire value of Random Forest lies in pushing $\rho$ down.

| Scenario | $\rho$ | Ensemble variance |
|----------|--------|--------------------|
| All same tree ($\rho=1$) | 1.0 | $\sigma^2$ (no improvement) |
| Independent trees ($\rho=0$) | 0 | $\sigma^2/B$ (huge improvement) |
| Random Forest ($\rho \approx 0.1$–$0.3$) | ~0.2 | $\approx 0.2\sigma^2 + \sigma^2/B$ |

### Decorrelating the trees: bagging + feature subsampling

Two sources of randomness push $\rho$ down toward 0 without needing truly independent datasets:

**1) Bootstrap sampling (bagging).** For each of the $B$ trees, draw $N$ examples **with
replacement** from the $N$-example training set. The probability a specific example is *not* drawn
in one draw is $1 - 1/N$; after $N$ draws, the probability it is never selected is
$(1-1/N)^N \to e^{-1} \approx 0.368$ as $N \to \infty$. So each bootstrap sample contains, on
average, $1 - 0.368 = 0.632 = 63.2\%$ of the original examples (with some repeated), and the other
$\approx 36.8\%$ are **out-of-bag (OOB)** for that tree. Different random draws give each tree a
genuinely different (though overlapping) training set, which is enough to make different trees
plausible — unlike training on identical data.

*Worked example:*

| Original Training Set | Bootstrap Sample 1 | Bootstrap Sample 2 |
|----------------------|---------------------|---------------------|
| A, B, C, D, E        | A, A, C, D, C (B, E are OOB) | B, D, E, E, A (C is OOB) |

**2) Random feature subsampling at each split.** Bagging alone still leaves trees correlated if one
feature is dominant: every bootstrap sample still contains that feature, so every tree's root (and
most other splits) picks it anyway, keeping $\rho$ high. Random Forest adds a second layer: at
**each node** of each tree, instead of searching all $p$ features for the best split, only a random
subset of $m$ features is considered. Default rules of thumb: $m = \sqrt{p}$ for classification,
$m = p/3$ for regression. By hiding the dominant feature at some nodes, weaker features get a chance
to define splits, producing structurally different trees — this is what actually pushes $\rho$
toward the ~0.1–0.3 range instead of staying near 1.

### Aggregating predictions

**Classification:** majority vote across all trees.
$$\hat{y} = \text{mode}(\hat{y}_1, \hat{y}_2, \dots, \hat{y}_B)$$

**Regression:** average across all trees.
$$\hat{y} = \frac{1}{B} \sum_{b=1}^B \hat{y}_b$$

### Out-of-bag (OOB) error

Since each tree trains on only ~63.2% of the data, the ~36.8% OOB for that tree can be used to
estimate generalization error without a separate validation set:
$$\text{OOB Error} = \frac{1}{N} \sum_{i=1}^N \mathbb{1}\!\left[\hat{y}_i^{OOB} \neq y_i\right]$$

where $\hat{y}_i^{OOB}$ aggregates predictions only from the trees for which example $i$ was OOB.
This acts like built-in cross-validation, correlates well with held-out test error in practice, and
wastes no data on an explicit validation split.

## Algorithm

```
INPUT: Training data (X, y), number of trees B, features per split m

FOR b = 1 to B:
    1. Draw bootstrap sample D_b from (X, y) — N samples with replacement
    2. Build a decision tree on D_b:
       - At each node:
         a. Randomly select m features from p total features
         b. Find the BEST split among these m features (max Gini/entropy reduction, or min MSE)
         c. Split the node
       - Grow tree until: leaf is pure, or min_samples_leaf reached, or max_depth hit
    3. Store tree b

PREDICTION:
    - For each tree b, get prediction ŷ_b for new point x
    - Classification: return the majority class vote
    - Regression: return the mean of all ŷ_b
```

## From-scratch implementation

`05-machine-learning/11-random-forest/bagging-variance-from-scratch.ipynb` isolates the bagging half
of this mechanism (bootstrap resampling + averaging, without feature subsampling), using sklearn's
`DecisionTreeClassifier` as the base learner — the point being demonstrated is the *ensembling*
mechanism, not re-deriving tree-splitting (already covered from scratch in `10-decision-tree`).

```python
def bootstrap_sample(X, y, rng):
    idx = rng.randint(0, len(y), size=len(y))
    return X[idx], y[idx]

def bagged_predict_proba(X_train, y_train, X_query, B, rng):
    proba_sum = np.zeros(len(X_query))
    for _ in range(B):
        Xb, yb = bootstrap_sample(X_train, y_train, rng)
        tree = DecisionTreeClassifier(random_state=rng.randint(1_000_000))
        tree.fit(Xb, yb)
        proba_sum += tree.predict_proba(X_query)[:, 1]
    return proba_sum / B
```

On a toy `make_moons` dataset, the notebook fits bagged ensembles for
$B \in \{1, 2, 5, 10, 25, 50, 100\}$, repeating the *entire* bagging procedure 25 independent times
per $B$ and measuring the variance of the ensemble's predicted probability at a fixed set of query
points across those repeats (see Experiment below for the numbers). This is a direct, measured
demonstration of the $\text{Var}(\text{average of }B) $ shrinkage argument derived above.

## Practical implementation

`sklearn.ensemble.RandomForestClassifier` / `RandomForestRegressor` implement the full algorithm
above (bagging **and** feature subsampling, plus OOB scoring and feature importances) in optimized
code. This is exactly the from-scratch bagging loop, with two additions the from-scratch step
deliberately left out: (1) `max_features` implements the random feature subsampling at each split
that decorrelates the trees beyond what bagging alone achieves, and (2) `oob_score=True` computes the
OOB error formula above automatically instead of needing a held-out set.

```python
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

rf = RandomForestClassifier(
    n_estimators=300,
    max_features='sqrt',
    max_depth=None,
    min_samples_leaf=1,
    oob_score=True,
    n_jobs=-1,
    random_state=42,
)
rf.fit(X_train, y_train)

print(f"OOB Accuracy: {rf.oob_score_:.4f}")
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))
```

`05-machine-learning/11-random-forest/random-forest-classification.ipynb` (travel-package purchase
prediction) and `random-forest-regression.ipynb` (used-car price prediction) apply this to two real
end-to-end projects, including data cleaning, encoding, hyperparameter search
(`GridSearchCV` over `n_estimators`, `max_depth`, `max_features`, `min_samples_leaf`), and feature
importance plots.

### Feature importance

Random Forest gives a natural measure of which features matter:

**Mean Decrease in Impurity (MDI / Gini importance):** for each feature $j$, sum how much it reduces
impurity (weighted by node sample count) across all trees and all splits on feature $j$, then
normalize so importances sum to 1:
$$\text{Importance}(j) = \frac{1}{B} \sum_{b=1}^B \sum_{\text{nodes splitting on } j} \frac{N_{node}}{N} \Delta \text{Impurity}$$

where $\Delta \text{Impurity} = \text{Impurity}_{parent} - \frac{N_{left}}{N_{node}} \text{Impurity}_{left} - \frac{N_{right}}{N_{node}} \text{Impurity}_{right}$
is the impurity reduction that node's split achieved.

MDI is fast but biased toward high-cardinality features (e.g. a random ID column can look important
just because it can always create a perfect split).

**Permutation importance (Mean Decrease in Accuracy):** evaluate the model, get baseline score $S_0$;
shuffle feature $j$'s values and re-evaluate to get $S_j$; importance is $S_0 - S_j$, averaged over
$K$ shuffles for stability:
$$\text{Permutation Importance}(j) = S_0 - \frac{1}{K}\sum_{k=1}^K S_{j,k}$$

Permutation importance is slower but unbiased by cardinality and works for any model, not just trees.

| Feature importance type | Fast? | Biased? | Works post-fit? |
|--------------------------|-------|---------|------------------|
| MDI (Gini importance) | Yes | Yes (high cardinality) | Yes |
| Permutation importance | No | No | Yes (any model) |

### Hyperparameters

| Hyperparameter | Controls | Default | Effect |
|-----------------|----------|---------|--------|
| `n_estimators` | Number of trees | 100 | More = better (diminishing returns + slower) |
| `max_depth` | Maximum tree depth | None (full) | Limit to reduce overfitting |
| `max_features` | Features per split | `sqrt(p)` | Lower → more diverse trees, but weaker individual trees |
| `min_samples_split` | Min samples to split a node | 2 | Higher → simpler trees (less overfitting) |
| `min_samples_leaf` | Min samples in a leaf | 1 | Higher → smoother predictions |
| `bootstrap` | Whether to use bootstrap sampling | True | False = use all data (bagging off) |
| `oob_score` | Whether to compute OOB score | False | True = free validation |
| `class_weight` | Weights for imbalanced classes | None | `'balanced'` adjusts for class imbalance |

**Tuning strategy:** start with `n_estimators=100` and increase until the OOB error plateaus
(usually 200–500 is enough — more trees essentially never hurt accuracy, only training time); sweep
`max_depth` (`None`, 5, 10, 20) and `max_features` (`sqrt(p)`, `log2(p)`, `0.33*p`) via OOB score or
cross-validation:

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'n_estimators': [100, 300, 500],
    'max_depth': [None, 10, 20],
    'max_features': ['sqrt', 'log2'],
    'min_samples_leaf': [1, 2, 4],
}
rf = RandomForestClassifier(oob_score=True, random_state=42, n_jobs=-1)
grid = GridSearchCV(rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
grid.fit(X_train, y_train)
```

## Experiment

This is the bootstrap-variance-vs-$B$ curve from the from-scratch step
(`bagging-variance-from-scratch.ipynb`).

**Hypothesis (stated before running):** a single tree's (B=1) prediction at a fixed query point
should vary substantially across independent bagging runs, since each run resamples the training
data differently. As $B$ grows, the ensemble's averaged prediction should vary less run to run, with
variance shrinking roughly like $\sigma^2/B$ for small $B$, then flattening as the correlation floor
$\rho \sigma^2$ from the Mathematical foundation starts to dominate.

**Setup:** `make_moons(n_samples=200, noise=0.35)` toy dataset, 30 fixed query points. For each
$B \in \{1, 2, 5, 10, 25, 50, 100\}$, repeat the whole bagging procedure (fresh bootstrap draws, fresh
trees) 25 times, recording the ensemble's predicted $P(\text{class}=1)$ at every query point each
repeat, then computing the variance across the 25 repeats (averaged over query points).

**Result:**

| $B$ | Mean prediction variance |
|-----|---------------------------|
| 1 | 0.04064 |
| 2 | 0.02896 |
| 5 | 0.01018 |
| 10 | 0.00585 |
| 25 | 0.00187 |
| 50 | 0.00105 |
| 100 | 0.00055 |

**Interpretation:** variance drops by roughly 74× from $B=1$ to $B=100$ (0.04064 → 0.00055), closely
tracking the $\sigma^2/B$ reference for small $B$ and only mildly exceeding it by $B=100$
(measured 0.00055 vs. reference $\sigma^2/100 \approx 0.00041$) — a small but visible correlation
floor, consistent with bagging alone (no feature subsampling, since this toy set only has 2 features)
leaving $\rho$ modestly above 0.

**Limitations:** single toy dataset, single noise level, only 2 features (so feature subsampling has
almost nothing to add here — $m=\sqrt{2}\approx 1.4$), and 25 repeats gives a noisy variance
estimate rather than a tight confidence interval. The OOB-error-vs-$n\_estimators$ code shown in
`Practical implementation`'s companion snippet (looping `RandomForestClassifier(n_estimators=n,
oob_score=True)` for `n` in `1..200` and plotting `1 - oob_score_`) is the analogous experiment for
the full Random Forest (bagging + feature subsampling) on a real dataset, for anyone extending this
notebook.

## Failure modes

- **Loses interpretability vs. a single tree.** A single tree can be printed and read; 300 trees
  cannot — Random Forest trades the auditability of a single decision tree (see `10-decision-tree`'s
  real-world usage) for stability. Feature importance is the main window back into what it learned.
- **Correlated trees don't reduce variance as much as expected.** If `max_features` is set too high
  (too few features excluded per split) or the dataset lacks real diversity (near-duplicate rows,
  very few dominant features), $\rho$ stays high and the ensemble variance stays close to $\sigma^2$
  regardless of $B$ — see the correlation-variance table in Mathematical foundation.
  Too few features sampled per split, conversely, weakens each individual tree (higher bias) even as
  it lowers $\rho$ — `max_features` is a bias/decorrelation tradeoff, not a free lunch.
  This is exactly what the Experiment's limitation section flags: with only 2 features, feature
  subsampling barely operates.
- **Can still overfit despite averaging.** Averaging reduces variance but does not touch bias; if
  every individual tree is grown very deep with almost no regularization (`min_samples_leaf=1`,
  unlimited depth) on a small or noisy dataset, the ensemble can still overfit — it will just overfit
  *less* than any one of its constituent trees, not zero.
- **Slow prediction and high memory** — every prediction requires running the input through all $B$
  trees, and all $B$ trees must be stored.
- **Poor for very sparse, high-dimensional data** (e.g. raw text/TF-IDF) — linear models or SVMs
  often outperform tree ensembles there, since axis-aligned splits don't exploit sparse, near-linear
  structure well.
- **Poor extrapolation for regression** — like a single tree, predictions are bounded by the range of
  leaf target values seen in training, since every leaf's prediction is a mean over training points.
- **Slower to train than a single tree** — training $B$ trees costs roughly $B\times$ a single tree's
  training time (mitigated by `n_jobs=-1`, since the trees are independent and train in parallel).
- **Missing values still generally need imputation** in scikit-learn's implementation (unlike
  boosted-tree libraries such as XGBoost/LightGBM, which handle them natively).

**Common mistakes:**

| Mistake | Why it's bad | Fix |
|---------|---------------|-----|
| Using only accuracy on imbalanced data | Misleading metric | Use `class_weight='balanced'` + F1/AUC |
| Too few trees (`n_estimators=10`) | High variance, unstable | Use ≥100; monitor OOB error plateau |
| Normalizing/scaling features | Wastes effort — makes no difference for tree ensembles | Skip scaling for tree-based models |
| Treating the forest as a black box | Misses available insight | Check feature importances + OOB error |
| Over-constraining `max_depth` | Fights the ensemble's own variance control | Let trees grow deep; let averaging control variance |

## Real-world usage

- **Default strong baseline for tabular data** — before reaching for gradient boosting or deep
  learning, a Random Forest with `n_estimators=300`, `max_features='sqrt'`, no depth limit is a
  reliable first model for classification/regression on structured/tabular datasets.
- **OOB score as a free validation signal** during rapid iteration, when a held-out set is expensive
  or data is limited.
- **Feature importance for exploratory analysis and feature selection**, especially early in a
  project before committing to a final model family.
- **Risk/uncertainty estimation in regression** — the standard deviation across tree predictions,
  $\text{std}(\hat{y}_1(x), \dots, \hat{y}_B(x))$, is a usable per-prediction uncertainty estimate:
  high spread flags points the model is unsure about (e.g. extrapolation, sparse regions of feature
  space).
- **Robust to outliers** — tree splits are based on ordering samples, not on distances/magnitudes, so
  extreme values distort a Random Forest far less than they distort linear/distance-based models.
- **Handles mixed feature types** — like its constituent trees, a random forest works directly with
  both numerical and categorical data without special preprocessing.
- **Handles high dimensionality reasonably well** — feature subsampling means no single tree needs to
  search all $p$ features, which helps when $p$ is large relative to $n$.
- **Trains in parallel** — because the $B$ trees are trained independently, fitting scales across CPU
  cores (`n_jobs=-1`), unlike sequential ensembles such as boosting.

### Single tree vs. random forest, at a glance

| Aspect | Decision tree | Random forest |
|--------|----------------|----------------|
| Variance | High | Low |
| Bias | Low (if deep) | Low |
| Overfitting | Severe | Much less |
| Interpretability | High (visualizable) | Low (black box) |
| Training speed | Fast | Slower ($\approx B\times$, parallelizable) |
| Prediction speed | Fast | Slower (runs all $B$ trees) |
| Feature importance | Basic (single tree's splits) | Robust (averaged over $B$ trees + OOB) |
| Missing values | Need imputation | Need imputation (scikit-learn)$^\dagger$ |

$^\dagger$ Correction from the original version of these notes, which claimed Random Forest "can
handle [missing values] approximately." Scikit-learn's `RandomForestClassifier`/`Regressor` do not
natively handle missing values — rows/features with NaNs still need imputation before fitting.
(Some other tree-ensemble libraries, e.g. XGBoost/LightGBM, do handle missing values natively — see
`## Failure modes` above.)

## Mental model

A random forest is many weak, overfit, decorrelated opinions averaged into one stable one — bagging
gives each tree a different (if overlapping) slice of the data, feature subsampling stops them from
all leaning on the same one or two dominant features, and averaging cancels out what's left of their
individually noisy, idiosyncratic errors while keeping the shared signal.

## Questions to think about

1. In the correlation-variance formula $\text{Var}(\text{ensemble}) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$,
   what happens to the marginal benefit of adding a 301st tree to a 300-tree forest versus adding a
   second tree to a one-tree forest? Why does this explain "diminishing returns" from `n_estimators`?
2. Why does bagging alone (bootstrap resampling, no feature subsampling) leave $\rho$ higher than
   full Random Forest, and why did the from-scratch experiment above (2 features) show only a small
   correlation floor even without feature subsampling — what would you expect to change if the toy
   dataset instead had 50 features, one of which was highly predictive?
3. The Mathematical foundation shows averaging leaves bias unchanged. Given that, why does the
   "Failure modes" section say a Random Forest "can still overfit despite averaging" — what has to be
   true of the individual trees' bias for this to happen?
4. OOB error is described as "built-in cross-validation." What assumption about how the bootstrap
   samples were drawn does this rely on, and would OOB error still be a valid generalization estimate
   if you deliberately biased the bootstrap sampling (e.g. always excluding the same 10% of rows)?
5. MDI (Gini) importance is biased toward high-cardinality features. Sketch, in words, why a random
   ID column could appear "important" by MDI even though it has zero true predictive power, and why
   permutation importance would correctly assign it near-zero importance.
6. `10-decision-tree`'s Failure modes says a single tree's instability motivates Random Forest.
   Having now derived the variance-reduction math, what specific property of decision trees (versus,
   say, linear regression) made them the natural base learner for bagging in the first place?
