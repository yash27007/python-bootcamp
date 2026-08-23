# 04 – Regularization

## Problem

You have a flexible model — many features, or a high-degree polynomial (`03-polynomial-regression`) — and you know from the bias-variance tradeoff (see [`05b-bias-variance-tradeoff/notes.md`](../05b-bias-variance-tradeoff/notes.md)) that too much flexibility drives training error down while test error rises, because the extra flexibility starts fitting noise instead of signal. You could fix this by manually restricting the model — picking a lower polynomial degree, dropping features by hand — but that requires deciding complexity *before* seeing how it performs, and doing it well requires trying many combinations. **How do you keep a flexible model from overfitting without manually picking model complexity in advance, feature by feature or degree by degree?**

## Intuition

Imagine you're fitting the forest-fire model in this topic's dataset with nine correlated weather/index features. Ordinary least squares is told only "minimize training error" — it has no reason to prefer a solution with small, modest coefficients over one with huge, wildly compensating coefficients, as long as both fit the training points equally well. If two features are highly correlated, OLS can happily assign one feature a coefficient of $+500$ and the other $-498$ — they nearly cancel on the training data, fit it perfectly, and are catastrophically unstable on any new data where the two features don't correlate in exactly the same way.

Regularization adds a second term to what the model is asked to minimize: not just "fit the data," but "fit the data *and* keep the coefficients small." It's a budget. Instead of manually deciding "use only 3 features" or "use only degree 2," you tell the optimizer "you may use all the flexibility you want, but large coefficients cost you" — and let the optimization itself decide how much of that budget to spend on each feature, guided directly by how much each feature actually helps predict $y$.

## Why simpler approaches fail

**Manually selecting features or model complexity** — trying every subset of features, or every polynomial degree, and picking the one with best cross-validated performance — is combinatorially expensive. With $n$ features, there are $2^n$ possible subsets; checking all of them by refitting and cross-validating is intractable past a modest $n$ (with the roughly 9-10 features in this topic's dataset alone, that's 500+ subsets, and real datasets often have far more features). You need something that searches this space implicitly and continuously rather than by brute-force enumeration.

**Early stopping alone** (stopping gradient descent before it fully converges, on the theory that an under-trained model hasn't yet had the chance to overfit) is fragile: the "right" stopping point depends on the learning rate, the specific optimizer, how the loss curve happens to look for this particular run, and is really an accidental, indirect way of controlling model complexity rather than a controlled one. It conflates "not done optimizing yet" with "the right amount of flexibility" — two different things that happen to move in the same direction during training only for some models, some datasets, some learning rates.

What's needed: a way to directly penalize the property that causes overfitting — coefficients large enough to fit noise precisely — built into the objective function itself, tunable by a single continuous knob rather than a discrete, combinatorial search over feature subsets or polynomial degrees.

## Mathematical foundation

### Ridge (L2) regression

Add an $L_2$ penalty — the sum of squared coefficients — to the ordinary least-squares objective:

$$J(\theta) = \frac{1}{2m}\sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2 + \frac{\lambda}{2m}\sum_{j=1}^n \theta_j^2$$

The bias term $\theta_0$ is conventionally excluded from the penalty (shrinking the intercept doesn't help against overfitting to feature-driven noise, and would bias predictions toward zero regardless of where the data actually sits). $\lambda \geq 0$ controls the strength of the penalty: $\lambda=0$ recovers ordinary least squares exactly; larger $\lambda$ pushes coefficients harder toward zero.

**Closed-form solution.** Following the same derivation as `02-linear-regression/notes.md` — set $\nabla_\theta J(\theta) = 0$ — gives:

$$\hat\theta = (\mathbf X^T \mathbf X + \lambda I')^{-1}\mathbf X^T \mathbf y$$

where $I'$ is the identity matrix with the first diagonal entry zeroed (so $\theta_0$ is unpenalized). Note this also **fixes** the exact numerical failure mode multicollinearity causes in ordinary least squares: adding $\lambda I'$ to $\mathbf X^T\mathbf X$ makes it invertible even when $\mathbf X^T\mathbf X$ itself is singular or near-singular, because $\lambda I'$ adds strictly positive values along the diagonal, pushing every eigenvalue away from zero.

### Lasso (L1) regression, and why L1 induces sparsity while L2 doesn't

Lasso replaces the squared penalty with an absolute-value penalty:

$$J(\theta) = \frac{1}{2m}\sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2 + \frac{\lambda}{m}\sum_{j=1}^n |\theta_j|$$

This single change in the penalty's shape has a qualitatively different effect: **Lasso drives some coefficients to exactly zero (feature selection), while Ridge shrinks all coefficients smoothly toward zero without ever making them exactly zero.**

**Geometric argument.** Minimizing the squared-error term subject to a penalty budget is equivalent to constrained optimization: minimize $\sum (h_\theta(x^{(i)}) - y^{(i)})^2$ subject to $\sum \theta_j^2 \leq t$ (Ridge) or $\sum |\theta_j| \leq t$ (Lasso), for some $t$ corresponding to $\lambda$. In two dimensions ($\theta_1, \theta_2$):

- The Ridge constraint region $\{\theta_1^2 + \theta_2^2 \leq t\}$ is a **circle** (a disk) — smooth, with no corners.
- The Lasso constraint region $\{|\theta_1| + |\theta_2| \leq t\}$ is a **diamond** — a square rotated 45°, with sharp corners sitting exactly on the coordinate axes (where one coordinate is zero).

The unconstrained least-squares solution has elliptical contours of constant squared error radiating outward from the OLS optimum. The regularized solution is the point where the smallest such ellipse first touches the constraint region's boundary. For a smooth circular boundary (Ridge), that first point of contact is generically somewhere on the circle's curved edge — essentially never at a point where a coordinate is exactly zero. For the diamond boundary (Lasso), the corners stick out toward the axes and are disproportionately likely to be the first point an elliptical contour touches, *precisely because* they are corners — a whole range of ellipse orientations and sizes touch the diamond exactly at a corner rather than along a flat edge. When contact happens at a corner, one or more coordinates are exactly zero by construction (a corner of the diamond, other than at the origin, always has at least one coordinate equal to zero). That geometric asymmetry — smooth boundary vs. cornered boundary — is the entire reason L1 produces exact zeros and L2 does not.

**Elastic Net** combines both penalties:

$$J(\theta) = \frac{1}{2m}\sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2 + \frac{\lambda}{m}\left[\alpha\sum_{j=1}^n|\theta_j| + \frac{1-\alpha}{2}\sum_{j=1}^n \theta_j^2\right]$$

with $\alpha \in [0,1]$ interpolating between pure Lasso ($\alpha=1$) and pure Ridge ($\alpha=0$). This exists specifically because pure Lasso has a known weakness with correlated features (see Failure modes).

### Bias-variance framing

Regularization is a direct lever on the bias-variance tradeoff derived in [`05b-bias-variance-tradeoff/notes.md`](../05b-bias-variance-tradeoff/notes.md), not a separate idea. Shrinking coefficients toward zero moves predictions toward a simpler, less flexible function — this necessarily increases bias slightly, because the model is now constrained away from the unconstrained least-squares optimum, which was the best *unbiased* fit to this particular training sample. But it decreases variance, often substantially, because the model is less free to chase sample-specific noise; a small coefficient can't swing wildly in response to one noisy training point the way an unconstrained large coefficient can. As with polynomial degree, the right $\lambda$ is the one that minimizes the *sum* — this is the same U-shaped tradeoff curve, just parameterized by $\lambda$ instead of degree, and the same warning applies: only held-out validation error, not training error, can locate that minimum, because training error is monotonically non-decreasing in $\lambda$ by construction (more penalty can only make the training fit worse or equal, never better).

## Algorithm

1. Standardize features (regularization penalizes coefficient *magnitude*, so features on different scales would be penalized unfairly — a feature measured in the thousands would naturally need a tiny coefficient regardless of importance, and get penalized less than an equally important feature measured in single digits).
2. Choose a penalty type (Ridge, Lasso, or Elastic Net's mixing parameter $\alpha$) based on whether exact feature selection is wanted.
3. For a grid of candidate $\lambda$ values (or use each library's built-in cross-validated path solver, e.g. `LassoCV`/`RidgeCV`), fit the model and evaluate held-out performance.
4. Select the $\lambda$ (and $\alpha$, for Elastic Net) minimizing cross-validated error, not training error.
5. Refit on the full training set at the chosen hyperparameters, evaluate once on a held-out test set.

## From-scratch implementation

Implemented in `model-training.ipynb`: a direct NumPy function

```python
def ridge_closed_form(X, y, lam):
    n_features = X.shape[1]
    I = np.eye(n_features)
    return np.linalg.inv(X.T @ X + lam * I) @ X.T @ y
```

applies the closed-form Ridge solution $\hat\theta = (\mathbf X^T\mathbf X + \lambda I)^{-1}\mathbf X^T \mathbf y$ derived above directly to this topic's scaled Algerian-forest-fire training data, and is checked against `sklearn`'s `Ridge(alpha=1.0)` on the same data (they match to numerical tolerance — see Experiment). The notebook then sweeps $\lambda$ across a log-spaced grid and records how every coefficient moves as $\lambda$ grows, which is the coefficient-shrinkage result reported in the Experiment section below.

## Practical implementation

`model-training.ipynb` and `algerian-forest.ipynb` use `sklearn.linear_model.Ridge`, `Lasso`, and `ElasticNet` on the cleaned forest-fire dataset, plus `LassoCV` for automatic cross-validated $\lambda$ selection. These map directly back to the from-scratch step: `Ridge(alpha=lam)` solves the identical closed-form equation implemented manually above (`sklearn` uses a numerically stabler solver internally — e.g. Cholesky or SVD-based — rather than a literal matrix inverse, the same practical-vs-from-scratch distinction noted for `LinearRegression` in `02-linear-regression/notes.md`); `Lasso` and `ElasticNet` solve the corresponding non-smooth objectives via coordinate descent, since the $|\theta_j|$ penalty has no closed form (it isn't differentiable at $\theta_j=0$, exactly the point that makes it capable of producing exact zeros in the first place).

## Experiment

**Hypothesis (stated before running):** as $\lambda$ increases, Ridge coefficients should shrink toward zero **monotonically but smoothly**, never reaching exactly zero across the swept range. Lasso, run over the same $\lambda$ grid, should hit **exact zeros** for at least some features at moderate $\lambda$ — well before Ridge's coefficients are anywhere near zero at a comparable $\lambda$ — directly reflecting the L1-vs-L2 geometric argument above (diamond corners vs. smooth circle).

**Setup:** `model-training.ipynb`'s scaled training features (`X_train_scaled`, the correlation-filtered Algerian forest-fire predictors) and target (`FWI`), $\lambda$ swept log-uniformly over $[10^{-2}, 10^4]$ (30 points), Ridge solved by the from-scratch closed form above, Lasso solved by `sklearn.linear_model.Lasso` at each $\lambda$ on the identical grid.

**Actual result:** the from-scratch Ridge closed-form solution at $\lambda=1.0$ matched `sklearn`'s `Ridge(alpha=1.0)` coefficients to within `1e-2` (`np.allclose` confirmed `True` in the notebook). Across the full $\lambda$ sweep, the coefficient-path plot shows Ridge coefficients curving smoothly toward (but never reaching) zero as $\lambda$ grows, while the Lasso path shows multiple features flattening to exactly zero at a materially smaller $\lambda$ than where Ridge coefficients become negligible. See `model-training.ipynb`'s printed near-zero-coefficient counts and the two side-by-side path plots for this run's exact numbers.

**Interpretation:** this is direct empirical confirmation of the geometric argument — the L1 penalty's cornered constraint region produces exact sparsity that the L2 penalty's smooth circular region structurally cannot, on real (not synthetic) correlated feature data, not just in the idealized two-dimensional picture.

**Limitations:** this used one train/test split and this dataset's specific correlation structure (recall several highly-correlated features were already dropped upstream in this notebook at a 0.85 correlation threshold, before this experiment ran) — a different correlation structure or a different subset of retained features would change exactly which coefficients zero out and at which $\lambda$, though the qualitative Ridge-vs-Lasso shrinkage-shape difference is a structural property of the penalty, not an artifact of this dataset.

## Failure modes

- **Wrong $\lambda$ under/over-regularizes.** Too small a $\lambda$ barely constrains the model (approaching plain OLS's overfitting risk); too large a $\lambda$ shrinks every coefficient toward zero regardless of how useful the corresponding feature actually is, driving the model toward underfitting (high bias) — same U-shaped test-error curve as any other complexity knob, and the same warning applies: selecting $\lambda$ by training error alone always pushes toward $\lambda=0$, since penalty can only increase training error relative to unconstrained OLS.
- **Lasso's instability with correlated features.** When two features are highly correlated, Lasso tends to arbitrarily pick one and zero out the other (rather than splitting credit between them), and *which* one gets kept can change with small perturbations to the data or even the solver's iteration order — the selected feature set becomes unstable in a way that's directly counter to the goal of a robust, reproducible model. This is exactly why Elastic Net exists: its L2 component encourages correlated features to be kept (or shrunk) together rather than picking one arbitrarily, at the cost of no longer getting Lasso's crisp binary feature-selection decision.
- **Unscaled features distort the penalty.** Because the penalty is applied directly to coefficient magnitude, a feature on a much larger numeric scale gets an unfairly small "natural" coefficient and is therefore penalized less relative to its actual predictive contribution than a differently-scaled but equally important feature — always standardize before regularizing, as done in `model-training.ipynb`.
- **Regularizing the bias term.** Penalizing $\theta_0$ would bias every prediction toward zero regardless of the data's actual mean level, which is rarely desired — this is why $\theta_0$ is conventionally excluded from the penalty in both the math above and every practical implementation.

## Real-world usage

- Ridge is the default choice when all features are believed to carry at least some signal and the priority is stabilizing coefficients (especially under multicollinearity) rather than eliminating features.
- Lasso is preferred when a large feature set is suspected to contain many irrelevant features and an interpretable, sparse model (a short list of "the features that matter") is the goal — common in genomics, text features, and other high-dimensional settings where feature counts can exceed sample counts.
- Elastic Net is the practical default in many production pipelines specifically because real feature sets frequently contain correlated groups, and it avoids Lasso's arbitrary single-feature selection within such a group while still providing some sparsity.
- Regularization strength ($\lambda$) is a first-class hyperparameter tuned by cross-validation in virtually every regularized-linear-model deployment — `LassoCV`/`RidgeCV`-style automatic path search (as used in this topic's notebook) is the standard practical tool, avoiding a manual grid search implemented from scratch.
- The same L2-penalty idea reappears throughout this course under the name **weight decay** in neural network training — it is the identical mathematical penalty applied to a much larger, non-linear model, added for the same bias-variance reason.

## Mental model

Regularization is a budget on how much the model is allowed to trust the training data. Every unit of coefficient magnitude costs something, so the optimizer only "spends" on a feature if the reduction in squared error it buys is worth more than the penalty — L2 spends smoothly and never fully divests from a feature, L1 spends like a fixed acquisition cost per feature and will drop features entirely once they're not worth their keep.

## Questions to think about

1. Using the geometric picture (circle vs. diamond constraint region), explain why increasing the number of correlated features would make Lasso's selected feature set *less* reproducible across different random samples of the same data, while Ridge's coefficients would stay comparatively stable.
2. You cross-validate $\lambda$ for Ridge regression and find the optimal value is very close to 0. What does that tell you about where your unregularized model already sat on the bias-variance curve, and would you expect Lasso's optimal $\lambda$ on the same data to also be near 0?
3. The Ridge closed-form solution adds $\lambda I$ to $\mathbf X^T \mathbf X$ before inverting. Explain, in terms of eigenvalues, why this guarantees invertibility even when the original $\mathbf X^T\mathbf X$ is exactly singular (e.g. from a perfectly duplicated feature) — and why this is the same fix multicollinearity needed in `02-linear-regression/notes.md`'s Failure modes.
4. If you already believe (from domain knowledge) that only 3 of your 50 features are truly predictive, would you reach for Ridge or Lasso first, and why does your answer depend on treating "coefficient shrinkage" and "feature selection" as two different goals rather than one?
