# 03 – Polynomial Regression

## Problem

Linear regression fits a straight line (or flat hyperplane) to data. Plenty of real relationships aren't straight — a moving object's position over time under acceleration follows a curve, dose-response effects in medicine often saturate, and marginal returns commonly diminish rather than staying constant. When the true relationship between $x$ and $y$ is curved, and you fit it with a straight line anyway, the model will be systematically wrong across the whole input range — not randomly wrong, but wrong in the same predictable direction wherever the curve bends away from the line. **Linear regression can't fit curved relationships, and you need a way to fit them without abandoning everything linear regression gives you** (a well-understood cost function, a closed-form solution, tractable optimization).

## Intuition

Suppose you're tracking an object's height over time as it's thrown into the air: at $t=0,1,2,3,4$ seconds, height is roughly $0, 15, 20, 15, 0$ meters. Plot it — it's a parabola, rising then falling. No straight line through these points is going to look good: a straight line is monotonic (always rising or always falling), but this data rises and then falls. Any single straight line will be forced to compromise, running roughly through the middle and missing every point by a lot.

Now here's the trick: nothing forces you to feed the model only $x = t$. What if you also feed it $x^2 = t^2$ as a second "feature"? The model $\hat y = \beta_0 + \beta_1 t + \beta_2 t^2$ can now bend — because $t^2$ grows non-linearly with $t$, giving the model a lever it didn't have before. Crucially, from the model's point of view, this is still *exactly* the same kind of fitting problem as multiple linear regression: it's a linear combination of "features" ($1, t, t^2$), weighted by parameters ($\beta_0, \beta_1, \beta_2$), fit to minimize squared error. The model doesn't know or care that the second feature happens to be the square of the first — it's just another number in each row of $\mathbf X$.

That's the entire idea of polynomial regression: **you're not inventing a new algorithm, you're feeding the same algorithm richer features.**

## Why simpler approaches fail

Forcing a straight line through curved data **underfits**: the model is not flexible enough to represent the true shape of the relationship, no matter how its two parameters ($\theta_0, \theta_1$) are chosen. This is a **bias** problem in the precise sense of [`05b-bias-variance-tradeoff/notes.md`](../05b-bias-variance-tradeoff/notes.md) — even averaged over infinitely many resampled training sets, a straight-line model's average prediction still can't match a curved true function, because the model family itself (all possible straight lines) doesn't contain anything that curves. More data does not fix this: collecting a million points along the same parabola still gives you the single best-fitting straight line, which is still a straight line, still wrong in the same systematic way at the same $x$ values.

You could try to fix this by hand — piecewise linear segments, manually chosen breakpoints — but this reintroduces the "eyeball it" problem from `02-linear-regression`'s Why-simpler-fails: no principled, reproducible way to choose the segments, and no single differentiable objective to optimize. What's needed is a way to add curve-fitting *capacity* to the model while keeping the parts of linear regression that work — a convex objective, a closed-form or gradient-based fit. Polynomial features are the simplest way to do exactly that.

## Mathematical foundation

### The model is still linear — in its parameters

For one input feature $x$, a degree-$k$ polynomial model is:

$$h_\beta(x) = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_k x^k$$

Define a feature transform $\phi(x) = [1, x, x^2, \dots, x^k]^T \in \mathbb{R}^{k+1}$. Then:

$$h_\beta(x) = \beta^T \phi(x)$$

This is *identical in form* to the multiple linear regression model $h_\theta(x) = \theta^T x$ from `02-linear-regression` — the only difference is that the feature vector is $\phi(x)$ (powers of a single original variable) instead of $x$ itself (several independently measured variables). The model is **non-linear in $x$** (the predicted curve visibly bends) but **linear in the parameters $\beta$** — each $\beta_j$ still enters the prediction as a simple multiplicative weight added to the others, with no $\beta_j$ multiplying another $\beta_{j'}$ or appearing inside a nonlinear function. That's the precise, checkable meaning of "linear model": linear in the *parameters being optimized*, regardless of what nonlinear transform was applied to the raw input to produce the features.

**Consequence:** every derivation from `02-linear-regression/notes.md` carries over unchanged. Build the design matrix
$$\mathbf X = \begin{bmatrix} 1 & x^{(1)} & (x^{(1)})^2 & \cdots & (x^{(1)})^k \\ 1 & x^{(2)} & (x^{(2)})^2 & \cdots & (x^{(2)})^k \\ \vdots & & & & \vdots \\ 1 & x^{(m)} & (x^{(m)})^2 & \cdots & (x^{(m)})^k \end{bmatrix}$$
and the same cost function $J(\beta) = \frac{1}{2m}\|\mathbf X\beta - \mathbf y\|^2$, the same normal equation
$$\hat\beta = (\mathbf X^T \mathbf X)^{-1}\mathbf X^T \mathbf y$$
and the same maximum-likelihood-under-Gaussian-noise justification for squared error, apply without modification. Polynomial regression is not a new algorithm — it is linear regression's exact machinery pointed at a richer, engineered set of features.

### Why degree matters, and why scale matters

Increasing $k$ increases the model's flexibility to bend — with $m$ data points, a degree-$(m-1)$ polynomial can pass through every point exactly (zero training error), which is a strong signal that high degree trades bias for variance, not a free improvement (see Experiment below).

Practically, raw powers of $x$ can differ by many orders of magnitude — if $x \sim 100$, then $x^3 \sim 10^6$. This does not change the mathematical solution (the normal equation is scale-covariant — it still finds the exact minimizer), but it makes $\mathbf X^T\mathbf X$ **numerically ill-conditioned**: columns with vastly different scales make the matrix inversion sensitive to floating-point round-off, and gradient descent (if used instead) converges far more slowly because the cost surface becomes a long, narrow ravine rather than a roughly circular bowl. Standardizing $x$ before forming polynomial features (or standardizing each polynomial column afterward) keeps the optimization numerically well-behaved without changing what function is ultimately being fit.

## Algorithm

1. Choose a maximum degree $k$.
2. Transform each raw input $x^{(i)}$ into the feature vector $\phi(x^{(i)}) = [1, x^{(i)}, (x^{(i)})^2, \dots, (x^{(i)})^k]$ (optionally scaling $x$ first for numerical stability).
3. Stack these into a design matrix $\mathbf X$, exactly as in ordinary multiple linear regression.
4. Fit $\hat\beta$ by the normal equation (or gradient descent) — no new fitting algorithm is needed.
5. Predict new points by transforming them with the same $\phi$ and evaluating $\hat\beta^T \phi(x_{\text{new}})$.
6. Select $k$ using held-out validation error (never training error alone — see Experiment and Failure modes) since training error is guaranteed to be non-increasing in $k$.

## From-scratch implementation

Implemented in `polynomial-regression.ipynb`: a manual feature-construction function

```python
def polynomial_features_manual(x, degree):
    x = np.asarray(x).reshape(-1)
    return np.column_stack([x ** p for p in range(0, degree + 1)])
```

builds $[1, x, x^2]$ column-by-column (equivalent to `np.vander(x, degree+1, increasing=True)`), with no `sklearn` involved. The resulting matrix is fed directly into the normal equation `theta = np.linalg.inv(X.T @ X) @ X.T @ y` — the identical closed-form solver derived and implemented in `02-linear-regression/notes.md` and `simple-linear-regression.ipynb`, applied here to engineered polynomial features instead of raw multiple-regression features. The notebook confirms this from-scratch fit's test MSE numerically matches the `sklearn` pipeline's test MSE at the same degree, which is the concrete demonstration that "polynomial regression" is not a separate algorithm.

## Practical implementation

`polynomial-regression.ipynb`'s main pipeline uses `sklearn.preprocessing.PolynomialFeatures` (which does exactly the manual power-expansion above, but efficiently and for multiple input variables including cross-terms like $x_1 x_2$, which get combinatorially expensive to write by hand past a couple of features) chained with `sklearn.preprocessing.StandardScaler` and `sklearn.linear_model.LinearRegression` inside an `sklearn.pipeline.Pipeline`. The mapping back to the from-scratch step is explicit and direct: `PolynomialFeatures(degree=d)` is `polynomial_features_manual(x, d)` generalized to multiple inputs and higher performance, and the `LinearRegression` step at the end of the pipeline is solving the exact same normal equation used in the from-scratch cell — the pipeline just automates feature construction, scaling, and fitting into one call instead of three manual steps.

## Experiment

**Hypothesis (stated before running):** sweeping polynomial degree upward on this topic's synthetic quadratic dataset (`y = 1.5x^2 - 3x + 5 + noise`), training error should decrease (or plateau) monotonically as degree increases, because a strictly more flexible model family can only fit its own training sample at least as well. Test error should instead be **U-shaped**: high at degree 1 (a straight line underfitting an inherently curved relationship — high bias), reaching a minimum near degree 2–3 (matching the true quadratic generating process), and rising again at high degrees (the extra flexibility starts fitting noise specific to the training sample — high variance). This is the same bias-variance shape derived in general in [`05b-bias-variance-tradeoff/notes.md`](../05b-bias-variance-tradeoff/notes.md); this experiment runs the concrete sweep on this topic's actual dataset rather than re-deriving that theory.

**Setup:** `polynomial-regression.ipynb`'s synthetic data (`x` uniform on $[0,10]$, $y = 1.5x^2 - 3x + 5$ plus Gaussian noise, `random_state=42`), degrees swept over $\{1, 2, 3, 4, 5, 8, 12, 16\}$, each fit with the `StandardScaler` → `PolynomialFeatures` → `LinearRegression` pipeline on the same train/test split, train and test MSE recorded at each degree.

**Actual result:** train MSE dropped sharply from degree 1 to degree 2 (≈151.6 → ≈22.2) and then decreased only marginally and non-monotonically through degree 16 (≈17.9 at degree 16). Test MSE dropped similarly from degree 1 to 2 (≈128.8 → ≈20.0), continued to *improve slightly* through degree 8 (≈19.8, its minimum in this run), and then rose sharply at degree 12 (≈27.3) and degree 16 (≈77.8) even as training error kept falling. See `polynomial-regression.ipynb`'s printed table and plot for the exact per-degree numbers from this run.

**Interpretation:** the qualitative U-shape predicted by the hypothesis is confirmed on the high-degree end — test error rises sharply past degree ~8 while training error keeps dropping, which is the textbook overfitting signature (a growing train/test gap). The minimum landing near degree 8 rather than exactly at the true degree (2) on this particular run, with test error at degrees 2–8 all fairly close together, illustrates that in the presence of noise the "best" degree by one held-out split is not guaranteed to exactly recover the true generating degree — it only has to be flexible enough to capture the dominant curvature, after which extra capacity for a while does little harm before variance takes over.

**Limitations:** single synthetic dataset, single train/test split (`random_state=42`) — the exact degree at which the minimum falls is sensitive to the specific noise draw and split, and a different seed could shift it earlier or later within the range where test errors are close. A production degree choice should use cross-validation (averaging over several splits) rather than one held-out split, precisely because one split can be misleading about which degree is truly best.

## Failure modes

- **Overfitting at high degree.** As shown directly in the experiment above: past some degree, additional flexibility fits noise specific to the training sample rather than the underlying signal, so test error rises even as training error keeps falling. This is the variance term from the bias-variance decomposition growing faster than bias shrinks.
- **Wild extrapolation.** High-degree polynomials are especially dangerous outside the training range: polynomial terms $x^k$ grow (or oscillate) rapidly once $x$ moves past the observed data, so predictions just beyond the training range can diverge to extreme, nonsensical values even when the fit looks excellent *inside* the training range. This is a sharper version of the general extrapolation warning in `02-linear-regression/notes.md` — the curvature that made the fit better inside the data makes it far worse outside it.
- **Numerical instability without scaling.** As discussed in the Mathematical foundation section, unscaled high powers of large $x$ values produce an ill-conditioned $\mathbf X^T\mathbf X$; symptoms include suspiciously large or unstable coefficients, or a fit that changes substantially with tiny changes to the data or degree, even though the model family hasn't fundamentally changed.
- **Choosing degree from training error alone.** Because training error is guaranteed non-increasing in degree, using it to select degree always pushes toward the highest degree tried — this is the same trap `05b-bias-variance-tradeoff/notes.md` describes generally: only held-out or cross-validated error can reveal the point where added flexibility stops helping.

## Real-world usage

- Polynomial regression is a special case of a much broader pattern: **feature engineering as a form of "linear model, richer features."** Any time you hand a linear model a non-linear transform of the raw input — polynomial terms, log/sqrt transforms, interaction terms $x_1 x_2$, spline basis functions — you're using exactly this idea: keep the well-understood linear-in-parameters machinery, and get non-linear predictive power by enriching what goes into $\mathbf X$ rather than complicating the fitting algorithm.
- In practice, low-degree polynomial terms (2 or 3) are common in tabular data pipelines where a specific curvature is expected (e.g. diminishing returns, U-shaped effects) and interpretability of individual coefficients is still wanted — high-degree polynomial expansion is rare in production because the failure modes above make it both fragile and hard to reason about, and tree-based or spline-based methods usually handle general non-linearity more robustly at higher flexibility.
- Understanding this "still linear in parameters" idea is a prerequisite for `04-regularization`: Ridge and Lasso penalties are added to exactly the same $\mathbf X$/$\beta$ setup used here, and are the standard practical control on polynomial degree's overfitting risk — regularizing away unneeded flexibility instead of manually capping the degree.

## Mental model

Polynomial regression is linear regression wearing a disguise — the model is still linear in its parameters. All you changed is what you feed it as $x$: instead of feeding it raw measured variables, you feed it powers of one variable, and the exact same normal equation (or gradient descent) that fits a straight line now fits a curve, because from the algorithm's perspective nothing about the fitting problem changed at all.

## Questions to think about

1. If polynomial regression is "just linear regression on more features," why can't you keep increasing the degree indefinitely with no downside, given that the normal equation always has an exact solution as long as $\mathbf X^T\mathbf X$ is invertible?
2. The experiment fit polynomial features on a single input variable $x$. If you instead had two input variables $x_1, x_2$ and wanted degree-2 features, what terms would `PolynomialFeatures(degree=2)` generate that a naive "just square each column" approach would miss — and why might that term matter for the fit?
3. Suppose you fit degree-1 through degree-10 polynomials and pick the degree with the lowest *training* error. Using the reasoning from `05b-bias-variance-tradeoff/notes.md`, explain precisely why this selection procedure is guaranteed to pick the highest degree tried, regardless of whether that degree actually generalizes best.
4. A colleague argues that scaling $x$ before computing polynomial features "changes the model you're fitting." Using the normal equation, explain what scaling changes and what it leaves invariant — does the *predicted curve* $\hat y(x)$ change, or only the numerical process used to find it?
