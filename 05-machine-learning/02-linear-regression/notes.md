# 02 – Linear Regression

## Problem

You have a set of examples where each example pairs some input measurements with a continuous numeric outcome — house size and price, hours studied and exam score, weight and height. You want a function that, given new input measurements, predicts the outcome. Specifically: **predict a continuous quantity as a linear combination of features.** That's the entire scope of this topic — not classification (discrete labels), not curve-fitting in general, just: find weights $\theta$ such that $\theta_0 + \theta_1 x_1 + \dots + \theta_d x_d$ is a good predictor of $y$.

## Intuition

Suppose you have three data points relating weight (kg) to height (cm): $(60, 165)$, $(70, 172)$, $(80, 178)$. Plot them — they roughly fall along a rising straight line. If someone hands you a new weight, 75 kg, you want a principled way to read off a predicted height, not a guess.

The simplest possible model is a straight line: $\text{height} = \theta_0 + \theta_1 \cdot \text{weight}$. $\theta_1$ is how many cm height increases per kg of weight; $\theta_0$ is where the line would cross weight $= 0$ (not meaningful on its own here, but necessary to let the line sit at the right height). Once you have numbers for $\theta_0$ and $\theta_1$, prediction is arithmetic — plug in the weight, get a height.

The hard part is not evaluating the line — it's choosing $\theta_0, \theta_1$ well. Different lines fit the three points differently well. Linear regression is precisely the answer to "which specific line (or hyperplane, with more features) fits best, and what does 'best' mean here, made precise enough to compute."

With more than one feature — say weight *and* age predicting height — the same idea generalizes: instead of a line in 2D, you get a hyperplane in higher dimensions, and instead of one slope you get one weight per feature, each answering "how much does the prediction change per unit of this feature, holding the others fixed."

## Why simpler approaches fail

**Eyeballing a line.** You could look at a scatter plot and sketch a line that "looks about right." This is not reproducible — two people draw two different lines, and neither has a way to say which is more correct beyond visual judgment. It also doesn't scale past two dimensions: you cannot eyeball a hyperplane in 5-dimensional feature space.

**Simple averaging.** You could ignore $x$ entirely and always predict $\bar y$ (the mean of all target values). This gives *a* number, and it minimizes squared error if you're not allowed to use $x$ at all — but it throws away all the information in $x$. If height and weight are correlated, predicting the same $\bar y$ for every weight is needlessly bad; the whole point is that different weights should produce different predictions.

**Connecting adjacent points / interpolation.** You could draw a line connecting each successive pair of points. This overfits immediately — it treats every observed value as exactly correct (ignoring measurement noise) and gives no rule for predicting outside the observed weights, or even for what to do when the "next" point in a new dataset doesn't fall exactly on a segment.

What's missing from all three: a **numeric criterion for "how good is this line"** that (a) uses every feature, (b) is well-defined for any candidate line, not just a few special ones, and (c) can be optimized systematically rather than by hand. That criterion is a **cost function**, and optimizing it is what turns "guess a line" into "solve for the best line."

## Mathematical foundation

### Setup and notation

Training set: $\{(x^{(i)}, y^{(i)})\}_{i=1}^{m}$, $m$ examples. For multiple features, $x^{(i)} \in \mathbb{R}^d$; add a constant feature $x_0^{(i)} = 1$ so the intercept can be written as an ordinary weight. Parameter vector $\theta = [\theta_0, \theta_1, \dots, \theta_d]^T$. The model (hypothesis):

$$h_\theta(x) = \theta^T x = \theta_0 + \theta_1 x_1 + \dots + \theta_d x_d$$

In matrix form over the whole training set: $\hat{\mathbf y} = \mathbf X \theta$, where $\mathbf X \in \mathbb{R}^{m \times (d+1)}$ is the design matrix (one row per example, a leading column of 1s).

### Why squared error, precisely

The cost function is the numeric criterion "why simpler approaches fail" said we needed. The standard choice is mean squared error:

$$J(\theta) = \frac{1}{2m}\sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2$$

This is not an arbitrary choice among many equally good ones. Two independent justifications:

**1. Differentiability and convexity.** $J(\theta)$ is a smooth, differentiable, convex quadratic function of $\theta$ — a paraboloid bowl with a single minimum, no local minima to get stuck in. Squared error is differentiable everywhere (unlike absolute error, which has a kink at zero), so gradient-based optimization has a well-defined direction to move at every point, and convexity guarantees that direction leads to *the* global optimum, not just *a* local one.

**2. It is exactly the maximum-likelihood answer under Gaussian noise.** This is the deeper justification, and it's worth deriving rather than asserting. Assume the data is generated as
$$y^{(i)} = \theta^T x^{(i)} + \varepsilon^{(i)}, \qquad \varepsilon^{(i)} \sim \mathcal N(0, \sigma^2) \text{ i.i.d.}$$
i.e. the true relationship is linear, and each observation is corrupted by independent Gaussian noise (a standard assumption when errors come from many small, additive, unrelated sources — measurement imprecision, unmodeled minor factors — which by the Central Limit Theorem tend toward a Gaussian). Under this model, $y^{(i)} \mid x^{(i)}; \theta \sim \mathcal N(\theta^T x^{(i)}, \sigma^2)$, so its density is
$$p(y^{(i)} \mid x^{(i)}; \theta) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right)$$
The likelihood of the whole dataset (independence across examples) is the product of these densities:
$$L(\theta) = \prod_{i=1}^m \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2}\right)$$
Maximizing $L(\theta)$ is equivalent to maximizing its log (log is monotonic increasing, so the maximizer is the same):
$$\log L(\theta) = m\log\frac{1}{\sqrt{2\pi}\sigma} - \frac{1}{2\sigma^2}\sum_{i=1}^m \left(y^{(i)} - \theta^T x^{(i)}\right)^2$$
The first term does not depend on $\theta$. So maximizing $\log L(\theta)$ over $\theta$ is exactly the same as **minimizing** $\sum_{i=1}^m (y^{(i)} - \theta^T x^{(i)})^2$ — which is (up to the constant $\frac{1}{2m}$ scale factor, which doesn't change where the minimum is) precisely $J(\theta)$.

**Conclusion:** minimizing squared error is not an arbitrary convention — it is the maximum-likelihood parameter estimate under the assumption of linear signal plus i.i.d. Gaussian noise. If you accept that noise model (and it's frequently a reasonable one), least squares is provably the "correct" way to fit, not merely a convenient one. This also explains why squared error weights large residuals so heavily: a Gaussian's density falls off as $\exp(-\text{error}^2)$, so under this noise model, large errors are considered disproportionately unlikely, and the objective reflects that.

### The normal equation (closed-form solution)

$J(\theta)$ is convex and differentiable, so its minimum occurs exactly where the gradient is zero. Write $J(\theta) = \frac{1}{2m}\|\mathbf X\theta - \mathbf y\|^2$ and differentiate with respect to the vector $\theta$:

$$\nabla_\theta J(\theta) = \frac{1}{m}\mathbf X^T(\mathbf X \theta - \mathbf y)$$

Setting this to zero:
$$\mathbf X^T \mathbf X \theta = \mathbf X^T \mathbf y$$

These are the **normal equations**. If $\mathbf X^T \mathbf X$ is invertible (full column rank — no exact linear dependence among features):

$$\boxed{\hat\theta = (\mathbf X^T \mathbf X)^{-1} \mathbf X^T \mathbf y}$$

This is a single, exact, closed-form solution — no iteration, no learning rate to tune. If $\mathbf X^T \mathbf X$ is singular (e.g. two features are exact linear combinations of each other, or $d+1 > m$), the inverse doesn't exist; use the Moore–Penrose pseudo-inverse $\theta = \mathbf X^+ \mathbf y$, which returns the minimum-norm solution among the (now infinite) minimizers.

**Geometric interpretation.** $\mathbf X\theta$, as $\theta$ ranges over all of $\mathbb R^{d+1}$, sweeps out the column space of $\mathbf X$ — every vector reachable as a linear combination of $\mathbf X$'s columns (i.e. every possible linear prediction). $\mathbf y$ generally does not lie in that subspace (there's noise; the model is only approximately correct). Minimizing $\|\mathbf X\theta - \mathbf y\|^2$ is finding the point in the column space of $\mathbf X$ that is closest to $\mathbf y$ in Euclidean distance — this is exactly the **orthogonal projection** of $\mathbf y$ onto the column space of $\mathbf X$. The normal equation $\mathbf X^T(\mathbf X\theta - \mathbf y) = 0$ says precisely that the residual vector $\mathbf X\theta - \mathbf y$ is orthogonal to every column of $\mathbf X$ — the residual carries no more signal that a linear combination of the features could explain; whatever's left is exactly what the model cannot reach.

### Gradient descent as the iterative alternative

The normal equation requires computing and inverting $\mathbf X^T\mathbf X$, a $(d+1)\times(d+1)$ matrix — $O(d^3)$ for the inversion. For very large $d$ (many features) or when no closed form exists for a related model, an iterative alternative is preferable: **gradient descent**. Since $\nabla_\theta J(\theta) = \frac{1}{m}\mathbf X^T(\mathbf X\theta - \mathbf y)$ points in the direction of steepest *increase*, repeatedly stepping in the opposite direction decreases $J$:

$$\theta := \theta - \alpha \nabla_\theta J(\theta) = \theta - \alpha \cdot \frac{1}{m}\mathbf X^T(\mathbf X\theta - \mathbf y)$$

with $\alpha$ the learning rate. Per-coordinate, this is the familiar update
$$\theta_j := \theta_j - \alpha \cdot \frac{1}{m}\sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)x_j^{(i)}$$
Because $J$ is convex with no local minima other than the global one, gradient descent with a sufficiently small $\alpha$ is guaranteed to converge to the same $\hat\theta$ the normal equation gives exactly — this is not a coincidence, it's two different methods finding the same unique minimum of the same convex function, and the from-scratch implementation below verifies this numerically.

## Algorithm

**Closed-form (normal equation):**
1. Build design matrix $\mathbf X$ (with bias column) and target vector $\mathbf y$.
2. Compute $\hat\theta = (\mathbf X^T \mathbf X)^{-1}\mathbf X^T \mathbf y$ (or the pseudo-inverse if $\mathbf X^T\mathbf X$ is singular).
3. Predict new points as $\hat y = \hat\theta^T x$.

**Iterative (batch gradient descent):**
1. Initialize $\theta$ (commonly all zeros).
2. Repeat until convergence: compute predictions $h_\theta(x^{(i)})$ for all $i$, compute the gradient $\frac{1}{m}\mathbf X^T(\mathbf X\theta - \mathbf y)$, update $\theta \mathrel{-}= \alpha \cdot \text{gradient}$.
3. Stop when the decrease in $J(\theta)$ between iterations falls below a threshold, the gradient norm is small, or a fixed iteration budget is reached.
4. Feature scaling (standardization $x' = (x-\mu)/\sigma$) before running gradient descent is standard practice — it keeps the cost surface closer to circular/symmetric rather than a narrow elongated bowl, which lets a single learning rate work well across all coordinates and speeds convergence substantially.

## From-scratch implementation

Implemented in `simple-linear-regression.ipynb` (final cells): batch gradient descent is coded directly in NumPy — building the bias-augmented design matrix `X_gd = np.c_[np.ones(...), X_train]`, then repeatedly applying the exact update rule from the Mathematical foundation section (`theta -= lr * (1/m) * X.T @ (X @ theta - y)`) for a fixed number of iterations. The same notebook then solves the normal equation directly (`np.linalg.inv(X_gd.T @ X_gd) @ X_gd.T @ y_gd`) on the same scaled data. Both are compared against `sklearn`'s `LinearRegression` fit on the identical data — see Experiment below for the result.

## Practical implementation

`simple-linear-regression.ipynb` and `multiple-linear-regression.ipynb` use `sklearn.linear_model.LinearRegression`, which internally solves (a variant of) the normal equation via a numerically stable factorization (SVD-based least squares) rather than a naive matrix inverse — same underlying idea as the from-scratch normal-equation cell above, just implemented more robustly for production use (handles rank-deficient `X` gracefully, avoids the numerical instability of literally inverting `X^T X`). `sklearn.linear_model.SGDRegressor` is the practical counterpart to the from-scratch gradient-descent cell: it performs the same iterative gradient-based update, but with stochastic/mini-batch updates and more sophisticated learning-rate schedules, so it scales to datasets too large to hold in memory for a single normal-equation solve. The mapping in both directions is: **from-scratch NumPy → production library, same math, different engineering for scale and numerical stability.**

## Experiment

**Hypothesis (stated before running):** the closed-form normal equation and batch gradient descent, run on the same scaled training data, should converge to the same coefficients (within a small numerical tolerance) — because both are solving the same convex optimization problem, and a convex quadratic has exactly one minimum.

**Setup:** `simple-linear-regression.ipynb`'s scaled `X_train`/`y_train` (height-weight data), gradient descent run for 5000 iterations at `lr=0.1`, compared against the closed-form normal equation and against `sklearn.LinearRegression.fit()` on the identical inputs.

**Actual result:** all three methods produced the same coefficients to at least 3 decimal places (intercept ≈ 156.47, slope ≈ 17.30); the notebook's explicit `np.allclose(theta_gd, theta_normal, atol=1e-3)` check evaluated to `True`. See the notebook's final cells for the exact printed values from this run.

**Interpretation:** this is direct empirical confirmation of the convexity argument above — there is exactly one minimum of $J(\theta)$ for this model, and both the iterative and closed-form routes find it.

**Limitations:** this was checked on one small dataset (height-weight, one feature) where the normal equation is cheap and well-conditioned; it does not demonstrate gradient descent's actual advantage (scalability to large $d$ or streaming data), only that the two methods agree where both are tractable. Gradient descent's result also depends on the learning rate and iteration count — an insufficiently small learning rate or too few iterations would show disagreement that is an optimization artifact, not evidence the two objectives differ.

## Failure modes

- **Multicollinearity.** When features are highly correlated (or exactly linearly dependent), $\mathbf X^T\mathbf X$ becomes near-singular (or exactly singular). Coefficients become unstable — small changes in the data cause large swings in $\hat\theta$, and individual coefficients lose their "holding other features constant" interpretation because features can't actually be varied independently. Diagnostic: high pairwise correlation, or a high variance inflation factor (VIF). This is the failure mode `04-regularization`'s Ridge penalty directly addresses.
- **Non-linearity.** If the true relationship between features and target is curved, a linear model has systematic, irreducible bias no matter how much data it sees — this is exactly the "high bias" case discussed in [`05b-bias-variance-tradeoff/notes.md`](../05b-bias-variance-tradeoff/notes.md). Diagnostic: structured (non-random) patterns in a residual-vs-fitted plot. `03-polynomial-regression` is the direct fix when the non-linearity is a smooth curve.
- **Heteroscedasticity.** When the noise variance $\sigma^2$ is not actually constant across the range of $x$ (e.g. errors grow larger for larger predicted values), the Gaussian-noise assumption underlying the MLE derivation above is violated. OLS coefficients remain unbiased, but the model's implicit assumption that all points are equally reliable is wrong, so standard errors/confidence intervals become invalid, and points with larger true variance get equal weight when they should get less.
- **Extrapolation.** A fitted line is only justified over (and near) the range of $x$ actually observed in training. Predicting far outside that range assumes the linear relationship continues unchanged, which is frequently false — nothing in the mathematics prevents you from evaluating $h_\theta(x)$ at any $x$, but nothing supports the prediction being meaningful there either.

## Real-world usage

- Linear regression (and its regularized variants) remains a standard baseline in applied ML — it's fast to fit, has interpretable coefficients, and its failure modes are well-understood diagnostics rather than opaque behavior.
- Coefficient interpretation ("a 1-unit change in $x_j$ is associated with a $\theta_j$ change in $y$, holding other features fixed") is used directly in econometrics, A/B test analysis, and scientific studies where interpretability matters as much as raw predictive accuracy.
- The normal equation / gradient descent duality generalizes: every model in this course that has a smooth, differentiable loss (logistic regression, neural network layers) is fit with some variant of gradient descent for the same underlying reason — no closed form exists once the model is no longer linear-in-parameters with squared loss.
- `SGDRegressor` and mini-batch gradient descent specifically are the pattern used at any scale where the full dataset doesn't fit in memory or a single normal-equation solve is too slow — the same update rule derived above underlies training of far larger models later in this course.

## Mental model

Linear regression finds the line (or hyperplane) minimizing total squared vertical distance to the data — and that specific choice of "squared" and "vertical distance to minimize" is exactly the maximum-likelihood answer if you assume the data is a linear signal corrupted by Gaussian noise. Geometrically, it's the projection of your target vector onto the space of everything a linear combination of your features could possibly produce.

## Questions to think about

1. Why does minimizing squared error correspond to maximum likelihood under *Gaussian* noise specifically — what part of the derivation would change if the noise were instead assumed to follow a Laplace distribution, and which loss function would that assumption justify instead?
2. Two features in your dataset have correlation 0.98. What specifically breaks in the normal equation's assumptions, and what does that predict about the coefficients you'd get if you refit the model on a slightly different sample of the same population?
3. You run gradient descent and the cost increases every iteration instead of decreasing. Using the update rule derived above, what single hyperparameter is almost certainly the cause, and why does an overly large value of it produce divergence rather than just slow convergence?
4. The geometric interpretation says the fitted values $\mathbf X\hat\theta$ are the projection of $\mathbf y$ onto the column space of $\mathbf X$. What does it mean, geometrically, for a model to have zero training error on every point — what would that imply about $\mathbf y$'s relationship to the column space of $\mathbf X$?
