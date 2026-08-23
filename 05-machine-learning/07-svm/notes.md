# 07 – Support Vector Machines

## Problem

`06-logistic-regression` finds *a* separating hyperplane, if one exists, by minimizing log loss — but log loss says nothing about *how well-placed* that hyperplane is relative to the data on either side of it. For a linearly-separable dataset there are infinitely many hyperplanes that classify every training point correctly; some hug one class closely and leave almost no room before crossing into the other class's territory, while others sit roughly in the middle with breathing room on both sides. All of them achieve zero training error. **Among all the hyperplanes that separate two classes, which one should you actually pick, and does that choice matter for how well the model generalizes to new points?**

## Intuition

Picture two clusters of points on a 2D plane, clearly separable by a straight line. Draw two candidate lines: Line A passes just barely past the nearest point of each class, practically grazing them. Line B sits as far as possible from the nearest point of *either* class — as if you inflated a corridor around the line until it just touched a point on each side, and centered the line in that corridor.

Both lines classify every training point correctly. But now imagine a new point arrives, close to where the training data was but not exactly on it — a slightly noisier measurement of the same underlying pattern. Line A, hugging the boundary of one class, is far more likely to misclassify that new point, because it left almost no margin for the natural jitter in real data. Line B, centered in the widest possible gap between the classes, tolerates that jitter far better.

That's the entire idea of a Support Vector Machine: don't just find *a* separator — find the one that maximizes the width of the empty corridor ("margin") between the two classes. The points that actually touch the edges of that corridor are called **support vectors**, because they alone determine where the boundary sits — every other point could be moved (as long as it stays outside the margin) without changing the fitted hyperplane at all.

## Why simpler approaches fail

Logistic regression's objective — minimize binary cross-entropy — has no term in it that rewards distance from the boundary once a point is already classified correctly (with high confidence, its loss just asymptotically approaches zero, but nothing in the objective explicitly compares *how far* different points sit from the line, and there is no unique "widest gap" solution the objective is designed to prefer over any other separating line). Concretely: take a linearly separable dataset and find two different hyperplanes, both fit by different reasonable procedures, that both achieve zero training error — logistic regression's own unregularized loss will push $\|\theta\|\to\infty$ regardless of which specific separating direction it started converging toward, so the *direction* of the boundary is determined largely by the fitting dynamics and any regularization, not by an explicit geometric guarantee. Two hyperplanes that both perfectly separate the training data can have very different distances to the nearest points of each class, and nothing about "minimize log loss" by itself distinguishes them on that basis.

**Just picking any zero-training-error separator is not enough** — training accuracy alone cannot tell you which of several equally-correct-on-training-data boundaries will generalize best, because generalization is a statement about *unseen* points, and "distance to the nearest training points" is the one geometric quantity that directly reasons about how much room there is for new points to land on the wrong side by chance. What's needed: an objective that directly and explicitly optimizes for that distance — the margin — rather than one that merely happens to find *some* separator as a side effect of minimizing a probability-based loss.

## Mathematical foundation

### Distance from a point to a hyperplane

A hyperplane in $n$-dimensional feature space is $\{x : w^Tx + b = 0\}$, where $w$ is a vector normal (perpendicular) to the hyperplane and $b$ shifts it away from the origin. The signed distance from any point $x_0$ to this hyperplane is

$$\text{distance} = \frac{w^Tx_0 + b}{\|w\|}$$

(Derivation sketch: move a distance $d$ from a point $x_1$ *on* the hyperplane, along the unit normal direction $w/\|w\|$, to reach $x_0$: $x_0 = x_1 + d\frac{w}{\|w\|}$. Substitute into $w^Tx_0+b$: $w^Tx_1 + b + d\frac{w^Tw}{\|w\|} = 0 + d\|w\|$, since $x_1$ is on the hyperplane so $w^Tx_1+b=0$, and $w^Tw = \|w\|^2$. Solving gives $d = \frac{w^Tx_0+b}{\|w\|}$.) This is the quantity SVM will directly maximize the minimum of, over all training points.

### Hard-margin optimization problem

Label classes as $y_i \in \{-1, +1\}$ (not $\{0,1\}$ — this sign convention makes the margin constraint below symmetric and clean). For a hyperplane that correctly separates the data with some margin, every point must satisfy $y_i(w^Tx_i+b) > 0$, and we additionally want every point to sit at least some fixed distance from the boundary. By rescaling $w$ and $b$ (the hyperplane $w^Tx+b=0$ is unchanged by any positive scaling of $(w,b)$), we can always choose the scale so that the *closest* points satisfy $y_i(w^Tx_i+b) = 1$ exactly. That fixes the margin's absolute width in terms of $\|w\|$: from the distance formula above, a point exactly on this boundary is at distance $1/\|w\|$ from the hyperplane, so the full margin width (both sides) is $2/\|w\|$.

Maximizing $2/\|w\|$ is equivalent to minimizing $\|w\|$, which (for smoothness — $\|w\|$ isn't differentiable at $w=0$, and the square gives a clean convex quadratic) is conventionally written as minimizing $\frac{1}{2}\|w\|^2$. This gives the **hard-margin SVM primal problem**:

$$\min_{w,b} \ \frac{1}{2}\|w\|^2 \qquad \text{subject to} \qquad y_i(w^Tx_i+b) \geq 1 \quad \forall i$$

Every constraint says "this point must be correctly classified *and* at least distance $1/\|w\|$ from the boundary." The points where the constraint is tight (equality, $y_i(w^Tx_i+b)=1$) are exactly the **support vectors** — geometrically, the points touching the edge of the margin corridor. This is a convex quadratic program: convex objective, linear constraints, so — like logistic regression's convex loss — there is one global optimum, findable by standard constrained-optimization solvers (used directly in the from-scratch implementation below).

### Soft margin: slack variables

Real data is rarely perfectly separable, and even when it is, a single noisy outlier deep inside the margin (or on the wrong side) can force the hard-margin problem to have no feasible solution at all, or to pick a needlessly narrow margin just to accommodate one point. The fix is to allow individual points to violate the margin, at a cost, via a **slack variable** $\xi_i \geq 0$ per point:

$$\min_{w,b,\xi} \ \frac{1}{2}\|w\|^2 + C\sum_{i=1}^m \xi_i \qquad \text{subject to} \qquad y_i(w^Tx_i+b) \geq 1-\xi_i, \quad \xi_i \geq 0 \ \ \forall i$$

$\xi_i = 0$ means point $i$ sits outside the margin as before; $0 < \xi_i \leq 1$ means it's inside the margin but still correctly classified; $\xi_i > 1$ means it's on the wrong side of the boundary entirely. The hyperparameter $C$ trades off margin width against how many/how badly points are allowed to violate it: large $C$ punishes violations heavily (pushes toward the hard-margin solution, narrower margin, less tolerance — lower bias/higher variance); small $C$ tolerates more violations in exchange for a wider margin (higher bias/lower variance) — the exact same bias-variance lever encountered as $\lambda$ in `04-regularization`, just parameterized inversely: $C$ plays the role $1/\lambda$ plays there.

Equivalently, eliminating $\xi_i$ (by noting the optimal $\xi_i = \max(0, 1-y_i(w^Tx_i+b))$) gives the unconstrained **hinge loss** form:

$$J(w,b) = \frac{1}{2}\|w\|^2 + C\sum_{i=1}^m \max\big(0,\ 1-y_i(w^Tx_i+b)\big)$$

which makes SVM's relationship to logistic regression explicit: both are "margin term + data-fit term" objectives on a linear score $w^Tx+b$, differing in which loss (hinge vs. log loss) penalizes the data-fit term.

### The kernel trick

A linear hyperplane can't separate data that isn't linearly separable — e.g. one class forming a ring around another. The classical fix is to map the data into a higher-dimensional space via some function $\phi(x)$ where it *becomes* linearly separable (analogous to `03-polynomial-regression`'s trick of adding polynomial features), then fit a linear SVM there. The problem: for a useful $\phi$ (e.g. one that implicitly represents all polynomial combinations of features up to some degree, or an infinite-dimensional feature space), explicitly computing $\phi(x)$ for every point can be computationally prohibitive or literally impossible (infinite dimensions).

The **kernel trick** notices that the SVM's dual optimization problem (and its prediction rule) only ever needs *dot products* $\phi(x_i)^T\phi(x_j)$ between transformed points — never $\phi(x)$ itself in isolation. If there's a function $K(x_i,x_j)$ that computes this dot product *directly from the original untransformed inputs*, without ever forming $\phi(x)$, then the entire algorithm can be run in the high-dimensional space's *geometry* while only ever doing arithmetic in the original, low-dimensional space.

**Polynomial kernel, worked out explicitly.** Take 2D inputs $x=(x_1,x_2)$ and the degree-2 mapping $\phi(x) = (x_1^2, \sqrt2\,x_1x_2, x_2^2)$ (all degree-2 monomials, with the $\sqrt2$ chosen to make the algebra below come out clean). Then

$$\phi(x)^T\phi(x') = x_1^2x_1'^2 + 2x_1x_2x_1'x_2' + x_2^2x_2'^2 = (x_1x_1'+x_2x_2')^2 = (x^Tx')^2$$

So $\phi(x)^T\phi(x') = (x^Tx')^2 = K(x,x')$ — the 3-dimensional dot product is computed with a single dot product and a square, entirely in the original 2D space. This generalizes to $K(x,x')=(x^Tx'+r)^d$, the polynomial kernel of degree $d$, which implicitly represents *all* monomials up to degree $d$ without ever constructing them.

**RBF kernel.** $K(x,x') = \exp(-\gamma\|x-x'\|^2)$ corresponds to an implicit $\phi$ mapping into an *infinite*-dimensional space (expand $\exp(-\gamma\|x-x'\|^2)$ via its Taylor series in $x^Tx'$ — every term is a polynomial kernel of increasing degree, summed to infinity). It would be literally impossible to compute $\phi(x)$ explicitly for this kernel, yet $K(x,x')$ is a single scalar exponential — trivial to compute. This is precisely why the kernel trick matters: it makes an infinite-dimensional feature space usable at all.

$\gamma$ controls how "local" the RBF kernel's notion of similarity is — large $\gamma$ means only very nearby points are considered similar (tight, wiggly decision boundaries, easy to overfit — see Failure modes); small $\gamma$ means even distant points are considered somewhat similar (smoother, more linear-like boundaries).

## Algorithm

1. Standardize features (margin geometry is scale-dependent — see Failure modes).
2. Choose a kernel: linear (no transformation, fastest, most interpretable) or nonlinear (polynomial, RBF) if the classes aren't linearly separable in the original feature space.
3. Choose $C$ (and, for RBF, $\gamma$) — typically by cross-validated grid search, trading margin width against training-fit tolerance.
4. Solve the (kernelized) quadratic program for $w,b$ (or, in dual form, for the per-point Lagrange multipliers $\alpha_i$ — points with $\alpha_i>0$ are exactly the support vectors; every other point has $\alpha_i=0$ and could be deleted from the training set without changing the fitted boundary at all).
5. Classify new points by the sign of $w^Tx+b$ (linear kernel) or the kernelized equivalent $\sum_i \alpha_iy_iK(x_i,x)+b$ (nonlinear kernel).

## From-scratch implementation

Implemented in `svm-from-scratch.ipynb`: a tiny 2D linearly-separable toy dataset (two well-separated clusters, $y_i\in\{-1,+1\}$), solved as the exact hard-margin constrained optimization problem derived above —

$$\min_{w,b} \ \tfrac{1}{2}\|w\|^2 \quad \text{s.t.} \quad y_i(w^Tx_i+b)\geq 1 \ \forall i$$

using `scipy.optimize.minimize` (SLSQP, which supports inequality constraints directly) on the primal variables $(w,b)$:

```python
from scipy.optimize import minimize, NonlinearConstraint

def objective(params):
    w = params[:2]
    return 0.5 * np.dot(w, w)

def constraint_fn(params):
    w, b = params[:2], params[2]
    return y * (X @ w + b) - 1   # >= 0 for every point

constraints = NonlinearConstraint(constraint_fn, 0, np.inf)
result = minimize(objective, x0=np.zeros(3), method='SLSQP', constraints=[constraints])
w_opt, b_opt = result.x[:2], result.x[2]
```

The notebook then identifies support vectors as the points where the constraint is (numerically) tight, $y_i(w^Tx_i+b)\approx 1$, and plots the fitted decision boundary $w^Tx+b=0$ together with the two margin boundaries $w^Tx+b=\pm1$ and the support vectors highlighted — visually confirming the margin corridor derived above and that only the support vectors touch its edges.

## Practical implementation

The three existing notebooks in this folder are the production-library step, mapped directly back to the math above:

- `Basic-SVC-Implementation.ipynb` fits `sklearn.svm.SVC(kernel='linear')` on a synthetic 2D dataset — this solves the exact same margin-maximization quadratic program as the from-scratch step, just via `sklearn`'s underlying `libsvm` solver (which works in the dual, and scales far better than a generic constrained optimizer like the SLSQP call above); `svc.coef_` recovers the fitted $w$ directly, since the kernel is linear.
- `SVM-Kernels-Implementation.ipynb` compares `SVC(kernel='rbf')`, `poly`, and `sigmoid` on non-linearly-separable data, exercising the kernel trick derived above — no explicit $\phi(x)$ is ever computed; only $K(x_i,x_j)$ is.
- `Support-Vector-Regression-Implementation.ipynb` uses `sklearn.svm.SVR`, the regression analogue: instead of maximizing a margin around a separating hyperplane, SVR fits a function and only penalizes points that fall *outside* an $\epsilon$-tube around it, with the identical hinge-style slack machinery ($\xi_i,\xi_i^*$) as soft-margin SVC applied to the two-sided regression residual — minimize $\frac12\|w\|^2 + C\sum_i(\xi_i+\xi_i^*)$ subject to $y_i-(w^Tx_i+b)\leq\epsilon+\xi_i$ and $(w^Tx_i+b)-y_i\leq\epsilon+\xi_i^*$.

## Experiment

**Hypothesis (stated before running):** as $C$ increases from small to large on a fixed dataset, the margin width should *shrink*, the number of support vectors should *decrease* (fewer points are allowed to sit inside a shrinking margin, and any point strictly outside the margin isn't a support vector), and training accuracy should *increase* (more tolerance for outliers removed) — reproducing the same bias-variance U-shape from `01-bias-variance-tradeoff` (referenced via `04-regularization`), just with $C$ playing the role of an inverse complexity/regularization knob instead of $\lambda$ or polynomial degree.

**Setup:** a moderately noisy, imperfectly-separable synthetic 2D binary classification dataset (some class overlap, e.g. `make_classification` with `class_sep` low enough to force some margin violations); fit `sklearn.svm.SVC(kernel='linear')` at a log-spaced sweep of $C$ values (e.g. $10^{-2}$ to $10^{3}$); record margin width ($2/\|w\|$ from `svc.coef_`), number of support vectors (`svc.support_vectors_.shape[0]`), and training accuracy at each $C$.

**Actual result:** see the corresponding cell in `Basic-SVC-Implementation.ipynb` (extend with the $C$-sweep) for this run's exact numbers and plot; the expected qualitative pattern is a monotonically shrinking margin width and shrinking support-vector count as $C$ grows, with training accuracy rising and then plateauing once $C$ is large enough that the margin has shrunk to just barely accommodate the non-separable points.

**Interpretation:** the number of support vectors is itself a rough proxy for model complexity/variance here — a small-$C$ model with many support vectors is relying on a wide, heavily-averaged consensus across many points (higher bias, lower variance, akin to underfitting), while a large-$C$ model with few support vectors is fit tightly to just the hardest-to-classify points near the boundary (lower bias, higher variance, akin to overfitting) — directly the same tradeoff, differently parameterized, as `04-regularization`'s $\lambda$ sweep.

**Limitations:** this uses one synthetic dataset and a linear kernel only; the relationship between $C$, margin width, and support-vector count for a nonlinear (RBF/poly) kernel is qualitatively similar but margin width is no longer as directly interpretable in the original feature space (it's a margin in the implicit, possibly infinite-dimensional, kernel-induced space) — a caveat worth stating rather than glossing over.

## Failure modes

- **Poor feature scaling breaks margin geometry.** Because the margin width is $2/\|w\|$ measured in the raw feature space's units, a feature on a scale of thousands (while another sits in $[0,1]$) dominates $\|w\|$ and therefore the geometric notion of "distance to nearest point" in a way that has nothing to do with which feature actually matters more — SVMs require standardized features (mean 0, unit variance) as a hard prerequisite, more strictly than logistic regression, precisely because the *entire* objective is a geometric distance argument.
- **RBF kernel with the wrong $\gamma$ overfits badly.** Too large a $\gamma$ makes the implicit similarity measure so local that the boundary wraps tightly around individual training points (near-zero training error, poor generalization — a direct visual analogue of a high-degree polynomial in `03-polynomial-regression` wiggling through every training point); too small a $\gamma$ makes the RBF kernel behave almost like a linear kernel, unable to capture real nonlinear structure. $\gamma$ needs cross-validated tuning jointly with $C$, not independently.
- **Poor scaling to large $N$.** The classical dual-form QP solvers scale roughly quadratic-to-cubic in the number of training points, because the kernel (Gram) matrix is $N\times N$ — this becomes prohibitively slow and memory-hungry well before the millions-of-rows scale that tree ensembles (gradient boosting) or linear models handle comfortably, which is a major reason SVMs are less common than they once were for very large tabular datasets, despite still being highly competitive on small-to-medium data.
- **No natural probability output.** Unlike logistic regression, SVM's raw output is a signed distance, not a probability — `sklearn`'s `SVC(probability=True)` bolts on Platt scaling (literally, fitting a logistic regression on top of the SVM's decision function — the same calibration technique mentioned in `06-logistic-regression`) after the fact, at extra computational cost, rather than producing calibrated probabilities as a byproduct of fitting.

## Real-world usage

- Still a strong, often first-choice, baseline for small-to-medium tabular datasets with a genuine geometric separation between classes, where the training-set size keeps the $O(N^2)$–$O(N^3)$ kernel computation tractable — text classification with modest vocabularies, bioinformatics (gene expression classification), image classification on small curated datasets.
- The kernel trick itself, independent of SVM specifically, is a general pattern reused anywhere a similarity function between two objects is easier to define/compute directly than an explicit feature embedding is (kernel PCA, Gaussian processes, some recommender-system similarity metrics).
- At scale (millions of rows, especially with many features), SVMs are usually displaced by linear models (with SGD-based solvers) or tree ensembles (gradient boosting, random forests) precisely because of the $O(N^2)$+ scaling failure mode above — knowing *why* SVMs don't scale is itself part of knowing when to reach for the alternative instead.

## Mental model

SVM doesn't just ask "can I draw a line that separates these classes" — it asks "what's the widest street I can drive between them, with the line running down the middle." The points touching the curb on either side (the support vectors) are the only ones that matter for where the street sits; every other point could vanish without moving it. The kernel trick lets that "street" be measured in a bent, higher-dimensional version of the data's geometry — computed entirely through similarity scores between original points, never by actually constructing that higher-dimensional space.

## Questions to think about

1. Why does the hard-margin SVM's constraint $y_i(w^Tx_i+b)\geq 1$ (rather than $\geq 0$) matter for pinning down a unique scale for $w$ and $b$, and what would go wrong with the optimization if the constraint were instead written as $\geq 0$?
2. A dataset is linearly separable, but only by a razor-thin margin, with two points from different classes sitting almost on top of each other. Would you expect a hard-margin or soft-margin SVM to generalize better here, and why does the answer depend on whether that near-collision is a genuine pattern in the data or a single noisy measurement?
3. Explain why increasing $C$ and increasing $\gamma$ (for an RBF kernel) both push the model toward overfitting, but via mechanistically different routes — one changes how much margin violation is tolerated, the other changes what "nearby" means in the first place.
4. The polynomial kernel example in this topic derives $K(x,x')=(x^Tx')^2$ for a specific 3-term $\phi$. Without expanding it explicitly, argue why $K(x,x')=(x^Tx')^d$ for general $d$ must correspond to *some* valid dot product in *some* higher-dimensional feature space (this property is what makes a function usable as a kernel at all).
5. Given this topic's Experiment result (fewer support vectors at large $C$) and `04-regularization`'s coefficient-shrinkage result (smaller coefficients at large $\lambda$), both parameters are described as controlling a bias-variance tradeoff — but they move it in *opposite* directions relative to their own magnitude (large $C$ = less regularization; large $\lambda$ = more). Why does this inverse relationship exist, and what should you check before assuming a "large hyperparameter value" always means "more regularized" across two different algorithms?
