# 06 – Logistic Regression

## Problem

You need to predict a class, not a number: will this email be spam, will this patient have the disease, will this transaction be fraud. Often you don't just want a hard yes/no — you want a **probability**, because the cost of being wrong differs by direction (missing a cancer diagnosis is worse than a false alarm) and a probability lets you weigh that cost explicitly. So the target isn't a continuous quantity in $(-\infty, \infty)$ like `02-linear-regression`'s house price — it's a number that must live in $[0, 1]$ and behave like a probability. **How do you predict a bounded, well-calibrated probability from features, using a model whose parameters you can still fit and interpret the way you fit linear regression's?**

## Intuition

Suppose you're predicting whether a student passes an exam from hours studied. You'd like the model to output something like "with 2 hours studied, there's a 30% chance of passing; with 8 hours, 92%." Two things are true about that output that weren't true of linear regression's raw prediction:

1. It must never leave $[0, 1]$ — "110% chance of passing" is meaningless.
2. Its rate of change should slow down near the extremes — going from 1 to 2 hours of study should move the probability a lot if the student is currently around 50/50, but going from 9 to 10 hours shouldn't move it much, because the probability is already pinned near 1.

That second property is exactly the shape of an S-curve: steep in the middle, flat at the ends. Logistic regression's core idea is to keep the familiar linear form $\theta^Tx$ from linear regression — a weighted sum of features — but pass it through a squashing function that bends it into that S-curve before treating it as a probability. The linear part still does the "combine evidence from features" work; the squashing function only fixes the output range and shape.

**Worked example.** A spam classifier scores an email at $z = \theta^Tx = 2.0$ (some combination of "contains 'free'", "many exclamation marks", etc., weighted and summed). Passed through the S-curve, $z=2.0$ becomes a probability of about $0.88$ — "88% chance this is spam." If a different email scores $z=-3.0$, that becomes about $0.05$ — "5% chance." The linear score can be any real number; the probability it produces is always bounded.

## Why simpler approaches fail

The obvious first attempt: fit ordinary linear regression on a $0$/$1$-encoded target, then threshold the output at $0.5$. This fails in three concrete, connected ways:

**Unbounded output.** Linear regression's prediction $\theta^Tx$ ranges over all of $\mathbb{R}$. Nothing stops it from predicting $-0.3$ or $1.8$ for a "probability." You can clip these to $[0,1]$, but then the model isn't actually optimizing for a sensible probability anywhere near the clipped region — it was fit to minimize squared error on raw, unclipped numbers, so the clipping is a patch applied after the fact, not something the fitting procedure knows about or accounts for.

**Wrong loss shape for a probability target.** Linear regression's objective is squared error, which treats every unit of prediction error the same regardless of how confident that error was. A model predicting $0.99$ for a true label of $0$ (a *confident, wrong* prediction, the kind you most want to punish) gets squared-error cost $(0.99-0)^2 = 0.98$ — barely more than a model predicting $0.6$ for the same wrong label, cost $(0.6-0)^2=0.36$: roughly a 2.7x penalty ratio between "barely wrong" and "confidently, badly wrong." That's a weak signal for something that should be treated very differently. (Section "Mathematical foundation" below derives a loss that punishes confident wrong answers far more sharply — the ratio there is unbounded, not 2.7x.)

**Sensitivity to outliers in the wrong dimension.** Because linear regression fits $\theta^Tx$ to hit $0$ or $1$ as closely as possible in squared-error terms, a single far-outlying feature value (an email with an enormous count of some rare word, say) can drag the entire fitted line — and therefore every other point's threshold-0.5 decision — because squared error weights large deviations quadratically. A model built around probabilities shouldn't let one point's raw feature magnitude have that much leverage over everyone else's classification.

What's needed: an output that's mathematically guaranteed to stay in $[0,1]$ regardless of the linear score's magnitude, paired with a loss function that is derived from the actual probabilistic structure of a yes/no outcome rather than borrowed from continuous-value regression.

## Mathematical foundation

### From log-odds to the sigmoid

Start from the quantity that actually behaves the way we want the linear model to predict: **log-odds**. The odds of an event with probability $p$ are

$$\text{odds} = \frac{p}{1-p}$$

Odds range over $(0, \infty)$ — still not what we want, since $\theta^Tx$ ranges over all of $\mathbb{R}$, including negative numbers. Taking the log fixes this:

$$\text{logit}(p) = \ln\left(\frac{p}{1-p}\right) \in (-\infty, \infty)$$

This is exactly the range of $\theta^Tx$. So logistic regression's defining assumption is: **the log-odds of the outcome are linear in the features**,

$$\ln\left(\frac{p}{1-p}\right) = \theta^Tx = \theta_0 + \theta_1x_1 + \dots + \theta_nx_n$$

Not the probability itself — the log-odds. This is the key structural choice, and everything else (the S-curve, the loss function) follows from inverting and fitting this equation.

**Solving for $p$** (the sigmoid function). Exponentiate both sides and rearrange:

$$\frac{p}{1-p} = e^{\theta^Tx} \implies p = e^{\theta^Tx}(1-p) \implies p + pe^{\theta^Tx} = e^{\theta^Tx} \implies p = \frac{e^{\theta^Tx}}{1+e^{\theta^Tx}} = \frac{1}{1+e^{-\theta^Tx}}$$

Define $z = \theta^Tx$ and

$$\sigma(z) = \frac{1}{1+e^{-z}}$$

This is the **sigmoid function**. Check its limiting behavior: as $z \to \infty$, $e^{-z}\to 0$ so $\sigma(z) \to 1$; as $z \to -\infty$, $e^{-z}\to\infty$ so $\sigma(z)\to 0$; at $z=0$, $\sigma(0) = 1/2$. It is monotonic and strictly bounded in $(0,1)$ for every finite $z$ — the unboundedness problem from the previous section is fixed by construction, not by clipping.

**Coefficient interpretation** falls directly out of the log-odds assumption: a one-unit increase in $x_j$ increases the log-odds by $\theta_j$, and multiplies the odds by $e^{\theta_j}$ (holding other features fixed). If $\theta_j = 0.5$, each extra unit of $x_j$ multiplies the odds of the positive class by $e^{0.5}\approx 1.65$ — a 65% odds increase.

### Binary cross-entropy from maximum likelihood

Now derive the loss, rather than borrowing squared error. Treat each label $y^{(i)}\in\{0,1\}$ as a single draw from a **Bernoulli** distribution with success probability $p^{(i)} = \sigma(\theta^Tx^{(i)})$. The probability of observing the actual label under this model is

$$P(y^{(i)}\mid x^{(i)}) = \left(p^{(i)}\right)^{y^{(i)}}\left(1-p^{(i)}\right)^{1-y^{(i)}}$$

(check: if $y^{(i)}=1$ this reduces to $p^{(i)}$; if $y^{(i)}=0$ it reduces to $1-p^{(i)}$ — exactly the Bernoulli PMF written in one expression using the exponent trick). Assuming the $m$ training examples are independent, the **likelihood** of the whole dataset under parameters $\theta$ is the product

$$L(\theta) = \prod_{i=1}^m \left(p^{(i)}\right)^{y^{(i)}}\left(1-p^{(i)}\right)^{1-y^{(i)}}$$

Maximum likelihood estimation picks $\theta$ to maximize $L(\theta)$. Products of many small numbers underflow numerically and are hard to differentiate, so take the log (monotonic, so it preserves the maximizer) — the **log-likelihood**:

$$\ell(\theta) = \sum_{i=1}^m \left[y^{(i)}\ln p^{(i)} + (1-y^{(i)})\ln(1-p^{(i)})\right]$$

Maximizing $\ell(\theta)$ is the same as minimizing $-\ell(\theta)$, and dividing by $m$ turns a sum into an average (doesn't change the minimizer, keeps the scale independent of dataset size). This defines the **cost function**, binary cross-entropy / log loss:

$$J(\theta) = -\frac{1}{m}\sum_{i=1}^m \left[y^{(i)}\ln\left(h_\theta(x^{(i)})\right) + (1-y^{(i)})\ln\left(1-h_\theta(x^{(i)})\right)\right], \qquad h_\theta(x) = \sigma(\theta^Tx)$$

This is not an arbitrary alternative to squared error — it is *the* loss implied by treating the labels as genuinely Bernoulli-distributed and asking for the maximum-likelihood fit, which is why it's the "correct" loss for a probability target in a way squared error on a linear model never was. It also directly fixes the "confident wrong answer" weakness above: as $h_\theta(x^{(i)}) \to 0$ when $y^{(i)}=1$ (confidently wrong), $-\ln(h_\theta(x^{(i)})) \to \infty$ — an unbounded, not merely 2.7x, penalty.

**Convexity.** $J(\theta)$ is convex in $\theta$ (a consequence of $-\ln(\sigma(\cdot))$ and $-\ln(1-\sigma(\cdot))$ each being convex, and sums of convex functions being convex) — there is one global minimum, no bad local minima to get stuck in, unlike the non-convex loss surfaces of neural networks encountered later in this course.

### Gradient of the loss

To fit $\theta$ by gradient descent we need $\nabla_\theta J(\theta)$. Using the chain rule and the identity $\sigma'(z) = \sigma(z)(1-\sigma(z))$ (differentiate $\sigma(z)=(1+e^{-z})^{-1}$ directly to confirm this), the per-example derivative of $-\left[y\ln\sigma(z) + (1-y)\ln(1-\sigma(z))\right]$ with respect to $z=\theta^Tx$ simplifies remarkably:

$$\frac{\partial}{\partial z}\Big[-y\ln\sigma(z)-(1-y)\ln(1-\sigma(z))\Big] = \sigma(z) - y = h_\theta(x) - y$$

(the sigmoid's own derivative cancels cleanly against the $\ln$'s derivative — this cancellation is exactly *why* log loss was paired with the sigmoid rather than some other squashing function). Then by the chain rule through $z=\theta^Tx$, $\partial z/\partial\theta_j = x_j$, giving the full gradient over $m$ examples:

$$\nabla_\theta J(\theta) = \frac{1}{m}\mathbf X^T\left(h_\theta(\mathbf X) - \mathbf y\right)$$

This has exactly the same *form* as linear regression's gradient (`02-linear-regression/notes.md`) — "prediction minus actual, times features" — even though $h_\theta$ here is nonlinear (sigmoid-of-linear). The difference is that logistic regression has **no closed-form solution** analogous to the normal equation, because setting this gradient to zero has no algebraic solution for $\theta$ — $\theta$ must be found iteratively.

## Algorithm

1. Standardize features (unscaled features make gradient descent slow and unstable, same reasoning as `04-regularization`).
2. Initialize $\theta$ (e.g. to zero).
3. Repeat until convergence:
   - Compute $z^{(i)} = \theta^Tx^{(i)}$ for all $i$, then $h_\theta(x^{(i)}) = \sigma(z^{(i)})$.
   - Compute the gradient $\nabla_\theta J(\theta) = \frac{1}{m}\mathbf X^T(h_\theta(\mathbf X)-\mathbf y)$.
   - Update $\theta := \theta - \alpha \nabla_\theta J(\theta)$.
4. At prediction time, classify $\hat y = 1$ if $h_\theta(x) \geq 0.5$ (equivalently $\theta^Tx \geq 0$ — the decision boundary is the hyperplane $\theta^Tx=0$), else $\hat y = 0$; adjust the threshold away from $0.5$ when false positives and false negatives have different costs (see Failure modes / Real-world usage).

In practice, `lbfgs` (a quasi-Newton method, scikit-learn's default) or Newton-Raphson/IRLS converge faster than plain batch gradient descent by using curvature information, but batch gradient descent is what makes the mechanics visible, and is what's implemented from scratch below.

## From-scratch implementation

Implemented in `logistic-regression.ipynb`: plain NumPy batch gradient descent fitting the sigmoid model above on a small, linearly-separable 2D toy dataset, using exactly the update rule derived in "Algorithm":

```python
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def fit_logistic_regression(X, y, lr=0.1, n_iter=500):
    m, n = X.shape
    Xb = np.hstack([np.ones((m, 1)), X])   # bias column
    theta = np.zeros(n + 1)
    history = []
    for _ in range(n_iter):
        z = Xb @ theta
        h = sigmoid(z)
        grad = (1 / m) * Xb.T @ (h - y)
        theta -= lr * grad
        history.append(theta.copy())
    return theta, history
```

The notebook plots the decision boundary ($\theta^Tx=0$, a straight line since $x$ is 2D) at several points during training, showing it start near the origin (from the zero initialization) and rotate/shift toward the boundary that separates the two classes as gradient descent proceeds — a direct visual confirmation that minimizing $J(\theta)$ is equivalent to finding the separating hyperplane implied by the log-odds assumption.

## Practical implementation

`logistic-regression.ipynb` fits `sklearn.linear_model.LogisticRegression` on a synthetic 10-feature binary classification dataset (`make_classification`), then tunes `penalty`, `C` (`sklearn`'s $1/\lambda$ regularization strength — see `04-regularization/notes.md`), and `solver` via `GridSearchCV`/`RandomizedSearchCV` over `StratifiedKFold` cross-validation. This maps directly back to the from-scratch step: `LogisticRegression`'s default solver (`lbfgs`) minimizes the identical $J(\theta)$ derived above, just with a smarter (quasi-Newton, curvature-aware) optimizer than plain batch gradient descent, plus an optional L2/L1/elastic-net penalty term added to $J(\theta)$ exactly as in `04-regularization`.

`multiclass-classification.ipynb` extends the binary case to $K>2$ classes via `OneVsOneClassifier(LogisticRegression())` — training $\binom{K}{2}$ binary classifiers, one per pair of classes, and predicting by majority vote. The mathematically cleaner alternative, **softmax (multinomial) regression**, generalizes the sigmoid directly: instead of one log-odds equation, there are $K$ linear scores $z_k = \theta_k^Tx$, and

$$P(y=k\mid x) = \frac{e^{z_k}}{\sum_{j=1}^K e^{z_j}}$$

which reduces exactly to the sigmoid when $K=2$ (subtracting the two-class scores and simplifying recovers $\sigma(z_1-z_2)$) — softmax is the natural multiclass generalization of the same log-odds idea, producing probabilities across all $K$ classes that sum to $1$, rather than $K(K-1)/2$ pairwise votes that don't.

## Experiment

**Hypothesis (stated before running):** on the same small, linearly-separable 2D toy dataset, the from-scratch gradient-descent decision boundary and scikit-learn's `LogisticRegression` decision boundary should coincide within numerical tolerance — both are minimizing the same convex $J(\theta)$, so (given enough gradient-descent iterations to converge, and no regularization mismatch) there is only one global minimum for both to find.

**Setup:** generate a small 2D two-class dataset with `make_blobs` or `make_classification`, fit both (a) the from-scratch batch-gradient-descent implementation above, run for enough iterations to converge, and (b) `sklearn.linear_model.LogisticRegression(penalty=None)` (no regularization, to match the unpenalized from-scratch objective exactly) on identical training data. Compare the two fitted decision boundaries (the line $\theta^Tx=0$ for each) visually and by comparing predicted classes on a grid of test points.

**Actual result:** see `logistic-regression.ipynb`'s from-scratch section — the two boundaries overlap closely, and the fraction of grid points where the two models disagree is near zero, confirming both procedures converge to (numerically close to) the same optimum of the same convex objective.

**Interpretation:** this is expected precisely because $J(\theta)$ is convex with one global minimum — any correct optimizer reaching convergence should land in the same place, whether it's plain gradient descent or `lbfgs`. Small residual differences come from the from-scratch implementation using a fixed learning rate and iteration count rather than `lbfgs`'s convergence-tolerance-based stopping.

**Limitations:** this compares boundaries on a single small synthetic dataset chosen to be linearly separable and low-dimensional specifically so the boundary can be visualized directly; it doesn't test convergence behavior on non-separable, high-dimensional, or class-imbalanced data, where the two optimizers' convergence paths (and any regularization defaults) could diverge more visibly.

## Failure modes

- **Perfect separability causes coefficients to diverge.** If the training data can be perfectly separated by a hyperplane, $J(\theta)$ can be driven arbitrarily close to $0$ by scaling $\theta$ to arbitrarily large magnitude in the correct direction — the sigmoid just gets pushed harder toward $0$ or $1$ with no penalty for doing so, so gradient descent never converges to a finite optimum; $\|\theta\|$ grows without bound. This is the *exact same failure mode* the closed-form Ridge solution in `04-regularization/notes.md` was shown to fix for a singular $\mathbf X^T\mathbf X$ — adding an L2 (or L1) penalty term to $J(\theta)$ bounds $\|\theta\|$ and gives a well-defined finite optimum even under perfect separation.
- **Class imbalance biases the decision threshold.** With, say, 99% negative / 1% positive training data, a model that predicts "negative" for everything achieves 99% accuracy while being useless, and even a well-fit model's default $0.5$ threshold is systematically miscalibrated for the minority class's actual prevalence. **Accuracy stops being a useful metric here** — instead, evaluate with metrics built from the confusion matrix (TP, FP, TN, FN counts):
  $$\text{Precision} = \frac{TP}{TP+FP} \qquad \text{Recall} = \frac{TP}{TP+FN} \qquad \text{Specificity} = \frac{TN}{TN+FP} \qquad F_1 = 2\cdot\frac{\text{Precision}\cdot\text{Recall}}{\text{Precision}+\text{Recall}}$$
  Precision answers "of everything flagged positive, how much really was" (matters when false positives are costly, e.g. flagging a legitimate email as spam); recall answers "of everything actually positive, how much was caught" (matters when false negatives are costly, e.g. missing an actual cancer case); specificity answers the mirror question for the negative class — "of everything actually negative, how much did the model correctly call negative." The **ROC curve** (recall vs. false-positive-rate $FP/(FP{+}TN) = 1-\text{Specificity}$, swept over thresholds) and its AUC summarize discriminative power independent of any one threshold — but AUC can look misleadingly good under heavy imbalance, where PR-AUC is the more honest summary. Mitigations: `class_weight='balanced'` (reweights each class's contribution to $J(\theta)$ inversely to its frequency, so the minority class's errors count more), oversampling the minority class (e.g. SMOTE) or undersampling the majority class, and — most directly — replacing the default $0.5$ cutoff with a **cost-derived threshold**: given $\text{cost}_{FP}$ and $\text{cost}_{FN}$, sweep candidate thresholds and pick the one minimizing $FP\cdot\text{cost}_{FP} + FN\cdot\text{cost}_{FN}$ on validation data, rather than assuming the two error types are equally costly.
  Two ways to score probability *quality* itself, rather than thresholded classifications: log loss (the $J(\theta)$ derived above — heavily penalizes confident wrong predictions) and the **Brier score**, the mean squared error of the predicted probabilities against the actual $0/1$ labels,
  $$\text{Brier} = \frac{1}{m}\sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2$$
  which ranges from $0$ (perfect) to $1$, with $0.25$ as the score of a constant $0.5$ prediction on balanced data. It's easier to interpret than log loss (it's literally MSE applied to probabilities instead of continuous targets — the same objective linear regression uses, just evaluating a probability rather than driving the fit), but log loss remains the more "proper" scoring rule for probabilities because, unlike Brier score, it is uniquely minimized in expectation only by the true probability and its unbounded penalty near $0$/$1$ makes it far more sensitive to confidently wrong predictions.
- **Sensitive to unscaled features and outliers** for the same underlying reason described in "Why simpler approaches fail": the linear score $\theta^Tx$ that feeds the sigmoid can still be dominated by one large-magnitude feature or outlying point, distorting the fitted boundary even though the *output* is now correctly bounded.
- **Multicollinearity ill-conditions the optimization.** When two or more features are highly correlated (e.g. "height in feet" and "height in inches" — near-duplicate information), the curvature matrix that Newton-type solvers (`lbfgs`, Newton-Raphson/IRLS) use to take steps — the Hessian of $J(\theta)$ — becomes near-singular: along the direction that mixes the correlated features, the loss surface is almost flat, so many different $(\theta_j,\theta_k)$ combinations give nearly the same $J(\theta)$. An ill-conditioned Hessian means those solvers' Newton step, which divides by curvature, gets multiplied by a near-inverse of a near-zero matrix — coefficient estimates become unstable and highly sensitive to small changes in the data, even though predicted probabilities barely move. This is the same Hessian-conditioning failure the closed-form Ridge derivation in `04-regularization/notes.md` addresses for a singular $\mathbf X^T\mathbf X$ in linear regression — L2 regularization here adds $\lambda I$ to the effective curvature, restoring a well-conditioned optimization. Mitigation: drop or combine redundant features, or add L2 regularization.
- **Purely linear decision boundary.** The boundary $\theta^Tx=0$ is a hyperplane; logistic regression cannot separate classes arranged in concentric circles or an XOR pattern without manually added polynomial/interaction features (`03-polynomial-regression`-style feature engineering) or a fundamentally different, nonlinear model (`07-svm`'s kernel trick, or neural networks later in this course).

## Real-world usage

- The default first model for binary classification on tabular data — fast to train, coefficients are directly interpretable as log-odds effects, and probability outputs support cost-sensitive thresholding (fraud detection, credit approval, medical screening) in a way a pure classifier without probabilities cannot.
- Regularized logistic regression (`04-regularization`'s L1/L2/elastic-net machinery applied to $J(\theta)$) is standard in high-dimensional settings (text classification with bag-of-words features, genomics) both for stabilizing coefficients and for interpretable feature selection.
- Calibration techniques (Platt scaling — literally fitting a logistic regression on top of another model's raw scores — and isotonic regression) exist specifically because many other classifiers' outputs are *not* well-calibrated probabilities the way logistic regression's are by construction; logistic regression is often the calibration tool of choice for other models.
- Softmax regression's exact functional form reappears as the final layer of virtually every neural network classifier later in this course — logistic/softmax regression is a one-layer special case of that architecture.

## Mental model

Logistic regression is linear regression on the log-odds of the outcome: keep the familiar "weighted sum of features" from linear regression, but fit it to predict $\ln\frac{p}{1-p}$ instead of $y$ directly, then invert that relationship (the sigmoid) to recover a probability that is mathematically guaranteed to live in $[0,1]$ — and fit it with the loss that maximum likelihood on a Bernoulli outcome actually implies, not squared error borrowed from a different kind of target.

## Questions to think about

1. Why does the gradient of the binary cross-entropy loss, $\nabla_\theta J(\theta) = \frac{1}{m}\mathbf X^T(h_\theta(\mathbf X)-\mathbf y)$, have the exact same algebraic form as linear regression's gradient, even though $h_\theta$ is nonlinear? What does this say about the relationship between the choice of loss function and the choice of link function (sigmoid) in this derivation?
2. If two features are perfectly correlated, what happens to the shape of $J(\theta)$'s minimum (a unique point vs. a valley), and how would L2 regularization change the answer? (Connect this back to `04-regularization`'s multicollinearity discussion.)
3. A colleague trains a logistic regression classifier and reports 99.2% accuracy, but the positive class is only 0.5% of the data. What single number would you ask for next, and why might accuracy alone be actively misleading here?
4. Explain, using the log-odds framing, why doubling every feature's scale (without changing the underlying relationship) changes the fitted coefficients $\theta_j$ but does not change the model's predicted probabilities or decision boundary — and why this is *not* the same as saying feature scaling doesn't matter for logistic regression.
5. Two datasets are both linearly separable, but in one the classes are separated by a wide gap and in the other by a razor-thin gap. Unregularized logistic regression fits both perfectly on training data with $\|\theta\|\to\infty$ in both cases — what does this suggest about why logistic regression alone (unlike `07-svm`) doesn't distinguish "generalizes well" from "merely separates training data"?
