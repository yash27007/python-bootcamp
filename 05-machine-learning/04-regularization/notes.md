# Regularization Notes

## 1) Ridge Regression (L2 Regularization)

### Model and notation
We use the linear hypothesis:

$$
 h_\theta(x) = \theta^T x
$$

with $x = [1, x_1, \dots, x_n]^T$ and $\theta = [\theta_0, \theta_1, \dots, \theta_n]^T$.

### Objective function
Ridge regression adds an $L2$ penalty to the least-squares objective:

$$
 J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2 + \frac{\lambda}{2m} \sum_{j=1}^{n} \theta_j^2
$$

- The bias term $\theta_0$ is typically **not** regularized.
- $\lambda \ge 0$ controls the strength of regularization.

### Interpretation
- Encourages smaller weights, which reduces variance and helps with multicollinearity.
- Does **not** force coefficients to exactly zero, so it usually keeps all features.

### Gradient descent updates

$$
 \frac{\partial J}{\partial \theta_0} = \frac{1}{m} \sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right)
$$

$$
 \frac{\partial J}{\partial \theta_j} = \frac{1}{m} \sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right) x_j^{(i)} + \frac{\lambda}{m} \theta_j\quad (j \ge 1)
$$

Update rule:

$$
 \theta_j := \theta_j - \alpha \left[\frac{1}{m} \sum_{i=1}^m \left(h_\theta(x^{(i)}) - y^{(i)}\right) x_j^{(i)} + \frac{\lambda}{m} \theta_j\right]
$$

### Closed-form solution
With feature matrix $X \in \mathbb{R}^{m \times (n+1)}$ and target vector $y$:

$$
 \theta = \left(X^T X + \lambda I'\right)^{-1} X^T y
$$

- $I'$ is the identity matrix with the first diagonal element set to 0 (so $\theta_0$ is not regularized).

### Probabilistic view (MAP)
Ridge is equivalent to a Gaussian prior on $\theta$:

$$
 \theta_j \sim \mathcal{N}(0, \sigma^2)\quad (j \ge 1)
$$

### Practical notes
- Standardize features for fair regularization across dimensions.
- Larger $\lambda$ shrinks coefficients more aggressively.

---

## 2) Lasso Regression and Elastic Net

### Lasso (L1 regularization)

#### Objective function

$$
 J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2 + \frac{\lambda}{m} \sum_{j=1}^{n} |\theta_j|
$$

- Bias term $\theta_0$ is typically not regularized.
- L1 penalty can force coefficients to exactly zero (feature selection).

#### Key properties
- Produces **sparse** models.
- Useful when many features are irrelevant.
- Not differentiable at $\theta_j = 0$, so solvers use coordinate descent or subgradients.

### Elastic Net (L1 + L2 regularization)

Elastic Net combines L1 and L2 penalties to get sparsity and stability:

$$
 J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2
 + \frac{\lambda}{m} \left[ \alpha \sum_{j=1}^{n} |\theta_j| + \frac{1-\alpha}{2} \sum_{j=1}^{n} \theta_j^2 \right]
$$

- $\alpha \in [0, 1]$ controls the mix:
  - $\alpha = 1$ gives Lasso.
  - $\alpha = 0$ gives Ridge.
- Encourages sparsity (L1) while keeping correlated features together (L2).

#### Why Elastic Net?
- Lasso tends to pick one feature from a group of correlated features and ignore the rest.
- Elastic Net keeps groups of correlated features together while still performing feature selection.

### Optimization notes
- Lasso and Elastic Net are commonly solved with coordinate descent.
- Standardize features before training for consistent regularization.

### Choosing $\lambda$ and $\alpha$
- Use cross-validation to select hyperparameters.
- Larger $\lambda$ increases regularization strength.
- For Elastic Net, tune both $\lambda$ and $\alpha$ for the best bias-variance tradeoff.
