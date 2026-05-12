# 02 - Linear Regression

## Notation and dataset setup

- Training set: $\{(x^{(i)}, y^{(i)})\}_{i=1}^{m}$, where $m$ is the number of samples.
- For simple regression, $x^{(i)}$ is a scalar. For multiple regression, $x^{(i)} \in \mathbb{R}^d$.
- We add a bias term by defining $x_0^{(i)} = 1$ so we can write the model in vector form.
- Parameter vector: $\theta = [\theta_0, \theta_1, \dots, \theta_d]^T$.
- Hypothesis (model) function: $h_\theta(x)$.

## 141. Simple Linear Regression Introduction

Simple linear regression models the relationship between **one input feature** and a **continuous target**.

### Problem setup

- Input feature: $x$ (e.g., house size)
- Target variable: $y$ (e.g., house price)
- Objective: learn a function that predicts $y$ from $x$ with minimal error.

### Model (hypothesis)

$$h_\theta(x) = \theta_0 + \theta_1 x$$

Interpretation:
- $\theta_0$ is the intercept (prediction when $x = 0$)
- $\theta_1$ is the slope (change in prediction per unit change in $x$)

### What learning means

Learning is the process of choosing $\theta_0$ and $\theta_1$ so that the line fits the training data as closely as possible according to a chosen error measure.

### Residuals

The error for a single example is the **residual**:

$$e^{(i)} = y^{(i)} - h_\theta(x^{(i)})$$

Residuals show how far each point is from the fitted line.

### Typical assumptions (for inference)

- The relationship between $x$ and $y$ is approximately linear.
- Errors are independent across samples.
- Errors have constant variance (homoscedasticity).
- Errors are roughly normally distributed (important for confidence intervals).

## 142. Understanding Simple Linear Regression Equations

### Slope-intercept form

$$h_\theta(x) = \theta_0 + \theta_1 x$$

- $\theta_1$ controls the tilt of the line.
- $\theta_0$ controls where the line crosses the $y$-axis.

### Vector form (basis for multiple regression)

Define:

$$\mathbf{x} = [1, x]^T, \quad \theta = [\theta_0, \theta_1]^T$$

Then:

$$h_\theta(x) = \theta^T \mathbf{x}$$

### Geometric view

In 2D, the model is a line that splits the plane into two sides based on the sign of $h_\theta(x) - y$. The best line minimizes the total error from points to the line.

### Residuals and error aggregation

Residuals are combined into a single cost value so we can optimize a single objective.

## 143. Cost Function

The cost function measures how well the model fits the data.

### Mean Squared Error (MSE)

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2$$

Often written with a $\frac{1}{2m}$ factor to simplify gradients:

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)}\right)^2$$

Why squared error:
- Penalizes larger errors more than smaller ones.
- Is differentiable everywhere, which enables gradient-based optimization.
- Produces a convex surface for linear regression, so there is a single global minimum.

### Why we use a loss function

A loss (or cost) function is the **single number** that summarizes model quality. Without it, there is no clear direction for optimization.

Key reasons:
- It **turns prediction errors into a numeric objective** we can minimize.
- It provides **gradients** (slopes) that tell us how to update parameters.
- It allows **fair comparison** between models and hyperparameters.

### Cost function example (numeric)

Suppose we have three samples:

$$\{(1, 2), (2, 3), (3, 5)\}$$

Let $h_\theta(x) = \theta_0 + \theta_1 x$ with $\theta_0 = 0$, $\theta_1 = 1$.

Predictions:
- $h_\theta(1) = 1$
- $h_\theta(2) = 2$
- $h_\theta(3) = 3$

Errors $h_\theta(x^{(i)}) - y^{(i)}$:
- $1 - 2 = -1$
- $2 - 3 = -1$
- $3 - 5 = -2$

Squared errors: $1, 1, 4$, sum = $6$.

Using $\frac{1}{2m}$:

$$J(\theta) = \frac{1}{2 \cdot 3} \cdot 6 = 1$$

Using $\frac{1}{m}$:

$$J(\theta) = \frac{1}{3} \cdot 6 = 2$$

Both give the **same minimum**, just scaled differently.

### Alternative losses (for context)

- **MAE:** less sensitive to outliers but not differentiable at 0.
- **Huber loss:** combines MSE and MAE behavior.

## 144. Convergence Algorithm (Gradient Descent)

Gradient descent is the standard iterative method used to minimize $J(\theta)$.

### Core idea

- Start with an initial guess for $\theta$.
- Move $\theta$ in the direction that decreases $J(\theta)$ the most.
- Repeat until the cost stops decreasing meaningfully.

### Update rules (simple linear regression)

For $m$ samples, the batch gradient descent updates are:

$$\theta_0 := \theta_0 - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)}\right)$$

$$\theta_1 := \theta_1 - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)}\right) x^{(i)}$$

Where:
- $\alpha$ is the learning rate.
- $m$ is the number of training examples.

### General update (multiple regression)

For each parameter $\theta_j$:

$$\theta_j := \theta_j - \alpha \cdot \frac{1}{m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)}\right) x_j^{(i)}$$

### Why it converges for linear regression

The cost function for linear regression is convex. With a proper learning rate, gradient descent converges to the global minimum.

### Why we use gradient descent

- **Scales to large datasets** where matrix inversion is expensive.
- **Works for many models** where no closed-form solution exists.
- **Online or streaming updates** are possible.
- **Flexibility** to add regularization or custom loss functions.

### Gradient descent example (one update step)

Using the same dataset:

$$\{(1, 2), (2, 3), (3, 5)\}$$

Start with $\theta_0 = 0$, $\theta_1 = 1$, learning rate $\alpha = 0.1$.

Compute the gradients:

$$\frac{1}{m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)}\right) = \frac{-4}{3}$$

$$\frac{1}{m} \sum_{i=1}^{m} \left(h_\theta(x^{(i)}) - y^{(i)}\right) x^{(i)} = -3$$

Update:

$$\theta_0 := 0 - 0.1 \cdot \left(-\frac{4}{3}\right) = 0.1333$$

$$\theta_1 := 1 - 0.1 \cdot (-3) = 1.3$$

After the update, the line moves closer to the data, reducing $J(\theta)$.

### Visual intuition (loss curve)

As gradient descent runs, the cost typically decreases like this:

```text
J(theta)
|
| *
|   *
|     *
|        *
|           *
|---------------> iterations
```

### Global minimum vs local minimum

For **linear regression with MSE**, the cost surface is convex, so there is **one global minimum** and no local minima.

For more complex models (e.g., neural networks), the loss surface can be non-convex with many local minima and saddle points, which is why optimization can be harder.

## 145. Convergence Algorithm Part 02 (Practical Details)

### Learning rate selection

- Too large: cost can diverge or oscillate.
- Too small: convergence becomes very slow.
- Common starting values: 0.1, 0.01, 0.001.

### Feature scaling

Scaling features helps gradient descent converge faster by making the cost surface more symmetric.

- Standardization: $x' = (x - \mu) / \sigma$
- Min-max scaling: $x' = (x - \min) / (\max - \min)$

### Convergence criteria

- Stop when the decrease in $J(\theta)$ between iterations is below a threshold.
- Stop when the gradient norm is sufficiently small.
- Stop after a fixed number of iterations (useful for reproducibility).

### Batch vs stochastic vs mini-batch

- **Batch:** uses all samples each step (stable, slower per step).
- **Stochastic:** uses one sample (noisy updates, can escape shallow regions).
- **Mini-batch:** uses a small subset (balanced speed and stability).

### Common debugging checks

- Plot $J(\theta)$ over iterations; it should decrease smoothly.
- If the cost increases, reduce $\alpha$ or scale features.
- If training is extremely slow, try a larger $\alpha$ or normalize features.

## 146. Multiple Linear Regression

Multiple linear regression uses **multiple input features** to predict a continuous target.

### Model equation

$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_d x_d$$

### Matrix form

$$\hat{\mathbf{y}} = \mathbf{X} \theta$$

Where:
- $\mathbf{X}$ is the design matrix of shape $(m, d+1)$ with a column of ones.
- $\theta$ is the parameter vector.

### Coefficient interpretation

- $\theta_j$ is the expected change in $\hat{y}$ for a 1-unit increase in $x_j$, **holding other features constant**.

### Practical considerations

- **Multicollinearity:** highly correlated features make coefficients unstable.
- **Scaling:** feature scales affect optimization and coefficient magnitude.
- **Categorical features:** use one-hot encoding (avoid dummy-variable trap by dropping one column).

## 147. Performance Metrics

Performance metrics quantify how close predictions are to true values.

### Common regression metrics

- **MSE:** mean squared error
- **MAE:** mean absolute error
- **RMSE:** root mean squared error
- **$R^2$:** proportion of variance explained
- **Adjusted $R^2$:** penalizes unnecessary features

### $R^2$ (coefficient of determination)

$$R^2 = 1 - \frac{\sum_{i=1}^{m} (y^{(i)} - \hat{y}^{(i)})^2}{\sum_{i=1}^{m} (y^{(i)} - \bar{y})^2}$$

- $R^2 = 1$ indicates perfect fit.
- $R^2 = 0$ means the model is no better than predicting the mean.
- Negative values mean the model performs worse than predicting the mean.

### Adjusted $R^2$

$$\bar{R}^2 = 1 - (1 - R^2) \cdot \frac{n - 1}{n - p - 1}$$

Where:
- $n$ is number of samples.
- $p$ is number of features.

Adjusted $R^2$ decreases when a new feature does not improve the model enough.

### Evaluation workflow

- Split data into train, validation, and test sets.
- Fit on train, tune on validation, report on test.
- Use cross-validation for more reliable estimates.

## 148. MSE, MAE, RMSE

These three metrics are used most often for regression error magnitude.

### MSE

$$\text{MSE} = \frac{1}{m} \sum_{i=1}^{m} \left(\hat{y}^{(i)} - y^{(i)}\right)^2$$

- Strongly penalizes large errors.
- Smooth and differentiable.

### MAE

$$\text{MAE} = \frac{1}{m} \sum_{i=1}^{m} \left|\hat{y}^{(i)} - y^{(i)}\right|$$

- More robust to outliers than MSE.
- Not differentiable at zero (less convenient for some optimizers).

### RMSE

$$\text{RMSE} = \sqrt{\text{MSE}}$$

- Same unit as the target variable.
- Penalizes large errors like MSE.

### When to use which

- **MAE:** when you want equal weight for all errors.
- **RMSE:** when large errors are especially bad.
- **MSE:** when you want a smooth objective for optimization.

## 149. Overfitting and Underfitting

### Underfitting

Underfitting happens when the model is too simple to capture the true pattern.

- High bias, low variance.
- Poor performance on training and test data.

### Overfitting

Overfitting happens when the model is too complex and captures noise.

- Low bias, high variance.
- Very low training error but high test error.

### Diagnostic signals

- **Learning curves:**
	- Underfitting: training and validation errors both high.
	- Overfitting: training error low, validation error high.
- **Residual plots:** structured patterns imply underfitting.

### How to reduce underfitting

- Add more informative features.
- Increase model complexity (polynomial features).
- Decrease regularization strength.

### How to reduce overfitting

- Use simpler features or lower polynomial degree.
- Add more data.
- Apply regularization (L1 or L2).
- Use cross-validation to tune complexity.

## 150. Linear Regression with OLS

**Ordinary Least Squares (OLS)** minimizes the sum of squared residuals.

### Objective

$$\min_{\theta} \sum_{i=1}^{m} \left(y^{(i)} - h_\theta(x^{(i)})\right)^2$$

### Normal equation (closed form)

$$\theta = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

### Notes and edge cases

- $\mathbf{X}^T \mathbf{X}$ must be invertible to use the exact inverse.
- If it is not invertible, use the pseudo-inverse:
	$$\theta = \mathbf{X}^+ \mathbf{y}$$
- OLS is efficient for small to medium feature sizes but can be expensive for very large $d$.

### OLS vs gradient descent

- OLS yields an exact solution in one step.
- Gradient descent is iterative and scales better to very large datasets.
- Gradient descent is preferred when you later extend to models without closed-form solutions.

### OLS derivation for simple linear regression (using $\beta_0$, $\beta_1$)

For simple linear regression, we often write the model as:

$$\hat{y} = \beta_0 + \beta_1 x$$

The OLS objective is to minimize the **sum of squared errors (SSE)**:

$$S(\beta_0, \beta_1) = \sum_{i=1}^{m} \left(y^{(i)} - (\beta_0 + \beta_1 x^{(i)})\right)^2$$

To find the minimum, set partial derivatives to zero.

**1) Partial derivative with respect to $\beta_0$: **

$$\frac{\partial S}{\partial \beta_0} = -2 \sum_{i=1}^{m} \left(y^{(i)} - \beta_0 - \beta_1 x^{(i)}\right) = 0$$

This gives the first normal equation:

$$\sum_{i=1}^{m} y^{(i)} = m \beta_0 + \beta_1 \sum_{i=1}^{m} x^{(i)}$$

**2) Partial derivative with respect to $\beta_1$: **

$$\frac{\partial S}{\partial \beta_1} = -2 \sum_{i=1}^{m} x^{(i)} \left(y^{(i)} - \beta_0 - \beta_1 x^{(i)}\right) = 0$$

This gives the second normal equation:

$$\sum_{i=1}^{m} x^{(i)} y^{(i)} = \beta_0 \sum_{i=1}^{m} x^{(i)} + \beta_1 \sum_{i=1}^{m} (x^{(i)})^2$$

Now define the sample means:

$$\bar{x} = \frac{1}{m} \sum_{i=1}^{m} x^{(i)}, \quad \bar{y} = \frac{1}{m} \sum_{i=1}^{m} y^{(i)}$$

Solving the normal equations yields the closed-form solution:

$$\beta_1 = \frac{\sum_{i=1}^{m} (x^{(i)} - \bar{x})(y^{(i)} - \bar{y})}{\sum_{i=1}^{m} (x^{(i)} - \bar{x})^2} = \frac{S_{xy}}{S_{xx}}$$

$$\beta_0 = \bar{y} - \beta_1 \bar{x}$$

Where:
- $S_{xy} = \sum (x^{(i)} - \bar{x})(y^{(i)} - \bar{y})$ is the covariance numerator.
- $S_{xx} = \sum (x^{(i)} - \bar{x})^2$ is the variance numerator of $x$.

Interpretation:
- $\beta_1$ measures how $y$ changes with $x$.
- $\beta_0$ shifts the line to match the average of $y$ when $x$ is at its mean.

## 151. Simple Linear Regression Practical

Practical content is intentionally omitted per request.

## 152. Multiple Linear Regression Practical

Practical content is intentionally omitted per request.

## 153. Polynomial Regression Intuition

Polynomial regression captures non-linear relationships by expanding features.

### Feature expansion

For one feature:

$$h_\theta(x) = \theta_0 + \theta_1 x + \theta_2 x^2 + \theta_3 x^3$$

Even though the curve is non-linear in $x$, the model is **linear in parameters** $\theta$.

### Why it helps

- Straight lines cannot fit curved patterns.
- Polynomial features allow the model to bend and capture curvature.

### Risks and best practices

- High-degree polynomials can overfit.
- Large values of $x$ can make $x^k$ explode, so scaling helps.
- Start with small degrees (2 or 3).
- Use cross-validation to select the degree.
- Combine with regularization if needed.
