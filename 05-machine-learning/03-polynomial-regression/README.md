# 03 – Polynomial Regression

| Topic | Status |
|-------|--------|
| Polynomial Regression Intuition | ✅ Complete |
| Pipeline in Polynomial Regression | ✅ Complete |

## Polynomial Regression Intuition

Polynomial regression is a **linear regression model applied to polynomial features**. It allows a linear model to fit **curved** relationships by expanding the input feature space.

### Why we need it

Simple linear regression fits a straight line. If the true relationship between input and target is curved, a straight line underfits.

Polynomial regression solves this by creating new features like $x^2$, $x^3$, etc., so the model can bend.

### Model form

For one feature $x$:

$$\hat{y} = \beta_0 + \beta_1 x + \beta_2 x^2 + \beta_3 x^3 + \dots + \beta_d x^d$$

Key idea:
- The model is **non-linear in $x$**.
- The model is **linear in parameters** $(\beta_0, \beta_1, \dots)$, so we can still use linear regression techniques.

### Geometric intuition

- Degree 1: straight line
- Degree 2: parabola (one bend)
- Degree 3: S-shaped curve (two bends)
- Higher degrees: more flexible curves

### When polynomial regression helps

- The scatter plot shows a curved trend.
- Residuals from linear regression show a pattern (not random).
- You want a simple, interpretable non-linear model.

### Risks

- High-degree polynomials can overfit.
- Extrapolation becomes unreliable outside the training range.
- Large feature values can cause huge polynomial terms.

### Best practices

- Start with small degrees (2 or 3).
- Use cross-validation to choose the degree.
- Combine with regularization if needed (Ridge or Lasso).
- Always scale features before polynomial expansion.

## Pipeline in Polynomial Regression

Polynomial regression is not a separate algorithm in sklearn. It is a **pipeline**: feature expansion + linear regression.

### Typical pipeline steps

1. **Split the data** into train and test sets.
2. **Scale features** (important because powers of $x$ grow quickly).
3. **Generate polynomial features** (e.g., $x$, $x^2$, $x^3$).
4. **Fit a linear regression model** on the expanded features.
5. **Evaluate** using metrics like MSE, RMSE, and $R^2$.

### Why scaling matters

- If $x$ is large, $x^2$ and $x^3$ become extremely large.
- This can cause unstable coefficients and slow optimization.
- Standardization keeps all polynomial terms on comparable scales.

### Choosing the degree

- Low degree: underfitting (too simple).
- High degree: overfitting (too flexible).
- Use validation or cross-validation to pick the best degree.

### Example pipeline (conceptual)

1. $X$ (original feature)
2. Scale $X$ to $X_{scaled}$
3. Expand: $[X_{scaled}, X_{scaled}^2, X_{scaled}^3]$
4. Fit linear regression on expanded features

### When to use polynomial regression

- You want a simple way to model non-linearity.
- Your dataset is small to medium.
- You need interpretability of coefficients.

### When not to use it

- High-dimensional data (polynomial explosion).
- Highly complex patterns (better handled by trees or neural nets).
- You need strong extrapolation beyond the observed range.
