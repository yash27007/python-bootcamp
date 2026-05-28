# Gradient Boosting: A Complete Guide

## Big Picture (Simple Summary)

Imagine you are trying to estimate the value of a house. Your first guess is $300,000. It's actually $350,000, so you're off by $50,000. Instead of starting over, you train a second model specifically to predict *the gap* ($50,000). Now your combined prediction is $350,000. Still a bit off? Train a third model to close the remaining gap. Repeat.

This is **Gradient Boosting**: an ensemble method that builds trees **sequentially**, where each new tree predicts the **residual errors** left by the previous trees, gradually improving the ensemble's prediction.

---

## 1) The Core Insight: Fitting Residuals

### 1.1 Simple Boosting (Residual Fitting)

Consider a regression problem with targets $y_i$ and predictions $F(x_i)$.

**Round 1:** Train a tree $h_1(x)$ → predictions $\hat{y}_1$  
**Residuals:** $r_i = y_i - \hat{y}_1(x_i)$  
**Round 2:** Train a tree $h_2(x)$ directly on the residuals $r_i$  
**Updated prediction:** $\hat{y} = \hat{y}_1 + h_2(x)$  
**New residuals:** $r_i' = y_i - \hat{y}_1(x_i) - h_2(x_i)$  
...and so on.

After T rounds:

$$F(x) = h_1(x) + h_2(x) + h_3(x) + \dots + h_T(x)$$

### 1.2 Why "Gradient"?

The connection to gradients comes from a more general framework. The residual $r_i = y_i - \hat{y}$ is **the negative gradient of the Mean Squared Error loss** with respect to the prediction:

$$r_i = -\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)} = -\frac{\partial \frac{1}{2}(y_i - F(x_i))^2}{\partial F(x_i)} = y_i - F(x_i)$$

By fitting the residuals, we're doing **gradient descent in function space** — each new tree steps in the direction that most reduces the loss.

This generalization is powerful: by changing the loss function, we get different kinds of residuals (**pseudo-residuals**), and the algorithm remains the same!

---

## 2) Gradient Boosting for General Loss Functions

### 2.1 The General Algorithm

Let $L(y, F(x))$ be any differentiable loss function.

**Pseudo-residuals** for a general loss at round t:

$$r_i^{(t)} = -\left[\frac{\partial L(y_i, F(x_i))}{\partial F(x_i)}\right]_{F = F^{(t-1)}}$$

This is the **negative gradient** of the loss evaluated at the current ensemble's prediction.

**The Full Gradient Boosting Algorithm:**

```
1. Initialize: F₀(x) = argmin_γ Σᵢ L(yᵢ, γ)
   (For MSE, this is just the mean of y: F₀(x) = mean(y))

2. For t = 1, 2, ..., T:
   a. Compute pseudo-residuals:
      rᵢ(t) = -∂L(yᵢ, F(xᵢ))/∂F(xᵢ)  evaluated at F = F_{t-1}

   b. Fit a regression tree h_t to the pseudo-residuals {(xᵢ, rᵢ(t))}

   c. Find optimal step size (leaf values):
      γ_t = argmin_γ Σᵢ L(yᵢ, F_{t-1}(xᵢ) + γ h_t(xᵢ))

   d. Update the ensemble:
      F_t(x) = F_{t-1}(x) + η · γ_t · h_t(x)

3. Return F_T(x)
```

Where $\eta$ (eta) is the **learning rate** (shrinkage parameter), typically 0.01 – 0.3.

### 2.2 Pseudo-Residuals for Different Loss Functions

| Loss Function | Use Case | Pseudo-residual $r_i$ |
|--------------|----------|----------------------|
| $\frac{1}{2}(y_i - F)^2$ (MSE) | Regression | $y_i - F(x_i)$ |
| $|y_i - F|$ (MAE) | Robust regression | $\text{sign}(y_i - F(x_i))$ |
| Huber Loss | Regression (outlier-robust) | MSE for small errors, MAE for large errors |
| Log Loss / Binary Cross-entropy | Classification | $y_i - \sigma(F(x_i))$ |
| Deviance | Multi-class | $(y_k - P(y=k|x))$ per class |

**Key insight:** The flexibility to plug in any loss function is what makes Gradient Boosting powerful. It generalizes AdaBoost (which uses exponential loss) and MSE boosting.

---

## 3) Worked Example: Regression with MSE Loss

**Training data:**

| x | y (actual) |
|---|-----------|
| 1 | 2.5 |
| 2 | 3.8 |
| 3 | 6.1 |
| 4 | 8.0 |
| 5 | 9.5 |

**Initial prediction (mean):** $F_0 = (2.5 + 3.8 + 6.1 + 8.0 + 9.5)/5 = 5.98$

**Round 1 — Compute Residuals:**

| x | y | F₀ | r₁ = y - F₀ |
|---|---|-----|------------|
| 1 | 2.5 | 5.98 | -3.48 |
| 2 | 3.8 | 5.98 | -2.18 |
| 3 | 6.1 | 5.98 | +0.12 |
| 4 | 8.0 | 5.98 | +2.02 |
| 5 | 9.5 | 5.98 | +3.52 |

**Fit a stump to residuals:** Best split at x=2.5

- Left (x ≤ 2): mean residual = (-3.48 - 2.18)/2 = **-2.83**
- Right (x > 2): mean residual = (0.12 + 2.02 + 3.52)/3 = **1.89**

**Update with learning rate η = 0.1:**

$$F_1(x) = F_0(x) + 0.1 \times h_1(x)$$

For x=1: F₁(1) = 5.98 + 0.1 × (-2.83) = **5.70** (actual: 2.5 — still far, but slightly closer!)

**Repeat for T = 200 rounds** → each step the prediction gradually improves.

---

## 4) Learning Rate and the Bias-Variance Tradeoff

### 4.1 Shrinkage

The learning rate $\eta$ controls how much each tree contributes:

$$F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x), \quad \eta \in (0, 1]$$

**Effect of learning rate:**
- $\eta = 1.0$: Full step — fast convergence, risk of overfitting
- $\eta = 0.1$: Each tree contributes 10% — slower learning, better generalization
- $\eta = 0.01$: Very small steps — very slow, needs many trees, often best final accuracy

**The tradeoff:**

| Learning Rate | n_estimators needed | Training speed | Generalization |
|---------------|--------------------|--------------:|---------------|
| 0.3 | ~100 | Fast | OK |
| 0.1 | ~300 | Medium | Good |
| 0.05 | ~600 | Slow | Better |
| 0.01 | ~3000 | Very slow | Often best |

**Rule of thumb:** Lower learning rate + more trees almost always outperforms higher learning rate + fewer trees, given enough computation budget.

### 4.2 Subsampling (Stochastic Gradient Boosting)

Friedman (2002) introduced **stochastic gradient boosting**: at each round, instead of using all N training examples, randomly sample a fraction $f$ (e.g., 50%) without replacement.

**Benefits:**
1. Faster training (fewer examples per tree)
2. Additional regularization (reduces overfitting)
3. Can be used to estimate generalization error

$$F_t(x) = F_{t-1}(x) + \eta \cdot h_t(x) \quad \text{where } h_t \text{ trained on random subset}$$

**Typical setting:** `subsample=0.5` to `0.8`

---

## 5) Regularization in Gradient Boosting

Gradient Boosting without regularization will overfit badly. Key regularization techniques:

### 5.1 Learning Rate (η) — Shrinkage
Already covered above. Most important regularizer.

### 5.2 Tree Complexity Constraints
- **max_depth:** Limit tree depth (typically 3–8 for GBM vs. unlimited for RF)
- **min_samples_leaf:** Minimum samples required in a leaf
- **max_leaf_nodes:** Maximum number of leaf nodes

### 5.3 Subsampling (Row Sampling)
`subsample < 1.0` — sample a fraction of rows for each tree.

### 5.4 Feature Subsampling (Column Sampling)
`max_features < 1.0` — use a random subset of features for each tree (similar to Random Forest).

### 5.5 Early Stopping
Monitor validation loss and stop training when it starts to increase:

```python
from sklearn.ensemble import GradientBoostingClassifier

# Use staged_predict to find optimal number of trees
model = GradientBoostingClassifier(n_estimators=1000, learning_rate=0.05)
model.fit(X_train, y_train)

# Find the tree count with best validation score
from sklearn.metrics import log_loss
val_losses = [log_loss(y_val, pred) 
              for pred in model.staged_predict_proba(X_val)]
optimal_n = val_losses.index(min(val_losses)) + 1
print(f"Optimal trees: {optimal_n}")
```

---

## 6) Gradient Boosting for Classification

### 6.1 Binary Classification

For binary classification, the ensemble outputs a log-odds score:

$$F(x) = \log\left(\frac{P(y=1|x)}{P(y=0|x)}\right)$$

Convert to probability with sigmoid:

$$P(y=1|x) = \sigma(F(x)) = \frac{1}{1 + e^{-F(x)}}$$

**Pseudo-residuals for binary cross-entropy loss:**

$$r_i = y_i - P(y_i = 1 | x_i) = y_i - \sigma(F(x_i))$$

These are the difference between actual class (0 or 1) and the model's predicted probability.

**Initial prediction:**

$$F_0 = \log\left(\frac{p}{1-p}\right), \quad p = \frac{\text{count of class 1}}{N}$$

### 6.2 Multiclass Classification

For K classes, train K separate trees per round — one tree per class. Each tree predicts the residual for its class. The final predictions use softmax:

$$P(y=k|x) = \frac{e^{F_k(x)}}{\sum_{j=1}^K e^{F_j(x)}}$$

---

## 7) Comparing GBM, AdaBoost, and Random Forest

| Feature | Random Forest | AdaBoost | Gradient Boosting |
|---------|--------------|----------|-------------------|
| Trees built | Parallel | Sequential | Sequential |
| Depth per tree | Deep | Shallow (stumps) | Shallow (3–8) |
| Learning method | Bagging | Reweighted samples | Gradient descent |
| Loss function | N/A | Exponential | Any differentiable |
| Outlier sensitivity | Low | Very high | Medium (depends on loss) |
| Interpretability | Low | Medium | Low |
| Speed | Fast (parallel) | Slow (sequential) | Slow (sequential) |
| Tuning effort | Low | Low | High |
| Typical accuracy | Good | Good | Better |

---

## 8) Hyperparameters

| Hyperparameter | What it Controls | Typical Range |
|----------------|-----------------|---------------|
| `n_estimators` | Number of trees (rounds) | 100–3000 |
| `learning_rate` | Shrinkage (η) | 0.01–0.3 |
| `max_depth` | Tree depth | 3–8 |
| `subsample` | Row sampling fraction | 0.5–0.9 |
| `max_features` | Feature sampling fraction | 0.5–1.0 |
| `min_samples_leaf` | Min samples per leaf | 1–50 |
| `min_impurity_decrease` | Minimum split gain | 0–0.01 |

**Tuning strategy:**

1. Fix `learning_rate=0.1`, tune `n_estimators` with early stopping
2. Tune `max_depth` (most impactful after n_estimators)
3. Add row and column subsampling (`subsample`, `max_features`)
4. Once best tree structure found, reduce `learning_rate` by 2x and increase `n_estimators` by 2x

---

## 9) Practical Implementation

```python
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, roc_auc_score
import numpy as np

# Classification
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

gb = GradientBoostingClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    max_features=0.8,
    min_samples_leaf=10,
    random_state=42,
    validation_fraction=0.2,   # Use 20% for early stopping
    n_iter_no_change=20,       # Stop if no improvement for 20 rounds
    tol=1e-4
)
gb.fit(X_train, y_train)
print(f"Optimal trees: {gb.n_estimators_}")

y_pred = gb.predict(X_val)
y_prob = gb.predict_proba(X_val)[:, 1]
print(classification_report(y_val, y_pred))
print(f"AUC-ROC: {roc_auc_score(y_val, y_prob):.4f}")

# Feature importance
import pandas as pd
import matplotlib.pyplot as plt

importances = pd.Series(gb.feature_importances_, index=X.columns)
importances.nlargest(15).plot(kind='barh')
plt.title("Feature Importances")
plt.tight_layout()
plt.show()
```

---

## 10) Pros and Cons

### Pros
1. **State-of-the-art on tabular data** — often best out-of-the-box performance
2. **Flexible loss functions** — MSE, MAE, Huber, Log Loss, etc.
3. **Robust to different feature scales** — tree-based, no need to standardize
4. **Handles mixed data types** — numerical and categorical
5. **Good feature importance** — built-in and permutation

### Cons
1. **Slow to train** — sequential, cannot be parallelized at the tree level
2. **Many hyperparameters** — requires careful tuning
3. **Memory intensive** — stores all trees
4. **Prone to overfitting** — must use regularization (especially learning rate + early stopping)
5. **Not interpretable** — hundreds of trees
6. **Sensitive to outliers** (with MSE loss) — use Huber or MAE loss for robustness

---

## 11) Gradient Boosting vs XGBoost vs LightGBM vs CatBoost

| Library | Key Improvement over vanilla GBM |
|---------|----------------------------------|
| **XGBoost** | Regularized objective (L1/L2 on leaves), exact split finding, parallelism |
| **LightGBM** | Leaf-wise tree growth (faster), GOSS, EFB — much faster on large data |
| **CatBoost** | Native categorical feature handling, ordered boosting, symmetric trees |

In practice, use **XGBoost** or **LightGBM** instead of sklearn's `GradientBoostingClassifier` for large datasets — they are much faster. See the XGBoost notes for details.

---

## Summary: The 5 Key Takeaways

1. **Gradient Boosting builds trees sequentially** where each new tree fits the **negative gradient (pseudo-residuals)** of the loss function — for MSE, this is just the ordinary residual $y_i - \hat{y}_i$.

2. **Any differentiable loss function** can be used: MSE for regression, log loss for classification, Huber for outlier-robust regression.

3. **Learning rate (shrinkage)** is the most important hyperparameter: smaller learning rate + more trees = better generalization, at the cost of training speed.

4. **Regularization is essential**: use `subsample`, `max_depth`, `max_features`, `min_samples_leaf`, and early stopping to prevent overfitting.

5. **Gradient Boosting often achieves the best accuracy** on structured/tabular data — but use optimized implementations like XGBoost or LightGBM in practice for speed.
