# Logistic Regression: A Complete Guide

## Big Picture (Simple Summary)

Imagine you are a doctor trying to predict if a patient has a disease (Class 1) or not (Class 0). You cannot just say "yes" or "no" with 100% certainty. Instead, you want a probability: "There is an 80% chance this patient has the disease."

**Logistic regression does exactly this.** It takes your data (like age, blood pressure, etc.), calculates a probability, and then uses a cutoff (usually 50%) to make a final decision.

---

## 1) What is Logistic Regression? (The "What")

It is a **supervised learning** algorithm for **classification** (not regression, despite the name).

- **Input:** Your data features (e.g., hours studied, test score).
- **Output:** A probability between 0 and 1.
- **Decision Rule:** If probability >= 0.5 → predict Class 1 (e.g., "Pass"). Else → predict Class 0 (e.g., "Fail").

**The Formula:**

$$h_\theta(x) = \sigma(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

This is called the **sigmoid function**. It squashes any number into a probability between 0 and 1.

**Example:**  
Suppose you build a model to predict if an email is spam (Class 1) or not (Class 0).  
If the model outputs `0.85`, that means "85% chance this is spam". With a 0.5 threshold, you classify it as spam.

---

## 2) Why NOT Linear Regression for Classification? (The "Why Not")

You might ask: *Why can't I just use a straight line to predict probabilities?*

Here's why linear regression fails:

| Problem | Explanation | Example |
|---------|-------------|---------|
| **Unbounded predictions** | Linear regression can predict values <0 or >1. Probabilities must be between 0 and 1. | Predicting a probability of -0.3 or 1.5 makes no sense. |
| **Wrong loss function** | Squared error (MSE) doesn't work well for probabilities. It doesn't heavily punish confident wrong answers. | A model that says "99% chance of rain" on a sunny day gets only a small squared error. |
| **Bad calibration** | Linear regression doesn't model log-odds, so the probabilities are unreliable. | You might get "probability 0.6" when the real chance is only 0.1. |

**Logistic regression fixes all this** by using the sigmoid function and a proper loss called **log loss** (explained in detail below).

---

## 3) Math Intuition (The "How")

### 3.1 Odds and Log-Odds (Logit)

Before understanding logistic regression, you need to understand **odds** and **log-odds**.

**Odds** = Probability of event happening / Probability of event not happening

$$\text{Odds} = \frac{p}{1-p}$$

**Example:** If probability of winning is 0.8 (80%), then:
- Odds = 0.8 / 0.2 = 4
- This means: "4 to 1 in favor" of winning

**Log-odds (Logit)** = The natural logarithm of odds

$$\text{Logit}(p) = \log\left(\frac{p}{1-p}\right)$$

**Why log-odds?** Because log-odds can be ANY number from negative infinity to positive infinity (unlike probability which is stuck between 0 and 1).

Logistic regression assumes:

$$\log\left(\frac{p}{1-p}\right) = \theta_0 + \theta_1 x_1 + \dots + \theta_n x_n$$

So it's **linear in log-odds**, not in probability. This is the key insight!

### 3.2 The Sigmoid Function (The Magic S-Curve)

The sigmoid function takes any real number and maps it to (0, 1):

$$\sigma(z) = \frac{1}{1 + e^{-z}}$$

**How it behaves:**
- When z is very large positive (e.g., z = 10) → e^(-10) ≈ 0.000045 → σ(z) ≈ 0.99995 (close to 1)
- When z = 0 → e^(0) = 1 → σ(z) = 0.5
- When z is very large negative (e.g., z = -10) → e^(10) ≈ 22026 → σ(z) ≈ 0.000045 (close to 0)

**Visualization (mental picture):** The sigmoid looks like an "S" shape that smoothly transitions from 0 to 1 as z increases.

### 3.3 Decision Boundary

The default boundary is at probability = 0.5 → which means z = 0 → θ^T x = 0.

This is a **straight line (hyperplane)** in feature space.

**Example (2 features):**  
Let's say θ₀ = -1, θ₁ = 2, θ₂ = -1

The decision boundary is: -1 + 2x₁ - x₂ = 0 → x₂ = 2x₁ - 1

- Points above this line (x₂ > 2x₁ - 1) → Class 1
- Points below this line (x₂ < 2x₁ - 1) → Class 0

### 3.4 Coefficient Interpretation (Very Important!)

Each coefficient θⱼ tells you how a feature affects the prediction.

**RULE 1:** A one-unit increase in feature xⱼ increases the **log-odds** by θⱼ (holding other features fixed).

**RULE 2:** A one-unit increase in feature xⱼ multiplies the **odds** by e^(θⱼ).

**Example:** Suppose you build a model to predict if a student passes (1) or fails (0) based on hours studied.

If θ_hours = 0.5, then:
- Each extra hour of study increases log-odds by 0.5
- Each extra hour multiplies the odds by e^(0.5) ≈ 1.65
- This means: The odds of passing increase by 65% for each additional hour of study

**Important:** If θⱼ is negative, it decreases the odds. If θⱼ is zero, the feature has no effect.

---

## 4) Log Loss Explained in DETAIL (Most Important Section!)

This is the heart of logistic regression. Understanding log loss is CRITICAL.

### 4.1 Why Can't We Use Squared Error?

Let me show you why squared error fails.

**Squared error for one example:** (prediction - actual)²

Suppose actual class = 1 (pass).

| Prediction (probability) | Squared Error |
|-------------------------|---------------|
| 0.99 (very confident, correct) | (0.99 - 1)² = 0.0001 |
| 0.90 | (0.90 - 1)² = 0.01 |
| 0.75 | (0.75 - 1)² = 0.0625 |
| 0.51 (barely correct) | (0.51 - 1)² = 0.2401 |
| 0.49 (barely wrong) | (0.49 - 1)² = 0.2601 |
| 0.25 | (0.25 - 1)² = 0.5625 |
| 0.01 (very confident, wrong!) | (0.01 - 1)² = 0.9801 |

**The problem:** The penalty for a "confident wrong" prediction (0.01) is 0.98. But the penalty for a "barely wrong" prediction (0.49) is 0.26. The ratio is only about 4x. That's too small! We want to DESTROY confident wrong predictions.

### 4.2 What is Log Loss? (The Solution)

Log loss uses the **logarithm** to create an enormous penalty for confident wrong predictions.

**For a single example:**

$$\text{Cost}(h_\theta(x), y) = 
\begin{cases}
-\log(h_\theta(x)) & \text{if } y = 1 \\
-\log(1 - h_\theta(x)) & \text{if } y = 0
\end{cases}$$

Let's understand this with a table.

**Case 1: Actual class = 1 (y = 1)**

| Prediction (probability of class 1) | Cost = -log(prediction) | Interpretation |
|-------------------------------------|------------------------|----------------|
| 0.99 (very confident, correct) | -log(0.99) ≈ 0.01 | Very small penalty ✅ |
| 0.90 | -log(0.90) ≈ 0.105 | Small penalty |
| 0.75 | -log(0.75) ≈ 0.288 | Moderate penalty |
| 0.51 (barely correct) | -log(0.51) ≈ 0.673 | Noticeable penalty |
| 0.49 (barely wrong) | -log(0.49) ≈ 0.713 | Still noticeable |
| 0.25 | -log(0.25) = 1.386 | Large penalty |
| 0.01 (very confident, wrong!) | -log(0.01) = 4.605 | HUGE penalty!!! 💀 |

**See the difference?** Going from 0.25 to 0.01 (confidently wrong) increases the penalty from 1.386 to 4.605 — a 3.3x increase. The squared error only gave a 1.7x increase.

**Case 2: Actual class = 0 (y = 0)** → Cost = -log(1 - prediction)

| Prediction (probability of class 1) | 1 - prediction | Cost = -log(1 - p) |
|-------------------------------------|----------------|-------------------|
| 0.01 (confident correct) | 0.99 | 0.01 ✅ |
| 0.10 | 0.90 | 0.105 |
| 0.25 | 0.75 | 0.288 |
| 0.49 (barely correct) | 0.51 | 0.673 |
| 0.51 (barely wrong) | 0.49 | 0.713 |
| 0.75 | 0.25 | 1.386 |
| 0.99 (confident wrong) | 0.01 | 4.605 💀 |

### 4.3 Why "Log" Makes Sense (The Intuition)

The logarithm function log(x) has a special property: as x approaches 0, -log(x) approaches +infinity.

**Visualize it:**
- log(1) = 0 (perfect prediction → zero penalty)
- log(0.5) = -0.693 → -log(0.5) = 0.693
- log(0.1) = -2.302 → -log(0.1) = 2.302
- log(0.01) = -4.605 → -log(0.01) = 4.605
- log(0.001) = -6.908 → -log(0.001) = 6.908

As prediction approaches 0 (completely wrong for class 1), the penalty goes to INFINITY. The model will do ANYTHING to avoid being confident and wrong.

### 4.4 The Combined Formula (Binary Cross-Entropy)

We can write both cases in ONE formula:

$$\text{Cost}(h_\theta(x), y) = -y \log(h_\theta(x)) - (1-y) \log(1-h_\theta(x))$$

**Check it:**
- If y = 1: second term vanishes (1-y = 0) → Cost = -log(h(x)) ✅
- If y = 0: first term vanishes (y = 0) → Cost = -log(1 - h(x)) ✅

### 4.5 The Complete Loss Function (Over All Training Data)

For m training examples:

$$J(\theta) = -\frac{1}{m} \sum_{i=1}^m \left[ y^{(i)} \log(h_\theta(x^{(i)})) + (1-y^{(i)}) \log(1 - h_\theta(x^{(i)})) \right]$$

This is also called the **negative log-likelihood** because maximizing the likelihood is equivalent to minimizing this loss.

### 4.6 Real Example: Comparing Squared Error vs Log Loss

Let's say you have 3 predictions on 3 different emails (spam = 1):

| Email | Actual | Model A Prediction | Model B Prediction |
|-------|--------|-------------------|-------------------|
| 1 | 1 | 0.99 | 0.60 |
| 2 | 1 | 0.98 | 0.61 |
| 3 | 0 | 0.02 | 0.40 |

**Which model is better?** Model A is very confident and correct. Model B is hesitant.

**Squared Error:**
- Model A: (0.99-1)² + (0.98-1)² + (0.02-0)² = 0.0001 + 0.0004 + 0.0004 = 0.0009
- Model B: (0.60-1)² + (0.61-1)² + (0.40-0)² = 0.16 + 0.1521 + 0.16 = 0.4721

Squared Error says Model A is 524x better. But is that fair?

**Log Loss:**
- Model A: -log(0.99) - log(0.98) - log(1-0.02) = 0.010 + 0.020 + 0.020 = 0.050
- Model B: -log(0.60) - log(0.61) - log(1-0.40) = 0.511 + 0.494 + 0.511 = 1.516

Log Loss says Model A is 30x better. Still strongly prefers Model A, but less extreme. Log loss is **proper** for probabilities — it gives the right incentive to output well-calibrated probabilities, not just extreme values.

---

## 5) Convexity and Convergence

### 5.1 What is Convexity?

A function is **convex** if the line segment between any two points on the graph lies above the graph.

**Why does this matter?** 
- Convex functions have ONE global minimum (no bad local minima)
- Any optimization method will eventually find the best answer
- Non-convex functions (like neural networks) can get stuck in suboptimal local minima

**Good news:** Log loss for logistic regression is CONVEX in θ.  
**Bad news:** Squared error for logistic regression is NON-CONVEX (that's another reason we don't use it).

### 5.2 How We Find the Best θ (Optimization)

There is **no closed-form solution** (unlike linear regression where you can do θ = (X^T X)^(-1) X^T y). We must use iterative methods.

**Method 1: Batch Gradient Descent**

Update rule:
$$\theta := \theta - \alpha \cdot \frac{1}{m} X^T (h_\theta(X) - y)$$

Where α is the learning rate (e.g., 0.01).

**Example:** If current prediction is 0.8 but actual is 1, the error (h-y) is negative, so θ increases slightly to make future predictions higher.

**Method 2: Stochastic Gradient Descent (SGD)**
- Update using ONE random example at a time
- Much faster for large datasets (millions of examples)
- Noisier updates but can escape shallow plateaus

**Method 3: Newton-Raphson / IRLS**
- Uses both gradient (first derivative) and Hessian (second derivative)
- Converges in fewer iterations
- Each iteration is more expensive (O(n³) for Hessian inversion)
- Great for small to medium datasets

**Method 4: L-BFGS (Quasi-Newton)**
- Approximates the Hessian without storing the full matrix
- Default in scikit-learn for logistic regression
- Good balance of speed and memory

### 5.3 Convergence Problems and Solutions

**Problem 1: Complete Separation**

When data is perfectly separable (e.g., all positive examples have x > 0, all negative have x < 0), the log loss can keep decreasing forever, causing coefficients to blow up to infinity.

**Solution:** Add regularization (see Section 6).

**Problem 2: Poor Feature Scaling**

If one feature is in range [0, 1] and another in [0, 1000000], the optimization will be slow and unstable.

**Solution:** Standardize features to mean=0, variance=1.

**Problem 3: Multicollinearity**

When two features are highly correlated (e.g., "height in feet" and "height in inches"), the Hessian becomes ill-conditioned.

**Solution:** Remove redundant features or use L2 regularization.

---

## 6) Regularization (Prevent Overfitting)

Regularization adds a penalty to large coefficients, forcing the model to be simpler.

### 6.1 L2 Regularization (Ridge)

Adds the sum of squared coefficients:

$$J(\theta) = \text{Log Loss} + \frac{\lambda}{2m} \sum_{j=1}^n \theta_j^2$$

**Effect:** Shrinks all coefficients toward zero (but never exactly zero).  
**Best for:** When you have many small to medium effects.

### 6.2 L1 Regularization (Lasso)

Adds the sum of absolute coefficients:

$$J(\theta) = \text{Log Loss} + \frac{\lambda}{m} \sum_{j=1}^n |\theta_j|$$

**Effect:** Can drive some coefficients to exactly zero → feature selection.  
**Best for:** When you suspect only a few features are important.

### 6.3 Elastic Net

A mix of L1 and L2.

### 6.4 Understanding λ and C

- **λ (lambda):** Regularization strength. Higher λ = more regularization.
- **In scikit-learn:** They use `C` where `C = 1/λ`. Smaller C = stronger regularization.

**Example:** 
- C = 0.01 → very strong regularization → high bias, low variance
- C = 1 → default
- C = 100 → very weak regularization → low bias, high variance (overfitting risk)

### 6.5 Choosing λ with Cross-Validation

```python
# Pseudocode
for lambda in [0.001, 0.01, 0.1, 1, 10, 100]:
    train model with regularization = lambda
    evaluate on validation set
pick lambda with best validation score
```

## 7  Performance Metrics (How to Evaluate)

### 7.1 Confusion Matrix (The Foundation)

|                      | Predicted YES (1) | Predicted NO (0) |
|----------------------|-------------------|------------------|
| **Actual YES (1)**   | True Positive (TP) | False Negative (FN) |
| **Actual NO (0)**    | False Positive (FP) | True Negative (TN) |

**Example:** Cancer test on 100 patients

|                      | Predicted Cancer | Predicted No Cancer |
|----------------------|------------------|---------------------|
| **Actually Cancer**  | 45 (TP)          | 5 (FN)              |
| **Actually Healthy** | 10 (FP)          | 40 (TN)             |

### 7.2 Key Metrics Explained

**Accuracy (not always good!)**
$$\text{Accuracy} = \frac{TP + TN}{TP + TN + FP + FN} = \frac{45 + 40}{100} = 85\%$$

**BUT** if only 5% have cancer, a model that always says "no cancer" gets 95% accuracy but is useless! That's why we need other metrics.

**Precision (When you say yes, how often are you right?)**
$$\text{Precision} = \frac{TP}{TP + FP} = \frac{45}{45 + 10} = 81.8\%$$

Use when false positives are costly. Example: Spam detection — flagging a good email as spam (FP) is very bad.

**Recall / Sensitivity (Of all actual yes, how many did you catch?)**
$$\text{Recall} = \frac{TP}{TP + FN} = \frac{45}{45 + 5} = 90\%$$

Use when false negatives are costly. Example: Cancer detection — missing a cancer patient (FN) is terrible.

**Specificity (Of all actual no, how many did you correctly say no to?)**
$$\text{Specificity} = \frac{TN}{TN + FP} = \frac{40}{40 + 10} = 80\%$$

**F1 Score (Harmonic mean of precision and recall)**
$$F1 = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}} = 2 \times \frac{0.818 \times 0.90}{0.818 + 0.90} \approx 0.857$$

F1 is best when you care about both precision and recall equally.

### 7.3 ROC Curve and AUC

**ROC Curve:** Plots True Positive Rate (Recall) vs False Positive Rate (1 - Specificity) as you vary the decision threshold.

- FPR = FP / (FP + TN) — how many negatives you incorrectly call positive

**AUC (Area Under ROC Curve):**
- 0.5 = random guessing (model is useless)
- 0.7-0.8 = acceptable
- 0.8-0.9 = excellent
- 0.9-1.0 = outstanding

**Example:** 
- Model A has AUC 0.95 → excellent
- Model B has AUC 0.55 → barely better than random

**When to use PR AUC instead:** When data is highly imbalanced (e.g., 1% positives). ROC can be misleadingly optimistic in that case.

### 7.4 Log Loss and Brier Score

**Log Loss** (we already covered): Measures probability quality. Lower is better.

**Brier Score:** Mean squared error of probabilities.
$$\text{Brier} = \frac{1}{m} \sum_{i=1}^m (h_\theta(x^{(i)}) - y^{(i)})^2$$

- Range: 0 to 1
- 0 = perfect predictions
- 0.25 = random (for balanced data)
- Lower is better

**Comparison:** Brier score is easier to interpret (it's just MSE), but log loss is more "proper" for probabilities.

---

## 8) Thresholding and Calibration

### 8.1 Changing the Decision Threshold

Default threshold is 0.5, but you can change it based on costs.

**Example: Loan Default Prediction**

| Scenario | Cost of False Positive | Cost of False Negative |
|----------|----------------------|----------------------|
| Approve loan to someone who defaults | $10,000 (big loss) | N/A |
| Deny loan to someone who would repay | $500 (lost profit) | N/A |

Since FP is 20x more expensive than FN, you should **lower the threshold** to approve fewer loans.

**If FP is more costly → increase threshold** (be more confident before saying yes)
**If FN is more costly → decrease threshold** (catch more positives, accept some FPs)

### 8.2 How to Choose the Threshold

1. Define the cost of each type of error
2. Calculate expected cost for each possible threshold
3. Pick threshold that minimizes expected cost

**Example calculation:**
- Cost(FP) = $10,000
- Cost(FN) = $500

For threshold = 0.5: FP=10, FN=5 → Cost = 10×10000 + 5×500 = $102,500
For threshold = 0.3: FP=20, FN=2 → Cost = 20×10000 + 2×500 = $201,000 (worse)
For threshold = 0.7: FP=2, FN=15 → Cost = 2×10000 + 15×500 = $27,500 (better!)

### 8.3 Calibration (Are Probabilities Trustworthy?)

A model is **calibrated** if: when it predicts 70% probability, the event actually happens ~70% of the time.

**Example of miscalibration:**
- Weather app predicts "30% chance of rain"
- But in reality, when it says 30%, it rains only 10% of the time → over-confident

**How to fix miscalibration:**
- **Platt scaling:** Fit a logistic regression on top of the model's outputs
- **Isotonic regression:** Non-parametric calibration (needs more data)

**When calibration matters:** Medical diagnosis (probability of disease), finance (probability of default)

---

## 9) Multiclass Logistic Regression (More Than 2 Classes)

### 9.1 One-vs-Rest (OvR) / One-vs-All (OvA)

Train K binary classifiers (one for each class). For class k, treat all examples of class k as positive, all others as negative.

**Example:** 3 classes: Dog, Cat, Bird
- Classifier 1: Dog vs (Cat or Bird)
- Classifier 2: Cat vs (Dog or Bird)
- Classifier 3: Bird vs (Dog or Cat)

Prediction: Pick the class whose classifier gives the highest probability.

**Pros:** Simple, works well  
**Cons:** Probabilities aren't calibrated across classes (they don't sum to 1)

### 9.2 Multinomial (Softmax Regression)

A single model that outputs probabilities for all classes that sum to 1.

**Softmax formula for class k:**

$$P(y=k | x) = \frac{e^{\theta_k^T x}}{\sum_{j=1}^K e^{\theta_j^T x}}$$

**Example:** Features: [weight=5kg, hasFeathers=1, sound='meow']

| Class | Score (θ^T x) | e^score | Probability |
|-------|---------------|---------|-------------|
| Dog | 2.1 | 8.17 | 0.15 |
| Cat | 3.5 | 33.1 | 0.60 |
| Bird | 1.8 | 6.05 | 0.25 |
| **Total** | | 47.32 | **1.00** |

The model predicts Cat with 60% confidence.

**When to use:** Classes are well-separated, you have enough data, and you want calibrated probabilities.

---

## 10) Handling Imbalanced Datasets

**Problem:** 99% of your data is "no", 1% is "yes". A model that always says "no" gets 99% accuracy but is useless.

### 10.1 Solutions

**Solution 1: Class Weights**
Assign higher weight to the minority class.

```python
# scikit-learn
model = LogisticRegression(class_weight='balanced')
# Or manually: class_weight={0: 1, 1: 10}  # 10x weight for class 1
```

**Effect:** Penalizes mistakes on minority class more heavily.

**Solution 2: Resampling**

- **Oversampling (SMOTE):** Create synthetic examples of the minority class
- **Undersampling:** Randomly remove examples from the majority class

**Solution 3: Use Appropriate Metrics**

- Don't use accuracy!
- Use Precision, Recall, F1, PR AUC

**Solution 4: Adjust Decision Threshold**

Lower the threshold to catch more positives.

**Solution 5: Use Algorithms Designed for Imbalanced Data**

- BalancedRandomForest
- XGBoost with `scale_pos_weight`

### 10.2 Example

Dataset: 1000 transactions, 10 are fraud (1%), 990 are legit (99%)

| Approach | Accuracy | Recall (fraud caught) | Precision (fraud claims correct) |
|----------|----------|----------------------|--------------------------------|
| Always say "legit" | 99% | 0% | N/A |
| Default logistic (0.5 threshold) | 98% | 40% | 50% |
| With class weights | 95% | 85% | 30% |
| + lower threshold to 0.3 | 92% | 92% | 20% |

There's always a tradeoff between recall and precision!

---

## 11) Assumptions and Limitations

### 11.1 Assumptions

1. **Linear relationship between features and log-odds** (not probability!)
   - Check: Plot log-odds vs each feature. Should be roughly linear.

2. **Independence of observations**
   - Logistic regression assumes each data point is independent (not time series or grouped data)

3. **No perfect multicollinearity**
   - Features should not be perfectly correlated (r = 1.0)
   - Solution: Remove one of the correlated features

4. **Large sample size** (rule of thumb: 10-20 events per feature)

### 11.2 Limitations

1. **Linear decision boundary only**
   - Cannot learn XOR, circles, or complex patterns
   - Solution: Add polynomial features (x₁², x₁x₂) or use a neural network

2. **Complete separation problem**
   - When data is perfectly separable, coefficients blow up
   - Solution: Add regularization

3. **Cannot handle missing values natively**
   - Must impute (fill in) missing data first

4. **Sensitive to outliers**
   - Extreme feature values can heavily influence the decision boundary

5. **Requires feature scaling for convergence**
   - Without scaling, optimization is slow

---

## 12) Practical Workflow (Step-by-Step)

### Step 1: Define the Problem
- What is Class 1 (the positive class)?
- What are the costs of false positives vs false negatives?

### Step 2: Explore Data
```python
# Check class balance
df['target'].value_counts(normalize=True)

# If imbalance > 10:1, plan to handle it
```

### Step 3: Preprocess
```python
from sklearn.preprocessing import StandardScaler

# Handle missing values
df = df.fillna(df.median())

# Scale features (crucial for logistic regression!)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Encode categorical variables
X = pd.get_dummies(X, drop_first=True)
```

### Step 4: Split Data
```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # preserve class balance
)
```

### Step 5: Train with Cross-Validation
```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l2'],
    'solver': ['lbfgs']
}

model = LogisticRegression(max_iter=1000)
grid = GridSearchCV(model, param_grid, cv=5, scoring='f1')
grid.fit(X_train, y_train)
```

### Step 6: Evaluate
```python
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

y_pred = grid.predict(X_test)
y_pred_proba = grid.predict_proba(X_test)[:, 1]

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(f"AUC: {roc_auc_score(y_test, y_pred_proba):.3f}")
```

### Step 7: Adjust Threshold (If Needed)
```python
from sklearn.metrics import precision_recall_curve

precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

# Find threshold that gives desired precision/recall tradeoff
desired_recall = 0.90
idx = np.argmin(np.abs(recalls - desired_recall))
optimal_threshold = thresholds[idx]

# Apply new threshold
y_pred_adjusted = (y_pred_proba >= optimal_threshold).astype(int)
```

### Step 8: Interpret Coefficients
```python
coefficients = grid.best_estimator_.coef_[0]
feature_names = X.columns

for name, coef in sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)[:10]:
    print(f"{name}: {coef:.3f} (odds ratio = {np.exp(coef):.2f})")
```

### Step 9: Calibrate (If Probability Quality Matters)
```python
from sklearn.calibration import CalibratedClassifierCV

calibrated = CalibratedClassifierCV(grid.best_estimator_, method='sigmoid', cv=5)
calibrated.fit(X_train, y_train)
```

---

## 13) Quick Comparison: Linear vs Logistic Regression

| Aspect | Linear Regression | Logistic Regression |
|--------|-------------------|----------------------|
| **Target variable** | Continuous (e.g., price, temperature) | Binary class (0/1) or probability |
| **Output range** | (-∞, +∞) | (0, 1) |
| **Equation** | y = θ^T x | p = 1/(1 + e^(-θ^T x)) |
| **Loss function** | Mean squared error | Log loss (binary cross-entropy) |
| **Convex?** | Yes (always) | Yes (with log loss) |
| **Closed-form solution?** | Yes (normal equation) | No (iterative only) |
| **Interpretation** | "Feature increases target by θ" | "Feature multiplies odds by e^θ" |
| **Use for** | Predicting sales, temperature, etc. | Predicting spam, disease, churn, etc. |

---

## 14) Common Pitfalls and How to Avoid Them

| Pitfall | Why It's Bad | How to Avoid |
|---------|--------------|---------------|
| Using accuracy on imbalanced data | Misleading results | Use F1, PR AUC, or balanced accuracy |
| Not scaling features | Slow convergence, unstable coefficients | Always use StandardScaler |
| Default threshold 0.5 | Suboptimal for business costs | Tune threshold based on cost matrix |
| No regularization | Overfitting, separation issues | Always use at least L2 regularization |
| Ignoring multicollinearity | Unstable coefficient estimates | Check correlation matrix, use L2 or drop features |
| Interpreting coefficients with unscaled features | Comparisons are meaningless | Interpret after scaling, or use standardized coefficients |

---

## Summary: The 5 Key Takeaways

1. **Logistic regression predicts probabilities** for classification using the sigmoid function: p = 1/(1+e^(-θ^T x))

2. **Log loss is the right loss function** because it heavily penalizes confident wrong predictions (unlike squared error)

3. **The loss function is convex** so optimization always finds the global minimum

4. **Regularization (L1 or L2) is essential** to prevent overfitting and handle separation

5. **Evaluation requires careful metric selection** — don't just use accuracy, especially with imbalanced data

---