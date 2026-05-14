# Cross Validation

## What is cross validation?
Cross validation is a set of evaluation techniques that reuse the available data to estimate how well a model will generalize to unseen data. Instead of relying on a single train/test split (which can be noisy or unlucky), cross validation repeatedly trains the model on different subsets of the data and evaluates it on the remaining subset. The final performance is summarized across all splits.

Think of it as **rotating the test set**: every example gets a chance to be in the test set at least once.

## Why cross validation is used

### 1) Better estimate of generalization performance
A single split can be overly optimistic or pessimistic depending on which points land in the test set. Cross validation averages performance across multiple splits, making the estimate more stable.

### 2) Efficient use of limited data
When the dataset is small, holding out a large test set wastes valuable training data. Cross validation uses most of the data for training in each fold while still evaluating on unseen data.

### 3) Model selection and hyperparameter tuning
Cross validation is the standard approach for comparing different models or hyperparameters. It reduces the risk of choosing a model that only performs well on a particular split.

### 4) Diagnosing bias and variance
By looking at the variation of scores across folds, you can infer whether the model is sensitive to data changes (high variance) or consistently underfits (high bias).

## Basic workflow (K-Fold)
1) Shuffle (optional) and split the dataset into K equal-sized folds.
2) For each fold:
	- Train on K-1 folds.
	- Evaluate on the remaining fold.
3) Aggregate the K scores (mean, std).

The final reported metric is often the mean score (and its standard deviation).

## Types of cross validation (with examples)

### 1) Hold-out (train/test split)
This is the simplest evaluation. It is not true cross validation but is often mentioned alongside it.

**Example:**
- Dataset: 1,000 rows
- Split: 80% training (800) and 20% testing (200)
- Train once, test once

**Pros:** simple, fast
**Cons:** high variance; result depends heavily on the split

### 2) K-Fold Cross Validation
Split data into K folds (commonly K=5 or K=10). Train K times, each time leaving out one fold for testing.

**Example:**
- Dataset: 1,000 rows
- K = 5
- Each fold has 200 rows
- Train on 800, test on 200, repeated 5 times
- Final score = average of 5 test scores

**Pros:** good balance between bias and variance
**Cons:** can be expensive for large datasets or slow models

### 3) Stratified K-Fold
Used for classification when classes are imbalanced. Each fold preserves class proportions.

**Example:**
- Dataset: 1,000 rows
- Class A: 900, Class B: 100
- In each fold, ~90% A and ~10% B

**Why it matters:** A normal K-Fold might create a fold with very few or no minority samples, causing unstable scores.

### 4) Leave-One-Out (LOOCV)
Each fold uses exactly one observation for testing and the rest for training. K equals the number of samples.

**Example:**
- Dataset: 100 rows
- 100 folds
- Train on 99, test on 1

**Pros:** maximum training data per fold, almost unbiased estimate
**Cons:** very expensive; high variance in scores; can be noisy for some models

### 5) Leave-P-Out
Generalization of LOOCV where P samples are left out for testing.

**Example:**
- Dataset: 50 rows
- P = 2
- Each iteration tests on a pair of samples

**Pros:** more stable than LOOCV
**Cons:** combinatorial explosion; rarely used for large datasets

### 6) Repeated K-Fold
Repeat K-Fold multiple times with different random shuffles to reduce variance.

**Example:**
- K = 5
- Repeats = 3
- Total evaluations = 15

**Pros:** more robust estimate
**Cons:** more computation

### 7) Nested Cross Validation
Used when hyperparameter tuning is involved. There are two loops:
- Outer loop: estimates generalization performance.
- Inner loop: selects hyperparameters.

**Example:**
- Outer 5-fold CV for performance
- Inner 3-fold CV to pick best C for logistic regression

**Why it matters:** prevents optimistic bias from tuning on the same folds you evaluate.

### 8) Time Series Cross Validation (Rolling / Forward Chaining)
Used for data with temporal order. You must respect time order and avoid training on future data.

**Example (rolling):**
- Train: months 1-3, test: month 4
- Train: months 1-4, test: month 5
- Train: months 1-5, test: month 6

**Pros:** realistic for forecasting
**Cons:** fewer independent test sets

### 9) Group K-Fold
Used when observations are grouped and groups must not be split across train/test.

**Example:**
- Dataset: medical records per patient
- Groups: patient_id
- All records of a patient must stay in one fold

**Why it matters:** prevents data leakage when samples within a group are correlated.

### 10) Monte Carlo / Shuffle-Split
Randomly split data into train/test multiple times.

**Example:**
- Repeat 10 times
- Each time: 80% train, 20% test
- Average the 10 scores

**Pros:** flexible train/test sizes
**Cons:** some samples may never appear in test sets

### 11) Bootstrap (for model stability)
Randomly sample with replacement to create training sets; test on out-of-bag samples.

**Example:**
- Sample N rows with replacement as train
- Evaluate on rows not selected (out-of-bag)

**Pros:** useful for small datasets and estimating uncertainty
**Cons:** not always appropriate for strict generalization error

## Practical examples

### Example 1: K-Fold for regression
Suppose you have 1,000 housing records and want to compare linear regression vs. random forest.

- Use 5-fold CV
- Compute RMSE for each fold
- Average RMSE across folds
- Choose the model with lower mean RMSE (and preferably lower variance)

This avoids making a decision based on a single lucky split.

### Example 2: Stratified K-Fold for classification
You have a fraud detection dataset with 98% non-fraud and 2% fraud.

- Use stratified 5-fold CV
- Evaluate with precision, recall, and F1
- Ensure each fold has ~2% fraud

This gives a more trustworthy estimate of minority-class performance.

### Example 3: Time Series CV for forecasting
You want to predict daily sales.

- Start with first 6 months for training
- Predict the next month
- Expand the training window and repeat

This respects the temporal order and simulates real forecasting.

## Common pitfalls and best practices

### Avoid data leakage
- Always apply preprocessing (scaling, encoding, feature selection) **inside** each training fold.
- In scikit-learn, use `Pipeline` to ensure transformations are fit only on training folds.

### Keep a final hold-out test set
Even with cross validation, it is good practice to keep a final unseen test set for a last unbiased evaluation.

### Choose a metric aligned with the task
- Classification: accuracy may be misleading on imbalanced data; use F1, ROC-AUC, PR-AUC.
- Regression: use MAE, RMSE, or R2 depending on the business context.

### Watch for high variance
If scores vary widely across folds, the model may be unstable. Consider simpler models, more data, or repeated CV.

## Quick comparison table

| Method | Best for | Key idea | Cost |
| --- | --- | --- | --- |
| Hold-out | Quick baseline | Single split | Low |
| K-Fold | General use | Rotate test fold | Medium |
| Stratified K-Fold | Imbalanced classes | Preserve class ratios | Medium |
| LOOCV | Very small data | Leave 1 out | High |
| Repeated K-Fold | More stability | Repeat splits | High |
| Nested CV | Model selection | Inner/outer loops | High |
| Time Series CV | Temporal data | Forward chaining | Medium |
| Group K-Fold | Grouped data | Keep groups intact | Medium |

## Summary
Cross validation is essential for reliable model evaluation and selection. It reduces the randomness of a single split, uses data efficiently, and helps detect overfitting. The choice of cross validation method depends on the data structure (imbalanced, grouped, temporal) and the computational budget.
