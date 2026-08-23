# K-Nearest Neighbours

| Topic | Status |
|-------|--------|
| KNN Intuition (Instance-based / Lazy Learning) | ✅ Complete |
| Distance Metrics | ✅ Complete |
| Effect of K (Bias/Variance) | ✅ Complete |
| KNN for Classification | ✅ Complete |
| KNN for Regression | ✅ Complete |

## KNN Intuition: Instance-Based / Lazy Learning

K-Nearest Neighbors (KNN) is one of the simplest algorithms in machine learning. It makes no assumption about the underlying data distribution (**non-parametric**) and does not build an explicit model during training (**lazy learning**).

The core idea: **similar data points exist close together** in feature space. To predict the label of a new point, look at its $K$ closest neighbors in the training set and let them "vote" (classification) or "average" (regression).

Because there is no training-time optimization, KNN is often called a **memory-based** method — it simply memorizes the entire training set and defers all computation to prediction time.

Prediction procedure:
1. Choose $K$ (number of neighbors) and a distance metric.
2. Compute the distance from the query point to every training point.
3. Sort distances ascending and take the $K$ closest points.
4. **Classification:** return the majority class among the $K$ neighbors.
5. **Regression:** return the mean (or weighted mean) of the target values among the $K$ neighbors.

## Distance Metrics

The notion of "closeness" is entirely determined by the distance metric, so the choice matters a lot.

**Euclidean Distance** ($p=2$, straight-line distance — most common for continuous features):
$$d(p, q) = \sqrt{\sum_{i=1}^n (q_i - p_i)^2}$$

**Manhattan Distance** ($p=1$, city-block distance — sum of absolute differences, useful when movement is grid-like or features are on different independent axes):
$$d(p, q) = \sum_{i=1}^n |q_i - p_i|$$

**Minkowski Distance** (generalization that unifies the two above via a parameter $p$):
$$d(p, q) = \left( \sum_{i=1}^n |q_i - p_i|^p \right)^{\frac{1}{p}}$$

- $p=1$ recovers Manhattan distance.
- $p=2$ recovers Euclidean distance.
- As $p \to \infty$, Minkowski distance approaches the **Chebyshev distance** (max absolute difference along any single dimension).

Because distance metrics are scale-sensitive, features should generally be **standardized/normalized** before applying KNN — otherwise a feature with a larger numeric range dominates the distance calculation.

## The Effect of K: Bias/Variance Tradeoff

$K$ is the single most important hyperparameter in KNN, and it directly controls the bias-variance tradeoff:

| K | Bias | Variance | Decision Boundary | Risk |
|---|------|----------|--------------------|------|
| Small (e.g. K=1) | Low | High | Very jagged, follows noise closely | Overfitting |
| Large (e.g. K=N) | High | Low | Very smooth, approaches the majority class everywhere | Underfitting |

- With **K=1**, the prediction is just the single nearest neighbor's label — extremely flexible but highly sensitive to noise and outliers (high variance).
- With a **very large K** (approaching the size of the dataset), the model essentially predicts the global majority class / global mean everywhere — very stable but ignores local structure (high bias).

The optimal $K$ balances these two extremes. It is typically found via **cross validation**: sweep $K$ over a range of values, evaluate CV accuracy (or CV RMSE for regression) at each $K$, and pick the point where held-out performance peaks (or the "elbow" where more neighbors stop giving a meaningful boost).

Practical guidance:
- $K$ is usually chosen as an odd number for binary classification to avoid tie votes.
- $\sqrt{N}$ is a common rule-of-thumb starting point for $K$.

## KNN for Classification

`KNeighborsClassifier` assigns the new point the majority class label among its $K$ nearest neighbors. Ties can be broken by distance-weighting (`weights='distance'`), which gives closer neighbors more influence than farther ones.

## KNN for Regression

`KNeighborsRegressor` predicts a continuous target by averaging (optionally distance-weighted) the target values of the $K$ nearest neighbors:

$$\hat{y}(x) = \frac{1}{K} \sum_{i \in N_K(x)} y_i$$

where $N_K(x)$ is the set of the $K$ nearest neighbors of $x$ in the training set.

## Cost Function & Training Phase

KNN is known as a **"lazy learner"**. Unlike Linear Regression or SVMs, KNN does not have a training phase where an explicit cost function is minimized to learn weights or parameters.

Instead, the algorithm simply stores the entire training dataset. The "cost" of computation is deferred to the prediction phase, where it must calculate the distance between the new point and all stored points (complexity of $O(N \times D)$). To optimize this query time, spatial data structures like **KD-Trees** or **Ball Trees** are often used.

## Pros and Cons

**Pros:**
- Simple and intuitive; no training phase.
- Naturally handles multi-class classification.
- Can model complex, non-linear decision boundaries given enough data.

**Cons:**
- Slow prediction for large datasets (no compact model is learned).
- Sensitive to feature scaling and irrelevant/noisy features.
- Suffers from the **curse of dimensionality**: in high dimensions, all points tend to become roughly equidistant, making "nearest" less meaningful.
