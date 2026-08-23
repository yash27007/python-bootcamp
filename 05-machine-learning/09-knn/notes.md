# K-Nearest Neighbours

## 1. Problem

We want to classify or predict a value for a new data point, but we don't want to (or can't) commit to a fixed functional form — a line, a plane, a fixed set of weights — ahead of time. Many real relationships in feature space have structure that changes locally: a linear model that fits well in one region of the space may be completely wrong in another. We need a method that can classify or predict using only the data itself, with no fitted parameters, letting the local neighborhood of a query point speak for itself rather than forcing a single global rule to hold everywhere.

## 2. Intuition

The core idea: **similar data points exist close together** in feature space. To predict the label of a new point, look at its $K$ closest neighbors in the training set and let them "vote" (classification) or "average" (regression).

Concretely: if you want to guess a house's price and you know the prices of the 5 most similar houses nearby (same neighborhood, similar size, similar age), averaging those 5 prices is often a very reasonable guess — no formula about "price as a function of square footage" required. K-Nearest Neighbors (KNN) formalizes exactly this. It makes no assumption about the underlying data distribution (**non-parametric**) and does not build an explicit model during training (**lazy learning**) — it simply memorizes the entire training set and defers all computation to prediction time, which is why it is also called a **memory-based** method.

Prediction procedure:
1. Choose $K$ (number of neighbors) and a distance metric.
2. Compute the distance from the query point to every training point.
3. Sort distances ascending and take the $K$ closest points.
4. **Classification:** return the majority class among the $K$ neighbors.
5. **Regression:** return the mean (or weighted mean) of the target values among the $K$ neighbors.

## 3. Why simpler approaches fail

A global linear model (e.g., linear or logistic regression) fits a single set of weights over the *entire* feature space — one hyperplane, or one linear decision boundary, everywhere. This works only when the true relationship between features and target really is (approximately) linear across the whole space. Many real datasets don't behave that way: the relationship between features and the target can have wildly different behavior in different regions — a boundary that curves one way in one cluster of the data and the opposite way in another, or a target that depends on features nonlinearly and non-uniformly.

A global linear model cannot represent this local structure — it must pick one fixed set of coefficients that is a compromise across the whole space, which means it systematically underfits regions where the true local behavior deviates from that global average. KNN sidesteps this entirely by not fitting *any* global function: every prediction is made fresh, using only the training points actually near the query point, so it can represent decision boundaries or regression surfaces of arbitrary local shape — at the cost (explored in Failure modes) of needing enough nearby data and well-behaved distances to make "nearby" meaningful.

## 4. Mathematical foundation

### Distance metrics

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

### The effect of K: bias/variance tradeoff

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

### Prediction formulas

**Classification:** the predicted class is the mode (majority vote) of the labels of the $K$ nearest neighbors, optionally weighting each neighbor's vote by $1/d$ so that closer neighbors count more (`weights='distance'`), which also helps break ties.

**Regression:**
$$\hat{y}(x) = \frac{1}{K} \sum_{i \in N_K(x)} y_i$$

where $N_K(x)$ is the set of the $K$ nearest neighbors of $x$ in the training set.

### Cost function and training phase

KNN is known as a **"lazy learner"**. Unlike Linear Regression or SVMs, KNN does not have a training phase where an explicit cost function is minimized to learn weights or parameters. Instead, the algorithm simply stores the entire training dataset. The "cost" of computation is deferred to the prediction phase, where it must calculate the distance between the new point and all stored points (complexity of $O(N \times D)$). To optimize this query time, spatial data structures like **KD-Trees** or **Ball Trees** are often used.

## 5. Algorithm

1. **Store** the entire training set $\{(x_i, y_i)\}$ — there is no fitting step.
2. Given a query point $x$, **compute the distance** from $x$ to every stored training point using the chosen metric (Section 4).
3. **Select** the $K$ points with the smallest distance to $x$.
4. **Classification:** return the majority class label among those $K$ points (optionally distance-weighted).
   **Regression:** return the (optionally weighted) mean of the target values among those $K$ points.
5. Repeat steps 2–4 independently for every new query point — there is no reuse of computation across predictions beyond the stored training data itself.

## 6. From-scratch implementation

`knn_from_scratch.ipynb` implements the algorithm above directly in NumPy on a small toy 2D dataset, with no scikit-learn model involved:

1. **Pairwise Euclidean distances** from a query point to every training point, computed directly from the Euclidean formula in Section 4 using vectorized NumPy operations (`np.linalg.norm` over broadcast differences).
2. **k-nearest selection** — `np.argsort` on the distance array, taking the first $K$ indices.
3. **Majority vote** — `collections.Counter` (or `np.bincount`) over the labels of those $K$ indices to produce the predicted class.
4. A visualization of the resulting decision boundary on the toy dataset, and a sanity check against `sklearn.neighbors.KNeighborsClassifier` on the same data to confirm the from-scratch predictions agree with the library implementation.

This makes concrete, in a handful of lines, exactly what Section 5's algorithm steps mean in code — there is no hidden optimization loop or numerical subtlety being skipped; distance-compute → sort → vote *is* the whole algorithm.

## 7. Practical implementation

`knn.ipynb` maps the from-scratch mechanism above onto scikit-learn's production implementations:

- **`KNeighborsClassifier`** replaces the from-scratch "compute all distances, sort, take K, majority vote" loop with an internally optimized version of the same steps — by default it uses a `KDTree`/`BallTree` (Section 4) instead of brute-force distance computation to every point, which is the practical answer to the $O(N \times D)$ prediction-time cost noted there, and becomes important once $N$ is large (see Failure modes).
- **`KNeighborsRegressor`** applies the exact same nearest-neighbor-then-average logic from Section 4's regression formula, again accelerated with a spatial index instead of brute-force distance computation.
- Both are wrapped in a `Pipeline` with `StandardScaler`, directly reflecting the scale-sensitivity point from Section 4 — the from-scratch toy dataset in Section 6 is small and already comparably scaled on both axes, but real feature sets rarely are, so the practical implementation makes standardization explicit.

## 8. Experiment

The existing `knn.ipynb` notebook already contains this topic's experiment, testing the bias-variance effect of $K$ described in Section 4:

- **Hypothesis (stated before running):** cross-validated classification accuracy should rise as $K$ increases from 1 (reducing variance/overfitting to noise), reach a peak, and then decline (or plateau) for very large $K$ as the model becomes too smooth and underfits — the classic bias-variance "elbow" shape.
- **Setup:** `KNeighborsClassifier` inside a `StandardScaler` pipeline, swept over $K = 1, \dots, 30$ on the Iris dataset, evaluated with 5-fold cross validation at each $K$; the best $K$ is then re-fit on a held-out train/test split and reported.
- **Result:** the notebook plots CV accuracy vs. $K$ (with a shaded standard-deviation band) and marks the best $K$; a companion `KNeighborsRegressor` experiment on California Housing reports RMSE and R² for a fixed $K=10$.
- **Interpretation:** the notebook's takeaway cell confirms the hypothesis — accuracy rises sharply from $K=1$ (overfit, noisy) then plateaus/slightly declines for very large $K$ (underfit), with the peak marking the sweet spot; the regression RMSE/R² are noted to depend heavily on the `StandardScaler` step given how differently scaled the housing features are.
- **Limitations:** the sweep is done on a single dataset (Iris) with only 5-fold CV, so the exact best-$K$ value is dataset-specific and would need re-running per problem; the regression experiment fixes $K=10$ rather than also sweeping it.

## 9. Failure modes

**Curse of dimensionality.** As the number of feature dimensions $D$ grows, distances between points become progressively less informative. Intuitively: in high dimensions, the volume of feature space grows exponentially with $D$, so a fixed number of training points becomes exponentially sparser, and the points that are "nearest" to a query point are, on average, almost as far away as the points that are "farthest" — the ratio between the nearest and farthest distances tends toward 1 as $D \to \infty$. When every point is roughly equidistant from every other point, the entire notion of "nearest neighbor" stops carrying useful information, and KNN's core assumption (similar points are close together) breaks down. In practice this means KNN tends to perform poorly on high-dimensional data unless dimensionality reduction (e.g., PCA) or feature selection is applied first.

**Sensitivity to feature scaling.** Because every distance metric in Section 4 sums (or takes the max of) differences across raw feature values, a feature measured on a much larger numeric scale (e.g., income in dollars vs. age in years) will dominate the distance calculation regardless of its actual predictive relevance. Unscaled KNN effectively lets whichever feature happens to have the largest numeric range decide "closeness" — standardizing (zero mean, unit variance) or normalizing features before fitting is close to mandatory, not optional, as reflected in the `StandardScaler` pipelines used in Sections 7 and 8.

**Slow prediction time with large $N$.** Because KNN performs no training-time compression of the data, every single prediction requires searching for nearest neighbors among all $N$ stored training points (Section 4's $O(N \times D)$ brute-force cost). Unlike a parametric model, where prediction cost is fixed once training is done, KNN's prediction cost grows with the size of the training set — this makes naive KNN impractical for latency-sensitive applications with millions of training points, unless spatial indexing (KD-Trees, Ball Trees) or approximate nearest-neighbor techniques are used to reduce the effective search cost.

## 10. Real-world usage

- **Recommendation systems:** "users/items similar to this one" is a direct nearest-neighbor query in an embedding space.
- **Anomaly detection:** points whose nearest neighbors are all unusually far away are flagged as outliers — a direct use of the same distance computation.
- **Image and pattern recognition baselines:** nearest-neighbor search over pixel or embedding features is a classic, easily-understood baseline before more complex models.
- **Imputation:** missing values can be filled in with the average of a data point's K nearest neighbors on the remaining features.
- **Approximate nearest-neighbor (ANN) search at scale:** the core KNN idea underlies large-scale vector search systems (e.g., for semantic search or retrieval-augmented generation), where approximate indexing structures make the same "find nearby points" query fast even with billions of vectors.

## 11. Mental model

KNN doesn't learn a rule — it learns nothing at all during training. Every prediction is answered by asking "who's standing near me right now?" and copying the answer from the crowd. That's powerful because it can represent arbitrarily local structure with no assumptions about global shape, and it's expensive because "who's standing near me" has to be recomputed, from scratch, against every other point, every single time you ask.

## 12. Questions to think about

1. Why does the "curse of dimensionality" argument imply that KNN's effectiveness tends to degrade as more (possibly irrelevant) features are added, even if those extra features contain no useful signal?
2. If you forgot to standardize features before running KNN and one feature had a numeric range 1000x larger than the others, what would you expect the fitted decision boundary to look like, and why?
3. Why does K=1 have zero bias in the sense of always agreeing with its single nearest training point, yet high variance — and why is a K approaching $N$ the opposite?
4. If prediction latency mattered enormously (e.g., real-time serving with a huge training set), what changes could you make to KNN's setup (algorithmic or data-structural) to make it fast enough, and what would each change trade away?
5. A linear model and a K=1 nearest-neighbor classifier are fit to the same dataset. Describe a dataset shape where the linear model would clearly outperform K=1 KNN, and another shape where the reverse would be true.
