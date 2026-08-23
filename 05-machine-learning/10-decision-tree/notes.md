# Decision Trees

| Topic | Status |
|-------|--------|
| Tree Structure (Root/Internal/Leaf) | ✅ Complete |
| Splitting Criteria (Gini, Entropy, MSE) | ✅ Complete |
| Overfitting and Pruning | ✅ Complete |

## Overview
Decision Trees are versatile, non-parametric supervised learning algorithms. They predict the value of a target variable by learning simple "if-then" decision rules inferred from the data features. They resemble a tree-like flowchart where each internal node represents a test on a feature, each branch represents an outcome of the test, and each leaf node holds a class label or continuous value.

## Tree Structure

A decision tree is made of three kinds of nodes:

- **Root node:** the topmost node, containing the full dataset before any split. The first, most informative feature test happens here.
- **Internal (decision) nodes:** intermediate nodes that each test a single feature (e.g., "is `petal_length` < 2.5?") and branch into child nodes based on the outcome.
- **Leaf (terminal) nodes:** nodes with no further children. Each leaf holds the final prediction — a class label (classification, typically the majority class of the samples that reached it) or a numeric value (regression, typically the mean of the samples that reached it).

Prediction works by starting at the root and following the branch dictated by each feature test until a leaf is reached; the leaf's stored value is the prediction.

## Splitting Criteria (Cost Functions for Trees)

Decision trees are built recursively (top-down, greedy) by splitting the dataset into subsets. At every node, the algorithm evaluates candidate splits (feature + threshold) and chooses the one that maximizes information gain or minimizes impurity — the "cost" of the split.

### 1) Classification Cost Functions

**Entropy:**
Measures the level of impurity or uncertainty in a group of examples $S$. For a dataset with $c$ classes, where $p_i$ is the probability of an item belonging to class $i$:
$$H(S) = - \sum_{i=1}^c p_i \log_2(p_i)$$

Entropy is 0 for a pure node (all one class) and maximal ($\log_2 c$) when classes are perfectly balanced.

**Information Gain:**
The reduction in entropy achieved by partitioning the dataset $S$ according to a feature $A$. The split that maximizes this gain is chosen.
$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

**Gini Impurity:**
A measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. It is computationally cheaper than Entropy (no logarithm) and is scikit-learn's default (`criterion='gini'`).
$$Gini(S) = 1 - \sum_{i=1}^c p_i^2$$

The algorithm seeks splits that **minimize** the weighted Gini impurity of the resulting children. In practice, Gini and Entropy usually produce very similar trees.

### 2) Regression Cost Functions

For Decision Tree Regression, splits are evaluated based on the variance of the targets, usually aiming to minimize the **Mean Squared Error (MSE)** in the child nodes:

$$MSE = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

Where $\hat{y}_i$ is the mean of the target values in that leaf node. The algorithm chooses the split that minimizes the weighted sum of the MSE of the left and right child nodes.

## Overfitting and Pruning

Because Decision Trees are grown greedily until every leaf is pure (or another stopping condition is hit), they are extremely prone to **overfitting**: a fully grown tree can perfectly memorize the training data (including its noise), producing a very deep, high-variance model that generalizes poorly.

Overfitting is controlled by **pruning**, which comes in two flavors:

### 1) Pre-pruning (Early Stopping)
Stop growing the tree before it becomes fully pure, by constraining its growth:

| Hyperparameter | Effect |
|-----------------|--------|
| `max_depth` | Caps how many levels deep the tree may grow. Smaller → simpler, higher-bias tree. |
| `min_samples_split` | Minimum number of samples a node must have before it is allowed to split. |
| `min_samples_leaf` | Minimum number of samples required to be at a leaf node. Prevents leaves fit to a single/few outlier points. |
| `max_leaf_nodes` | Caps the total number of leaves in the tree. |

### 2) Post-pruning (Cost Complexity Pruning)
Grow the full tree first, then remove branches that don't provide enough predictive power. Scikit-learn implements **Minimal Cost-Complexity Pruning** via the `ccp_alpha` parameter.

The idea: for a subtree $T$, define a cost-complexity measure that penalizes both misclassification/impurity and tree size:
$$R_\alpha(T) = R(T) + \alpha \cdot |\tilde{T}|$$

where $R(T)$ is the total impurity (or MSE) of the tree's leaves, $|\tilde{T}|$ is the number of leaf nodes, and $\alpha \geq 0$ is the complexity parameter controlling the size/accuracy tradeoff. Increasing $\alpha$ prunes more aggressively (fewer leaves, simpler tree); $\alpha = 0$ recovers the fully grown, unpruned tree.

In practice, `cost_complexity_pruning_path` computes a sequence of candidate $\alpha$ values and their corresponding pruned trees; the best $\alpha$ is then chosen via cross-validation.

## Pros and Cons

**Pros:**
- Highly interpretable — can be visualized and understood by non-experts.
- Requires little data preprocessing (no need to scale/normalize features).
- Naturally handles both numerical and categorical data, and captures non-linear relationships.

**Cons:**
- Prone to overfitting without pruning; high variance (small data changes can produce a very different tree).
- Greedy splitting is not guaranteed to find the globally optimal tree.
- Can create biased trees if classes are imbalanced (needs `class_weight` balancing).
- Poor extrapolation for regression — predictions are bounded by the range of leaf values seen in training.
