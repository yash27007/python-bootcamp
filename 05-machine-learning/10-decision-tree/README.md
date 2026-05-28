# 10 – Decision Trees

## Overview
Decision Trees are versatile, non-parametric supervised learning algorithms. They predict the value of a target variable by learning simple "if-then" decision rules inferred from the data features. They resemble a tree-like flowchart where each internal node represents a test on a feature, each branch represents an outcome of the test, and each leaf node holds a class label or continuous value.

## Splitting Criteria (Cost Functions for Trees)
Decision trees are built recursively by splitting the dataset into subsets. The algorithm chooses the split that maximizes information gain or minimizes impurity (the "cost").

### 1. Classification Cost Functions

**Entropy:**
Measures the level of impurity or uncertainty in a group of examples $S$. For a dataset with $c$ classes, where $p_i$ is the probability of an item belonging to class $i$:
$$H(S) = - \sum_{i=1}^c p_i \log_2(p_i)$$

**Information Gain:**
The reduction in entropy achieved by partitioning the dataset $S$ according to a feature $A$. The split that maximizes this gain is chosen.
$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

**Gini Impurity:**
A measure of how often a randomly chosen element from the set would be incorrectly labeled if it was randomly labeled according to the distribution of labels in the subset. It is computationally faster than Entropy.
$$Gini(S) = 1 - \sum_{i=1}^c p_i^2$$

The algorithm seeks splits that **minimize** Gini impurity.

### 2. Regression Cost Functions

For Decision Tree Regression, splits are evaluated based on the variance of the targets, usually aiming to minimize the **Mean Squared Error (MSE)** in the child nodes:

$$MSE = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$
Where $\hat{y}_i$ is the mean of the target values in that leaf node. The algorithm chooses the split that minimizes the weighted sum of the MSE of the left and right child nodes.

## Optimization & Pruning
Because Decision Trees are prone to overfitting (growing until every leaf is pure), they are often regulated through:
- **Pre-pruning (Early Stopping):** Limiting tree depth, requiring a minimum number of samples to split, or a minimum number of samples in a leaf.
- **Post-pruning (Cost Complexity Pruning):** Growing the full tree and then removing branches that do not provide significant predictive power, minimizing an objective function that penalizes tree size.
