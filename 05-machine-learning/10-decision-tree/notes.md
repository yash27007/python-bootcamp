# Decision Trees

## Problem

Real-world decisions are usually a sequence of yes/no questions, not a single formula. "Should this
loan be approved?" isn't answered by one linear score — a human loan officer implicitly reasons:
"Is income above $50k? If yes, is debt-to-income below 30%? If no to that, is there a co-signer?
..." Hand-coding these rules as nested `if`/`elif` statements works for a handful of rules an expert
already knows, but it doesn't scale: real datasets have dozens of features, the "right" thresholds
aren't known in advance, and the rules that actually separate the classes have to be *discovered*
from data, not guessed by a human staring at a spreadsheet.

The problem decision trees solve: **learn a sequence of yes/no (or threshold) decisions directly
from data, instead of hand-coding if-else rules**, such that following the sequence from the root to
a leaf produces an accurate prediction.

## Intuition

A decision tree is a game of 20 questions played against the data. It is built from three kinds of
nodes:

- **Root node:** the topmost node, containing the full dataset before any split. The first, most
  informative feature test happens here.
- **Internal (decision) nodes:** intermediate nodes that each test a single feature (e.g., "is
  `petal_length` < 2.5?") and branch into child nodes based on the outcome.
- **Leaf (terminal) nodes:** nodes with no further children. Each leaf holds the final prediction — a
  class label (classification, typically the majority class of the samples that reached it) or a
  numeric value (regression, typically the mean of the samples that reached it).

Prediction works by starting at the root and following the branch dictated by each feature test
until a leaf is reached; the leaf's stored value is the prediction.

Concretely: to classify a tumor as benign/malignant, the tree might first ask "is `mean radius` <
15?" — if yes, ask "is `worst texture` < 20?" — and so on, until it reaches a leaf where, say, 95 of
the 100 training examples that ended up there were benign, so it predicts benign. Each question is
chosen because it's the single most useful question to ask *given everything already known from
earlier questions* — this greedy, recursive question-asking is the entire algorithm.

## Why simpler approaches fail

A linear model (logistic regression, a linear SVM) draws exactly one straight boundary (or
hyperplane) through feature space and predicts based on which side of it a point falls. This fails
whenever the true decision rule is not linearly separable — the canonical case is an **XOR-like**
rule: "approve the loan if (`has_collateral` AND `low_debt`) OR (`no_collateral` AND `high_income`)".
Plot this in 2D and the positive class occupies two diagonally-opposite quadrants; no single straight
line separates them from the negative class, no matter how the line is rotated or shifted. A
logistic regression or linear SVM on these two raw features is stuck near 50% accuracy.

A decision tree handles this trivially: it doesn't need one global boundary. It can ask
"`has_collateral`?" first, then ask a *different* follow-up question in each branch
(`low_debt` in the collateral branch, `high_income` in the no-collateral branch). The tree's
axis-aligned, recursively-partitioned decision boundary is a staircase that can carve out
arbitrarily shaped (in particular, non-convex and disjoint) regions — something a single linear
boundary structurally cannot do.

## Mathematical foundation

Decision trees are built recursively (top-down, greedy) by splitting the dataset into subsets. At
every node, the algorithm evaluates candidate splits (feature + threshold) and chooses the one that
maximizes information gain or minimizes impurity — the "cost" of the split.

### Classification cost functions

**Entropy:** measures the level of impurity or uncertainty in a group of examples $S$. For a dataset
with $c$ classes, where $p_i$ is the probability (empirical frequency) of an item belonging to class
$i$:
$$H(S) = - \sum_{i=1}^c p_i \log_2(p_i)$$

Entropy is 0 for a pure node (all one class) and maximal ($\log_2 c$) when classes are perfectly
balanced.

**Information Gain:** the reduction in entropy achieved by partitioning the dataset $S$ according to
a feature $A$. The split that maximizes this gain is chosen.
$$IG(S, A) = H(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} H(S_v)$$

**Gini Impurity:** a measure of how often a randomly chosen element from the set would be incorrectly
labeled if it was randomly labeled according to the distribution of labels in the subset. It is
computationally cheaper than Entropy (no logarithm) and is scikit-learn's default
(`criterion='gini'`).
$$Gini(S) = 1 - \sum_{i=1}^c p_i^2$$

The algorithm seeks splits that **minimize** the weighted Gini impurity of the resulting children —
i.e. it maximizes the *Gini reduction*, the impurity analogue of information gain:
$$\Delta Gini(S, A) = Gini(S) - \sum_{v \in Values(A)} \frac{|S_v|}{|S|} Gini(S_v)$$

In practice, Gini and Entropy usually produce very similar trees.

### Regression cost function

For Decision Tree Regression, splits are evaluated based on the variance of the targets, usually
aiming to minimize the **Mean Squared Error (MSE)** in the child nodes:
$$MSE = \frac{1}{N} \sum_{i=1}^N (y_i - \hat{y}_i)^2$$

Where $\hat{y}_i$ is the mean of the target values in that leaf node. The algorithm chooses the split
that minimizes the weighted sum of the MSE of the left and right child nodes.

### Cost-complexity pruning (the math behind post-pruning)

Because trees are grown greedily until every leaf is pure (or another stopping condition is hit),
they are extremely prone to **overfitting**. Scikit-learn implements **Minimal Cost-Complexity
Pruning** via the `ccp_alpha` parameter. For a subtree $T$, define a cost-complexity measure that
penalizes both misclassification/impurity and tree size:
$$R_\alpha(T) = R(T) + \alpha \cdot |\tilde{T}|$$

where $R(T)$ is the total impurity (or MSE) of the tree's leaves, $|\tilde{T}|$ is the number of leaf
nodes, and $\alpha \geq 0$ is the complexity parameter controlling the size/accuracy tradeoff.
Increasing $\alpha$ prunes more aggressively (fewer leaves, simpler tree); $\alpha = 0$ recovers the
fully grown, unpruned tree.

## Algorithm

```
BUILD_TREE(node, data, depth):
    if stopping_condition(data, depth):        # pure node, max_depth, min_samples, ...
        make node a leaf (majority class / mean target)
        return

    best_feature, best_threshold = None, None
    best_score = current impurity/MSE of `data`

    for feature in features:
        for threshold in candidate_thresholds(feature, data):
            left, right = split(data, feature, threshold)
            score = weighted_impurity(left) + weighted_impurity(right)   # Gini/Entropy/MSE
            if score improves on best_score:
                best_feature, best_threshold, best_score = feature, threshold, score

    if no split improves impurity:
        make node a leaf
        return

    split data into left/right using (best_feature, best_threshold)
    BUILD_TREE(node.left, left, depth+1)
    BUILD_TREE(node.right, right, depth+1)
```

Prediction: start at the root, at each internal node follow the branch matching the feature test,
until a leaf is reached; return the leaf's stored value.

### Pruning strategies (controlling overfitting)

**1) Pre-pruning (early stopping)** — stop growing the tree before it becomes fully pure, by
constraining its growth:

| Hyperparameter | Effect |
|-----------------|--------|
| `max_depth` | Caps how many levels deep the tree may grow. Smaller → simpler, higher-bias tree. |
| `min_samples_split` | Minimum number of samples a node must have before it is allowed to split. |
| `min_samples_leaf` | Minimum number of samples required to be at a leaf node. Prevents leaves fit to a single/few outlier points. |
| `max_leaf_nodes` | Caps the total number of leaves in the tree. |

**2) Post-pruning (cost-complexity pruning)** — grow the full tree first, then remove branches that
don't provide enough predictive power relative to the $R_\alpha(T)$ tradeoff above. In practice,
`cost_complexity_pruning_path` computes a sequence of candidate $\alpha$ values and their
corresponding pruned trees; the best $\alpha$ is then chosen via cross-validation.

## From-scratch implementation

The one operation that makes up the entire `BUILD_TREE` algorithm above is: *given one feature
column and the labels, find the threshold that minimizes weighted Gini impurity of the children*.
`05-machine-learning/10-decision-tree/decision-tree.ipynb` (section 0) implements exactly this in
plain NumPy on a small toy dataset (`hours studied -> pass/fail`, 10 points, one point out of order
so the answer isn't visually obvious):

```python
def gini(y):
    _, counts = np.unique(y, return_counts=True)
    p = counts / counts.sum()
    return 1 - np.sum(p ** 2)

def weighted_gini_of_split(x_col, y, threshold):
    left_mask = x_col <= threshold
    right_mask = ~left_mask
    n_left, n_right, n = left_mask.sum(), right_mask.sum(), len(y)
    if n_left == 0 or n_right == 0:
        return np.inf
    return (n_left / n) * gini(y[left_mask]) + (n_right / n) * gini(y[right_mask])

def best_split_gini(x_col, y):
    candidates = np.unique(x_col)
    thresholds = (candidates[:-1] + candidates[1:]) / 2
    scores = np.array([weighted_gini_of_split(x_col, y, t) for t in thresholds])
    best_idx = np.argmin(scores)
    return thresholds[best_idx], scores[best_idx], thresholds, scores
```

Running this on `hours = [1..10]`, `passed = [0,0,0,0,1,0,1,1,1,1]` gives:
parent Gini impurity 0.5000, best threshold `hours <= 4.5`, weighted children Gini 0.1667 (a
reduction of 0.3333). As a sanity check, the notebook then fits `DecisionTreeClassifier(max_depth=1)`
on the same data and reads off `tree_.threshold[0]` — it agrees exactly (`4.5`), confirming the
from-scratch search reproduces what sklearn's compiled implementation does internally.

Real decision trees just repeat this threshold search over *every* feature at *every* node,
recursively, until a stopping condition is hit — that's the whole algorithm.

## Practical implementation

`sklearn.tree.DecisionTreeClassifier` / `DecisionTreeRegressor` implement exactly the `BUILD_TREE`
algorithm above, but in optimized C, searching every feature (not just one) at every node, with
`criterion='gini'` (default) or `'entropy'` for classification and `'squared_error'` for regression.
`ccp_alpha` implements the cost-complexity pruning math from the Mathematical foundation section.
`sklearn.tree.plot_tree` visualizes the resulting structure directly.

Sections 1–2 of `decision-tree.ipynb` apply this to two real datasets:

- `DecisionTreeClassifier` on `load_breast_cancer` (unpruned vs. `max_depth=3`), visualized with
  `plot_tree`.
- `DecisionTreeRegressor` on `fetch_california_housing`, reporting RMSE/R².

This is the exact same Gini-threshold search performed in the from-scratch step — just applied
recursively, over all 30 (or 8) features, at every node of a much deeper tree, instead of once over
one feature by hand.

## Experiment

This is the unpruned-vs-pruned comparison in section 1 of `decision-tree.ipynb`.

**Hypothesis (stated before running):** an unpruned tree, grown until every leaf is pure, will reach
(near-)100% training accuracy but will generalize *worse* — lower test accuracy — than a
depth-limited (`max_depth=3`) tree, because the unpruned tree has enough capacity to memorize noise
in the training set (classic overfitting from unconstrained, greedy growth).

**Setup:** `load_breast_cancer` (569 samples, 30 features), 70/30 stratified train/test split,
`random_state=42`. Two `DecisionTreeClassifier`s trained on the identical split: one with
`max_depth=None` (grows until pure), one with `max_depth=3`.

**Result:**

| Model | Depth | Leaves | Train acc | Test acc |
|-------|-------|--------|-----------|----------|
| Unpruned | 6 | 16 | 1.0000 | 0.9181 |
| Depth-limited (`max_depth=3`) | 3 | 7 | 0.9799 | 0.9240 |

**Interpretation:** the hypothesis holds, and more strongly than "just" reduced overfitting — the
depth-limited tree has both *lower* train accuracy *and higher* test accuracy than the unpruned
tree. It gave up memorizing four extra percentage points of training data and got a more reliable
model in return. This is the bias-variance tradeoff made concrete: the unpruned tree has near-zero
bias but high variance; capping depth trades a little bias for a large variance reduction, and on
this dataset that trade pays off out-of-sample.

**Limitations:** this is a single train/test split on one dataset with one specific `max_depth`
choice; the gap would need to be confirmed with cross-validation before treating `max_depth=3` as
"the" right depth in general, and the effect size (0.6 points of test accuracy) is modest here
because breast-cancer classes are fairly separable to begin with — on noisier data the gap between
unpruned and pruned trees is typically much larger.

## Failure modes

- **High variance / instability:** small changes in the training data can produce a very different
  tree — a different feature or threshold winning the root split cascades into a completely
  different structure below it. This is the single biggest weakness of decision trees, and it is
  exactly the problem `11-random-forest` exists to fix (averaging many decorrelated trees).
- **Greedy suboptimality:** splitting is chosen greedily one node at a time; a locally-best split can
  lead to a globally worse tree than a split that looks locally suboptimal but sets up better splits
  later. Finding the globally optimal tree is NP-hard, so greedy search is the practical compromise.
- **Overfitting without pruning:** a fully grown tree can perfectly memorize the training data
  (including its noise) — see the Experiment above.
- **Class imbalance bias:** trees can be biased toward the majority class if classes are imbalanced;
  needs `class_weight` balancing.
- **Poor extrapolation for regression:** predictions are bounded by the range of leaf target values
  seen in training — a tree cannot predict outside the range of `y` it was trained on.

## Real-world usage

- **Interpretable, auditable decisions**: credit scoring, medical triage, fraud flags — anywhere a
  human needs to see *why* the model made a call, a shallow tree can be printed and read directly.
- **Base learner for ensembles**: random forests, gradient boosting machines (XGBoost, LightGBM) all
  use decision trees (typically shallow ones for boosting) as the building block being combined.
- **Feature importance / exploratory data analysis**: even when a tree isn't the final deployed
  model, fitting one quickly surfaces which features and thresholds separate the classes.
- **Mixed data with no preprocessing**: trees need no feature scaling and handle numeric and
  categorical features, and non-linear/non-monotonic relationships, without manual transformation.

## Mental model

A decision tree is a nested game of 20 questions that greedily picks, at each step, the single
question (feature + threshold) that most purifies the remaining group — recursively, until each
remaining group is (nearly) a single answer.

## Questions to think about

1. Why does Gini impurity use $p_i^2$ while Entropy uses $p_i \log_2 p_i$ — what does each penalize
   differently, and why do they still usually pick the same splits in practice?
2. If a feature has 1,000 unique continuous values, how many candidate thresholds does the greedy
   split search need to evaluate at a single node, and why does this make deep trees on
   high-cardinality features expensive?
3. Why can't a decision tree represent a smooth diagonal decision boundary (e.g. $y > x$) efficiently
   no matter how deep it grows, even though it can represent an XOR pattern trivially?
4. `min_samples_leaf` and `max_depth` both limit tree complexity — construct a scenario (in words)
   where they produce meaningfully different trees on the same data.
5. The Experiment above found the depth-limited tree beat the unpruned tree on *both* fewer training
   examples memorized *and* higher test accuracy. Is it possible for pruning to help test accuracy
   even when the unpruned tree isn't visibly "worse" on a validation curve? What would that scenario
   look like?
6. Why does a single tree's instability (failure mode above) get fixed by training many trees on
   *bootstrap resamples* specifically, rather than just training the same tree-growing algorithm
   once and calling it done?
