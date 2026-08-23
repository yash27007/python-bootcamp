# 01 – Introduction to ML

## Problem

Suppose you want a program that decides whether an email is spam. You could sit down and write rules by hand: "if it contains 'free money', flag it; if the sender is unknown and there are three exclamation marks, flag it; ..." This works for a while, but spam changes constantly, legitimate email varies enormously, and every new pattern requires you to go back and edit the rules. The real problem machine learning exists to solve is this: **how do we get a machine to produce its own rule from examples of correct behavior, instead of a human writing that rule by hand?**

Formally: given a set of examples $(x_i, y_i)$ — inputs $x_i$ and the outputs $y_i$ we want for them — we want an algorithm that searches for a function $f$ such that $f(x_i) \approx y_i$ for the examples we have, **and** $f$ continues to work on inputs we have not seen yet. That second requirement — generalization — is what separates machine learning from simply memorizing a lookup table.

## Intuition

Say you're given ten (size in sq. ft., price) pairs for houses in one neighborhood and asked to predict the price of an eleventh house of a new size. A human could plot the ten points, notice they roughly line up, and read the price off a line drawn through them. Machine learning automates exactly that: it doesn't store "house #7 costs $310,000," it finds the line (or curve, or boundary) that the *pattern* in the data implies, so it can answer for a house it has never seen.

Two very different ways to do this "finding":

- **Look at the neighbors.** To predict the eleventh house's price, find the 3 houses in your data most similar in size, and average their prices. You never wrote down an equation — you just compare the new point to old points every time. This is **instance-based learning**.
- **Fit a global rule.** Find the single line $\hat{y} = wx + b$ that best fits all ten houses, then plug the eleventh house's size into that line. You did work once, up front, and now prediction is just arithmetic. This is **model-based learning**.

Both are "learning from data" in the sense defined above; they differ in *where* the pattern lives — in the stored examples themselves, or in a compact set of learned parameters.

## Why simpler approaches fail

The obvious non-learning alternative is a **fixed lookup table or hardcoded if-else chain**: enumerate every input you can think of and hardcode the output. Two things break this immediately:

1. **It cannot generalize to unseen inputs.** A lookup table only has an answer for exact inputs it has seen before. A house of size 1,743 sq. ft. that was never in your table gets no prediction at all (or an arbitrary default). Real-world inputs are continuous or combinatorially vast — you cannot enumerate them.
2. **It doesn't scale with complexity.** As the number of relevant features grows (size, location, age, number of rooms, ...), the number of hand-written rules needed to cover every combination explodes. A human cannot maintain that.

What's missing is a way to **interpolate and extrapolate** from the examples you do have to the examples you don't — i.e., a notion of *similarity* (instance-based) or a *parametric functional form* (model-based) that lets the same procedure answer for a point never seen during construction. That is precisely what makes something "learning" rather than "storage."

## Mathematical foundation

Model-based learning needs a mathematical object to represent a "rule." The simplest and most common such object in ML is a **hyperplane** — a linear boundary or surface in feature space.

### Line (2D)

A line separating two regions of the plane can be written in vector form:
$$\mathbf{w}^T \mathbf{x} + b = 0$$
where $\mathbf{x} = [x_1, x_2]^T$ is a point, $\mathbf{w} = [w_1, w_2]^T$ is a **weight vector** (the coefficients), and $b$ is a **bias/offset** term. Expanded, this is $w_1 x_1 + w_2 x_2 + b = 0$ — the same object as $ax + by + c = 0$ from algebra, just renamed for ML notation.

**Why this form is a "rule":** the vector $\mathbf{w}$ is normal (perpendicular) to the line. For a point not on the line, the sign of $\mathbf{w}^T\mathbf{x} + b$ tells you which side of the line it's on:
$$\mathbf{w}^T \mathbf{x} + b > 0 \implies \text{one class}, \qquad \mathbf{w}^T \mathbf{x} + b < 0 \implies \text{the other class}$$
This *is* a classification rule, expressed as two numbers ($\mathbf{w}, b$) instead of a table of examples. The distance from any point to the line is
$$\text{distance} = \frac{|\mathbf{w}^T \mathbf{x} + b|}{\|\mathbf{w}\|}$$
which becomes important later for margin-based methods like SVMs.

### Plane (3D) and hyperplane ($n$D)

The same equation generalizes without changing form. In 3D, $\mathbf{w}^T\mathbf{x} + b = 0$ with $\mathbf{w} = [w_1, w_2, w_3]^T$ defines a plane; in $n$ dimensions it defines a **hyperplane** that splits the space into two half-spaces, $\mathbf{w}^T\mathbf{x} + b > 0$ and $\mathbf{w}^T\mathbf{x} + b < 0$. This is why "linear classifier" and "hyperplane" are used almost interchangeably: fitting a linear classifier *means* finding the $\mathbf{w}, b$ that best separates the classes.

Linear regression uses the identical algebraic object, but reads it differently: instead of a boundary, $\hat{y} = \mathbf{w}^T\mathbf{x} + b$ is treated as a *predicted value* on a separate output axis, not a separator of two regions.

**The key conceptual point:** a "rule" learned from data, in the model-based sense, is nothing more than a small number of parameters ($\mathbf{w}, b$) that define a geometric object (a boundary or a value-generating surface) in feature space. Learning means *searching for the parameters*, not searching for the rule's syntax the way a human writing if-else statements would.

## Algorithm

**Model-based learning (general recipe):**
1. Choose a parametric family of functions $f(x; \theta)$ (e.g. a hyperplane, parameterized by $\mathbf{w}, b$).
2. Choose a loss function measuring how wrong $f(x_i; \theta)$ is versus $y_i$, summed/averaged over the training data.
3. Search over $\theta$ to minimize the loss (e.g. gradient descent, closed-form solution).
4. At prediction time, evaluate $f(x_{\text{new}}; \theta)$ — no reference to the original data needed.

**Instance-based learning (general recipe, e.g. kNN):**
1. Store the entire training set.
2. At prediction time, given a new point $x_{\text{new}}$, compute its distance to every stored point.
3. Take the $k$ nearest stored points.
4. Predict by majority vote (classification) or average (regression) of their labels.

Note where the "work" happens: model-based does it once, up front, at training time; instance-based defers all of it to prediction time, every time.

## From-scratch implementation

The from-scratch step here is geometric rather than algorithmic: `01-introduction/instance-vs-model-based-and-geometry.ipynb` plots the hyperplane equation directly with NumPy — no library "fits" anything. Section 1 picks $\mathbf{w} = [2, -3]$, $b = 6$, solves $2x - 3y + 6 = 0$ for $y$ across a grid of $x$ values, and plots the resulting line together with its normal vector $\mathbf{w}$, making the abstract equation $\mathbf{w}^T\mathbf{x} + b = 0$ visible as an actual boundary. Section 2 does the same in 3D: $\mathbf{w} = [1, 1, 2]$, $b = -4$ defines a plane, solved for $z$ and rendered as a surface. This is the "from-scratch" step for this topic because it builds the geometric object the rest of ML relies on using nothing but the raw equation and NumPy arithmetic — no `sklearn` fitting involved yet.

## Practical implementation

Section 3 of the same notebook maps the instance-based vs. model-based distinction onto real library implementations, fit on `load_iris` restricted to 2 features so decision boundaries are visualizable:

- **Instance-based → `KNeighborsClassifier(n_neighbors=5)`.** No parameters are learned; `.fit()` just stores the data. Prediction re-scans the stored points every call — this is the library version of "look at the neighbors" from the Intuition section.
- **Model-based → `LogisticRegression()`.** `.fit()` solves for a compact $\mathbf{w}, b$ per class boundary. Prediction is a single matrix multiply against the new point — the library version of "fit a global rule," using exactly the hyperplane equation derived above.

Both are then visualized with `DecisionBoundaryDisplay.from_estimator(...)`, which colors the predicted class over a mesh of the 2D feature space — turning the abstract "rule" (stored data vs. weight vector) into a picture of a boundary, directly connecting back to the Mathematical foundation section.

## Experiment

**Hypothesis (stated before running):** because kNN's boundary is built from local majority votes among stored points, it should look jagged and follow the local density of training points; because logistic regression's boundary is a fixed linear equation per class, it should look smooth and (piecewise) straight — regardless of local density.

**Setup:** both models trained on the same 2-feature, 3-class `iris` subset; decision boundaries rendered side by side with `DecisionBoundaryDisplay`.

**Actual result** (from `instance-vs-model-based-and-geometry.ipynb`): the KNN plot shows visibly ragged, non-linear region boundaries that bend around clusters of points, while the logistic regression plot shows straight-line boundaries between each pair of classes. Training accuracy was reported for both (KNN ≈ 0.8+, LogisticRegression slightly lower on this 2-feature subset) — the point of the experiment is boundary *shape*, not accuracy.

**Interpretation:** this confirms the hypothesis and makes concrete what "instance-based" and "model-based" mean geometrically — one is a function of local density, the other a function of a fixed small parameter set.

**Limitations:** iris with only 2 (of 4) features is a toy, low-dimensional case chosen for visualizability; the comparison used training accuracy, not held-out generalization, so it says nothing yet about which approach generalizes better — that requires the cross-validation and bias-variance material in later topics.

## Failure modes

- **Instance-based learning doesn't scale.** Every prediction requires comparing against (up to) the entire training set, so both memory and prediction-time cost grow with data size. In production systems with millions of stored points, naive kNN prediction can become too slow without approximate nearest-neighbor structures.
- **Instance-based learning is sensitive to feature scaling and noise.** Because predictions come from raw distances, a feature on a larger numeric scale dominates the distance calculation unless features are normalized; noisy or irrelevant features corrupt every neighbor lookup.
- **Model-based learning can't fit non-linear boundaries without feature engineering.** A plain linear model like logistic regression can only draw straight-line (hyperplane) boundaries. If the true decision boundary is curved, the model will systematically misclassify points near the curve no matter how much data it gets — the fix is either a non-linear model or manually engineered non-linear features (e.g. polynomial terms) fed into the same linear form.

## Real-world usage

- Model-based methods (linear/logistic regression, neural networks, gradient-boosted trees) dominate production systems that need fast, low-latency predictions at scale — the training cost is paid once, offline.
- Instance-based methods appear where local structure matters more than a global rule, or where the "model" needs to update instantly as new data arrives without retraining (recommendation systems' nearest-neighbor lookups, some anomaly detection, retrieval-augmented systems that combine a learned embedding model with an instance-based nearest-neighbor search over a vector database).
- Understanding the line/plane/hyperplane equation is a prerequisite for essentially every linear model, SVMs, and the first layer of intuition for neural network weights — it is the single most reused piece of geometry in this curriculum.

## Mental model

Instance-based learning **memorizes the neighborhood** and answers by comparison every time; model-based learning **fits a global rule once** and answers by evaluating that rule — the same distinction as "look it up" versus "calculate it."

## Questions to think about

1. If you had a training set of 50 million examples and needed predictions in under 1ms, would you reach for instance-based or model-based learning first, and why?
2. Suppose you scale one feature of your dataset by 1000x (e.g. converting meters to millimeters) without touching the others. Which of the two approaches from this topic is affected, and why does the hyperplane equation make the model-based one's sensitivity different from the instance-based one's?
3. A linear model draws a straight decision boundary. Describe two different ways you could get a linear model to represent a *curved* boundary without switching to a fundamentally different algorithm class.
4. Why does "generalization" require more than fitting the training examples exactly — construct a small example where a rule that perfectly matches every training point still gives a bad answer on a new point.
