# 01 – Introduction to ML

| Topic | Status |
|-------|--------|
| What is AI, ML, Deep Learning, Data Science | ✅ Complete |
| Differences and overlap | ✅ Complete |
| Types of ML Techniques | ✅ Complete |
| Equation of Line, 3D, Hyperplane | ✅ Complete |
| Instance-based vs Model-based Learning | ✅ Complete |

## What is Artificial Intelligence (AI)?

**Artificial Intelligence (AI)** is the broad field of building systems that can perform tasks that typically require human intelligence. This includes perception, language understanding, reasoning, planning, and decision-making.

AI is an umbrella term that includes:
- **Rule-based systems:** Handcrafted logic and rules (expert systems, search algorithms).
- **Learning-based systems:** Systems that learn from data (machine learning).
- **Hybrid systems:** Combinations of rules and learning.

Important idea: **AI does not require learning.** A chess engine that uses rules and search is still AI. Modern AI often relies on ML, but AI is broader than ML.

## What is Machine Learning (ML)?

**Machine Learning (ML)** is a subset of AI that focuses on systems that learn patterns from data to make predictions or decisions.

Core ingredients in ML:
- **Data:** Examples $x$ (features) and sometimes labels $y$.
- **Model:** A function $f(x; \theta)$ with parameters $\theta$.
- **Loss function:** A measure of error between predictions and targets.
- **Optimization:** Adjusting parameters to minimize loss.

ML emphasizes **generalization**: the model should perform well on unseen data, not just memorize training examples.

Types of ML problems:
- **Supervised learning:** Learn from labeled data.
- **Unsupervised learning:** Discover structure in unlabeled data.
- **Reinforcement learning:** Learn by interacting with an environment.

## What is Deep Learning (DL)?

**Deep Learning (DL)** is a subset of ML that uses **multi-layer neural networks** to learn complex representations.

Key traits:
- **Representation learning:** Learns features automatically from raw data.
- **Hierarchy:** Lower layers learn simple patterns; higher layers learn complex concepts.
- **Scales with data:** Performance often improves with large datasets and compute.

Deep learning is not a different goal than ML; it is a **family of models and training techniques** within ML.

## What is Data Science?

**Data Science** is a discipline focused on extracting insight and value from data. It is broader than ML and includes the full workflow from raw data to decision-making.

Common components:
- **Data collection and cleaning:** Fixing missing values, removing duplicates.
- **Exploratory data analysis (EDA):** Understanding distributions, correlations.
- **Statistical analysis:** Hypothesis testing, confidence intervals.
- **Modeling:** Machine learning, forecasting, simulation.
- **Visualization and communication:** Dashboards, reports, storytelling.
- **Domain knowledge:** Business or scientific context.

Data Science overlaps with ML but is not a subset of AI. It can be purely descriptive and analytical without building autonomous systems.

## Differences Between AI, ML, DL, and Data Science

Think of the relationship as nested circles:
- **AI** is the broadest field: intelligent systems.
- **ML** is a subset of AI: systems that learn from data.
- **DL** is a subset of ML: deep neural networks.
- **Data Science** overlaps with ML but is a separate discipline focused on insights from data.

Comparison table:

| Area | Focus | Typical Output |
|------|-------|----------------|
| AI | Intelligent behavior | Decisions, actions, reasoning |
| ML | Learning patterns | Predictions or classifications |
| DL | Neural representation learning | High-accuracy models on complex data |
| Data Science | Understanding data | Insights, dashboards, experiments |

Examples:
- A rule-based medical expert system is **AI** but not ML.
- A spam classifier trained on labeled emails is **ML** (and AI).
- A CNN that recognizes objects in images is **DL** and **ML**.
- A sales dashboard that reveals trends is **Data Science**.

Why this matters:
- Use **Data Science** when you need insight and explanation.
- Use **ML** when you need predictions and automation.
- Use **DL** when data is complex and large (images, text, audio).
- Use **AI** when the system must reason, plan, or act intelligently.

## Types of ML Techniques

Machine Learning (ML) is a set of methods that learn patterns from data to make predictions or decisions. The most common way to categorize ML techniques is by the kind of supervision available during training, but there are also useful categories based on data structure, the task type, and the learning objective.

### 1) Supervised learning

Supervised learning trains a model on labeled data, where each example has an input $x$ and a target output $y$. The model learns a mapping $f(x) \rightarrow y$.

Common tasks:
- **Regression:** Predict a continuous value (house price, temperature).
- **Classification:** Predict a discrete label (spam vs. not spam, disease vs. healthy).

Typical pipeline:
1. Collect labeled dataset $(x_i, y_i)$.
2. Split into train/validation/test sets.
3. Choose a model and loss function.
4. Train by minimizing loss on training data.
5. Tune hyperparameters on validation data.
6. Evaluate on test data for generalization.

Common algorithms:
- Linear regression, logistic regression
- k-Nearest Neighbors (kNN)
- Decision trees, random forests, gradient boosting
- Support Vector Machines (SVM)
- Neural networks

Key ideas:
- **Bias-variance tradeoff:** Simple models may underfit (high bias), complex models may overfit (high variance).
- **Generalization:** Performance on unseen data matters more than training accuracy.

### 2) Unsupervised learning

Unsupervised learning uses unlabeled data and seeks patterns, structure, or representations.

Common tasks:
- **Clustering:** Group similar data points (customer segmentation).
- **Dimensionality reduction:** Compress data into fewer features (PCA, t-SNE, UMAP).
- **Density estimation:** Model data distribution.
- **Association rule learning:** Find co-occurrence patterns (market basket analysis).

Common algorithms:
- k-Means, hierarchical clustering, DBSCAN
- PCA, ICA, t-SNE, UMAP
- Gaussian Mixture Models (GMM)

Key ideas:
- Unsupervised methods are exploratory and often used for feature engineering.
- Evaluation is harder because there is no ground-truth label.

### 3) Semi-supervised learning

Semi-supervised learning uses a small amount of labeled data and a large amount of unlabeled data. It is common when labeling is expensive.

Approaches:
- Self-training and pseudo-labeling
- Consistency regularization
- Graph-based methods

Use cases:
- Medical imaging (few labeled scans, many unlabeled scans)
- Web content classification

### 4) Self-supervised learning

Self-supervised learning creates labels from the data itself using pretext tasks. It learns representations that can later be fine-tuned for downstream tasks.

Examples:
- Predicting masked words in text (language models)
- Predicting missing patches in images
- Contrastive learning (e.g., SimCLR)

Why it matters:
- Powerful for large datasets without human labels
- Foundation for transfer learning in NLP and vision

### 5) Reinforcement learning (RL)

RL is about learning a policy by interacting with an environment. The agent takes actions and receives rewards; it learns to maximize cumulative reward.

Core components:
- **Agent:** Learner/decision-maker.
- **Environment:** World the agent interacts with.
- **State $s$:** Current situation.
- **Action $a$:** Choice made by the agent.
- **Reward $r$:** Feedback signal.
- **Policy $\pi(a|s)$:** Strategy for choosing actions.

Common algorithms:
- Q-learning, SARSA
- Policy gradients, actor-critic methods

Use cases:
- Game playing (Atari, Go)
- Robotics and control
- Resource allocation

### 6) Other useful categorizations

**Batch vs. Online learning:**
- *Batch:* Train once on a fixed dataset.
- *Online:* Continuously update as new data arrives.

**Parametric vs. Non-parametric:**
- *Parametric:* Fixed number of parameters (linear regression).
- *Non-parametric:* Model complexity grows with data (kNN, decision trees).

**Generative vs. Discriminative:**
- *Generative:* Model $p(x, y)$ or $p(x|y)$ (Naive Bayes, GMM).
- *Discriminative:* Model $p(y|x)$ or decision boundary (logistic regression, SVM).

**Instance-based vs. Model-based:**
- Covered in detail below.

## Equation of a Line, 3D Plane, and Hyperplane

Geometric intuition is central to ML. Many models learn decision boundaries that are lines, planes, or hyperplanes in a feature space.

### 1) Equation of a line (2D)

In 2D, a line can be represented in several forms.

**Slope-intercept form:**
$$y = mx + b$$
Here $m$ is the slope and $b$ is the $y$-intercept.

**Point-slope form:**
$$y - y_1 = m(x - x_1)$$
Useful when you know one point and the slope.

**General form:**
$$ax + by + c = 0$$
This can represent any line (except when both $a$ and $b$ are zero).

**Vector form:**
$$\mathbf{w}^T \mathbf{x} + b = 0$$
Let $\mathbf{x} = [x, y]^T$, then $\mathbf{w} = [a, b]^T$. This form scales to higher dimensions.

Interpretation in ML:
- For binary linear classification, the decision boundary is a line: $\mathbf{w}^T \mathbf{x} + b = 0$.
- The sign of $\mathbf{w}^T \mathbf{x} + b$ determines the class.

### 2) Equation of a plane (3D)

In 3D, a plane is the set of points $\mathbf{x} = [x, y, z]^T$ satisfying:
$$ax + by + cz + d = 0$$

Vector form:
$$\mathbf{w}^T \mathbf{x} + b = 0$$
with $\mathbf{w} = [a, b, c]^T$ and $b = d$.

Interpretation in ML:
- A linear classifier in 3D uses a plane as the decision boundary.
- The vector $\mathbf{w}$ is normal (perpendicular) to the plane.

### 3) Hyperplane (nD)

In $n$ dimensions, the decision boundary becomes a hyperplane:
$$\mathbf{w}^T \mathbf{x} + b = 0$$

Geometric meaning:
- $\mathbf{w}$ is normal to the hyperplane.
- $b$ controls the offset from the origin.
- The hyperplane splits the space into two half-spaces:
	- $\mathbf{w}^T \mathbf{x} + b > 0$
	- $\mathbf{w}^T \mathbf{x} + b < 0$

Distance from a point to the hyperplane:
$$\text{distance} = \frac{|\mathbf{w}^T \mathbf{x} + b|}{\|\mathbf{w}\|}$$
This is important for margin-based methods like SVM.

Connection to linear regression:
- Linear regression predicts a value using $\hat{y} = \mathbf{w}^T \mathbf{x} + b$.
- The model is a hyperplane in feature space; the target $y$ lives on a different axis.

## Instance-based vs. Model-based Learning

This is a practical distinction about how knowledge is stored and used.

### 1) Instance-based learning

The model stores training instances and makes predictions by comparing new points with stored examples.

Characteristics:
- **Lazy learning:** Training is minimal; most work happens at prediction time.
- **Local generalization:** Predictions depend on nearby samples.
- **Memory heavy:** Stores the dataset or a large part of it.

Common methods:
- k-Nearest Neighbors (kNN)
- Locally weighted regression

Example (kNN classification):
1. Store all training data.
2. For a new point, find the $k$ closest training examples.
3. Predict the majority label among those neighbors.

Pros:
- Simple and intuitive.
- Can model complex decision boundaries if enough data is available.

Cons:
- Slow prediction for large datasets.
- Sensitive to feature scaling and noisy data.
- Requires a good distance metric.

### 2) Model-based learning

The model learns parameters from the data, summarizing patterns in a compact form.

Characteristics:
- **Eager learning:** More work happens during training.
- **Global generalization:** The model captures overall structure.
- **Efficient prediction:** Once trained, predictions are fast.

Common methods:
- Linear regression, logistic regression
- Decision trees, random forests
- Neural networks

Example (linear regression):
1. Learn parameters $\mathbf{w}, b$ by minimizing a loss (e.g., MSE).
2. Predict with $\hat{y} = \mathbf{w}^T \mathbf{x} + b$.

Pros:
- Compact representation of knowledge.
- Fast inference and deployment friendly.
- Often more robust with proper regularization.

Cons:
- Can underfit if model is too simple.
- Training can be expensive for large models.

### 3) How to choose

Use instance-based learning when:
- You need a very flexible model.
- Training data changes frequently.
- You can afford higher inference time.

Use model-based learning when:
- You need fast predictions at scale.
- You want a compact model that generalizes well.
- You have sufficient data to train a stable model.

In practice, modern ML systems often combine both ideas (e.g., a model-based neural network with a retrieval component for instance-based reasoning).
