# Phase 1: Machine Learning First-Principles Retrofit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrofit every topic in `05-machine-learning/` (17 existing + 1 new) into the first-principles `notes.md` template defined in `AGENTS.md`, with from-scratch NumPy implementations, measurable experiments, failure modes, and reasoning questions — replacing "concept explainer" style notes with problem-first, derivation-first teaching material.

**Architecture:** Each topic gets a rewritten `notes.md` following the 12-section template (Problem, Intuition, Why-simpler-fails, Math, Algorithm, From-scratch, Practical, Experiment, Failure-modes, Real-world, Mental-model, Questions) and, where one doesn't already exist demonstrating it, a from-scratch NumPy implementation cell/notebook proving the concept before the practical (sklearn) implementation. Existing notebooks and good derivations are preserved and extended, not deleted. Work is organized into 7 tasks grouped by conceptual family (foundations, regression, margin/probabilistic classifiers, trees, boosting, unsupervised) so each task tells one coherent story; tasks are dispatched **sequentially** per subagent-driven-development's rule even though the families are file-independent.

**Tech Stack:** Python 3.13, NumPy/SciPy (from-scratch), scikit-learn (practical), pandas/matplotlib/seaborn, `.venv` (uv-managed), `.venv/bin/jupyter`.

**Spec:** `docs/superpowers/specs/2026-08-23-first-principles-curriculum-design.md` and `AGENTS.md` (binding template + quality bar) — read both before starting any task.

## Global Constraints

- Repo root: `/home/yashwanth-aravind/ml-course/python-bootcamp`. Python env: `.venv` (uv-managed) — `.venv/bin/python` / `.venv/bin/jupyter`.
- Every `notes.md` follows the 12-section template in `AGENTS.md` exactly, in order: Problem, Intuition, Why simpler approaches fail, Mathematical foundation, Algorithm, From-scratch implementation, Practical implementation, Experiment, Failure modes, Real-world usage, Mental model, Questions to think about. Problem / Why-simpler-fails / Mathematical foundation / Mental model / Questions are never skipped. Use LaTeX (`$...$` / `$$...$$`) for all math, every symbol explained in prose.
- Preserve correct existing derivations — read the current `notes.md`/`README.md` first and restructure/extend into the template rather than deleting good material. Where a topic has no `notes.md` yet, write one from scratch.
- From-scratch implementation: plain Python/NumPy/SciPy only. This is a teaching demonstration proving the concept works (e.g. gradient descent updating weights by hand, or a from-scratch Gini-impurity split search) — not a full reimplementation of a production algorithm. Keep it small (one clearly-scoped function or short script, one toy dataset, a print/plot showing it converges/works). Where a from-scratch demo would add no insight over the existing math derivation (e.g. XGBoost's exact tree-boosting internals), a smaller illustrative piece suffices (e.g. from-scratch weighted-error boosting update for AdaBoost, not a full XGBoost reimplementation) — use judgment, document the choice in the notes.md prose, never skip the section entirely.
- Practical implementation: the existing scikit-learn notebook(s) for that topic, executed end-to-end, explicitly connected in `notes.md` prose to the from-scratch step ("the from-scratch update above is exactly what `SGDRegressor` does internally, at scale").
- Experiment section: a real hypothesis test using the topic's existing notebook or a small addition to it — hypothesis stated before running, actual result compared to expected, limitations named. Reuses existing notebook results where they already constitute a valid experiment (e.g. pruned vs unpruned tree comparison already in `10-decision-tree`) rather than manufacturing a redundant one.
- Every notebook touched or created MUST execute end-to-end with no errors: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace <path>.ipynb`. Non-null `execution_count` on every code cell.
- Datasets: reuse the existing notebook's dataset where one already exists and is a sklearn/keras built-in or already-committed CSV in that topic's folder. No new manual downloads.
- `05-machine-learning/README.md`: status column stays `✅ Complete` for retrofitted topics (already true) — no change needed there. Each topic's own `README.md` gets rewritten to the AGENTS.md orientation format (what you'll learn / why it matters / prerequisites / what you'll build / where it appears in real systems / what's next) if it doesn't already read that way — keep it concise (short stub, not a duplicate of notes.md).
- New topic `bias-variance-tradeoff`: create as `05-machine-learning/05b-bias-variance-tradeoff/` (letter-suffixed to avoid renumbering existing topics 06+, per the spec's additive-only numbering rule) with `README.md` + `notes.md` + one small notebook.
- Commit granularity: one commit per task (matches the family grouping), not one giant commit for the whole phase.
- No unit-test framework applies. "Tests pass" means: every notebook executes cleanly, and every notes.md's 12 sections are present with real content (no "TBD"/placeholder text) covering that topic's actual algorithm.

---

### Task 1: Foundations — `01-introduction`, `05-cross-validation`, new `05b-bias-variance-tradeoff`

**Files:**
- Modify: `05-machine-learning/01-introduction/notes.md` (retrofit into template; existing notebook `instance-vs-model-based-and-geometry.ipynb` stays as the practical step, already covers instance-based vs model-based)
- Modify: `05-machine-learning/05-cross-validation/notes.md` (retrofit; existing `cross-validation.ipynb` stays as practical + experiment step)
- Create: `05-machine-learning/05b-bias-variance-tradeoff/README.md`, `notes.md`, `bias-variance-tradeoff.ipynb`

**Interfaces:** None (leaf content task).

**Content requirements:**

- **`01-introduction`**: Problem = how do we get a machine to produce a rule from data instead of a human writing one by hand. Why-simpler-fails = a fixed lookup table / hardcoded if-else can't generalize to unseen inputs. Math = the hyperplane equation (already in the notebook) explained as "a rule is a boundary in feature space." From-scratch = the notebook's manual decision-boundary plotting already qualifies — cite it as the from-scratch step, add a short prose bridge in notes.md. Practical = `KNeighborsClassifier`/`LogisticRegression` from the existing notebook. Experiment = compare instance-based vs model-based decision boundaries on the same data (existing notebook), stated as hypothesis/result. Failure modes = instance-based doesn't scale / model-based can't fit non-linear boundaries without feature engineering. Mental model = one sentence contrasting "memorize the neighborhood" vs "fit a global rule."
- **`05-cross-validation`**: Problem = how do we estimate generalization error without a held-out production environment. Why-simpler-fails = a single train/test split has high variance and can accidentally be lucky/unlucky. Math = the K-Fold estimator as an average of K point estimates, its variance vs a single split's variance. From-scratch = a small NumPy K-Fold index-splitter (given N and K, yield train/test index arrays) — a handful of lines, proves the mechanic. Practical = the existing notebook's `KFold`/`StratifiedKFold`/`LeaveOneOut`/`GridSearchCV`/`RandomizedSearchCV` work. Experiment = the existing notebook's per-fold score printout already is one — frame it as hypothesis ("mean±std should be more stable than any single split") vs actual. Failure modes = LOOCV's high variance on unstable models (already touched on from the final review's parked LOOCV-std finding — resolve that here: add the one-sentence caveat that per-fold LOOCV std reflects per-sample noise, not model variance, and isn't directly comparable to K-Fold std). Mental model = "cross-validation trades compute for a lower-variance estimate of a number you can't otherwise observe."
- **`05b-bias-variance-tradeoff`** (new): Problem = why does a model that fits training data perfectly sometimes perform worse on new data than one that fits training data worse. Why-simpler-fails = "just minimize training error" as a strategy, and why it silently produces overfitting. Math = the bias-variance decomposition of expected test error, $\mathbb{E}[(y-\hat f(x))^2] = \text{Bias}^2 + \text{Variance} + \sigma^2$ — derive it, explain each term. From-scratch: NumPy experiment fitting polynomials of increasing degree to noisy synthetic data (a fixed-seed sine or cubic + noise), computing empirical bias²/variance/test-error across many resampled training sets, plotting the classic U-shaped test-error curve against model complexity. Practical = connect explicitly to `04-regularization` (upcoming task) and `05-cross-validation` (this task) as the two main tools that manage this tradeoff. Experiment = the polynomial-degree sweep above, hypothesis stated before running (test error should be U-shaped; train error should be monotonically decreasing). Failure modes = high-variance regime (overfitting) vs high-bias regime (underfitting), and why more data helps variance but not bias. Real-world = model selection, why "just add more parameters" isn't free. Mental model = "bias is being wrong on average; variance is being unstable across different training sets — you cannot always reduce both at once." README follows the standard orientation format.

- [ ] **Step 1:** Read each topic's current `notes.md`/`README.md` and the family's shared narrative above.
- [ ] **Step 2:** Retrofit `01-introduction/notes.md` and `05-cross-validation/notes.md` into the 12-section template per the content requirements. Write `05b-bias-variance-tradeoff/README.md` + `notes.md`.
- [ ] **Step 3:** Write `05b-bias-variance-tradeoff/bias-variance-tradeoff.ipynb` (the polynomial-degree bias/variance sweep experiment described above), with markdown narration cells.
- [ ] **Step 4:** Execute `05b-bias-variance-tradeoff/bias-variance-tradeoff.ipynb` with `jupyter nbconvert --to notebook --execute --inplace`; confirm exit 0, non-null `execution_count` on every code cell. (`01-introduction` and `05-cross-validation`'s existing notebooks are unchanged — no re-execution needed unless you added cells to them, in which case execute those too.)
- [ ] **Step 5:** `git add` the 3 topic folders, commit: `git commit -m "Phase 1 Task 1: first-principles retrofit — introduction, cross-validation, bias-variance tradeoff"`.

---

### Task 2: Regression family — `02-linear-regression`, `03-polynomial-regression`, `04-regularization`

**Files:**
- Modify: `05-machine-learning/02-linear-regression/notes.md` (retrofit; existing notes.md has substantial derivation already — restructure into template, don't discard the math)
- Create: `05-machine-learning/03-polynomial-regression/notes.md` (none exists yet)
- Modify: `05-machine-learning/04-regularization/notes.md` (retrofit)

**Interfaces:** Builds narratively on Task 1's `05b-bias-variance-tradeoff` (regularization is a bias-variance tool) — reference it by relative link, don't duplicate its derivation.

**Content requirements:**

- **`02-linear-regression`**: Problem = predict a continuous quantity as a linear combination of features. Why-simpler-fails = "eyeball a line" / simple averaging doesn't generalize or handle multiple features. Math = least-squares objective, why squared error (differentiable, penalizes large errors more, connects to Gaussian-noise MLE — derive this connection explicitly since the existing notes.md likely states least-squares without justifying the choice), normal equation derivation $\hat\beta = (X^TX)^{-1}X^Ty$, geometric interpretation (projection onto column space of X), gradient descent as the iterative alternative. From-scratch = manual gradient descent in NumPy for simple linear regression (the existing notebook may already have this — check `simple-linear-regression.ipynb` and `multiple-linear-regression.ipynb`; if present, cite as the from-scratch step; if only sklearn `LinearRegression` is used, add a short from-scratch gradient-descent cell). Practical = `LinearRegression`/`SGDRegressor` from the existing notebooks. Experiment = compare normal-equation solution vs gradient-descent solution converging to the same coefficients (hypothesis: they should match within tolerance). Failure modes = multicollinearity, non-linearity, heteroscedasticity, extrapolation. Mental model = "linear regression finds the line minimizing total squared vertical distance, which is exactly the maximum-likelihood answer if you assume Gaussian noise."
- **`03-polynomial-regression`** (no existing notes.md): Problem = linear regression can't fit curved relationships. Why-simpler-fails = forcing a straight line through curved data underfits (bias). Math = polynomial features are still a *linear* model in the transformed feature space ($x \to [x, x^2, x^3, ...]$) — derive why this is still "linear regression," just on engineered features. From-scratch = small NumPy example building polynomial features manually (`np.vander` or manual powers) and fitting via the normal equation from Task 2's linear-regression derivation. Practical = the existing `polynomial-regression.ipynb`'s `PolynomialFeatures` + `LinearRegression` pipeline. Experiment = degree sweep showing underfit → good fit → overfit (this directly reuses `05b-bias-variance-tradeoff`'s experiment shape — link to it rather than re-deriving bias/variance theory, but do run the concrete sweep here on this topic's actual dataset). Failure modes = high-degree polynomials overfit and extrapolate wildly outside the training range. Real-world = feature engineering as a form of "linear model, richer features." Mental model = "polynomial regression is linear regression wearing a disguise — the model is still linear in its parameters."
- **`04-regularization`**: Problem = how do we keep a flexible model (e.g. high-degree polynomial, many features) from overfitting without manually picking model complexity. Why-simpler-fails = cross-validating over every possible feature subset is combinatorially expensive; early stopping alone is fragile. Math = Ridge ($L_2$) and Lasso ($L_1$) penalty terms added to the loss, derive why $L_1$ induces sparsity (geometric argument: diamond-shaped constraint region vs circular) while $L_2$ shrinks smoothly, bias-variance framing (regularization trades a small bias increase for a larger variance decrease — link to Task 1's `05b-bias-variance-tradeoff`). From-scratch = NumPy ridge regression via the closed-form solution $\hat\beta = (X^TX + \lambda I)^{-1}X^Ty$, showing coefficients shrinking as $\lambda$ increases. Practical = existing `model-training.ipynb`/`algerian-forest.ipynb` (`Ridge`/`Lasso`/`ElasticNet`). Experiment = coefficient-path plot (coefficients vs $\lambda$) — hypothesis: coefficients shrink toward zero monotonically, Lasso hits exact zeros before Ridge does. Failure modes = wrong $\lambda$ under/over-regularizes; Lasso's instability with correlated features (motivates ElasticNet). Mental model = "regularization is a budget on how much the model is allowed to trust the training data."

- [ ] **Step 1:** Read the 3 topics' current `notes.md`/`README.md`/notebooks.
- [ ] **Step 2:** Retrofit `02-linear-regression/notes.md` and `04-regularization/notes.md`; write `03-polynomial-regression/notes.md` from scratch. Follow the 12-section template.
- [ ] **Step 3:** For any topic whose notebook lacks a from-scratch cell per the content requirements above, add one (markdown + code cells) to that topic's existing notebook rather than creating a new file, unless the notebook is already at natural capacity — in that case add a new small notebook `from-scratch-<topic>.ipynb` in the same folder.
- [ ] **Step 4:** Execute every notebook touched (existing + any new) with `jupyter nbconvert --to notebook --execute --inplace`; confirm exit 0, non-null `execution_count` on every code cell.
- [ ] **Step 5:** `git add` the 3 topic folders, commit: `git commit -m "Phase 1 Task 2: first-principles retrofit — linear regression, polynomial regression, regularization"`.

---

### Task 3: Margin & probabilistic classifiers — `06-logistic-regression`, `07-svm`

**Files:**
- Modify: `05-machine-learning/06-logistic-regression/notes.md` (retrofit)
- Create: `05-machine-learning/07-svm/notes.md` (none exists yet)

**Content requirements:**

- **`06-logistic-regression`**: Problem = predict a probability/class, not a continuous number — linear regression's unbounded output doesn't fit $[0,1]$. Why-simpler-fails = thresholding a linear regression's raw output at 0.5 produces nonsensical "probabilities" outside $[0,1]$ and is sensitive to outliers in a way a probability model shouldn't be. Math = the sigmoid function and its derivation from log-odds ($\text{logit}(p) = \ln\frac{p}{1-p} = w^Tx + b$), binary cross-entropy loss derived from maximum likelihood on a Bernoulli outcome, gradient of the loss w.r.t. weights. From-scratch = NumPy gradient descent fitting logistic regression on a small 2D toy dataset, plotting the decision boundary converging over iterations. Practical = existing `logistic-regression.ipynb`/`multiclass-classification.ipynb` (`LogisticRegression`, softmax for multiclass). Experiment = compare the from-scratch boundary to sklearn's `LogisticRegression` boundary on the same data — hypothesis: they should coincide within numerical tolerance. Failure modes = perfectly separable data causes coefficients to diverge to infinity without regularization; class imbalance biases the decision threshold. Mental model = "logistic regression is linear regression on the log-odds of the outcome."
- **`07-svm`** (no existing notes.md — 3 existing notebooks: Basic-SVC, SVR, Kernels): Problem = among all lines that separate two classes, which one generalizes best. Why-simpler-fails = logistic regression's decision boundary isn't chosen to maximize distance to the nearest points — two boundaries can both perfectly classify training data but generalize very differently. Math = margin maximization objective, derive the hard-margin optimization problem geometrically (distance from a point to a hyperplane), soft margin with slack variables $C$, the kernel trick (why $\phi(x)^T\phi(x')$ can be computed as $K(x,x')$ without explicitly computing $\phi$ — show this for the polynomial or RBF kernel). From-scratch = NumPy/SciPy: solve a small hard-margin SVM as a constrained optimization (e.g. via `scipy.optimize.minimize` on the dual or primal for a tiny 2D linearly-separable toy set) and visualize the margin and support vectors. Practical = the existing 3 notebooks (`SVC`, `SVR`, kernels). Experiment = vary $C$ and show margin width vs number of support vectors vs training accuracy (bias/variance tradeoff again — reference Task 1). Failure modes = poor scaling of features breaks margin geometry (SVMs need standardized features); RBF kernel with wrong $\gamma$ overfits badly; doesn't scale well to very large $N$ (motivates why tree ensembles/linear models are often preferred at scale). Real-world = still used where margins and small-to-medium tabular data with clear geometric separation matter. Mental model = "SVM asks not just 'can I separate these classes' but 'what's the widest street I can drive between them.'"

- [ ] **Step 1:** Read current `06-logistic-regression/notes.md` and `07-svm`'s 3 notebooks + `README.md`.
- [ ] **Step 2:** Retrofit `06-logistic-regression/notes.md`; write `07-svm/notes.md` from scratch, per content requirements.
- [ ] **Step 3:** Add a from-scratch gradient-descent cell to the logistic-regression notebook if not already present. Add a new small notebook `05-machine-learning/07-svm/svm-from-scratch.ipynb` implementing the hard-margin toy-data demo described above (the 3 existing SVM notebooks remain the practical step, unmodified unless they need re-execution).
- [ ] **Step 4:** Execute every notebook touched/created; confirm exit 0, non-null `execution_count` on every code cell.
- [ ] **Step 5:** `git add` the 2 topic folders, commit: `git commit -m "Phase 1 Task 3: first-principles retrofit — logistic regression, SVM"`.

---

### Task 4: Trees — `10-decision-tree`, `11-random-forest`

**Files:**
- Modify: `05-machine-learning/10-decision-tree/notes.md` (retrofit — already has decent math from the prior plan)
- Modify: `05-machine-learning/11-random-forest/notes.md` (retrofit)

**Content requirements:**

- **`10-decision-tree`**: existing notes.md already covers splitting criteria and pruning (from the earlier plan) — restructure into the 12-section template rather than rewriting the math from scratch. Add: Problem framed explicitly ("how do we make a sequence of yes/no decisions from data instead of hand-coding if-else rules"), Why-simpler-fails ("a single linear boundary — logistic regression/SVM — can't represent an XOR-like rule; decision trees can"), From-scratch (a small NumPy function computing Gini impurity for a split and searching over thresholds on one feature of a toy dataset — the existing `plot_tree` unpruned-vs-pruned comparison already IS the Experiment step, cite it), Failure modes (already touched — high variance/instability to small data changes, which directly motivates `11-random-forest` next), Mental model, Questions.
- **`11-random-forest`**: Problem = a single decision tree is unstable (high variance) — small changes in training data produce very different trees. Why-simpler-fails = training multiple trees on the exact same data just reproduces the same tree. Math = bootstrap aggregating (bagging) — why averaging many high-variance, low-bias models reduces variance without increasing bias (derive the variance-reduction argument for averaging $B$ i.i.d.-ish estimators, then explain why bootstrap resampling + random feature subsets decorrelates the trees so this argument actually holds). From-scratch = NumPy bagging demonstration: fit $B$ decision trees (can use sklearn's `DecisionTreeClassifier` as the base learner, since the point being demonstrated is the *ensembling* mechanism, not re-deriving tree-splitting) on bootstrap resamples of a toy dataset, show variance of predictions shrinking as $B$ grows. Practical = existing `random-forest-regression.ipynb`/`random-forest-classification.ipynb`. Experiment = out-of-bag error vs number of trees curve (or bootstrap-variance-vs-B from the from-scratch step). Failure modes = loses interpretability vs a single tree; correlated trees (too few features sampled per split, or too little data diversity) don't reduce variance as much as expected; can still overfit with very deep trees despite averaging. Mental model = "a random forest is many weak, overfit, decorrelated opinions averaged into one stable one."

- [ ] **Step 1:** Read current `notes.md`/notebooks for both topics.
- [ ] **Step 2:** Retrofit both `notes.md` files per content requirements.
- [ ] **Step 3:** Add a from-scratch Gini-search cell to the decision-tree notebook (or a small new notebook if it doesn't fit cleanly) and a from-scratch bagging-variance-reduction cell/notebook for random forest.
- [ ] **Step 4:** Execute every notebook touched/created; confirm exit 0, non-null `execution_count` on every code cell.
- [ ] **Step 5:** `git add` the 2 topic folders, commit: `git commit -m "Phase 1 Task 4: first-principles retrofit — decision tree, random forest"`.

---

### Task 5: Probabilistic & instance-based — `08-naive-bayes`, `09-knn`

**Files:**
- Modify: `05-machine-learning/08-naive-bayes/notes.md` (retrofit — already has Bayes' theorem derivation from the prior plan)
- Modify: `05-machine-learning/09-knn/notes.md` (retrofit — already has distance metrics from the prior plan)

**Content requirements:**

- **`08-naive-bayes`**: restructure existing derivation into the template. Add: Problem ("classify using probability when we only have limited data to estimate a full joint distribution"), Why-simpler-fails ("estimating the full joint $P(x_1,...,x_n \mid y)$ needs exponentially much data as features grow — the naive conditional-independence assumption is what makes this tractable at all; state explicitly this assumption is usually false and explain why the classifier still works well in practice despite that"), the existing GaussianNB/MultinomialNB notebook work is the practical + experiment step (cite it), Failure modes (the independence assumption breaking down with strongly correlated features; zero-frequency problem motivating Laplace smoothing — derive the smoothing formula if not already present), Mental model, Questions.
- **`09-knn`**: restructure existing derivation into the template. Add: Problem ("classify/predict using only the data itself, with no fitted parameters"), Why-simpler-fails ("a global linear model can't represent local structure with wildly different behavior in different regions of feature space"), From-scratch (a small NumPy k-NN: compute pairwise Euclidean distances to all training points, take the k nearest, majority-vote — a handful of lines on a toy 2D dataset), the existing K-sweep notebook is the Experiment step (cite it, it already tests the bias-variance effect of K), Failure modes (curse of dimensionality — distances become less meaningful in high dimensions, derive/explain briefly; sensitive to feature scaling; slow at prediction time with large N), Mental model, Questions.

- [ ] **Step 1:** Read current `notes.md`/notebooks for both topics.
- [ ] **Step 2:** Retrofit both `notes.md` files per content requirements.
- [ ] **Step 3:** Add a from-scratch NumPy k-NN cell/small notebook for `09-knn`. `08-naive-bayes` needs no new from-scratch code beyond what Laplace-smoothing derivation requires in notes.md prose (the existing GaussianNB/MultinomialNB notebook already demonstrates the mechanism clearly enough that a from-scratch reimplementation adds little — document this choice in notes.md per the Global Constraints judgment call).
- [ ] **Step 4:** Execute every notebook touched/created; confirm exit 0, non-null `execution_count` on every code cell.
- [ ] **Step 5:** `git add` the 2 topic folders, commit: `git commit -m "Phase 1 Task 5: first-principles retrofit — naive bayes, KNN"`.

---

### Task 6: Boosting family — `12-adaboost`, `13-gradient-boosting`, `14-xgboost`

**Files:**
- Modify: `05-machine-learning/12-adaboost/notes.md` (retrofit)
- Modify: `05-machine-learning/13-gradient-boosting/notes.md` (retrofit)
- Modify: `05-machine-learning/14-xgboost/notes.md` (retrofit)

**Interfaces:** Builds narratively on Task 4's `11-random-forest` (bagging) — boosting is presented as the other family of ensembling, explicitly contrasted: bagging reduces variance by averaging independent models, boosting reduces bias by sequentially correcting errors.

**Content requirements:**

- **`12-adaboost`**: Problem = combine many *weak* learners (barely better than random) into one strong learner, sequentially. Why-simpler-fails = bagging's independent-models approach doesn't help if every individual model is systematically biased (underfit) rather than high-variance — averaging biased models just gives a biased average. Math = derive the AdaBoost weight-update rule: sample weights increase for misclassified points, each weak learner's vote weight $\alpha_t = \frac12\ln\frac{1-\epsilon_t}{\epsilon_t}$ derived from minimizing exponential loss. From-scratch = NumPy: a tiny AdaBoost loop using sklearn `DecisionTreeClassifier(max_depth=1)` "stumps" as the weak learner, manually updating and renormalizing sample weights each round, on a toy 2D dataset — show training error dropping round by round. Practical = existing `adaboost-regression.ipynb`/`adaboost-classification.ipynb`. Experiment = training error vs number of boosting rounds (hypothesis: monotonically decreasing, eventually may overfit). Failure modes = sensitive to noisy labels/outliers (they get up-weighted repeatedly); can overfit with too many rounds. Mental model = "AdaBoost is a class focusing more and more on the questions it keeps getting wrong."
- **`13-gradient-boosting`**: Problem = generalize AdaBoost's "focus on mistakes" idea to arbitrary differentiable loss functions, not just exponential loss on classification. Why-simpler-fails = AdaBoost's exponential-loss weight-update trick doesn't generalize cleanly to regression or other loss functions. Math = derive gradient boosting as functional gradient descent — each new weak learner is fit to the *negative gradient* (pseudo-residuals) of the loss w.r.t. the current ensemble's predictions; show this reduces to fitting residuals directly for squared-error loss. From-scratch = NumPy: fit successive shallow `DecisionTreeRegressor`s to residuals on a toy 1D regression problem, show the ensemble's predictions converging to the target function round by round (plot). Practical = existing `GradientBoost-Classification.ipynb`/`Gradientboost-Regression.ipynb`. Experiment = training loss vs number of boosting rounds, and effect of learning rate (shrinkage) on convergence speed vs overfitting. Failure modes = slow to train (sequential, can't parallelize across rounds), sensitive to learning rate/number of estimators, easy to overfit without early stopping. Mental model = "gradient boosting fits a sequence of models to the errors of the previous ensemble, using the gradient of the loss to know which direction 'fixing the error' means for any loss function, not just squared error."
- **`14-xgboost`**: Problem = plain gradient boosting is slow and prone to overfitting at scale — production systems need speed, regularization, and to handle missing data automatically. Why-simpler-fails = naive gradient boosting recomputes full residuals with no built-in regularization on tree complexity and no principled handling of missing values or parallelism. Math = XGBoost's regularized objective (adds an explicit tree-complexity penalty $\Omega(f) = \gamma T + \frac12\lambda\|w\|^2$ to the loss), second-order (Newton) approximation of the loss using both gradient and Hessian for faster/more accurate split-finding — derive why using the Hessian gives a better local approximation than gradient alone. From-scratch = not a full reimplementation (correctly out of scope per Global Constraints) — instead, a small NumPy illustration comparing a first-order (gradient-only) approximate step vs a second-order (gradient+Hessian, Newton) approximate step on a simple 1D loss curve, showing the second-order step lands closer to the true minimum in fewer steps. Practical = existing `XgboostBoost-Classification-Implementation.ipynb`/`Xgboost-Regression-Implementation.ipynb`. Experiment = compare `GradientBoostingClassifier` (Task's plain gradient boosting) vs `XGBClassifier` on the same dataset/split for training time and accuracy (hypothesis: XGBoost trains faster and generalizes at least as well due to built-in regularization). Failure modes = many hyperparameters to tune, can still overfit with too many rounds/too little regularization, less interpretable than a single tree. Real-world = the default choice for tabular data competitions and many production tabular pipelines. Mental model = "XGBoost is gradient boosting with a built-in complexity budget and a smarter (second-order) way of deciding each step."

- [ ] **Step 1:** Read current `notes.md`/notebooks for all 3 topics.
- [ ] **Step 2:** Retrofit all 3 `notes.md` files per content requirements, in the narrative order above (AdaBoost → gradient boosting → XGBoost), each explicitly building on the previous.
- [ ] **Step 3:** Add the from-scratch AdaBoost-loop notebook/cell, the from-scratch residual-fitting gradient-boosting notebook/cell, and the from-scratch first-order-vs-second-order-step illustration for XGBoost. Add the plain-GB-vs-XGBoost timing/accuracy experiment cell to the xgboost notebook.
- [ ] **Step 4:** Execute every notebook touched/created; confirm exit 0, non-null `execution_count` on every code cell.
- [ ] **Step 5:** `git add` the 3 topic folders, commit: `git commit -m "Phase 1 Task 6: first-principles retrofit — AdaBoost, gradient boosting, XGBoost"`.

---

### Task 7: Unsupervised — `15-unsupervised-learning`, `18-pca`, `16-anomaly-detection`

**Files:**
- Modify: `05-machine-learning/15-unsupervised-learning/notes.md` (retrofit)
- Create: `05-machine-learning/18-pca/notes.md` (none exists yet)
- Modify: `05-machine-learning/16-anomaly-detection/notes.md` (retrofit)

**Content requirements:**

- **`15-unsupervised-learning`**: Problem = find structure in data with no labels at all. Why-simpler-fails = supervised methods have nothing to optimize against without $y$. Math = K-Means as minimizing within-cluster sum of squares (derive the alternating-minimization argument: fixing centroids optimizes assignments, fixing assignments optimizes centroids as the mean), hierarchical clustering's linkage criteria, DBSCAN's density-reachability definition. From-scratch = NumPy Lloyd's-algorithm K-Means (assign-to-nearest-centroid / recompute-centroid loop) on a toy 2D dataset, visualize convergence over iterations. Practical = the existing 4 notebooks (KMeans, Hierarchical, DBSCAN, Silhouette). Experiment = silhouette score vs number of clusters $K$ (the existing Silhouette-Analysis notebook already is this — cite it as the Experiment step, hypothesis: silhouette peaks near the "true" number of clusters in synthetic data with known structure). Failure modes = K-Means assumes spherical, similarly-sized clusters and requires choosing K in advance; sensitive to initialization (motivates K-Means++, mention briefly) and feature scaling. Mental model = "K-Means alternates between 'who belongs to which group' and 'where is each group's center' until neither answer changes."
- **`18-pca`** (no existing notes.md — 2 existing notebooks): Problem = high-dimensional data is hard to visualize, expensive to model, and often has redundant/correlated features. Why-simpler-fails = dropping features by hand loses information arbitrarily; correlated features waste model capacity. Math = derive PCA as the direction of maximum variance — covariance matrix, why its eigenvectors are the principal components and eigenvalues are the variance explained along each (either via the Lagrangian-constrained-optimization argument or the SVD connection — pick one and explain it clearly rather than gesturing at both), projection onto the top-$k$ components. From-scratch = NumPy: compute the covariance matrix of a small toy dataset, `np.linalg.eig` to get eigenvectors/eigenvalues, project data onto the top 2 components, compare visually to `sklearn.decomposition.PCA`'s result on the same data. Practical = existing `PCA-Principal-Component-Analysis.ipynb`/`PCA-2.ipynb`. Experiment = explained-variance-ratio vs number of components (scree plot) — hypothesis: a small number of components should capture most of the variance if features are correlated. Failure modes = PCA is linear (can't capture non-linear manifold structure — briefly mention non-linear alternatives like t-SNE/UMAP exist but are out of scope here); components are not inherently interpretable; sensitive to feature scaling (must standardize first). Mental model = "PCA rotates the coordinate system to point along the directions the data actually varies in, so you can keep only the directions that matter."
- **`16-anomaly-detection`**: restructure existing derivation into the template. Add: Problem ("find the rare, unusual points when you have few or no labeled examples of what 'anomalous' looks like"), Why-simpler-fails ("supervised classification needs labeled anomalies, which are by definition rare and often unseen at training time"), the existing 3 notebooks (Isolation Forest, DBSCAN, LOF) are the practical + experiment step (cite them — if a comparison across the 3 methods on the same dataset doesn't already exist, add a short comparison cell to one of them as the Experiment), Failure modes (contamination-rate assumption sensitivity, high-dimensional distance metrics degrading similarly to KNN's curse-of-dimensionality issue — cross-reference Task 5's `09-knn`), Mental model, Questions.

- [ ] **Step 1:** Read current `notes.md`/notebooks for all 3 topics.
- [ ] **Step 2:** Retrofit `15-unsupervised-learning/notes.md` and `16-anomaly-detection/notes.md`; write `18-pca/notes.md` from scratch.
- [ ] **Step 3:** Add the from-scratch K-Means notebook/cell, the from-scratch covariance/eigendecomposition PCA notebook/cell, and (if not already present) a short cross-method comparison cell in the anomaly-detection notebooks.
- [ ] **Step 4:** Execute every notebook touched/created; confirm exit 0, non-null `execution_count` on every code cell.
- [ ] **Step 5:** `git add` the 3 topic folders, commit: `git commit -m "Phase 1 Task 7: first-principles retrofit — unsupervised learning, PCA, anomaly detection"`.

---

## Verification (after all 7 tasks)

Run from repo root:

```bash
.venv/bin/python - <<'EOF'
import json, pathlib
paths = list(pathlib.Path("05-machine-learning").glob("*/*.ipynb"))
bad = []
for p in paths:
    d = json.load(open(p))
    for c in d["cells"]:
        if c["cell_type"] == "code" and not c.get("execution_count"):
            bad.append(str(p)); break
print(f"{len(paths)} notebooks checked")
print("NOT executed:", bad or "none")

for topic in sorted(pathlib.Path("05-machine-learning").iterdir()):
    if not topic.is_dir(): continue
    nm = topic / "notes.md"
    if not nm.exists():
        print("MISSING notes.md:", topic); continue
    txt = nm.read_text(encoding="utf-8", errors="ignore").lower()
    required = ["## problem","## intuition","why simpler","## mathematical foundation","## algorithm",
                "from-scratch","## practical","## experiment","failure mode","real-world","## mental model","questions to think"]
    missing = [r for r in required if r not in txt]
    if missing:
        print(f"{topic}: missing sections {missing}")
EOF
```

Expect `NOT executed: none`, no `MISSING notes.md`, and no topic with missing template sections.
