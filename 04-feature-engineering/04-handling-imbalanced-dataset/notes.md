# Handling Imbalanced Datasets

## Problem

Many real classification tasks have classes that occur in very different proportions — fraud is
rare compared to legitimate transactions, disease is rare compared to health, defects are rare
compared to passing units. Standard training procedures (and the metrics used to judge them)
implicitly assume classes matter equally per-example. When one class vastly outnumbers another, a
model can score extremely well on the naive metric while completely failing at the one thing it
was built for: correctly identifying the rare, usually more important, class.

## Intuition

If 900 of 1,000 patients in a dataset are healthy and only 100 have a disease, a "model" that
answers "healthy" to every single patient, without looking at any data, is right 90% of the time.
That 90% number *looks* like a good model. It is a model that has never once correctly identified
a sick patient — the exact failure a real diagnostic tool cannot afford. The problem isn't the
model; it's that "accuracy" was the wrong question to be asking of an imbalanced problem in the
first place.

## Why simpler approaches fail

- **Training on the raw imbalanced data and evaluating with accuracy** rewards a model for
  matching the majority class and gives essentially no training signal (and no evaluation
  penalty) for ignoring the minority class entirely — demonstrated numerically below.
- **Blind class balancing by simple duplication (naive upsampling)** copies existing minority rows
  verbatim; a model can now memorize those exact duplicated points rather than learning the
  general shape of the minority class, increasing overfitting risk without adding real new
  information.
- **Blind downsampling** throws away real majority-class data to force balance, discarding
  information the model could have used, and becomes especially costly when the minority class is
  already small (little data survives).

## Conceptual foundation: what SMOTE actually synthesizes

SMOTE (Synthetic Minority Over-sampling Technique) generates *new* minority-class points instead
of duplicating existing ones, and the mechanism is exactly a nearest-neighbor computation followed
by linear interpolation:

1. Pick a real minority-class point $p$.
2. Find its $k$ nearest neighbors **among the minority class only**, using Euclidean distance —
   the identical distance computation and "sort, keep the closest $k$" procedure covered from
   first principles (distance metrics, the role of $K$, feature scaling) in
   [`../../05-machine-learning/09-knn/notes.md`](../../05-machine-learning/09-knn/notes.md).
3. Pick one of those $k$ neighbors at random, $q$.
4. Generate a synthetic point $p_{\text{synthetic}} = p + \lambda (q - p)$ for a random
   $\lambda \in [0, 1]$ — a point somewhere on the line segment between $p$ and $q$.

SMOTE *is* KNN's distance-and-neighbor-search machinery, reused for synthesis rather than
prediction: instead of asking "what's the label of my $k$ nearest neighbors," it asks "generate a
new point somewhere between me and one of my $k$ nearest neighbors." Because every synthetic point
is constrained to lie between two real minority points, SMOTE cannot invent an entirely new region
of feature space that didn't already have minority-class support nearby.

## Algorithm

**SMOTE**, as implemented from scratch in the notebook:
1. For each synthetic sample to generate: pick a random minority point $p$.
2. Compute the Euclidean distance from $p$ to every other minority point.
3. Take the $k$ closest (excluding $p$ itself) as candidate neighbors.
4. Choose one neighbor $q$ uniformly at random from those $k$.
5. Interpolate: $p_{\text{synthetic}} = p + \lambda(q-p)$, $\lambda \sim \text{Uniform}(0,1)$.

## From-scratch implementation

`04-handling-imbalanced-dataset.ipynb` implements exactly the 5 steps above with plain NumPy (no
`imblearn` call) on a toy 2D dataset (`make_classification`, 898/102 imbalance) and visualizes the
result:

```
Real minority points: 102   Synthetic points generated: 50
```

The plot shows the 50 synthetic points (marked with an `x`) scattered strictly *between* pairs of
real minority points, never coinciding with a real point and never landing in a region with no
nearby minority support — visually confirming the interpolation mechanism, not just the textual
description of it.

## Practical implementation

The notebook covers `sklearn.utils.resample` for naive up/downsampling and `imblearn.over_sampling
.SMOTE` as the library implementation of the exact interpolation mechanism built from scratch
above — `SMOTE().fit_resample(X, y)` runs the same "find $k$ nearest minority neighbors, interpolate"
loop internally, at production scale and with additional safeguards (e.g. automatic $k$ adjustment
for very small minority classes) that the from-scratch version omits for clarity.

## Experiment

**Hypothesis 1 (accuracy is misleading):** a model that always predicts the majority class scores
high accuracy on an imbalanced dataset while providing zero value for identifying the minority
class.

**Setup:** the notebook's own 900/100-imbalanced synthetic dataset; accuracy computed both by hand
from the class counts and via a real fitted `sklearn.dummy.DummyClassifier(strategy=
"most_frequent")` on a held-out test split.

**Actual result:**
```
Class counts: {0: 900, 1: 100}
Naive 'always predict class 0' accuracy (by hand): 0.9000
DummyClassifier(most_frequent) test accuracy: 0.9000
```

**Interpretation:** both computations agree exactly — a model using zero information about the
features reaches 90% accuracy while identifying not a single minority-class case. Accuracy alone
cannot distinguish this model from one that actually learned something about the minority class.

**Hypothesis 2 (resampling before the split leaks information):** applying SMOTE to the *entire*
dataset before splitting into train/test lets synthetic training points be interpolated from real
points that end up in the test fold, producing an optimistically biased test metric.

**Setup:** 80-row synthetic dataset, 85/15 imbalance, classes overlapping (means 0.8 apart, both
std=1 — realistic rather than trivially separable), 300 trials comparing SMOTE-before-split
("leaky") against SMOTE-after-split, fit on the training fold only ("correct").

**Actual result:**
```
n=80, 85/15 imbalance, overlapping classes, 300 trials (test_size=0.4):
  Leaky (SMOTE before split)   mean test accuracy: 0.7338
  Correct (SMOTE after split)  mean test accuracy: 0.7125
  Mean difference (leaky - correct): +0.0213
  Trials where leaky > correct: 181/300   |   correct > leaky: 119/300
```

**Interpretation:** the leaky pipeline is optimistically biased on average (+0.0213 accuracy, and
wins outright in 181/300 trials vs. 119/300 for the correct pipeline) — a real, directionally
consistent effect from resampling before the split, exactly parallel to the imputation-leakage
result in `../01-missing-values/notes.md`.

**Limitations:** synthetic Gaussian data, logistic regression only, one imbalance ratio and one
class-overlap setting; real datasets with more features or stronger nonlinear class boundaries
could show a larger or smaller gap.

## Failure modes

- **Applying SMOTE before the train/test split (data leakage)** — the same category of bug as
  `../01-missing-values/notes.md`'s imputation-before-split failure, demonstrated with real,
  measured numbers above (+0.0213 average accuracy inflation, 181/300 trials favoring the leaky
  pipeline). The fix is identical in spirit: fit resampling on the training fold only, leave the
  test fold real and untouched.
- **Class weighting vs. resampling — a real tradeoff, not a strictly-better option.** Both
  `class_weight='balanced'` and SMOTE-resampling were run on the same overlapping-class data and
  compared on minority-class recall (the metric accuracy hides):
  ```
  unweighted                  accuracy=0.8500   minority-class recall=0.0556
  class_weight='balanced'     accuracy=0.6833   minority-class recall=0.6111
  SMOTE-resampled             accuracy=0.6750   minority-class recall=0.5556
  ```
  Both weighting and resampling trade a substantial drop in raw accuracy for a large gain in
  minority-class recall — the unweighted model's 85% accuracy comes almost entirely from ignoring
  the minority class (5.6% recall), which is precisely the accuracy-is-misleading problem in
  practice, not just in theory. Neither weighting nor resampling is "better" in the abstract; the
  right choice depends on which class's errors are more costly in the actual application.

## Real-world usage

Fraud detection, medical diagnosis, defect detection, churn prediction, and rare-event
forecasting are all naturally imbalanced classification problems where accuracy is close to
useless as an evaluation metric — precision, recall, F1, and precision-recall curves (rather than
ROC curves, which can look deceptively good under heavy imbalance) are the standard tools instead.
SMOTE and its variants (Borderline-SMOTE, ADASYN) are standard preprocessing steps in production
imbalanced-classification pipelines, always applied strictly inside the training-fold boundary
of cross-validation to avoid the leakage failure mode measured above. Class weighting is often
preferred in large-scale production systems specifically because it requires no extra synthetic
data generation step and composes cleanly with any model exposing a `class_weight` or
per-sample-weight parameter.

## Mental model

Accuracy answers "how often is the model right overall" — on imbalanced data that question is
dominated by the class with more examples, which is often the class you care about least. SMOTE
is KNN's neighbor-search machinery repurposed to manufacture new points between real ones, not a
separate algorithm — and like every other fitted preprocessing step, it is only leakage-free when
fit strictly inside the training fold.

## Questions to think about

1. The unweighted model reached 85% accuracy with 5.6% minority recall on the same test data where
   `class_weight='balanced'` reached 68% accuracy with 61% recall. If this were a disease-screening
   model, which of the two would you deploy, and what does that choice imply about which metric you
   should have been optimizing for from the start?
2. SMOTE interpolates only between minority points that are close together. What would you expect
   SMOTE to do — and where would its synthetic points land — for a minority class that actually
   forms two well-separated clusters in feature space, rather than one blob?
3. The SMOTE-before-split leakage effect (+0.0213 mean accuracy) was smaller than some
   individual-trial swings. Why is running many trials (300, in this experiment) necessary to
   trust the *direction* of a leakage effect, rather than relying on a single train/test split?
4. Class weighting changes the loss function; SMOTE changes the training data. Name one type of
   model where `class_weight` has no natural equivalent (i.e., the model has no loss function to
   reweight) — what would you have to do instead for that model?
5. The naive majority-class baseline scored 90% accuracy while catching zero minority cases. Design
   a single alternative metric (in words) that would score that same naive model at 0% or near-0%,
   and confirm it would score a perfect classifier at 100%.
