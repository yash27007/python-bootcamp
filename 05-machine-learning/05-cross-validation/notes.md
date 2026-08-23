# 05 – Cross Validation

## Problem

You've trained a model. Before you ship it, you need to answer one question: **how well will it perform on data it has never seen?** You don't have production traffic yet — there is no held-out environment you can point the model at and simply read off the true error rate. All you have is the dataset you already collected. So the real problem is: **how do we estimate generalization error using only the data we already have, without access to the future data the model will actually be judged on?**

## Intuition

The obvious idea: split your dataset into two pieces, train on one (say 80%), test on the other (20%), and report the test score. This is a genuine estimate of generalization — the model never saw the test 20% during training.

But which 20%? If you happen to split off 20% that's easier than average (fewer edge cases, more "typical" examples), your reported score will be optimistic. If you happen to split off a harder-than-average 20%, your reported score will be pessimistic. Either way, you get **one number**, and that number depends on an arbitrary choice — which rows landed in the test set — that has nothing to do with how good the model actually is.

Cross-validation's answer: don't pick one split, **rotate through several**, so every example gets to be in the test set at least once, and report the *distribution* of scores (mean and spread) instead of a single lucky-or-unlucky number.

## Why simpler approaches fail

A single train/test split has **high variance as an estimator**: re-running the same split procedure with a different random seed can shift the reported test score by several percentage points on a small dataset, purely from which rows ended up in the 20% test slice — not from any real change in model quality. This is a genuine statistical problem, not a coding inconvenience: you are trying to estimate a population quantity (true generalization error) from a sample, and a single split gives you a single realization of a random variable, with no sense of how much that realization could have differed by chance.

This matters most exactly when it's most tempting to skip cross-validation: small datasets, where a single 20% test set might be only a few dozen rows, and where an unlucky split can flip your comparison between two candidate models entirely. You need an estimate that isn't hostage to one random draw.

## Mathematical foundation

Let $\hat{L}$ denote a test-set score computed from one split (e.g. accuracy, RMSE — sign convention doesn't matter). A single split's score is one sample of a random variable whose randomness comes entirely from which rows landed in train vs. test.

**K-Fold as averaging point estimates.** Partition the $N$ examples into $K$ disjoint folds $F_1, \dots, F_K$, each of size $\approx N/K$. For each fold $k$, train on all data except $F_k$, and evaluate on $F_k$ to get a point estimate $\hat{L}_k$. The K-Fold estimator is the mean of these $K$ point estimates:
$$\hat{L}_{\text{CV}} = \frac{1}{K}\sum_{k=1}^{K} \hat{L}_k$$

**Why averaging reduces variance.** If the $\hat{L}_k$ were independent with variance $\sigma^2$ each, the variance of their mean would be
$$\text{Var}(\hat{L}_{\text{CV}}) = \frac{\sigma^2}{K}$$
— i.e. averaging $K$ estimates shrinks variance by a factor of $K$ compared to relying on a single one ($K=1$). In reality the folds are *not* fully independent (they share overlapping training data — fold 1's training set and fold 2's training set overlap in $K-2$ folds' worth of data), so the true variance reduction is less than the naive $\sigma^2/K$, but the direction of the effect holds: averaging over rotated splits is a strictly lower-variance estimator of generalization error than any single split, at the cost of $K\times$ the compute (training $K$ models instead of 1).

**The mean ± std you compute across folds is not free information about the model's true variance** — see the LOOCV caveat below for the boundary case where this reasoning breaks.

## Algorithm

**K-Fold (general form):**
1. Shuffle (optional but usually recommended) and partition the $N$ examples into $K$ roughly equal folds.
2. For $k = 1, \dots, K$:
   - Train the model on all folds except $F_k$.
   - Evaluate it on $F_k$, record score $\hat{L}_k$.
3. Report $\hat{L}_{\text{CV}} = \text{mean}(\hat{L}_1, \dots, \hat{L}_K)$ and $\text{std}(\hat{L}_1, \dots, \hat{L}_K)$.

**Variants**, all following the same "rotate the test fold" skeleton with different partitioning rules:

| Method | Best for | Key idea | Cost |
| --- | --- | --- | --- |
| Hold-out | Quick baseline | Single split | Low |
| K-Fold | General use | Rotate test fold | Medium |
| Stratified K-Fold | Imbalanced classes | Preserve class ratios per fold | Medium |
| LOOCV | Very small data | Leave exactly 1 out ($K = N$) | High |
| Repeated K-Fold | More stability | Repeat K-Fold with different shuffles | High |
| Nested CV | Model selection | Inner loop tunes, outer loop evaluates | High |
| Time Series CV | Temporal data | Forward chaining, respect time order | Medium |
| Group K-Fold | Grouped/correlated data | Keep each group in a single fold | Medium |

- **Stratified K-Fold** preserves class proportions in every fold — necessary for imbalanced classification, otherwise a fold could randomly end up with almost no minority-class examples.
- **LOOCV** is the $K=N$ extreme: train on $N-1$ points, test on the 1 left out, repeat for every point.
- **Nested CV** wraps an inner CV loop (for hyperparameter search) inside an outer CV loop (for the final performance estimate), so the number used to select hyperparameters is never the same number reported as the model's generalization estimate — avoiding optimistic bias.
- **Group K-Fold** and **Time Series CV** modify the *partitioning rule* to respect a constraint the plain random split would violate (correlated groups; temporal ordering), preventing information leakage from test back into train.

## From-scratch implementation

The mechanic underlying every K-Fold variant is just index arithmetic: given $N$ and $K$, decide which row indices go into each fold's test set, and everything else goes into that fold's training set.

```python
import numpy as np

def kfold_indices(n, k, shuffle=True, seed=42):
    """Yield (train_idx, test_idx) arrays for k-fold CV over n samples."""
    idx = np.arange(n)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(idx)

    fold_sizes = np.full(k, n // k, dtype=int)
    fold_sizes[: n % k] += 1  # distribute the remainder across the first folds

    current = 0
    for size in fold_sizes:
        test_idx = idx[current : current + size]
        train_idx = np.concatenate([idx[:current], idx[current + size:]])
        yield np.sort(train_idx), np.sort(test_idx)
        current += size

# sanity check against sklearn's KFold
from sklearn.model_selection import KFold
n, k = 23, 5
scratch_folds = list(kfold_indices(n, k, shuffle=False))
sk_folds = list(KFold(n_splits=k, shuffle=False).split(np.arange(n)))
for (my_tr, my_te), (sk_tr, sk_te) in zip(scratch_folds, sk_folds):
    assert np.array_equal(my_tr, sk_tr) and np.array_equal(my_te, sk_te)
print("scratch kfold_indices matches sklearn.KFold exactly")
```

This handful of lines is the entire mechanic that every CV variant in the table above builds on: `StratifiedKFold` adds a class-balance constraint to which indices go where; `GroupKFold` adds a group-membership constraint; `TimeSeriesSplit` replaces random partitioning with a forward-chaining rule. None of them do anything conceptually beyond "produce train/test index arrays" — the library versions differ only in *which* indices they're allowed to put together.

## Practical implementation

`05-cross-validation/cross-validation.ipynb` runs the library versions of these ideas on `load_breast_cancer`, mapped back to the mechanic above:

- **`KFold(n_splits=5, shuffle=True, random_state=42)`** — the direct library equivalent of `kfold_indices` above, wired into `cross_val_score` to train and score a `RandomForestClassifier` across all 5 rotations automatically.
- **`StratifiedKFold`** — same rotation, but each fold's test indices are chosen to preserve the overall class balance, which the from-scratch version above doesn't do (it only respects size, not class labels).
- **`LeaveOneOut`** — the $K=N$ extreme; the notebook runs all ~569 folds (one per sample) and reports mean ± std of the per-sample (0/1) scores.
- **`GridSearchCV` / `RandomizedSearchCV`** — hyperparameter search that internally runs K-Fold (5-fold, Stratified for classification) for every candidate hyperparameter combination, exhaustively for `GridSearchCV`, via random sampling of a fixed budget for `RandomizedSearchCV`.

## Experiment

**Hypothesis (stated before running):** the K-Fold mean±std should be a more stable/trustworthy estimate of generalization accuracy than any single split's score — i.e. individual fold scores should scatter around the mean rather than agreeing exactly, demonstrating why reporting a single split's number would have been misleading.

**Setup:** `RandomForestClassifier(n_estimators=100, random_state=42)` evaluated with `KFold(5)`, `StratifiedKFold(5)`, and `LeaveOneOut()` on `load_breast_cancer` (569 samples, binary classification, class balance 212/357).

**Actual result** (from the executed notebook): K-Fold produced 5 per-fold accuracies with a non-zero spread around their mean (mean ± std reported directly in the notebook's printed summary), confirming that any single one of those 5 folds, taken alone, would have reported a different number than the mean. Stratified K-Fold's mean was very close to plain K-Fold's here, because `load_breast_cancer`'s class imbalance (37% minority class) is mild enough that random folds already come out roughly balanced. LOOCV's reported std was computed over 569 individual 0/1 scores.

**Interpretation:** the non-zero spread across K-Fold's 5 scores is direct empirical evidence for the "why simpler approaches fail" argument — a single split could have landed on any one of those 5 values, over- or under-stating the model's real performance.

**Limitations:** this was run once, with one fixed `random_state`; the comparison doesn't itself quantify *how much* variance a single hold-out split would have shown without repeating that hold-out split many times (`ShuffleSplit`/Monte Carlo CV would make that comparison explicit — not run here, but a natural next experiment).

### Resolved caveat: LOOCV's per-fold "std" is not comparable to K-Fold's

LOOCV's per-fold scores are 0/1 (each fold tests exactly one sample, so the model is either right or wrong on that one point). The std computed across those 569 binary outcomes reflects **which individual samples happen to be easy or hard to classify** — essentially per-sample noise in the dataset — not the model's sensitivity to *which training set it saw*, since LOOCV's $N-1$ training sets overlap almost completely with each other (any two LOOCV training sets differ by only 2 rows out of $N-1$). K-Fold's per-fold std, by contrast, reflects variation across genuinely different training sets (each K-Fold training set differs from another by roughly $2N/K$ rows) evaluated on genuinely different multi-sample test sets. **The two stds are measuring different things and should not be read side-by-side as "LOOCV is more/less stable than K-Fold."** If you want an estimate of how sensitive a model is to which training data it saw, K-Fold's fold-to-fold spread (or repeated K-Fold) is the right tool; LOOCV's spread instead tells you about the distribution of per-example difficulty.

## Failure modes

- **A single train/test split** can accidentally be lucky or unlucky — the entire motivation for this topic.
- **LOOCV has high variance for unstable models**, despite each individual fold using almost all the data. Because the $N$ training sets in LOOCV overlap almost entirely, the fitted models across folds are highly correlated with each other; for a model whose fit is sensitive to small perturbations in the training set (e.g. deep unpruned trees, high-variance models generally), that correlation means LOOCV's *overall* estimate is not the "almost unbiased, therefore automatically better" answer it's sometimes assumed to be — it trades bias for a different variance profile, and its per-fold std should not be interpreted the way K-Fold's is (see caveat above).
- **Data leakage through preprocessing.** Fitting a scaler, encoder, or feature selector on the *whole* dataset before splitting into folds leaks test-fold information into training, inflating every reported score. Always fit preprocessing only on each fold's training portion (`Pipeline` in scikit-learn enforces this automatically).
- **Ignoring structure in the data.** Plain K-Fold on grouped data (e.g. multiple rows per patient) or temporal data (e.g. daily sales) can put correlated or future information into the training fold, again leaking information and producing an optimistic estimate. Group K-Fold and Time Series CV exist specifically to prevent this.

## Real-world usage

- Cross-validation is the standard tool for comparing candidate models or hyperparameter settings before choosing what to ship — almost every `GridSearchCV`/`RandomizedSearchCV`-style hyperparameter search in production ML pipelines uses K-Fold or Stratified K-Fold internally.
- Nested CV is used whenever a paper or production report needs to claim "this is our unbiased estimate of generalization performance" *and* the model involved hyperparameter tuning — using the same folds for tuning and final reporting is a common, silent source of overoptimistic published results.
- Time Series CV / forward-chaining validation is the only correct approach for forecasting models — using ordinary K-Fold on time-ordered data trains on the future to predict the past, an error that would never show up as a bug, only as a model that mysteriously underperforms once deployed.

## Mental model

Cross-validation trades compute for a lower-variance estimate of a number you can't otherwise observe — the model's true generalization error.

## Questions to think about

1. You have 200 rows and want to compare two candidate models. Would you prefer 5-fold or 10-fold CV, and what's the tradeoff you're making as $K$ increases toward $N$?
2. A colleague reports "LOOCV gave us a lower std across folds than 5-fold CV, so LOOCV is the more reliable estimate." Using the caveat above, explain what's wrong with that conclusion.
3. You're building a model to predict tomorrow's stock price from the last 30 days of prices. Why would plain `KFold(shuffle=True)` silently produce a misleadingly good CV score, and what would you use instead?
4. Suppose fitting your `StandardScaler` on the full dataset before running K-Fold changes your reported accuracy from 91% to 94%. Which number is correct, and why?
