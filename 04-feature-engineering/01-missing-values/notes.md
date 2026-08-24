# Missing Values

## Problem

Real datasets have gaps. A sensor drops a reading, a survey respondent skips a question, a
database join finds no matching row — the result is the same either way: some cells in the table
have no value. Most algorithms cannot simply skip over a missing cell mid-computation — a mean, a
distance calculation, a matrix multiplication, a gradient step all assume every cell holds a
number. Left alone, a single `NaN` propagates and silently breaks (or crashes) anything downstream
that touches it. Something has to be decided about every missing entry before the data reaches a
model: drop it, or fill it with something — and *what* to fill it with is not a free choice, it
depends on *why* the value is missing in the first place.

## Intuition

Imagine a class roster where some students' ages are blank. If the teacher just forgot to write a
few ages down at random, filling in "the class average age" for those blanks is a reasonable
guess — nothing about *why* they're blank tells you anything about what the true age was. But if
the ages are blank specifically for students who skipped the form because they didn't want to
reveal they were older than their classmates, "the class average" is a systematically wrong guess
for exactly those rows — the *reason* the value is missing is correlated with the value itself.
Choosing an imputation method without first asking "why is this missing?" is choosing a bandage
without checking what kind of wound it is.

## Why simpler approaches fail

- **Dropping every row with a missing value** throws away any row that has a gap in *any* column,
  even columns you don't care about. On the Titanic dataset used in this notebook, `df.dropna()`
  keeps only **182 of 891 rows (20.4%)** — a naive fix that discards 80% of the data just because
  one column (`age`, `deck`, `embarked`) happened to have a gap somewhere.
- **Dropping every column with a missing value** avoids losing rows but discards entire features —
  `deck`, `age`, and `embarked` vanish along with whatever predictive signal they carried, even
  though most of `age` and `embarked`'s values are perfectly fine.
- **A single fill value applied blindly** (mean, median, or mode) treats every missing entry as
  interchangeable, ignoring both the shape of the distribution (mean is pulled by outliers; median
  isn't) and, more fundamentally, *why* the value is missing at all — covered next.

## Conceptual foundation: the MCAR / MAR / MNAR taxonomy

This is a genuine three-way distinction about the *mechanism* generating the gaps, not a list of
imputation methods — it decides which fixes are even statistically valid:

**MCAR — Missing Completely At Random.** The probability that a value is missing is unrelated to
any variable, observed or not. A sensor randomly drops a reading regardless of what that reading
would have been or what else is going on. Under MCAR, the *observed* rows are a random,
representative subsample of the full data, so simple methods (listwise deletion, unconditional
mean/median imputation) introduce no systematic bias — they just lose some precision.

**MAR — Missing At Random.** The probability of missingness depends on *other observed variables*,
but not on the missing value itself, once those other variables are accounted for. Example: older
passengers were less likely to report their age, but conditional on knowing "this passenger is
elderly" (inferred from, say, their title or class), the fact that their exact age is missing
carries no further information about what that age actually was. Simple unconditional imputation
now biases estimates (it ignores the systematic relationship with other columns) — methods that
condition on other variables (KNN imputation, MICE/`IterativeImputer`) are needed instead.

**MNAR — Missing Not At Random.** The probability of missingness depends on the *missing value
itself*. Wealthy passengers might be the ones who declined to report their fare, precisely because
it was unusually high — the missingness is informative about the very thing that's missing. No
imputation method that only looks at observed data can fully correct for this: the mechanism has
to be modeled explicitly, or the bias has to be acknowledged and tested with domain knowledge and
sensitivity analysis.

The practical consequence: before picking a fill method, ask which of these three describes the
gap. MCAR licenses the simplest fixes. MAR requires conditioning on other columns. MNAR requires
acknowledging that no purely statistical fix removes the bias.

## Algorithm

**Mean/median imputation (the from-scratch case below):**
1. Compute the statistic (mean or median) over the *observed* (non-missing) values of the column.
2. Replace every missing entry in that column with the statistic.

**KNN imputation (conceptually, detailed in the notebook and linked below):**
1. For a row with a missing value in column $c$, find its $K$ nearest neighbors using the row's
   *other*, non-missing columns as the distance features.
2. Fill the missing value with the (distance-weighted) average of column $c$ among those neighbors.

## From-scratch implementation

`01-missing-values.ipynb` implements manual mean and median imputation directly with NumPy —
`np.nanmean` / `np.nanmedian` to compute the statistic over observed values, `np.where` to fill —
and checks it against `sklearn.impute.SimpleImputer` fit on the identical column:

```
manual mean statistic:    29.69911764705882  | sklearn mean statistic:    29.69911764705882
manual median statistic:  28.0                | sklearn median statistic:  28.0
manual vs sklearn MEAN imputed values match:   True
manual vs sklearn MEDIAN imputed values match: True
```

The manual implementation and `SimpleImputer` compute the exact same numbers, because there is
nothing more to mean/median imputation than "compute one statistic, broadcast it into the gaps."

## Practical implementation

The notebook covers the full practical surface: finding and visualizing missing values
(`df.isna().sum()`), row/column deletion and its cost (measured above), mean/median/mode
imputation via `pandas.Series.fillna`, and `sklearn.impute.SimpleImputer` / `KNNImputer` /
`IterativeImputer` as the library-level equivalents of the from-scratch statistic computation.

**Beyond a single statistic — KNN imputation.** Mean/median imputation gives every missing entry
in a column the *same* fill value, which is exactly wrong under MAR (where the right fill value
depends on the row's other columns). `sklearn.impute.KNNImputer` fixes this by filling a missing
value with the average of that feature among the row's $K$ nearest neighbors in the other feature
columns — the same "find the closest points and average their values" idea covered from first
principles (distance metrics, the role of $K$, feature scaling) in
[`../../05-machine-learning/09-knn/notes.md`](../../05-machine-learning/09-knn/notes.md). KNN
imputation *is* that algorithm, applied to fill a feature instead of predict a target — the
notebook runs `KNNImputer(n_neighbors=5)` on the Titanic data and compares its fills for `age`
against the global-mean fill, row by row.

## Experiment

**Hypothesis:** fitting an imputer (or computing an imputation statistic) on the *entire* dataset
before splitting into train/test leaks information from the test set into training, producing an
optimistic bias in the reported test metric compared to fitting the imputer on the training fold
only.

**Setup — Titanic data (891 rows, ~20% missing `age`):** a "leaky" pipeline fills `age` with the
mean of *all* 891 rows, then splits into train/test; a "correct" pipeline splits first, computes
the mean from the training fold only, and applies that fixed statistic to both folds. Both train a
`LogisticRegression` on `pclass`, `sex`, `age`, `fare` to predict `survived`, repeated over 100
random splits.

**Actual result:**
```
Global mean age (whole dataset, what the leaky pipeline uses): 29.6991
Example train-only mean age (seed=0): 29.9153  (global mean differs by -0.2162)

Leaky pipeline   mean test accuracy over 100 splits: 0.7909
Correct pipeline mean test accuracy over 100 splits: 0.7907
Mean difference (leaky - correct): +0.0002
```

The imputed value itself measurably differs between the two pipelines (a real, always-present
effect — the global mean necessarily incorporates the test fold's own observed ages), but the
downstream accuracy gap is tiny here because a single scalar statistic carries very little
information relative to 891 rows and only ~20% missingness.

**Amplified setup:** the same exact bug, but on a synthetic 40-row dataset with 60% missingness
(smaller sample and higher missing fraction make the imputation statistic a much noisier — and
more test-fold-dependent — estimate), 300 trials, `test_size=0.5`:

```
n=40, 60% missing, 300 trials (test_size=0.5):
  Leaky pipeline   mean test accuracy: 0.6325
  Correct pipeline mean test accuracy: 0.6232
  Mean difference (leaky - correct):   +0.0093
  Trials where leaky > correct: 73/300   |   correct > leaky: 39/300
```

**Interpretation:** the leaky pipeline is optimistically biased in both experiments, and the bias
grows as the imputation statistic becomes noisier relative to the data (smaller n, higher missing
fraction) — exactly what the "test rows contributed to the statistic" mechanism predicts. The
effect is small for a single scalar mean on a large, low-missingness dataset, and larger for
smaller/higher-missingness data or for imputers that use more information (KNNImputer,
IterativeImputer) — the mechanism is identical either way.

**Limitations:** single logistic regression on 4 features, one random-split protocol, one dataset
family (Gaussian-ish synthetic + Titanic); a different model/metric or a stronger correlation
between missingness and the target (MNAR-like) would likely show a larger gap.

## Failure modes

- **Imputing before the train/test split (data leakage)** — demonstrated concretely above: fitting
  the imputation statistic on the full dataset lets test-fold information influence what fills the
  training data, producing an optimistically biased performance estimate. The fix is always the
  same regardless of imputer complexity: fit on train, apply to test — never the reverse order.
- **Applying MCAR-appropriate methods to MAR/MNAR data.** Mean imputation on an MNAR column (like
  fare values withheld by wealthy passengers) doesn't just add noise — it introduces systematic
  bias in a specific, target-relevant direction, and no amount of additional data fixes it, because
  the *mechanism itself* is the source of the bias, not sample size.
- **Mode imputation collapsing rare categories.** Filling every missing categorical value with the
  single most frequent category (as done for `embarked` in this notebook) can noticeably inflate
  that category's frequency if the missing fraction is large, distorting anything downstream that
  depends on the category's true prevalence (e.g. target-encoding statistics, downstream model
  calibration).

## Real-world usage

Every production ML pipeline that ingests real-world data — sensor logs, user-submitted forms,
joined data warehouses — hits missing values as a matter of course, not an edge case. The
MCAR/MAR/MNAR distinction is the first triage question a data scientist should ask before choosing
a fix, because picking the wrong fix for the actual missingness mechanism produces confidently
wrong downstream models rather than obviously broken ones. KNN and iterative (MICE-style)
imputation are standard defaults in production feature-engineering pipelines precisely because
real missingness is rarely pure MCAR. The leakage failure mode above is a direct instance of the
single most common data-leakage bug in applied ML: fitting *any* preprocessing step (imputers,
scalers, encoders) on data that includes information from the evaluation set.

## Mental model

A missing value isn't just "a blank to fill in" — it's the residue of a *mechanism*, and the right
fix depends entirely on what that mechanism was. Any statistic used to fill a gap must be fit on
training data only, exactly like a model's parameters — an imputer is a fitted object, not a
free-floating formula, and leaking test data into its fit is the same class of bug as leaking test
data into the model's own training.

## Questions to think about

1. The `embarked` column had only 2 missing values out of 891 (mode imputation used here). If
   instead 40% of `embarked` were missing, would mode imputation still be defensible? What would
   you check first before deciding?
2. The measured leakage effect on Titanic accuracy was +0.0002 — nearly zero — while the amplified
   synthetic experiment showed +0.0093. Name two properties of a dataset (besides sample size and
   missing fraction) that would make imputation leakage's effect on a downstream metric larger.
3. `KNNImputer` fills a missing value using the row's *other* feature columns. Under which of
   MCAR/MAR/MNAR does this actually fix the bias that mean imputation would introduce, and under
   which does it not help at all? Why?
4. Suppose a column is MNAR because high values are systematically missing (e.g. top earners don't
   report income). Would *any* imputation method — including KNN or iterative imputation —
   actually recover an unbiased estimate of the missing values? What would you need beyond the
   observed data to do so?
5. The correct leakage-free pipeline fits the imputer on the training fold and applies the exact
   same fitted statistic to the test fold. Why must the *test* fold use the *train* statistic
   rather than its own mean/median, even though computing the test fold's own statistic would
   technically "work" in isolation?
