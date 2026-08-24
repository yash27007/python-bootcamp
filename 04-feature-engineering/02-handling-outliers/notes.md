# Handling Outliers

## Problem

A single extreme value can dominate a statistic computed over an entire dataset. The mean is a
sum divided by count — one value ten times larger than everything else pulls the whole average
toward it; a model fit by minimizing squared error (linear regression, k-means, PCA) is
disproportionately influenced by whichever points sit farthest from the rest, because the squared
term explodes for large deviations. Left unexamined, a handful of extreme values (or plain data
errors — a misplaced decimal, a sensor spike, a `-1` used as a missing-value sentinel) can distort
summary statistics and model fits so badly that the result describes the outliers more than it
describes the data.

## Intuition

Fifteen students score between 22 and 98 on a test. Add one more "student" who scored 900 (a data
entry error — someone typed an extra zero). The class average, which was previously a reasonable
summary of "typical performance," is now dragged upward by one bad row — the median barely moves,
because it only cares about rank order, not magnitude. This is the core intuition behind every
outlier-handling technique: **some statistics (mean, standard deviation, squared-error model fits)
are sensitive to extreme values; others (median, IQR, rank-based methods) are much less so** —
detecting outliers is largely about using the robust statistics to flag what the sensitive ones
would otherwise be silently corrupted by.

## Why simpler approaches fail

- **Eyeballing a histogram or sorted list** doesn't scale past a few dozen points and is not
  reproducible — two people can disagree on where "normal" ends, and there is no principled
  boundary to apply consistently to new data.
- **A fixed absolute threshold** (e.g. "flag anything above 100") only works if you already know
  the expected range in advance and that range never shifts — it breaks the moment the data's
  scale changes (different units, different time period, different subpopulation).
- **Standard deviation-based thresholds computed the naive way** are themselves outlier-sensitive:
  the same extreme value that you're trying to detect also inflates the mean and standard
  deviation used to detect it, weakening the very test meant to catch it. This is exactly why the
  IQR method (built from quartiles, which are far more robust to extreme values) is generally
  preferred over a naive mean ± k·std rule for skewed or contaminated data.

## Mathematical foundation

**The IQR (interquartile range) rule.** Given the first quartile $Q_1$ (25th percentile) and third
quartile $Q_3$ (75th percentile) of a distribution:

$$
\text{IQR} = Q_3 - Q_1
$$

$$
\text{lower fence} = Q_1 - k \cdot \text{IQR} \qquad \text{upper fence} = Q_3 + k \cdot \text{IQR}
$$

with $k = 1.5$ the conventional choice (the boundary Tukey's box plot draws its whiskers to). A
value outside $[\text{lower fence}, \text{upper fence}]$ is flagged. Because $Q_1$, $Q_3$ (and
hence IQR) are computed from rank order, not magnitude, a single extreme value cannot pull the
fences toward itself the way it pulls a mean.

**The Z-score rule.** For a value $x$ from a distribution with mean $\mu$ and standard deviation
$\sigma$:

$$
z = \frac{x - \mu}{\sigma}
$$

$z$ measures "how many standard deviations from the mean" $x$ sits; a common threshold is
$|z| > 3$. Unlike the IQR rule, both $\mu$ and $\sigma$ are themselves sensitive to the outliers
being tested for — a large enough outlier inflates $\sigma$ and can prevent its own $z$-score from
crossing the threshold, which is one reason the IQR method is generally more robust on skewed or
heavily contaminated data.

## Algorithm

**IQR-based detection:**
1. Compute $Q_1$, $Q_3$ from the data (25th/75th percentiles).
2. Compute $\text{IQR} = Q_3 - Q_1$ and the fences $Q_1 - 1.5\cdot\text{IQR}$,
   $Q_3 + 1.5\cdot\text{IQR}$.
3. Flag any value outside the fences.

**Z-score-based detection:**
1. Compute the mean $\mu$ and standard deviation $\sigma$ of the data.
2. Compute $z = (x-\mu)/\sigma$ for every value.
3. Flag any value with $|z|$ above a chosen threshold (commonly 3).

## From-scratch implementation

`02-handling-outliers.ipynb` implements both rules directly from their definitions with NumPy —
`np.percentile` for the IQR fences, a manual `(x - mean) / std` for Z-scores — on
`[-10, -29, 45, 32, 56, 78, 33, 48, 98, 77, 86, 22, 80, 97, 67, 59, 45, 170, 232, 220]`, and checks
the Z-score computation against `scipy.stats.zscore` and the IQR statistic against
`sklearn.preprocessing.RobustScaler` (which internally centers/scales by median and IQR):

```
IQR fence: (np.float64(-28.125), np.float64(158.875))
IQR-flagged outliers:   [-29. 170. 232. 220.]
Z-score-flagged outliers (|z|>3): []

manual z-scores match scipy.stats.zscore: True
RobustScaler center_ (median) = 63.0  |  np.median = 63.0
RobustScaler scale_  (IQR)    = 46.75  |  manual Q3-Q1 = 46.75
```

Notably, **the IQR rule flags 4 outliers here while the Z-score rule (|z|>3) flags none** — on
this small, right-skewed sample the large values inflate the mean and standard deviation enough
that no single point's z-score crosses 3, exactly the robustness gap described in "Why simpler
approaches fail" above. The manual computations match the library equivalents exactly.

## Practical implementation

The notebook covers the 5-number summary (`np.quantile`), the IQR fence computation, and
`seaborn.boxplot` — whose whiskers *are* the IQR fences and whose individual markers beyond them
*are* the flagged outliers, making the box plot a direct visualization of the exact rule derived
above, not a separate technique.

## Experiment

**Hypothesis:** naively removing every point an IQR rule flags as an outlier can *bias* a fitted
model's estimate away from the true relationship, when some of the flagged points are genuine
extreme observations rather than errors.

**Setup:** 200 synthetic points where `y = 2.0*x + 5.0 + noise` holds genuinely across the whole
range — 190 points cluster near `x=50`, and 10 points have `x` uniformly between 150 and 300
(rare, but real, not corrupted). The IQR rule is applied to `x` and flags points as outliers
purely because they're numerically extreme; a `LinearRegression` is fit both on the full data and
on the data with flagged points removed, then compared to the known true slope (2.0).

**Actual result:**
```
Points flagged as outliers by the IQR rule: 10 / 200
True slope: 2.0
Slope estimated on FULL data (genuine extremes kept):        2.0150
Slope estimated after REMOVING IQR-flagged points:            1.9126
R^2 (full-data model, evaluated on all 200 points):            0.9670
R^2 (outlier-removed model, evaluated on the SAME all points): 0.9644
```

**Interpretation:** keeping the genuine extreme values gives a slope estimate (2.0150) closer to
the true slope (2.0) than removing them does (1.9126) — a real, measured ~5x increase in slope
error from blindly applying the IQR rule. The extreme points weren't noise; they were the part of
the data that best constrained the slope at the high end of the range, and removing them threw
away exactly the information needed to estimate it accurately.

**Limitations:** a single synthetic linear relationship, one random seed, one outlier rule (IQR).
A dataset where the extreme values genuinely *are* corrupted (not a real linear continuation)
would show the opposite result — removing them would improve the fit — which is precisely why this
decision cannot be automated without domain judgment about what the extreme values represent.

## Failure modes

- **Removing genuine extreme values, not just erroneous ones** — demonstrated concretely above: a
  statistical outlier rule cannot distinguish "rare but real" from "measurement error," and
  removing the former measurably biases downstream estimates away from the truth.
- **Choosing $k$ or the Z-score threshold without checking the data's actual shape.** The
  $k{=}1.5$ IQR convention and the $|z|{>}3$ convention are defaults tuned for roughly
  bell-shaped data — applied to strongly skewed or multimodal data, both can flag far too many or
  far too few points, as shown directly above where the two rules disagreed entirely on the same
  data (4 outliers vs. 0).
- **Detecting outliers on the full dataset before any train/test split**, mirroring the imputation
  leakage failure mode in `../01-missing-values/notes.md` — computing $Q_1$/$Q_3$/mean/std (or
  deciding which rows to drop) using rows that will later be used as test data leaks the same kind
  of information a leaky imputer does; the fences/mean/std should be computed on training data
  only.

## Real-world usage

Outlier detection is a first-pass data-quality check in almost every real pipeline — flagging
sensor glitches, data-entry typos, and payment-processing anomalies before they corrupt downstream
aggregates or model fits. Financial fraud detection and network intrusion detection are, at their
core, outlier-detection problems (often with far more sophisticated multivariate methods —
Isolation Forest, One-Class SVM, Local Outlier Factor — that generalize the same "how unusual is
this point" idea beyond a single feature). `RobustScaler`'s median/IQR-based scaling (used above)
is the standard choice for feature scaling when a dataset is known to contain real outliers that
shouldn't be allowed to dominate a `StandardScaler`'s mean/std the way they would a naive
approach.

## Mental model

An outlier detector answers "is this value unusual?" — it can never by itself answer "is this
value *wrong*?" Confusing the two questions is the single most common mistake in outlier handling:
statistical unusualness is a necessary trigger for investigation, never sufficient grounds for
automatic deletion.

## Questions to think about

1. On the sample dataset, the IQR rule flagged 4 outliers and the Z-score rule flagged 0, on the
   *same* data. Which rule would you trust more here, and why does the mean/std computation itself
   being outlier-sensitive matter to your answer?
2. The regression experiment showed removing genuine extreme values *increased* slope error. Under
   what change to the experiment (something about how the extreme points were generated) would
   removing them instead *decrease* the error?
3. `RobustScaler`'s `center_`/`scale_` matched the manual median/IQR exactly. Why would using
   `StandardScaler` (mean/std) instead of `RobustScaler` on a dataset containing genuine outliers
   change how much those outliers influence every other feature's scaled values?
4. If you were building an automated pipeline that needed to run unattended (no human review of
   flagged points), what would you do differently from simply dropping every IQR-flagged row,
   given the failure mode demonstrated above?
5. Both IQR and Z-score are univariate — they look at one column at a time. Sketch a case (two
   features, described in words) where a point is not extreme in either feature individually, but
   is clearly an outlier when both features are considered together. What does this imply about
   applying univariate outlier detection column-by-column to a multi-feature dataset?
