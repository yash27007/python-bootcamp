# Descriptive Statistics

## Problem

A dataset — 150 iris flowers, a million log lines, a spreadsheet of sales — is too large for a
human to reason about one row at a time. Before any model, chart, or decision, you need a handful
of numbers that compress the dataset without lying about it: where is the data centered, how
spread out is it, what shape does it take, and how do variables move together. That compression is
what descriptive statistics does. It **summarises and describes** the main features of a dataset —
unlike inferential statistics (covered in `03-inferential-statistics`), which draws conclusions
that go *beyond* the data you actually have, descriptive statistics only describes *what is in
front of you*.

## Intuition

Take 20 exam scores. "The class did well" is a claim; "the mean is 78, the median is 81, the
standard deviation is 9" is a description you can check and compare across classes. Each summary
number answers a different question:
- **Where is the "typical" value?** → central tendency (mean, median, mode).
- **How spread out are the values?** → dispersion (range, variance, std dev, IQR).
- **Is the data lopsided or symmetric?** → shape (skewness, kurtosis).
- **Do two variables move together?** → covariance/correlation.

No single number is enough on its own — a mean of 78 could come from everyone scoring near 78, or
half the class scoring 40 and half scoring 100. That's exactly why dispersion and shape exist
alongside central tendency.

### Types of data (what summary is even valid)

Which summaries make sense depends on the data type:

| Type | Description | Examples |
|------|-------------|---------|
| **Nominal** | Categories with no order | Colour, Gender, Country |
| **Ordinal** | Categories with a meaningful order | Star rating, Education level |
| **Interval** | Numeric, equal gaps, no true zero | Temperature (°C), Year |
| **Ratio** | Numeric, equal gaps, true zero exists | Height, Weight, Income |

A mean of "star rating" (ordinal) is common in practice but is already a mild abuse — there's no
guarantee the gap between 3★ and 4★ equals the gap between 4★ and 5★. A mean of "country" (nominal)
is meaningless.

## Why simpler approaches fail

The simplest possible summary is "look at the numbers" or "eyeball the first 10 rows." That fails
immediately past a few dozen data points — human working memory can't average 150 numbers reliably,
let alone spot that a distribution is bimodal from a column of digits. The next-simplest fix, "just
report the mean," fails for a specific and important reason: **a single number cannot capture both
location and spread**, and worse, the mean itself is not robust — one extreme outlier (a data-entry
typo of 9,900,000 instead of 99,000) drags a mean far from where the bulk of the data actually
sits, while the median barely moves. This is why the field developed a *vocabulary* of complementary
summaries (mean, median, mode; range, variance, IQR; skewness; correlation) rather than a single
metric — each captures something the others cannot, and using a mean/std-dev pair on the wrong kind
of distribution (see **Failure modes**) actively misleads.

## Mathematical foundation

### Measures of central tendency

**Mean (arithmetic average)**
$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$
Sensitive to outliers. Best for symmetric, continuous data.

**Median** — the middle value when data is sorted. For even *n*: average of the two middle values.
Robust to outliers — preferred for skewed distributions (e.g. income, house prices).

**Mode** — the most frequent value. Can be used with any data type (even nominal). A distribution
can be unimodal, bimodal, or multimodal.

### Measures of dispersion (spread)

**Range**
$$\text{Range} = x_{\max} - x_{\min}$$
Simple but highly sensitive to outliers.

**Variance**
$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n}(x_i - \bar{x})^2$$
Average of squared deviations. Denominator is *n − 1* (Bessel's correction) for **sample**
variance — see the Experiment section below for why that correction exists and what happens
without it.

**Standard deviation**
$$s = \sqrt{s^2}$$
Same units as the data — the most interpretable spread measure.

**Interquartile range (IQR)**
$$\text{IQR} = Q_3 - Q_1$$
- **Q1** = 25th percentile, **Q3** = 75th percentile.
- Robust to outliers.
- Used in box plots and outlier detection (Tukey's fence: $Q_1 - 1.5 \cdot \text{IQR}$,
  $Q_3 + 1.5 \cdot \text{IQR}$).

### Shape of the distribution

**Skewness** measures asymmetry of the distribution around its mean.

| Skewness | Shape | Relationship |
|----------|-------|--------------|
| = 0 | Symmetric | Mean = Median = Mode |
| > 0 | Right-skewed (positive) | Mean > Median > Mode |
| < 0 | Left-skewed (negative) | Mean < Median < Mode |

$$\text{Skewness} = \frac{\frac{1}{n}\sum(x_i - \bar{x})^3}{s^3}$$

**Kurtosis** measures the "tailedness" of the distribution.

| Kurtosis | Description |
|----------|-------------|
| = 3 (excess = 0) | **Mesokurtic** – normal distribution |
| > 3 (excess > 0) | **Leptokurtic** – heavy tails, sharp peak |
| < 3 (excess < 0) | **Platykurtic** – light tails, flat peak |

### Five-number summary & box plot

$$\min,\ Q_1,\ \text{Median},\ Q_3,\ \max$$

A **box plot** (box-and-whisker plot) visualises this summary:
- Box spans Q1 → Q3 (the IQR)
- Whiskers extend to $Q_1 - 1.5\cdot\text{IQR}$ and $Q_3 + 1.5\cdot\text{IQR}$
- Points beyond whiskers are **outliers**

### Covariance & correlation

**Covariance**
$$\text{Cov}(X, Y) = \frac{1}{n-1}\sum(x_i - \bar{x})(y_i - \bar{y})$$
Direction of the linear relationship between two variables.

**Pearson correlation coefficient**
$$r = \frac{\text{Cov}(X,Y)}{s_X \cdot s_Y}, \quad r \in [-1, 1]$$

| r value | Interpretation |
|---------|----------------|
| +1 | Perfect positive linear |
| 0 | No linear relationship |
| −1 | Perfect negative linear |

> Correlation ≠ Causation — see **Failure modes**.

### Quick reference cheat sheet

| Measure | Robust to Outliers? | Data Type |
|---------|--------------------|-----------|
| Mean | No | Interval / Ratio |
| Median | Yes | Ordinal and above |
| Mode | N/A | Any |
| Std Dev | No | Interval / Ratio |
| IQR | Yes | Ordinal and above |
| Pearson r | No | Interval / Ratio |

## Algorithm

To describe a numeric dataset end-to-end:
1. Confirm the data type (nominal/ordinal/interval/ratio) — this constrains which summaries are
   even meaningful.
2. Compute central tendency: mean, median, mode.
3. Compute dispersion: range, variance (n−1), std dev, IQR.
4. Compute shape: skewness, kurtosis; compare mean vs median vs mode to sanity-check the sign of
   skewness.
5. Plot: histogram (shape), box plot (five-number summary + outliers).
6. If there are multiple numeric variables: compute the covariance/correlation matrix, visualise
   with a heatmap, and remember correlation only captures *linear* association.
7. If there's a categorical grouping variable: compute grouped statistics (`groupby`) before
   drawing conclusions from the pooled data — see Simpson's paradox in **Failure modes**.

## From-scratch implementation

Mean, variance, standard deviation, and Pearson correlation, implemented directly from their
mathematical definitions in pure Python (no `np.mean`/`np.var`/`np.std`/`np.corrcoef`), then
verified against NumPy on the real Iris `sepal_length` / `petal_length` columns
(`descriptive_statistics.ipynb`, "From-Scratch" section):

```python
def mean_scratch(x):
    return sum(x) / len(x)

def variance_scratch(x, ddof=1):
    m = mean_scratch(x)
    return sum((xi - m) ** 2 for xi in x) / (len(x) - ddof)

def std_scratch(x, ddof=1):
    return variance_scratch(x, ddof) ** 0.5

def covariance_scratch(x, y):
    mx, my = mean_scratch(x), mean_scratch(y)
    n = len(x)
    return sum((xi - mx) * (yi - my) for xi, yi in zip(x, y)) / (n - 1)

def correlation_scratch(x, y):
    return covariance_scratch(x, y) / (std_scratch(x) * std_scratch(y))
```

Actual output, run on `sepal_length` (`x`) and `petal_length` (`y`), n = 150:

```
mean   scratch=5.843333  numpy=5.843333
var    scratch=0.685694  numpy=0.685694
std    scratch=0.828066  numpy=0.828066
corr   scratch=0.871754  numpy=0.871754
```

Exact match to six decimal places — confirming the closed-form definitions and NumPy's
implementation compute the same quantity.

## Practical implementation

`descriptive_statistics.ipynb` walks the whole pipeline on the real Iris dataset (150 flowers, 3
species, loaded via `sklearn.datasets.load_iris` — no network dependency):
- `df.describe()` — mean/std/min/quartiles/max for every numeric column in one call, the
  practical/vectorised analogue of the from-scratch functions above.
- `df[col].skew()` / `df[col].kurt()` — shape statistics, mapped directly to the Skewness/Kurtosis
  formulas above (pandas returns *excess* kurtosis, i.e. already centered so that normal = 0).
- Box plots (`sns.boxplot`) and a custom `detect_outliers_iqr` function implementing Tukey's fence
  exactly as derived in **Mathematical foundation**.
- `df[numeric_cols].corr()` plus a `sns.heatmap` — the Pearson correlation matrix and its visual
  form.
- `df.groupby('species')[numeric_cols].agg(['mean', 'std'])` — grouped statistics, directly
  relevant to avoiding Simpson's paradox (see **Failure modes**).

## Experiment

**Hypothesis:** dividing the sum of squared deviations by `n` (the naive/population formula)
systematically *underestimates* the true variance when computed from small samples, and dividing by
`n-1` (Bessel's correction) removes that bias.

**Setup:** draw 200,000 independent samples of size 5 from a known population `N(50, 5²)` (true
variance = 25). For every sample, compute variance both ways — `/n` and `/(n-1)` — then average
each estimator across all 200,000 trials and compare to the known true variance.

**Actual result** (`descriptive_statistics.ipynb`, "Experiment" section):

```
true population variance (sigma^2):   25.0000
mean of /n   (biased) estimator:      20.0335  (bias = -4.9665, 19.9% too low)
mean of /(n-1) (unbiased) estimator:  25.0419  (bias = 0.0419)
```

**Interpretation:** the `/n` estimator is biased low by ~20% at sample size 5 — a large,
systematic distortion, not sampling noise (200,000 trials averages the noise out). The `/(n-1)`
estimator lands within 0.2% of the true value. The intuition for *why*: a sample's own mean is,
by construction, the point that minimizes the sum of squared deviations for that specific sample —
so squared deviations from the *sample* mean are always slightly smaller than squared deviations
from the (unknown) *true* population mean would be. Dividing by the smaller `n-1` compensates for
that systematic undercount.

**Limitations:** the bias shrinks as sample size grows (it is a small-sample effect — at n=1000 the
two estimators are nearly identical), and this experiment used a normal population; the same
correction applies regardless of the population's shape, since the derivation of Bessel's
correction only relies on linearity of expectation, not normality.

## Failure modes

- **Mean/std misleading on skewed distributions.** For income, house prices, or any right-skewed
  quantity, the mean is pulled toward the long tail and can exceed the median substantially — "the
  average household income" is a classic case where the *median* is the more honest single-number
  summary, and the standard deviation is nearly meaningless as a "typical spread" for skewed data.
- **Mean/std misleading on multimodal distributions.** If a distribution has two separate humps
  (e.g. response times from two different server tiers mixed in one column), the mean lands in the
  valley between the humps — a value that may be *rare or impossible* in the actual data. A single
  mean ± std dev summary implies unimodality it does not have; always plot the histogram before
  trusting the summary numbers.
- **Simpson's paradox.** A trend that appears in several groups of data can reverse when the groups
  are combined. Classic example: a treatment can have a *higher* success rate than a control in
  every age subgroup individually, yet a *lower* overall success rate once the subgroups are pooled,
  because the subgroups have very different sizes. This is exactly why `groupby` statistics
  (Practical implementation, above) must be checked before trusting an aggregate correlation or
  mean — a hidden confounding variable (age, in the example) can flip the sign of an effect.
- **Correlation ≠ causation, and correlation only measures *linear* association.** Two variables can
  have a strong nonlinear relationship (e.g. $y = x^2$ over a symmetric range of $x$) and a Pearson
  correlation near 0 — the formula literally cannot see nonlinear structure. And even a strong,
  real linear correlation says nothing about which variable (if either) causes the other, or
  whether a third variable causes both.
- **Outlier sensitivity compounds across measures.** Mean, standard deviation, variance, and
  Pearson correlation are all built from squared or first-power deviations from the mean — a single
  extreme value distorts all four simultaneously. Median and IQR are robust precisely because they
  ignore the tails.

## Real-world usage

Descriptive statistics are the first step of virtually every data science workflow — before
training any model, `.describe()`, histograms, and box plots are how you catch data-entry errors,
detect skew that motivates a log transform, and spot outliers that need investigation before they
corrupt a mean-based feature. Correlation matrices/heatmaps are a standard first pass at feature
selection (dropping near-duplicate features) and at spotting multicollinearity before linear
regression. Grouped statistics (`groupby`) are the default sanity check before reporting any
aggregate metric, specifically to catch Simpson's-paradox-style reversals in A/B test results or
demographic breakdowns.

## Mental model

A single summary number is a lossy compression of the dataset — mean and std dev alone assume the
data is unimodal and roughly symmetric; if you haven't looked at the shape (histogram, skewness,
box plot), you don't actually know whether that compression is honest.

## Questions to think about

1. You're given `mean = 50, std = 10` for a dataset and told nothing else. What two distributions
   could produce those numbers but tell completely different stories? What single visualization
   would tell them apart?
2. Why does the sample variance formula divide by `n-1` instead of `n`, and why does that
   correction matter more at small sample sizes than large ones? (Tie your answer to the Experiment
   section's actual measured bias.)
3. A company reports "average correlation between marketing spend and revenue across all regions is
   +0.6" but a per-region breakdown shows some regions have negative correlation. What must be true
   for that to happen, and what would you ask to see before trusting the aggregate number?
4. Pearson correlation between two variables comes out to 0.02. Does that mean the variables are
   unrelated? Construct a concrete counterexample.
5. If you only had storage budget for two numbers per dataset (not five, not the full histogram),
   would you pick (mean, std dev) or (median, IQR)? What kind of data would make you regret each
   choice?
