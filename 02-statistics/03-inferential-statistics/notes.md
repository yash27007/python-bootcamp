# Inferential Statistics

## Problem

You can never measure an entire population — every voter, every widget off a production line,
every user of a product. You can only ever observe a sample. Descriptive statistics
(`01-descriptive-statistics`) can summarise *that sample*, but the real question is almost always
about the population: does this drug actually work, is this website redesign actually better, is
this factory actually producing 500g widgets on average? Inferential statistics lets you **draw
conclusions about a population** from a sample, with a quantified level of certainty about how
wrong that conclusion might be.

## Intuition

Take 20 widgets off a production line and weigh them. Their average is never going to be *exactly*
500g even if the true long-run average is 500g — every sample has sampling noise. The core question
inferential statistics answers is: given that noise, how far can the sample mean plausibly be from
the true population mean, and is the difference we see large enough to conclude something real is
going on, rather than just noise?

### Population vs sample

| Concept | Population | Sample |
|---------|-----------|--------|
| Definition | All individuals of interest | A subset drawn from the population |
| Size | N (usually unknown/huge) | n (manageable) |
| Mean | μ (parameter) | x̄ (statistic) |
| Std Dev | σ (parameter) | s (statistic) |

**Sampling methods:** simple random, stratified, cluster, systematic — the sample must represent
the population, or every conclusion below is about the wrong thing.

### Point estimation vs interval estimation

| Type | Description | Example |
|------|-------------|---------|
| **Point estimate** | Single value guess | $\bar{x} = 72$ |
| **Interval estimate** | Range of plausible values | $(68.5,\ 75.5)$ with 95% confidence |

A point estimate throws away the uncertainty information entirely — "the mean is 72" sounds exact
but a single sample statistic is almost never exactly the population parameter. An interval
estimate keeps that uncertainty explicit.

## Why simpler approaches fail

The simplest possible inference is "compute the sample mean and treat it as the population mean" —
a point estimate with no uncertainty attached. That fails because a different sample from the same
population would give a different mean; without quantifying *how much* it would vary, you can't
tell a real effect from noise. The next-simplest fix — "eyeball whether two sample means look
different" — fails for the same reason: two samples from *identical* populations will almost never
have identical means just by chance, so "they look different" is not evidence of anything without a
formal measure of how different they'd be expected to look under pure chance. That's exactly what
the sampling distribution, standard error, confidence intervals, and hypothesis testing formalize:
turning "they look different" into "here's the probability that a difference this large would occur
if there were truly no difference."

## Mathematical foundation

### Sampling distribution

When you take many samples and compute the mean of each, the **distribution of those sample means**
is the sampling distribution. From the Central Limit Theorem (`02-probability`):
$$\bar{X} \sim \mathcal{N}\!\left(\mu,\ \frac{\sigma^2}{n}\right)$$

**Standard error (SE)** — standard deviation of the sampling distribution:
$$SE = \frac{\sigma}{\sqrt{n}}$$

As sample size increases, SE decreases — larger samples give more precise estimates.

### Confidence intervals (CI)

A **95% confidence interval** means: if we repeated the sampling process 100 times and built 100
such intervals, approximately 95 of those intervals would contain the true population parameter.
(It does *not* mean "there is a 95% chance the true value is in this one interval" — see
**Failure modes**.)

**CI for population mean (σ known)**
$$\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

**CI for population mean (σ unknown, use t-distribution)**
$$\bar{x} \pm t_{\alpha/2,\ n-1} \cdot \frac{s}{\sqrt{n}}$$

Common z-values:

| Confidence Level | z (two-tailed) |
|-----------------|----------------|
| 90% | 1.645 |
| 95% | 1.96 |
| 99% | 2.576 |

### Hypothesis testing

A formal framework for deciding whether sample data supports or contradicts a claim about the
population.

**Steps**
1. **State hypotheses** — $H_0$ (Null): default assumption, no effect. $H_1$/$H_a$ (Alternative):
   what you want to prove.
2. **Choose significance level α** (typically 0.05).
3. **Collect data, compute test statistic.**
4. **Compute p-value** (or compare test statistic to critical value).
5. **Make decision.**

**Decision rule**
- If **p-value < α** → Reject H₀ (result is statistically significant)
- If **p-value ≥ α** → Fail to reject H₀

**Type I & Type II errors**

|  | H₀ True | H₀ False |
|--|---------|----------|
| **Reject H₀** | Type I Error (False Positive) α | Correct (Power) |
| **Fail to reject H₀** | Correct | Type II Error (False Negative) β |

- **α** = P(Type I Error) = significance level
- **β** = P(Type II Error)
- **Power** = 1 − β = probability of correctly rejecting a false H₀

### Common statistical tests

**One-sample t-test** — tests whether the sample mean differs from a known/hypothesised value.
$$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}, \quad df = n - 1$$

**Independent (two-sample) t-test** — tests whether two independent groups have different means.
$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_p^2(1/n_1 + 1/n_2)}}$$

**Paired t-test** — tests the mean difference in **paired/matched** observations (before vs after).

**ANOVA (Analysis of Variance)** — tests whether **3+ group means** are all equal.
$$F = \frac{\text{Between-group variance}}{\text{Within-group variance}}$$
If F is large → at least one group mean differs (but doesn't say which one). Post-hoc tests (Tukey,
Bonferroni) find where differences lie.

**Chi-square test (χ²)** — tests for associations between **categorical variables**.
$$\chi^2 = \sum \frac{(O - E)^2}{E}$$
O = Observed frequency, E = Expected frequency.

### p-value intuition

> The p-value is the probability of observing results *at least as extreme* as the sample data,
> **assuming H₀ is true**.

- p = 0.03 → "If H₀ were true, there's only a 3% chance of seeing this result by chance."
- Small p-value → evidence against H₀ → reject H₀.

**Common misconceptions** (see **Failure modes** for the full treatment):
- p-value is NOT the probability that H₀ is true.
- Statistical significance ≠ practical significance.
- Always report **effect size** (Cohen's d, η²) alongside p-values.

### Effect size

Measures the **magnitude** of an effect, independent of sample size.

| Measure | Formula | Used with |
|---------|---------|-----------|
| Cohen's d | $d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}$ | t-tests |
| η² (eta squared) | $\eta^2 = SS_{between}/SS_{total}$ | ANOVA |
| Cramér's V | $V = \sqrt{\chi^2 / (n \cdot \min(r-1, c-1))}$ | Chi-square |

Cohen's d conventions: 0.2 = small, 0.5 = medium, 0.8 = large.

### Assumptions to check

| Test | Assumptions |
|------|------------|
| t-test | Normality of data (or n ≥ 30), homogeneity of variance |
| ANOVA | Normality, equal variances (Levene's test), independence |
| Chi-square | Expected cell counts ≥ 5, independence |

Normality tests: Shapiro-Wilk (n < 50), Kolmogorov-Smirnov, Q-Q plot.

## Algorithm

**Quick reference: which test?**

```
Is the outcome variable numeric or categorical?
├── Numeric
│   ├── 1 group vs known value → One-sample t-test
│   ├── 2 independent groups   → Independent t-test
│   ├── 2 paired/matched groups → Paired t-test
│   └── 3+ groups              → ANOVA
└── Categorical
    └── Association between two categorical vars → Chi-square test
```

General procedure for any inference:
1. Decide point estimate vs interval estimate vs formal hypothesis test based on the question
   ("what's the value" vs "is there a difference").
2. Check the relevant assumptions table above (normality, variance homogeneity, expected cell
   counts) before trusting a parametric test's p-value.
3. If assumptions are shaky and the sample is small, prefer a resampling-based method (bootstrap
   CI, permutation test — see **From-scratch implementation**) that makes no distributional
   assumption.
4. Compute the test statistic and p-value; compare to α.
5. Always compute and report effect size alongside the p-value — a significant result with a tiny
   effect size may not matter practically.

## From-scratch implementation

Two resampling-based methods, implemented directly (no `scipy.stats` formula), each compared
against the equivalent parametric method on the same real data (`inferential_statistics.ipynb`).

**1. Bootstrap confidence interval for the mean**, vs the parametric t-based CI, on the 20 widget
weights from the one-sample-t-test example:

```python
n_boot = 100_000
boot_means = np.empty(n_boot)
for i in range(n_boot):
    resample = rng.choice(weights, size=len(weights), replace=True)
    boot_means[i] = resample.mean()

boot_lo, boot_hi = np.percentile(boot_means, [2.5, 97.5])
```

Actual output:

```
sample mean:                    499.700 g
bootstrap 95% CI (percentile):  (498.500, 500.950)
parametric t-based 95% CI:      (498.358, 501.042)
```

No normal/t-distribution assumption anywhere — just resample the data itself with replacement
100,000 times and read the interval directly off the resulting distribution of means. The bootstrap
interval (498.500, 500.950) closely tracks the parametric interval (498.358, 501.042); both contain
500g, consistent with failing to reject the factory's claim.

**2. Permutation test**, vs the independent two-sample t-test, on the Group A / Group B exam scores:

```python
observed_diff = group_a.mean() - group_b.mean()
pooled = np.concatenate([group_a, group_b])
n_a = len(group_a)
count_extreme = 0
for i in range(n_perm):          # n_perm = 100_000
    shuffled = rng.permutation(pooled)
    perm_a, perm_b = shuffled[:n_a], shuffled[n_a:]
    diff = perm_a.mean() - perm_b.mean()
    if abs(diff) >= abs(observed_diff):
        count_extreme += 1
perm_p_value = count_extreme / n_perm
```

Actual output:

```
observed mean difference (A - B): 7.300
permutation test p-value (two-sided, 100000 shuffles): 0.00012
independent t-test p-value:                        0.00004
```

The logic: under $H_0$ (no real difference between groups), the "A" and "B" labels are
interchangeable — so randomly reshuffling which values are labeled A and which are B, and checking
how often that produces a difference at least as extreme as the one actually observed, *is* the
p-value, by definition, with zero distributional assumptions. The permutation p-value (0.00012) and
the t-test p-value (0.00004) agree in order of magnitude and conclusion (reject $H_0$ at α = 0.05);
they aren't identical because the t-test additionally assumes normally-distributed data, while the
permutation test uses only the data's actual empirical distribution.

## Practical implementation

`inferential_statistics.ipynb` walks the parametric methods end to end, each directly implementing
a formula above:
- **Confidence intervals** — `stats.t.interval()` on simulated height samples, plus a visualization
  drawing 20 independent 95% CIs and showing how many actually capture the true population mean
  (a direct, visual demonstration of what "95% confidence" means across repeated sampling — see
  **Failure modes** for what it does *not* mean).
- **One-sample t-test** — `stats.ttest_1samp()` on the factory-widget-weight scenario.
- **Independent two-sample t-test** — `stats.ttest_ind()`, preceded by Levene's test to choose
  `equal_var`, plus Cohen's d for effect size.
- **Paired t-test** — `stats.ttest_rel()` on before/after blood-pressure measurements.
- **ANOVA** — `stats.f_oneway()` on three fertiliser groups, visualised with a box plot.
- **Chi-square test** — `stats.chi2_contingency()` on a gender × product-preference contingency
  table.
- **Normality testing** — Q-Q plots and `stats.shapiro()`, the practical check for the "Assumptions
  to check" table above.
- **p-value visualisation** — the t-distribution's rejection region shaded directly, making the
  p-value definition ("probability of a result this extreme, assuming H₀") visible as an area under
  the curve.

## Experiment

The bootstrap-vs-parametric-CI and permutation-vs-t-test comparisons above **are** this topic's
experiments. Both share the same structure: hypothesis (a distribution-free resampling method
should closely approximate the parametric method's answer on data that roughly satisfies the
parametric method's assumptions), setup (real small samples, 100,000 resamples/permutations),
result (bootstrap CI (498.500, 500.950) vs parametric (498.358, 501.042); permutation p = 0.00012
vs t-test p = 0.00004), and interpretation (close agreement validates both methods on this data;
resampling methods matter most precisely when the parametric assumptions — normality, known
variance structure — are in doubt, which this particular dataset satisfies well enough that the two
approaches nearly coincide). **Limitation:** with only n=20 (weights) and n=10-per-group (A/B
scores), both experiments are small-sample demonstrations; the resampling methods' real advantage
shows up more clearly on data that visibly violates normality, which these examples were not
designed to do.

## Failure modes

- **Misinterpreting the confidence interval.** A 95% CI does **not** mean "there is a 95%
  probability that the true parameter lies in this specific interval." The true parameter is a
  fixed (if unknown) number — it either is or isn't in any given interval, with probability 1 or 0.
  The correct reading is about the *procedure*: if you repeated the sampling and interval-building
  process many times, about 95% of the resulting intervals would contain the true value. The
  simulation in **Practical implementation** (20 CIs, counting how many capture the true mean) makes
  this concrete — the 95% is a property of the long-run procedure, not of any one interval.
- **p-hacking.** Running many tests, trying many variable subsets or cutoffs, and reporting only
  the analysis that produced p < 0.05, inflates the true false-positive rate far above the nominal
  α. If you try 20 independent hypotheses at α = 0.05, you expect about one "significant" result by
  chance alone even if nothing real is going on — reporting only that one result as "the finding" is
  p-hacking, whether done deliberately or by unconsciously trying several analyses until one looks
  good.
- **The multiple-comparisons problem** — the systematic version of p-hacking. Running $m$
  independent tests at α = 0.05 each gives a family-wise error rate of roughly
  $1 - (1-0.05)^m$, which already exceeds 40% at $m=10$. Corrections (Bonferroni: use α/m per test;
  Tukey's HSD for post-hoc ANOVA comparisons) exist specifically to control this — this is exactly
  why ANOVA's F-test says only "at least one group differs" and requires a *separate*,
  correction-aware post-hoc test to say *which* groups differ, rather than just running pairwise
  t-tests on every pair.
- **Statistical significance ≠ practical significance.** With a large enough sample, even a
  practically meaningless difference (e.g. 0.01g heavier widgets) will produce a very small
  p-value. This is why effect size (Cohen's d, η², Cramér's V) must always be reported alongside a
  p-value — the p-value answers "is there evidence of an effect," and effect size answers "does the
  effect matter."
- **Treating "fail to reject H₀" as "H₀ is proven true."** A non-significant result means the data
  didn't provide strong enough evidence to reject the null — it does not mean the null is
  confirmed. Low statistical power (small sample, small true effect) can produce "fail to reject"
  even when a real effect exists (Type II error).

## Real-world usage

A/B testing in product and marketing is applied hypothesis testing end to end — confidence
intervals on conversion-rate lift, t-tests or chi-square tests on whether a change moved a metric,
and effect sizes to decide whether a statistically significant lift is big enough to ship. Clinical
trials use paired and independent t-tests (before/after, treatment/control) with pre-registered
significance levels specifically to guard against p-hacking. ANOVA is standard in experimental
design whenever more than two treatment groups are compared (e.g. multiple ad creatives, multiple
drug dosages). Chi-square tests underlie categorical association checks in survey analysis and
feature-selection screening for categorical variables. Bootstrap methods are the default when a
metric's sampling distribution has no clean closed form (e.g. a ratio of two random quantities, or
a machine learning model's cross-validated accuracy) — resample the data, recompute the metric each
time, and read the interval off the resulting distribution, exactly as in the widget-weight example
above.

## Mental model

Inference is not "what did I measure" (that's descriptive statistics) — it's "how much would this
measurement vary if I repeated the whole sampling process," and every tool in this topic (standard
error, confidence interval, p-value, effect size) is a different way of quantifying that variability
so a difference can be judged real versus noise.

## Questions to think about

1. Explain, without using the word "probability" applied to the parameter itself, what a 95%
   confidence interval actually guarantees. Why is "95% chance the true value is in this interval"
   wrong?
2. A researcher runs 15 different hypothesis tests on subsets of the same dataset and reports the 2
   that came back p < 0.05 as "the finding." What's wrong with this, and what would the expected
   number of false positives be under pure chance at α = 0.05 with 15 independent tests?
3. A sample of 1,000,000 users shows a new button color increases click rate by 0.02 percentage
   points with p < 0.0001. Should the company ship the change? What additional number would you ask
   for before deciding, and why does a large sample make this question harder rather than easier to
   answer from the p-value alone?
4. Why does the bootstrap confidence interval in this topic's Experiment not require assuming the
   data is normally distributed, while the parametric t-based interval does? In what situation would
   you expect the two intervals to diverge more than they did here?
5. ANOVA's F-test rejects $H_0$ for three fertiliser groups. Why can't you conclude "fertiliser A
   differs from fertiliser B" directly from that result, and what do you need to run next?
