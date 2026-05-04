# Inferential Statistics

Inferential statistics lets you **draw conclusions about a population** from a sample. Instead of describing the data you have (descriptive statistics), you use the sample to make *inferences* about the larger population with a quantified level of certainty.

---

## 1. Population vs Sample

| Concept | Population | Sample |
|---------|-----------|--------|
| Definition | All individuals of interest | A subset drawn from the population |
| Size | N (usually unknown/huge) | n (manageable) |
| Mean | μ (parameter) | x̄ (statistic) |
| Std Dev | σ (parameter) | s (statistic) |

**Sampling methods:** Simple random, stratified, cluster, systematic.

---

## 2. Sampling Distribution

When you take many samples and compute the mean of each, the **distribution of those sample means** is the sampling distribution.

From the **Central Limit Theorem**:
$$\bar{X} \sim \mathcal{N}\!\left(\mu,\ \frac{\sigma^2}{n}\right)$$

**Standard Error (SE):** Standard deviation of the sampling distribution.
$$SE = \frac{\sigma}{\sqrt{n}}$$

As sample size increases, SE decreases — larger samples give more precise estimates.

---

## 3. Point Estimation vs Interval Estimation

| Type | Description | Example |
|------|-------------|---------|
| **Point estimate** | Single value guess | $\bar{x} = 72$ |
| **Interval estimate** | Range of plausible values | $(68.5,\ 75.5)$ with 95% confidence |

---

## 4. Confidence Intervals (CI)

A **95% confidence interval** means: if we repeated the sampling 100 times, approximately 95 of those intervals would contain the true population parameter.

### CI for Population Mean (σ known)
$$\bar{x} \pm z_{\alpha/2} \cdot \frac{\sigma}{\sqrt{n}}$$

### CI for Population Mean (σ unknown, use t-distribution)
$$\bar{x} \pm t_{\alpha/2,\ n-1} \cdot \frac{s}{\sqrt{n}}$$

Common z-values:
| Confidence Level | z (two-tailed) |
|-----------------|----------------|
| 90% | 1.645 |
| 95% | 1.96 |
| 99% | 2.576 |

---

## 5. Hypothesis Testing

A formal framework for deciding whether sample data supports or contradicts a claim about the population.

### Steps
1. **State hypotheses**
   - $H_0$ (Null): Default assumption, no effect
   - $H_1$ / $H_a$ (Alternative): What you want to prove
2. **Choose significance level α** (typically 0.05)
3. **Collect data, compute test statistic**
4. **Compute p-value** (or compare test statistic to critical value)
5. **Make decision**

### Decision Rule
- If **p-value < α** → Reject H₀ (result is statistically significant)
- If **p-value ≥ α** → Fail to reject H₀

### Type I & Type II Errors

|  | H₀ True | H₀ False |
|--|---------|----------|
| **Reject H₀** | Type I Error (False Positive) α | Correct (Power) |
| **Fail to reject H₀** | Correct | Type II Error (False Negative) β |

- **α** = P(Type I Error) = significance level
- **β** = P(Type II Error)
- **Power** = 1 − β = probability of correctly rejecting a false H₀

---

## 6. Common Statistical Tests

### One-Sample t-Test
Tests whether the sample mean differs from a known/hypothesised value.
$$t = \frac{\bar{x} - \mu_0}{s/\sqrt{n}}, \quad df = n - 1$$

### Independent (Two-Sample) t-Test
Tests whether two independent groups have different means.
$$t = \frac{\bar{x}_1 - \bar{x}_2}{\sqrt{s_p^2(1/n_1 + 1/n_2)}}$$

### Paired t-Test
Tests the mean difference in **paired/matched** observations (before vs after).

### ANOVA (Analysis of Variance)
Tests whether **3+ group means** are all equal.
$$F = \frac{\text{Between-group variance}}{\text{Within-group variance}}$$
- If F is large → at least one group mean differs (but doesn't say which one)
- Post-hoc tests (Tukey, Bonferroni) find where differences lie.

### Chi-Square Test (χ²)
Tests for associations between **categorical variables**.
$$\chi^2 = \sum \frac{(O - E)^2}{E}$$
- O = Observed frequency, E = Expected frequency

---

## 7. p-Value Intuition

> The p-value is the probability of observing results *at least as extreme* as the sample data, **assuming H₀ is true**.

- p = 0.03 → "If H₀ were true, there's only a 3% chance of seeing this result by chance."
- Small p-value → evidence against H₀ → reject H₀.

**Common misconceptions:**
- p-value is NOT the probability that H₀ is true.
- Statistical significance ≠ practical significance.
- Always report **effect size** (Cohen's d, η²) alongside p-values.

---

## 8. Effect Size

Measures the **magnitude** of an effect, independent of sample size.

| Measure | Formula | Used with |
|---------|---------|-----------|
| Cohen's d | $d = \frac{\bar{x}_1 - \bar{x}_2}{s_{pooled}}$ | t-tests |
| η² (eta squared) | $\eta^2 = SS_{between}/SS_{total}$ | ANOVA |
| Cramér's V | $V = \sqrt{\chi^2 / (n \cdot \min(r-1, c-1))}$ | Chi-square |

Cohen's d conventions: 0.2 = small, 0.5 = medium, 0.8 = large.

---

## 9. Assumptions to Check

| Test | Assumptions |
|------|------------|
| t-test | Normality of data (or n ≥ 30), homogeneity of variance |
| ANOVA | Normality, equal variances (Levene's test), independence |
| Chi-square | Expected cell counts ≥ 5, independence |

Normality tests: Shapiro-Wilk (n < 50), Kolmogorov-Smirnov, Q-Q plot.

---

## Quick Reference: Which Test?

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
