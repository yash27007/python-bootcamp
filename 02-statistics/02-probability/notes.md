# Probability

## Problem

Real data is noisy: the same coin flipped twice doesn't land the same way twice, the same medical
test gives different results on different patients, the same model prediction is right sometimes
and wrong others. Any system that has to make decisions under that noise — a spam filter, a medical
diagnosis, an A/B test — needs a rigorous way to reason about "how likely" something
is, and to combine multiple uncertain pieces of evidence into one answer. Probability is the
mathematical framework for **quantifying uncertainty**. It is the backbone of statistical inference,
machine learning algorithms, and data science.

## Intuition

Before any formula: probability assigns a number between 0 and 1 to "how often would this happen if
we repeated the situation many times" (or, in the Bayesian reading, "how much should I believe
this"). Flip a fair coin many times and the fraction of heads settles near 0.5 — not because any
individual flip is predictable, but because the *long-run average* is. That settling-down behavior
is the Law of Large Numbers, and it's the intuitive foundation everything else in this topic builds
on: individual outcomes are unpredictable, but their aggregate behavior is not.

### Core definitions

| Term | Definition |
|------|-----------|
| **Experiment** | Any process that produces an outcome |
| **Sample Space (Ω)** | The set of all possible outcomes |
| **Event (E)** | A subset of the sample space |
| **Probability P(E)** | A number in [0, 1] measuring likelihood |

### Probability axioms (Kolmogorov)

1. $P(E) \geq 0$ for every event E
2. $P(\Omega) = 1$ (something must happen)
3. For mutually exclusive events: $P(A \cup B) = P(A) + P(B)$

Everything else in probability theory — every rule below — is derivable from these three axioms.

## Why simpler approaches fail

The naive approach to uncertainty is "estimate a single most-likely outcome and act on it" — but
that throws away exactly the information that matters when outcomes are rare or asymmetric in
cost. The medical test example makes this concrete: a test with 99% accuracy sounds like "trust it,"
but applied to a rare disease (1% prevalence), a positive result is still only ~50% likely to be a
true positive (see **Mathematical foundation** → Bayes' theorem, and the from-scratch simulation
below). Reasoning about the *single most likely outcome only* would say "the test is 99% accurate,
trust a positive" — dangerously wrong. You need the *full conditional structure* (prior, likelihood,
posterior), not a point estimate, and that structure is exactly what Bayes' theorem formalizes.
Likewise, treating dependent events as if they were independent (a simpler assumption) gives
wrong answers whenever real-world variables are actually correlated — see **Failure modes**.

## Mathematical foundation

### Rules of probability

**Complement rule**
$$P(A^c) = 1 - P(A)$$

**Addition rule**
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$
For **mutually exclusive** events: $P(A \cup B) = P(A) + P(B)$

**Multiplication rule**
$$P(A \cap B) = P(A) \cdot P(B|A)$$
For **independent** events: $P(A \cap B) = P(A) \cdot P(B)$

### Conditional probability

$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

Read: "Probability of A given B has occurred."

**Independence.** Events A and B are independent if:
$$P(A|B) = P(A) \quad \Leftrightarrow \quad P(A \cap B) = P(A) \cdot P(B)$$

### Bayes' theorem

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

| Term | Name |
|------|------|
| $P(A)$ | **Prior** – belief before seeing data |
| $P(B\|A)$ | **Likelihood** – probability of data given hypothesis |
| $P(A\|B)$ | **Posterior** – updated belief after seeing data |
| $P(B)$ | **Marginal likelihood** (normalising constant) |

**Worked example:** Medical test with 99% accuracy, disease prevalence 1%. Even a positive test
means only ~50% chance you have the disease — this is the **base-rate fallacy**, verified by
simulation in **From-scratch implementation** below.

### Random variables

A **random variable** X maps outcomes of an experiment to numbers.

| Type | Description | Example |
|------|-------------|---------|
| **Discrete** | Countable values | Number of heads in 10 flips |
| **Continuous** | Uncountable, interval of values | Height of a person |

**Expected value (mean)**
$$E[X] = \sum x \cdot P(X=x) \quad \text{(discrete)}$$
$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x)\, dx \quad \text{(continuous)}$$

**Variance**
$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

### Common probability distributions

**Discrete**

*Bernoulli* — single trial with success probability *p*.
$$P(X=1) = p, \quad P(X=0) = 1-p$$
$E[X] = p$, $\text{Var}(X) = p(1-p)$

*Binomial* — *n* independent Bernoulli trials, each with probability *p*.
$$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$$
$E[X] = np$, $\text{Var}(X) = np(1-p)$

*Poisson* — number of events in a fixed interval, with average rate λ.
$$P(X=k) = \frac{e^{-\lambda}\lambda^k}{k!}$$
$E[X] = \lambda = \text{Var}(X)$

**Continuous**

*Uniform* — all values in [a, b] equally likely.
$$f(x) = \frac{1}{b-a}, \quad x \in [a,b]$$

*Normal (Gaussian)*
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
Notation: $X \sim \mathcal{N}(\mu, \sigma^2)$. 68–95–99.7 rule: 68% within 1σ, 95% within 2σ, 99.7%
within 3σ.

*Standard normal*
$$Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$$

*Exponential* — time between Poisson events.
$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

### Central Limit Theorem (CLT)

> Regardless of the underlying distribution, the **sample mean** of a sufficiently large sample
> (n ≥ 30) follows an approximately **normal distribution**.

$$\bar{X} \sim \mathcal{N}\!\left(\mu, \frac{\sigma^2}{n}\right)$$

This is why the normal distribution is so important in statistics — it justifies the use of many
parametric tests (see `03-inferential-statistics`).

### Law of Large Numbers

> As the number of trials increases, the sample mean converges to the true population mean.

$$\bar{X}_n \xrightarrow{p} \mu \quad \text{as } n \to \infty$$

## Algorithm

To answer "what's the probability of X given evidence Y":
1. Identify the sample space and whether the event of interest is a simple event or a compound one
   (union, intersection, complement).
2. If conditioning on evidence: identify the prior $P(A)$, the likelihood $P(B|A)$, and whether you
   need the full Bayes' theorem or a simpler conditional-probability formula.
3. Check independence explicitly — don't assume it. If $P(A \cap B) \ne P(A)P(B)$ empirically,
   treating the events as independent will give a wrong compound probability.
4. For a random variable: identify whether it's discrete or continuous, and which named
   distribution (if any) models it, to get closed-form $E[X]$/$\text{Var}(X)$ instead of estimating
   them numerically.
5. When aggregating across many independent trials (a sample mean, a sum of many random
   contributions), invoke the CLT — the aggregate is approximately normal even if each individual
   trial is not.

## From-scratch implementation

Rather than trust the Bayes' theorem algebra in the abstract, simulate the medical-test scenario
directly and check that the empirical result matches the formula (`probability.ipynb`,
"From-Scratch" section):

```python
rng = np.random.default_rng(0)
n_people = 5_000_000
sensitivity = 0.99   # P(+ | disease)
specificity = 0.99   # P(- | no disease)

disease = rng.random(n_people) < p_disease          # p_disease = 0.01
test_pos = np.where(
    disease,
    rng.random(n_people) < sensitivity,
    rng.random(n_people) < (1 - specificity),
)

empirical_posterior = disease[test_pos].mean()
```

Actual output:

```
simulated people:                 5000000
positive tests observed:          98854
Monte Carlo  P(disease | +test):  0.4991
Bayes formula P(disease | +test): 0.5000
absolute difference:              0.00087
```

Five million simulated people, no formula assumed — just generate the disease status, generate the
test result conditioned on that status, and directly count. The empirical fraction (0.4991) matches
the closed-form Bayes' theorem answer (0.5000) to within simulation noise, confirming the base-rate
fallacy is real and not an algebraic trick: even a "99% accurate" test is closer to a coin flip than
a diagnosis when the underlying condition is rare.

## Practical implementation

`probability.ipynb` covers the practical, `scipy.stats`-based tour, each piece mapped back to the
math above:
- **Law of Large Numbers**, coin-flip simulation — the running proportion of heads converging to
  0.5 as `n` grows, a direct visual of the theorem in **Mathematical foundation**.
- **Bayes' theorem**, computed directly with the prior/likelihood/total-probability formula (the
  closed-form counterpart to the Monte Carlo simulation above).
- **Discrete distributions** — `scipy.stats.binom`, `scipy.stats.poisson`, with PMFs plotted and
  mean/std dev read off the distribution objects (matching the $E[X]$/$\text{Var}(X)$ formulas
  above).
- **Continuous distributions** — `scipy.stats.norm`, with PDF/CDF plotted, ±1/2/3σ regions shaded
  to make the 68–95–99.7 rule visible, and `.cdf()` used to compute real interval probabilities and
  z-scores.
- **CLT**, demonstrated by drawing repeated samples from a **uniform** population (deliberately
  non-normal) at increasing sample sizes (5, 10, 30, 100) and showing the sample-mean distribution
  becoming visibly more normal as `n` grows — the practical, visual counterpart to the CLT formula.
- Several right-skewed continuous distributions (exponential, gamma, chi-squared) plotted
  side by side for comparison.

## Experiment

The Monte Carlo Bayes' theorem verification above **is** this topic's experiment: hypothesis
(simulated posterior will match the closed-form Bayes' theorem answer), setup (5,000,000 simulated
people, known prior/sensitivity/specificity), actual result (0.4991 vs 0.5000, difference 0.00087),
and interpretation (the base-rate fallacy is a real, measurable phenomenon, not just an algebra
trick). Limitation: this verifies one instance (99%/99%/1% prevalence) — the qualitative conclusion
(a "highly accurate" test can still be a coin flip on a positive result) generalizes to any low
base-rate condition, but the exact posterior probability changes with the specific numbers.

## Failure modes

- **Base-rate neglect.** People (and naive systems) systematically ignore the prior probability
  $P(A)$ and over-weight the likelihood $P(B|A)$ — exactly the medical-test scenario above. Any
  alert system, fraud detector, or classifier evaluated only on accuracy/sensitivity without
  accounting for how rare the positive class actually is will produce a flood of false positives
  once deployed on real, imbalanced data.
- **Conflating correlation with causation.** Two variables being statistically dependent
  ($P(A \cap B) \ne P(A)P(B)$) says nothing about which one causes the other, or whether a third,
  unobserved variable causes both. Probability theory alone — without a causal model or a
  randomized experiment — cannot distinguish "A causes B," "B causes A," and "C causes both A and
  B."
- **Misapplying independence.** The multiplication rule $P(A \cap B) = P(A) \cdot P(B)$ only holds
  for genuinely independent events; assuming independence when events are actually correlated
  (e.g. treating two sensor readings from the same faulty device as independent evidence) will
  systematically overstate confidence — compounding many "independent" pieces of correlated
  evidence produces false certainty far more often than the math predicts.
- **The CLT requires a large enough sample and finite variance.** For n well below 30, or for
  heavy-tailed distributions without finite variance (e.g. certain power laws), the sample mean is
  not well-approximated by a normal distribution — using CLT-justified methods (t-tests,
  z-intervals) anyway silently produces overconfident results.

## Real-world usage

Bayes' theorem underlies spam filters (Naive Bayes classifiers), medical diagnostic reasoning, and
Bayesian A/B testing. Logistic regression directly models $P(Y=1|X)$ — a conditional probability —
making it a probabilistic model, not just a curve-fit. Gaussian-distribution assumptions underlie
many classical ML algorithms (Linear Discriminant Analysis, the residual-normality assumption
behind ordinary least squares' confidence intervals). Maximum Likelihood Estimation (MLE) — finding
the parameters that make the observed data most probable under a model — is the training objective
behind logistic regression, and (via cross-entropy loss) most of deep learning classification. The
CLT is why almost every classical hypothesis test (`03-inferential-statistics`) is justified to use
a normal or t-distribution for the sampling distribution of a mean, regardless of what the
underlying population looks like.

## Mental model

Probability is not about predicting one outcome — it's a formal accounting system for uncertainty
that lets you combine a prior belief with new evidence (Bayes' theorem) and know how much the
aggregate of many uncertain events will converge to a predictable pattern (Law of Large
Numbers/CLT), even when no individual event is predictable.

## Questions to think about

1. A test is "99% accurate" — why is that phrase alone not enough to compute $P(\text{disease} |
   +\text{test})$? What two additional numbers do you need, and why does prevalence matter as much
   as accuracy?
2. If you flip a fair coin 10 times and get 8 heads, does the coin "owe" you more tails on the next
   10 flips to balance out ("gambler's fallacy")? What does the Law of Large Numbers actually
   guarantee, and what does it not guarantee?
3. Two sensors on the same physical device report readings that are correlated due to a shared
   power supply issue, but you treat them as independent evidence when computing a combined
   confidence score. In which direction does your combined confidence become wrong — overconfident
   or underconfident — and why?
4. The CLT says sample means become approximately normal for large n "regardless of the underlying
   distribution." Construct a scenario (or name a real-world quantity) where you'd be skeptical of
   applying the CLT even with n = 50, and explain what property of the underlying distribution
   breaks the guarantee.
5. Why does Bayes' theorem require dividing by $P(B)$ (the marginal likelihood) rather than just
   using $P(B|A) \cdot P(A)$ directly as the answer? What would go wrong numerically if you skipped
   that normalization?
