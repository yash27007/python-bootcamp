# Probability

## What you'll learn

The mathematical framework for quantifying uncertainty: probability axioms and rules, conditional
probability and Bayes' theorem, random variables and their distributions (Bernoulli, Binomial,
Poisson, Uniform, Normal, Exponential), and the two theorems — the Law of Large Numbers and the
Central Limit Theorem — that explain why aggregates of uncertain events become predictable even
when individual events aren't.

## Why it matters

Bayes' theorem is the base-rate fallacy made rigorous: a "99% accurate" medical test applied to a
rare condition is still close to a coin flip on a positive result, and getting that wrong has real
consequences for any classifier, alert system, or diagnostic tool evaluated only on accuracy. The
CLT is why almost every classical hypothesis test in `03-inferential-statistics` is allowed to use a
normal or t-distribution, regardless of the population's actual shape.

## Prerequisites

- `01-descriptive-statistics` (mean, variance — reused directly in the random-variable formulas
  here)

## What you'll build

- A 5,000,000-person Monte Carlo simulation of the medical-test / Bayes'-theorem scenario, verifying
  the empirical posterior probability against the closed-form formula (0.4991 simulated vs 0.5000
  theoretical)

See [`notes.md`](notes.md) for the full write-up including real captured output, and
[`probability.ipynb`](probability.ipynb) (all cells executed) for the practical tour — Law of Large
Numbers via coin-flip simulation, `scipy.stats` discrete/continuous distributions, and the Central
Limit Theorem demonstrated by sampling repeatedly from a deliberately non-normal (uniform)
population.

## Where it shows up in real systems

Naive Bayes classifiers are a direct application of Bayes' theorem. Logistic regression models
$P(Y=1|X)$ — a conditional probability — making it a probabilistic model by construction. Gaussian
assumptions underlie Linear Discriminant Analysis and the residual-normality assumption behind
ordinary least squares. Maximum Likelihood Estimation, introduced here, is the training objective
behind logistic regression and (via cross-entropy loss) most deep learning classifiers.

## What's next

`03-inferential-statistics` — using the sampling distribution and CLT from this topic to draw
conclusions about a population from a sample, with a quantified level of certainty.
