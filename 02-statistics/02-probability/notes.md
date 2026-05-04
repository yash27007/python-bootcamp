# Probability

Probability is the mathematical framework for **quantifying uncertainty**. It is the backbone of statistical inference, machine learning algorithms, and data science.

---

## 1. Core Definitions

| Term | Definition |
|------|-----------|
| **Experiment** | Any process that produces an outcome |
| **Sample Space (Ω)** | The set of all possible outcomes |
| **Event (E)** | A subset of the sample space |
| **Probability P(E)** | A number in [0, 1] measuring likelihood |

### Probability Axioms (Kolmogorov)
1. $P(E) \geq 0$ for every event E
2. $P(\Omega) = 1$ (something must happen)
3. For mutually exclusive events: $P(A \cup B) = P(A) + P(B)$

---

## 2. Rules of Probability

### Complement Rule
$$P(A^c) = 1 - P(A)$$

### Addition Rule
$$P(A \cup B) = P(A) + P(B) - P(A \cap B)$$
For **mutually exclusive** events: $P(A \cup B) = P(A) + P(B)$

### Multiplication Rule
$$P(A \cap B) = P(A) \cdot P(B|A)$$
For **independent** events: $P(A \cap B) = P(A) \cdot P(B)$

---

## 3. Conditional Probability

$$P(A|B) = \frac{P(A \cap B)}{P(B)}, \quad P(B) > 0$$

Read: "Probability of A given B has occurred."

### Independence
Events A and B are **independent** if:
$$P(A|B) = P(A) \quad \Leftrightarrow \quad P(A \cap B) = P(A) \cdot P(B)$$

---

## 4. Bayes' Theorem

$$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$

| Term | Name |
|------|------|
| $P(A)$ | **Prior** – belief before seeing data |
| $P(B\|A)$ | **Likelihood** – probability of data given hypothesis |
| $P(A\|B)$ | **Posterior** – updated belief after seeing data |
| $P(B)$ | **Marginal likelihood** (normalising constant) |

**Example:** Medical test with 99% accuracy, disease prevalence 1%.  
Even a positive test means only ~50% chance you have the disease — this is the **base rate fallacy**.

---

## 5. Random Variables

A **random variable** X maps outcomes of an experiment to numbers.

| Type | Description | Example |
|------|-------------|---------|
| **Discrete** | Countable values | Number of heads in 10 flips |
| **Continuous** | Uncountable, interval of values | Height of a person |

### Expected Value (Mean)
$$E[X] = \sum x \cdot P(X=x) \quad \text{(discrete)}$$
$$E[X] = \int_{-\infty}^{\infty} x \cdot f(x)\, dx \quad \text{(continuous)}$$

### Variance
$$\text{Var}(X) = E[(X - \mu)^2] = E[X^2] - (E[X])^2$$

---

## 6. Common Probability Distributions

### Discrete Distributions

#### Bernoulli
Single trial with success probability *p*.
$$P(X=1) = p, \quad P(X=0) = 1-p$$
- $E[X] = p$, $\text{Var}(X) = p(1-p)$

#### Binomial
*n* independent Bernoulli trials, each with probability *p*.
$$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}$$
- $E[X] = np$, $\text{Var}(X) = np(1-p)$

#### Poisson
Number of events in a fixed interval, with average rate λ.
$$P(X=k) = \frac{e^{-\lambda}\lambda^k}{k!}$$
- $E[X] = \lambda = \text{Var}(X)$

### Continuous Distributions

#### Uniform
All values in [a, b] equally likely.
$$f(x) = \frac{1}{b-a}, \quad x \in [a,b]$$

#### Normal (Gaussian)
$$f(x) = \frac{1}{\sigma\sqrt{2\pi}} e^{-\frac{(x-\mu)^2}{2\sigma^2}}$$
- Notation: $X \sim \mathcal{N}(\mu, \sigma^2)$
- 68–95–99.7 rule: 68% within 1σ, 95% within 2σ, 99.7% within 3σ

#### Standard Normal
$$Z = \frac{X - \mu}{\sigma} \sim \mathcal{N}(0, 1)$$

#### Exponential
Time between Poisson events.
$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

---

## 7. Central Limit Theorem (CLT)

> Regardless of the underlying distribution, the **sample mean** of a sufficiently large sample (n ≥ 30) follows an approximately **normal distribution**.

$$\bar{X} \sim \mathcal{N}\!\left(\mu, \frac{\sigma^2}{n}\right)$$

This is why the normal distribution is so important in statistics — it justifies the use of many parametric tests.

---

## 8. Law of Large Numbers

> As the number of trials increases, the sample mean converges to the true population mean.

$$\bar{X}_n \xrightarrow{p} \mu \quad \text{as } n \to \infty$$

---

## Key Takeaways for ML

- **Naive Bayes classifier** is a direct application of Bayes' theorem.
- **Logistic regression** models $P(Y=1|X)$ — conditional probability.
- **Gaussian distribution** assumptions underlie many ML algorithms (LDA, linear regression residuals).
- **Maximum Likelihood Estimation (MLE)** finds parameters that maximise the likelihood function.
