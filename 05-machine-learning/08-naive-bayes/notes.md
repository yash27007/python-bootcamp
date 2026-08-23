# Naïve Bayes

## Overview
Naive Bayes is a family of probabilistic machine learning models based on applying **Bayes' Theorem** with a "naive" assumption of conditional independence between every pair of features given the value of the class variable.

## Bayes' Theorem
Bayes' theorem calculates the posterior probability $P(y|X)$ from $P(y)$, $P(X)$, and $P(X|y)$:

$$P(y|X) = \frac{P(X|y)P(y)}{P(X)}$$

Where:
- $P(y|X)$ is the **posterior probability** of class $y$ given predictor (features) $X$.
- $P(y)$ is the **prior probability** of class $y$.
- $P(X|y)$ is the **likelihood** which is the probability of the predictor $X$ given class $y$.
- $P(X)$ is the **prior probability of the predictor** (evidence).

## The "Naive" Assumption
The algorithm is called "naive" because it assumes that all features in $X = (x_1, x_2, ..., x_n)$ are mutually independent conditional on the class $y$:

$$P(x_i | y, x_1, ..., x_{i-1}, x_{i+1}, ..., x_n) = P(x_i | y)$$

Thus, the overall likelihood becomes:
$$P(X|y) = \prod_{i=1}^n P(x_i | y)$$

## The Decision Rule
The final classification is made by finding the class $y$ that maximizes the posterior probability (Maximum A Posteriori estimation). Since $P(X)$ is constant across all classes, we can drop it:

$$\hat{y} = \arg\max_{y} P(y) \prod_{i=1}^n P(x_i | y)$$

## Variants of Naive Bayes
There are different ways to model $P(x_i | y)$ depending on the nature of the data:
1. **Gaussian Naive Bayes:** Used for continuous features. It assumes that features follow a normal distribution.
2. **Multinomial Naive Bayes:** Used for discrete counts (e.g., word frequencies in text classification).
3. **Bernoulli Naive Bayes:** Used for binary/boolean features.

## Cost Function Formulation
Unlike algorithms like Logistic Regression or Neural Networks that minimize a distinct "cost function" (e.g., Log-Loss) via gradient descent, Naive Bayes models the data distribution directly. It fundamentally relies on **Maximum Likelihood Estimation (MLE)** to estimate the parameters (probabilities and means/variances) analytically rather than iteratively minimizing an error surface.

For **Gaussian Naive Bayes**, the per-feature likelihood is modeled as a normal distribution whose mean $\mu_{i,y}$ and variance $\sigma^2_{i,y}$ are estimated (via MLE) directly from the training examples of class $y$:

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_{i,y}^2}} \exp\left(-\frac{(x_i - \mu_{i,y})^2}{2\sigma_{i,y}^2}\right)$$

For **Multinomial Naive Bayes**, the parameters are estimated as relative frequencies (with Laplace/additive smoothing $\alpha$ to avoid zero probabilities for unseen feature values):

$$P(x_i \mid y) = \frac{N_{y,i} + \alpha}{N_y + \alpha n}$$

where $N_{y,i}$ is the total count of feature $i$ across all training examples of class $y$, $N_y$ is the total count of all features for class $y$, and $n$ is the number of features (vocabulary size for text).

## Worked Numeric Example

Consider a tiny toy dataset for predicting whether to `Play` (Yes/No) based on two binary features: `Windy` (Yes/No) and `Rainy` (Yes/No).

| # | Windy | Rainy | Play |
|---|-------|-------|------|
| 1 | Yes   | No    | No   |
| 2 | No    | No    | Yes  |
| 3 | No    | Yes   | No   |
| 4 | Yes   | Yes   | No   |
| 5 | No    | No    | Yes  |
| 6 | Yes   | No    | Yes  |

**Priors** (out of 6 examples: 3 Yes, 3 No):
$$P(\text{Play=Yes}) = \frac{3}{6} = 0.5 \qquad P(\text{Play=No}) = \frac{3}{6} = 0.5$$

**Likelihoods**, computed within each class:
- Among Play=Yes (rows 2, 5, 6): Windy=Yes occurs 1/3 time, Rainy=No occurs 3/3 time.
  $$P(\text{Windy=Yes}\mid\text{Yes}) = \frac13 \qquad P(\text{Rainy=No}\mid\text{Yes}) = \frac33 = 1$$
- Among Play=No (rows 1, 3, 4): Windy=Yes occurs 2/3 time, Rainy=No occurs 1/3 time.
  $$P(\text{Windy=Yes}\mid\text{No}) = \frac23 \qquad P(\text{Rainy=No}\mid\text{No}) = \frac13$$

**Query:** predict `Play` for a new day where `Windy=Yes, Rainy=No`.

Using the naive independence assumption, compute the (unnormalized) posterior score for each class:

$$\text{score(Yes)} = P(\text{Yes}) \cdot P(\text{Windy=Yes}\mid\text{Yes}) \cdot P(\text{Rainy=No}\mid\text{Yes}) = 0.5 \times \frac13 \times 1 = 0.1\overline{6}$$

$$\text{score(No)} = P(\text{No}) \cdot P(\text{Windy=Yes}\mid\text{No}) \cdot P(\text{Rainy=No}\mid\text{No}) = 0.5 \times \frac23 \times \frac13 = 0.1\overline{1}$$

Normalizing by dividing by the evidence $P(X) = \text{score(Yes)} + \text{score(No)} \approx 0.2\overline{7}$:

$$P(\text{Yes}\mid X) \approx \frac{0.1\overline{6}}{0.2\overline{7}} \approx 0.6 \qquad P(\text{No}\mid X) \approx \frac{0.1\overline{1}}{0.2\overline{7}} \approx 0.4$$

Since $P(\text{Yes}\mid X) > P(\text{No}\mid X)$, the model predicts **Play = Yes**.
