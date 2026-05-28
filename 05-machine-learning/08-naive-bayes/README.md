# 08 – Naïve Bayes

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
