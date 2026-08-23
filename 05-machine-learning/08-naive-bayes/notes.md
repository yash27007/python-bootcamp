# Naïve Bayes

## 1. Problem

We want to classify an example described by features $X = (x_1, x_2, \dots, x_n)$ into a class $y$, and we want the answer to come with a *probability*, not just a label — "90% chance this email is spam" is more useful than a bare yes/no when downstream decisions (block, flag, ignore) depend on confidence.

The natural way to do this is to model the full joint distribution of the features given the class, $P(x_1, x_2, \dots, x_n \mid y)$, and pick the class that makes the observed features most probable (via Bayes' theorem). The problem is data: estimating that full joint distribution requires seeing every meaningful *combination* of feature values within each class, and combinations grow exponentially as features are added. With any realistic number of features, we will never have enough data to estimate it directly. Naive Bayes exists to make probabilistic classification tractable when data is limited relative to the number of features.

## 2. Intuition

Suppose we're predicting whether to `Play` outdoors based on whether it's `Windy` and whether it's `Rainy`. If we tried to model $P(\text{Windy}, \text{Rainy} \mid \text{Play})$ jointly, we would need enough examples of *every combination* — (Windy=Yes, Rainy=Yes), (Windy=Yes, Rainy=No), (Windy=No, Rainy=Yes), (Windy=No, Rainy=No) — within each class, just for two binary features. Add a third feature and the number of combinations doubles again.

Naive Bayes sidesteps this by assuming the features are conditionally independent given the class: once you know it's a "Play=Yes" day, knowing it's windy tells you nothing extra about whether it's also rainy. That assumption lets us estimate each feature's distribution *on its own*, one at a time, which needs vastly less data, and then simply multiply the individual probabilities together to get a combined score. It's "naive" precisely because real features are rarely truly independent — but the resulting classifier is fast, needs little data, and in practice is a strong baseline.

## 3. Why simpler approaches fail

The "simple" approach is to directly estimate the joint likelihood $P(x_1, \dots, x_n \mid y)$ from training data — e.g., by counting how often each exact combination of feature values occurs within each class. This fails because the number of possible combinations grows **exponentially** with the number of features: with $n$ binary features there are $2^n$ possible feature vectors per class. Even a modest $n = 20$ gives over a million combinations — far more than any realistic dataset contains examples. Most combinations would never be observed in training, so most likelihoods would be estimated as exactly zero, making the joint-distribution approach useless in practice.

The naive conditional-independence assumption is precisely what breaks this exponential blow-up: instead of estimating one joint distribution over all $n$ features per class, we estimate $n$ separate one-dimensional distributions per class — a linear amount of work instead of exponential. This assumption is **usually false** in real data (features are often correlated), yet the classifier tends to work well anyway. The reason is that Naive Bayes only needs to get the *ranking* of $\arg\max_y P(y)\prod_i P(x_i\mid y)$ correct, not accurate probability values — even when the independence assumption distorts the magnitude of the posterior, it frequently distorts it similarly across classes, leaving the class with the highest score unchanged. This is discussed further in the Failure modes section, where it does break down.

## 4. Mathematical foundation

### Bayes' theorem

Bayes' theorem calculates the posterior probability $P(y \mid X)$ from $P(y)$, $P(X)$, and $P(X \mid y)$:

$$P(y \mid X) = \frac{P(X \mid y) P(y)}{P(X)}$$

Where:
- $P(y \mid X)$ is the **posterior probability** of class $y$ given predictor (features) $X$.
- $P(y)$ is the **prior probability** of class $y$.
- $P(X \mid y)$ is the **likelihood** — the probability of observing predictor $X$ given class $y$.
- $P(X)$ is the **prior probability of the predictor** (the evidence).

### The "naive" independence assumption

The algorithm is called "naive" because it assumes that all features in $X = (x_1, x_2, \dots, x_n)$ are mutually independent conditional on the class $y$:

$$P(x_i \mid y, x_1, \dots, x_{i-1}, x_{i+1}, \dots, x_n) = P(x_i \mid y)$$

Thus, the overall likelihood factors into a product of per-feature likelihoods:

$$P(X \mid y) = \prod_{i=1}^n P(x_i \mid y)$$

This is exactly the step that turns an exponential joint-estimation problem into $n$ independent, cheaply-estimable one-dimensional problems (see Section 3).

### The decision rule (MAP)

The final classification is made by finding the class $y$ that maximizes the posterior probability (Maximum A Posteriori estimation). Since $P(X)$ is constant across all candidate classes for a given query, it can be dropped from the comparison:

$$\hat{y} = \arg\max_{y} P(y) \prod_{i=1}^n P(x_i \mid y)$$

### Variants of Naive Bayes

There are different ways to model $P(x_i \mid y)$ depending on the nature of the data:
1. **Gaussian Naive Bayes:** used for continuous features. Assumes each feature follows a normal distribution within each class.
2. **Multinomial Naive Bayes:** used for discrete counts (e.g., word frequencies in text classification).
3. **Bernoulli Naive Bayes:** used for binary/boolean features.

### Parameter estimation (MLE)

Unlike algorithms like Logistic Regression or Neural Networks that minimize a distinct cost function (e.g., log-loss) via gradient descent, Naive Bayes models the data distribution directly. It relies on **Maximum Likelihood Estimation (MLE)** to estimate its parameters (probabilities, or means/variances) analytically rather than iteratively minimizing an error surface.

For **Gaussian Naive Bayes**, the per-feature likelihood is modeled as a normal distribution whose mean $\mu_{i,y}$ and variance $\sigma^2_{i,y}$ are estimated (via MLE) directly from the training examples of class $y$:

$$P(x_i \mid y) = \frac{1}{\sqrt{2\pi\sigma_{i,y}^2}} \exp\left(-\frac{(x_i - \mu_{i,y})^2}{2\sigma_{i,y}^2}\right)$$

For **Multinomial Naive Bayes**, the parameters are estimated as relative frequencies (with Laplace/additive smoothing $\alpha$ to avoid zero probabilities for unseen feature values — derived in Section 9):

$$P(x_i \mid y) = \frac{N_{y,i} + \alpha}{N_y + \alpha n}$$

where $N_{y,i}$ is the total count of feature $i$ across all training examples of class $y$, $N_y$ is the total count of all features for class $y$, and $n$ is the number of features (vocabulary size for text).

## 5. Algorithm

1. **Estimate priors** $P(y)$ for every class, from class frequencies in the training set.
2. **Estimate per-feature likelihoods** $P(x_i \mid y)$ for every feature $i$ and class $y$ (Gaussian parameters, multinomial counts, or Bernoulli probabilities, depending on variant).
3. **At prediction time**, for a new example $X$, compute the unnormalized score $P(y) \prod_i P(x_i \mid y)$ for every class.
4. **Predict** the class with the highest score: $\hat{y} = \arg\max_y P(y)\prod_i P(x_i \mid y)$.
5. Optionally normalize the scores by their sum to report calibrated-looking posterior probabilities $P(y \mid X)$.

### Worked numeric example

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

## 6. From-scratch implementation

**Scope decision:** this topic does not include a new from-scratch NumPy reimplementation of Naive Bayes. The worked numeric example above already walks through every arithmetic step of the MAP decision rule by hand (priors → per-feature likelihoods → product → normalize → argmax), which is where a from-scratch implementation would add insight for this algorithm — there is no iterative optimization loop, gradient computation, or hidden numerical-stability subtlety to expose that the hand-worked example doesn't already surface directly. Reimplementing `GaussianNB`/`MultinomialNB` in NumPy would mostly reproduce the same formulas from Section 4 as a few `np.mean`/`np.var`/count-and-divide lines wrapped in a class, without teaching anything the manual derivation and worked example haven't already taught. Per the project's judgment call on when from-scratch code is warranted, that step is skipped here in favor of going straight to the practical library implementation, which is exercised on real datasets in `naive-bayes.ipynb`.

## 7. Practical implementation

`naive-bayes.ipynb` maps the two variants from Section 4 directly onto their scikit-learn implementations:

- **`GaussianNB` on the Iris dataset** — Iris's four continuous, roughly bell-shaped measurements are a natural fit for the Gaussian likelihood formula from Section 4: the notebook fits `GaussianNB`, which under the hood estimates $\mu_{i,y}$ and $\sigma^2_{i,y}$ per feature per class exactly as derived above, then classifies held-out flowers by the same MAP rule from Section 4.
- **`MultinomialNB` on 20-Newsgroups text** (`sci.space` vs `rec.sport.hockey`) — raw documents are vectorized into word counts with `CountVectorizer` (bag-of-words), giving exactly the count data the multinomial likelihood formula from Section 4 (with Laplace smoothing, scikit-learn's default `alpha=1.0`) is built for.

Both fits use the closed-form MLE/count-based parameter estimation from Section 4 — there is no gradient descent or iterative fitting loop involved, consistent with the "Cost Function" discussion there.

## 8. Experiment

The existing `naive-bayes.ipynb` notebook already contains this topic's experiment:

- **Hypothesis (stated before running):** `GaussianNB` should perform well on Iris because its four features are approximately normally distributed within each species class; `MultinomialNB` should perform well on the 20-Newsgroups word-count text because, despite the naive independence assumption being technically wrong for word co-occurrences, bag-of-words classification is a case where Naive Bayes is known to be a strong, fast baseline.
- **Setup:** `GaussianNB` fit on an Iris train/test split (stratified, 70/30); `MultinomialNB` fit on `CountVectorizer` bag-of-words features (5000-word vocabulary cap, English stop words removed) built from the `sci.space` and `rec.sport.hockey` categories of 20-Newsgroups.
- **Result:** both models are evaluated with accuracy and a confusion matrix (`ConfusionMatrixDisplay`) on held-out test data; see the notebook's printed accuracy values and confusion-matrix plots for the exact numbers on the current environment/seed.
- **Interpretation:** the notebook's own takeaway cell confirms the hypothesis — `GaussianNB` benefits from Iris's genuinely near-normal features, and `MultinomialNB` achieves strong separation between the two clearly distinct newsgroup vocabularies even though word occurrences are not really conditionally independent.
- **Limitations:** the 20-Newsgroups task uses only two well-separated categories; accuracy would likely be lower (and the independence violation more damaging) with more, more-similar categories, or with a smaller/noisier vocabulary cap.

## 9. Failure modes

**The independence assumption breaking down with strongly correlated features.** When features are highly correlated (e.g., two features that are near-duplicates of each other, or word pairs that always co-occur), Naive Bayes effectively "double counts" the shared information — each correlated feature contributes its own multiplicative factor to the score as if it were independent new evidence, even though it's largely repeating what another feature already said. This inflates the posterior probability magnitude and can bias the class ranking itself when the degree of correlation differs between classes, not just its calibration. Because the classifier still often gets the argmax right even when its numeric posteriors are distorted, this failure mode shows up more clearly in the *confidence* of the predictions being over- or under-stated than in raw accuracy — but it does degrade accuracy when the correlation structure differs meaningfully between classes.

**The zero-frequency problem, and Laplace smoothing.** Consider estimating a categorical/multinomial likelihood by raw relative frequency:

$$P(x_i \mid y) = \frac{N_{y,i}}{N_y}$$

If a particular feature value (e.g., a specific word) never appears in the training examples of class $y$ (so $N_{y,i} = 0$), this formula gives $P(x_i \mid y) = 0$ exactly. Because the decision rule *multiplies* all per-feature likelihoods together, a single zero-probability feature forces the entire product $P(y)\prod_i P(x_i \mid y)$ to zero — regardless of how strongly every other feature favors class $y$. In text classification this is common: any test document containing a word never seen in a class's training vocabulary would otherwise zero out that class's score entirely, which is clearly too harsh a penalty for one unseen word.

**Laplace (additive) smoothing** fixes this by pretending every feature value has been seen $\alpha$ extra times (typically $\alpha = 1$) before counting, for every class. Adding $\alpha$ to every numerator, we must also add $\alpha$ once for every one of the $n$ possible feature values to the denominator, so the estimate still sums to 1 over all values:

$$P(x_i \mid y) = \frac{N_{y,i} + \alpha}{N_y + \alpha n}$$

With $\alpha > 0$, no likelihood is ever exactly zero, so no single unseen feature value can zero out an entire class's score — it instead gets a small nonzero, appropriately pessimistic probability. This is the formula already used in Section 4 and applied by default (`alpha=1.0`) in scikit-learn's `MultinomialNB`.

## 10. Real-world usage

- **Spam and content filtering:** the classic Naive Bayes use case — bag-of-words features over email/message text, fast enough to score every incoming message in real time.
- **Text classification / sentiment analysis baselines:** Naive Bayes is a standard first baseline before trying more expensive models, precisely because it is cheap to train (closed-form MLE, no iterative optimization) and surprisingly hard to beat on high-dimensional sparse text features.
- **Medical / diagnostic screening tools:** where features (symptoms, test results) are reasonably independent given the diagnosis, and a fast, interpretable probability estimate is valuable.
- **Real-time recommendation and document-routing systems:** where prediction latency matters and the independence assumption is an acceptable trade for speed and small memory footprint.

## 11. Mental model

Naive Bayes turns classification into "let every feature vote independently for each class, using Bayes' theorem to weigh each vote by how typical that feature value is for that class, and add a small prior nudge for how common the class is overall — then trust whichever class collected the loudest total vote." It's naive because it assumes every feature votes without listening to what the others are saying, but that naivety is exactly what makes the vote countable at all with limited data.

## 12. Questions to think about

1. Why does the naive independence assumption reduce the amount of training data needed from *exponential* in the number of features to roughly *linear*? Walk through the counting argument for $n$ binary features.
2. Suppose two features in your dataset are perfectly correlated (one is always equal to the other). What does Naive Bayes effectively do to their combined influence on the posterior, and why?
3. If a test document contains a word that never appeared anywhere in training (for *any* class), does Laplace smoothing alone fix the resulting likelihood being unreliable? What does scikit-learn's `CountVectorizer`/vocabulary handling do with such out-of-vocabulary words, and why does that matter?
4. Naive Bayes often gets the correct *class ranking* even when its independence assumption is badly violated. Why might a wrong assumption still preserve the argmax, even though it distorts the actual posterior probability values?
5. Why is Gaussian Naive Bayes a poor fit for a feature that is strongly bimodal (e.g., clearly clustered into two separate humps) within a single class? What would happen to the fitted Gaussian, and what would that do to likelihood estimates for points near the "valley" between the humps?
