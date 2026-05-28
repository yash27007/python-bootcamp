# Anomaly Detection: A Complete Guide

## Big Picture (Simple Summary)

Imagine you are a bank monitoring millions of credit card transactions per day. 99.99% are normal purchases. But a few are fraudulent. You need to automatically flag the suspicious ones — even though you may not know exactly what "fraudulent" looks like, and even if fraudsters constantly change their behavior.

**Anomaly detection** (also called outlier detection) is the task of identifying data points that are significantly different from the majority of the data. These unusual points are called **anomalies**, **outliers**, or **novelties**.

Unlike classification, you often don't have labeled "anomaly" examples — the algorithm must learn what "normal" looks like and flag deviations.

---

## 1) Types of Anomalies

### 1.1 Point Anomalies
A single data point is unusual compared to the rest of the data.

**Example:** A transaction of $50,000 when all other transactions are under $500.

### 1.2 Contextual Anomalies (Conditional Anomalies)
A data point is anomalous in a specific context but not globally.

**Example:** 30°C temperature is normal in summer but anomalous in winter.  
**Example:** Buying 100 ski jackets is normal for a ski shop in November, but anomalous in July.

### 1.3 Collective Anomalies
A group of data points is anomalous together, even if each point is individually normal.

**Example:** A series of small transactions ($9.99, $9.99, $9.99, ...) — individually normal, but 50 of them in a row is suspicious (card testing).

---

## 2) Statistical Methods

### 2.1 Z-Score (Standard Score)

For a feature with mean $\mu$ and standard deviation $\sigma$, the Z-score of a data point $x$ is:

$$z = \frac{x - \mu}{\sigma}$$

**Rule of thumb:** A point is an outlier if $|z| > 3$ (more than 3 standard deviations from the mean).

**Assumption:** Data follows a **normal (Gaussian) distribution**.

**Worked Example:**

Heights in a class: mean = 170 cm, std = 10 cm

| Person | Height | Z-Score | Anomaly? |
|--------|--------|---------|---------|
| A | 175 | (175-170)/10 = +0.5 | No |
| B | 195 | (195-170)/10 = +2.5 | Borderline |
| C | 140 | (140-170)/10 = **-3.0** | Flagged |
| D | 225 | (225-170)/10 = **+5.5** | Definitely anomaly |

**Limitation:** Z-score assumes Gaussian distribution, is sensitive to the outliers themselves (they inflate the mean and std), and only works for univariate data.

**Modified Z-Score (more robust):** Uses median (M) and median absolute deviation (MAD):

$$\text{Modified } z = \frac{0.6745(x_i - \text{median})}{MAD}$$

Where $MAD = \text{median}(|x_i - \text{median}|)$. Flag if $|z| > 3.5$.

### 2.2 Interquartile Range (IQR) Method

The **IQR** is the range between the 25th and 75th percentiles:

$$IQR = Q_3 - Q_1$$

**Outlier bounds:**
- Lower fence: $Q_1 - 1.5 \times IQR$
- Upper fence: $Q_3 + 1.5 \times IQR$

Points outside these fences are flagged as outliers.

**Worked Example:**

Salaries (in thousands): [30, 35, 38, 40, 42, 45, 50, 55, 200]

- Q1 = 37, Q3 = 52, IQR = 15
- Lower fence = 37 - 1.5 × 15 = **14.5**
- Upper fence = 52 + 1.5 × 15 = **74.5**

Salary of $200k is an outlier (above 74.5).

**Why IQR?** Robust to the outliers themselves (uses percentiles, not mean/std).

```python
import numpy as np

def iqr_outliers(data):
    Q1 = np.percentile(data, 25)
    Q3 = np.percentile(data, 75)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    return (data < lower) | (data > upper)
```

### 2.3 Mahalanobis Distance (Multivariate Outlier Detection)

For multivariate data, Euclidean distance doesn't account for correlations between features. **Mahalanobis distance** fixes this:

$$D_M(x) = \sqrt{(x - \mu)^T \Sigma^{-1} (x - \mu)}$$

Where:
- $\mu$ = mean vector of the data
- $\Sigma$ = covariance matrix
- $\Sigma^{-1}$ = inverse covariance matrix (accounts for correlations and scale)

**Interpretation:** If the data is multivariate Gaussian, $D_M^2$ follows a chi-squared distribution with p degrees of freedom. A point is an outlier if:

$$D_M^2 > \chi^2_{p, 0.975}$$

**Why not just Euclidean distance?** Suppose Feature 1 ranges [0, 1000] and Feature 2 ranges [0, 1]. Euclidean distance is dominated by Feature 1. Also, if features are correlated, a point might look "far" in one dimension but actually be in the normal range of the joint distribution.

```python
from scipy.spatial.distance import mahalanobis
from scipy.stats import chi2
import numpy as np

# Compute robust covariance (less sensitive to outliers)
from sklearn.covariance import MinCovDet
mcd = MinCovDet(random_state=42).fit(X)

# Mahalanobis distances
distances = mcd.mahalanobis(X)

# Threshold: chi-squared distribution
threshold = chi2.ppf(0.975, df=X.shape[1])
outliers = distances > threshold
print(f"Outliers detected: {outliers.sum()} ({100*outliers.mean():.1f}%)")
```

---

## 3) Isolation Forest

### 3.1 The Core Idea

**Isolation Forest** (Liu et al., 2008) is based on a simple observation:

> Anomalies are few and different. Normal points require more splits to isolate; anomalies can be isolated with very few splits.

**Algorithm:**
1. Build many random **isolation trees** (iTrees)
2. Each iTree recursively splits the data with random feature + random split value
3. For each data point, record the average number of splits (path length) needed to isolate it across all trees

**The anomaly score:**

$$s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}$$

Where:
- $E[h(x)]$ = average path length (number of splits) to isolate point $x$
- $c(n)$ = expected path length for n samples (normalization factor): 
$$c(n) = 2H(n-1) - \frac{2(n-1)}{n}$$
where $H(i) = \ln(i) + 0.5772$ (Euler-Mascheroni constant)

**Score interpretation:**
- $s \approx 1$: Very short path → anomaly (easy to isolate)
- $s \approx 0.5$: Normal points → no clear anomalies
- $s \ll 0.5$: Very long path → dense, normal region

### 3.2 Visual Intuition

Imagine a scatter plot with 100 normal points in a dense cluster and 1 outlier far away.

**Normal point (in the cluster):** A random split might not separate it from others. You need many, many splits to finally isolate it from the last few neighbors.

**Outlier (far from cluster):** The first random split likely puts it in a partition by itself (or nearly so). Path length ≈ 1–2 splits.

The outlier is **naturally easier to isolate** → shorter average path length → higher anomaly score.

### 3.3 Why Isolation Forest is Powerful

1. **No distance computation** → works in high dimensions (unlike k-NN based methods)
2. **No density estimation** → avoids curse of dimensionality
3. **Linear time complexity** $O(N \log N)$
4. **No assumption about data distribution**
5. **Handles high-dimensional data** well

### 3.4 Hyperparameters

| Parameter | Description | Default | Typical Range |
|-----------|-------------|---------|---------------|
| `n_estimators` | Number of iTrees | 100 | 100–500 |
| `max_samples` | Training samples per tree | 'auto' (256) | 'auto' or 0.1–1.0 × N |
| `contamination` | Expected fraction of anomalies | 'auto' | 0.01–0.2 |
| `max_features` | Features per split | 1.0 | 0.5–1.0 |

**Why max_samples=256?** After 256 samples, path lengths saturate — anomalies vs. normal points are already distinguishable. Using all N samples doesn't improve detection but is much slower.

### 3.5 Implementation

```python
from sklearn.ensemble import IsolationForest
import numpy as np
import matplotlib.pyplot as plt

# Fit Isolation Forest
clf = IsolationForest(
    n_estimators=200,
    max_samples='auto',
    contamination=0.05,  # Expect 5% outliers
    random_state=42,
    n_jobs=-1
)
clf.fit(X_train)

# Predict: 1 = normal, -1 = anomaly
predictions = clf.predict(X_test)
scores = clf.decision_function(X_test)  # More negative = more anomalous
anomaly_scores = -clf.score_samples(X_test)  # Positive anomaly score (higher = more anomalous)

print(f"Anomalies detected: {(predictions == -1).sum()}")

# If you have labels for evaluation
from sklearn.metrics import classification_report, roc_auc_score
# Convert: -1 → 1 (anomaly), 1 → 0 (normal)
y_pred_binary = (predictions == -1).astype(int)
print(roc_auc_score(y_true, anomaly_scores))
```

---

## 4) Local Outlier Factor (LOF)

### 4.1 Idea

LOF (Breunig et al., 2000) compares the **local density** of a point to the local density of its neighbors. An anomaly is a point that is in a region of much lower density than its neighbors.

**Key concept: reachability distance**

$$\text{reach-dist}_k(A, B) = \max(k\text{-dist}(B), d(A, B))$$

Where $k\text{-dist}(B)$ is the distance from B to its $k$-th nearest neighbor.

This smooths out density estimates by preventing very small distances.

**Local reachability density of point A:**

$$\text{lrd}_k(A) = \frac{k}{\sum_{B \in N_k(A)} \text{reach-dist}_k(A, B)}$$

High lrd = A is in a dense region.

**LOF score:**

$$\text{LOF}_k(A) = \frac{\sum_{B \in N_k(A)} \frac{\text{lrd}_k(B)}{\text{lrd}_k(A)}}{k} = \frac{\text{average lrd of A's neighbors}}{\text{lrd of A}}$$

**Interpretation:**
- LOF ≈ 1 → A has similar density to its neighbors → normal
- LOF >> 1 → A is much less dense than its neighbors → anomaly
- LOF < 1 → A is denser than its neighbors (rare, possible in very dense clusters)

### 4.2 LOF vs Isolation Forest

| Aspect | Isolation Forest | LOF |
|--------|-----------------|-----|
| Approach | Global (random splits) | Local (density comparison) |
| Works in high dim | Yes | No (distances meaningless) |
| Speed | Fast O(N log N) | Slower O(N²) |
| Detects | Global outliers | Local outliers (relative to neighbors) |
| Best for | General purpose | When density varies significantly |

**Example where LOF excels:** Dense cluster A (100 points in radius 1) and sparse cluster B (100 points in radius 10). A point slightly outside cluster A might be flagged as anomalous by Isolation Forest (low global density) but is normal relative to cluster A — LOF will correctly judge it as normal.

```python
from sklearn.neighbors import LocalOutlierFactor

lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05,
    novelty=False  # Set True for predict() on new data
)
predictions = lof.fit_predict(X)
scores = -lof.negative_outlier_factor_  # Higher = more anomalous

print(f"Anomalies: {(predictions == -1).sum()}")
```

---

## 5) One-Class SVM

### 5.1 Idea

One-Class SVM (Schölkopf et al., 2001) learns a **boundary in feature space** (using the kernel trick) that encloses most of the training data. Points outside this boundary are anomalies.

**Objective:**
$$\min_{w, \rho, \xi} \frac{1}{2}||w||^2 + \frac{1}{\nu n} \sum_{i=1}^n \xi_i - \rho$$

Where:
- $\nu \in (0, 1)$: Upper bound on fraction of outliers (hyperparameter)
- $\rho$: The offset of the hyperplane from origin (threshold)
- $\xi_i \geq 0$: Slack variables (allow some points to be outside the boundary)

**Decision function:** $f(x) = \text{sign}(w^T \phi(x) - \rho)$

- $f(x) = +1$: Normal
- $f(x) = -1$: Anomaly

### 5.2 When to Use One-Class SVM

- You have **only normal** training data (no anomaly examples at all — "novelty detection")
- Data lives in a high-dimensional space but has low intrinsic dimensionality
- You can use kernel tricks (RBF kernel to detect non-spherical normal regions)

**Limitation:** Very slow on large datasets. Use Isolation Forest or LOF instead for big data.

```python
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

oc_svm = OneClassSVM(nu=0.05, kernel='rbf', gamma='scale')
oc_svm.fit(X_train_scaled)  # Train on NORMAL data only

predictions = oc_svm.predict(X_test_scaled)  # 1 = normal, -1 = anomaly
scores = oc_svm.decision_function(X_test_scaled)  # More positive = more normal
```

---

## 6) Autoencoder-Based Anomaly Detection

### 6.1 Idea

An **autoencoder** is a neural network that learns to compress data into a low-dimensional representation and then reconstruct it:

```
Input X (n features)
     ↓
Encoder (compresses to latent dim d << n)
     ↓
Latent representation z (bottleneck)
     ↓
Decoder (reconstructs from z)
     ↓
Reconstruction X̂ (n features)
```

The autoencoder is trained on **normal data** to minimize **reconstruction error**:

$$L = ||X - \hat{X}||^2 = \sum_{i=1}^n (x_i - \hat{x}_i)^2$$

**Key insight:** After training, the autoencoder is good at reconstructing normal patterns (they fit within the learned latent space). When given an anomaly, the latent space can't represent it well → **high reconstruction error** → anomaly detected.

$$\text{Anomaly Score}(x) = ||x - \text{autoencoder}(x)||^2$$

Set a threshold (e.g., 95th percentile of training reconstruction errors): if score > threshold → anomaly.

### 6.2 Advantages Over Other Methods

| Aspect | Isolation Forest | LOF | Autoencoder |
|--------|-----------------|-----|------------|
| High-dim data | Yes | No | **Yes** |
| Non-linear patterns | No | No | **Yes** |
| Sequence/temporal | No | No | **Yes (with LSTM)** |
| Interpretability | Medium | Medium | Low |
| Requires GPU | No | No | Sometimes |
| Training data needed | Little | Little | **Much more** |

### 6.3 Implementation (Keras/TensorFlow)

```python
import numpy as np
import tensorflow as tf
from tensorflow import keras

# Build autoencoder
input_dim = X_train.shape[1]
encoding_dim = 8  # Bottleneck size

# Encoder
inputs = keras.Input(shape=(input_dim,))
x = keras.layers.Dense(64, activation='relu')(inputs)
x = keras.layers.Dense(32, activation='relu')(x)
encoded = keras.layers.Dense(encoding_dim, activation='relu')(x)

# Decoder
x = keras.layers.Dense(32, activation='relu')(encoded)
x = keras.layers.Dense(64, activation='relu')(x)
outputs = keras.layers.Dense(input_dim, activation='linear')(x)

autoencoder = keras.Model(inputs, outputs)
autoencoder.compile(optimizer='adam', loss='mse')

# Train on normal data only
history = autoencoder.fit(
    X_train_normal, X_train_normal,
    epochs=100,
    batch_size=64,
    validation_split=0.1,
    callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)]
)

# Compute reconstruction errors
reconstructions = autoencoder.predict(X_test)
reconstruction_errors = np.mean((X_test - reconstructions) ** 2, axis=1)

# Set threshold (e.g., 95th percentile on training data)
train_reconstructions = autoencoder.predict(X_train_normal)
train_errors = np.mean((X_train_normal - train_reconstructions) ** 2, axis=1)
threshold = np.percentile(train_errors, 95)

# Classify
predictions = (reconstruction_errors > threshold).astype(int)
print(f"Anomalies detected: {predictions.sum()}")
```

---

## 7) Practical Guide: Which Method to Choose?

| Situation | Recommended Method |
|-----------|-------------------|
| Large dataset, no labels | **Isolation Forest** (first choice) |
| Need local density comparison | **LOF** |
| Only normal training data (novelty detection) | **One-Class SVM** or **Isolation Forest** |
| Complex patterns, sequence data | **Autoencoder** |
| Simple univariate checks | **Z-score** or **IQR** |
| Correlated, multivariate features | **Mahalanobis Distance** |
| Unknown number of anomaly types | **DBSCAN** (noise points = anomalies) |

---

## 8) Evaluating Anomaly Detection Models

Anomaly detection evaluation is tricky because:
1. Data is extremely imbalanced (few anomalies)
2. You may have no labeled anomalies at all

### 8.1 With Labeled Data

| Metric | Formula | Use When |
|--------|---------|---------|
| **Precision** | TP / (TP + FP) | Cost of false alarms is high |
| **Recall** | TP / (TP + FN) | Missing anomalies is costly |
| **F1 Score** | 2·P·R / (P + R) | Balance both |
| **AUC-ROC** | Area under ROC curve | Want threshold-free evaluation |
| **AUC-PR** | Area under Precision-Recall curve | **Best for imbalanced data** |

**Why AUC-PR over AUC-ROC for anomaly detection?** When only 1% of data is anomalous, even a bad model can have high AUC-ROC (by getting all normal points right). AUC-PR focuses on the minority class (anomalies) — much more informative.

```python
from sklearn.metrics import (roc_auc_score, average_precision_score,
                             classification_report)

# Isolation Forest scores: more negative = more anomalous
anomaly_scores = -clf.score_samples(X_test)

auc_roc = roc_auc_score(y_true, anomaly_scores)
auc_pr  = average_precision_score(y_true, anomaly_scores)

print(f"AUC-ROC: {auc_roc:.4f}")
print(f"AUC-PR: {auc_pr:.4f}")

# Threshold-based evaluation
threshold = np.percentile(anomaly_scores, 95)
y_pred = (anomaly_scores > threshold).astype(int)
print(classification_report(y_true, y_pred, target_names=['Normal', 'Anomaly']))
```

### 8.2 Without Labels

- **Visual inspection:** Plot anomaly score distribution, manually inspect flagged points
- **Domain knowledge:** Ask subject matter experts to review flagged examples
- **Business metrics:** Track fraud loss prevented, system alerts triggered, etc.

---

## 9) Real-World Applications

| Domain | Application | What's "Anomalous" |
|--------|-------------|-------------------|
| Finance | Fraud detection | Unusual spending patterns |
| Cybersecurity | Network intrusion | Unusual traffic patterns |
| Manufacturing | Predictive maintenance | Sensor readings before failure |
| Healthcare | Patient monitoring | Vital signs deviating from normal |
| E-commerce | Bot detection | Inhuman browsing speed/patterns |
| Infrastructure | Server monitoring | Memory/CPU spikes |

---

## Summary: The 5 Key Takeaways

1. **Anomaly detection ≠ classification** — you typically train only on normal data (or without any labels) and flag anything that deviates significantly.

2. **Isolation Forest** is the best default: fast, scalable, distribution-free, and works well in high dimensions — start with it for any tabular dataset.

3. **Statistical methods (Z-score, IQR, Mahalanobis)** are powerful baselines for low-dimensional, well-distributed data — always try them first.

4. **Threshold choice matters**: use domain knowledge to balance false positives (alert fatigue) vs false negatives (missed anomalies). Use precision-recall curves, not just a fixed score cutoff.

5. **AUC-PR is the right metric** for anomaly detection evaluation, not accuracy or AUC-ROC — the extreme class imbalance makes precision-recall far more informative.
