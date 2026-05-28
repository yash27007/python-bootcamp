# Unsupervised Learning: A Complete Guide

## Big Picture (Simple Summary)

In supervised learning, you have a teacher — labeled data tells the model what is right and wrong. In **unsupervised learning**, there are no labels. The algorithm must discover hidden structure, patterns, and groupings on its own. It's like handing someone thousands of photographs with no captions and asking them to organize them into groups.

The two main tasks are:
1. **Clustering** — group similar things together (K-Means, DBSCAN, Hierarchical)
2. **Dimensionality Reduction** — compress data into fewer dimensions while preserving structure (PCA, t-SNE, UMAP)

---

## Part 1: Clustering

---

## 2) K-Means Clustering

### 2.1 The Core Idea

K-Means partitions N data points into K clusters by minimizing the total **within-cluster sum of squares (WCSS)**, also called **inertia**:

$$J = \sum_{k=1}^K \sum_{x_i \in C_k} ||x_i - \mu_k||^2$$

Where:
- $C_k$ = set of points in cluster k
- $\mu_k$ = centroid (mean) of cluster k
- $||x_i - \mu_k||^2$ = squared Euclidean distance from point to centroid

**Goal:** Find cluster assignments and centroids that minimize J.

### 2.2 The Algorithm (Lloyd's Algorithm)

```
1. Choose K (number of clusters)
2. Initialize K centroids (randomly or with K-Means++)
3. REPEAT until convergence:
   a. Assignment step: Assign each point to its nearest centroid
      for each point xᵢ: cᵢ = argmin_k ||xᵢ - μk||²
   b. Update step: Move each centroid to the mean of its assigned points
      μk = (1/|Ck|) Σ_{xᵢ ∈ Ck} xᵢ
4. STOP when assignments don't change (convergence)
```

**The algorithm always converges** (J can only decrease or stay the same each iteration), but not necessarily to the **global** optimum — it may find a local minimum.

### 2.3 K-Means++ Initialization

Random initialization often leads to poor local minima. **K-Means++** (Arthur & Vassilvitskii, 2007) fixes this with a smarter initialization:

1. Choose the first centroid $\mu_1$ uniformly at random from the data
2. For each subsequent centroid $\mu_k$:
   - For each data point $x_i$, compute $d(x_i)^2$ = squared distance to the nearest already-chosen centroid
   - Choose next centroid with probability proportional to $d(x_i)^2$
   - Points farther from existing centroids are more likely to be chosen as new centroids
3. Continue until K centroids are chosen

**Effect:** Centers are spread out → much better starting positions → faster convergence → better solutions.

In scikit-learn, `init='k-means++'` is the default.

### 2.4 Choosing K — The Elbow Method

Since K is a hyperparameter, you need to select it. The **elbow method** plots inertia vs K:

$$\text{For K = 1, 2, ..., 15: compute inertia}$$

The "elbow" in the curve (where inertia stops decreasing sharply) suggests the optimal K.

**Example:**
| K | Inertia |
|---|---------|
| 1 | 10,000 |
| 2 | 5,000 |
| 3 | 2,500 |
| 4 | 1,800 |
| 5 | 1,600 |
| 6 | 1,550 |
| 7 | 1,530 |

Elbow at K=4 or K=5 (inertia stops decreasing sharply after that).

**Silhouette Score** (more reliable than elbow):

For each point $i$, the silhouette score measures how well it fits its cluster compared to neighboring clusters:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$

Where:
- $a(i)$ = average distance from point $i$ to other points in its own cluster (cohesion)
- $b(i)$ = average distance from point $i$ to points in the nearest other cluster (separation)

$s(i) \in [-1, 1]$:
- Near +1 → well clustered
- Near 0 → on border between clusters
- Near -1 → probably in the wrong cluster

**Average silhouette score** across all points: Higher is better. Choose K that maximizes this.

```python
from sklearn.metrics import silhouette_score

scores = []
for k in range(2, 15):
    km = KMeans(n_clusters=k, random_state=42)
    labels = km.fit_predict(X)
    scores.append(silhouette_score(X, labels))

# Plot scores
plt.plot(range(2, 15), scores, marker='o')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for Optimal K')
plt.show()
```

### 2.5 Limitations of K-Means

| Limitation | Description | Solution |
|-----------|-------------|---------|
| Must specify K | Don't know K in advance | Use elbow/silhouette; try DBSCAN |
| Assumes spherical clusters | Fails on elongated, non-convex shapes | Use DBSCAN or GMM |
| Sensitive to scale | Features in larger ranges dominate | **Always standardize features!** |
| Sensitive to outliers | Outliers pull centroids toward them | Use K-Medoids or DBSCAN |
| Local minima | Random initialization matters | Use K-Means++, run multiple times |

### 2.6 Implementation

```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Always scale first!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Elbow method
inertias = []
K_range = range(1, 16)
for k in K_range:
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    km.fit(X_scaled)
    inertias.append(km.inertia_)

plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, marker='o')
plt.xlabel('K')
plt.ylabel('Inertia')
plt.title('Elbow Method')

# Silhouette analysis
from sklearn.metrics import silhouette_score
sil_scores = []
for k in range(2, 16):
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    labels = km.fit_predict(X_scaled)
    sil_scores.append(silhouette_score(X_scaled, labels))

plt.subplot(1, 2, 2)
plt.plot(range(2, 16), sil_scores, marker='o', color='orange')
plt.xlabel('K')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.tight_layout()
plt.show()

# Fit final model
optimal_k = 4
final_model = KMeans(n_clusters=optimal_k, init='k-means++', n_init=10, random_state=42)
labels = final_model.fit_predict(X_scaled)
print(f"Cluster sizes: {np.bincount(labels)}")
```

---

## 3) Hierarchical Clustering

### 3.1 The Idea

Instead of specifying K upfront, hierarchical clustering builds a **tree of clusters (dendrogram)** that shows all possible groupings from N individual points to 1 big cluster.

You can then "cut" the tree at any height to get any number of clusters K.

### 3.2 Agglomerative (Bottom-Up) Approach

**Start:** N clusters, each containing one data point.

**Repeat until 1 cluster remains:**
1. Find the two closest clusters (using a **linkage criterion**)
2. Merge them into one cluster
3. Update the distance matrix

### 3.3 Linkage Criteria (How to Measure Distance Between Clusters)

| Linkage | Distance Between Clusters A and B | Characteristic |
|---------|----------------------------------|----------------|
| **Single** | $\min_{a \in A, b \in B} d(a, b)$ (minimum distance) | Tends to create long, chained clusters |
| **Complete** | $\max_{a \in A, b \in B} d(a, b)$ (maximum distance) | Creates compact, roughly equal clusters |
| **Average** | $\frac{1}{|A||B|} \sum_{a \in A} \sum_{b \in B} d(a, b)$ (mean distance) | Good compromise |
| **Ward** | Minimizes the increase in total WCSS when merging | Most popular; creates compact clusters |

**Ward's linkage** merges clusters that result in the **smallest increase in inertia** — similar to K-Means objective.

### 3.4 The Dendrogram

The dendrogram shows the merge history:
- X-axis: data points
- Y-axis: distance at which clusters were merged (height)
- Higher merge height = less similar clusters

**How to choose K from a dendrogram:** Look for the largest vertical gap (longest lines that don't cross a horizontal cut). Draw a horizontal line through the gap → the number of lines it crosses = K.

```python
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build the linkage matrix
Z = linkage(X_scaled, method='ward')

# Plot dendrogram
plt.figure(figsize=(15, 6))
dendrogram(
    Z,
    truncate_mode='lastp',  # Show only last p merged clusters
    p=30,
    leaf_rotation=90,
    leaf_font_size=8,
    show_contracted=True
)
plt.title('Hierarchical Clustering Dendrogram (Ward Linkage)')
plt.xlabel('Cluster / Sample Index')
plt.ylabel('Distance')
plt.axhline(y=6, color='r', linestyle='--', label='Cut at distance=6')
plt.legend()
plt.show()

# Get flat cluster labels by cutting at distance or n_clusters
labels = fcluster(Z, t=4, criterion='maxclust')  # t=4 clusters
# OR
labels = fcluster(Z, t=6.0, criterion='distance')  # cut at distance=6
```

### 3.5 K-Means vs Hierarchical

| Aspect | K-Means | Hierarchical |
|--------|---------|-------------|
| K required? | Yes | No (choose after) |
| Scales to large N? | Yes | No (O(N²) memory) |
| Deterministic? | No (random init) | Yes |
| Result | Flat partition | Full hierarchy (dendrogram) |
| Cluster shape | Spherical only | Flexible |
| Outlier handling | Poor | Better |

---

## 4) DBSCAN — Density-Based Spatial Clustering

### 4.1 Motivation

K-Means and hierarchical clustering fail when:
- Clusters have irregular shapes
- Clusters are nested or have varying density
- There are noise/outlier points

**DBSCAN (Density-Based Spatial Clustering of Applications with Noise)** fixes this.

### 4.2 Key Concepts

DBSCAN has two hyperparameters:
- **ε (eps):** The radius of a neighborhood around each point
- **min_samples (MinPts):** Minimum number of points required in a ε-neighborhood for a point to be a "core point"

**Three types of points:**

1. **Core point:** Has at least `min_samples` points within radius ε (including itself)
2. **Border point:** Within ε of a core point, but has fewer than `min_samples` neighbors itself
3. **Noise point:** Neither core nor border — it's an outlier, labeled as -1

**Directly density-reachable:** Point B is directly density-reachable from A if:
- A is a core point
- B is within distance ε of A

**Density-reachable:** B is density-reachable from A if there exists a chain of points $A = p_1, p_2, \dots, p_n = B$ where each $p_{i+1}$ is directly density-reachable from $p_i$.

**A cluster** = all points mutually density-connected (reachable through core points).

### 4.3 The Algorithm

```
Mark all points as unvisited.

FOR each unvisited point P:
    Mark P as visited
    Find all neighbors within radius ε: N(P) = {q : dist(P,q) ≤ ε}
    
    IF |N(P)| < min_samples:
        Mark P as NOISE (potential outlier, may be reclassified later)
    ELSE:
        Start new cluster C
        Add P to C
        For each point Q in N(P):
            If Q is NOISE → add to cluster C (reclassify as border point)
            If Q is unvisited:
                Mark Q as visited
                Find Q's neighbors N(Q)
                If |N(Q)| >= min_samples: add all of N(Q) to the seed set
                Add Q to cluster C
```

### 4.4 How to Choose ε and min_samples

**min_samples:** Rule of thumb = 2 × number of features (or at least 3)

**ε (eps):** Use the **k-distance plot**:
1. For each point, compute the distance to its $k$-th nearest neighbor (use k = min_samples - 1)
2. Sort these distances in ascending order
3. Plot them — the "elbow" suggests a good ε

```python
from sklearn.neighbors import NearestNeighbors
import numpy as np
import matplotlib.pyplot as plt

# k-distance plot
k = 5  # min_samples - 1
nn = NearestNeighbors(n_neighbors=k)
nn.fit(X_scaled)
distances, _ = nn.kneighbors(X_scaled)
distances = np.sort(distances[:, k-1], axis=0)

plt.plot(distances)
plt.xlabel('Points sorted by distance')
plt.ylabel(f'{k}th nearest neighbor distance')
plt.title('k-distance plot — Elbow suggests ε')
plt.show()
```

### 4.5 DBSCAN vs K-Means

| Aspect | K-Means | DBSCAN |
|--------|---------|--------|
| K needed? | Yes | No |
| Cluster shape | Spherical | Any shape |
| Outlier handling | None | Explicitly identifies noise |
| Scales to large data | Yes (fast) | Can be slow (O(N log N) with tree) |
| Cluster density | Uniform assumed | Variable density struggle |
| Deterministic? | No | Yes |

```python
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X_scaled)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)

print(f"Number of clusters: {n_clusters}")
print(f"Number of noise points: {n_noise} ({100*n_noise/len(labels):.1f}%)")
```

---

## Part 2: Dimensionality Reduction

---

## 5) Principal Component Analysis (PCA)

### 5.1 The Problem

High-dimensional data (e.g., 1000 features) suffers from:
- **Curse of dimensionality:** In high dimensions, all points are "far apart" — distances become meaningless
- **Computational cost:** Models train slowly on many features
- **Overfitting:** More features → more parameters → easier to overfit
- **Visualization:** Can't visualize more than 3 dimensions

**PCA** finds the directions of maximum variance in the data and projects the data onto a lower-dimensional subspace.

### 5.2 The Math: Eigenvectors and Eigenvalues

**Step 1: Center the data**

$$x_i \leftarrow x_i - \bar{x}, \quad \bar{x} = \frac{1}{N} \sum_{i=1}^N x_i$$

**Step 2: Compute the covariance matrix**

$$\mathbf{\Sigma} = \frac{1}{N-1} \mathbf{X}^T \mathbf{X} \in \mathbb{R}^{p \times p}$$

Entry $\Sigma_{jk}$ = covariance between feature j and feature k.

**Step 3: Compute eigenvectors and eigenvalues of Σ**

$$\mathbf{\Sigma} \mathbf{v}_j = \lambda_j \mathbf{v}_j$$

- $\mathbf{v}_j$ = **eigenvector (principal component)**: a direction in feature space
- $\lambda_j$ = **eigenvalue**: the variance of the data along direction $\mathbf{v}_j$

**Sort eigenvectors by eigenvalue** (largest first) — the first principal component $\mathbf{v}_1$ is the direction of maximum variance.

**Step 4: Project the data**

Select the top $k$ eigenvectors to form the projection matrix $\mathbf{W} \in \mathbb{R}^{p \times k}$:

$$\mathbf{Z} = \mathbf{X} \mathbf{W}$$

$\mathbf{Z} \in \mathbb{R}^{N \times k}$ is the dimensionality-reduced dataset.

### 5.3 Explained Variance Ratio

How much information does each principal component capture?

$$\text{Explained Variance Ratio (PC}_j\text{)} = \frac{\lambda_j}{\sum_{i=1}^p \lambda_i}$$

**Cumulative explained variance** tells you how many components you need to keep a certain percentage of information:

$$\text{Cumulative}(k) = \frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^p \lambda_j}$$

**Example:**
| PC | Variance | Explained Ratio | Cumulative |
|----|----------|----------------|------------|
| PC1 | 120 | 40% | 40% |
| PC2 | 90 | 30% | 70% |
| PC3 | 45 | 15% | 85% |
| PC4 | 30 | 10% | 95% |
| PC5 | 15 | 5% | 100% |

If you want to retain 95% of variance, keep the first 4 PCs (down from 5 features → only a small reduction here, but in real data with 1000 features, you might keep 50 PCs for 95% variance).

### 5.4 Intuitive Example: PCA in 2D → 1D

Imagine a 2D dataset of (height, weight) that is highly correlated (tall people tend to weigh more). The data forms an elongated ellipse.

- **PC1** = direction of maximum spread = approximately the line height/weight are both high (the "big person" axis). Variance λ₁ is large.
- **PC2** = perpendicular to PC1 = captures the spread orthogonal to the main direction (thin-tall vs heavy-short). Variance λ₂ is small.

By keeping only PC1, we compress 2D → 1D with minimal information loss.

### 5.5 PCA vs Feature Selection

| Aspect | PCA | Feature Selection |
|--------|-----|------------------|
| Creates new features? | Yes (linear combinations) | No (subsets of originals) |
| Interpretability | Low (PC1 is a mix of all features) | High (keeps original features) |
| Handles correlated features? | Yes (decorrelates them) | No |
| Use for | Visualization, compression, preprocessing | When interpretability matters |

### 5.6 Implementation

```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

# Always scale before PCA!
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit PCA
pca = PCA()  # Compute all components first
pca.fit(X_scaled)

# Explained variance plot (Scree plot)
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.bar(range(1, len(pca.explained_variance_ratio_)+1), pca.explained_variance_ratio_)
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')
plt.title('Scree Plot')

plt.subplot(1, 2, 2)
cumvar = np.cumsum(pca.explained_variance_ratio_)
plt.plot(range(1, len(cumvar)+1), cumvar, marker='o')
plt.axhline(y=0.95, color='r', linestyle='--', label='95% variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('How many components to keep?')
plt.legend()
plt.tight_layout()
plt.show()

# Choose number of components
n_components = np.argmax(cumvar >= 0.95) + 1
print(f"Components needed for 95% variance: {n_components}")

# Fit with chosen number of components
pca_final = PCA(n_components=n_components)
X_pca = pca_final.fit_transform(X_scaled)
print(f"Reduced shape: {X_pca.shape} (was {X_scaled.shape})")

# 2D visualization (for any dataset)
pca_2d = PCA(n_components=2)
X_2d = pca_2d.fit_transform(X_scaled)

plt.figure(figsize=(8, 6))
scatter = plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y, cmap='viridis', alpha=0.7)
plt.colorbar(scatter)
plt.xlabel(f'PC1 ({pca_2d.explained_variance_ratio_[0]:.1%} variance)')
plt.ylabel(f'PC2 ({pca_2d.explained_variance_ratio_[1]:.1%} variance)')
plt.title('PCA — 2D Projection')
plt.show()
```

---

## 6) t-SNE and UMAP (Visualization Only)

PCA is linear and may not capture non-linear structure. **t-SNE (t-distributed Stochastic Neighbor Embedding)** and **UMAP (Uniform Manifold Approximation and Projection)** are non-linear dimensionality reduction methods for visualization.

### 6.1 t-SNE Intuition

t-SNE preserves **local structure** (nearby points in high dimensions stay nearby in 2D):

1. Convert distances between high-dimensional points to **probabilities** (Gaussian kernel):
$$p_{j|i} = \frac{\exp(-||x_i - x_j||^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-||x_i - x_k||^2 / 2\sigma_i^2)}$$

2. Initialize 2D positions randomly, define similar probabilities in 2D using a **t-distribution** (heavier tails to prevent crowding):
$$q_{ij} = \frac{(1 + ||y_i - y_j||^2)^{-1}}{\sum_{k \neq l} (1 + ||y_k - y_l||^2)^{-1}}$$

3. Minimize the **KL divergence** between the high-dimensional and 2D probability distributions using gradient descent.

**Important caveats for t-SNE:**
- Distances/sizes of clusters are **NOT meaningful** — only topology
- Run multiple times with different seeds
- `perplexity` hyperparameter controls the balance between local and global structure (typical: 5–50)
- **Very slow** on large datasets (O(N²))

### 6.2 UMAP

UMAP is newer, faster, and often better than t-SNE:
- Preserves more global structure
- Much faster (O(N log N))
- Can be used for actual dimensionality reduction (not just visualization)

```python
# t-SNE
from sklearn.manifold import TSNE

tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.title('t-SNE Visualization')
plt.show()

# UMAP
import umap
reducer = umap.UMAP(n_components=2, n_neighbors=15, min_dist=0.1, random_state=42)
X_umap = reducer.fit_transform(X_scaled)

plt.scatter(X_umap[:, 0], X_umap[:, 1], c=y, cmap='tab10', alpha=0.7)
plt.title('UMAP Visualization')
plt.show()
```

---

## 7) Evaluating Clustering

Clustering evaluation is hard because there's no ground truth. Two scenarios:

### 7.1 When You Have Ground Truth Labels

| Metric | Description | Range |
|--------|-------------|-------|
| **Adjusted Rand Index (ARI)** | How similar predicted clusters are to true labels, adjusted for chance | -1 to 1 (1 = perfect) |
| **Normalized Mutual Information (NMI)** | Information shared between predicted and true clusters | 0 to 1 (1 = perfect) |
| **Homogeneity** | Each cluster contains only members of a single class | 0 to 1 |
| **Completeness** | All members of a class are in the same cluster | 0 to 1 |

### 7.2 Without Ground Truth Labels

| Metric | Description | Range |
|--------|-------------|-------|
| **Silhouette Score** | How well-separated clusters are | -1 to 1 (higher better) |
| **Davies-Bouldin Index** | Average ratio of within-cluster spread to between-cluster separation | Lower is better |
| **Calinski-Harabasz Index** | Ratio of between-cluster to within-cluster variance | Higher is better |

```python
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             silhouette_score, davies_bouldin_score,
                             calinski_harabasz_score)

# With ground truth
ari = adjusted_rand_score(y_true, labels)
nmi = normalized_mutual_info_score(y_true, labels)

# Without ground truth
sil = silhouette_score(X_scaled, labels)
db = davies_bouldin_score(X_scaled, labels)
ch = calinski_harabasz_score(X_scaled, labels)

print(f"Silhouette: {sil:.3f} (higher = better)")
print(f"Davies-Bouldin: {db:.3f} (lower = better)")
print(f"Calinski-Harabasz: {ch:.1f} (higher = better)")
```

---

## 8) Summary — When to Use What

| Algorithm | Use When |
|-----------|----------|
| **K-Means** | Spherical clusters, K known, large datasets, need speed |
| **Hierarchical** | Want to explore all K options, small/medium dataset, need dendrogram |
| **DBSCAN** | Unknown number of clusters, irregular shapes, need outlier detection |
| **PCA** | Reduce dimensions before modeling, correlated features, visualization |
| **t-SNE / UMAP** | 2D/3D visualization only, exploring cluster structure |

---

## Key Takeaways

1. **K-Means** minimizes inertia using the EM algorithm; always use K-Means++ init and standardize features first.

2. **Hierarchical clustering** builds a full dendrogram — cut at the right height to get any K; Ward linkage is typically best.

3. **DBSCAN** needs no K, handles arbitrary shapes, and explicitly labels outliers; choose ε via the k-distance plot elbow.

4. **PCA** projects data onto directions of maximum variance; choose components to retain 95%+ of variance; always scale first.

5. **Silhouette score** is the most reliable metric for comparing clustering solutions when ground truth is unavailable.
