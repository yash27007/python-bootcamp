# 15 – Unsupervised Learning

Unsupervised learning discovers hidden structure in unlabeled data. This section covers the two major tasks: **clustering** (finding groups) and **dimensionality reduction** (compressing data while preserving structure).

---

## Folder Structure

```
15-unsupervised-learning/
├── 01-kmeans-clustering/
│   └── KMeans-Clustering.ipynb
├── 02-hierarchical-clustering/
│   └── Hierarchical-Clustering.ipynb
├── 03-dbscan-clustering/
│   └── DBSCAN-Clustering.ipynb
└── 04-silhouette-analysis/
    └── Silhouette-Analysis.ipynb
```

---

## Topics

| # | Topic | Notebook | Status |
|---|-------|----------|--------|
| 1 | **K-Means Clustering** | [KMeans-Clustering.ipynb](01-kmeans-clustering/KMeans-Clustering.ipynb) | ✅ Complete |
| 2 | **Hierarchical Clustering** | [Hierarchical-Clustering.ipynb](02-hierarchical-clustering/Hierarchical-Clustering.ipynb) | ✅ Complete |
| 3 | **DBSCAN Clustering** | [DBSCAN-Clustering.ipynb](03-dbscan-clustering/DBSCAN-Clustering.ipynb) | ✅ Complete |
| 4 | **Silhouette Analysis** | [Silhouette-Analysis.ipynb](04-silhouette-analysis/Silhouette-Analysis.ipynb) | ✅ Complete |

---

## 1 — K-Means Clustering

**Objective**: Minimize within-cluster sum of squares (inertia):

$$J = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

**Algorithm (Lloyd's)**: Alternate between (1) assigning each point to the nearest centroid and (2) moving each centroid to the mean of its assigned points. Guaranteed to converge.

**Key topics covered**:
- Mathematical proof of convergence
- K-Means++ initialization — $O(\log K)$ approximation guarantee
- Elbow method for choosing K
- Limitations: spherical clusters, scale sensitivity, outlier sensitivity
- Full implementation from scratch

---

## 2 — Hierarchical Clustering

**Idea**: Build a full tree (dendrogram) of all possible cluster groupings. Cut the tree at any height to get any number of clusters K — no need to specify K upfront.

**Linkage criteria**:

| Method | Formula | Best for |
|--------|---------|---------|
| Single | $\min_{a \in A, b \in B} d(a,b)$ | Elongated chains |
| Complete | $\max_{a \in A, b \in B} d(a,b)$ | Compact clusters |
| Average | $\frac{1}{\|A\|\|B\|}\sum d(a,b)$ | Compromise |
| **Ward** | $\frac{\|A\|\|B\|}{\|A\|+\|B\|}\|\mu_A-\mu_B\|^2$ | **Most popular** |

**Key topics covered**:
- Lance-Williams update formula
- Ward's method derivation (minimizes WCSS increase)
- Reading and cutting dendrograms
- Complexity: $O(n^2)$ space — not scalable past ~10K points

---

## 3 — DBSCAN

**Core idea**: A cluster is a dense region of points separated from other dense regions by sparse regions.

**Point types**:
- **Core point**: ≥ `min_samples` neighbors within radius ε
- **Border point**: within ε of a core point, but not a core point itself
- **Noise point**: neither core nor border — labeled **-1**

**Two hyperparameters**:
- **ε (eps)**: neighborhood radius → choose via k-distance plot elbow
- **min_samples**: density threshold → rule of thumb: `2 × n_features`

**Key topics covered**:
- Formal definitions: density-reachability, density-connectivity
- BFS-based cluster expansion algorithm
- k-distance plot for ε selection
- Arbitrary cluster shapes, explicit outlier detection
- Full implementation from scratch

---

## 4 — Silhouette Analysis

**Purpose**: Evaluate clustering quality without ground-truth labels (internal metric).

**Formula**: For each point $i$:

$$s(i) = \frac{b(i) - a(i)}{\max(a(i),\, b(i))} \in [-1, 1]$$

- $a(i)$ = mean distance to points in the same cluster (cohesion)
- $b(i)$ = mean distance to the nearest other cluster (separation)
- $s(i) \approx +1$: well-clustered; $s(i) \approx 0$: on boundary; $s(i) < 0$: likely misassigned

**Key topics covered**:
- Full mathematical derivation
- Silhouette plots (per-point visualization)
- Using silhouette to choose K (more reliable than elbow)
- Other internal metrics: Davies-Bouldin Index, Calinski-Harabasz Index
- External metrics: ARI, NMI, V-Measure (when labels are available)

---

## Quick Algorithm Reference

| Algorithm | K needed? | Cluster shape | Outliers | Scales to large n? |
|-----------|-----------|---------------|----------|-------------------|
| K-Means | Yes | Spherical | No | Yes |
| Hierarchical | No (post-hoc) | Flexible | Partial | No ($O(n^2)$) |
| DBSCAN | No | Any shape | Explicit | Yes ($O(n \log n)$) |

---

## Key Prerequisites

- StandardScaler before any clustering algorithm
- PCA for 2D visualization of high-dimensional results
- Always evaluate with at least one internal metric (silhouette) + visual inspection
