# Unsupervised Learning: Clustering

## 1. Problem

Most of the data that exists has no labels. Nobody sat down and tagged every customer transaction as "fraud/not fraud," every user session as "churner/loyal," or every gene-expression profile as "cancer subtype A/B/C." Yet there is often real structure hiding in that unlabeled data — natural groups of similar points, dense regions separated by sparse ones, redundant or correlated dimensions.

**Unsupervised learning** is the task of finding that structure without being told what to look for. There is no $y$. Given only a matrix of feature vectors $X \in \mathbb{R}^{N \times p}$, the goal is to discover groupings, densities, or a lower-dimensional description of the data on its own terms.

This note covers **clustering** — partitioning data into groups of similar points: K-Means, Hierarchical Clustering, and DBSCAN, plus how to evaluate a clustering with no ground truth (Silhouette Analysis). Dimensionality reduction (PCA) is covered separately in `18-pca`, since it answers a different question ("which directions matter?" rather than "which group do you belong to?") — but the underlying problem (no labels, structure must be discovered) is the same, and PCA is frequently used as a preprocessing step before clustering high-dimensional data.

## 2. Intuition

Imagine handing someone a box of thousands of unlabeled photographs and asking them to organize it. With no captions telling them "this is a dog," "this is a beach," they still manage to sort photos into piles — beach photos together, dog photos together, blurry photos in their own pile — purely by noticing which photos *look like* which other photos. Nobody supervised the sorting; the structure came from similarity itself.

That's the entire idea behind clustering: define a notion of "similar" (usually distance in feature space) and group points so that within-group similarity is high and between-group similarity is low.

Three different ways to formalize "similar":
- **K-Means**: a point belongs with whichever group has the closest *center*. Groups are defined by their centroid.
- **Hierarchical clustering**: don't commit to one grouping — build the entire family tree of groupings, from every point alone up to one giant cluster, and let the analyst pick a cut height afterward.
- **DBSCAN**: a point belongs with a group if it sits in a dense neighborhood connected to other dense neighborhoods, regardless of the group's overall shape. No centroid required.

## 3. Why simpler approaches fail

Supervised methods (logistic regression, decision trees, gradient boosting, ...) all work the same way at bottom: define a loss function that measures the gap between a prediction and the true label $y_i$, then minimize that loss. Every piece of that machinery — cross-entropy, squared error, information gain — is a function of $(\hat{y}_i, y_i)$.

**With no labels, there is nothing to optimize against.** There is no "ground truth" to compare a candidate grouping to, so a supervised loss simply cannot be written down. This is not a matter of a supervised method performing poorly on unlabeled data — it's a category error; supervised methods have no defined objective at all without $y$.

Unsupervised methods sidestep this by defining a *self-referential* objective computed purely from $X$: how tightly grouped are the points assigned to each cluster ($K$-Means's within-cluster sum of squares), how far apart are two groups before they're allowed to merge (hierarchical linkage), how connected is a dense region (DBSCAN's density-reachability). The absence of $y$ is exactly why these objectives had to be invented — they are what supervised learning did not need.

A second, related failure: even naive heuristics like "just eyeball scatter plots and draw circles around groups" break down past 2–3 dimensions, and break down with any real number of points. Clustering algorithms replace human eyeballing with an explicit, repeatable mathematical criterion.

## 4. Mathematical foundation

### 4.1 K-Means: minimizing within-cluster sum of squares

K-Means partitions $N$ points into $K$ clusters by minimizing the **within-cluster sum of squares (WCSS)**, also called **inertia**:

$$J(\{C_k\}, \{\mu_k\}) = \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

where $C_k$ is the set of points assigned to cluster $k$, and $\mu_k$ is that cluster's centroid (mean).

$J$ depends on two things simultaneously: which points are assigned to which cluster ($\{C_k\}$), and where each cluster's centroid sits ($\{\mu_k\}$). Jointly minimizing over both is combinatorially hard (choosing an optimal partition of $N$ points into $K$ groups is NP-hard). K-Means instead uses **alternating minimization**: fix one set of variables, optimize the other exactly, then swap and repeat.

**Step A — fix centroids, optimize assignments.** If $\{\mu_k\}$ is fixed, $J$ decomposes into an independent term per point:

$$J = \sum_{i=1}^N \|x_i - \mu_{c_i}\|^2$$

Each point's contribution depends only on which cluster it's assigned to. The minimum over $c_i$ for a fixed $x_i$ is achieved by assigning $x_i$ to whichever centroid is nearest:

$$c_i = \arg\min_k \|x_i - \mu_k\|^2$$

This is exact and trivial — no iteration needed once centroids are fixed.

**Step B — fix assignments, optimize centroids.** If $\{C_k\}$ is fixed, $J$ decomposes into one independent sub-problem per cluster:

$$J_k(\mu_k) = \sum_{x_i \in C_k} \|x_i - \mu_k\|^2$$

This is a sum of squared distances to a single point $\mu_k$ — a classic least-squares problem. Take the gradient and set it to zero:

$$\nabla_{\mu_k} J_k = -2\sum_{x_i \in C_k} (x_i - \mu_k) = 0 \implies \mu_k = \frac{1}{|C_k|}\sum_{x_i \in C_k} x_i$$

The minimizer is exactly the mean of the assigned points — which is where the name "K-**Means**" comes from.

**Why alternating minimization converges.** Each step (A or B) can only decrease $J$ or leave it unchanged — never increase it, because each step solves its sub-problem exactly. $J \geq 0$ is bounded below, and the number of possible distinct partitions of $N$ points into $K$ groups is finite, so the sequence of $J$ values is non-increasing and bounded below: it must converge in a finite number of steps (in practice, it stops as soon as no point changes cluster). This guarantees convergence to *a* local minimum — not necessarily the global one, since a different starting point can lead the alternating process into a different stable configuration. See §9 for what this means practically.

### 4.2 Hierarchical clustering: linkage criteria

Instead of one flat partition, agglomerative hierarchical clustering starts with $N$ singleton clusters and repeatedly merges the two closest clusters until one cluster remains, recording the whole merge history as a tree. "Closest" between two *clusters* (as opposed to two points) needs its own definition — this is the **linkage criterion**:

| Linkage | Distance between clusters $A$, $B$ | Effect |
|---|---|---|
| **Single** | $\min_{a\in A,\, b\in B} d(a,b)$ | Chains together nearest neighbors; tends to produce long, straggly clusters |
| **Complete** | $\max_{a\in A,\, b\in B} d(a,b)$ | Merges only when *every* pair is close; produces compact, roughly equal-sized clusters |
| **Average** | $\frac{1}{|A||B|}\sum_{a\in A}\sum_{b\in B} d(a,b)$ | A compromise between single and complete |
| **Ward** | The increase in total WCSS ($J$ from §4.1) caused by merging $A$ and $B$ | Directly minimizes the same objective K-Means minimizes; the most commonly used default |

Ward's linkage merges whichever pair of clusters causes the **smallest increase in inertia** — it is, in effect, greedily building toward the same objective K-Means optimizes, but via bottom-up merges instead of alternating minimization, and without committing to a fixed $K$ in advance.

### 4.3 DBSCAN: density-reachability

DBSCAN defines a cluster not by a centroid or a merge order, but by density connectivity, using two hyperparameters: a neighborhood radius $\varepsilon$ and a minimum neighbor count `min_samples`.

- **Core point**: a point with at least `min_samples` points (including itself) within radius $\varepsilon$.
- **Border point**: not a core point itself, but lies within $\varepsilon$ of some core point.
- **Noise point**: neither core nor border — an outlier, labeled $-1$.

**Directly density-reachable**: point $B$ is directly density-reachable from $A$ if $A$ is a core point and $B$ lies within distance $\varepsilon$ of $A$.

**Density-reachable**: $B$ is density-reachable from $A$ if there exists a chain $A = p_1, p_2, \dots, p_n = B$ where each $p_{i+1}$ is directly density-reachable from $p_i$. This is transitive closure over "directly density-reachable" — it lets a cluster snake through an arbitrarily shaped dense region, one core point's neighborhood at a time.

**A cluster** is a maximal set of points that are mutually density-connected through this chain of core points. Because this definition never assumes a centroid or a convex shape, DBSCAN can discover clusters of arbitrary geometry — crescents, rings, nested shapes — that K-Means and (to a lesser extent) hierarchical clustering cannot.

## 5. Algorithm

**K-Means (Lloyd's algorithm):**
```
1. Choose K.
2. Initialize K centroids (random, or K-Means++ — see below).
3. REPEAT:
   a. Assignment step: for each point x_i, c_i = argmin_k ||x_i - mu_k||^2
   b. Update step: for each cluster k, mu_k = mean of points assigned to k
   UNTIL assignments stop changing (or centroid movement < tol)
```
This is exactly the alternating minimization of §4.1, steps A and B, repeated to convergence.

**Agglomerative hierarchical clustering:**
```
1. Start: N clusters, each a single point.
2. REPEAT until 1 cluster remains:
   a. Find the two closest clusters under the chosen linkage.
   b. Merge them; record the merge height (distance) in the dendrogram.
3. Cut the resulting tree at any height to obtain any number of flat clusters K.
```
Reading a dendrogram to choose $K$: look for the largest vertical gap between merge heights (the longest line segment the cut doesn't cross); a horizontal cut through that gap crosses as many lines as the natural number of clusters.

**DBSCAN:**
```
Mark all points unvisited.
FOR each unvisited point P:
    Mark P visited.
    N(P) = points within radius eps of P.
    IF |N(P)| < min_samples:
        Mark P as NOISE (may be reclassified as a border point later).
    ELSE:
        Start new cluster C; add P to C.
        For each point Q in N(P):
            IF Q was NOISE: reclassify Q as a border point of C.
            IF Q is unvisited:
                Mark Q visited; compute N(Q).
                IF |N(Q)| >= min_samples: add N(Q) to the seed set to expand C.
                Add Q to C.
```

**K-Means++ initialization** (briefly — full weight given in §9, since it exists specifically to fix K-Means's initialization sensitivity): instead of picking all $K$ initial centroids uniformly at random, pick the first centroid uniformly at random, then repeatedly pick the next centroid with probability proportional to its squared distance from the nearest centroid already chosen. This spreads the initial centroids out across the data rather than risking two of them landing close together, which empirically produces faster convergence and better final solutions. `sklearn`'s `KMeans` defaults to `init='k-means++'`.

## 6. From-scratch implementation

Implemented as `KMeansScratch` in `01-kmeans-clustering/KMeans-Clustering.ipynb` (cell 12) — a full Lloyd's-algorithm loop (assignment step / update step, alternating until centroid movement falls below tolerance, with multiple random restarts to reduce the local-minimum risk described in §9), validated against `sklearn.cluster.KMeans` on the same data (inertia and centroids match closely).

To make the *iteration-by-iteration convergence* visible (rather than only the final converged state), a second, minimal implementation was added to the same notebook: a bare assign→recompute loop on a toy 2D three-blob dataset that records the centroid positions and inertia after every iteration, then plots the cluster assignment at several snapshots (initialization, iteration 1, iteration 2, convergence) side by side with inertia vs. iteration number. The plot makes concrete what §4.1 proves algebraically: $J$ drops sharply in the first iteration or two and then flattens out as assignments stabilize.

## 7. Practical implementation

The production version in all four notebooks is `sklearn.cluster.KMeans`, `scipy.cluster.hierarchy.linkage`/`dendrogram`/`fcluster`, and `sklearn.cluster.DBSCAN`. Each maps directly back to §5:

- `KMeans(n_clusters=K, init='k-means++', n_init=10)` runs Lloyd's algorithm (§5) `n_init` times with different K-Means++ initializations and keeps the run with lowest inertia — exactly the from-scratch loop in §6, with the K-Means++ initialization from §5 and multiple restarts baked in.
- `linkage(X, method='ward')` builds the merge tree bottom-up using Ward's linkage (§4.2); `dendrogram()` visualizes it; `fcluster(Z, t=K, criterion='maxclust')` cuts it at a given number of clusters.
- `DBSCAN(eps=..., min_samples=...)` implements the core/border/noise algorithm of §5 directly.

**Always scale features first** (`StandardScaler`) for all three algorithms — each depends on Euclidean distance, and any feature with a larger numeric range will dominate the distance calculation regardless of its actual importance.

Choosing hyperparameters: the elbow method plots inertia vs. $K$ for K-Means (looking for where the inertia-vs-$K$ curve stops dropping sharply); DBSCAN's $\varepsilon$ is chosen from a **k-distance plot** — sort each point's distance to its $k$-th nearest neighbor (with $k \approx$ `min_samples` $- 1$) and look for the elbow, since points beyond that distance are typically noise. `min_samples` has a rule of thumb of roughly $2\times$ the number of features (at least 3).

Full worked example, elbow-method table, and k-distance-plot code live in the four topic notebooks:
- `01-kmeans-clustering/KMeans-Clustering.ipynb`
- `02-hierarchical-clustering/Hierarchical-Clustering.ipynb`
- `03-dbscan-clustering/DBSCAN-Clustering.ipynb`
- `04-silhouette-analysis/Silhouette-Analysis.ipynb`

## 8. Experiment

**Hypothesis:** if synthetic data is generated with a known number of well-separated blob clusters (say, 4), the silhouette score computed for candidate $K \in \{2, \dots, 15\}$ should peak at (or very near) $K=4$ — the true number of clusters — since the silhouette score directly measures how well-separated and internally cohesive a candidate clustering is.

$$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}, \qquad s(i) \in [-1, 1]$$

where $a(i)$ is the mean distance from point $i$ to other points in its own cluster (cohesion — lower is tighter) and $b(i)$ is the mean distance from $i$ to points in the nearest *other* cluster (separation — higher is more distinct). Averaging $s(i)$ over all points gives one number per candidate $K$; the $K$ that maximizes the average silhouette is the recommended choice.

This experiment — generate synthetic blobs with a known ground-truth $K$, sweep $K$ from 2 to 15, plot the average silhouette score against $K$, and check whether the peak lands on the true $K$ — is exactly what `04-silhouette-analysis/Silhouette-Analysis.ipynb` runs. **Result:** the silhouette curve peaks at the true number of generating clusters on the synthetic dataset, confirming the hypothesis; on messier or overlapping clusters the peak becomes less sharp, and silhouette is noted there as *more reliable than the elbow method* (which can show an ambiguous "elbow" even on clean data) because it has a bounded, interpretable range and directly rewards separation rather than just penalizing spread. **Limitation:** silhouette score itself assumes roughly convex clusters measured by Euclidean distance — on DBSCAN-style arbitrarily-shaped clusters it can be misleading, since it still measures compactness/separation geometrically rather than density-connectivity.

## 9. Failure modes

- **K-Means assumes spherical, similarly-sized clusters.** Because the objective is squared Euclidean distance to a single centroid, K-Means implicitly models each cluster as an isotropic Gaussian blob. Elongated, non-convex, or very differently-sized clusters get sliced incorrectly (e.g. two crescent-moon clusters get cut straight through). DBSCAN or Gaussian Mixture Models handle this better.
- **K requires being chosen in advance**, and K-Means has no internal signal for "this K is wrong" — the elbow method and silhouette score (§8) are external diagnostics bolted on afterward, not something the algorithm reports itself. Hierarchical clustering sidesteps this by deferring the choice to after the tree is built; DBSCAN sidesteps it entirely by inferring the number of clusters from density.
- **Sensitive to initialization.** Because Lloyd's algorithm only guarantees convergence to *a* local minimum of $J$ (§4.1), different random starting centroids can converge to different, sometimes much worse, final clusterings. **K-Means++** (§5) mitigates this by spreading out the initial centroids probabilistically rather than picking them uniformly at random; running with multiple restarts (`n_init`) and keeping the lowest-inertia result further reduces (but does not eliminate) the risk.
- **Sensitive to feature scaling.** Since every algorithm here (K-Means, Ward linkage, DBSCAN) is built on Euclidean distance, a feature measured in the thousands will dominate a feature measured in single digits regardless of which one is actually informative. `StandardScaler` before fitting is not optional.
- **Sensitive to outliers.** A single far-away point pulls a K-Means centroid toward it (since the objective is a *sum of squares*, and squares punish large deviations disproportionately). DBSCAN handles this better by design — it labels sparse/far points as noise rather than forcing them into a cluster.
- **Hierarchical clustering doesn't scale.** Building and storing the full pairwise distance matrix is $O(N^2)$ in both time and memory, which becomes impractical past roughly tens of thousands of points.
- **DBSCAN struggles with varying density.** A single global $\varepsilon$ cannot simultaneously suit a tight cluster and a loose one — points in the loose cluster may all get marked as noise, or the tight cluster and loose cluster may get merged incorrectly. (Density-based methods designed for varying density, like HDBSCAN, exist but are out of scope here.)

## 10. Real-world usage

- **Customer segmentation** — grouping customers by purchase behavior for targeted marketing, typically K-Means or hierarchical clustering on engineered behavioral features.
- **Anomaly/outlier flagging as a side effect** — DBSCAN's noise label (`-1`) is reused directly as an anomaly detector (see `16-anomaly-detection`).
- **Document/topic clustering** — grouping similar text embeddings.
- **Image compression / color quantization** — K-Means on pixel colors reduces an image to $K$ representative colors.
- **Preprocessing before supervised learning** — cluster assignments or distances-to-centroid used as engineered features.

**Evaluating a clustering in production**, when ground-truth labels are unavailable (the common case): use internal metrics — **silhouette score** (§8), **Davies-Bouldin index** (lower is better; ratio of within-cluster spread to between-cluster separation), and **Calinski-Harabasz index** (higher is better; ratio of between-cluster to within-cluster variance). When ground-truth *is* available for validation purposes (e.g. a held-out labeled subset), external metrics apply: **Adjusted Rand Index** (agreement with true labels, corrected for chance, range $[-1, 1]$), **Normalized Mutual Information**, **Homogeneity** (each cluster contains only one true class) and **Completeness** (all members of a true class land in one cluster).

## 11. Mental model

**K-Means alternates between "who belongs to which group" and "where is each group's center" until neither answer changes.** Hierarchical clustering builds the entire family tree of every possible grouping and lets you pick a cut afterward. DBSCAN doesn't ask "which center is closest" at all — it asks "am I in a dense, connected neighborhood," which is what lets it find any-shaped clusters and call sparse points what they are: noise, not a forced-fit group member.

## 12. Questions to think about

1. K-Means's alternating minimization is guaranteed to never increase $J$. Why does that *not* imply it will find the global minimum of $J$? Construct (conceptually) a small dataset and initialization where Lloyd's algorithm gets stuck in a bad local minimum.
2. Ward's linkage minimizes the increase in the same WCSS objective K-Means minimizes. Given that, why might Ward's linkage hierarchical clustering and K-Means still produce different final clusters for the same $K$?
3. DBSCAN has no notion of a "centroid." What does it mean, conceptually, for a border point to belong to a cluster despite not itself satisfying the core-point density condition?
4. If you ran the elbow method and the silhouette score sweep on the same dataset and they disagreed on the best $K$, which would you trust more, and why — referring back to what each one actually measures?
5. Why does feature scaling matter identically for K-Means, Ward-linkage hierarchical clustering, and DBSCAN, even though the three algorithms define "cluster" in completely different ways?
