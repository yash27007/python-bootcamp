# Principal Component Analysis (PCA)

## 1. Problem

Real datasets often have far more features than are actually informative. A tabular dataset might have hundreds of columns; an image is a vector of thousands of pixel intensities. High-dimensional data creates three concrete problems:

- **It's hard to visualize.** Humans can plot 2 or 3 dimensions; a dataset with 30 features cannot be looked at directly.
- **It's expensive to model.** More features mean more parameters, slower training, and (per the curse of dimensionality — see `09-knn`'s Failure modes for the distance-concentration argument) sparser, less informative geometry to learn from.
- **Features are often redundant or correlated.** Height and weight, or a dozen pixel intensities in the same region of an image, carry overlapping information. A model spends capacity re-learning the same correlation structure across multiple correlated inputs instead of using that capacity on genuinely independent signal.

**PCA** addresses this by finding a small number of new, uncorrelated directions that capture most of the variation in the data, so the data can be described — approximately, but with minimal information loss — using far fewer numbers.

## 2. Intuition

Picture a 2D scatter plot of (height, weight) for a population. Because taller people tend to weigh more, the points don't fill up the whole plane evenly — they form a tilted, elongated ellipse, or cigar shape, oriented roughly along the "bigger people" direction.

If you had to describe each point using only *one* number instead of two, which axis would you use? Not the original height axis, and not the original weight axis — the best single number is the position **along the long axis of the cigar** — the direction the data actually stretches out in. Most of the spread of the data is captured by that one direction; the leftover spread, perpendicular to it (roughly "tall-and-thin vs. short-and-heavy" once you've accounted for overall size), is much smaller.

That long axis is the **first principal component**. It isn't one of the original features (height or weight) — it's a new axis, a specific linear combination of both, chosen because the data varies the most along it. The second principal component is the next-best direction, forced to be perpendicular to the first, capturing whatever variance is left over. PCA is exactly this idea generalized to $p$ dimensions: rotate the coordinate system to align with the directions the data actually varies in, then keep only the first few.

## 3. Why simpler approaches fail

**Dropping features by hand** — e.g. "let's just remove half the columns" — throws away information arbitrarily. There's no principled way to know, by inspection, which raw features are safe to discard, especially once there are dozens of them and the important signal is spread across combinations of features rather than sitting in any one of them.

**Keeping all correlated features as-is** doesn't reduce dimensionality, and it actively wastes model capacity: two features that are 95% redundant with each other are still treated by most models as two separate, independent inputs, meaning the model spends parameters and training data essentially re-deriving the same correlation the raw features already encode. A model trained on 500 correlated features doesn't get 500 features' worth of *independent* signal — it gets far less, but pays the full computational and overfitting cost of 500 dimensions anyway.

What's needed is a way to find the directions where the data actually carries independent information, and to rank them by how much variance (i.e., how much distinguishing information) each one carries — so that the least informative directions can be discarded first, with a quantifiable, not arbitrary, amount of information lost. That is precisely what PCA computes.

## 4. Mathematical foundation

### 4.1 Variance and covariance

For a single centered feature $x$ (mean subtracted, so $\bar{x}=0$), variance measures spread:

$$\text{Var}(x) = \frac{1}{N-1}\sum_{i=1}^N x_i^2$$

For two centered features $x$ and $y$, covariance measures how much they vary *together*:

$$\text{Cov}(x, y) = \frac{1}{N-1}\sum_{i=1}^N x_i y_i$$

Positive covariance means the features tend to move in the same direction (like height and weight); covariance near zero means they vary independently.

### 4.2 The covariance matrix

For a centered data matrix $X \in \mathbb{R}^{N \times p}$ (each of the $N$ rows is a point, each of the $p$ columns a feature, and each column has had its mean subtracted), the full **covariance matrix** collects every pairwise variance/covariance at once:

$$\Sigma = \frac{1}{N-1} X^T X \in \mathbb{R}^{p \times p}$$

Entry $\Sigma_{jk}$ is the covariance between feature $j$ and feature $k$; the diagonal $\Sigma_{jj}$ is just the variance of feature $j$. $\Sigma$ is symmetric ($\Sigma_{jk} = \Sigma_{kj}$) and positive semi-definite.

### 4.3 Why the eigenvectors of the covariance matrix are the principal components

The goal, formalized: find a unit direction $v \in \mathbb{R}^p$ ($\|v\|=1$) such that projecting the data onto $v$ maximizes the variance of the projection. The projection of point $x_i$ onto direction $v$ is the scalar $x_i^T v$, so the variance of the projected data is:

$$\text{Var}(Xv) = \frac{1}{N-1}\sum_i (x_i^T v)^2 = \frac{1}{N-1} v^T X^T X v = v^T \Sigma v$$

So the problem is: maximize $v^T \Sigma v$ subject to the constraint $\|v\|^2 = v^Tv = 1$ (without a unit-length constraint, $v^T\Sigma v$ could be made arbitrarily large just by scaling $v$ up — the constraint forces the optimization to be about *direction*, not magnitude).

This is a constrained optimization problem — solve it with a Lagrange multiplier $\lambda$:

$$\mathcal{L}(v, \lambda) = v^T \Sigma v - \lambda (v^T v - 1)$$

Take the gradient with respect to $v$ and set it to zero:

$$\nabla_v \mathcal{L} = 2\Sigma v - 2\lambda v = 0 \quad\Longrightarrow\quad \Sigma v = \lambda v$$

This is exactly the **eigenvector equation**. The direction $v$ that maximizes variance subject to being a unit vector must be an eigenvector of $\Sigma$, and the corresponding Lagrange multiplier $\lambda$ is its eigenvalue. Substituting back:

$$v^T \Sigma v = v^T (\lambda v) = \lambda (v^Tv) = \lambda$$

So **the variance captured along eigenvector $v$ is exactly its eigenvalue $\lambda$.** Since $\Sigma$ is a real symmetric $p \times p$ matrix, it has exactly $p$ real eigenvalues and an orthonormal set of $p$ eigenvectors (spectral theorem). Sorting them by eigenvalue from largest to smallest gives:

- $v_1$ (largest $\lambda_1$): the direction of maximum variance — the **first principal component**.
- $v_2$ (next largest $\lambda_2$, constrained orthogonal to $v_1$ by the eigendecomposition itself): the direction of maximum *remaining* variance — the **second principal component**.
- ...and so on through $v_p$.

Each eigenvector is a principal component (a direction in feature space); each eigenvalue is the amount of variance the data has along that direction.

### 4.4 Projection onto the top-$k$ components

Stack the top $k$ eigenvectors (by eigenvalue) as columns of a projection matrix $W \in \mathbb{R}^{p\times k}$. The dimensionality-reduced representation is:

$$Z = XW \in \mathbb{R}^{N \times k}$$

Each row of $Z$ is the original point's coordinates in the new, lower-dimensional basis defined by the top $k$ principal components.

### 4.5 Explained variance ratio

How much of the total variance does component $j$ capture?

$$\text{EVR}(j) = \frac{\lambda_j}{\sum_{i=1}^p \lambda_i}$$

and the cumulative variance retained by keeping the top $k$ components:

$$\text{Cumulative}(k) = \frac{\sum_{j=1}^k \lambda_j}{\sum_{j=1}^p \lambda_j}$$

This is what makes PCA's information loss *quantifiable* rather than arbitrary (§3): you can choose $k$ to retain, say, 95% of the total variance, and know precisely how much was traded away.

## 5. Algorithm

```
1. Center the data: X <- X - mean(X, axis=0)     (and standardize — see §7 — if features
                                                    are on different scales)
2. Compute the covariance matrix: Sigma = (1/(N-1)) X^T X
3. Compute eigenvalues/eigenvectors of Sigma: Sigma v_j = lambda_j v_j
4. Sort eigenvectors by eigenvalue, descending.
5. Choose k (e.g. via cumulative explained variance >= a target threshold).
6. Form W from the top k eigenvectors; project: Z = X W
```

## 6. From-scratch implementation

A minimal NumPy implementation follows §5 directly on a small toy dataset: center the data, compute the covariance matrix with `X_centered.T @ X_centered / (N-1)`, get eigenvectors/eigenvalues with `np.linalg.eig` (or the numerically preferred `np.linalg.eigh` for symmetric matrices), sort by eigenvalue descending, and project onto the top 2 eigenvectors. The resulting 2D projection is plotted side-by-side against `sklearn.decomposition.PCA(n_components=2)` fit on the same data — the two scatter plots match up to an axis sign flip (eigenvectors are only defined up to sign, since $v$ and $-v$ satisfy the same eigenvector equation and capture identical variance).

A fuller version of the same idea — a reusable `PCAFromScratch` class (`fit`/`transform`/`inverse_transform`), validated numerically against `sklearn.decomposition.PCA` on the Iris dataset (matching explained-variance ratios and projected coordinates up to sign) — lives in `PCA-Principal-Component-Analysis.ipynb`, §11 ("Implementation from Scratch"). That same notebook (§4, §5) also separately demonstrates eigendecomposition on a synthetic matrix and compares the eigendecomposition route to the SVD route to computing PCA, noting that SVD is what `sklearn` actually uses internally because it's more numerically stable (it avoids explicitly forming $X^TX$, which can amplify numerical error).

## 7. Practical implementation

`sklearn.decomposition.PCA` is the production version of exactly the algorithm in §5, computed via SVD instead of an explicit eigendecomposition of the covariance matrix for numerical stability (see §6) — the two are mathematically equivalent decompositions of the same variance structure. `pca.components_` holds the principal-component directions (equivalent to $W^T$ from §4.4), `pca.explained_variance_` the eigenvalues $\lambda_j$, and `pca.explained_variance_ratio_` the values from §4.5.

**Always standardize before PCA** (`StandardScaler`, zero mean and unit variance per feature) unless every feature is already on a comparable scale. PCA maximizes *variance*, and variance is scale-dependent: a feature measured in the thousands (e.g. income in dollars) will dominate the covariance matrix and hijack the first principal component purely due to units, not because it's actually more informative than a feature measured in single digits (e.g. years of education).

Existing worked examples:
- `PCA-Principal-Component-Analysis.ipynb` — full derivation, eigendecomposition-vs-SVD comparison, explained variance / scree plot, reconstruction and information loss on digit images, PCA vs. feature selection, Kernel PCA for non-linear structure.
- `PCA-2.ipynb` — applied PCA (standardize → `PCA(n_components=2)` → visualize) on the breast cancer dataset, colored by diagnosis.

## 8. Experiment

**Hypothesis:** if the underlying features of a dataset are correlated (as in the breast-cancer or digits datasets, where many measurements move together), a small number of principal components should capture a large majority of the total variance — because correlated features share redundant information that collapses onto shared directions, rather than each feature contributing independent variance.

**Setup:** fit `PCA()` with no component limit on the standardized dataset, examine `explained_variance_ratio_` per component (a **scree plot**: bar chart of variance ratio vs. component index) and its cumulative sum vs. number of components kept.

**Result:** on the breast cancer dataset (`PCA-Principal-Component-Analysis.ipynb`, §12), a large majority of the variance is captured by a handful of components out of the original 30 features — the cumulative-variance curve rises steeply for the first several components and then flattens, and the number of components needed to reach a 95% variance threshold is far smaller than the original feature count. The from-scratch/`sklearn` comparison in §6 confirms this isn't an artifact of one implementation: both the manual eigendecomposition and `sklearn`'s SVD-based PCA agree on the ranked eigenvalues (up to numerical precision), which is what the scree plot is built from.

**Interpretation:** the steep early drop in the scree plot is a direct, visual signature of feature correlation — exactly the redundancy described in §3. A dataset with genuinely independent features would instead show a nearly flat scree plot (every component carrying roughly equal variance), since there would be no redundant directions to collapse.

**Limitations:** the scree plot's "how many components" answer is about variance retained, not about *predictive* or *task* relevance — a low-variance component can occasionally still be the one that separates two classes. And the specific shape of the curve is dataset-dependent; there's no universal "right" cutoff, only a target retained-variance threshold chosen per use case.

## 9. Failure modes

- **PCA is linear.** It can only find and remove *linear* redundancy — directions that are literally straight lines through feature space. Data that lies on a curved (non-linear) manifold — e.g. a spiral or a Swiss-roll shape — has structure PCA cannot capture; the top linear components can badly misrepresent such data's true intrinsic dimensionality. Non-linear alternatives exist for visualization — **t-SNE** (converts distances to neighbor probabilities and matches them in a low-dimensional embedding via KL-divergence minimization; preserves local structure but not distances or global scale, and is $O(N^2)$) and **UMAP** (faster, preserves more global structure, and can be used for actual dimensionality reduction, not just visualization) — but the mechanics of both are out of scope for this note.
- **Principal components are not inherently interpretable.** Each component is a linear combination of *all* original features (with the loadings/weights in `components_` telling you the mixture), so "PC1" rarely has an obvious real-world meaning the way an original feature like "age" does. This is the direct tradeoff against feature selection (which keeps original, interpretable features but cannot decorrelate them).
- **Sensitive to feature scaling** (§7, restated as a failure mode because it's the most common practical mistake) — unscaled PCA silently produces components dominated by whichever feature has the largest numeric range, not the most informative one.
- **Sensitive to outliers.** Because PCA maximizes variance and variance is a squared quantity, a small number of extreme outliers can distort the covariance matrix enough to pull the top principal components toward the outliers rather than the genuine structure of the bulk of the data.
- **Reconstruction from a truncated projection is lossy by construction** — keeping only the top $k$ components discards the variance in the remaining $p-k$ components. This is deliberate (that's the point of dimensionality reduction), but it means PCA is not a lossless compression method, and reconstructed data (via `inverse_transform`) will be a blurred/simplified version of the original.

## 10. Real-world usage

- **Preprocessing before modeling** — reducing correlated, high-dimensional features (e.g. hundreds of sensor readings, or pixel intensities) down to a smaller set of components before feeding them into a downstream model, cutting training time and overfitting risk.
- **Visualization** — projecting high-dimensional data to 2D/3D to inspect cluster structure by eye (frequently paired with the clustering methods in `15-unsupervised-learning` — PCA is used there to visualize K-Means/DBSCAN cluster assignments in 2D even when the original clustering happened in a much higher-dimensional space).
- **Noise reduction** — since noise tends to be spread roughly equally across all directions while real signal concentrates in a few high-variance directions, discarding the low-variance trailing components can filter out some noise.
- **Compression** — image and signal compression pipelines that trade a controlled amount of information loss for a much smaller representation.
- **Exploratory data analysis** — inspecting `components_` (loadings) to understand which original features move together.

## 11. Mental model

**PCA rotates the coordinate system to point along the directions the data actually varies in, so you can keep only the directions that matter.** The rotation doesn't change the data itself — it changes which axes you're allowed to describe it with — and the eigenvalues tell you, precisely, how much you'd lose by dropping each axis.

## 12. Questions to think about

1. The Lagrangian argument in §4.3 shows that the direction of maximum variance must be an eigenvector of $\Sigma$. Why does the *same* argument, applied to the variance remaining after removing the first principal component, force the second principal component to be orthogonal to the first — rather than that orthogonality being an assumption imposed from outside?
2. If two features are perfectly correlated (one is an exact linear function of the other), what happens to the covariance matrix's eigenvalues, and how does that show up in a scree plot?
3. Why is standardizing before PCA sometimes the *wrong* choice — i.e., can you think of a situation where a feature's larger natural variance genuinely reflects its greater importance, and standardizing would erase real signal rather than remove a scaling artifact?
4. PCA's projection minimizes reconstruction error (in a mean-squared sense) among all linear projections to $k$ dimensions, at the same time as it maximizes variance. Why are those the same optimization, rather than two coincidentally similar ones? (Hint: think about the Pythagorean relationship between a point, its projection, and the residual.)
5. A dataset's first two principal components explain 40% of total variance combined. Does that number alone tell you whether a 2D PCA scatter plot is a *trustworthy* visualization of the data's cluster structure? What would you check before trusting it?
