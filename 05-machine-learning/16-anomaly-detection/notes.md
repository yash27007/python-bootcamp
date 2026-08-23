# Anomaly Detection

## 1. Problem

A bank processes millions of credit card transactions a day. Nearly all of them are legitimate. A tiny fraction are fraudulent — and fraud patterns keep changing, so even a list of "known fraud signatures" goes stale. The task: **find the rare, unusual points when you have few or no labeled examples of what "anomalous" looks like.**

This is a distinct problem from classification. In classification, "fraud" and "not fraud" are both classes with (ideally) plenty of labeled training examples of each. In anomaly detection, the "anomaly" class is by definition rare, often not represented at all in training data, and frequently not even well-defined in advance — new kinds of fraud, new sensor failure modes, new attack patterns appear that were never seen when the system was built.

**Anomaly detection** (also called outlier or novelty detection) is the task of identifying data points that are significantly different from the majority, usually by learning what "normal" looks like from mostly- or entirely-unlabeled data and flagging deviations from it.

## 2. Intuition

Three flavors of "unusual," each needing a different kind of detector:

- **Point anomalies** — a single data point is unusual on its own. A $50,000 transaction when everything else is under $500.
- **Contextual anomalies** — a point is unusual only *given its context*, not globally. 30°C is unremarkable in summer, anomalous in winter; buying 100 ski jackets is normal for a ski shop in November, suspicious in July.
- **Collective anomalies** — no single point looks wrong, but a *group* of points together does. A string of $9.99 charges is individually unremarkable, but fifty of them back-to-back on one card is a classic card-testing pattern.

Most of the methods in this note (Isolation Forest, DBSCAN-as-anomaly-detector, LOF) target point anomalies — they ask "is this point far from, or in a much sparser region than, the bulk of the data?" — using different definitions of "far" or "sparse."

## 3. Why simpler approaches fail

**Supervised classification needs labeled anomalies to learn from — and anomalies are, by definition, rare, and often unseen at training time.** Even if a bank has some labeled fraud examples, a classifier trained on them learns to recognize the *specific* fraud patterns present in that data. A genuinely new fraud technique, structurally different from anything in the training set, doesn't look like the labeled positive examples the classifier learned from — and a supervised model has no special mechanism for flagging "this doesn't look like the negative examples either," because that was never the objective it was trained against. Worse, extreme class imbalance (fraud might be 0.1% of transactions) means a supervised classifier can often get near-perfect accuracy by simply predicting "normal" every time, technically minimizing its loss while being useless for the actual task.

What's needed instead is a way to characterize what "normal" looks like directly from the abundant unlabeled (mostly-normal) data, and flag anything that deviates significantly from that characterization — without needing to have seen a labeled example of every possible way to be anomalous.

## 4. Mathematical foundation

### 4.1 Statistical baselines

**Z-score.** For a feature with mean $\mu$ and standard deviation $\sigma$, the Z-score of a value $x$ is:

$$z = \frac{x - \mu}{\sigma}$$

Rule of thumb: flag $|z| > 3$. This assumes the feature is roughly normally distributed, and it is univariate — it only looks at one feature at a time. It's also fragile: the outliers themselves inflate $\mu$ and $\sigma$, which can mask exactly the points it's supposed to catch.

**Modified Z-score** replaces mean/std with the more outlier-robust median and median absolute deviation (MAD):

$$\text{Modified } z = \frac{0.6745(x_i - \text{median})}{\text{MAD}}, \qquad \text{MAD} = \text{median}(|x_i - \text{median}|)$$

Flag $|z| > 3.5$.

**Interquartile range (IQR).** $IQR = Q_3 - Q_1$ (the spread between the 25th and 75th percentiles); flag anything outside $[Q_1 - 1.5\cdot IQR,\ Q_3 + 1.5\cdot IQR]$. Robust to the outliers themselves, since it's built from percentiles rather than mean/std.

**Mahalanobis distance** extends this to multivariate data, accounting for correlation between features:

$$D_M(x) = \sqrt{(x-\mu)^T \Sigma^{-1} (x-\mu)}$$

where $\mu$ is the mean vector and $\Sigma$ the covariance matrix. Plain Euclidean distance is dominated by whichever feature has the largest numeric range and ignores correlation between features (a point can look far away along one axis yet be entirely within the normal joint distribution once correlation is accounted for) — $\Sigma^{-1}$ corrects for both. If the data is multivariate Gaussian, $D_M(x)^2$ follows a $\chi^2$ distribution with $p$ degrees of freedom, giving a principled threshold: flag if $D_M(x)^2 > \chi^2_{p,\,0.975}$.

### 4.2 Isolation Forest

Core observation (Liu, Ting & Zhou, 2008): **anomalies are few and different, so they're easier to isolate with random splits than normal points are.** Build many random trees (iTrees), each recursively splitting the data on a random feature at a random threshold; for each point, record the average path length (number of splits) needed to isolate it alone in a leaf, across all trees. An anomaly — sitting apart from the dense bulk of the data — tends to get separated out after only one or two splits, since most random splits will already put it alone; a normal point buried inside a dense cluster needs many more splits to be pried away from its many close neighbors.

The anomaly score:

$$s(x, n) = 2^{-\frac{E[h(x)]}{c(n)}}$$

where $E[h(x)]$ is the average path length to isolate $x$ across all trees, and $c(n)$ normalizes for tree size:

$$c(n) = 2H(n-1) - \frac{2(n-1)}{n}, \qquad H(i) = \ln(i) + 0.5772\ (\text{Euler–Mascheroni constant})$$

$s \to 1$ for short average path length (anomaly); $s \to 0.5$ for typical path length (normal, no clear signal either way); $s \ll 0.5$ for long path length (dense, clearly normal region).

### 4.3 Local Outlier Factor (LOF)

LOF (Breunig et al., 2000) compares a point's *local* density to its neighbors' local density, rather than isolating it globally. Built from a chain of definitions:

$$k\text{-dist}(A) = \text{distance from } A \text{ to its } k\text{-th nearest neighbor}$$

$$\text{reach-dist}_k(A,B) = \max(k\text{-dist}(B),\ d(A,B))$$

The reachability distance smooths the density estimate by preventing very small point-to-point distances from dominating.

$$\text{lrd}_k(A) = \frac{k}{\sum_{B \in N_k(A)} \text{reach-dist}_k(A,B)}$$

(local reachability density — high lrd means $A$ sits in a dense region).

$$\text{LOF}_k(A) = \frac{\sum_{B \in N_k(A)} \text{lrd}_k(B)}{k \cdot \text{lrd}_k(A)} = \frac{\text{average lrd of } A\text{'s neighbors}}{\text{lrd of } A}$$

$\text{LOF} \approx 1$: $A$'s density matches its neighbors' (normal). $\text{LOF} \gg 1$: $A$ is in a markedly sparser region than its neighbors (anomaly). $\text{LOF} < 1$: $A$ is denser than its neighbors (rare; typically still normal, just in a very tight sub-cluster).

### 4.4 DBSCAN as an anomaly detector

DBSCAN (full derivation in `15-unsupervised-learning`, §4.3) already produces a built-in outlier label as a side effect of clustering: any point that is neither a core point nor within $\varepsilon$ of one is labeled **noise** ($-1$). Formally, $p$ is noise if $|N_\varepsilon(p)| < \text{min\_samples}$ and $p$ is not within $\varepsilon$ of any core point. Reusing that noise label directly as "anomaly" requires no separate scoring function — a point is flagged simply for failing to belong to any sufficiently dense region.

## 5. Algorithm

**Isolation Forest:** build `n_estimators` random binary trees, each on a random subsample (`max_samples`, default 256 — chosen because path lengths saturate around that many points; more data past that point doesn't sharpen the separation between anomaly and normal path lengths, it just costs more to compute) with random feature/threshold splits at each node; average each point's path length across all trees; convert to the score in §4.2; threshold using the expected `contamination` fraction.

**LOF:** for each point, find its $k$ nearest neighbors, compute `reach-dist` to each, derive `lrd`, then compare each point's `lrd` to the average `lrd` of its neighbors, per §4.3.

**DBSCAN-as-detector:** run the DBSCAN clustering algorithm (`15-unsupervised-learning`, §5); every point labeled `-1` is an anomaly.

## 6. From-scratch implementation

A from-scratch `lof_scratch` implementation (k-distance → reach-dist → lrd → LOF, matching §4.3 term-by-term) is in `03-local-outlier-factor/Local-Outlier-Factor.ipynb`, §7 ("Implementation from Scratch"), validated by comparing its ranking of anomaly scores against `sklearn.neighbors.LocalOutlierFactor`'s on the same data.

Isolation Forest and DBSCAN are not reimplemented from scratch here — Isolation Forest's value lies in an ensemble of many random trees over resampled data, which is a straightforward but not conceptually illuminating engineering exercise once the single-tree splitting logic is understood (and decision trees are already implemented from scratch in `10-decision-tree`); DBSCAN's from-scratch implementation lives in `15-unsupervised-learning/03-dbscan-clustering/DBSCAN-Clustering.ipynb` and is reused conceptually here rather than duplicated.

## 7. Practical implementation

- `sklearn.ensemble.IsolationForest` implements §4.2/§5 directly — `contamination` sets the expected anomaly fraction used for thresholding, `predict()` returns $\pm 1$ (anomaly/normal), `score_samples()`/`decision_function()` give the continuous score.
- `sklearn.neighbors.LocalOutlierFactor` implements §4.3 — `novelty=False` (default) is for outlier detection on the fitted data itself; `novelty=True` allows fitting on training data and later calling `predict()` on new points.
- `sklearn.cluster.DBSCAN` reused as in §4.4 — anomalies are simply the points with `label_ == -1`.
- Statistical baselines (§4.1) map directly to `numpy`/`scipy` one-liners: percentile-based IQR fences, and `sklearn.covariance.MinCovDet` for a robust covariance estimate feeding `scipy.spatial.distance.mahalanobis` / `scipy.stats.chi2` for the Mahalanobis threshold.

Two additional production-relevant methods, not derived from scratch above but implemented and mapped to their underlying idea in the notebooks, are worth preserving here since they're genuinely useful alternatives:

- **One-Class SVM** (Schölkopf et al., 2001) learns a boundary in (possibly kernel-transformed) feature space enclosing most of the training data:
$$\min_{w,\rho,\xi} \tfrac{1}{2}\|w\|^2 + \tfrac{1}{\nu n}\sum_i \xi_i - \rho, \qquad f(x) = \text{sign}(w^T\phi(x) - \rho)$$
where $\nu \in (0,1)$ upper-bounds the expected outlier fraction and $\xi_i$ are slack variables. Suited to **novelty detection** — training on *only* normal data, with no anomalies present at all, unlike Isolation Forest and LOF which can be run directly on mixed data. Implemented in `01-isolation-forest/Isolation-Forest-Anomaly-Detection.ipynb`, §7. Slow on large datasets relative to Isolation Forest/LOF.
- **Autoencoder-based detection**: a neural network trained to compress and reconstruct *normal* data; reconstruction error $\|x - \hat{x}\|^2$ is the anomaly score, since a network trained only on normal patterns reconstructs anomalies poorly. Useful for high-dimensional and non-linear structure (e.g. images, sequences with LSTM encoders) where distance-based methods degrade (see §9). Threshold typically set at a high percentile (e.g. 95th) of training reconstruction error. Requires substantially more training data than the other methods here.

## 8. Experiment

**Hypothesis:** on the same dataset, Isolation Forest, DBSCAN, and LOF will not agree perfectly on which points are anomalous, because each defines "anomalous" differently — Isolation Forest by ease of random-split isolation (a global notion), DBSCAN by failing a hard density threshold, and LOF by relative local density compared to neighbors. In particular, LOF should be better than the other two at flagging a point that sits just outside a *locally dense* sub-cluster embedded in otherwise sparser data, since Isolation Forest and DBSCAN's global/fixed-threshold notions of "normal" don't adapt to locally varying density the way LOF's neighbor-relative comparison does.

**Setup and result:** `02-dbscan-anomaly-detection/DBSCAN-Anomaly-Detection.ipynb` already runs a side-by-side comparison of DBSCAN against Isolation Forest across several synthetic shapes (blobs, moons, circles, each with injected uniform-random outliers) — §9, "Multi-algorithm comparison on various data shapes" — visualizing which points each method flags. To close the gap to a full three-way comparison, a short additional cell was added to `03-local-outlier-factor/Local-Outlier-Factor.ipynb` that runs Isolation Forest, DBSCAN, and LOF on the same dataset (a dense Gaussian cluster, a sparse Gaussian cluster, and injected far-away uniform outliers — precisely the varying-density scenario the hypothesis targets) and reports AUC-ROC / AUC-PR for all three, plus how many *legitimate* sparse-cluster-boundary points each method incorrectly flags as anomalies. Result: on the specific comparison the hypothesis targeted — false-flagging legitimate points on the sparse cluster's edge — DBSCAN and LOF both correctly left every boundary point unflagged, while Isolation Forest incorrectly flagged one, confirming that a fixed global contamination threshold can misjudge a locally-sparse-but-legitimate region the way LOF's neighbor-relative comparison does not. However, Isolation Forest still posted the *highest overall* AUC-ROC and AUC-PR on this dataset, because it also has near-perfect recall on the far uniform outliers, which dominate the overall score. **Interpretation:** the three methods agree almost completely on the easy, far-away outliers and disagree specifically at the harder boundary the hypothesis targeted — exactly the kind of disagreement predicted, but it shows up as a difference in *where the mistakes are*, not as one method being strictly better than the others overall.

**Limitations:** this is one synthetic setup with one specific density pattern and a small number of true anomalies, so the "1 false positive vs 0" gap is illustrative rather than statistically robust; the ranking of methods is not universal, and on data without meaningfully varying density (uniform clusters), Isolation Forest and DBSCAN can match or beat LOF while running faster.

## 9. Failure modes

- **Contamination-rate sensitivity.** Isolation Forest, LOF, and One-Class SVM all take an expected outlier fraction (`contamination` or $\nu$) as a hyperparameter used to set the decision threshold. If the true anomaly rate in production drifts from what was assumed at training/threshold-setting time, the flagged fraction becomes systematically wrong — too permissive (missed anomalies) or too aggressive (alert fatigue from false positives) — even though the underlying *scores* may still be ranking points correctly.
- **High-dimensional distance metrics degrade.** LOF, Mahalanobis distance, and (to a lesser extent) DBSCAN all fundamentally depend on distances or local neighborhoods being meaningful. This is the same curse-of-dimensionality failure documented for KNN in `09-knn`'s Failure modes: as dimensionality grows, the ratio between nearest and farthest distances tends toward 1, so "nearest neighbor" and "local density" stop carrying useful information. Isolation Forest is comparatively robust here since it never computes a distance at all — it isolates points via random splits along single features — which is precisely why it's often recommended as the default for high-dimensional tabular data.
- **DBSCAN's noise-as-anomaly approach requires a single global $\varepsilon$**, which (per `15-unsupervised-learning`, Failure modes) struggles when normal data itself has varying density — some genuinely normal points in a sparse-but-legitimate region get mislabeled as noise/anomalies.
- **Statistical baselines (Z-score, IQR) are univariate** — they check one feature at a time and cannot catch a point that is anomalous only in combination across features (an otherwise-normal age combined with an otherwise-normal income might be jointly anomalous — e.g. "5 years old with a $500k salary" — while each value alone looks fine).
- **Collective anomalies are invisible to all point-wise methods above.** None of Isolation Forest, LOF, DBSCAN, or the statistical baselines look at *sequences* or *groups* of points together — a collective anomaly (§2) requires a different formulation (e.g. change-point detection or sequence models) entirely outside this note's scope.

## 10. Real-world usage

| Domain | Application | What's "anomalous" |
|---|---|---|
| Finance | Fraud detection | Unusual spending patterns |
| Cybersecurity | Network intrusion detection | Unusual traffic patterns |
| Manufacturing | Predictive maintenance | Sensor readings preceding failure |
| Healthcare | Patient monitoring | Vital signs deviating from baseline |
| E-commerce | Bot detection | Inhuman browsing speed/patterns |
| Infrastructure | Server monitoring | CPU/memory usage spikes |

**Choosing a method in practice:** Isolation Forest is a reasonable first choice for large, tabular, high-dimensional data (fast, distribution-free, robust to dimensionality per §9); LOF when local density is known to vary meaningfully across the data; One-Class SVM or Isolation Forest when only normal training data is available (true novelty detection); statistical baselines (Z-score/IQR/Mahalanobis) as fast, interpretable first passes on low-dimensional or well-understood data; DBSCAN when the number/shape of anomaly clusters is itself unknown and clustering the data is independently useful.

**Evaluation** is complicated by extreme class imbalance — even a poor model can post a deceptively high accuracy or AUC-ROC by getting the overwhelming majority-normal class right. **AUC-PR (area under the precision-recall curve)** is the preferred metric here specifically because it focuses on the minority (anomaly) class rather than being dominated by the abundant normal class the way AUC-ROC and accuracy both are. When ground truth is entirely unavailable, evaluation falls back to visual inspection of the score distribution, domain-expert review of flagged points, and tracked business outcomes (fraud loss prevented, alert volume).

## 11. Mental model

**Anomaly detection is not classification with a rare class — it's "learn what normal looks like, and measure how far a point is from that, without ever needing a labeled example of every way to be abnormal."** Isolation Forest measures how easy a point is to isolate at random; LOF measures how much sparser a point's neighborhood is than its neighbors' neighborhoods; DBSCAN just calls anything outside a dense region noise. All three encode the same intuition — normal points are hard to tell apart from each other, anomalies aren't — through entirely different mechanics.

## 12. Questions to think about

1. Isolation Forest's anomaly score never computes a pairwise distance, while LOF is built entirely out of distances and neighbor counts. Given the curse-of-dimensionality argument in `09-knn`, why does that structural difference predict which method degrades faster as feature count grows?
2. Why does a fixed `contamination` parameter (used by Isolation Forest, LOF, and One-Class SVM to set a threshold) create a specific, nameable failure mode in production that a purely rank-based evaluation (like AUC-PR) would not reveal during offline testing?
3. LOF's local reachability density compares a point to its *neighbors'* densities, not to the dataset's overall density. Construct a scenario (conceptually) where this local comparison causes LOF to *miss* an anomaly that Isolation Forest would catch easily.
4. DBSCAN's noise label requires no separate "anomaly score" at all — a point is either noise or it isn't. What capability does that binary approach lose compared to a continuous anomaly score (like Isolation Forest's or LOF's), and why might that matter operationally (e.g. when choosing how many alerts a team can review per day)?
5. Why is a supervised classifier trained on 99.9% normal / 0.1% fraud data, evaluated only on accuracy, an almost meaningless model regardless of how sophisticated the classifier is — and why does AUC-PR expose that meaninglessness while AUC-ROC can still look deceptively reasonable?
