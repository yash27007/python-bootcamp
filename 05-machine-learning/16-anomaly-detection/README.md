# 16 – Anomaly Detection

Anomaly detection identifies data points significantly different from the majority — without requiring labeled anomaly examples. The algorithm learns "normal" and flags deviations.

---

## Folder Structure

```
16-anomaly-detection/
├── 01-isolation-forest/
│   └── Isolation-Forest-Anomaly-Detection.ipynb
├── 02-dbscan-anomaly-detection/
│   └── DBSCAN-Anomaly-Detection.ipynb
└── 03-local-outlier-factor/
    └── Local-Outlier-Factor.ipynb
```

---

## Topics

| # | Topic | Notebook | Status |
|---|-------|----------|--------|
| 1 | **Isolation Forest & Statistical Methods** | [Isolation-Forest-Anomaly-Detection.ipynb](01-isolation-forest/Isolation-Forest-Anomaly-Detection.ipynb) | ✅ Complete |
| 2 | **DBSCAN for Anomaly Detection** | [DBSCAN-Anomaly-Detection.ipynb](02-dbscan-anomaly-detection/DBSCAN-Anomaly-Detection.ipynb) | ✅ Complete |
| 3 | **Local Outlier Factor (LOF)** | [Local-Outlier-Factor.ipynb](03-local-outlier-factor/Local-Outlier-Factor.ipynb) | ✅ Complete |

---

## 1 — Isolation Forest & Statistical Methods

### Statistical Baselines

| Method | Formula | Use When |
|---|---|---|
| Z-Score | z = (x − μ)/σ | Univariate, Gaussian data |
| Modified Z-Score | Uses median + MAD | More robust than Z-score |
| IQR | [Q1 − 1.5·IQR, Q3 + 1.5·IQR] | Non-parametric, robust |
| Mahalanobis | sqrt((x−μ)ᵀ Σ⁻¹ (x−μ)) | Multivariate, correlated features |

### Isolation Forest (Liu et al., 2008)

**Core idea**: Anomalies are easier to isolate — they require fewer random splits in a random binary tree.

**Anomaly Score**: s(x,n) = 2^(−E[h(x)]/c(n))

- s → 1: very short path → **anomaly**
- s → 0.5: average path → normal

**Why max_samples=256**: Path lengths saturate after 256 points. More data doesn't help; default of 256 is optimal.

**Strengths**: O(n log n), no distance computation, no distributional assumption, works well in high dimensions.

---

## 2 — DBSCAN for Anomaly Detection

DBSCAN noise points (label **-1**) are anomalies — points in low-density regions with no cluster membership.

**Point types**:
- Core point (≥ min_samples neighbors in ε-ball) → normal
- Border point (near a core point) → normal
- **Noise point** → **anomaly**

**Strengths**: Arbitrary cluster shapes, no contamination rate needed, deterministic.

**Weaknesses**: No continuous score, sensitive to ε, fails with varying-density normal data, can't predict new points.

**ε selection**: k-distance plot elbow (k = min_samples − 1).

---

## 3 — Local Outlier Factor (LOF)

LOF detects points in a region of **much lower density than their neighbors**.

Formula chain:
```
k-dist(A) → reach-dist(A,B) = max(k-dist(B), d(A,B)) → lrd(A) = k / Σ reach-dist → LOF(A) = avg_lrd(neighbors) / lrd(A)
```

- LOF ≈ 1 → normal (similar density to neighbors)
- LOF >> 1 → **anomaly** (much sparser than neighbors)

**Key advantage over Isolation Forest**: Handles datasets where normal data has **varying density** — a loose cluster is still normal relative to itself.

**Novelty mode**: Set `novelty=True` to fit on training data and predict on new samples.

---

## Algorithm Reference

| Method | High-dim? | Varying density? | Predict new? | Speed |
|---|---|---|---|---|
| **Isolation Forest** | **Yes** | No | **Yes** | O(n log n) |
| **LOF** | No | **Yes** | Novelty mode | O(n²) |
| **DBSCAN** | No | No | No | O(n log n) |
| Mahalanobis | Up to ~50D | No | Yes | O(n) |
| Z-Score / IQR | No (univariate) | — | Yes | O(n) |

## Evaluation

Use **AUC-PR** (Average Precision) as the primary metric — class imbalance makes it far more informative than accuracy or AUC-ROC.

```python
from sklearn.metrics import roc_auc_score, average_precision_score
auc_pr = average_precision_score(y_true, anomaly_scores)  # preferred
auc_roc = roc_auc_score(y_true, anomaly_scores)
```
