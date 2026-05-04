# Descriptive Statistics

Descriptive statistics **summarise and describe** the main features of a dataset. Unlike inferential statistics (which draw conclusions beyond the data), descriptive statistics only describe *what is in front of you*.

---

## 1. Types of Data

| Type | Description | Examples |
|------|-------------|---------|
| **Nominal** | Categories with no order | Colour, Gender, Country |
| **Ordinal** | Categories with a meaningful order | Star rating, Education level |
| **Interval** | Numeric, equal gaps, no true zero | Temperature (°C), Year |
| **Ratio** | Numeric, equal gaps, true zero exists | Height, Weight, Income |

---

## 2. Measures of Central Tendency

### Mean (Arithmetic Average)
$$\bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i$$

- Sensitive to outliers.
- Best for symmetric, continuous data.

### Median
- The **middle value** when data is sorted.
- For even *n*: average of the two middle values.
- Robust to outliers — preferred for skewed distributions (e.g. income, house prices).

### Mode
- The **most frequent** value.
- Can be used with any data type (even nominal).
- A distribution can be unimodal, bimodal, or multimodal.

---

## 3. Measures of Dispersion (Spread)

### Range
$$\text{Range} = x_{\max} - x_{\min}$$
Simple but highly sensitive to outliers.

### Variance
$$s^2 = \frac{1}{n-1} \sum_{i=1}^{n}(x_i - \bar{x})^2$$
Average of squared deviations. Denominator is *n − 1* (Bessel's correction) for sample variance.

### Standard Deviation
$$s = \sqrt{s^2}$$
Same units as the data — the most interpretable spread measure.

### Interquartile Range (IQR)
$$\text{IQR} = Q_3 - Q_1$$
- **Q1** = 25th percentile, **Q3** = 75th percentile.
- Robust to outliers.
- Used in box plots and outlier detection (Tukey's fence: $Q_1 - 1.5 \cdot \text{IQR}$, $Q_3 + 1.5 \cdot \text{IQR}$).

---

## 4. Shape of Distribution

### Skewness
Measures asymmetry of the distribution around its mean.

| Skewness | Shape | Relationship |
|----------|-------|--------------|
| = 0 | Symmetric | Mean = Median = Mode |
| > 0 | Right-skewed (positive) | Mean > Median > Mode |
| < 0 | Left-skewed (negative) | Mean < Median < Mode |

$$\text{Skewness} = \frac{\frac{1}{n}\sum(x_i - \bar{x})^3}{s^3}$$

### Kurtosis
Measures the "tailedness" of the distribution.

| Kurtosis | Description |
|----------|-------------|
| = 3 (excess = 0) | **Mesokurtic** – normal distribution |
| > 3 (excess > 0) | **Leptokurtic** – heavy tails, sharp peak |
| < 3 (excess < 0) | **Platykurtic** – light tails, flat peak |

---

## 5. Five-Number Summary & Box Plot

$$\min,\ Q_1,\ \text{Median},\ Q_3,\ \max$$

A **box plot** (box-and-whisker plot) visualises this summary:
- Box spans Q1 → Q3 (the IQR)
- Whiskers extend to $Q_1 - 1.5\cdot\text{IQR}$ and $Q_3 + 1.5\cdot\text{IQR}$
- Points beyond whiskers are **outliers**

---

## 6. Covariance & Correlation

### Covariance
$$\text{Cov}(X, Y) = \frac{1}{n-1}\sum(x_i - \bar{x})(y_i - \bar{y})$$
Direction of the linear relationship between two variables.

### Pearson Correlation Coefficient
$$r = \frac{\text{Cov}(X,Y)}{s_X \cdot s_Y}, \quad r \in [-1, 1]$$

| r value | Interpretation |
|---------|----------------|
| +1 | Perfect positive linear |
| 0 | No linear relationship |
| −1 | Perfect negative linear |

> Correlation ≠ Causation

---

## 7. Quick Reference Cheat Sheet

| Measure | Robust to Outliers? | Data Type |
|---------|--------------------|-----------| 
| Mean | No | Interval / Ratio |
| Median | Yes | Ordinal and above |
| Mode | N/A | Any |
| Std Dev | No | Interval / Ratio |
| IQR | Yes | Ordinal and above |
| Pearson r | No | Interval / Ratio |
