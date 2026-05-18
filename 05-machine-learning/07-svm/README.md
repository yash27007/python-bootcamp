# 07 – Support Vector Machines

## 1. SVM Intuition (Hard & Soft Margin)
Support Vector Machines (SVM) is a supervised machine learning algorithm used for both classification and regression. The primary goal of SVM is to find a hyperplane in an N-dimensional space (where N is the number of features) that distinctly classifies the data points.

### Hard Margin
When the dataset is strictly linearly separable, we can use a hard margin SVM. It ensures that all data points are classified correctly without any margin violations. The goal is to maximize the margin ($2/||w||$) between the positive and negative classes, ensuring no points fall inside the margin.

### Soft Margin
In real-world scenarios, datasets are rarely strictly linearly separable. Outliers can ruin the hard margin. Soft Margin SVM introduces a slack variable ($\xi_i$) that allows some misclassifications or margin violations to achieve a better overall fit and robust generalized model. A hyperparameter $C$ controls the trade-off.

## 2. SVM Math (Cost Function)

### Hard Margin Cost Function
For a hard margin setup, the optimization problem is formulated as:
Minimize:
$$ \frac{1}{2} ||w||^2 $$
Subject to:
$$ y_i (w \cdot x_i + b) \geq 1 \quad \forall i $$

### Soft Margin Cost Function
For soft margin, we introduce slack variables $\xi_i \ge 0$:
Minimize:
$$ \frac{1}{2} ||w||^2 + C \sum_{i=1}^n \xi_i $$
Subject to:
$$ y_i (w \cdot x_i + b) \geq 1 - \xi_i \quad \forall i $$

Hinge Loss is typically used: $$J(w, b) = \frac{1}{2} ||w||^2 + C \sum_{i=1}^n \max(0, 1 - y_i(w \cdot x_i + b))$$

## 3. SVM Kernels
When data is not linearly separable, SVM uses the **Kernel Trick** to project it into a higher-dimensional space where it can be linearly separated without explicitly computing the coordinates.
- **Linear Kernel**: $K(x_i, x_j) = x_i \cdot x_j$
- **Polynomial Kernel**: $K(x_i, x_j) = (\gamma (x_i \cdot x_j) + r)^d$
- **RBF (Radial Basis Function) Kernel**: $K(x_i, x_j) = \exp(-\gamma ||x_i - x_j||^2)$
- **Sigmoid Kernel**: $K(x_i, x_j) = \tanh(\gamma (x_i \cdot x_j) + r)$

## 4. SVC Implementation
In Python, Support Vector Classification (SVC) is implemented via `sklearn.svm.SVC`:

```python
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC(C=1.0, kernel='rbf', gamma='scale')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
```

## 5. Support Vector Regression (SVR)
Support Vector Regression tries to fit as many instances as possible *inside* an $\epsilon$-tube around the regression function.

**Cost Function for SVR**:

Minimize:
$$
\frac{1}{2} ||w||^2 + C \sum_{i=1}^n (\xi_i + \xi_i^*)
$$

Subject to: 
$$
\begin{aligned}
y_i - (w \cdot x_i + b) &\le \epsilon + \xi_i \\
(w \cdot x_i + b) - y_i &\le \epsilon + \xi_i^* \\
\xi_i, \xi_i^* &\ge 0
\end{aligned}
$$
