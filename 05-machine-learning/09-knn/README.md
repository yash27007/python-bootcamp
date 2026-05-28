# 09 – K-Nearest Neighbours

## Overview
K-Nearest Neighbors (KNN) is a simple, non-parametric, and lazy learning algorithm used for both classification and regression. The fundamental idea is that similar data points exist in close proximity.

## How it Works
1. Choose the number of neighbors, $K$.
2. Calculate the distance between the query data point and all the points in the training dataset.
3. Sort the computed distances in ascending order and select the $K$ points with the smallest distances.
4. **For Classification:** Return the mode (most frequent class) among the $K$ neighbors.
5. **For Regression:** Return the mean or median of the continuous target values of the $K$ neighbors.

## Distance Metrics
The algorithm relies heavily on how "distance" is defined. Common distance formulas include:

**1. Euclidean Distance** (Most common for continuous variables, $p=2$):
$$d(p, q) = \sqrt{\sum_{i=1}^n (q_i - p_i)^2}$$

**2. Manhattan Distance** (City-block distance, $p=1$):
$$d(p, q) = \sum_{i=1}^n |q_i - p_i|$$

**3. Minkowski Distance** (Generalized formula):
$$d(p, q) = \left( \sum_{i=1}^n |q_i - p_i|^p \right)^{\frac{1}{p}}$$

## Cost Function & Training Phase
KNN is known as a **"lazy learner"**. Unlike Linear Regression or SVMs, KNN does not have a training phase where an explicit cost function is minimized to learn weights or parameters. 

Instead, the algorithm simply stores the entire training dataset. The "cost" of computation is deferred to the prediction phase, where it must calculate the distance between the new point and all stored points (complexity of $O(N \times D)$). To optimize this query time, spatial data structures like **KD-Trees** or **Ball Trees** are often used.
