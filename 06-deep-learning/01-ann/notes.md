# 01 – Artificial Neural Networks

| Topic | Status |
|-------|--------|
| Perceptron Intuition | ✅ Complete |
| ANN Architecture & Forward Propagation | ✅ Complete |
| Backpropagation & Weight Updates | ✅ Complete |
| Activation Functions (Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Softmax) | ✅ Complete |
| Loss vs Cost Function | ✅ Complete |
| Regression & Classification Loss Functions | ✅ Complete |
| Optimisers (SGD, Momentum, AdaGrad, RMSProp, Adam) | ✅ Complete |
| Vanishing/Exploding Gradient Problems | ✅ Complete |
| Weight Initialisation & Dropout | ✅ Complete |
| ANN Regression & Classification Projects | ✅ Complete |

## Perceptron Intuition

The **perceptron** (Rosenblatt, 1958) is the simplest artificial neuron and the historical starting point for neural networks. It takes a vector of inputs $\mathbf{x} = [x_1, x_2, \dots, x_n]$, multiplies each by a learned weight, sums them with a bias term, and passes the result through a step (threshold) activation:

$$z = \mathbf{w}^T \mathbf{x} + b, \qquad \hat{y} = \begin{cases} 1 & z \ge 0 \\ 0 & z < 0 \end{cases}$$

Geometrically, the perceptron carves the input space with a hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$ — exactly the linear-classifier boundary introduced in `05-machine-learning`. A single perceptron can only separate **linearly separable** data (it cannot learn XOR). This limitation motivated stacking many perceptron-like units into layers — the **multi-layer perceptron (MLP)**, which is what "ANN" usually refers to today.

The key upgrade from a raw perceptron to a modern artificial neuron is replacing the hard step function with a **smooth, differentiable activation** (sigmoid, ReLU, etc.), because differentiability is what allows gradient-based training (backpropagation) to work.

## ANN Architecture & Forward Propagation

An **Artificial Neural Network (ANN)** — also called a Multi-Layer Perceptron (MLP) or a feed-forward network — is organized into layers:

- **Input layer:** one node per feature, no computation performed.
- **Hidden layer(s):** each neuron computes a weighted sum of the previous layer's outputs, adds a bias, and applies a non-linear activation.
- **Output layer:** produces the final prediction — a single linear unit for regression, a sigmoid unit for binary classification, or a softmax layer for multi-class classification.

**Forward propagation** is the process of pushing an input through the network to get a prediction. For layer $l$:

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}, \qquad \mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})$$

where $\mathbf{a}^{[0]} = \mathbf{x}$ is the input, $\mathbf{W}^{[l]}$ and $\mathbf{b}^{[l]}$ are the weight matrix and bias vector of layer $l$, and $g^{[l]}$ is that layer's activation function. Stacking multiple non-linear layers lets an ANN approximate highly non-linear functions — this is formalized by the **Universal Approximation Theorem**, which states that a feed-forward network with a single sufficiently-wide hidden layer and a non-linear activation can approximate any continuous function on a compact domain to arbitrary precision.

## Backpropagation & Weight Updates

**Backpropagation** is the algorithm used to compute the gradient of the loss with respect to every weight in the network, so that gradient descent can update them. It is simply a systematic application of the **chain rule** of calculus, propagating error signals backward from the output layer to the input layer.

Given a loss $L$ computed at the output, the gradient with respect to a weight in layer $l$ is obtained by chaining derivatives back through every layer that lies between the weight and the loss:

$$\frac{\partial L}{\partial \mathbf{W}^{[l]}} = \frac{\partial L}{\partial \mathbf{a}^{[l]}} \cdot \frac{\partial \mathbf{a}^{[l]}}{\partial \mathbf{z}^{[l]}} \cdot \frac{\partial \mathbf{z}^{[l]}}{\partial \mathbf{W}^{[l]}}$$

In practice this is computed efficiently layer-by-layer: define the error signal at layer $l$ as $\boldsymbol{\delta}^{[l]} = \frac{\partial L}{\partial \mathbf{z}^{[l]}}$. At the output layer $\boldsymbol{\delta}^{[L]}$ comes directly from the loss derivative; for earlier layers it is computed recursively:

$$\boldsymbol{\delta}^{[l]} = \left( (\mathbf{W}^{[l+1]})^T \boldsymbol{\delta}^{[l+1]} \right) \odot g'^{[l]}(\mathbf{z}^{[l]})$$

where $\odot$ is element-wise multiplication. The weight gradients then follow as $\frac{\partial L}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^T$.

Once gradients are known, **gradient descent** updates every parameter in the direction that reduces the loss:

$$\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta \frac{\partial L}{\partial \mathbf{W}^{[l]}}, \qquad \mathbf{b}^{[l]} \leftarrow \mathbf{b}^{[l]} - \eta \frac{\partial L}{\partial \mathbf{b}^{[l]}}$$

where $\eta$ is the **learning rate**. In practice this update is computed on mini-batches of data (mini-batch gradient descent), which balances the stability of full-batch gradients against the speed and regularizing noise of pure stochastic (single-example) updates. Keras/TensorFlow computes all of this automatically via **automatic differentiation** — the user never has to hand-derive $\boldsymbol{\delta}^{[l]}$.

## Activation Functions

Activation functions inject non-linearity into the network; without them, stacking linear layers would collapse into a single linear transform no matter how many layers exist.

**Sigmoid:**
$$\sigma(z) = \frac{1}{1 + e^{-z}}, \qquad \sigma'(z) = \sigma(z)(1-\sigma(z))$$
Squashes outputs to $(0, 1)$. Used at the **output layer for binary classification** (interpretable as a probability). Rarely used in hidden layers of deep networks because it saturates for large $|z|$, causing vanishing gradients.

**Tanh:**
$$\tanh(z) = \frac{e^z - e^{-z}}{e^z + e^{-z}}$$
Squashes to $(-1, 1)$, zero-centered (which helps optimization compared to sigmoid), but still saturates at the extremes and suffers vanishing gradients in deep networks.

**ReLU (Rectified Linear Unit):**
$$\text{ReLU}(z) = \max(0, z)$$
The default choice for hidden layers in modern deep networks. Cheap to compute, does not saturate for $z > 0$, and empirically trains faster. Downside: **"dying ReLU"** — a neuron whose input is always negative outputs 0 and its gradient is 0, so it stops learning.

**Leaky ReLU:**
$$\text{LeakyReLU}(z) = \begin{cases} z & z \ge 0 \\ \alpha z & z < 0 \end{cases} \quad (\alpha \approx 0.01)$$
Fixes dying ReLU by allowing a small non-zero gradient when $z < 0$.

**ELU (Exponential Linear Unit):**
$$\text{ELU}(z) = \begin{cases} z & z \ge 0 \\ \alpha(e^z - 1) & z < 0 \end{cases}$$
Smooth for negative inputs (unlike Leaky ReLU's sharp corner), pushes mean activations closer to zero which can speed up learning, at the cost of being more expensive to compute (involves an exponential).

**Softmax** (used at the **output layer for multi-class classification**):
$$\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}$$
Converts a vector of raw scores ("logits") into a probability distribution over $K$ classes that sums to 1.

**Rule of thumb:** ReLU (or a variant) in hidden layers; sigmoid for binary output; softmax for multi-class output; linear (no activation) for regression output.

## Loss vs Cost Function

These terms are often used interchangeably but have a precise distinction:

- **Loss function** measures the error for a **single training example**: $L(\hat{y}_i, y_i)$.
- **Cost function** is the **average (or sum) of the loss over the whole training set (or a mini-batch)**: $J(\theta) = \frac{1}{m}\sum_{i=1}^{m} L(\hat{y}_i, y_i)$.

Training minimizes the cost function $J(\theta)$ via gradient descent; the loss function is the per-example building block that the cost is computed from.

## Regression & Classification Loss Functions

**Mean Squared Error (MSE)** — the standard regression loss:
$$\text{MSE} = \frac{1}{m}\sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$
Penalizes large errors quadratically, which makes it sensitive to outliers but gives smooth gradients that are easy to optimize.

**Binary Cross-Entropy** — for binary classification with a sigmoid output $\hat{y}_i \in (0,1)$:
$$L = -\frac{1}{m}\sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]$$
Heavily penalizes confident-but-wrong predictions (as $\hat{y}_i \to 0$ while $y_i = 1$, the loss $\to \infty$).

**Categorical Cross-Entropy** — for multi-class classification with a softmax output and one-hot targets $y_{i,k}$:
$$L = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})$$
When targets are given as integer class labels instead of one-hot vectors, Keras's `sparse_categorical_crossentropy` computes the same quantity without requiring explicit one-hot encoding.

## Optimisers

All of the optimizers below build on plain gradient descent, $\theta \leftarrow \theta - \eta \nabla J(\theta)$, by changing *how* the update direction and/or step size is computed.

**SGD (Stochastic Gradient Descent):** updates parameters using the gradient computed from a single example or a small mini-batch rather than the full dataset. This makes each step noisy but cheap, and the noise itself can help escape shallow local minima. Convergence can be slow and oscillatory, especially in ravines of the loss surface.

**Momentum:** accumulates an exponentially-decaying moving average of past gradients and uses that as the update direction: $v_t = \beta v_{t-1} + (1-\beta)\nabla J(\theta)$, $\theta \leftarrow \theta - \eta v_t$. This dampens oscillations across steep dimensions and accelerates progress along consistent, shallow directions — much like a ball rolling downhill and gathering speed.

**AdaGrad:** adapts the learning rate **per parameter**, dividing by the square root of the accumulated sum of past squared gradients. Parameters that receive large or frequent gradients get their effective learning rate shrunk, while rarely-updated parameters keep a relatively larger rate — useful for sparse features. Its main weakness is that the accumulated sum only grows, so the effective learning rate eventually shrinks to near zero and learning stalls.

**RMSProp:** fixes AdaGrad's ever-shrinking learning rate by using an **exponentially-decaying** moving average of squared gradients instead of a raw cumulative sum, so the effective step size does not monotonically vanish. It adapts well to non-stationary objectives and is a strong default for RNNs.

**Adam (Adaptive Moment Estimation):** combines Momentum's idea of averaging the gradient itself (the first moment) with RMSProp's idea of adapting the step size using an average of squared gradients (the second moment), plus a bias-correction term for the early training steps. Adam is the most widely used default optimizer in deep learning today because it converges quickly and needs little learning-rate tuning; both notebooks in this topic use `Adam`.

## Vanishing/Exploding Gradient Problems

During backpropagation, gradients are products of many layer-wise Jacobians (chain rule). In deep networks:

- **Vanishing gradients:** if each layer's derivative factor has magnitude $< 1$ (common with sigmoid/tanh, whose derivative maxes out at 0.25 and 0.25 respectively), the product across many layers shrinks toward zero. Early layers then receive almost no gradient signal and stop learning — training stalls even though the loss is not yet low.
- **Exploding gradients:** if the factors have magnitude $> 1$ (common with poorly-scaled weight initialization), the product grows exponentially with depth, causing huge, unstable weight updates (or `NaN` losses).

Mitigations covered in this course:
- Use **ReLU-family activations** instead of sigmoid/tanh in hidden layers (derivative is 1, not shrinking, for positive inputs).
- Use proper **weight initialization** (Xavier/He, below).
- **Batch Normalization** to keep layer inputs in a well-scaled range (introduced in the CNN topic).
- **Gradient clipping** to cap gradient magnitude and prevent explosion.
- Architectures with explicit "gradient highways" such as **LSTM/GRU** (topic 04) and residual connections, which are designed specifically to combat vanishing gradients in deep/recurrent networks.

## Weight Initialisation & Dropout

**Weight initialization** matters because starting all weights at the same value (e.g. zero) makes every neuron in a layer compute the same output and receive the same gradient — they never differentiate ("symmetry problem"). Initialization must also keep the variance of activations (and gradients) roughly stable across layers to avoid vanishing/exploding signals from the very first forward pass.

- **Xavier/Glorot initialization:** draws weights from a distribution with variance $\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$. Designed for activations whose derivative is symmetric around 0 and roughly linear near the origin — i.e. **sigmoid and tanh**.
- **He initialization:** draws weights with variance $\text{Var}(W) = \frac{2}{n_{in}}$, accounting for the fact that **ReLU** zeroes out roughly half of its inputs, so it needs a larger variance to preserve signal magnitude. This is the standard choice for ReLU-based networks (Keras's `he_normal`/`he_uniform` initializers).

**Dropout** is a regularization technique that randomly "drops" (zeroes out) a fraction $p$ of neurons' outputs during each training step, forcing the network to not rely too heavily on any single neuron and effectively training an ensemble of thinned sub-networks that share weights. At test/inference time, dropout is turned off and outputs are used at full scale (Keras handles this scaling automatically). It is a very effective way to reduce overfitting in fully-connected layers, and is used in `ann-classification.ipynb` below.

## ANN Regression & Classification Projects

Two companion notebooks put the above theory into practice with Keras `Sequential` models:

- **`ann-regression.ipynb`** — trains an MLP on the `fetch_california_housing` dataset (continuous target: median house value). Uses `StandardScaler`-normalized features, `Dense` hidden layers with ReLU activations, `Adam` optimizer, `MSE` loss, and `EarlyStopping` to prevent overfitting. Reports test **MAE** and **RMSE**, and plots training/validation loss curves.
- **`ann-classification.ipynb`** — trains an MLP on the `load_breast_cancer` dataset (binary target: malignant/benign). Uses `Dropout` layers for regularization, a sigmoid output neuron, binary cross-entropy loss, and reports test accuracy plus a confusion matrix alongside accuracy/loss curves.
