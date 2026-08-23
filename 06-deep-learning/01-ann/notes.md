# 01 – Artificial Neural Networks

## Problem

`05-machine-learning` builds models that assume a specific *shape* of relationship between inputs and outputs: linear regression assumes the target is a linear combination of features, logistic regression assumes the log-odds of a class are linear in the features, and even the kernel trick in `07-svm` only works because someone chose a kernel that implicitly encodes the right non-linear shape in advance. **What do you do when you don't know the shape of the function that maps inputs to outputs at all** — when the true relationship between, say, a house's features and its price, or a tumor's measurements and malignancy, is some unknown non-linear function of the inputs that no fixed-form model (linear, polynomial-of-known-degree, or a hand-picked kernel) is guaranteed to capture?

## Intuition

Imagine trying to separate two classes of points scattered on a 2D plane where the true boundary between them is a circle, not a line. No single straight cut through the plane gets every point right — you'd need to bend the decision boundary. A biological neuron gives a hint at how to do this: it receives many weighted signals from other neurons, sums them, and "fires" (or doesn't) based on whether that sum crosses a threshold. Chain many such units together, in layers, and something interesting happens: each unit draws its own simple boundary, but stacking their outputs through further weighted sums and non-linear "firing" decisions lets the combination bend and combine those simple boundaries into arbitrarily complex shapes — a circle, a spiral, an XOR-like pattern, or a true price-prediction surface.

That's the entire idea of an **Artificial Neural Network (ANN)**: instead of committing to one fixed functional form up front, learn a flexible composition of simple linear-then-nonlinear building blocks, and let gradient-based training discover which composition fits the data.

## Why simpler approaches fail

The **perceptron** (Rosenblatt, 1958) is the simplest artificial neuron and the historical starting point for neural networks. It takes a vector of inputs $\mathbf{x} = [x_1, x_2, \dots, x_n]$, multiplies each by a learned weight, sums them with a bias term, and passes the result through a step (threshold) activation:

$$z = \mathbf{w}^T \mathbf{x} + b, \qquad \hat{y} = \begin{cases} 1 & z \ge 0 \\ 0 & z < 0 \end{cases}$$

Geometrically, this single perceptron carves the input space with one hyperplane $\mathbf{w}^T \mathbf{x} + b = 0$ — exactly the linear decision boundary that `05-machine-learning/06-logistic-regression/notes.md` derives for logistic regression (there, the boundary is $\theta^Tx = 0$, the same hyperplane shape, just a different fitting objective). A single perceptron — and a single logistic regression unit — can only separate **linearly separable** data. There is no learning rate, no regularization strength, no amount of training data that fixes this: it is a structural limitation of the *hypothesis class*, not an optimization failure.

**Concrete, provable failure case: XOR.** Consider four points, $(0,0)\to 0$, $(0,1)\to 1$, $(1,0)\to 1$, $(1,1)\to 0$. No single straight line in the 2D plane can separate the two $y=1$ points from the two $y=0$ points, because they alternate around the square's diagonal — any line that puts $(0,1)$ and $(1,0)$ on one side necessarily puts at least one of $(0,0)$ or $(1,1)$ on the same side too. This isn't a hard-to-optimize case; it is a case where *no weights exist* that solve it with one linear layer. The "From-scratch implementation" section below trains a single linear layer on exactly this dataset and shows it converges to 50% accuracy (chance level) no matter how long it runs — then shows a 2-layer network solves it perfectly.

This limitation motivated stacking many perceptron-like units into layers — the **multi-layer perceptron (MLP)**, which is what "ANN" usually refers to today. The key upgrade from a raw perceptron to a modern artificial neuron is replacing the hard step function with a **smooth, differentiable activation** (sigmoid, ReLU, etc.), because differentiability is what allows gradient-based training (backpropagation, below) to work at all — a step function's derivative is zero almost everywhere, so gradient descent would get no useful signal from it.

## Mathematical foundation

### Architecture and forward propagation

An ANN — also called a Multi-Layer Perceptron (MLP) or a feed-forward network — is organized into layers:

- **Input layer:** one node per feature, no computation performed.
- **Hidden layer(s):** each neuron computes a weighted sum of the previous layer's outputs, adds a bias, and applies a non-linear activation.
- **Output layer:** produces the final prediction — a single linear unit for regression, a sigmoid unit for binary classification, or a softmax layer for multi-class classification.

**Forward propagation** is the process of pushing an input through the network to get a prediction. For layer $l$:

$$\mathbf{z}^{[l]} = \mathbf{W}^{[l]} \mathbf{a}^{[l-1]} + \mathbf{b}^{[l]}, \qquad \mathbf{a}^{[l]} = g^{[l]}(\mathbf{z}^{[l]})$$

where $\mathbf{a}^{[0]} = \mathbf{x}$ is the input, $\mathbf{W}^{[l]}$ and $\mathbf{b}^{[l]}$ are the weight matrix and bias vector of layer $l$, and $g^{[l]}$ is that layer's activation function. Stacking multiple non-linear layers lets an ANN approximate highly non-linear functions — this is formalized by the **Universal Approximation Theorem**, which states that a feed-forward network with a single sufficiently-wide hidden layer and a non-linear activation can approximate any continuous function on a compact domain to arbitrary precision. This is the formal answer to the Problem section's question: an ANN doesn't need to be told the function's shape in advance, because a wide-enough non-linear composition can represent essentially any shape.

### Backpropagation and weight updates

**Backpropagation** is the algorithm used to compute the gradient of the loss with respect to every weight in the network, so that gradient descent can update them. It is simply a systematic application of the **chain rule** of calculus, propagating error signals backward from the output layer to the input layer.

Given a loss $L$ computed at the output, the gradient with respect to a weight in layer $l$ is obtained by chaining derivatives back through every layer that lies between the weight and the loss:

$$\frac{\partial L}{\partial \mathbf{W}^{[l]}} = \frac{\partial L}{\partial \mathbf{a}^{[l]}} \cdot \frac{\partial \mathbf{a}^{[l]}}{\partial \mathbf{z}^{[l]}} \cdot \frac{\partial \mathbf{z}^{[l]}}{\partial \mathbf{W}^{[l]}}$$

In practice this is computed efficiently layer-by-layer: define the error signal at layer $l$ as $\boldsymbol{\delta}^{[l]} = \frac{\partial L}{\partial \mathbf{z}^{[l]}}$. At the output layer $\boldsymbol{\delta}^{[L]}$ comes directly from the loss derivative; for earlier layers it is computed recursively:

$$\boldsymbol{\delta}^{[l]} = \left( (\mathbf{W}^{[l+1]})^T \boldsymbol{\delta}^{[l+1]} \right) \odot g'^{[l]}(\mathbf{z}^{[l]})$$

where $\odot$ is element-wise multiplication. The weight gradients then follow as $\frac{\partial L}{\partial \mathbf{W}^{[l]}} = \boldsymbol{\delta}^{[l]} (\mathbf{a}^{[l-1]})^T$.

Once gradients are known, **gradient descent** updates every parameter in the direction that reduces the loss:

$$\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta \frac{\partial L}{\partial \mathbf{W}^{[l]}}, \qquad \mathbf{b}^{[l]} \leftarrow \mathbf{b}^{[l]} - \eta \frac{\partial L}{\partial \mathbf{b}^{[l]}}$$

where $\eta$ is the **learning rate**. In practice this update is computed on mini-batches of data (mini-batch gradient descent), which balances the stability of full-batch gradients against the speed and regularizing noise of pure stochastic (single-example) updates. Keras/TensorFlow computes all of this automatically via **automatic differentiation** — the user never has to hand-derive $\boldsymbol{\delta}^{[l]}$ (the "From-scratch implementation" section below does derive and code it by hand, exactly once, so the automation isn't a black box).

### Activation functions

Activation functions inject the non-linearity that "Why simpler approaches fail" argued is required; without them, stacking linear layers would collapse into a single linear transform no matter how many layers exist ($\mathbf{W}^{[2]}(\mathbf{W}^{[1]}\mathbf{x}) = (\mathbf{W}^{[2]}\mathbf{W}^{[1]})\mathbf{x}$ is still just one linear map).

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

### Loss and cost functions

These terms are often used interchangeably but have a precise distinction:

- **Loss function** measures the error for a **single training example**: $L(\hat{y}_i, y_i)$.
- **Cost function** is the **average (or sum) of the loss over the whole training set (or a mini-batch)**: $J(\theta) = \frac{1}{m}\sum_{i=1}^{m} L(\hat{y}_i, y_i)$.

Training minimizes the cost function $J(\theta)$ via gradient descent; the loss function is the per-example building block that the cost is computed from.

**Mean Squared Error (MSE)** — the standard regression loss:
$$\text{MSE} = \frac{1}{m}\sum_{i=1}^{m} (y_i - \hat{y}_i)^2$$
Penalizes large errors quadratically, which makes it sensitive to outliers but gives smooth gradients that are easy to optimize.

**Binary Cross-Entropy** — for binary classification with a sigmoid output $\hat{y}_i \in (0,1)$:
$$L = -\frac{1}{m}\sum_{i=1}^{m} \left[ y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i) \right]$$
Heavily penalizes confident-but-wrong predictions (as $\hat{y}_i \to 0$ while $y_i = 1$, the loss $\to \infty$). This is the loss used by the from-scratch XOR network below.

**Categorical Cross-Entropy** — for multi-class classification with a softmax output and one-hot targets $y_{i,k}$:
$$L = -\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^{K} y_{i,k} \log(\hat{y}_{i,k})$$
When targets are given as integer class labels instead of one-hot vectors, Keras's `sparse_categorical_crossentropy` computes the same quantity without requiring explicit one-hot encoding.

### Optimizers

All of the optimizers below build on plain gradient descent, $\theta \leftarrow \theta - \eta \nabla J(\theta)$, by changing *how* the update direction and/or step size is computed.

**SGD (Stochastic Gradient Descent):** updates parameters using the gradient computed from a single example or a small mini-batch rather than the full dataset. This makes each step noisy but cheap, and the noise itself can help escape shallow local minima. Convergence can be slow and oscillatory, especially in ravines of the loss surface.

**Momentum:** accumulates an exponentially-decaying moving average of past gradients and uses that as the update direction: $v_t = \beta v_{t-1} + (1-\beta)\nabla J(\theta)$, $\theta \leftarrow \theta - \eta v_t$. This dampens oscillations across steep dimensions and accelerates progress along consistent, shallow directions — much like a ball rolling downhill and gathering speed.

**AdaGrad:** adapts the learning rate **per parameter**, dividing by the square root of the accumulated sum of past squared gradients. Parameters that receive large or frequent gradients get their effective learning rate shrunk, while rarely-updated parameters keep a relatively larger rate — useful for sparse features. Its main weakness is that the accumulated sum only grows, so the effective learning rate eventually shrinks to near zero and learning stalls.

**RMSProp:** fixes AdaGrad's ever-shrinking learning rate by using an **exponentially-decaying** moving average of squared gradients instead of a raw cumulative sum, so the effective step size does not monotonically vanish. It adapts well to non-stationary objectives and is a strong default for RNNs.

**Adam (Adaptive Moment Estimation):** combines Momentum's idea of averaging the gradient itself (the first moment) with RMSProp's idea of adapting the step size using an average of squared gradients (the second moment), plus a bias-correction term for the early training steps. Adam is the most widely used default optimizer in deep learning today because it converges quickly and needs little learning-rate tuning; both practical notebooks in this topic use `Adam`.

## Algorithm

Training an ANN by (mini-batch) gradient descent proceeds as:

1. Initialize weights randomly (small values; He or Xavier scale — see below) and biases to zero.
2. **Forward pass:** compute $\mathbf{z}^{[l]}, \mathbf{a}^{[l]}$ for every layer $l = 1, \dots, L$, ending in prediction $\hat y = \mathbf{a}^{[L]}$.
3. Compute the loss $L(\hat y, y)$ (or cost $J(\theta)$ over a mini-batch).
4. **Backward pass:** compute $\boldsymbol{\delta}^{[L]}$ from the loss derivative, then recurse $\boldsymbol{\delta}^{[l]} = \left((\mathbf{W}^{[l+1]})^T\boldsymbol{\delta}^{[l+1]}\right)\odot g'^{[l]}(\mathbf{z}^{[l]})$ backward through every layer, computing $\partial L/\partial \mathbf{W}^{[l]}$ and $\partial L/\partial \mathbf{b}^{[l]}$ along the way.
5. **Update:** $\mathbf{W}^{[l]} \leftarrow \mathbf{W}^{[l]} - \eta\,\partial L/\partial \mathbf{W}^{[l]}$ for every layer (optionally via a smarter optimizer than plain gradient descent — Momentum/RMSProp/Adam above).
6. Repeat steps 2–5 for many epochs, over mini-batches of the training set, until the loss converges or a stopping criterion (e.g. `EarlyStopping` on validation loss) is met.

## From-scratch implementation

Implemented in `ann-from-scratch-xor.ipynb`: a 2-layer MLP (2 inputs → 4-unit `tanh` hidden layer → 1 sigmoid output) built with plain NumPy, no autograd. The notebook:

1. First fits a **single linear layer** (`sigmoid(w^Tx + b)`) on the XOR dataset with the exact gradient-descent update rule from "Algorithm," and confirms it converges to 50% accuracy — a direct, executed demonstration of the "Why simpler approaches fail" claim.
2. Defines the 2-layer network's forward pass exactly as in "Mathematical foundation": $z^{[1]}=XW^{[1]}+b^{[1]}$, $a^{[1]}=\tanh(z^{[1]})$, $z^{[2]}=a^{[1]}W^{[2]}+b^{[2]}$, $\hat y = \sigma(z^{[2]})$.
3. Derives and codes the **manual backward pass** — $\delta^{[2]} = \hat y - y$ (the simplified form that BCE + sigmoid output combine to), $\delta^{[1]} = (\delta^{[2]}(W^{[2]})^T)\odot\tanh'(z^{[1]})$, and the resulting weight gradients — as a direct, two-layer instance of the general $\boldsymbol{\delta}^{[l]}$ recursion above.
4. Runs full-batch gradient descent for 10,000 epochs, plots the loss curve, and confirms the trained network reaches **100% accuracy** on XOR with a visibly curved (non-linear) decision boundary — solving exactly the problem the single linear layer provably cannot.

This is the conceptual root that the rest of the deep learning section (CNNs, RNNs, LSTMs, attention) builds on: every one of those architectures is still forward propagation, a loss, backpropagation via the chain rule, and a gradient update — just with more structured layers than a plain dense one.

## Practical implementation

Two companion notebooks put the same theory into practice with Keras `Sequential` models — replacing the from-scratch manual forward/backward/update loop with Keras's `model.fit()`, which performs the identical algorithm (forward pass → loss → backprop via automatic differentiation → optimizer update) internally, just implemented in optimized, GPU-capable C++/XLA rather than hand-written NumPy loops:

- **`ann-regression.ipynb`** — trains an MLP on the `fetch_california_housing` dataset (continuous target: median house value). Uses `StandardScaler`-normalized features, `Dense` hidden layers with ReLU activations, `Adam` optimizer, `MSE` loss, and `EarlyStopping` to prevent overfitting. Reports test **MAE** and **RMSE**, and plots training/validation loss curves.
- **`ann-classification.ipynb`** — trains an MLP on the `load_breast_cancer` dataset (binary target: malignant/benign). Uses `Dropout` layers for regularization, a sigmoid output neuron, binary cross-entropy loss (the same loss the from-scratch notebook implements by hand), and reports test accuracy plus a confusion matrix alongside accuracy/loss curves.

The mapping back to "From-scratch implementation" is direct: `Dense(units, activation=...)` is $\mathbf{z}^{[l]}=\mathbf{W}^{[l]}\mathbf{a}^{[l-1]}+\mathbf{b}^{[l]}$ followed by $g^{[l]}(\cdot)$; `model.compile(loss=..., optimizer=...)` selects the loss and the optimizer update rule; `model.fit()` runs the forward/backward/update loop for the requested number of epochs.

## Experiment

**Hypothesis (stated before running):** if EarlyStopping and Dropout are doing their job, both notebooks' training and validation loss curves should track closely together and both trend downward, rather than validation loss diverging upward while training loss keeps falling (which would signal overfitting).

**Setup:** `ann-regression.ipynb` and `ann-classification.ipynb` each hold out a validation split, train with `Adam`, and record loss (and accuracy, for the classifier) per epoch via Keras's `History` callback, then plot training vs. validation curves.

**Result:** both notebooks' plotted curves show training and validation loss decreasing together and flattening near the end of training rather than diverging — `ann-regression.ipynb` reports final test MAE/RMSE consistent with the validation loss at the epoch `EarlyStopping` selected, and `ann-classification.ipynb` reports test accuracy and a confusion matrix consistent with its flattened validation accuracy curve.

**Interpretation:** the close tracking of train/validation curves is the intended, observable signature of `EarlyStopping` (regression) and `Dropout` (classification) doing their regularizing job — the model is not simply memorizing the training set.

**Limitations:** both experiments use one dataset, one architecture, and one train/validation split each — they demonstrate that the mitigations work in this setting, not that the same hyperparameters generalize to arbitrary datasets or architectures.

## Failure modes

During backpropagation, gradients are products of many layer-wise Jacobians (chain rule). In deep networks:

- **Vanishing gradients:** if each layer's derivative factor has magnitude $< 1$ (common with sigmoid/tanh, whose derivative maxes out at 0.25 and 1 respectively but is $<1$ almost everywhere and saturates toward 0 at the extremes), the product across many layers shrinks toward zero. Early layers then receive almost no gradient signal and stop learning — training stalls even though the loss is not yet low.
- **Exploding gradients:** if the factors have magnitude $> 1$ (common with poorly-scaled weight initialization), the product grows exponentially with depth, causing huge, unstable weight updates (or `NaN` losses).

**Weight initialization** matters for the same underlying reason: starting all weights at the same value (e.g. zero) makes every neuron in a layer compute the same output and receive the same gradient — they never differentiate (the "symmetry problem," which is why the from-scratch notebook above initializes $W^{[1]}, W^{[2]}$ randomly, not at zero). Initialization must also keep the variance of activations (and gradients) roughly stable across layers to avoid vanishing/exploding signals from the very first forward pass.

- **Xavier/Glorot initialization:** draws weights from a distribution with variance $\text{Var}(W) = \frac{2}{n_{in} + n_{out}}$. Designed for activations whose derivative is symmetric around 0 and roughly linear near the origin — i.e. **sigmoid and tanh**.
- **He initialization:** draws weights with variance $\text{Var}(W) = \frac{2}{n_{in}}$, accounting for the fact that **ReLU** zeroes out roughly half of its inputs, so it needs a larger variance to preserve signal magnitude. This is the standard choice for ReLU-based networks (Keras's `he_normal`/`he_uniform` initializers).

Mitigations for vanishing/exploding gradients covered in this course:
- Use **ReLU-family activations** instead of sigmoid/tanh in hidden layers (derivative is 1, not shrinking, for positive inputs).
- Use proper **weight initialization** (Xavier/He, above).
- **Batch Normalization** to keep layer inputs in a well-scaled range (introduced in the CNN topic).
- **Gradient clipping** to cap gradient magnitude and prevent explosion.
- Architectures with explicit "gradient highways" such as **LSTM/GRU** (topic `04-lstm-gru`) and residual connections, which are designed specifically to combat vanishing gradients in deep/recurrent networks.

**Dropout** is a regularization technique — not a fix for vanishing/exploding gradients but for overfitting — that randomly "drops" (zeroes out) a fraction $p$ of neurons' outputs during each training step, forcing the network to not rely too heavily on any single neuron and effectively training an ensemble of thinned sub-networks that share weights. At test/inference time, dropout is turned off and outputs are used at full scale (Keras handles this scaling automatically). It is used in `ann-classification.ipynb`.

## Real-world usage

Plain feed-forward ANNs (MLPs) are the building block underneath every architecture covered later in this section: a CNN (`02-cnn`) ends its pipeline in `Dense` layers; an RNN/LSTM's output head is a `Dense` layer; a Transformer's feed-forward sublayer is exactly this MLP structure. Outside of deep learning proper, MLPs remain a common baseline for structured/tabular regression and classification tasks where feature interactions are too complex for a linear model but the data doesn't have the spatial or sequential structure that would justify a CNN or RNN — for example, tabular risk scoring, sensor-reading regression, or tasks like the housing-price and cancer-diagnosis notebooks above.

## Mental model

An ANN is "linear regression, stacked with a bend at every layer": each layer does the same weighted-sum-plus-bias arithmetic a linear model does, but the non-linear activation between layers is what lets the composition represent shapes — like XOR's — that no single linear layer ever can. Backpropagation is nothing more than the chain rule applied mechanically, layer by layer, to find how much each weight contributed to the final error.

## Questions to think about

1. Why does replacing every activation function in an MLP with the identity function (linear activation) collapse the whole network into a single linear model, regardless of depth?
2. The from-scratch notebook's single linear layer got stuck at exactly 50% accuracy on XOR, not some other number. Why 50%, specifically, for this dataset?
3. If you initialized the from-scratch MLP's $W^{[1]}$ and $W^{[2]}$ to all zeros instead of small random values, what would happen during training, and why does the "symmetry problem" argument predict it?
4. Sigmoid and tanh both saturate at their extremes. Why does ReLU not have a symmetric vanishing-gradient problem for very large positive inputs, and what problem does it have instead?
5. Given the Universal Approximation Theorem's guarantee about a single wide-enough hidden layer, why does practice favor deep (many-layer) networks over very wide, shallow ones?
