# 02 – Convolutional Neural Networks

## Problem

`01-ann/notes.md` establishes that a plain feed-forward network (MLP) can approximate essentially any function once you feed it enough hidden units — including, in principle, image classification, by flattening an image into a long vector of pixel values and feeding that vector to a `Dense` layer. **Why does this straightforward approach fail badly in practice, and what property of images does a plain ANN ignore that a purpose-built architecture should exploit?**

## Intuition

Flattening a $100\times100$ pixel image into a 10,000-length vector throws away something every human vision system relies on constantly: **spatial locality**. A cat's ear is a small, local pattern — a particular arrangement of edges and shading confined to a small patch of the image — and that pattern looks essentially the same whether it appears in the top-left or bottom-right corner of the photo. A flattened vector destroys this: pixel 47 and pixel 48 (horizontally adjacent) end up no more "related" in the vector's structure than pixel 47 and pixel 9,000 (on the opposite side of the image), and a `Dense` layer must independently learn a *separate* set of weights to recognize "cat ear" in every possible position it could occur, because it has no built-in notion that nearby pixels are correlated or that a pattern found in one location might reappear in another.

The visual cortex processes images very differently: early neurons respond to simple, local stimuli (edges, orientations, color contrast) within a small **receptive field** — a limited patch of the visual field, not the whole image — and progressively deeper neurons combine those simple signals into detectors for shapes, textures, and eventually whole objects and faces. A **Convolutional Neural Network (CNN)** is directly inspired by this: instead of a `Dense` layer connecting every output to every input pixel, a CNN layer slides a small, shared filter across the image, so each output value only depends on a local neighborhood of the input — mimicking a local receptive field, and reusing the same filter weights at every position instead of learning separate weights per location.

## Why simpler approaches fail

**Concrete parameter-count blowup.** Consider a modest $100\times100\times3$ RGB image (30,000 input values) feeding a single `Dense` hidden layer of just 256 units. Every hidden unit connects to every input value, so this one layer alone has

$$30{,}000 \times 256 = 7{,}680{,}000 \text{ weights (plus 256 biases)}$$

for a single hidden layer, on a genuinely small image — a real photo is easily $10\times$ larger per side, which is $100\times$ more pixels, pushing this into the hundreds of millions of parameters for one layer. Compare that to a convolutional layer using 256 filters of size $3\times3\times3$ (a common configuration): each filter has $3\times3\times3=27$ weights (+1 bias), so

$$256 \times (27 + 1) = 7{,}168 \text{ weights total} - \text{regardless of the image's height or width.}$$

That is roughly **1,000× fewer parameters** for a comparable number of output feature maps, because every filter's weights are **shared** across every spatial position instead of learned independently for each one. Beyond the sheer compute/memory cost, that many independent parameters relative to typical dataset sizes makes a dense-on-flattened-pixels model prone to severe overfitting — it has enough capacity to memorize training images rather than learn generalizable patterns, and it must independently re-learn the same pattern (e.g. "vertical edge") at every pixel location as if they were unrelated concepts, since flattening discards any notion that nearby pixels are related.

## Mathematical foundation

### Images as tensors

A grayscale image is a 2D array of pixel intensities; a color image is a 3D tensor of shape (height, width, channels), typically 3 channels for RGB. In Keras, image batches are 4D tensors of shape `(batch, height, width, channels)`.

### The convolution operation

A **kernel** (or **filter**) is a small matrix of learnable weights (e.g. $3\times3$ or $5\times5$). Convolution slides this kernel across the input, and at each position computes the element-wise product between the kernel and the underlying image patch, then sums the result into a single output value:

$$(I * K)(i, j) = \sum_{m}\sum_{n} I(i+m, j+n) \, K(m, n)$$

A single kernel produces one 2D **feature map** that highlights wherever its learned pattern (e.g. a vertical edge) appears in the image. A convolutional layer typically learns many kernels in parallel, each producing its own feature map, stacked together as the layer's output channels.

Two hyperparameters control how the kernel moves and what happens at the image boundary:

- **Stride:** the step size the kernel moves each time (stride 1 = move one pixel at a time; stride 2 = skip every other position, halving the output resolution).
- **Padding:** whether to pad the input's border with zeros before convolving.
    - `"valid"` padding: no padding — the output shrinks slightly relative to the input (output size $= \lfloor (n - k)/s \rfloor + 1$ for input size $n$, kernel size $k$, stride $s$).
    - `"same"` padding: pad just enough so the output has the same height/width as the input (with stride 1).

### Parameter sharing and translation invariance

The "Why simpler approaches fail" parameter count above is a direct consequence of two properties baked into the convolution operation:

- **Parameter sharing:** a kernel's weights are reused at every spatial position, so a $3\times3\times3$ kernel has only 27 weights (+bias) regardless of image size, versus a `Dense` layer needing one independent weight per pixel per output neuron. This is *why* the parameter count above shrinks by orders of magnitude, and it drastically reduces overfitting risk since far fewer parameters must be estimated from the same amount of training data.
- **Translation invariance:** because the same kernel scans the whole image, a pattern (e.g. an edge or a cat's ear) is detected wherever it appears, without needing to be retrained per position — directly solving the Intuition section's complaint about a `Dense` layer needing separate weights per location. Pooling (below) further strengthens this by tolerating small shifts.
- **Local connectivity:** each output value depends only on a small local patch of the input, matching the local/spatial structure of images (nearby pixels are correlated; distant pixels usually aren't directly related) — a `Dense` layer ignores this structure entirely and treats every pixel as independent.
- **Hierarchical feature learning:** stacking conv→pool blocks lets the network build increasingly abstract representations (edges → textures → parts → objects) without hand-engineering features, mirroring the brain's hierarchical visual processing described in "Intuition," unlike classical computer vision pipelines that hand-engineer each stage.

### Pooling

Pooling layers **downsample** feature maps, reducing spatial resolution (and hence computation) while retaining the most important information and adding a further degree of local translation invariance.

- **Max pooling:** for each window (e.g. $2\times2$), output the maximum value. This preserves the strongest activation of a detected feature within that window, which is usually what matters for recognition.
- **Average pooling:** output the average value within each window — smoother, but can dilute strong signals.

`MaxPooling2D` is the standard choice in CNN classifiers and is used in this topic's notebook.

### Flattening and fully connected layers

After several conv→pool blocks, the network holds a stack of small, deep feature maps that summarize *what* patterns are present, but the classifier needs a single probability distribution over classes. The **Flatten** layer reshapes the final 3D feature-map tensor `(height, width, channels)` into a 1D vector, which is then fed into one or more **fully connected (`Dense`)** layers — exactly the ANN layers from `01-ann`. The final `Dense` layer uses a softmax activation (multi-class) or sigmoid (binary) to produce class probabilities.

Typical CNN architecture:

$$\text{Input} \to [\text{Conv2D} \to \text{ReLU} \to \text{MaxPooling2D}]\times N \to \text{Flatten} \to \text{Dense} \to \text{Dense(softmax)}$$

Convolutional layers act as an automatic, learned feature extractor; the dense layers at the end act as the classifier operating on those extracted features — the flattening step is precisely where the "spatial-structure-aware" part of the network hands off to the plain-MLP part covered in `01-ann`.

## Algorithm

Forward pass of a convolutional layer, for one output feature map from one kernel:

1. For each valid position $(i, j)$ (per the stride/padding rule), extract the image patch of the kernel's size centered/anchored at that position.
2. Compute the element-wise product of the patch and the kernel, and sum it into a single scalar: $(I*K)(i,j) = \sum_m\sum_n I(i+m,j+n)K(m,n)$.
3. Repeat for every valid $(i,j)$ to fill in the output feature map.
4. Apply the layer's activation (typically ReLU) element-wise to the feature map.
5. Repeat steps 1–4 for every kernel in the layer, stacking the resulting feature maps as output channels.
6. Optionally apply pooling (step size and window per hyperparameters) to downsample.
7. Stack conv→pool blocks, then flatten and feed into `Dense` layers per the architecture above.

Training uses the same backpropagation and gradient descent machinery from `01-ann/notes.md`: gradients flow backward through the `Dense` layers, then back through pooling and convolutional layers (a kernel's gradient is accumulated across every position it was applied to, since parameter sharing means the same weight influenced many output values).

## From-scratch implementation

Implemented in `cnn-from-scratch-convolution.ipynb`: the convolution formula above, coded directly as two nested Python loops over output positions, applied to an $8\times8$ toy grayscale image containing a single vertical edge (dark left half, bright right half), using a hand-picked $3\times3$ vertical Sobel-style edge-detection kernel (no learning — the kernel here is fixed to make the mechanics visible). The notebook:

1. Computes the feature map manually, confirming it spikes at the column where the edge sits and stays near zero on flat regions.
2. Cross-checks the result against `scipy.signal.convolve2d` (noting that true mathematical convolution flips the kernel, so the kernel is passed pre-flipped to match) and confirms the outputs are numerically identical (`np.allclose`).
3. Cross-checks the result against `tf.nn.conv2d` (which, like all deep learning frameworks, performs cross-correlation — no flip — matching the manual loop's convention directly) and confirms the outputs match.
4. Visualizes all three feature maps side-by-side to show they agree.

This confirms that `notes.md`'s convolution formula is not a simplification of what frameworks actually compute — it *is* what they compute, just executed with optimized, vectorized/GPU code instead of Python loops. The only thing that changes when a `Conv2D` layer is trained is that the kernel values (fixed here, hand-picked Sobel weights) become learnable parameters, updated by the same backpropagation and gradient descent mechanism derived in `01-ann/notes.md`.

## Practical implementation

`cnn-image-classification.ipynb` trains a Keras `Sequential` CNN (`Conv2D` + `MaxPooling2D` stack, followed by `Flatten` and `Dense` layers) on `tf.keras.datasets.fashion_mnist` — 10 classes of clothing items in $28\times28$ grayscale images. The notebook covers: normalizing pixel values, building and training the CNN, plotting training/validation accuracy and loss curves, reporting test accuracy, and visualizing a grid of sample test images with their true vs. predicted labels.

The mapping back to "From-scratch implementation" is direct: each `Conv2D(filters, kernel_size, activation="relu")` layer performs exactly the manual convolution loop above, once per filter, over every training image, with the filter weights initialized randomly (per `01-ann`'s weight-initialization discussion) and updated by backpropagation rather than fixed by hand; `MaxPooling2D` performs the max-pooling operation described above; `Flatten` and `Dense` are unchanged from `01-ann`.

> **Note:** a trained CNN like this is a natural candidate for a lightweight **Streamlit** web app: the model is serialized (`model.save(...)`), loaded in a Streamlit script, and wrapped with a simple UI (`st.file_uploader` for an image, `st.image` to display it, and the model's `predict()` output rendered with `st.bar_chart` or `st.write`) so a user can upload a clothing image and see the predicted class and confidence interactively. This deployment step is a suggested follow-on exercise, not delivered as part of this notebook — no Streamlit app is included here.

## Experiment

**Hypothesis (stated before running):** if the convolutional feature extractor is learning useful, generalizable patterns rather than overfitting the 60,000 Fashion-MNIST training images, training and validation accuracy should rise together and training and validation loss should fall together across epochs, rather than validation metrics stalling or reversing while training metrics keep improving.

**Setup:** `cnn-image-classification.ipynb` trains the `Conv2D`/`MaxPooling2D`/`Dense` CNN described above on Fashion-MNIST with a held-out validation split, recording accuracy and loss per epoch via Keras's `History` callback, then plots training vs. validation curves for both metrics.

**Result:** the notebook's plotted curves show training and validation accuracy rising together and loss falling together across epochs, with the model reaching a reported test accuracy consistent with the final validation accuracy; the sample-prediction grid visualizes specific correct and incorrect predictions alongside their true labels.

**Interpretation:** the close tracking between training and validation curves indicates the CNN's learned filters generalize to unseen images rather than memorizing the training set — consistent with the parameter-sharing argument in "Why simpler approaches fail" (far fewer independent parameters than a flatten-and-`Dense` approach on the same images, so less capacity to overfit).

**Limitations:** this is a single dataset (Fashion-MNIST — small, grayscale, centered, low-resolution images) and a single architecture; it demonstrates that this specific CNN generalizes on this specific benchmark, not that any CNN architecture generalizes on arbitrary, larger, or noisier real-world images.

## Failure modes

- **Small training sets still overfit CNNs**, despite parameter sharing reducing the *count* of parameters relative to a dense-on-flattened-pixels model — a CNN with many layers/filters can still have millions of parameters, more than enough to memorize a small dataset. Mitigations: data augmentation, dropout in the `Dense` head, and transfer learning from a pretrained backbone.
- **Excessive downsampling loses spatial detail** needed for fine-grained tasks (e.g. exact object localization) — aggressive pooling/striding trades resolution for translation invariance and compute savings, which is the wrong trade for tasks like segmentation that need precise per-pixel output.
- **Vanishing/exploding gradients in very deep CNNs**, for the same chain-rule reasons discussed in `01-ann/notes.md` — mitigated in practice with Batch Normalization (introduced here: normalizing each layer's inputs to zero mean/unit variance per mini-batch, which keeps activations in a well-scaled range and lets deeper CNNs train reliably) and residual/skip connections in deeper architectures (ResNet-style).
- **Kernel size and receptive field mismatch** — a small kernel (e.g. $3\times3$) in a shallow network may never build a large enough effective receptive field to recognize large objects; stacking many conv layers (each expanding the effective receptive field) or using larger strides/kernels addresses this, but naive over-large kernels re-introduce much of the parameter cost convolution was meant to avoid.
- **Distribution shift between train and deployment images** — a CNN trained on centered, evenly-lit, single-object images (like Fashion-MNIST) can degrade sharply on cluttered, differently-lit, or differently-cropped real-world photos, since translation invariance handles position shifts but not fundamentally different image statistics.

## Real-world usage

CNNs (and CNN-derived backbones) remain the standard architecture for image classification, object detection, and segmentation in production computer vision systems — medical image screening, manufacturing defect detection, autonomous vehicle perception, satellite imagery analysis, and content moderation all build on the same conv→pool→dense pattern developed here, typically starting from a large pretrained backbone (transfer learning) rather than training from scratch on a small task-specific dataset. The parameter-sharing argument from "Why simpler approaches fail" is precisely why this is computationally feasible at production image resolutions, where a flatten-and-`Dense` approach would be intractable.

## Mental model

A CNN is "a small, shared pattern-detector, swept across the whole image, instead of one giant pattern-detector wired individually to every pixel": convolution trades the ability to have position-specific weights (which a `Dense` layer has but rarely needs for images) for a massive reduction in parameters and the ability to recognize a pattern wherever it appears — exactly the trade the parameter-count comparison in "Why simpler approaches fail" makes concrete.

## Questions to think about

1. The from-scratch notebook confirms `scipy.signal.convolve2d` matches the manual loop only after flipping the kernel, while `tf.nn.conv2d` matches without any flip. What is actually different between mathematical convolution and the cross-correlation that deep learning frameworks call "convolution," and why doesn't the distinction matter for a *learned* kernel?
2. If you doubled every spatial dimension of the input image (keeping kernel size fixed), how would the parameter count of a `Conv2D` layer change? How would the parameter count of an equivalent flatten-then-`Dense` layer change? What does that difference tell you about why CNNs scale to high-resolution images and dense-on-pixels approaches don't?
3. Max pooling and average pooling both downsample a feature map. In what situation would average pooling's "dilution" of a strong signal actually be preferable to max pooling's "keep only the strongest activation"?
4. Translation invariance means a CNN detects a pattern regardless of *where* it appears in the image. Give a real-world image task where this assumption is actively wrong (i.e. position genuinely matters for the correct answer), and explain what would need to change in the architecture to handle it.
5. Why does stacking several small (e.g. $3\times3$) convolutional layers, rather than using one large (e.g. $7\times7$) layer, both keep the effective receptive field large and keep the parameter count low?
