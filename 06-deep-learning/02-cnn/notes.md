# 02 – Convolutional Neural Networks

| Topic | Status |
|-------|--------|
| CNN Intuition (Human Brain vs CNN) | ✅ Complete |
| Images, Convolution, Padding, Pooling | ✅ Complete |
| Flattening & Fully Connected Layers | ✅ Complete |
| CNN Classification Project + Streamlit Deployment | ✅ Complete |

## CNN Intuition (Human Brain vs CNN)

The visual cortex processes images hierarchically: early neurons respond to simple, local stimuli (edges, orientations, color contrast in a small receptive field), and progressively deeper neurons combine those simple signals into detectors for shapes, textures, and eventually whole objects and faces. Crucially, biological visual neurons are **locally connected** — each neuron only "sees" a small patch of the visual field (its receptive field), not the entire image at once.

A **Convolutional Neural Network (CNN)** is directly inspired by this structure. Instead of a plain ANN's `Dense` layer where every output neuron connects to every input pixel (an enormous number of independent weights for a high-resolution image), a CNN layer slides a small, shared filter across the image, so each output value only depends on a local neighborhood of the input — mimicking a local receptive field. Layers early in a CNN learn to detect edges and simple textures; deeper layers combine those into parts (eyes, wheels) and eventually whole-object concepts, mirroring the brain's hierarchical processing.

## Images, Convolution, Padding, Pooling

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

### Pooling

Pooling layers **downsample** feature maps, reducing spatial resolution (and hence computation) while retaining the most important information and adding a degree of local translation invariance.

- **Max pooling:** for each window (e.g. $2\times2$), output the maximum value. This preserves the strongest activation of a detected feature within that window, which is usually what matters for recognition.
- **Average pooling:** output the average value within each window — smoother, but can dilute strong signals.

`MaxPooling2D` is the standard choice in CNN classifiers and is used in this topic's notebook.

## Flattening & Fully Connected Layers

After several conv→pool blocks, the network holds a stack of small, deep feature maps that summarize *what* patterns are present, but the classifier needs a single probability distribution over classes. The **Flatten** layer reshapes the final 3D feature-map tensor `(height, width, channels)` into a 1D vector, which is then fed into one or more **fully connected (`Dense`)** layers — exactly the ANN layers from topic `01-ann`. The final `Dense` layer uses a softmax activation (multi-class) or sigmoid (binary) to produce class probabilities.

Typical CNN architecture:

$$\text{Input} \to [\text{Conv2D} \to \text{ReLU} \to \text{MaxPooling2D}]\times N \to \text{Flatten} \to \text{Dense} \to \text{Dense(softmax)}$$

Convolutional layers act as an automatic, learned feature extractor; the dense layers at the end act as the classifier operating on those extracted features.

## Why CNNs Beat Plain ANNs on Images

- **Parameter sharing:** a kernel's weights are reused at every spatial position, so a $3\times3\times3$ kernel has only 27 weights (+bias) regardless of image size, versus a `Dense` layer needing one weight per pixel per output neuron. This drastically reduces parameter count and overfitting risk.
- **Translation invariance:** because the same kernel scans the whole image, a pattern (e.g. an edge or a cat's ear) is detected wherever it appears, without needing to be retrained per position. Pooling further strengthens this by tolerating small shifts.
- **Local connectivity:** each output value depends only on a small local patch of the input, matching the local/spatial structure of images (nearby pixels are correlated; distant pixels usually aren't directly related) — a `Dense` layer ignores this structure entirely and treats every pixel as independent.
- **Hierarchical feature learning:** stacking conv→pool blocks lets the network build increasingly abstract representations (edges → textures → parts → objects) without hand-engineering features, unlike classical computer vision pipelines.

## CNN Classification Project + Streamlit Deployment

`cnn-image-classification.ipynb` trains a Keras `Sequential` CNN (`Conv2D` + `MaxPooling2D` stack, followed by `Flatten` and `Dense` layers) on `tf.keras.datasets.fashion_mnist` — 10 classes of clothing items in $28\times28$ grayscale images. The notebook covers: normalizing pixel values, building and training the CNN, plotting training/validation accuracy and loss curves, reporting test accuracy, and visualizing a grid of sample test images with their true vs. predicted labels.

A trained CNN like this is a natural candidate for a lightweight **Streamlit** web app: the model is serialized (`model.save(...)`), loaded in a Streamlit script, and wrapped with a simple UI (`st.file_uploader` for an image, `st.image` to display it, and the model's `predict()` output rendered with `st.bar_chart` or `st.write`) so a user can upload a clothing image and see the predicted class and confidence interactively. This deployment step is a natural follow-on exercise once the notebook's saved model exists, and is outside the scope of the notebook itself.
