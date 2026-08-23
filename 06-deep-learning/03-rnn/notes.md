# 03 – Recurrent Neural Networks

## Problem

`01-ann` and `02-cnn` both assume a fixed-size input: an MLP takes a fixed-length feature vector, a CNN takes a fixed-shape image tensor. Many real problems don't come in that shape at all — they come as **sequences** whose length varies and whose **order carries meaning**: a movie review is a variable-length string of words, and "the acting was great, not the plot" means something different from "the plot was great, not the acting" even though both sentences contain the same words. A time series (stock price, sensor reading) has the same variable-length property, plus the extra constraint that today's value is best predicted using yesterday's, and the day before that, not some arbitrary fixed-size snapshot. Neither an ANN nor a CNN has any built-in notion of "this input comes after that one" — something new is needed that processes a sequence element by element and carries information from earlier elements forward.

## Intuition

Reading a sentence one word at a time, you don't throw away everything you've read so far when a new word arrives — you keep a running mental summary ("so far, this review sounds positive... but now 'not' just showed up, that might flip things") and update it as each new word comes in. A **Recurrent Neural Network (RNN)** formalizes exactly that: it keeps a **hidden state** — a fixed-size vector summarizing "everything relevant seen so far" — and at every timestep it combines the current input with that running summary to produce an updated summary. The same update rule (the same weights) is reused at every position in the sequence, the way the same reading strategy is reused for every new word regardless of where in the sentence it appears.

## Why simpler approaches fail

Two naive fixes using only the tools from `01-ann`/`02-cnn` both break down:

- **Flatten the sequence into a fixed-size vector and feed it to a `Dense` layer.** This requires picking one fixed length in advance (padding shorter sequences, truncating longer ones), and once flattened, a `Dense` layer's weights are position-specific: the weight connecting "word at position 5" to the output is a *different* weight than the one connecting "word at position 50." The model has no way to recognize that the same word means roughly the same thing whether it appears at position 5 or position 50 — there is no weight sharing across positions, so the network must independently relearn what a word means at every possible position, and it still discards information about relative order beyond the fixed positional slots.
- **Treat each timestep independently (e.g., average word vectors, or classify word-by-word and vote).** This discards order entirely — "dog bites man" and "man bites dog" would produce the identical bag-of-words representation despite opposite meanings — and cannot represent the fact that an early input should influence a decision made many steps later (e.g., resolving what a pronoun near the end of a sentence refers to, based on a noun near the beginning).

What's needed is an architecture that (a) shares the same weights across every position in the sequence, so it doesn't need separate parameters per position, and (b) carries forward a running summary that lets earlier inputs influence outputs computed many steps later. That is exactly what the recurrence relation below provides.

## Mathematical foundation

### RNN architecture (vs ANN)

An RNN processes a sequence $x_1, x_2, \dots, x_T$ one timestep at a time, maintaining a hidden state $h_t$ that is updated at every step and carried forward to the next:

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
$$\hat{y}_t = g(W_{hy} h_t + b_y)$$

Here $W_{xh}$ maps the current input into the hidden space, $W_{hh}$ maps the previous hidden state forward (this is the "recurrent" connection that gives the network memory), and $W_{hy}$ maps the hidden state to an output. Crucially, **the same weights $W_{xh}, W_{hh}, W_{hy}$ are reused at every timestep** — this is a form of parameter sharing across time, analogous to how a CNN kernel is shared across space, and it is the direct fix for the position-specific-weights failure mode above.

Depending on the task, an RNN can be used as:
- **Many-to-one:** a full sequence in, a single output at the end (e.g. sentiment classification — used in this topic's notebook).
- **Many-to-many:** an output at every timestep (e.g. part-of-speech tagging).
- **One-to-many / sequence-to-sequence:** covered conceptually in topic `05-attention-transformers`.

### Forward & backward propagation through time

**Forward propagation** in an RNN simply applies the recurrence formula above step by step, starting from an initial hidden state $h_0$ (usually zeros), producing $h_1, h_2, \dots, h_T$ and, if needed, an output at each step.

**Backpropagation Through Time (BPTT)** is backpropagation applied to this "unrolled" computation graph. Conceptually, the RNN is unrolled into a chain of $T$ copies of the same cell (one per timestep), each sharing the same weights, and the standard chain rule is applied across the unrolled graph. The gradient of the loss with respect to the recurrent weight $W_{hh}$ accumulates contributions from **every timestep**, because $W_{hh}$ was used repeatedly:

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial h_t} \left( \prod_{k=t}^{2} \frac{\partial h_k}{\partial h_{k-1}} \right) \frac{\partial h_1}{\partial W_{hh}}$$

The product term $\prod \frac{\partial h_k}{\partial h_{k-1}}$ — a product of many Jacobians, one per timestep between $t$ and the start — is the crux of the vanishing/exploding gradient problem covered in "Failure modes" below: the longer the sequence, the more factors are multiplied together.

## Algorithm

Training a vanilla RNN by BPTT proceeds as:

1. Initialize $W_{xh}, W_{hh}, W_{hy}$ (small random values) and biases; set $h_0 = \mathbf{0}$.
2. **Forward pass:** for $t = 1, \dots, T$, compute $h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$, reusing the same $W_{xh}, W_{hh}, b_h$ at every step; compute $\hat y_t$ where needed (or only $\hat y_T$ for a many-to-one task).
3. Compute the loss $L$ (e.g. binary cross-entropy for sentiment classification, using the final $\hat y_T$).
4. **Backward pass (BPTT):** unroll the computation graph over all $T$ steps and apply the chain rule backward through it, accumulating the gradient of $L$ with respect to $W_{xh}, W_{hh}, W_{hy}$ across every timestep, per the summation formula above.
5. **Update:** apply a gradient-descent step (typically Adam) to all shared weights.
6. Repeat over mini-batches/epochs until convergence.

## From-scratch implementation

Implemented in `rnn-from-scratch-unroll.ipynb`: a single `SimpleRNN` cell's forward pass, in plain NumPy, unrolled by hand over a 4-timestep toy sequence of 3-dimensional input vectors.

1. Initializes $W_{xh}$, $W_{hh}$, $b_h$ **once** and explicitly reuses those exact same arrays for all four timesteps — a direct, executed demonstration of the parameter-sharing claim in "Mathematical foundation."
2. Computes $h_1, h_2, h_3, h_4$ one at a time by hand, applying $h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$ explicitly at each step (rather than hiding the recurrence inside a loop), then wraps the same four steps into a `simple_rnn_forward` loop and confirms the loop reproduces the identical hand-computed values.
3. Plots each hidden unit's value across the four timesteps, visualizing the hidden state's evolution.
4. Perturbs only the *first* input $x_1$ and re-runs the unroll, showing the final hidden state $h_4$ changes — proof that information genuinely propagates forward through the chain of $W_{hh}$ multiplications, not just from the most recent input.
5. Cross-checks the entire hand-computed forward pass against `tf.keras.layers.SimpleRNN` loaded with the exact same weights, confirming both produce identical hidden states.

This is the same recurrence formula and the same parameter-sharing property described above — worked through explicitly, timestep by timestep, instead of executed inside a framework's compiled loop.

## Practical implementation

**Word embedding layers.** Text must be converted to numbers before a neural network can process it. A naive approach — one-hot encoding each word from a vocabulary of size $V$ — produces extremely sparse, high-dimensional vectors that carry no notion of similarity (any two distinct words are equally "different"). An **embedding layer** instead learns a dense, low-dimensional vector representation for each word in the vocabulary, as part of training the overall model. Keras's `Embedding(input_dim=vocab_size, output_dim=embedding_dim)` layer is essentially a lookup table: it maps each integer word-index to a trainable dense vector of length `embedding_dim`. Words that behave similarly in the training data tend to end up with similar embedding vectors (nearby in the embedding space), which gives the downstream RNN a much richer, denser signal to work with than one-hot vectors, and drastically reduces the number of input dimensions the recurrent layer must process.

**`rnn-imdb-sentiment.ipynb`** builds a binary sentiment classifier (positive/negative movie review) using `tf.keras.datasets.imdb`, restricted to the top 10,000 most frequent words. Reviews are padded/truncated to a fixed length so they can be batched, then passed through an `Embedding` layer followed by a `SimpleRNN` layer and a sigmoid output neuron. This maps directly back to "From-scratch implementation": the `Embedding` layer converts each token into the dense input vector $x_t$ that the from-scratch notebook fed in by hand, and Keras's `SimpleRNN` layer runs the exact recurrence $h_t = \tanh(W_{xh}x_t + W_{hh}h_{t-1} + b_h)$ across the whole padded sequence — the same computation the from-scratch notebook verified cell-by-cell, just vectorized and compiled rather than written as an explicit Python loop, and with the weights learned by gradient descent instead of set by hand.

## Experiment

**Hypothesis (stated before running):** if the `SimpleRNN`-based sentiment classifier is learning useful sequential structure rather than memorizing the training set, its training and validation accuracy should both improve together in early epochs and then the validation curve should plateau or start diverging from the training curve as overfitting sets in — since a vanilla RNN, per "Failure modes" below, has limited capacity to exploit truly long-range dependencies and will lean on simpler local patterns that are relatively easy to overfit to.

**Setup:** `rnn-imdb-sentiment.ipynb` trains the `Embedding` → `SimpleRNN` → sigmoid model on the IMDB training split with a held-out validation split, recording per-epoch training/validation accuracy and loss via Keras's `History` callback.

**Result:** the notebook's plotted training/validation accuracy and loss curves (see `rnn-imdb-sentiment.ipynb`) and reported test accuracy — cited here rather than reproduced, since the notebook is the source of record — show the expected pattern of improving accuracy with a validation curve that plateaus below the training curve, consistent with the hypothesis.

**Interpretation:** the model learns useful signal from word order and embeddings beyond a bag-of-words baseline, but a vanilla `SimpleRNN`'s accuracy on this task is a baseline to be compared against the LSTM/GRU models in `04-lstm-gru`, which is exactly how `lstm-gru-time-series.ipynb`'s three-way comparison is framed in that topic.

**Limitations:** a single dataset, one architecture, one train/validation split, and no hyperparameter search — the experiment demonstrates the expected training dynamics in this setting, not that a `SimpleRNN` is competitive with gated architectures in general.

## Failure modes

Because each $\frac{\partial h_k}{\partial h_{k-1}}$ term involves multiplying by $W_{hh}$ and the derivative of $\tanh$ (which is $\le 1$, and typically much less for saturated inputs), the product $\prod_{k=t}^{2}\frac{\partial h_k}{\partial h_{k-1}}$ shrinks exponentially as the number of timesteps between $t$ and $1$ grows. This is the same **vanishing gradient** phenomenon from `01-ann`, but here it is caused by *depth in time* rather than depth in layers.

Practical consequence: a vanilla ("Simple") RNN struggles to learn **long-range dependencies** — connections between tokens that are far apart in the sequence (e.g. resolving a pronoun that refers to a noun introduced 50 words earlier). Gradients from distant timesteps become vanishingly small by the time they reach early timesteps, so those early inputs barely influence the weight updates. The symmetric failure mode, **exploding gradients**, can also occur when $W_{hh}$ has large eigenvalues, causing unstable training (often mitigated in practice with gradient clipping).

This limitation — vanilla RNNs cannot reliably carry information across many timesteps — directly motivates the **LSTM** and **GRU** architectures covered in the next topic (`04-lstm-gru`), which introduce gating mechanisms specifically designed to preserve gradient flow over long sequences.

## Real-world usage

Vanilla RNNs are rarely deployed as-is in modern systems — LSTM/GRU (next topic) or attention-based architectures (`05-attention-transformers`) have mostly superseded them for anything beyond short sequences — but the recurrence idea underlies all of them, and `SimpleRNN` remains a useful, cheap baseline for short-sequence tasks (e.g. short-window sensor classification) and a pedagogical stepping stone: understanding where and why it fails on long sequences is exactly what motivates every architecture built to fix it. The word-embedding-layer pattern used here (`Embedding` → sequence model) is the standard entry point for almost every NLP pipeline covered later in this repository (`07-nlp`), regardless of what sequence model follows the embedding layer.

## Mental model

An RNN is "the same small function, applied over and over, carrying a running summary forward": at every timestep it reads the current input and its own previous summary, and writes a new summary using the *same* weights every time — parameter sharing across time, the sequence analogue of a CNN's parameter sharing across space. Its Achilles' heel is exactly what makes it work: the same repeated multiplication that carries information forward through the hidden state also shrinks gradients exponentially on the way back, which is why long sequences defeat it.

## Questions to think about

1. Why does reusing the *same* $W_{xh}, W_{hh}$ at every timestep (rather than learning separate weights per position) let an RNN generalize to sequences of a length it never saw during training, in a way that a `Dense` layer applied to a flattened, fixed-length sequence cannot?
2. The perturbation experiment in `rnn-from-scratch-unroll.ipynb` shows $h_4$ changes when $x_1$ changes. Would you expect that change to be larger or smaller if the toy sequence had 40 timesteps instead of 4, and why does the BPTT gradient formula predict that?
3. `rnn-imdb-sentiment.ipynb` pads/truncates every review to the same fixed length before feeding it to the model. Does this padding reintroduce any of the position-specific-weight problems described in "Why simpler approaches fail," or does the recurrence structure avoid that? Why?
4. If you used a linear activation instead of $\tanh$ in the hidden-state update, what would happen to the vanishing/exploding gradient behavior, and why does $\tanh$'s bounded output range matter here beyond just introducing non-linearity?
5. A many-to-one RNN (e.g. the sentiment classifier) only produces output at the final timestep. What information, if any, is lost compared to a many-to-many architecture that produces an output at every timestep, and when would that loss matter?
