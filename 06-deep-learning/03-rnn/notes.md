# 03 – Recurrent Neural Networks

| Topic | Status |
|-------|--------|
| RNN Architecture (vs ANN) | ✅ Complete |
| Forward & Backward Propagation Through Time | ✅ Complete |
| Problems with Vanilla RNN | ✅ Complete |
| Word Embedding Layers | ✅ Complete |
| IMDB Sentiment Analysis (Simple RNN) | ✅ Complete |

## Why Sequence Models

Plain ANNs and CNNs assume inputs are fixed-size, independent vectors — there is no notion of "order" or "what came before." Many real-world problems are fundamentally **sequential**: text (word order changes meaning — "dog bites man" vs. "man bites dog"), time series (today's value depends on yesterday's), audio, and video. Two properties make sequences special:

- **Order matters:** the same set of tokens in a different order can have a completely different meaning.
- **Variable length:** sentences, sensor readings, and audio clips come in different lengths, but a `Dense` layer expects a fixed-size input vector.

**Recurrent Neural Networks (RNNs)** are designed specifically to handle sequences by processing one element at a time while maintaining a **hidden state** that summarizes everything seen so far.

## RNN Architecture (vs ANN)

An ANN maps a fixed input directly to an output with no memory of previous inputs. An RNN instead processes a sequence $x_1, x_2, \dots, x_T$ one timestep at a time, maintaining a hidden state $h_t$ that is updated at every step and carried forward to the next:

$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
$$\hat{y}_t = g(W_{hy} h_t + b_y)$$

Here $W_{xh}$ maps the current input into the hidden space, $W_{hh}$ maps the previous hidden state forward (this is the "recurrent" connection that gives the network memory), and $W_{hy}$ maps the hidden state to an output. Crucially, **the same weights $W_{xh}, W_{hh}, W_{hy}$ are reused at every timestep** — this is a form of parameter sharing across time, analogous to how a CNN kernel is shared across space.

Depending on the task, an RNN can be used as:
- **Many-to-one:** a full sequence in, a single output at the end (e.g. sentiment classification — used in this topic's notebook).
- **Many-to-many:** an output at every timestep (e.g. part-of-speech tagging).
- **One-to-many / sequence-to-sequence:** covered conceptually in topic `05-attention-transformers`.

## Forward & Backward Propagation Through Time

**Forward propagation** in an RNN simply applies the recurrence formula above step by step, starting from an initial hidden state $h_0$ (usually zeros), producing $h_1, h_2, \dots, h_T$ and, if needed, an output at each step.

**Backpropagation Through Time (BPTT)** is backpropagation applied to this "unrolled" computation graph. Conceptually, the RNN is unrolled into a chain of $T$ copies of the same cell (one per timestep), each sharing the same weights, and the standard chain rule is applied across the unrolled graph. The gradient of the loss with respect to the recurrent weight $W_{hh}$ accumulates contributions from **every timestep**, because $W_{hh}$ was used repeatedly:

$$\frac{\partial L}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial L_t}{\partial h_t} \left( \prod_{k=t}^{2} \frac{\partial h_k}{\partial h_{k-1}} \right) \frac{\partial h_1}{\partial W_{hh}}$$

The product term $\prod \frac{\partial h_k}{\partial h_{k-1}}$ — a product of many Jacobians, one per timestep between $t$ and the start — is the crux of the vanishing/exploding gradient problem discussed next: the longer the sequence, the more factors are multiplied together.

## Problems with Vanilla RNN

Because each $\frac{\partial h_k}{\partial h_{k-1}}$ term involves multiplying by $W_{hh}$ and the derivative of $\tanh$ (which is $\le 1$, and typically much less for saturated inputs), the product $\prod_{k=t}^{2}\frac{\partial h_k}{\partial h_{k-1}}$ shrinks exponentially as the number of timesteps between $t$ and $1$ grows. This is the same **vanishing gradient** phenomenon from `01-ann`, but here it is caused by *depth in time* rather than depth in layers.

Practical consequence: a vanilla ("Simple") RNN struggles to learn **long-range dependencies** — connections between tokens that are far apart in the sequence (e.g. resolving a pronoun that refers to a noun introduced 50 words earlier). Gradients from distant timesteps become vanishingly small by the time they reach early timesteps, so those early inputs barely influence the weight updates. The symmetric failure mode, **exploding gradients**, can also occur when $W_{hh}$ has large eigenvalues, causing unstable training (often mitigated in practice with gradient clipping).

This limitation — vanilla RNNs cannot reliably carry information across many timesteps — directly motivates the **LSTM** and **GRU** architectures covered in the next topic (`04-lstm-gru`), which introduce gating mechanisms specifically designed to preserve gradient flow over long sequences.

## Word Embedding Layers

Text must be converted to numbers before a neural network can process it. A naive approach — one-hot encoding each word from a vocabulary of size $V$ — produces extremely sparse, high-dimensional vectors that carry no notion of similarity (any two distinct words are equally "different").

An **embedding layer** instead learns a dense, low-dimensional vector representation for each word in the vocabulary, as part of training the overall model. Keras's `Embedding(input_dim=vocab_size, output_dim=embedding_dim)` layer is essentially a lookup table: it maps each integer word-index to a trainable dense vector of length `embedding_dim`. Words that behave similarly in the training data tend to end up with similar embedding vectors (nearby in the embedding space), which gives the downstream RNN a much richer, denser signal to work with than one-hot vectors, and drastically reduces the number of input dimensions the recurrent layer must process.

In `rnn-imdb-sentiment.ipynb`, the embedding layer is the very first layer of the network, converting padded integer-encoded reviews into dense vector sequences before they are fed into the `SimpleRNN` layer.

## IMDB Sentiment Analysis (Simple RNN)

`rnn-imdb-sentiment.ipynb` builds a binary sentiment classifier (positive/negative movie review) using `tf.keras.datasets.imdb`, restricted to the top 10,000 most frequent words. Reviews are padded/truncated to a fixed length so they can be batched, then passed through an `Embedding` layer followed by a `SimpleRNN` layer and a sigmoid output neuron. The notebook trains the model, plots training/validation accuracy and loss curves, and reports test accuracy — providing a concrete baseline to compare against the LSTM/GRU models in the next topic.
