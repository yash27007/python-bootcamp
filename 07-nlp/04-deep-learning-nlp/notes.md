# 04 – Deep Learning for NLP

| Topic | Status |
|-------|--------|
| Introduction to NLP in Deep Learning | ✅ Complete |
| Word Embedding Layers (Keras) | ✅ Complete |
| Text Classification with RNN | ✅ Complete |
| Text Classification with LSTM | ✅ Complete |

## Introduction to NLP in Deep Learning

Classical NLP pipelines (topics 01–03) hand-engineer features: tokenise, stem/lemmatise, then represent text as BOW, TF-IDF, or pre-trained Word2Vec vectors before handing the result to a traditional classifier. **Deep learning for NLP** instead learns the text representation *and* the classifier jointly, end-to-end, as part of a single neural network trained on the task itself.

The typical deep-learning NLP pipeline for text classification is:

1. **Tokenise & integer-encode**: map each word in the vocabulary to an integer index (this is different from one-hot/BOW — it's just an ID lookup key).
2. **Pad/truncate to a fixed length**: neural network layers (especially the ones after embedding, like RNN/LSTM/Dense) generally expect fixed-shape batches.
3. **Embedding layer**: map each integer word ID to a dense vector, learned during training.
4. **Sequence layer** (RNN/LSTM/GRU): process the sequence of embeddings, capturing order and context.
5. **Dense output layer**: produce the final prediction (e.g. sigmoid for binary sentiment classification).

The key shift from topic 03 is that the embeddings here are **learned specifically for the downstream task** (end-to-end, via backpropagation through the whole network) rather than learned upfront by a separate self-supervised objective (Word2Vec) and then reused.

## Word Embedding Layers (Keras)

Keras's `Embedding` layer is a trainable lookup table: it maps each integer word index to a dense vector of a chosen dimensionality.

```python
from tensorflow.keras.layers import Embedding
Embedding(input_dim=vocab_size, output_dim=embedding_dim)
```

- `input_dim`: the size of the vocabulary (largest word index + 1).
- `output_dim`: the dimensionality of the dense embedding vectors (e.g. 32, 64, 128).

**Learned vs. pretrained embeddings**:
- **Learned from scratch**: the `Embedding` layer's weights are randomly initialized and updated by gradient descent along with the rest of the network, specialising the embeddings to the specific task and dataset. This is what this topic's notebook does.
- **Pretrained**: the layer's weights are initialized from vectors trained elsewhere on a much larger corpus (e.g. Word2Vec, GloVe) and either frozen (`trainable=False`) or fine-tuned. Pretrained embeddings help most when the task's own training data is small, since they bring in semantic knowledge learned from a much larger corpus.

Internally, the `Embedding` layer is mathematically equivalent to a `Dense` layer applied to a one-hot input, but implemented as an efficient lookup rather than a matrix multiplication — this is the practical, scalable version of the one-hot-encoding idea from topic 02, made dense and learnable.

## Padding/Truncating Sequences

Real documents vary in length, but a batch of sequences fed to a neural network must share a common shape. Keras's `pad_sequences` utility standardises sequence length by:

- **Padding** shorter sequences with a filler value (typically `0`) up to a fixed `maxlen`, either at the start (`padding='pre'`, the default — often preferred for RNN/LSTM since it keeps the most recent, most relevant tokens closest to the end of the sequence) or the end (`padding='post'`).
- **Truncating** longer sequences down to `maxlen`, again either from the start (`truncating='pre'`) or the end (`truncating='post'`, the default).

The choice of `maxlen` trades off information loss (too short, and long documents get cut) against compute/memory cost and the risk of the model having to learn to ignore a lot of padding (too long). The padding value `0` is deliberately reserved as a special index (not used for any real word) so that layers can optionally mask it out (`mask_zero=True` in `Embedding`) to prevent it from influencing the sequence computation.

## Text Classification with RNN / LSTM

A **Recurrent Neural Network (RNN)** processes a sequence one token at a time, maintaining a hidden state $h_t$ that is updated at each step as a function of the current input and the previous hidden state:
$$h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$$
This lets the network, in principle, use information from earlier in the sequence when processing later tokens — unlike BOW/TF-IDF or AvgWord2Vec, order matters.

Plain RNNs struggle with **long-range dependencies** because gradients propagated back through many time steps tend to vanish (or explode), so information from early tokens gets washed out by the time the network reaches the end of a long sequence.

**LSTM (Long Short-Term Memory)** addresses this with a more elaborate cell that maintains a separate **cell state** $C_t$ alongside the hidden state, regulated by three gates:
- **Forget gate**: decides what to discard from the cell state.
- **Input gate**: decides what new information to write into the cell state.
- **Output gate**: decides what part of the cell state to expose as the hidden state/output.

These gates let gradients flow through the cell state largely unimpeded across many time steps, which is why LSTMs handle longer sequences and longer-range dependencies far better than plain RNNs in practice.

For **text classification** with either architecture, the typical Keras model stacks:
```
Embedding → LSTM (or SimpleRNN) → Dense(1, activation='sigmoid')
```
The recurrent layer's final hidden state (or `return_sequences=False` output) summarises the whole sequence into a fixed-size vector, which the final `Dense` layer turns into a class probability (e.g. positive/negative sentiment on the IMDB dataset used in this topic's notebook). This topic's notebook uses an LSTM specifically, since it converges more reliably than a plain RNN on the moderately long movie-review sequences in IMDB.
