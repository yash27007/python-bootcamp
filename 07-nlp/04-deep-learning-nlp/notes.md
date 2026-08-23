# 04 – Deep Learning for NLP

## Problem

`07-nlp/03-word-embeddings` solved "words as isolated symbols" by learning dense vectors from co-occurrence statistics — but Word2Vec's vectors are **fixed and context-independent**: once trained, `model.wv["bank"]` returns the same vector no matter what sentence `"bank"` appears in, and that training happens as a separate, upfront, self-supervised step, disconnected from whatever downstream task (classification, translation, ...) the embeddings will ultimately be used for. Meanwhile, `06-deep-learning/03-rnn` and `04-lstm-gru` developed machinery (recurrent hidden state, LSTM gating) for processing *sequences* while respecting order — but applied it to generic sequential data, not text specifically. **How do we combine "words as dense vectors" with "order-aware sequence processing," and train both pieces jointly, end-to-end, for a specific downstream task, rather than bolting a fixed, separately-trained embedding onto a sequence model as an afterthought?**

## Intuition

Think back to `03-word-embeddings`'s AvgWord2Vec classifier: it averaged every word's embedding into one vector per document and fed that to logistic regression. Averaging worked well enough to distinguish hockey articles from space articles (95% accuracy), because those categories are dominated by a handful of very topic-specific words. But averaging is a blunt instrument — it treats a document as an unordered bag of vectors, throwing away the sequence entirely. **Deep learning for NLP** replaces "average the vectors, then classify" with "feed the *sequence* of vectors into a recurrent model that reads them in order, updating a running summary as it goes" — the same recurrence mechanic from `06-deep-learning/03-rnn`/`04-lstm-gru`, but now the inputs at each timestep are word embedding vectors instead of, e.g., stock prices or sensor readings. Crucially, in this setup the embedding vectors themselves are *not* fixed in advance by a separate Word2Vec run — they start random and are updated by backpropagation through the exact same loss the classifier is trained on, so the whole pipeline (word representation *and* sequence processing *and* final classification) is one differentiable system, learned jointly for one objective.

## Why simpler approaches fail

AvgWord2Vec (`03-word-embeddings`) computes $\vec{v}_{doc} = \frac{1}{n}\sum_i \vec{v}_{w_i}$ — a sum that is, by construction, invariant to the order the terms are summed in. `"dog bites man"` and `"man bites dog"` contain the exact same three words, so they average to the *identical* vector, despite having opposite meanings. Any task where meaning depends on word order — negation (`"not good"` vs `"good"`), who-did-what-to-whom, sarcasm, multi-clause sentences — is invisible to an averaging-based representation no matter how good the underlying word vectors are; the information was never in the input the classifier saw. This is exactly the same "order is discarded" failure that `07-nlp/02-feature-extraction` diagnosed for plain (unigram) bag-of-words, one level up: n-grams patched BOW's order-blindness locally (a bigram like `"not_good"` is one token), but a sequence model captures order globally, across the whole document, without needing to pre-decide a fixed window size.

## Mathematical foundation

### The Keras `Embedding` layer as a learned lookup table

The core conceptual point of this topic: Keras's `Embedding` layer is a **trainable weight matrix** $E \in \mathbb{R}^{|V| \times d}$ (`input_dim=|V|`, `output_dim=d`), and looking up word index $i$ simply reads off row $i$:
$$\text{Embedding}(i) = E[i, :] = \mathbf{e}_i^\top E$$
where $\mathbf{e}_i$ is word $i$'s one-hot vector — so the `Embedding` layer is mathematically identical to a `Dense` layer with no bias and no activation applied to a one-hot input, just implemented as an $O(d)$ row lookup instead of an $O(|V| \cdot d)$ matrix multiply against a mostly-zero vector. This is the same one-hot-encoding idea from `07-nlp/02-feature-extraction`, made dense — but the critical difference from `03-word-embeddings` is **what determines the values in $E$ and how**:

| | Word2Vec (`03-word-embeddings`) | Keras `Embedding` (this topic) |
|---|---|---|
| Objective $E$ is optimized for | CBOW/Skip-Gram: predict a masked word from its context (self-supervised, task-agnostic) | Whatever the downstream model is trained for (e.g. sentiment classification) |
| When trained | Once, upfront, separately from any downstream model | Jointly, end-to-end, alongside the LSTM/Dense layers, during the same training run |
| Gradient source | Cross-entropy loss on the CBOW/Skip-Gram prediction task | Backpropagated all the way from the final classification loss, through the LSTM, through the embedding lookup |
| Reusability | One embedding matrix usable across many downstream tasks | Specialized to the one task it was trained on (unless you deliberately reuse the weights elsewhere) |

Formally: in Word2Vec, $E = V_{in}$ is updated by $\nabla_{V_{in}} \mathcal{L}_{\text{CBOW/SG}}$ (a self-supervised next/context-word prediction loss, no labels needed). In this topic's model, $E$ is updated by $\nabla_E \mathcal{L}_{\text{task}}$, where $\mathcal{L}_{\text{task}}$ is e.g. binary cross-entropy on sentiment labels, and the gradient reaches $E$ only *after* flowing backward through the `Dense` output layer and every LSTM timestep first — the chain rule connects "was this review classified correctly" all the way back to "how should word 4271's embedding vector move." Two consequences follow directly from this: (1) a task-trained embedding can end up organized around whatever distinctions matter *for that task* (e.g. sentiment-bearing words spread out more than they would in a general-purpose Word2Vec space) rather than general semantic similarity; and (2) it needs enough task-labeled training data to learn good vectors from scratch, which is why **pretrained** embeddings (initializing $E$ from Word2Vec/GloVe, optionally `trainable=False` to freeze them) are preferred when labeled data is scarce — borrowing the semantic structure a self-supervised objective already learned from a much larger, unlabeled corpus.

### Sequence layer: recurrence over embedded tokens

Once each token is embedded, the model needs to process the *sequence* of embedding vectors $\mathbf{x}_1, \ldots, \mathbf{x}_T$ (one per token position) while preserving order — this is exactly the recurrence relation derived in `06-deep-learning/03-rnn`:
$$h_t = \tanh(W_x \mathbf{x}_t + W_h h_{t-1} + b)$$
and, because plain RNN hidden states are overwritten at every step and suffer vanishing gradients over long sequences (`03-rnn`'s failure mode), `04-lstm-gru`'s gated cell state $C_t$, regulated by forget/input/output gates, is used in practice for anything beyond short sequences — exactly the LSTM architecture derived in that topic, applied here with $\mathbf{x}_t = E[\text{token}_t, :]$ instead of a generic per-timestep feature vector.

### Padding as a shape constraint, not a modeling idea

Neural network layers process fixed-shape batches, but raw documents vary in length. `pad_sequences` standardizes every sequence to a fixed `maxlen` by padding shorter sequences with a reserved index (`0`, never a real word) and truncating longer ones — `padding='pre'`/`'post'` chooses which end gets the filler, `truncating='pre'`/`'post'` chooses which end gets cut. `0` is reserved specifically so `Embedding(..., mask_zero=True)` can optionally tell the recurrent layer to skip padded positions rather than let them influence $h_t$/$C_t$ as if they were real tokens.

## Algorithm

1. Build a vocabulary and integer-encode every token (an index, not a one-hot vector or a Word2Vec lookup — just an arbitrary but fixed word→ID mapping).
2. Pad/truncate every sequence to a fixed `maxlen`.
3. `Embedding` layer: look up each token ID's row in the trainable matrix $E$, producing a sequence of dense vectors $\mathbf{x}_1, \ldots, \mathbf{x}_{maxlen}$.
4. Feed the sequence through an LSTM (or SimpleRNN), producing a final hidden state that summarizes the whole sequence into one fixed-size vector.
5. `Dense(1, activation='sigmoid')` maps that summary vector to a class probability.
6. Compute the task loss (binary cross-entropy), backpropagate through the `Dense` layer, through every LSTM timestep, and into $E$ itself.
7. Update all weights (embedding matrix included) via gradient descent; repeat over epochs.

## From-scratch implementation

This topic deliberately does **not** re-derive new from-scratch mechanics, and that's a considered choice rather than an omission: everything this model does is a *composition* of two mechanics already derived from scratch elsewhere in this repository — embedding lookup (one-hot → dense vector, `03-word-embeddings`'s CBOW forward pass) and gated recurrence (`04-lstm-gru`'s forget/input/output gate derivation). Reimplementing an LSTM cell's forward/backward pass in NumPy a second time, here, with a different input source (learned embeddings instead of the toy sequences used in `04-lstm-gru`), would repeat that topic's mechanics almost verbatim without adding new insight — the "from-scratch value" of this topic is entirely in seeing *how* the two known mechanics compose (embedding lookup feeds recurrence, both trained by one shared gradient), not in re-deriving either mechanic's own math. Per the repository's from-scratch-only-where-it-adds-insight rule, that compositional understanding is documented here in prose instead of padded out with a redundant NumPy LSTM implementation: reading `03-word-embeddings`'s CBOW forward pass alongside `04-lstm-gru`'s gate equations, and substituting the embedding lookup's output vector as the LSTM's per-timestep input $x_t$, is a complete, correct from-scratch account of what this topic's Keras model computes internally — the only genuinely new fact is that both pieces are trained by *one* backward pass instead of two separate ones (see Mathematical foundation above).

## Practical implementation

`deep-learning-nlp.ipynb` builds exactly this architecture on IMDB movie reviews (binary sentiment, 25,000 train / 25,000 test reviews, pre-tokenized as integer sequences by Keras):

```python
model = keras.Sequential([
    layers.Input(shape=(MAX_LEN,)),
    layers.Embedding(input_dim=VOCAB_SIZE, output_dim=32),
    layers.LSTM(...),
    layers.Dense(1, activation="sigmoid"),
])
```

with `VOCAB_SIZE=10000`, `MAX_LEN=200` (pad/truncate every review to 200 tokens), and a 32-dimensional learned embedding (328,353 total trainable parameters, of which $10000 \times 32 = 320{,}000$ belong to the embedding matrix alone — the largest single parameter block in the model, and the one this topic's math section is about). This is the from-scratch composition above, executed at production scale: `Embedding` performs the same row-lookup as the toy CBOW example, `LSTM` performs the same gated recurrence as `04-lstm-gru`'s derivation, and Keras's autodiff handles the joint backward pass through both.

## Experiment

**Hypothesis**: because the `Embedding` layer here is trained jointly with the LSTM specifically for sentiment classification (rather than for general-purpose word similarity), and because the LSTM processes review text in order rather than averaging it, the model should reach meaningfully higher accuracy than a bag-of-words or averaged-embedding baseline would on the same task, while remaining vulnerable to overfitting given IMDB's modest training set size (25,000 reviews) relative to the 320,000-parameter embedding matrix alone.

**Setup**: train the `Embedding(10000, 32) → LSTM → Dense(1, sigmoid)` model above for 8 epochs (with early stopping on validation loss) on 20,000 IMDB training reviews (5,000 held out for validation), then evaluate on the full 25,000-review test set.

**Result**: training accuracy climbed from 73.4% (epoch 1) to 93.3% (epoch 4), while validation accuracy peaked around 87.8% (epoch 2) and then declined as validation loss rose (0.30 → 0.46 by epoch 5) — a clear overfitting signature, consistent with the hypothesis that a large embedding matrix trained on a comparatively modest labeled set is prone to memorizing training-specific patterns. Final **test accuracy: 86.94%** (test loss 0.3144). Qualitatively, decoding individual test reviews back to text and comparing true vs. predicted sentiment showed correct low-confidence-of-negativity scores (e.g. score 0.106, correctly predicted negative) on a clearly negative review ("terrible performances", "flat flat flat").

**Limitations**: 86.9% is respectable but well short of what fine-tuned pretrained-embedding or Transformer-based models reach on IMDB (typically 90%+); the model was capped at `VOCAB_SIZE=10000` and `MAX_LEN=200`, so rarer words and anything past the first 200 tokens of a review are invisible to it; and the visible overfitting gap between training and validation accuracy indicates the reported test accuracy likely understates what more regularization (dropout, more data, or fewer epochs) could achieve.

## Failure modes

- **Overfitting a large embedding matrix on limited labeled data**: as seen directly in the Experiment above — the embedding matrix alone has $320{,}000$ free parameters, and with only 20,000 training examples, the model starts memorizing training-specific word associations rather than generalizing (visible as rising validation loss after epoch 2).
- **Fixed `MAX_LEN` truncates long documents**: any content past the first (or last, depending on truncation direction) 200 tokens is invisible to the model regardless of how relevant it is to the label.
- **Vanishing/exploding gradients over very long sequences**: even with LSTM's gating, extremely long sequences (well beyond IMDB's typical review length) can still be difficult to train on, which is part of why Transformer-based architectures (attention over the whole sequence at once, rather than one step at a time) have largely superseded RNN/LSTM for long-document NLP in current practice — noted here as **planned**, not covered by this repository yet.
- **Task-specialized embeddings don't transfer**: because $E$ here is trained purely for sentiment classification, the resulting vectors are not a general-purpose semantic embedding space the way Word2Vec's are — reusing this model's embedding matrix for an unrelated task (e.g. topic classification) would likely perform worse than either retraining or using a general-purpose pretrained embedding.
- **Padding token side effects if unmasked**: without `mask_zero=True`, the LSTM processes padding positions as if they were real (zero-valued-embedding) tokens, which can dilute the final hidden state's summary of a short, heavily-padded sequence.

## Real-world usage

This Embedding→LSTM(/GRU)→Dense pattern (or its Transformer-based successors) underlies production text classification systems: sentiment analysis, spam/toxicity detection, intent classification for chatbots, and document tagging. The core lesson generalizes even where LSTMs have been replaced by Transformers: an embedding layer trained end-to-end with the rest of the model (or fine-tuned from a pretrained checkpoint) plus some order-aware sequence mechanism, jointly optimized for the actual downstream objective, consistently outperforms fixed, separately-trained representations fed into a simple classifier — the same "train it end-to-end for what you actually care about" principle that motivated this topic in the first place.

## Mental model

**Word2Vec learns a dictionary once, upfront, for general use; a task-trained `Embedding` layer learns a dictionary tailored to one specific job, revised by every mistake the whole model makes on that job.** Averaging that dictionary's entries (AvgWord2Vec) throws away sentence order; feeding them through an LSTM one word at a time, instead, lets the model actually read.

## Questions to think about

1. Why does backpropagating the classification loss all the way into the `Embedding` layer's weights (rather than freezing pretrained Word2Vec vectors) risk overfitting more, specifically when the labeled training set is small? Connect your answer to the parameter count of the embedding matrix versus the size of the training set in this topic's IMDB experiment.
2. AvgWord2Vec reached 95% accuracy on hockey-vs-space classification (`03-word-embeddings`) while this topic's LSTM model reached 86.9% on IMDB sentiment. Does this mean averaging beats recurrence? What differs between the two tasks that would explain the gap regardless of which representation is "better" in general?
3. If you froze the `Embedding` layer's weights at their Word2Vec-pretrained values (`trainable=False`) instead of training them from scratch jointly with the LSTM, what would you expect to change about the overfitting behavior seen in this topic's experiment, and why?
4. The from-scratch section argues that this topic needs no new from-scratch mechanic because it composes two already-derived ones. What would be a legitimate reason to still write new from-scratch code for a "composition" topic like this one — i.e., under what circumstance would composing two known mechanics itself introduce a new idea worth deriving?
5. `mask_zero=True` tells the LSTM to skip padded positions. Without it, how would a very short review (e.g. 20 real tokens padded to `MAX_LEN=200`) versus a long one (190 real tokens) be affected differently by the presence of padding, and what would you expect to see in the model's accuracy on short reviews specifically as a result?
