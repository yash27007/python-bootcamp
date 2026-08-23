# 05 – Attention Mechanism & Transformers

## Problem

`04-lstm-gru`'s "Failure modes" ended on two limitations that LSTM/GRU gating does not fix, because they are not gradient-flow problems at all:

1. **Sequential processing.** Backpropagation Through Time processes one timestep at a time — step $t$ cannot start until step $t-1$'s hidden state exists. No amount of gating removes that dependency, so training cannot be parallelized across the sequence dimension, which becomes the dominant cost on large datasets and long sequences.
2. **A fixed-size bottleneck.** Even an LSTM's cell state $C_t$ is a single vector of fixed dimensionality. However good the gating is at protecting it from vanishing gradients, an entire sequence — 10 tokens or 10,000 — still has to be summarized into that one vector by the time an encoder-decoder model needs to use it.

Classic sequence-to-sequence (seq2seq) models built from RNNs/LSTMs make bottleneck #2 explicit: an **encoder** reads the whole input and compresses it into a single fixed-size **context vector** (typically its final hidden state $h_T$), and a **decoder** is initialized from that one vector and generates the output autoregressively. This decouples input length from output length — useful for translation, where source and target sentences rarely match in length — but it means the decoder must reconstruct an entire output sequence from one summary vector produced once, before decoding even starts.

The problem this topic solves: **how do you let a model relate any two positions in a sequence directly, and compute those relationships in parallel, instead of forcing everything through one fixed-size vector and one sequential pass?**

## Intuition

Imagine translating a 40-word sentence by reading it once, closing your eyes, and then having to produce the entire translation from memory of a single mental summary — no re-reading, no glancing back at specific words as you go. That is what a vanilla encoder-decoder does: the decoder only ever sees $h_T$, the encoder's one final hidden state, however long the input was.

**Attention** fixes this by letting the decoder "glance back" at the input while producing each output token — not at one fixed summary, but at a weighted combination of *all* the encoder's hidden states, with the weights recomputed fresh at every decoding step to reflect what's relevant *right now*. Translating "the cat sat" into French, when producing "chat" the decoder's attention weights should concentrate on the encoder state for "cat"; when producing "assis," they should shift toward "sat."

**Self-attention** takes the same weighted-glancing idea and turns it inward: instead of one sequence (the decoder) attending to another (the encoder), every position in a *single* sequence attends to every other position within that same sequence, deciding how much each other token should contribute to its own updated representation. The **Transformer** builds an entire encoder/decoder purely out of this self-attention operation (plus simple feed-forward layers) — no recurrence at all, so every position's computation can run in parallel.

## Why simpler approaches fail

The "simpler approach" here is the vanilla encoder-decoder (seq2seq) architecture introduced above, built from the RNN/LSTM cells of `03-rnn` / `04-lstm-gru`, and it fails for reasons that compound on top of that topic's own failure modes:

- **The fixed-context bottleneck.** The entire input sequence — regardless of length — must be compressed into one fixed-size vector $h_T$. For short sequences this is a mild approximation; as sequence length grows, information from early tokens gets diluted or lost by the time the encoder reaches the end, and the decoder has no way to "look back" at specific relevant parts of the input — it only ever has the one summary vector, produced once, used throughout decoding. Empirically, translation quality degrades sharply as sentence length increases with plain encoder-decoder models.
- **The sequential-training bottleneck.** Even setting the fixed-vector problem aside, an RNN/LSTM encoder must still process the input one token at a time — step $t$ depends on step $t-1$'s hidden state — so training cannot be parallelized across the sequence dimension, unlike a computation where every position could in principle be processed simultaneously.

**Attention** (Bahdanau et al., 2014) was the first fix, and it addresses only the first bottleneck: instead of a single fixed context vector, the decoder, at each generation step, looks back at **all** of the encoder's hidden states $h_1, \dots, h_T$ (not just the final one) and computes a weighted combination of them. An alignment score $e_{t,i}$ between the decoder's current state and every encoder hidden state $h_i$ is normalized into weights via softmax:

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j} \exp(e_{t,j})}$$

and the decoder's context vector for that step is the weighted sum $c_t = \sum_i \alpha_{t,i} h_i$, letting the decoder attend more strongly to whichever encoder positions are most relevant to the current output token. This solved the long-sequence degradation problem — but the encoder and decoder underneath are still RNNs/LSTMs, so the second bottleneck (sequential processing) remains untouched.

The **Transformer** (Vaswani et al., "Attention Is All You Need", 2017) removes both bottlenecks at once by discarding recurrence entirely and building the encoder and decoder purely out of attention and feed-forward layers:

- **No more fixed vector:** self-attention lets every position attend directly to every other position, with a *direct, constant-length path* between any two positions no matter how far apart they are — unlike an RNN, where information between distant tokens must flow through every intermediate timestep and is subject to the vanishing-gradient dynamics `03-rnn`/`04-lstm-gru` describe.
- **No more sequential bottleneck:** attention over a sequence is a set of matrix multiplications that can be computed for all positions simultaneously, so training is fully parallelizable on GPUs/TPUs — the direct fix for `04-lstm-gru`'s "Failure modes" observation that BPTT cannot be parallelized across timesteps.

This combination — parallel training and strong long-range modeling — is why Transformers displaced RNN-based seq2seq as the dominant NLP architecture (and, later, spread to vision and multi-modal models).

## Mathematical foundation

### Self-attention: Q, K, V and scaled dot-product attention

Each input token's embedding (a row of the input matrix $X \in \mathbb{R}^{n \times d_{\text{model}}}$, for a sequence of $n$ tokens) is projected into three vectors via learned weight matrices:

$$Q = X W^Q, \qquad K = X W^K, \qquad V = X W^V$$

- **Query ($Q$):** "what am I looking for?" — the current token's request for information.
- **Key ($K$):** "what do I contain?" — each token's advertisement of its own content, compared against queries.
- **Value ($V$):** "what do I actually offer?" — the content that gets aggregated once a token is deemed relevant.

The output is a weighted sum of value vectors, with weights from comparing queries against keys via **scaled dot-product attention**:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- $QK^T$ computes a similarity score between every query and every key — a full $n \times n$ pairwise comparison across the sequence.
- Dividing by $\sqrt{d_k}$ (the dimensionality of the key vectors) prevents the dot products from growing too large in magnitude as $d_k$ increases; unscaled, large-magnitude scores push softmax into regions with vanishingly small gradients, which would make the attention weights hard to learn.
- $\text{softmax}(\cdot)$ turns each row of scores into a probability distribution (attention weights) over all positions.
- The final output for each token is the value vectors weighted by those probabilities: each token's new representation is a context-aware blend of the whole sequence, with more weight on the tokens most relevant to it.

This is the *same* weighted-sum-over-relevance idea as Bahdanau attention above ($c_t = \sum_i \alpha_{t,i} h_i$), generalized: instead of one decoder state attending to a separate encoder sequence, every position in one sequence attends to every position in that same sequence, and the alignment scores are computed by a learned dot-product ($QK^T/\sqrt{d_k}$) rather than a small feed-forward network.

### Multi-head attention

Rather than computing a single attention distribution, **multi-head attention** runs $h$ independent scaled dot-product attention computations ("heads") in parallel, each with its own learned $W_i^Q, W_i^K, W_i^V$ projecting into a smaller dimensionality $d_k = d_{\text{model}}/h$, then concatenates all heads' outputs and projects back to $d_{\text{model}}$ with one more learned matrix $W^O$:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h)\, W^O, \qquad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Each head can specialize in a different kind of relationship (e.g. one head tracking syntactic dependencies, another semantic similarity), giving the model a richer representation than a single attention computation could produce alone.

### Positional encoding

Self-attention treats the input as an unordered **set** — permuting the input rows just permutes $\text{Attention}(Q,K,V)$'s output rows identically, with no other change — unlike an RNN, which processes tokens in order and therefore has an implicit sense of position. Since word order matters for meaning, Transformers must inject positional information explicitly.

The original Transformer does this with a fixed **sinusoidal positional encoding**, added directly to each token's input embedding:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

where $pos$ is the token's position in the sequence and $i$ indexes the embedding dimension. Sine/cosine functions at different frequencies per dimension let the model infer relative positions (since $PE_{pos+k}$ can be expressed as a linear function of $PE_{pos}$), and it generalizes to sequence lengths not seen during training. A simpler alternative — a trainable **learned positional embedding** indexed by position and added to the token embedding — lets the model learn whatever positional representation works best for a given task; this is the variant used in this topic's practical notebook, following the standard Keras text-classification-with-Transformer pattern.

### Layer normalization

**Layer Normalization** (Ba et al., 2016) normalizes activations *across the feature dimension* for each individual example (as opposed to Batch Normalization, which normalizes across the batch dimension for each feature) — well suited to sequence models where sequence length varies and batch statistics are less stable:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu$ and $\sigma^2$ are the mean and variance across the features of a single example, and $\gamma, \beta$ are learned scale/shift parameters. In a Transformer block, layer normalization wraps both the attention and feed-forward sub-layers, usually with a residual/skip connection (`x + Sublayer(LayerNorm(x))` or `LayerNorm(x + Sublayer(x))` depending on the variant) — this stabilizes training of deep stacks of attention layers, directly mitigating the vanishing/exploding-gradient concerns from `01-ann` in a very deep model.

### Masked self-attention and encoder-decoder attention (decoder-side)

A full Transformer decoder (needed for sequence-generation tasks like translation) differs from an encoder block in two ways:

- **Masked multi-head self-attention:** when generating the output autoregressively (left to right, one token at a time), the decoder must not "see" future tokens it hasn't generated yet. This is enforced by adding a mask to the attention scores that sets every position *after* the current one to $-\infty$ before the softmax, so their attention weight becomes exactly 0 — preserving the autoregressive property while still allowing the whole target sequence to be fed in at once during training (parallel training, with the mask preventing information leakage from future positions).
- **Encoder-decoder (cross) attention:** a second multi-head attention sub-layer where the **queries** come from the decoder's own masked-self-attention representations, but the **keys and values** come from the encoder's output. This is the direct conceptual descendant of Bahdanau attention above — each decoder position attends over the entire encoded input sequence to decide what source information is relevant to the token being generated right now.

## Algorithm

A single **Transformer encoder block** (the piece used in this topic's classification notebook) combines self-attention, multi-head attention, layer norm, and residual connections into one self-contained unit:

1. Compute **multi-head self-attention** over the input sequence.
2. Add a residual connection and apply layer normalization: $z = \text{LayerNorm}(x + \text{MultiHeadAttention}(x))$.
3. Pass $z$ through a **position-wise feed-forward network** — two `Dense` layers with a non-linearity between them, applied independently and identically to every position.
4. Add a second residual connection and layer normalization:

$$\text{output} = \text{LayerNorm}\big(z + \text{FeedForward}(z)\big)$$

A full **Transformer encoder** stacks $N$ of these blocks on top of an input token embedding + positional encoding/embedding. For a full encoder-decoder Transformer (translation, etc.), the decoder stack (masked self-attention → encoder-decoder cross-attention → feed-forward, each with its own residual + layer norm) sits on top, attending to the encoder's final output. For a classification task, only the encoder is needed: the encoder's output sequence is pooled (e.g. global average pooling over the sequence dimension) and passed through a final `Dense` classification head — exactly the architecture built in this topic's practical notebook.

## From-scratch implementation

**Part 1 of `attention-and-transformer.ipynb`** implements scaled dot-product self-attention from scratch in plain NumPy, and meets the bar for this section directly:

1. A toy sequence of 6 tokens (`['the', 'cat', 'sat', 'on', 'the', 'mat']`) is represented as random $8$-dimensional embeddings $X \in \mathbb{R}^{6 \times 8}$.
2. Fixed-seed random projection matrices $W^Q, W^K, W^V \in \mathbb{R}^{8 \times 4}$ compute $Q = XW^Q$, $K = XW^K$, $V = XW^V$ — exactly the projections in "Mathematical foundation."
3. A `scaled_dot_product_attention(Q, K, V)` function computes `scores = Q @ K.T / sqrt(d_k)`, a numerically-stable `softmax`, and `output = weights @ V` — the full $\text{softmax}(QK^T/\sqrt{d_k})V$ formula, verified to produce attention weight rows that each sum to 1.
4. The resulting $6 \times 6$ attention-weight matrix is **visualized as a heatmap** (`plt.imshow`, with per-cell weight values annotated), with tokens labeled on both axes — directly showing, for a concrete toy sequence, how much each token (as query, row) attends to every other token (as key, column).

This is single-head attention only; it does not re-implement multi-head attention from scratch. That is a deliberate scope choice rather than an omission: multi-head attention (per "Mathematical foundation") is mechanically $h$ independent copies of exactly this same `scaled_dot_product_attention` computation, run on different learned projections of $Q, K, V$ and concatenated — the single-head NumPy demo already shows the computation that gets replicated $h$ times, so re-implementing it $h$ times over would add repetition, not insight. The natural place multi-head attention *is* exercised end-to-end is Part 2, via `layers.MultiHeadAttention` — this is the from-scratch → practical mapping "Practical implementation" makes explicit below.

## Practical implementation

**Part 2 of `attention-and-transformer.ipynb`** builds a Transformer-encoder text classifier on `tf.keras.datasets.imdb` (25,000 movie reviews, binary sentiment), directly mapped back to Part 1 and "Mathematical foundation":

- **`TransformerEncoderBlock`** (custom Keras layer): `layers.MultiHeadAttention` computes the exact $\text{Concat}(\text{head}_1,\dots,\text{head}_h)W^O$ formula from "Mathematical foundation" — the multi-head generalization of the single-head `scaled_dot_product_attention` function from Part 1 — followed by a residual connection, `layers.LayerNormalization`, a position-wise feed-forward (`Dense(ff_dim, relu)` → `Dense(embed_dim)`), a second residual connection, and a second `LayerNormalization` — implementing the encoder-block algorithm step by step.
- **`TokenAndPositionEmbedding`**: a token `Embedding` plus a **learned** positional `Embedding` (indexed by position, added to the token embedding) — the learned-positional-embedding alternative described in "Mathematical foundation."
- The full model: `Input` → `TokenAndPositionEmbedding` → one `TransformerEncoderBlock` → `GlobalAveragePooling1D` → `Dropout` → `Dense(20, relu)` → `Dropout` → `Dense(1, sigmoid)`, trained with `binary_crossentropy` / Adam and early stopping on validation loss.

The from-scratch NumPy attention matrix in Part 1 shows exactly what `MultiHeadAttention` computes internally, per head, inside Part 2: for every token, a weighted blend of the whole sequence's value vectors, with weights driven by query-key similarity — the only differences in the practical version are (a) $h$ heads instead of 1, (b) learned rather than fixed-seed-random projection weights, and (c) the whole computation vectorized/compiled and trained by gradient descent instead of hand-set.

## Experiment

**Hypothesis (stated before running):** a single Transformer encoder block — with learned positional embeddings, multi-head self-attention, and a small feed-forward sublayer — should be able to reach competitive sentiment-classification accuracy on IMDB despite having no recurrence at all, because self-attention gives every token a direct path to every other token in the review (per "Why simpler approaches fail"), which should be enough signal for a document-level classification task even without an RNN's explicit sequential processing.

**Setup:** `attention-and-transformer.ipynb` Part 2 trains the encoder-only classifier described in "Practical implementation" on the IMDB dataset (vocabulary capped at 10,000 words, sequences padded/truncated to 200 tokens, 32-dimensional embeddings, 2 attention heads), with early stopping on validation loss, and evaluates on the held-out test set.

**Result:** test accuracy **0.8748**, test loss **0.3092** (see `attention-and-transformer.ipynb`, final cells, for training/validation curves).

**Interpretation:** this result is consistent with the hypothesis — a single encoder block with no recurrence reaches strong (>87%) accuracy on this document-classification task, supporting the claim that self-attention's direct token-to-token paths are sufficient signal here, without needing the sequential processing that `03-rnn`/`04-lstm-gru` rely on.

**Limitations:** one dataset (IMDB), one architecture configuration (a single encoder block, 2 heads, no hyperparameter search), no direct head-to-head comparison against the `03-rnn`/`04-lstm-gru` models on the identical IMDB split in this notebook — the comparison is qualitative (architecture and mechanism), not a controlled accuracy comparison.

## Failure modes

- **Quadratic cost in sequence length.** The $QK^T$ step in scaled dot-product attention computes a full $n \times n$ score matrix, so both compute and memory scale as $O(n^2)$ in sequence length $n$. Doubling the sequence length quadruples the attention computation — this becomes the dominant cost for very long sequences (long documents, high-resolution images-as-patches, genomic sequences), and is the direct tradeoff for the "direct constant-length path between any two positions" advantage described in "Why simpler approaches fail": that direct path is exactly what makes every pair of positions require its own score, unlike an RNN's $O(n)$ sequential cost.
- **No inherent notion of order.** Because self-attention is permutation-*equivariant* by construction, positional encoding is not optional — an incorrectly configured or omitted positional signal silently degrades a Transformer to treating the input as a bag of tokens.
- **Data-hungry.** Transformers lack the built-in inductive biases an RNN has (recurrence itself is a strong prior that order matters) or a CNN has (translation invariance via convolution); they generally need more training data or pretraining to reach strong performance from scratch.
- **Fixed context window.** Despite removing the *fixed-vector* bottleneck of seq2seq, standard self-attention still requires a fixed maximum sequence length (`MAX_LEN` in the practical notebook); sequences longer than that must be truncated or handled with specialized long-context architectures (sparse/linear attention variants), which are outside this topic's scope.

## Real-world usage

- **Encoder-only Transformers** (e.g. BERT-style models): text classification, sentiment analysis, named-entity recognition — the architecture built in this topic's practical notebook.
- **Decoder-only Transformers** (e.g. GPT-style models): autoregressive text generation, relying on the masked self-attention from "Mathematical foundation."
- **Encoder-decoder Transformers**: machine translation, summarization — the direct architectural descendant of the seq2seq models in "Why simpler approaches fail," with cross-attention replacing Bahdanau attention.
- Beyond NLP: image patches treated as a "sequence" (Vision Transformers), protein sequences, and multi-modal models combining text/image/audio — anywhere data can be tokenized into a sequence, self-attention's direct pairwise relationships and parallel training make it a strong default, subject to the quadratic-cost failure mode above for very long sequences.
- LSTM/GRU (`04-lstm-gru`) remain preferable when sequences are extremely long (making $O(n^2)$ attention prohibitive) or compute/parameter budgets are tight and full parallelization isn't the bottleneck.

## Mental model

A Transformer replaces "compress everything into one fixed-size vector, then process it one step at a time" (seq2seq's two bottlenecks) with "let every position directly ask every other position a question (query vs. key) and pull in a weighted answer (value) — all at once, in parallel." Multi-head attention runs several such question-asking processes side by side so different heads can specialize; positional encoding puts back the order information that a purely set-based operation would otherwise lose; layer norm and residuals keep a deep stack of these blocks trainable. The price for this direct, parallel, all-pairs connectivity is that its cost grows quadratically with sequence length — the fixed vector's forgetting problem is traded for a compute-and-memory scaling problem.

## Questions to think about

1. `04-lstm-gru`'s "Failure modes" names two separate limitations LSTM/GRU do not fix: sequential processing and a fixed-size context vector. Which of these does Bahdanau-style attention (added on top of an RNN encoder-decoder) fix, and which does it leave untouched? Which does the Transformer fix, and how?
2. The scaled dot-product attention formula divides by $\sqrt{d_k}$. If this scaling were removed and $d_k$ were large, what would happen to the softmax output, and why would that make the attention weights harder to learn via gradient descent?
3. The from-scratch notebook implements a single attention head. Multi-head attention runs $h$ of these in parallel on different learned projections of the same $Q, K, V$ inputs. What could a second head learn that the first head's projection could not represent, given they share the same input embeddings but different $W^Q, W^K, W^V$?
4. Self-attention is permutation-equivariant: permuting the input rows permutes the output rows identically, with no other change. Concretely, what would go wrong in the IMDB sentiment classifier if the positional embedding were removed entirely — what information would the model lose access to, and what kinds of movie reviews would this affect most?
5. Attention's cost scales as $O(n^2)$ in sequence length. For a task with sequences of length 100,000 (e.g. a full-length book), what would the practical implications be of using unmodified self-attention, and what would you look for in an alternative architecture?
