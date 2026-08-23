# 05 – Attention Mechanism & Transformers

| Topic | Status |
|-------|--------|
| Encoder–Decoder (Seq2Seq) Architecture | ✅ Complete |
| Problems with Encoder-Decoder | ✅ Complete |
| Attention Mechanism | ✅ Complete |
| What & Why Transformers | ✅ Complete |
| Self-Attention Layer | ✅ Complete |
| Multi-Head Attention | ✅ Complete |
| Positional Encoding | ✅ Complete |
| Layer Normalisation | ✅ Complete |
| Complete Encoder Transformer | ✅ Complete |
| Decoder (Masked Multi-Head Attention, Encoder-Decoder Attention) | ✅ Complete |

## Encoder–Decoder (Seq2Seq) Architecture

Many sequence tasks map one sequence to another sequence of a *different* length — machine translation (English sentence → French sentence), summarization (article → summary), or question answering. The classic solution before Transformers was the **encoder-decoder (sequence-to-sequence)** architecture built from RNNs/LSTMs:

- The **encoder** (an RNN/LSTM, as in topics `03-rnn` / `04-lstm-gru`) reads the entire input sequence one token at a time and compresses everything into a single fixed-size **context vector** — typically its final hidden state $h_T$.
- The **decoder** (another RNN/LSTM) is initialized with that context vector and generates the output sequence one token at a time, feeding each generated token back in as the next input (autoregressive generation).

This architecture elegantly decouples input length from output length — a natural fit for translation, where the source and target sentences rarely have the same number of tokens.

## Problems with Encoder-Decoder

The classic seq2seq design has a critical bottleneck: **the entire input sequence must be compressed into one fixed-size vector**, regardless of how long the input is. For short sequences this works reasonably well, but as sequence length grows:

- Information from early tokens gets diluted or lost by the time the encoder reaches the end of a long sequence (the same vanishing-gradient-driven "forgetting" problem from `03-rnn`/`04-lstm-gru`, now compounded by the fact that *everything* must fit through a single vector).
- The decoder has no way to "look back" at specific parts of the input relevant to what it is generating *right now* — it only has the one summary vector, produced once, used throughout decoding.
- Empirically, translation quality degrades sharply as sentence length increases with plain encoder-decoder models.

This single-fixed-vector bottleneck directly motivated the **attention mechanism**.

## Attention Mechanism

**Attention** (Bahdanau et al., 2014) removes the fixed-context bottleneck by letting the decoder, at each generation step, look back at **all** of the encoder's hidden states $h_1, \dots, h_T$ (not just the final one) and compute a **weighted combination** of them — with weights that depend on how relevant each encoder position is to what the decoder is producing right now.

Concretely, at decoder step $t$, an alignment score $e_{t,i}$ is computed between the decoder's current state and every encoder hidden state $h_i$, then normalized into weights via softmax:

$$\alpha_{t,i} = \frac{\exp(e_{t,i})}{\sum_{j} \exp(e_{t,j})}$$

The decoder's context vector for that step is the weighted sum $c_t = \sum_i \alpha_{t,i} h_i$, letting the decoder "attend" more strongly to the encoder positions most relevant to the current output token (e.g. when translating, attending to the source word being translated right now). This solved the long-sequence degradation problem and set the stage for the next leap: what if attention were used *not just* to connect encoder and decoder, but as the primary computation *inside* the encoder and decoder themselves?

## What & Why Transformers

The **Transformer** (Vaswani et al., "Attention Is All You Need", 2017) took that idea to its conclusion: discard the recurrence entirely and build the encoder and decoder purely out of **attention** and feed-forward layers. This has major practical advantages over RNN-based seq2seq:

- **Parallelization:** an RNN must process tokens sequentially (step $t$ depends on step $t-1$), which is slow to train. A Transformer's attention computation looks at all positions simultaneously, so it can be fully parallelized on GPUs/TPUs, dramatically speeding up training on large datasets.
- **Long-range dependencies:** attention connects any two positions in a sequence with a direct, constant-length path (no matter how far apart they are), unlike an RNN where information between distant tokens must flow through every intermediate timestep and is subject to vanishing gradients (per `03-rnn`).

This combination of parallel training and strong long-range modeling is why Transformers became the dominant architecture for NLP (and, later, vision and multi-modal models).

## Self-Attention Layer

**Self-attention** lets every position in a sequence attend to every other position *within the same sequence* (as opposed to the encoder-decoder attention above, which connects two different sequences). Each input token's embedding is projected into three vectors via learned weight matrices:

$$Q = X W^Q, \qquad K = X W^K, \qquad V = X W^V$$

- **Query ($Q$):** "what am I looking for?" — represents the current token's request for information.
- **Key ($K$):** "what do I contain?" — represents each token's advertisement of its own content, compared against queries.
- **Value ($V$):** "what do I actually offer?" — the content that gets aggregated once a token is deemed relevant.

The output is a weighted sum of value vectors, where the weights come from comparing queries against keys via the **scaled dot-product attention** formula:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

$QK^T$ computes a similarity score between every query and every key (a full pairwise comparison across the sequence); dividing by $\sqrt{d_k}$ (the dimensionality of the key vectors) prevents the dot products from growing too large in magnitude as $d_k$ increases, which would otherwise push the softmax into regions with vanishingly small gradients. The softmax turns the scores into a probability distribution (attention weights) over all positions, and the final output for each token is the value vectors weighted by those probabilities — i.e. each token's new representation is a context-aware blend of the whole sequence, with more weight on the tokens most relevant to it.

The companion notebook implements this formula from scratch in NumPy on a tiny toy sequence and visualizes the resulting attention-weight matrix as a heatmap, to build direct intuition for what "attention weights" mean before using Keras's built-in layer.

## Multi-Head Attention

Rather than computing a single attention distribution, **multi-head attention** runs $h$ independent scaled dot-product attention computations ("heads") in parallel, each with its own learned $W^Q, W^K, W^V$ projections into a smaller dimensionality $d_k = d_{\text{model}}/h$, then concatenates all heads' outputs and projects back to $d_{\text{model}}$ with one more learned matrix $W^O$:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) W^O, \qquad \text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Each head can specialize in a different kind of relationship (e.g. one head might learn to track syntactic dependencies, another semantic similarity), giving the model a richer representation than a single attention computation could. Keras provides this directly as `layers.MultiHeadAttention`, used in this topic's Transformer-encoder notebook.

## Positional Encoding

Unlike an RNN, which processes tokens in order and therefore has an implicit sense of position, self-attention treats the input as an unordered **set** — $\text{Attention}(Q,K,V)$ gives the same result regardless of token order (permuting the input rows just permutes the output rows identically). Since word order matters for meaning, Transformers must inject positional information explicitly.

The original Transformer does this with a fixed **sinusoidal positional encoding**, added directly to each token's input embedding:

$$PE_{(pos, 2i)} = \sin\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right), \qquad PE_{(pos, 2i+1)} = \cos\!\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)$$

where $pos$ is the token's position in the sequence and $i$ indexes the embedding dimension. Using sine/cosine functions at different frequencies per dimension lets the model infer relative positions (since $PE_{pos+k}$ can be expressed as a linear function of $PE_{pos}$), and it generalizes to sequence lengths not seen during training. An alternative, simpler approach — used in the companion notebook, following the standard Keras text-classification-with-Transformer pattern — is a **learned positional embedding**: a trainable `Embedding` layer indexed by position, added to the token embedding, letting the model learn whatever positional representation works best for the task.

## Layer Normalisation

**Layer Normalization** (Ba et al., 2016) normalizes the activations *across the feature dimension* for each individual example (as opposed to Batch Normalization, which normalizes across the batch dimension for each feature) — making it well suited to sequence models where sequence length varies and batch statistics are less stable:

$$\text{LayerNorm}(x) = \gamma \odot \frac{x - \mu}{\sqrt{\sigma^2 + \epsilon}} + \beta$$

where $\mu$ and $\sigma^2$ are the mean and variance computed across the features of a single example, and $\gamma, \beta$ are learned scale/shift parameters. In a Transformer block, layer normalization is applied around both the attention and feed-forward sub-layers (commonly with a residual/skip connection: `x + Sublayer(LayerNorm(x))` or `LayerNorm(x + Sublayer(x))` depending on the exact variant), which stabilizes training of deep stacks of attention layers — directly mitigating the vanishing/exploding gradient concerns from `01-ann` in a very deep model.

## Complete Encoder Transformer

A single **Transformer encoder block** combines everything above into a self-contained unit:

1. **Multi-head self-attention** over the input sequence (with a residual connection and layer normalization).
2. A **position-wise feed-forward network** — two `Dense` layers with a non-linearity between them, applied independently and identically to every position (with another residual connection and layer normalization).

$$\text{output} = \text{LayerNorm}\big(x + \text{FeedForward}(\text{LayerNorm}(x + \text{MultiHeadAttention}(x)))\big)$$

A full **Transformer encoder** stacks $N$ of these blocks on top of an input embedding + positional encoding. For a classification task (as in this topic's notebook), the encoder's output sequence is typically pooled (e.g. global average pooling over the sequence dimension) and passed through a final `Dense` classification head. This is exactly the architecture built in `attention-and-transformer.ipynb`'s second half: `Embedding` + positional embedding → one Transformer encoder block (`MultiHeadAttention` + `LayerNormalization` + feed-forward `Dense` layers) → pooling → `Dense` sigmoid output, trained on IMDB sentiment classification.

## Decoder (Masked Multi-Head Attention, Encoder-Decoder Attention)

The full Transformer (used for sequence-to-sequence tasks like translation) also has a **decoder** stack, which differs from the encoder block in two ways:

- **Masked multi-head self-attention:** when generating the output sequence autoregressively (one token at a time, left to right), the decoder must not be allowed to "see" future tokens it hasn't generated yet. This is enforced by adding a mask to the attention scores that sets all positions *after* the current one to $-\infty$ before the softmax, so their attention weight becomes exactly 0. This preserves the autoregressive property while still allowing parallel training (the whole target sequence is fed in at once during training, with the mask preventing information leakage from future positions).
- **Encoder-decoder (cross) attention:** a second multi-head attention sub-layer where the **queries** come from the decoder's own (masked self-attention) representations, but the **keys and values** come from the encoder's output. This is the direct conceptual descendant of the original Bahdanau attention mechanism described above — it lets each decoder position attend over the entire encoded input sequence to decide what source information is relevant to the token currently being generated.

A full decoder block is therefore: masked self-attention → encoder-decoder cross-attention → feed-forward, each wrapped with residual connections and layer normalization, mirroring the encoder block's structure. This course's notebook focuses on the encoder-only classification use case (the more common pattern for tasks like sentiment analysis), since a decoder is only needed for sequence-generation tasks — but understanding masked self-attention and cross-attention conceptually completes the picture of how the full encoder-decoder Transformer (e.g. for translation) operates.
