# 02 – Pretraining Objective: A Tiny GPT-Style Decoder

## Problem

[`06-deep-learning/05-attention-transformers`](../../06-deep-learning/05-attention-transformers/notes.md)
built self-attention and a full Transformer **encoder** block, and used it for
*classification* — the encoder reads an entire movie review at once, pools the output,
and predicts one label. Nothing in that topic ever generates text. GPT-style large
language models do something structurally different: given some text so far, predict the
single next token, append it, and repeat — an *autoregressive* process that produces
arbitrarily long text one token at a time. This topic asks the question
`06-deep-learning/05-attention-transformers` left open: **what has to change about
self-attention, and about the training objective itself, to go from "encode a whole
sequence for classification" to "generate a sequence one token at a time"?**

## Intuition

`06-deep-learning/05-attention-transformers`'s encoder is allowed to be a little bit of
a cheat: when it decides how much attention token 3 ("sat") should pay to token 5
("mat"), it's fine, because the whole sentence "the cat sat on the mat" is already sitting
in memory — there's no future to worry about leaking from, since the entire input is
given upfront and the task is just "classify this fixed sentence."

Generation cannot use that shortcut. When a GPT-style model is producing the fourth word
of a sentence it is *writing*, words five, six, and seven don't exist yet — there's
nothing there to look at, and even during training (where the full target sentence is
available so training can be parallelized) the model must not be allowed to peek ahead at
the answer, or it would trivially "predict" the next word by copying it, and learn nothing
useful for the one situation that matters: generating text where the future genuinely
isn't known yet. The fix is mechanical and almost embarrassingly simple: keep exactly the
same attention computation from `06-deep-learning/05-attention-transformers`, but for
every position, forcibly zero out attention to every position after it. A decoder-only
Transformer is that idea — a stack of self-attention blocks *with the future blacked out*
— trained with one objective: given everything so far, predict what comes next.

## Why simpler approaches fail

The "simpler approach" here is literally the previous topic's architecture: take
`06-deep-learning/05-attention-transformers`'s encoder, with its *unmasked* self-attention
(every position freely attends to every other position, past and future alike), and try
to use it for generation instead of classification.

This fails for a structural reason, not a performance reason. Suppose the training
objective is "predict token $t$ from the rest of the sequence." If the encoder's
self-attention lets position $t$'s query attend to position $t$'s own key/value (or any
position $\geq t$), the model can satisfy that objective **without learning anything**:
attention weight $w_{t,t} \to 1$ lets the model simply copy token $t$'s value vector
straight through to its own output and "predict" it perfectly, for every training example,
immediately, with loss $\to 0$ and zero generalization. This isn't a subtle overfitting
risk that shows up after some training — it is available to the optimizer from step one
as the single cheapest way to minimize the loss, and gradient descent has no way to prefer
the harder, useful solution over it. Worse, this failure is invisible at training time
(loss looks great) and only shows up at inference time, when the model must generate token
$t$ without token $t$ already sitting in the input — at that point the model has never had
to actually predict from the past alone, so its outputs are essentially untrained noise.

Bidirectional attention (BERT-style encoders, exactly `06-deep-learning/05`'s
architecture) sidesteps this by changing the *objective* instead, not the architecture —
masked language modeling randomly hides ~15% of input tokens and asks the model to
recover them from the (still bidirectional) rest of the sequence. That is a legitimate,
widely-used pretraining objective, but it produces a model built to *fill in blanks in a
fixed-length sequence it can already see*, not one built to *extend a sequence
open-endedly from left to right* — the latter is what generation requires, and it is what
this topic builds. Two different problems, two different objectives, two different
masking strategies — deliberately not "one better than the other."

## Mathematical foundation

### Causal (masked) self-attention

Start from exactly `06-deep-learning/05-attention-transformers`'s scaled dot-product
attention:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}}\right)V,
\qquad Q = XW^Q,\; K = XW^K,\; V = XW^V
$$

For a sequence of length $n$, $QK^T \in \mathbb{R}^{n \times n}$ is a full matrix of
pairwise query-key similarity scores, where entry $(i, j)$ says how much query position
$i$ should attend to key position $j$. Unmasked, softmax normalizes each *row* $i$ over
**all** $j \in \{1, \ldots, n\}$ — including $j > i$, future positions relative to $i$.

**The causal mask** adds a mask matrix $M \in \mathbb{R}^{n \times n}$ before the softmax:

$$
\text{CausalAttention}(Q, K, V) = \text{softmax}\!\left(\frac{QK^T}{\sqrt{d_k}} + M\right)V,
\qquad
M_{i,j} = \begin{cases} 0 & j \le i \\ -\infty & j > i \end{cases}
$$

**Why $-\infty$, and why this is sufficient to prevent leakage — derived, not asserted:**
softmax over row $i$ computes
$\text{softmax}(z)_j = e^{z_j} / \sum_k e^{z_k}$. As $z_j \to -\infty$,
$e^{z_j} \to 0$, so the post-softmax weight $\alpha_{i,j} \to 0$ *exactly* (in floating
point, exactly 0, not merely small) for every $j > i$. The attention output at position
$i$ is $\sum_j \alpha_{i,j} v_j$ — since $\alpha_{i,j} = 0$ for all future $j$, **the
output at position $i$ is, by construction, a weighted sum over only $v_1, \ldots, v_i$,
containing zero contribution from any future value vector.** This isn't an approximate or
learned property that training might violate — it is enforced structurally, before any
gradient is computed, the same way `06-deep-learning/05-attention-transformers`'s
"Masked self-attention and encoder-decoder attention" paragraph describes it. This is
exactly what makes next-token prediction *well-defined* as a training objective: since
position $i$'s output provably cannot depend on positions $> i$, using that output to
predict token $i+1$ is training the model to predict from the past alone — the same
condition inference time will actually have. Without the mask, as shown in "Why simpler
approaches fail," the objective is satisfiable by leakage and therefore *not* actually
testing (or teaching) next-token prediction at all.

Practically, $M$ is built once as a fixed lower-triangular matrix of $1$s (`torch.tril`),
and any position where that matrix is $0$ gets its score overwritten with $-\infty$ before
the softmax — see "From-scratch implementation" below and the notebook's `CausalSelfAttention`.

### The autoregressive language-modeling objective

Let $x_1, x_2, \ldots, x_T$ be a sequence of tokens. The **chain rule of probability**
lets any joint distribution over the full sequence be factored, with no independence
assumption at all, as a product of one-token-at-a-time conditionals:

$$
P(x_1, \ldots, x_T) = \prod_{t=1}^{T} P(x_t \mid x_{<t})
$$

where $x_{<t} = (x_1, \ldots, x_{t-1})$. This factorization is exact for *any*
distribution — it is not a modeling assumption yet. The modeling assumption is deciding to
approximate each conditional $P(x_t \mid x_{<t})$ with one shared neural network
$P_\theta(x_t \mid x_{<t})$ — a decoder-only causal Transformer, applied at every position
$t$ in parallel during training (exactly what the causal mask makes valid, per above).

**Maximum likelihood.** Training maximizes the probability the model assigns to the real
training data, i.e. maximizes $\prod_t P_\theta(x_t \mid x_{<t})$ over a training corpus.
Products of many small probabilities underflow numerically and are hard to differentiate
through directly, so the equivalent (monotonic) log form is used instead — maximizing a
sum of log-probabilities:

$$
\max_\theta \sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
\quad\Longleftrightarrow\quad
\min_\theta \underbrace{-\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})}_{\text{cross-entropy loss}}
$$

This *is* cross-entropy loss on next-token prediction, applied at every position: the
model outputs a categorical distribution over the vocabulary at each position (a softmax
over logits), and the loss at position $t$ is $-\log P_\theta(x_t \mid x_{<t})$ — the
negative log-probability the model assigned to the token that actually came next.
Averaging this over all positions and all training sequences gives the scalar loss that
gradient descent minimizes; in PyTorch this is exactly `F.cross_entropy(logits, targets)`
where `targets[i]` is the correct next token for position `i`, and it is only a valid,
leakage-free training signal because of the causal mask above.

**Perplexity**, a standard way to report language-model loss, is simply
$\text{PPL} = e^{\text{cross-entropy loss}}$ — the effective branching factor the model
is choosing among at each step. A model outputting a uniform distribution over a
vocabulary of size $|V|$ has cross-entropy $\log|V|$ and perplexity exactly $|V|$; any
perplexity below $|V|$ means the model has learned *something* nonrandom about which
tokens tend to follow which — this is the exact quantity the notebook measures before and
after training.

## Algorithm

A **decoder-only Transformer block** (this topic's core unit, per "Mathematical
foundation"):

1. Compute **causal multi-head self-attention** over the input sequence — identical
   projections and scaled-dot-product formula to `06-deep-learning/05-attention-transformers`,
   with the $M$ mask added before softmax.
2. Residual connection + LayerNorm: $z = \text{LayerNorm}(x + \text{CausalAttention}(x))$.
3. Position-wise feed-forward network (two linear layers with a nonlinearity between).
4. Second residual + LayerNorm: $\text{output} = \text{LayerNorm}(z + \text{FeedForward}(z))$.

A full decoder-only model stacks $N$ of these blocks on top of a token embedding + learned
positional embedding (exactly `06-deep-learning/05-attention-transformers`'s
`TokenAndPositionEmbedding` pattern), then a final linear layer projecting back to
vocabulary-sized logits at every position.

**Training loop:**

1. Slice the training corpus into `(context, target)` windows, where `target` is `context`
   shifted one token to the right (position $i$ of `target` is the correct next token after
   position $i$ of `context`).
2. Forward pass: get per-position logits over the vocabulary for the whole window at once
   (parallel, thanks to the mask).
3. Compute cross-entropy loss between predicted logits and the shifted targets.
4. Backpropagate, take an optimizer step.
5. Repeat for a fixed number of steps.

**Generation loop (inference):**

1. Start from a prompt (a sequence of tokens).
2. Forward pass, take the logits at the **last** position only.
3. Convert to a probability distribution via softmax (optionally temperature-scaled).
4. Sample the next token from that distribution (`torch.multinomial`) — not necessarily
   the single most likely one.
5. Append the sampled token to the sequence, repeat from step 2.

## From-scratch implementation

`001_tiny_gpt_pretraining.ipynb` builds and actually trains a genuinely tiny decoder-only
GPT, real gradient steps, real loss curve, real sampled generation:

- **Tokenizer choice: character-level, not `13-llms-from-scratch/01-tokenizer-from-scratch`'s
  BPE.** BPE's benefit (shrinking sequence length by merging frequent pairs) only pays off
  at a corpus scale large enough for merge statistics to be meaningful; at this notebook's
  ~1.5KB corpus it would mostly rediscover whole words as tokens after a handful of merges
  — real, but adding no new insight over Task 1. Character-level tokenization keeps the
  notebook's complexity budget on the actually-new material (causal attention, the
  pretraining objective) rather than tokenizer plumbing already covered.
- A **causal self-attention module** (PyTorch, `CausalSelfAttention`) that is
  `06-deep-learning/05-attention-transformers`'s `scaled_dot_product_attention(Q,K,V)` —
  same formula — extended with exactly one new ingredient: a registered lower-triangular
  buffer mask, applied via `scores.masked_fill(mask == 0, -inf)` before the softmax, per
  "Mathematical foundation." A standalone sanity check confirms the mask mechanically: for
  query position 0, only column 0 of the attention-weight row is nonzero; for query
  position 3, columns 0–3 are nonzero and columns 4–5 (future) are exactly 0.
- A **`TinyGPT`** model: token embedding + learned positional embedding → 2 stacked
  `Block`s (causal self-attention + feed-forward, each with residual + LayerNorm, per
  "Algorithm") → final LayerNorm → linear head to vocabulary logits. Embedding dim 32,
  2 attention heads, context window 48 characters — **21,093 parameters total** (a real
  small GPT-2 is ~124 million — roughly 5,900x larger).
- **Actual training**: 300 real AdamW steps, batch size 32, `torch.set_num_threads(1)`,
  on a small hand-written English paragraph (1,486 characters, 37-character vocabulary,
  deliberately natural prose rather than a repeating template so the model cannot trivially
  memorize a fixed pattern). Wall-clock: **1.90 seconds**. Training loss: **3.68 → 1.74**
  (real printed curve every 50 steps, plus a plotted `training_loss_curve.png`
  monotonically decreasing).
- **Actual generation**: `model.generate()` samples autoregressively from the trained
  model at two temperatures (1.0 and 0.7), starting from the prompt `"The old "`.

## Practical implementation

There is no separate "practical/library" step here beyond the from-scratch build itself —
unlike `06-deep-learning/05-attention-transformers`, where Part 2 swapped in
`layers.MultiHeadAttention` on a real dataset, doing the equivalent for a real
GPT-scale decoder (`transformers`' `GPT2LMHeadModel`, or training one from
`nanoGPT`-style code on a real corpus) is explicitly out of toy-scale reach — real
pretraining runs for days-to-months on many GPUs/TPUs over hundreds of billions of tokens,
which `AGENTS.md`'s no-heavy-training constraint rules out running for real in this
environment. The honest mapping instead: the `TinyGPT` class above **is** architecturally
what `GPT2LMHeadModel` is — token embedding, learned positional embedding, a stack of
causal-self-attention-plus-feed-forward blocks, a linear head to vocabulary logits,
trained with cross-entropy on next-token prediction — just with `n_layers=2`,
`embed_dim=32` instead of GPT-2's `n_layers=12`, `embed_dim=768` (small), scaling further
to GPT-3-class models' `n_layers=96`, `embed_dim=12288`. Using a real pretrained
GPT-2-family checkpoint via `transformers` for actual downstream text generation (as
opposed to training one from scratch) is a natural next step, but is deliberately not
this topic's job — this topic is about the pretraining *mechanism*, not about consuming
someone else's already-pretrained weights.

## Experiment

**Hypothesis (stated before running):** if causal self-attention plus cross-entropy on
next-token prediction is actually a valid, well-defined training signal (per
"Mathematical foundation"), then over real gradient steps on this tiny model (a) training
loss should measurably decrease, (b) held-out validation perplexity should land below the
uniform-random baseline of $|V|$, and (c) the model's predicted probability for a common
in-corpus continuation should get measurably more confident (less random) than an
untrained model's near-uniform predictions.

**Setup:** `001_tiny_gpt_pretraining.ipynb`, `TinyGPT` (21,093 params) trained for 300
AdamW steps (`lr=3e-3`, batch size 32, `block_size=48`) on the 1,486-character corpus
described above, `torch.set_num_threads(1)`. Before and after training, probed the
model's top-5 predicted next characters after the common in-corpus prefix `"the "`.

**Result (actual, from the executed notebook):**

- Training loss: **3.6758 (step 0) → 1.7356 (step 299)**, monotonically decreasing (see
  the printed per-50-step values and `training_loss_curve.png` in the notebook).
- Wall-clock training time: **1.90 seconds** for 300 steps (well under the "few minutes"
  ceiling, single CPU thread).
- Validation perplexity: **13.91**, vs. a uniform-random-character baseline of
  **37** (`vocab_size`) — the model is meaningfully better than chance at predicting
  held-out characters, though far from a low real-model perplexity (single digits or below,
  typical of well-trained language models on natural text).
- Top next-character prediction after `"the "`: **before training**, near-uniform
  (top pick `' '` at probability 0.063, barely above the 0.027 uniform baseline);
  **after training**, sharply more confident (top pick `'l'` at probability 0.179 — nearly
  3x the untrained top probability).

**Interpretation:** all three hypothesized effects held, with real measured numbers, not
just a qualitative "it seemed to work." The mechanism — causal masking making next-token
prediction leakage-free, cross-entropy correctly driving the loss down via real gradient
steps — is doing exactly what "Mathematical foundation" derives it should. This is the
narrow, honest claim this experiment supports.

**Limitations, stated directly:** one 1,486-character corpus, one tiny architecture, no
hyperparameter search, no comparison against an alternative objective (e.g. masked
language modeling) on the same data. Most importantly: **none of the above numbers say
anything about generation quality** — see "Failure modes" for why the sampled text is
still nonsensical despite loss provably decreasing and perplexity provably beating chance.

## Failure modes

**The generated text is not coherent English, and this is expected, not a bug.** Real
samples from the trained model (temperature 1.0):

```
The old houted vier aget, sin p gy toueainid
EChe ld sterly thid tsthull st n ye aiarmey
```

The text contains real, learned English-shaped structure — space-separated
word-length token runs, `"the"` appearing correctly, plausible-looking letter clusters
(`"sthull"`, `"aiarmey"`) — but it is not real words and not a grammatical sentence.
This is the honestly expected outcome at this scale, for three concrete, measurable
reasons, not a mysterious failure:

1. **Capacity.** 21,093 parameters is roughly 5,900x smaller than GPT-2 small
   (~124M parameters) and many orders of magnitude smaller than a modern LLM
   (billions to hundreds of billions of parameters). A model this small has nowhere near
   enough representational capacity to encode English grammar, word identity, and
   long-range coherence simultaneously — it can only capture the crudest local statistics
   (which letters commonly follow which, roughly how long a "word" between spaces is).
2. **Data.** 1,486 characters is a rounding error next to a real pretraining corpus
   (hundreds of billions to trillions of tokens). There simply is not enough data here
   for the model to see most English words more than a handful of times, let alone learn
   robust word-level or sentence-level structure.
3. **Training steps.** 300 gradient steps is a toy budget; real pretraining runs for
   hundreds of thousands to millions of steps. Training loss was still visibly decreasing
   at step 299 (per "Experiment") — this model is measurably *undertrained*, not converged.

These three factors compound multiplicatively, not additively, in real LLM
**scaling laws**: Kaplan et al. (2020) and Hoffmann et al.'s "Chinchilla" paper (2022)
empirically characterize how pretraining loss depends jointly on model size, dataset
size, and compute, and derive compute-optimal tradeoffs between them (mentioned by name
here, not derived — deriving them is out of this topic's scope and requires training runs
far beyond what `AGENTS.md`'s no-heavy-training constraint allows). The qualitative
takeaway that *does* transfer down to this toy scale: pushing any one of
capacity/data/steps up without the others produces diminishing returns, which is exactly
why real LLM pretraining scales all three together rather than maximizing just one.

**Repetition and instability** are the other classic small-model/undertrained symptom
(not fully visible in this notebook's short 200-character samples, but the standard
failure mode as generation length grows): a model with weak long-range structure tends to
fall into short repeated loops once it drifts away from anything resembling its training
distribution, because it has no real mechanism for tracking "what have I already said" —
only local character/word statistics.

## Real-world usage

- **Every GPT-family model** (GPT-2/3/4, Llama, Mistral, Claude's own underlying
  architecture family, and effectively every modern general-purpose LLM) is a decoder-only
  Transformer trained with exactly this objective — cross-entropy on next-token
  prediction under a causal mask — at a scale of billions of parameters, trillions of
  training tokens, and enormous compute budgets. Nothing about the *mechanism* changes at
  that scale; what changes is capacity, data, and steps (per "Failure modes"), plus
  substantial systems engineering (distributed training, mixed precision, KV-caching at
  inference time) that is out of this topic's scope.
- **Perplexity**, computed here as a toy sanity check, is a standard benchmark metric
  reported for real language models (e.g. on held-out WikiText or C4 splits) — lower is
  better, and it is directly comparable across models trained on the same evaluation set.
- This is the pretraining stage specifically — the resulting raw model is a text
  *completer*, not an instruction-follower or chat assistant; see
  `13-llms-from-scratch/03-instruction-tuning` for what has to change to get from "predicts
  plausible continuations" to "follows instructions."

## Mental model

An encoder (`06-deep-learning/05-attention-transformers`) is allowed to see the whole
sentence at once because its job is to *understand* a fixed input. A decoder built for
generation cannot use that shortcut, because its job is to *produce* text the future
tokens of which don't exist yet — so causal self-attention blacks out every future
position with $-\infty$ before the softmax, making "predict the next token from
everything so far" the only thing the model is mathematically capable of doing. Stack
enough of those blocks, train with cross-entropy on real next-token targets, and you get
the exact mechanism every modern LLM's pretraining runs on — just at a scale (parameters
× data × steps) this toy deliberately does not attempt to reach, per real scaling laws.

## Questions to think about

1. "Why simpler approaches fail" argues that without the causal mask, attention weight
   $w_{t,t} \to 1$ would let the model minimize loss by copying, not learning. Concretely,
   what would the *training* loss curve look like if you removed the mask from this
   notebook's `CausalSelfAttention` and re-ran training — and what would the *generated*
   text look like afterward? Why would those two observations disagree so sharply?
2. The mask uses $-\infty$ (in practice, a very large negative number) rather than simply
   setting the post-softmax weight to 0 directly. Why does it matter that the masking
   happens *before* the softmax rather than after? (Hint: what would happen to the
   remaining, un-masked weights' normalization if you zeroed weights out after softmax
   instead?)
3. This experiment measured perplexity (13.91) well below the uniform baseline (37) at the
   same time the generated text was still nonsensical. Explain concretely why both of
   those can be true simultaneously — what does perplexity measure that generation
   coherence doesn't, and vice versa?
4. "Failure modes" names three compounding factors — capacity, data, training steps.
   If you could increase exactly one of the three for this notebook's model while holding
   the other two fixed, which would you expect to move perplexity the most, and why?
   What does that predict about which lever matters most as models get closer to
   real-world scale?
5. `13-llms-from-scratch/01-tokenizer-from-scratch` built a real BPE tokenizer, but this
   notebook used character-level tokenization instead. If you swapped in the BPE
   tokenizer from Task 1 for this exact experiment (same corpus, same model size, same
   number of steps), what would you expect to happen to `block_size` (context length in
   tokens vs. characters covering the same amount of text) and to the difficulty of the
   next-token prediction task itself? Would you expect loss/perplexity numbers to be
   directly comparable to this notebook's character-level ones?
