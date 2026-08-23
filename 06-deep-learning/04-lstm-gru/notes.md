# 04 – LSTM & GRU

## Problem

`03-rnn`'s "Failure modes" showed that vanilla RNNs cannot learn **long-range dependencies**: because $\frac{\partial L}{\partial W_{hh}}$ involves a product of many Jacobians, one per timestep, and each factor is $\le 1$ in magnitude (from $W_{hh}$ and $\tanh'$), that product shrinks exponentially as the number of timesteps grows. Gradients from distant timesteps vanish before they can meaningfully influence early weight updates, so a vanilla RNN's hidden state effectively "forgets" information from many steps ago — even though, for tasks like resolving a pronoun that refers to a noun 50 words earlier, that exact information is what's needed. The problem here: **how do you let information — and gradients — flow across many timesteps without shrinking to nothing?**

## Intuition

The vanilla RNN's hidden state is **fully overwritten** at every step — $h_t$ is a fresh $\tanh$-squashed combination of $h_{t-1}$ and $x_t$, with no way to preserve part of $h_{t-1}$ untouched. Imagine instead a notebook with a conveyor belt running alongside it: at each timestep, you can choose to (a) erase some of what's already written on the belt, (b) write some new information onto it, and (c) read some of what's currently on the belt to decide your next action — but whatever you don't erase stays on the belt, unchanged, carried forward automatically. That conveyor belt is the **cell state** $C_t$ of an LSTM, and the "choose how much to erase / write / read" decisions are made by three learned **gates** — small neural networks in their own right, each outputting a number between 0 and 1 per feature, acting as a "how much to let through" valve.

## Why simpler approaches fail

This section is answered by the link above: the "simpler approach" here is exactly the vanilla RNN of `03-rnn`, and it fails specifically because its hidden state is overwritten by a repeated squashing multiplication rather than preserved by addition — the mechanism that causes gradients to vanish over long sequences. LSTM and GRU exist specifically to fix that mechanism, not to solve a different problem.

## Mathematical foundation

### LSTM architecture (forget, input, output gates)

At each timestep $t$, an LSTM cell receives the current input $x_t$, the previous hidden state $h_{t-1}$, and the previous cell state $C_{t-1}$, and uses three sigmoid **gates** (each in the range $[0, 1]$) plus a candidate update:

**Forget gate** — decides what fraction of the old cell state to keep:
$$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$$

**Input gate** — decides how much of the new candidate information to write in:
$$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C [h_{t-1}, x_t] + b_C)$$

**Cell state update** — combines the retained old memory with the new candidate, gated by how much of each to use:
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**Output gate** — decides how much of the (squashed) cell state to expose as the hidden state:
$$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

Here $[h_{t-1}, x_t]$ denotes concatenation, $\sigma$ is the sigmoid function, and $\odot$ is element-wise multiplication. Intuitively: the forget gate lets the cell drop irrelevant old memory, the input gate lets it write new relevant information, and the output gate lets it decide what part of its memory is relevant to expose right now.

**Why this fixes the vanishing-gradient problem:** because $C_t$ is updated by **addition** ($f_t \odot C_{t-1} + i_t \odot \tilde C_t$) rather than by full replacement through a matrix multiplication and squashing non-linearity, the derivative $\partial C_t / \partial C_{t-1} \approx f_t$ — a per-feature multiplication by a learned gate value, not a repeated matrix-and-$\tanh$ transformation. As long as the forget gate keeps values close to 1 for information that should be remembered, gradients can flow backward through $C_t$ across many timesteps largely unattenuated, unlike the vanilla RNN's $\prod \frac{\partial h_k}{\partial h_{k-1}}$ product from `03-rnn`, which shrinks by construction.

### GRU architecture

The **Gated Recurrent Unit (GRU)** (Cho et al., 2014) is a simplification of the LSTM that merges the cell state and hidden state into a single state vector $h_t$, and uses only **two** gates instead of three:

**Update gate** — plays a role similar to the LSTM's combined forget+input gates, controlling how much of the past hidden state to carry forward vs. replace with a new candidate:
$$z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)$$

**Reset gate** — controls how much of the previous hidden state to use when computing the new candidate:
$$r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)$$

**Candidate activation and final update:**
$$\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h)$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

Because a GRU has fewer gates and no separate cell state, it has **fewer parameters** than an LSTM with the same hidden size, which makes it faster to train and less prone to overfitting on smaller datasets, while typically achieving comparable performance to LSTM on many tasks. Note the GRU's final update is also additive in the same spirit as the LSTM's cell-state update ($h_t$ is a convex combination of $h_{t-1}$ and $\tilde h_t$, not a full overwrite), which is why it shares the LSTM's resistance to vanishing gradients despite having a simpler gate structure.

## Algorithm

LSTMs and GRUs are trained with the same overall recipe as any RNN — **Backpropagation Through Time (BPTT)** — applying the chain rule through the unrolled sequence of cells:

1. Initialize all gate weight matrices ($W_f, W_i, W_C, W_o$ for LSTM, or $W_z, W_r, W_h$ for GRU) and biases.
2. **Forward pass:** for $t = 1, \dots, T$, apply the gate equations above in order (forget → input/candidate → cell-state update → output/hidden-state, for LSTM; update → reset/candidate → hidden-state, for GRU), reusing the same weights at every timestep.
3. Compute the loss from the relevant output(s).
4. **Backward pass (BPTT):** unroll the graph and apply the chain rule backward through every gate at every timestep. The key structural difference from a vanilla RNN's BPTT is the (approximately) additive path through $C_t$ (LSTM) or $h_t$ (GRU) described in "Mathematical foundation" — gradients propagate with far less attenuation than through a vanilla RNN's fully-multiplicative recurrence.
5. **Update:** apply an optimizer (typically Adam) to all gate weights and biases.
6. Repeat over mini-batches/epochs. In Keras, all of this is handled internally by `layers.LSTM(...)` / `layers.GRU(...)`; the user only specifies the number of units and stacks it into a model exactly like a `SimpleRNN`.

## From-scratch implementation

Implemented in `lstm-from-scratch-cell.ipynb`: one LSTM cell's forward step, in plain NumPy, applying all four gate equations to a toy input vector.

1. Initializes $W_f, W_i, W_C, W_o$ and biases, a toy input $x_t \in \mathbb{R}^3$, and zero-initialized $h_{t-1}, C_{t-1} \in \mathbb{R}^4$.
2. Computes the forget gate $f_t$, then the input gate $i_t$ and candidate $\tilde C_t$, then the cell-state update $C_t = f_t \odot C_{t-1} + i_t \odot \tilde C_t$, then the output gate $o_t$ and hidden state $h_t = o_t \odot \tanh(C_t)$ — one equation at a time, matching "Mathematical foundation" exactly, with intermediate values printed at each step.
3. Wraps the four equations into a reusable `lstm_cell_step` function, then takes a **second** step with a new toy input, using the first step's $h_t, C_t$ as the new "previous" state, and explicitly verifies that $C_{t+1}$ equals `f_t * C_t + i_t * C_tilde_t` — the additive property that "Mathematical foundation" argues is the reason LSTM gradients don't vanish the same way a vanilla RNN's do.
4. Cross-checks the entire hand-computed step against `tf.keras.layers.LSTMCell` loaded with the same weights (reassembled into Keras's `[i, f, C, o]` gate-concatenation layout), confirming both produce identical $h_t$ and $C_t$.

A single forward step is enough to see the mechanic — the point isn't to unroll a long sequence by hand (that's `03-rnn`'s from-scratch notebook), it's to see, numerically, that the cell state is updated by a weighted **sum** of old and new information rather than an overwrite.

## Practical implementation

**`lstm-gru-time-series.ipynb`** compares plain `LSTM`, plain `GRU`, and `Bidirectional(LSTM(...))` head-to-head on a windowed time-series forecasting task. Real stock price data requires an external download, which is outside this course's "standard built-ins only" constraint; instead, the notebook uses a numpy-generated, reproducible noisy sine wave (`np.random.seed` fixed for reproducibility) standing in for a smoothly-trending-but-noisy time series like a stock price. The task is: given a sliding window of the last $N$ time steps, predict the next value — exactly the framing used in real univariate time-series forecasting (stock prices, sensor readings, weather).

This maps directly back to "From-scratch implementation": Keras's `layers.LSTM(units)` runs the exact four gate equations the from-scratch notebook verified step-by-step, across the full input window, for every window in the dataset — the same arithmetic, vectorized and compiled, with weights learned by gradient descent instead of set by hand. `layers.GRU(units)` runs the two-gate equations from "Mathematical foundation" the same way.

**Bidirectional RNNs.** A standard (unidirectional) RNN/LSTM/GRU only has access to **past** context when producing the hidden state at time $t$ — it processes the sequence left-to-right. For tasks where the *entire* sequence is available at inference time (e.g. classifying a whole sentence, rather than generating text token-by-token), it is often beneficial to also look at **future** context. A **Bidirectional RNN** runs two separate recurrent layers over the same input — one processing the sequence forward ($x_1 \to x_T$), one processing it backward ($x_T \to x_1$) — and concatenates their hidden states at each timestep: $h_t = [\overrightarrow{h_t} \, ; \, \overleftarrow{h_t}]$. In Keras this is a simple wrapper, `layers.Bidirectional(layers.LSTM(units))`. It roughly doubles the number of parameters and compute compared to the unidirectional version, but often improves accuracy on sequence tasks where the full sequence is available, because each timestep's representation is informed by the whole sequence, not just what came before it — not appropriate for autoregressive generation where future tokens are unknown at generation time.

## Experiment

**Hypothesis (stated before running):** since GRU has fewer parameters than LSTM for the same hidden size (per "Mathematical foundation"), and Bidirectional(LSTM) has access to future context that plain LSTM/GRU do not, we'd expect: (a) LSTM and GRU to reach broadly comparable test MSE on this single-step-ahead windowed forecasting task, since it doesn't require resolving genuinely long-range dependencies that would favor LSTM's extra gating capacity, and (b) Bidirectional(LSTM) to have no inherent advantage here specifically *because* forecasting is a causal task — at inference time, future values are exactly what's being predicted and are not available, so the "sees the future" advantage that helps bidirectional models on classification/tagging does not transfer to this setup within the windowed-training regime the notebook uses.

**Setup:** `lstm-gru-time-series.ipynb` builds windowed training examples from the synthetic noisy sine wave, trains separate `LSTM`, `GRU`, and `Bidirectional(LSTM(...))` models with the same window size and training configuration, and reports test MSE for all three plus predicted-vs-actual plots.

**Result:** the notebook's reported test MSE values and predicted-vs-actual plots (see `lstm-gru-time-series.ipynb`) are cited here as the source of record rather than reproduced.

**Interpretation:** where LSTM and GRU land close together in test MSE, that's consistent with the hypothesis that this particular task doesn't stress long-range dependency resolution enough to separate them; where Bidirectional(LSTM) doesn't dominate the unidirectional models, that's consistent with the causal-forecasting argument above, since within each training window the bidirectional model can look only within that window, not truly into the future being predicted.

**Limitations:** one synthetic dataset, one window size, one architecture per gate type, no hyperparameter search — the results characterize this setup, not a general ranking of LSTM vs. GRU vs. Bidirectional across all time-series tasks.

## Failure modes

LSTM/GRU substantially mitigate, but do not eliminate, the vanishing-gradient problem: for **extremely** long sequences (hundreds to thousands of timesteps), even a forget gate close to 1 compounds multiplicatively enough to still attenuate gradients significantly, and the sequential, step-by-step nature of BPTT for any RNN variant (vanilla, LSTM, or GRU) means training cannot be parallelized across timesteps the way a Transformer's attention mechanism can (`05-attention-transformers`) — this is a computational, not just a gradient-flow, limitation. A second, separate limitation that gating does nothing to address: the cell state $C_t$ (and hidden state $h_t$) is a single vector of **fixed dimensionality**, regardless of how long the input sequence is — so however well the gates protect it from vanishing gradients, an entire sequence still has to be summarized into that one fixed-size vector, which becomes a bottleneck for very long sequences (`05-attention-transformers` addresses this directly). Additional practical failure modes:

- **Exploding gradients** can still occur (less commonly than in vanilla RNNs, but not impossible), typically mitigated with gradient clipping, same as for vanilla RNNs.
- **Overfitting on small datasets:** LSTM's extra gates give it more parameters than a GRU or vanilla RNN of the same hidden size, which can overfit faster on limited data — one reason GRU is often preferred when data is scarce.
- **Bidirectional models cannot be used for autoregressive generation**, since they require the full sequence (including "future" tokens relative to any given position) at inference time — using one where only past context will be available at inference time is a silent conceptual error, not just a minor accuracy loss.

## Real-world usage

- **Vanilla RNN:** short sequences, minimal long-range dependency, when compute/parameter budget is extremely tight.
- **LSTM:** long sequences with important long-range dependencies; more expressive gating (separate forget/input/output control) can help on complex tasks, at the cost of more parameters and compute. Historically dominant in speech recognition, machine translation (pre-Transformer), and time-series forecasting.
- **GRU:** similar benefits to LSTM for long-range dependencies, but with fewer parameters — a good default when data is limited or training speed matters, and often performs comparably to LSTM.
- **Bidirectional (wrapping LSTM or GRU):** whenever the **full sequence is available at inference time** (classification, tagging, encoding for translation) — common in named-entity recognition and sentiment/document classification pipelines.

Both LSTM and GRU have been largely superseded by attention-based architectures (`05-attention-transformers`) for large-scale NLP, precisely because of the sequential-training limitation noted in "Failure modes" — but they remain in active use for smaller-scale sequence tasks, embedded/resource-constrained settings, and univariate time-series forecasting, where a Transformer's overhead isn't justified.

## Mental model

An LSTM is "a vanilla RNN with a protected memory lane": instead of overwriting its entire summary at every step, it keeps a cell state that is only ever modified by addition — gated erasing, gated writing, gated reading — so information (and gradient) can travel down that lane across many timesteps largely intact. A GRU is the same idea with the cell state and hidden state merged and fewer gates, trading a little expressiveness for fewer parameters and faster training.

## Questions to think about

1. `notes.md`'s cell-state-update equation is $C_t = f_t \odot C_{t-1} + i_t \odot \tilde C_t$. If $f_t$ were fixed at exactly $\mathbf{1}$ and $i_t$ at exactly $\mathbf{0}$ for every timestep, what would happen to $C_t$ across a long sequence, and why is that both the LSTM's greatest strength and a potential failure mode if the gates get "stuck" there?
2. The from-scratch notebook's second step verifies $C_{t+1} = f_t \odot C_t + i_t \odot \tilde C_t$ numerically. Why does this additive structure make $\partial C_{t+1}/\partial C_t \approx f_t$ (a single elementwise multiplication) rather than the repeated matrix-and-$\tanh$ product that appears in a vanilla RNN's $\partial h_k/\partial h_{k-1}$?
3. GRU merges the cell state and hidden state into one vector and drops the output gate entirely. What information can an LSTM's separate cell state $C_t$ and hidden state $h_t$ represent that a GRU's single $h_t$ cannot, and when might that matter?
4. Why is a Bidirectional LSTM a poor choice for a language-generation task (predicting the next word one token at a time) even though it might improve accuracy on a sentiment-classification task using the same underlying LSTM cell?
5. `lstm-gru-time-series.ipynb`'s hypothesis predicted LSTM and GRU would land close together on this task specifically because it doesn't stress long-range dependencies. What property of a task (independent of sequence length) would you look for to predict that LSTM's extra gating capacity should give it a real edge over GRU?
