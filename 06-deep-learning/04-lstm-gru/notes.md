# 04 – LSTM & GRU

| Topic | Status |
|-------|--------|
| Why LSTM? | ✅ Complete |
| LSTM Architecture (Forget, Input, Output Gates) | ✅ Complete |
| LSTM Training Process | ✅ Complete |
| GRU Architecture | ✅ Complete |
| Bidirectional RNN | ✅ Complete |
| Windowed Time-Series Forecasting (LSTM/GRU/BiLSTM) | ✅ Complete |

## Why LSTM?

Topic `03-rnn` showed that vanilla RNNs suffer from **vanishing gradients** across long sequences: the hidden state $h_t$ is repeatedly overwritten by a $\tanh$-squashed combination of the previous state and current input, so information from many steps ago gets diluted and gradients from distant timesteps shrink toward zero during BPTT. This makes it hard for a vanilla RNN to learn dependencies spanning more than a handful of timesteps.

The **Long Short-Term Memory (LSTM)** network (Hochreiter & Schmidhuber, 1997) solves this by adding a separate **cell state** $C_t$ that acts as a conveyor belt of information running through the sequence, modified only by carefully controlled, learned **gates** — rather than being fully overwritten at every step like the vanilla RNN's hidden state. Because information can flow through the cell state largely unchanged when the gates allow it, gradients can also flow backward through many timesteps largely unattenuated, letting the network learn long-range dependencies that vanilla RNNs cannot.

## LSTM Architecture (Forget, Input, Output Gates)

At each timestep $t$, an LSTM cell receives the current input $x_t$, the previous hidden state $h_{t-1}$, and the previous cell state $C_{t-1}$, and uses three sigmoid **gates** (each in the range $[0, 1]$, acting like a learned "how much to let through" valve) plus a candidate update:

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

Here $[h_{t-1}, x_t]$ denotes concatenation, $\sigma$ is the sigmoid function, and $\odot$ is element-wise multiplication. Intuitively: the forget gate lets the cell drop irrelevant old memory, the input gate lets it write new relevant information, and the output gate lets it decide what part of its memory is relevant to expose right now. Because $C_t$ is updated by addition ($f_t \odot C_{t-1} + i_t \odot \tilde C_t$) rather than by full replacement, gradients can flow back through $C_t$ across many timesteps without vanishing as quickly as in a vanilla RNN.

## LSTM Training Process

LSTMs are trained with the same overall recipe as any RNN: **Backpropagation Through Time (BPTT)**, applying the chain rule through the unrolled sequence of cells, then updating all gate weight matrices ($W_f, W_i, W_C, W_o$ and their biases) with an optimizer such as Adam. The key structural difference from a vanilla RNN's BPTT is that gradients have an (approximately) additive path through the cell state $C_t$ — the derivative $\partial C_t / \partial C_{t-1} \approx f_t$ (rather than a repeated matrix multiplication through $\tanh$ and a weight matrix), so as long as the forget gate keeps values close to 1 for information that should be remembered, gradients can propagate over long sequences with far less attenuation than in a vanilla RNN. In Keras, all of this gradient computation is handled internally by `layers.LSTM(...)`; the user only specifies the number of units and stacks it into a `Sequential`/functional model exactly like a `SimpleRNN`.

## GRU Architecture

The **Gated Recurrent Unit (GRU)** (Cho et al., 2014) is a simplification of the LSTM that merges the cell state and hidden state into a single state vector $h_t$, and uses only **two** gates instead of three:

**Update gate** — plays a role similar to the LSTM's combined forget+input gates, controlling how much of the past hidden state to carry forward vs. replace with a new candidate:
$$z_t = \sigma(W_z [h_{t-1}, x_t] + b_z)$$

**Reset gate** — controls how much of the previous hidden state to use when computing the new candidate:
$$r_t = \sigma(W_r [h_{t-1}, x_t] + b_r)$$

**Candidate activation and final update:**
$$\tilde{h}_t = \tanh(W_h [r_t \odot h_{t-1}, x_t] + b_h)$$
$$h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t$$

Because a GRU has fewer gates and no separate cell state, it has **fewer parameters** than an LSTM with the same hidden size, which makes it faster to train and less prone to overfitting on smaller datasets, while typically achieving comparable performance to LSTM on many tasks.

## Bidirectional RNNs

A standard (unidirectional) RNN/LSTM/GRU only has access to **past** context when producing the hidden state at time $t$ — it processes the sequence left-to-right. For tasks where the *entire* sequence is available at inference time (e.g. classifying a whole sentence, rather than generating text token-by-token), it is often beneficial to also look at **future** context.

A **Bidirectional RNN** runs two separate recurrent layers over the same input — one processing the sequence forward ($x_1 \to x_T$), one processing it backward ($x_T \to x_1$) — and concatenates (or sums) their hidden states at each timestep:

$$h_t = [\overrightarrow{h_t} \, ; \, \overleftarrow{h_t}]$$

In Keras this is a simple wrapper: `layers.Bidirectional(layers.LSTM(units))`. It roughly doubles the number of parameters and compute compared to the unidirectional version, but often improves accuracy on sequence classification tasks because each timestep's representation is informed by the whole sequence, not just what came before it.

## When to Prefer LSTM / GRU / Bidirectional

- **Vanilla RNN:** short sequences, minimal long-range dependency, when compute/parameter budget is extremely tight.
- **LSTM:** long sequences with important long-range dependencies; more expressive gating (separate forget/input/output control) can help on complex tasks, at the cost of more parameters and compute.
- **GRU:** similar benefits to LSTM for long-range dependencies, but with fewer parameters — a good default when data is limited or training speed matters, and often performs comparably to LSTM.
- **Bidirectional (wrapping LSTM or GRU):** whenever the **full sequence is available at inference time** (classification, tagging, encoding for translation) — not appropriate for autoregressive generation where future tokens are unknown at generation time.

`lstm-gru-time-series.ipynb` compares plain `LSTM`, plain `GRU`, and `Bidirectional(LSTM(...))` head-to-head on the same synthetic time-series task so these tradeoffs can be observed directly in test MSE and predicted-vs-actual plots.

## Windowed Time-Series Forecasting (LSTM/GRU/BiLSTM)

Real stock price data requires an external download, which is outside this course's "standard built-ins only" constraint. Instead, `lstm-gru-time-series.ipynb` uses the same **windowed sequence-prediction framing** that a stock-price predictor would use — a numpy-generated, reproducible noisy sine wave (`np.random.seed` fixed for reproducibility) standing in for a smoothly-trending-but-noisy time series like a stock price. The task is: given a sliding window of the last $N$ time steps, predict the next value. This is exactly the framing used in real univariate time-series forecasting (stock prices, sensor readings, weather), and the notebook trains and compares `LSTM`, `GRU`, and `Bidirectional(LSTM(...))` models on it, reporting test MSE for all three and plotting predicted-vs-actual curves.
