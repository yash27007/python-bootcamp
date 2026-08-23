# 04 – LSTM & GRU

Detailed notes (forget/input/output gate equations, GRU update/reset gates, additive cell-state argument for gradient flow, bidirectional RNNs): [notes.md](notes.md)

From-scratch: a manual NumPy LSTM cell working through all four gate equations on a toy input, two steps, cross-checked against `tf.keras.layers.LSTMCell` — [lstm-from-scratch-cell.ipynb](lstm-from-scratch-cell.ipynb)

| Topic | Status |
|-------|--------|
| Why LSTM? | ✅ Complete |
| LSTM Architecture (Forget, Input, Output Gates) | ✅ Complete |
| LSTM Training Process | ✅ Complete |
| GRU Architecture | ✅ Complete |
| Bidirectional RNN | ✅ Complete |
| Windowed Time-Series Forecasting (LSTM/GRU/BiLSTM) | ✅ Complete |
