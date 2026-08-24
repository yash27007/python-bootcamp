# 01 – Tensors and Autograd

Detailed notes (why manual backprop doesn't scale, reverse-mode AD derivation, the bridge to the
from-scratch MLP): [notes.md](notes.md)

Practical + cross-check: `torch.Tensor`/`.backward()`, a tiny hand-checked example, and the from-scratch
MLP's gradients vs. PyTorch autograd's gradients on identical weights — [01-tensors-and-autograd.ipynb](01-tensors-and-autograd.ipynb)

## What you'll learn

Why `06-deep-learning/01-ann/notes.md`'s hand-derived backprop (a $\delta^{[l]}$ recursion, coded by hand
in `ann-from-scratch-xor.ipynb`) doesn't scale past a couple of layers, and what closes that gap: reverse-
mode automatic differentiation — a computation graph recorded during the forward pass, walked backward via
the chain rule, generically, for any graph shape, so no one has to re-derive backprop by hand for every new
architecture. Then the real thing: PyTorch `torch.Tensor` with `requires_grad=True` and `.backward()`.

| Topic | Status |
|-------|--------|
| Why manual backprop doesn't scale | ✅ Complete |
| Computation graphs & the chain rule | ✅ Complete |
| Reverse-mode vs. forward-mode AD | ✅ Complete |
| `torch.Tensor`, `requires_grad`, `.backward()`, `.grad` | ✅ Complete |
| Autograd vs. from-scratch manual gradient cross-check | ✅ Complete |

## Why it matters

The from-scratch MLP's `backward()` function is exact, but it exists because a human sat down and derived
the chain-rule recursion for one specific 2-layer graph. Every new layer type or connection pattern would
need the same derivation repeated by hand — a bottleneck that makes deep architectures with dozens or
hundreds of layers practically impossible to build by hand. Autograd removes that bottleneck by recording
the graph automatically and applying the same chain-rule mechanics generically, to any graph.

## Prerequisites

- `06-deep-learning/01-ann/notes.md` and `ann-from-scratch-xor.ipynb` — this topic bridges directly from
  the hand-derived backprop there; read those first, they are cited (not repeated) throughout.
- Basic NumPy familiarity (`01-python-foundation`).

## What you'll build

- A small worked autograd example (`x = torch.tensor(..., requires_grad=True)`, a short operation chain,
  `.backward()`, inspect `.grad`) to build intuition before touching the MLP.
- The exact 2-layer XOR MLP from `ann-from-scratch-xor.ipynb`, rebuilt with PyTorch `Tensor`s instead of
  plain NumPy arrays and no hand-written `backward()` — one `.backward()` call replaces the manual
  `backward()` function entirely.
- A cross-check, actually run: the from-scratch NumPy gradients and PyTorch's autograd gradients, computed
  on identical weights and data, compared with `np.allclose()` — confirmed to match exactly.

## Where it appears in real systems

Every production deep learning framework (PyTorch, TensorFlow, JAX) is built on reverse-mode AD — it is the
mechanism that makes training architectures with millions of parameters (CNNs, Transformers) tractable at
all, since it computes the full gradient in one backward pass regardless of parameter count.

## What's next

`02-nn-module-and-training-loop` — the from-scratch MLP's other scaling problem (manually tracking every
weight tensor) and PyTorch's `nn.Module` + standard training loop that automates it.
