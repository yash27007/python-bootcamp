# 02 – nn.Module and the Training Loop

Detailed notes (why manual weight-tensor tracking doesn't scale, the 5-step training loop mapped to the
from-scratch loop): [notes.md](notes.md)

Practical + experiment: an `nn.Module` trained on XOR and `make_moons`, and `optimizer.step()` vs. the
from-scratch manual gradient-descent update, compared on identical data — [02-nn-module-and-training-loop.ipynb](02-nn-module-and-training-loop.ipynb)

## What you'll learn

Why the from-scratch MLP's approach — four loose NumPy tensors (`W1, b1, W2, b2`), threaded by hand through
`forward()`, `backward()`, and an explicit per-tensor update loop — doesn't scale to real architectures with
dozens or hundreds of parameters. PyTorch's `nn.Module` groups parameters into a container that tracks them
automatically, and the standard training-loop pattern (forward → loss → `loss.backward()` →
`optimizer.step()` → `zero_grad()`) is shown to be the automated version of the from-scratch loop's five
steps, line by line.

| Topic | Status |
|-------|--------|
| Why manual weight-tensor tracking doesn't scale | ✅ Complete |
| `nn.Module` as a parameter container + `forward()` | ✅ Complete |
| The 5-step PyTorch training loop, mapped to the from-scratch loop | ✅ Complete |
| Training an `nn.Module` on XOR and `make_moons` | ✅ Complete |
| `optimizer.step()` vs. manual gradient-descent update comparison | ✅ Complete |

## Why it matters

`01-tensors-and-autograd` removed the need to hand-derive gradients; this topic removes the second manual
burden the from-scratch notebook carried — explicitly naming and updating every weight tensor. Together they
turn the from-scratch loop's five hand-written steps into five fixed, architecture-independent method calls
that work unchanged whether the model has 4 parameters or 4 million.

## Prerequisites

- `01-tensors-and-autograd` — this topic's `.backward()` step relies directly on the autograd mechanics
  established there.
- `06-deep-learning/01-ann/ann-from-scratch-xor.ipynb` — the training loop this topic bridges from,
  line by line.

## What you'll build

- A small `nn.Module` (2–3 `nn.Linear` layers) structurally identical to the from-scratch MLP, trained on
  the same XOR dataset with `nn.BCELoss()` and `torch.optim.SGD` — actually trained, real loss curve.
- The same architecture trained on `sklearn.datasets.make_moons`, showing the pattern scale past 4 toy
  points.
- A hypothesis-first experiment comparing `torch.optim.SGD`'s update against the from-scratch notebook's
  manual `W -= lr * dW` update on the same problem — actually run, confirming similar convergence.

## Where it appears in real systems

Every non-trivial PyTorch model (CNN, RNN, Transformer) is a class subclassing `nn.Module`; the same 5-step
training loop runs, unchanged in structure, inside every high-level training utility (PyTorch Lightning,
Hugging Face `Trainer`) and every from-scratch PyTorch training script in production use.

## What's next

Later PyTorch topics build specific architectures (CNNs, RNNs, Transformers) on top of this same
`nn.Module` + training-loop foundation — the pattern established here does not change, only the layers
composed inside `forward()` do.
