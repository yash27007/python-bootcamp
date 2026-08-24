# 01 – Tensors and Autograd

## Problem

`06-deep-learning/01-ann/notes.md` derived backpropagation for a 2-layer MLP by hand: a
$\delta^{[2]} = \hat y - y$ output error term, a $\delta^{[1]} = (\delta^{[2]}(W^{[2]})^T)\odot\tanh'(z^{[1]})$
recursion into the hidden layer, and explicit weight-gradient formulas
$\partial L/\partial W^{[l]} = (a^{[l-1]})^T\delta^{[l]}$ — and `ann-from-scratch-xor.ipynb` coded every one of
those formulas by hand in NumPy, specialized to exactly that one 2-layer architecture.

That derivation was tractable because the network had two layers and one hidden non-linearity. **What
happens when the architecture is a 50-layer CNN, or a Transformer with attention, residual connections,
and layer norm scattered across dozens of sublayers?** NumPy arrays carry no memory of how they were
computed — `a1 = np.tanh(z1)` gives you `a1`'s values, not the fact that `a1` came from a `tanh` applied to
`z1`, which came from a matrix multiply of `X` and `W1`. Every time the architecture changes, someone has to
re-derive the chain-rule recursion for the new graph of operations and re-code it by hand, correctly, for
every new layer type, activation, and connection pattern. This does not scale, and it is exactly the gap
that automatic differentiation ("autograd") closes: track the sequence of operations that produced a value,
so the chain rule can be applied to *any* graph of operations mechanically, without a human re-deriving it
per architecture.

## Intuition

Think of computing a value like $L = \left(\sin(x_1 \cdot x_2) + x_2\right)^2$ by hand, on paper, with
intermediate steps labeled:

```
a = x1 * x2
b = sin(a)
c = b + x2
L = c ** 2
```

Each line is a small, simple operation whose local derivative you already know from calculus 101:
$\partial a/\partial x_1 = x_2$, $\partial b/\partial a = \cos(a)$, $\partial c/\partial b = 1$,
$\partial c/\partial x_2 = 1$, $\partial L/\partial c = 2c$. To get $\partial L/\partial x_1$, the chain rule
says: multiply the local derivatives along the path from $x_1$ to $L$:
$\partial L/\partial x_1 = \partial L/\partial c \cdot \partial c/\partial b \cdot \partial b/\partial a \cdot \partial a/\partial x_1$.

That is *all* automatic differentiation is: build a record of every small operation as it happens (the
"computation graph"), remember each operation's cheap local derivative, and walk the graph backward
multiplying local derivatives together via the chain rule. It never symbolically re-derives calculus (like
a computer-algebra system would) and it never approximates with finite differences (like numerical
differentiation would) — it mechanically composes exact local derivatives that were computed once, during
the forward pass. The from-scratch XOR notebook's $\delta^{[1]} = (\delta^{[2]}(W^{[2]})^T)\odot\tanh'(z^{[1]})$
line is a human doing exactly this graph-walk by hand for one specific two-node-deep graph; autograd does
the same walk for a graph with millions of nodes, automatically, no matter its shape.

## Why simpler approaches fail

Two "simpler" alternatives to autograd exist, and both fail for the reason `notes.md`'s from-scratch
derivation already hints at:

**Numerical differentiation** — approximate $\partial L/\partial \theta_i$ via finite differences,
$\big(L(\theta_i + \epsilon) - L(\theta_i - \epsilon)\big) / 2\epsilon$, for every parameter $\theta_i$
independently. This requires **two full forward passes per parameter**. A network with a million
parameters needs two million forward passes to get one gradient — computationally infeasible for any
real model, and it is also numerically imprecise (the result depends on the choice of $\epsilon$, too
large and you get truncation error, too small and you get floating-point cancellation error).

**Manual symbolic backprop, per architecture** — exactly what `06-deep-learning/01-ann/notes.md` and
`ann-from-scratch-xor.ipynb` did: derive $\delta^{[l]}$ and the weight-gradient formulas on paper, then
hand-code them for that one architecture. This is exact and cheap to *run* (one backward pass, not one per
parameter), but it does not scale to *build*: every new layer type (convolution, LSTM gate, attention head,
batch norm) needs its own hand-derived backward formula, and every new way of composing existing layers
(skip connections, branching, weight sharing) needs the composition's chain rule worked out by hand too.
A production deep learning codebase changes architecture constantly during research; re-deriving backprop
by hand for each change is a bottleneck that would make modern deep learning practically impossible.

Autograd keeps the *cheap-to-run* property of manual backprop (one backward pass, not one per parameter)
while removing the *re-derive-by-hand-per-architecture* cost: the local derivative of each primitive
operation (multiply, add, `tanh`, matmul, `sin`, ...) is coded once, in the framework, and the chain rule
composition across arbitrarily many of them is handled by walking the recorded graph — not by a human.

## Mathematical foundation

### The computation graph

Any composite function, however deep, is a sequence (a directed acyclic graph, in general) of primitive
operations, each with known inputs, known outputs, and a known **local derivative** — the Jacobian of that
one operation with respect to its own inputs. For a chain $x \to v_1 \to v_2 \to \cdots \to v_n = L$, the
total derivative of the final output with respect to the input is the product of every local Jacobian along
the path, by the chain rule:

$$\frac{\partial L}{\partial x} = \frac{\partial v_n}{\partial v_{n-1}}\frac{\partial v_{n-1}}{\partial v_{n-2}}\cdots\frac{\partial v_1}{\partial x}$$

For a graph with many inputs feeding into many intermediate nodes (not a single chain), the same idea
applies with sums over paths: the derivative with respect to any node is the sum, over every path from that
node to the output, of the product of local derivatives along that path (this is the multivariate chain
rule). There are two directions in which this product can be evaluated, and they cost very different
amounts depending on the shape of the graph.

### Forward-mode vs. reverse-mode AD

**Forward-mode AD** propagates derivatives *with* the computation, from inputs to outputs: alongside each
value $v_i$, carry $\dot v_i = \partial v_i/\partial x_j$ for one chosen input $x_j$, updated by the chain
rule as each new $v_i$ is computed. One forward-mode pass computes the derivative of **every** output with
respect to **one** input. To get the gradient of a scalar loss $L$ with respect to $n$ input parameters,
this requires $n$ separate passes — one per input — which is the same cost profile that made numerical
differentiation impractical above.

**Reverse-mode AD** does the opposite: it first runs the full forward pass, recording the graph and every
local derivative, then propagates derivatives *backward* from the output, computing
$\bar v_i = \partial L/\partial v_i$ for each node, seeded with $\bar L = \partial L/\partial L = 1$ at the
single output and pushed backward via the chain rule, accumulating contributions from every downstream node
that used $v_i$. One reverse-mode pass computes the derivative of **one** output with respect to **every**
input, simultaneously.

**Why reverse-mode wins for deep learning:** training a network means computing the gradient of one scalar
loss $L$ with respect to potentially millions of parameters — a many-inputs, one-output problem, precisely
the shape reverse-mode is built for. One forward pass plus one backward pass yields the entire gradient,
$O(1)$ passes regardless of parameter count. Forward-mode would need one pass *per parameter*, $O(n)$ passes
for $n$ parameters — infeasible for millions of parameters, exactly like numerical differentiation. (Forward
mode is instead the efficient choice for the opposite shape: few inputs, many outputs — e.g. sensitivity of
many outputs to one or two physical design parameters — which is uncommon in ML training but does appear in
some Jacobian-vector-product use cases.) `.backward()` in PyTorch (below) *is* one reverse-mode pass; the
from-scratch $\delta^{[l]}$ recursion in `06-deep-learning/01-ann/notes.md` is the identical
many-inputs/one-output backward walk, worked out by hand for one specific 2-layer graph instead of
implemented generically for any graph.

## Algorithm

Reverse-mode automatic differentiation, generically:

1. **Forward pass:** execute the computation normally, producing the output value(s). As each primitive
   operation runs, record it as a node in a graph, storing (a) its inputs, (b) enough information to compute
   its local derivative (e.g. `tanh`'s local derivative needs its own output; `matmul`'s needs both operand
   matrices), and (c) which nodes consumed its output.
2. **Seed the backward pass:** set the gradient of the final scalar output with respect to itself to 1
   ($\bar L = 1$).
3. **Backward pass:** visit nodes in reverse topological order (outputs before their inputs). For each node,
   multiply its stored local derivative by the gradient accumulated at its output(s), and **accumulate**
   (sum) the result into each of its input nodes' gradients — summing matters whenever a node's output feeds
   more than one downstream node, since the multivariate chain rule sums contributions over every path.
4. When the backward pass reaches a leaf node marked as requiring a gradient (a parameter), its accumulated
   gradient is the answer: $\partial L/\partial \theta$.

This is architecture-agnostic: nothing above assumes a specific layer count, layer type, or connection
pattern — the graph recorded in step 1 can be any DAG, which is exactly what removes the "re-derive per
architecture" cost identified in "Why simpler approaches fail."

## From-scratch implementation

This topic does **not** re-derive or re-implement backpropagation from scratch — that work already exists,
done in full, in `06-deep-learning/01-ann/notes.md` (the general $\boldsymbol{\delta}^{[l]}$ recursion) and
`06-deep-learning/01-ann/ann-from-scratch-xor.ipynb` (that recursion coded by hand in NumPy for a 2-layer
`tanh`/sigmoid MLP on the XOR dataset, function `backward()`). Re-deriving the same chain-rule mechanics
here would duplicate that work rather than add insight.

Instead, the point of this topic is the bridge: **the from-scratch `backward()` function *is* one
hand-executed instance of the reverse-mode AD algorithm above**, specialized to exactly one graph shape (a
2-layer `tanh` → sigmoid MLP, BCE loss) that the author worked out on paper first. Line by line:

| From-scratch (`ann-from-scratch-xor.ipynb`, `backward()`) | General reverse-mode AD step above |
|---|---|
| `delta2 = a2 - y` | Seed the backward pass at the output node ($\bar L$ passed through the loss+sigmoid combination) |
| `dW2 = a1.T @ delta2 / m` | Multiply the local derivative of the `matmul` node ($a^{[1]}$) by the accumulated output gradient ($\delta^{[2]}$) |
| `dtanh = 1 - np.tanh(z1) ** 2` | Compute the local derivative of the `tanh` node |
| `delta1 = (delta2 @ W2.T) * dtanh` | Push the gradient backward through the `matmul` node ($W^{[2]T}$) and then multiply by the `tanh` node's local derivative — one backward step per graph edge |
| `dW1 = X.T @ delta1 / m` | Multiply the local derivative of the first `matmul` node ($X$) by the accumulated gradient at that node |

The "Practical implementation" section below runs the identical 2-layer computation through PyTorch's
autograd engine instead, and numerically confirms the two agree — the point is not that autograd computes a
*different* answer, but that it computes the *same* answer as the hand-derived formulas above, without a
human having derived a `backward()` function specific to this graph at all.

## Practical implementation

PyTorch's `torch.Tensor` is a drop-in replacement for `np.ndarray` — same shape/dtype/indexing semantics,
same underlying contiguous memory layout — with one addition: passing `requires_grad=True` tells PyTorch to
record every operation performed on that tensor into a computation graph (step 1 of the algorithm above).
Calling `.backward()` on a scalar output runs the reverse pass (steps 2–3); each leaf tensor's `.grad`
attribute then holds its accumulated gradient (step 4) — the direct PyTorch analogue of `dW1`, `db1`, `dW2`,
`db2` in the from-scratch notebook.

The companion notebook (`01-tensors-and-autograd.ipynb`) runs, with real output:

1. A small worked example — `x = torch.tensor(..., requires_grad=True)`, a short chain of operations, and
   `.backward()` — inspecting `.grad` and confirming it matches a hand-computed derivative, to build
   intuition for what `.backward()` actually does before touching the MLP.
2. The **exact same 2-layer XOR MLP** as `ann-from-scratch-xor.ipynb` (same weight values, same forward
   equations $z^{[1]}=XW^{[1]}+b^{[1]}$, $a^{[1]}=\tanh(z^{[1]})$, $z^{[2]}=a^{[1]}W^{[2]}+b^{[2]}$,
   $\hat y=\sigma(z^{[2]})$, same BCE loss), built with `torch.Tensor(..., requires_grad=True)` weights
   instead of plain NumPy arrays, and **no hand-written `backward()` function at all** — one call to
   `loss.backward()` computes every gradient.
3. **The cross-check** (also this topic's Experiment, below): the from-scratch NumPy `backward()` and
   PyTorch's `.backward()` are run on **identical weight values and identical input data**, and their
   resulting gradients (`dW1, db1, dW2, db2` vs. `W1.grad, b1.grad, W2.grad, b2.grad`) are compared with
   `np.allclose()`.

## Experiment

**Hypothesis (stated before running):** since both the from-scratch `backward()` and PyTorch's autograd
implement the same mathematical operation — reverse-mode differentiation of the same BCE-loss-of-a-2-layer-
tanh/sigmoid-MLP computation graph, at the same weight values — their computed gradients should agree to
floating-point precision, i.e. `np.allclose(manual_grad, autograd_grad)` should be `True` for every one of
`W1`, `b1`, `W2`, `b2`.

**Setup:** initialize `W1, b1, W2, b2` once (NumPy arrays with a fixed seed), run the from-scratch forward
+ `backward()` on the 4-point XOR dataset to get `dW1, db1, dW2, db2`; separately, build PyTorch tensors
from those *exact same* NumPy values with `requires_grad=True`, run the identical forward computation, call
`.backward()`, and read off `W1.grad, b1.grad, W2.grad, b2.grad`.

**Actual result (from the executed notebook):**

```
max |dW1 diff| = 6.94e-18
max |db1 diff| = 6.94e-18
max |dW2 diff| = 2.78e-17
max |db2 diff| = 2.78e-17
np.allclose(dW1, W1.grad) = True
np.allclose(db1, b1.grad) = True
np.allclose(dW2, W2.grad) = True
np.allclose(db2, b2.grad) = True
ALL GRADIENTS MATCH: True
```

**Interpretation:** PyTorch's autograd, applied to the same graph, at the same weights, reproduces the
hand-derived gradient exactly (to floating-point precision) — confirming autograd is not a different or
approximate method, but the same reverse-mode chain-rule computation the from-scratch notebook performed by
hand, generalized so it never has to be re-derived for a new graph.

**Limitations:** this cross-check exercises one small, 2-layer graph with a handful of parameters and 4
data points — it demonstrates correctness-of-agreement on a case simple enough to hand-verify, not a
performance or scalability claim (autograd's actual value is precisely that it doesn't need to be re-derived
as the graph grows, which this toy comparison can't exercise directly).

## Failure modes

- **Forgetting `.zero_grad()` (gradient accumulation):** PyTorch's `.backward()` *accumulates* into
  `.grad` — it adds the newly computed gradient to whatever is already stored there, rather than replacing
  it (this supports legitimate use cases like accumulating gradients across several mini-batches before a
  single update). If a training loop calls `.backward()` repeatedly without clearing `.grad` first (usually
  via `optimizer.zero_grad()` or `tensor.grad = None`), gradients from previous steps silently add into the
  current step's update, corrupting training in a way that doesn't crash — it just trains wrong, slowly and
  non-obviously.
- **In-place operations breaking the graph:** operations like `x += 1` or `x.add_(1)` (the trailing
  underscore is PyTorch's in-place-op convention) mutate a tensor's memory directly. If that tensor is
  needed, unmodified, by the backward pass of some other node in the graph (e.g. `tanh`'s local derivative
  needs the value it originally computed), an in-place overwrite can invalidate what the graph recorded,
  raising a `RuntimeError` at `.backward()` time ("a variable needed for gradient computation has been
  modified by an inplace operation") — or, in rarer cases, silently returning a wrong gradient.
- **`.detach()` / `.item()` confusion:** `.detach()` returns a new tensor sharing the same data but *removed*
  from the computation graph (no gradient flows through it) — needed when a value should be used but not
  backpropagated through (e.g. logging predictions, or freezing part of a network). `.item()` extracts a
  plain Python scalar from a single-element tensor, discarding both graph *and* tensor-ness — useful for
  logging a loss value, but calling `.item()` on something still needed downstream in the graph silently
  removes it from future differentiation. Using `.detach()` when a gradient *was* needed (or vice versa)
  produces no error but a wrong training signal.

## Real-world usage

Every deep learning framework in production use today (PyTorch, TensorFlow/Keras, JAX) is built on reverse-
mode automatic differentiation — it is the mechanism that made deep learning practical at the scale of
modern architectures (`06-deep-learning/01-ann/notes.md`'s note that "Keras/TensorFlow computes all of this
automatically via automatic differentiation" is the exact same mechanism this topic makes explicit). Beyond
neural network training, reverse-mode AD underlies gradient-based optimization broadly: physics simulators
that are differentiated end-to-end, probabilistic programming (gradients of a likelihood with respect to
model parameters), and any pipeline where "find the input/parameter that minimizes some scalar objective" is
the task.

## Mental model

Autograd is the from-scratch notebook's hand-written `backward()` function, generalized: instead of a human
looking at one specific 2-layer computation graph and deriving its $\delta^{[l]}$ recursion on paper once,
the framework records *any* graph as it is built during the forward pass and walks it backward automatically,
multiplying the same kind of local derivatives the human used, for graphs of arbitrary shape and depth. The
math (reverse-mode chain rule) is identical either way — what changes is who does the derivation: a person,
once, per architecture, or the framework, every time, for free.

## Questions to think about

1. Why does reverse-mode AD need exactly one forward pass and one backward pass to get the gradient with
   respect to *every* parameter simultaneously, while forward-mode AD would need one full pass per
   parameter to get the same result?
2. The from-scratch notebook's `delta1 = (delta2 @ W2.T) * dtanh` line multiplies by `W2.T`, the transpose of
   the weight matrix used in the *forward* pass. Why does the backward pass use the transpose, in terms of
   the local-derivative-of-a-matmul-node argument from "Mathematical foundation"?
3. If a tensor is created with `requires_grad=False` (PyTorch's default) and is never touched again, what
   does PyTorch's computation graph look like around it, and what happens if you try to call `.backward()`
   on a loss that depends on it?
4. `optimizer.zero_grad()` (covered fully in `02-nn-module-and-training-loop`) exists specifically because
   `.grad` accumulates rather than overwrites. Construct a concrete two-step training scenario where
   forgetting it would make the loss visibly fail to decrease, and explain the mechanism.
5. The cross-check experiment above initializes PyTorch's weights from the *exact same* NumPy values used in
   the from-scratch run. Why is that necessary for `np.allclose()` to be a meaningful test — what would a
   mismatch actually tell you if the weights had instead been initialized independently in each?
