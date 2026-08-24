# 04 – GPU, Mixed Precision, and Profiling

## Problem

Every notebook in this repository — every Keras model in
`06-deep-learning`, every PyTorch model in `09-pytorch/01` through `03` —
has trained on CPU, because this development environment has no GPU
(`torch.cuda.is_available()` returns `False` here, verified below). For
the small toy datasets and small models used so far (4-point XOR, 569-row
Breast Cancer Wisconsin, small MLPs), CPU training finishes in seconds.
**Real models don't stay that small.** A production-scale CNN, RNN, or
Transformer can have millions to billions of parameters and be trained on
datasets far larger than anything in this repo — at that scale, CPU
training that would take seconds here can take weeks, making it
practically unusable. Separately, even once a GPU is available, training
every number as a 32-bit float (fp32) — the default, and what every
notebook so far has implicitly used — spends more memory and compute per
operation than is always necessary. **This topic asks: what does a GPU
actually do differently from a CPU that makes it faster for this kind of
workload, why does reducing numeric precision help further without
breaking training, and how do you *measure* where time is actually going
before reaching for either?**

## Intuition

A CPU is built like a small team of a few, very capable generalists: a
handful of cores (often 4–64), each able to execute complex, varied
instructions quickly, with sophisticated branch prediction and large
per-core caches — excellent for tasks with lots of different logic and
sequential dependencies. A GPU is built like a stadium full of workers
doing the exact same simple arithmetic step, all at once: thousands of
small, simple compute units, each much less capable individually than a
CPU core, but present in overwhelming numbers, and all executing the
*same* instruction on different pieces of data simultaneously
(SIMD/SIMT). Training a neural network is, at its core, an enormous
number of small, identical multiply-then-add operations — matrix
multiplication — repeated over and over across a batch. That workload is
exactly what a stadium of simple workers doing the same step in unison
outperforms a small team of generalists at, and exactly what a CPU's
"few powerful cores" design isn't shaped for.

Mixed precision's intuition: a highly precise, expensive-to-produce
32-bit measurement isn't always needed for every step of a computation
that gets recombined and re-measured thousands of times afterward. Most
of a network's matrix multiplications tolerate a coarser 16-bit
representation just fine; the moments that don't (accumulating many
values into a sum, or comparing a gradient against a very small learning
rate) are kept at full 32-bit precision. Doing most of the work in the
cheaper representation and only the sensitive parts in the expensive one
gets most of the speed and memory savings without most of the risk.

## Why simpler approaches fail

**"Just run everything on CPU":** this is exactly what this repository
has done throughout, and it has been the right choice *for toy-scale
work* — every notebook so far, on CPU, still finishes in seconds to a few
minutes. It stops being viable the moment model size or dataset size
grows past toy scale: a CPU's small number of powerful cores has no way
to parallelize a billion-parameter model's matrix multiplications the way
a GPU's thousands of simple cores can, so training time scales in a way
that becomes impractical (hours become days, days become weeks) long
before a GPU's equivalent training run would.

**"Just always use fp32":** fp32 is the safe default precisely because it
avoids ever worrying about numeric range or precision — but it uses twice
the memory of fp16/bf16 per number, and GPU hardware built for
mixed-precision workloads can execute fp16/bf16 matrix multiplications
significantly faster than fp32 ones. Refusing to ever use lower precision
leaves that speed and memory on the table even when it costs nothing in
accuracy — but naively switching *everything* to fp16, with no
countermeasure, introduces the numeric instability the next section
explains and defends against.

## Conceptual foundation

*(This section documents a deliberate substitution: this is a
hardware/systems topic, not one with a derivable closed-form mathematical
foundation — there is no equation to derive the way `01-tensors-and-autograd`
derives the chain rule. Per `08-mlops-deployment/04-model-packaging-versioning/notes.md`'s
precedent for the same situation, this section is titled "Conceptual
foundation" in place of "Mathematical foundation," and explains the
underlying mechanism in place of a derivation.)*

**What a GPU parallelizes.** A single `nn.Linear` forward pass computes
$y = xW^T + b$ — for a batch of $m$ examples, an input dimension $n$, and
output dimension $k$, this is an $(m \times n) \cdot (n \times k)$ matrix
multiplication: $m \times n \times k$ individual multiply-then-accumulate
(MAC) operations, all independent of each other until the final
summation per output element. A CPU core executes these mostly
sequentially (with some vectorization via SIMD instructions like AVX, but
still on the order of tens of values at once per core, across a handful
of cores). A GPU is built from thousands of small arithmetic units
(CUDA cores on NVIDIA hardware, further accelerated for exactly this
matrix-multiply pattern by dedicated "tensor cores" on modern GPUs) that
can each independently execute one MAC operation, all at the same time —
turning an operation that takes $m \times n \times k$ *sequential* steps
on a CPU core into one that takes far fewer *parallel* steps on a GPU,
because thousands of independent MACs are computed simultaneously rather
than one after another.

**Why mixed precision speeds training up without destroying stability.**
Floating-point numbers trade off range and precision for the number of
bits used to represent them:

| Format | Bits | Notes |
|---|---|---|
| fp32 (single precision) | 32 | Default in this repo so far; wide range, high precision |
| fp16 (half precision) | 16 | Half the memory/bandwidth of fp32; narrower exponent range — can overflow/underflow more easily |
| bf16 (bfloat16) | 16 | Same exponent range as fp32 (so no overflow risk fp16 has), but fewer mantissa bits (less precision per value) |

Two mechanisms make mixed precision safe rather than just fast:

1. **Automatic mixed precision (`torch.autocast`)** selectively runs only
   the operations that tolerate reduced precision well — mainly matrix
   multiplications and convolutions, which dominate total FLOPs — in
   fp16/bf16, while automatically keeping numerically sensitive
   operations (loss reductions, softmax, batch-norm statistics) in fp32.
   This is a per-operation decision PyTorch makes internally based on a
   maintained list of which ops are safe to downcast, not a blanket
   "everything is fp16" switch.
2. **fp32 master weights + a loss scaler (`GradScaler`)** address the two
   specific failure modes reduced precision introduces. Gradients
   computed in fp16 can be extremely small (products of many small
   derivatives) and underflow to exactly zero in fp16's narrower range
   before they're used to update anything — `GradScaler` multiplies
   ("scales up") the loss by a large factor *before* `.backward()`, so
   the resulting gradients are scaled proportionally larger and stay
   representable in fp16, then *unscales* them back down before the
   optimizer step, so the actual update applied matches what an
   unscaled fp32 computation would have produced. Separately, keeping a
   master copy of the weights in fp32 (which `optimizer.step()` updates)
   avoids the accumulation error of repeatedly adding small fp16 gradient
   updates to fp16-precision weights over many steps, where the update
   itself can be too small to change the fp16 value at all.

## Algorithm

**GPU training loop (conceptually — see "Practical implementation" for
real code):**
1. Move the model's parameters and each batch's tensors onto the GPU
   (`model.to("cuda")`, `x.to("cuda")`) — this is a one-time model move
   plus a per-batch data move.
2. Run the forward pass and loss computation inside an `autocast` context,
   so eligible ops run in fp16/bf16 automatically.
3. Scale the loss up (`scaler.scale(loss)`), call `.backward()` on the
   scaled loss so the resulting (also-scaled) gradients don't underflow.
4. Unscale and step (`scaler.step(optimizer)`) — internally unscales
   gradients back to their true magnitude before applying the optimizer's
   update rule; `scaler.update()` adjusts the scale factor for the next
   iteration based on whether any overflow (inf/NaN gradient) was
   detected this step.
5. Repeat per batch/epoch, exactly as the fp32 CPU loop in
   `02-nn-module-and-training-loop`/`03-datasets-dataloaders-checkpointing`
   does structurally — mixed precision changes *how* steps 2–4 execute
   internally, not the five-step shape of the loop itself.

**Profiling (actually run on CPU below):**
1. Wrap the region of code to measure in `torch.profiler.profile(...)`.
2. Run the code (here: a few training epochs) inside that context.
3. Read the captured per-operator breakdown (`prof.key_averages().table(...)`)
   to see which operations actually consumed the most CPU (or GPU) time —
   *before* deciding what to optimize.

## From-scratch implementation

**N/A for this topic, explicitly.** Per `AGENTS.md`'s "From-scratch"
guidance ("when it adds real insight... don't reimplement mature
production systems for their own sake") and the brief's own judgment
call, hardware parallelism and GPU driver/kernel scheduling are not
concepts a small NumPy reimplementation can meaningfully illuminate — a
from-scratch "GPU simulator" would either be a trivial loop that teaches
nothing about real parallel hardware, or a large undertaking (writing
actual CUDA kernels) far outside this topic's scope and this
environment's capability (no GPU present to run them on). The genuine
insight this topic offers — *why* parallel hardware and reduced
precision help, and *how* to verify where time is spent before optimizing
— is delivered instead through the "Conceptual foundation" mechanism
explanation above and the real, executed CPU profiler trace below.

## Practical implementation

The companion notebook (`04-gpu-mixed-precision-profiling.ipynb`) is
split into two honestly-labeled parts:

**GPU/mixed-precision code — written and reviewed, NOT executed here.**
`train_one_epoch_amp()` is real, correct PyTorch code implementing the
algorithm above (`.to("cuda")`, `torch.autocast(device_type="cuda",
dtype=torch.float16)`, `torch.cuda.amp.GradScaler`), guarded by
`if torch.cuda.is_available():`. Verified first, in this environment:

```
torch.cuda.is_available() -> False
```

so the guard's `else` branch runs instead, printing an honest skip
message — the function is defined and reviewed for correctness against
the mechanism described above, but has never actually executed, in this
notebook or anywhere in this environment. This mirrors
`AGENTS.md`'s discipline for the un-executed Dockerfile precedent from
Phase 3: real, reviewed code, honestly marked as not run rather than
faked with fabricated output.

**CPU profiling — actually run, with real output.** `torch.profiler`
requires no GPU when scoped to `activities=[ProfilerActivity.CPU]`; the
notebook profiles a real training loop (the same `MLP` architecture and
Breast Cancer Wisconsin dataset from `03-datasets-dataloaders-checkpointing`,
trained with `Adam` for 3 profiled epochs after 1 unprofiled warm-up
epoch) and prints the real captured operator-level breakdown — see
"Experiment" below for the actual trace.

## Experiment

**Hypothesis (stated before running):** for a small fully-connected MLP,
matrix-multiply-family ops underlying `nn.Linear` (`aten::mm`,
`aten::addmm`, and their autograd backward-pass counterparts) should
dominate total CPU time, since that's where nearly all the
floating-point work in this model happens; data loading and Python-loop
overhead should be comparatively small.

**Setup:** `MLP` with three `nn.Linear` layers (hidden width 64) trained
on the 455-row Breast Cancer Wisconsin training split, batch size 32,
`Adam(lr=1e-2)`, `BCELoss`; one unprofiled warm-up epoch, then 3 epochs
wrapped in `torch.profiler.profile(activities=[ProfilerActivity.CPU],
record_shapes=True)`.

**Actual result (real captured trace, executed in this environment):**

```
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
                                                   Name    Self CPU %      Self CPU   CPU total %     CPU total  CPU time avg    # of Calls
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
    autograd::engine::evaluate_function: AddmmBackward0         0.25%       1.546ms        47.41%     297.238ms       2.202ms           135
                                         AddmmBackward0         0.19%       1.178ms        46.90%     294.006ms       2.178ms           135
                                               aten::mm        46.50%     291.533ms        46.51%     291.586ms       1.296ms           225
                               Optimizer.step#Adam.step         1.88%      11.805ms        27.18%     170.368ms       3.786ms            45
                                             aten::sqrt        23.53%     147.522ms        23.53%     147.522ms     546.378us           270
                                           aten::linear         0.06%     405.272us        21.42%     134.317ms     994.938us           135
                                            aten::addmm        21.15%     132.568ms        21.24%     133.151ms     986.305us           135
enumerate(DataLoader)#_SingleProcessDataLoaderIter._...         1.06%       6.660ms         2.14%      13.445ms     280.111us            48
                                           aten::select         0.72%       4.492ms         0.85%       5.332ms       1.953us          2730
                                             aten::add_         0.42%       2.662ms         0.76%       4.744ms       8.785us           540
                                               aten::to         0.09%     558.365us         0.65%       4.054ms       3.465us          1170
                                         aten::_to_copy         0.35%       2.186ms         0.56%       3.496ms       2.988us          1170
-------------------------------------------------------  ------------  ------------  ------------  ------------  ------------  ------------
Self CPU time total: 626.915ms

Top op by total CPU time: autograd::engine::evaluate_function: AddmmBackward0  (297.24 ms total, 135 calls)
Share of total CPU time in matmul-family ops (addmm/linear/mm): 74.6%
```

**Interpretation:** the trace confirms the core hypothesis — matmul-family
ops (`aten::mm`/`aten::addmm`, plus their `AddmmBackward0` autograd
counterparts) account for ~75% of total CPU time, consistent with a model
whose compute is dominated by matrix multiplication. The trace also
surfaces something the hypothesis didn't predict: `Optimizer.step#Adam.step`
(27% of total time, mostly `aten::sqrt` for Adam's per-parameter variance
normalization) is a non-trivial secondary cost that a plain-SGD optimizer
would not pay at all — a concrete reminder that profiling, not intuition,
is what tells you where time actually goes. Data loading (~2%) and
Python-loop overhead are, as hypothesized, comparatively small at this
scale.

**Limitations:** small model (three `nn.Linear` layers, hidden width 64),
small dataset (455 training examples), 3 profiled epochs, one CPU, one
machine — exact percentages will differ for a larger model, different
batch size, different optimizer, or different hardware; profiler
instrumentation itself adds measurable overhead (see "Failure modes").

## Failure modes

- **Mixed-precision overflow/underflow without a loss scaler.** Running
  the backward pass in fp16 without `GradScaler` risks gradients
  underflowing to exactly zero (silently stalling learning for affected
  parameters) or, less commonly, activations overflowing to `inf`
  (poisoning the loss with `NaN`). `GradScaler`'s scale-up-before-backward,
  unscale-before-step pattern exists specifically to prevent this — using
  `autocast` without a paired `GradScaler` on the backward pass reintroduces
  the exact risk mixed precision is supposed to avoid.
- **Profiler overhead skewing the measurement it's taking.** `torch.profiler`
  instruments every operation it observes, and that instrumentation itself
  costs CPU time — the "Self CPU time total: 626.915ms" reported above is
  measured *while being profiled*, and will be somewhat higher than the
  same 3 epochs would take unprofiled. This is a standard measurement
  observer effect: profile to find *relative* hot spots (which op
  dominates), not to get an exact absolute wall-clock number — for that,
  time the unprofiled run separately.
- **Assuming GPU speedup without profiling first (premature optimization).**
  Moving a workload to a GPU or switching to mixed precision only helps
  time actually spent in the operations they accelerate (dense
  matrix/convolution math). If a real bottleneck were instead disk I/O,
  data preprocessing, or Python-level control flow — none of which a GPU
  or `autocast` speeds up — buying/renting a GPU or adding `autocast`
  would add engineering complexity for little or no real speedup. This
  topic's CPU profiler trace is exactly the tool that distinguishes these
  cases: profile before optimizing, not after.

## Real-world usage

Virtually all real-scale deep learning training runs — vision models,
language models, recommendation systems — happen on GPUs (or
purpose-built accelerators like TPUs), with mixed precision as close to a
default as CPU float32 was in this repo's earlier notebooks: PyTorch's
`torch.cuda.amp` (or the newer unified `torch.amp` API) and NVIDIA's
Apex library exist specifically to make `autocast`+`GradScaler` a drop-in
addition to an existing training loop with minimal code change. Framework
defaults increasingly assume mixed precision is on unless explicitly
disabled (e.g. Hugging Face `Trainer`'s `fp16=True`/`bf16=True` flags).
Profiling before optimizing is likewise standard production practice:
`torch.profiler` traces (often visualized in TensorBoard or Chrome's
trace viewer) are the first diagnostic step before any GPU/precision
optimization work, in the same spirit `08-mlops-deployment/08-monitoring`'s
"measure before you act" discipline applies to production model
monitoring.

## Mental model

A CPU is a small team of expert generalists; a GPU is a stadium of simple
workers doing the same step in unison — exactly matched to a neural
network's core operation (millions of small, identical multiply-adds).
Mixed precision is "do the bulk, tolerant work in a cheaper unit, keep an
accurate master copy and a safety margin (scaling) around the sensitive
parts" — not "throw away precision everywhere and hope." And a profiler
is a stopwatch with a magnifying glass: it tells you, with real numbers,
which specific operation is actually consuming the time you're trying to
save — the only honest starting point before reaching for a GPU or
mixed precision at all.

## Questions to think about

1. The CPU profiler trace found `Optimizer.step#Adam.step` responsible
   for 27% of total time, almost entirely `aten::sqrt`. If this model
   used plain SGD (`02-nn-module-and-training-loop/notes.md`'s comparison
   optimizer) instead of Adam, would you expect that 27% to mostly
   disappear, mostly persist, or shift into the matmul ops? Justify from
   what each optimizer's update rule actually computes.
2. `GradScaler`'s scale factor is adjusted automatically each step
   (`scaler.update()`), decreasing it if an `inf`/`NaN` gradient was
   detected and increasing it otherwise. Why would a *fixed*, hand-picked
   scale factor be worse than this adaptive scheme — think about what
   happens if the fixed factor is chosen too small vs. too large for a
   given model's typical gradient magnitudes.
3. `autocast` keeps some operations in fp32 even inside its context (loss
   reductions, softmax) rather than downcasting everything. Pick one such
   operation and explain, in terms of what it numerically computes (a sum
   over many values, or an exponential), why running it in fp16 would be
   riskier than running a single matrix multiply in fp16.
4. The profiler trace's "Self CPU time total" (626.915ms) measures 3
   profiled epochs *including* profiler instrumentation overhead. Design
   a follow-up experiment (described, not necessarily run) that would let
   you estimate how much of that total is the instrumentation itself,
   versus the training loop it's observing.
5. Given this topic's own conclusion ("profile before optimizing"), what
   would be a legitimate reason a team might still add `.to("cuda")` and
   `autocast` to a training script *before* profiling it on their
   specific new model — i.e., under what conditions is skipping the
   profile-first step actually reasonable, and under what conditions is
   it premature optimization?
