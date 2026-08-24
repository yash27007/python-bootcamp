# 04 – GPU, Mixed Precision, and Profiling

Detailed notes (what a GPU actually parallelizes vs. a CPU, why mixed precision speeds
training up without destroying numerical stability, the "Conceptual foundation" substitution
documented explicitly since this is a hardware/systems topic, not a derivable-math one):
[notes.md](notes.md)

Practical + experiment: real, reviewed `.to("cuda")`/`autocast`/`GradScaler` code honestly
marked as not executed (no GPU in this environment — verified), plus a real, actually-run
CPU `torch.profiler` trace over a training loop —
[04-gpu-mixed-precision-profiling.ipynb](04-gpu-mixed-precision-profiling.ipynb)

## What you'll learn

Why every notebook in this repo has trained on CPU (no GPU available here), what a GPU's many
small parallel compute units actually parallelize that a CPU's few powerful cores don't, and
the mechanism behind mixed-precision training (fp16/bf16 compute, fp32 master weights, and a
loss scaler) that makes it faster without silently breaking numerical stability. Also: why
profiling has to come *before* reaching for either — this environment's CPU profiler trace
demonstrates that discipline directly.

| Topic | Status |
|-------|--------|
| Why CPU training doesn't scale to real model sizes | ✅ Complete |
| GPU parallelism: many small MAC units vs. few powerful CPU cores | ✅ Complete |
| Mixed precision mechanism: autocast + fp32 master weights + GradScaler | ✅ Complete |
| `.to("cuda")`/`autocast`/`GradScaler` code — written, reviewed, honestly marked unexecuted | ✅ Complete |
| Real CPU `torch.profiler` trace over a real training loop | ✅ Complete |
| From-scratch: explicitly N/A, reasoning documented | ✅ Complete |

## Why it matters

This is the last topic in `09-pytorch`, and the one that connects everything built so far to
what real-scale training actually requires: the same `nn.Module` + training loop from
`02-nn-module-and-training-loop`, and the same `Dataset`/`DataLoader` pipeline from
`03-datasets-dataloaders-checkpointing`, run unchanged on a GPU with mixed precision at real
scale — with profiling as the honest way to know whether that move is even worth making for a
given workload.

## Prerequisites

- `02-nn-module-and-training-loop` and `03-datasets-dataloaders-checkpointing` — this topic's
  profiled training loop and (unexecuted) GPU training loop are the same pattern from those
  topics.
- `.venv/bin/python -c "import torch; print(torch.cuda.is_available())"` returns `False` in
  this environment — confirmed at the top of the companion notebook.

## What you'll build

- Real, correct PyTorch code for `.to("cuda")`, `torch.autocast(device_type="cuda", ...)`, and
  `torch.cuda.amp.GradScaler`, guarded by `torch.cuda.is_available()` and honestly printing a
  skip message in this GPU-less environment rather than fabricating output.
- A real CPU-only `torch.profiler` trace over an actual training loop (the
  `03-datasets-dataloaders-checkpointing` `MLP`/dataset), with a stated hypothesis about which
  operation dominates CPU time, confirmed against the real captured trace.

## Where it appears in real systems

Virtually every real-scale deep learning training run happens on a GPU with mixed precision
close to the default (PyTorch's `torch.amp`, Hugging Face `Trainer`'s `fp16=True`/`bf16=True`).
Profiling before optimizing (`torch.profiler`, often visualized in TensorBoard) is standard
practice before any GPU/precision optimization work in production ML engineering.

## What's next

This completes `09-pytorch`'s 4-topic build-out — see [`../README.md`](../README.md) for the
full section index. Later sections (`11-generative-ai` onward) build specific architectures on
this same PyTorch foundation, with the same "toy scale on CPU, real scale on GPU with mixed
precision" split this topic establishes.
