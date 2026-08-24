# 09 – PyTorch

First-principles PyTorch: autograd, `nn.Module` and the training loop, real data pipelines
and checkpointing, and the hardware/systems side of training at scale (GPU, mixed precision,
profiling) — each topic built by first bridging from a from-scratch NumPy implementation
already in this repo (`06-deep-learning/01-ann/ann-from-scratch-xor.ipynb`), then mapping to
the real framework.

| # | Topic | Status | Description |
|---|-------|--------|--------------|
| 01 | [Tensors and Autograd](./01-tensors-and-autograd/) | ✅ Complete | Reverse-mode automatic differentiation, `torch.Tensor`/`.backward()`, cross-checked against the from-scratch MLP's hand-derived gradients |
| 02 | [nn.Module and the Training Loop](./02-nn-module-and-training-loop/) | ✅ Complete | `nn.Module` as an automatic parameter container, the 5-step PyTorch training loop mapped line-by-line to the from-scratch loop |
| 03 | [Datasets, DataLoaders, and Checkpointing](./03-datasets-dataloaders-checkpointing/) | ✅ Complete | `Dataset`/`DataLoader` as a batching/shuffling iterator abstraction, periodic `torch.save` checkpointing with a real crash-and-resume experiment |
| 04 | [GPU, Mixed Precision, and Profiling](./04-gpu-mixed-precision-profiling/) | ✅ Complete | What a GPU parallelizes vs. a CPU, the mixed-precision (autocast + GradScaler) mechanism, a real CPU `torch.profiler` trace |

## Prerequisites

- `06-deep-learning/01-ann` — the from-scratch NumPy MLP every topic in this section bridges
  from.
- `08-mlops-deployment/04-model-packaging-versioning` and `08-mlops-deployment/07-cicd` —
  content-addressing and automation ideas topic 03 connects back to.

## Environment note

This repository's development environment has no GPU (`torch.cuda.is_available()` is
`False` here, on a CPU-only PyTorch build — see `pyproject.toml`). Every topic's from-scratch
and practical PyTorch code is actually run on CPU; topic 04's GPU-specific code
(`.to("cuda")`, `torch.autocast(device_type="cuda", ...)`, `GradScaler`) is written and
reviewed for correctness but honestly marked as not executed in this environment, per this
repo's discipline for un-runnable-here code (see `AGENTS.md`).

## What's next

Later sections (`11-generative-ai` onward, per `AGENTS.md`'s repository structure) build
specific architectures — generative models, agents, LLM components — on top of this section's
`nn.Module` + training-loop + data-pipeline foundation, at toy scale on CPU per those
sections' own no-heavy-training constraint.
