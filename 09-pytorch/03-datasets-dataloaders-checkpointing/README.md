# 03 – Datasets, DataLoaders, and Checkpointing

Detailed notes (why a big in-memory array and a no-checkpoint training loop both break down
at real scale, the `Dataset`/`DataLoader` iterator split, checkpointing as periodic state
serialization): [notes.md](notes.md)

From-scratch + practical + experiment: a manual batching/shuffling generator in plain Python,
a real PyTorch `Dataset`/`DataLoader` pipeline over the Breast Cancer Wisconsin dataset,
a training loop that checkpoints every 3 epochs, and a real crash-and-resume verifying the
reloaded model continues training correctly — [03-datasets-dataloaders-checkpointing.ipynb](03-datasets-dataloaders-checkpointing.ipynb)

## What you'll learn

Why loading an entire dataset into one NumPy array (every notebook in this repo so far) has
no answer for data larger than memory, and why a training loop with no way to save its
progress has to restart from scratch on any crash. `Dataset`/`DataLoader` splits "fetch one
item" from "decide what order/batches to fetch items in"; checkpointing periodically
serializes model **and** optimizer state so a crashed run can resume, not just restart.

| Topic | Status |
|-------|--------|
| Why one big in-memory array and no checkpointing both break down at scale | ✅ Complete |
| `Dataset`/`DataLoader` as an iterator abstraction | ✅ Complete |
| Manual batching/shuffling generator, from scratch | ✅ Complete |
| Real `Dataset`+`DataLoader` training with periodic `torch.save` checkpoints | ✅ Complete |
| Real crash simulation + checkpoint reload + verified resume | ✅ Complete |
| Failure mode demonstrated: dropped optimizer state on resume | ✅ Complete |

## Why it matters

`01-tensors-and-autograd` and `02-nn-module-and-training-loop` removed the need to hand-derive
gradients and hand-track parameter tensors; this topic removes the third manual burden every
toy notebook so far has quietly relied on — that all the data fits in memory and training
never gets interrupted. Neither assumption survives contact with real training runs.

## Prerequisites

- `02-nn-module-and-training-loop` — the 5-step training loop this topic checkpoints.
- `08-mlops-deployment/04-model-packaging-versioning/notes.md` — the content-addressing idea
  this topic's checkpoint files connect back to.

## What you'll build

- A plain-Python manual batching/shuffling generator over a toy list, with no `torch` — what
  `DataLoader` automates, made visible.
- A real `torch.utils.data.Dataset`/`DataLoader` pipeline over the Breast Cancer Wisconsin
  dataset, and an `MLP` trained on it with `Adam`, checkpointing every 3 epochs.
- A real "simulate a crash, resume in a fresh process" experiment: a brand-new model and
  optimizer, populated only from a saved checkpoint file, that continues training with a
  smoothly decreasing loss — plus a side-by-side demo of what happens when optimizer state is
  dropped on resume.

## Where it appears in real systems

Every production training pipeline uses this exact `Dataset`/`DataLoader` split (usually with
parallel background loading via `num_workers`) for datasets far too large to hold in memory,
and checkpointing is standard practice for any training run long enough to risk interruption —
especially on preemptible/spot cloud instances, which are specifically designed around the
assumption that training can be reclaimed and resumed at any moment.

## What's next

`04-gpu-mixed-precision-profiling` addresses the next scaling problem this topic's toy dataset
doesn't surface: CPU training itself becoming the bottleneck once models and datasets grow
past toy scale, and how to speed that up (and measure the speedup) correctly.
