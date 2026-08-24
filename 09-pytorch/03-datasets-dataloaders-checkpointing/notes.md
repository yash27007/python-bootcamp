# 03 – Datasets, DataLoaders, and Checkpointing

## Problem

Every notebook so far in this repo — from `05-machine-learning` through
`09-pytorch/01-tensors-and-autograd` and `02-nn-module-and-training-loop` —
loads its entire dataset into one NumPy array before training starts:
`X, y = load_breast_cancer(return_X_y=True)`, or a 4-point XOR array typed
directly into a cell. This works because every one of those datasets is
small enough to fit in RAM whole. **Two problems break this pattern at real
scale:**

1. **The data itself doesn't fit.** A real image, audio, or text dataset
   can be gigabytes to terabytes — far larger than available RAM — so
   "load it all into one array first" is not merely inconvenient, it is
   impossible. Something has to load *pieces* of the dataset on demand.
2. **A long training run has no memory of its own progress.** Every
   training loop this repo has written so far runs to completion in one
   uninterrupted Python process. A real training run can take hours or
   days; if the process is killed (an out-of-memory error, a machine
   reboot, a cloud spot instance being reclaimed, someone `Ctrl-C`-ing by
   accident) with no way to recover, every epoch already trained is
   thrown away, and training has to start over from scratch.

This topic answers both: **how do you feed a model data it doesn't need to
hold in memory all at once, and how do you make a training run recoverable
after a crash instead of catastrophic?**

## Intuition

**Datasets/DataLoaders:** imagine a librarian handing a reader one book at
a time from a warehouse of a million books, instead of trying to stack all
million books on the reader's desk at once. The librarian doesn't need to
know anything about what's *inside* each book — just how to fetch book
number `i` when asked, and how many books there are in total. A separate
process decides *which* books to fetch and in what order (maybe shuffled,
maybe in batches of 5 at a time) — that's a completely different concern
from "how do I retrieve book `i`." Splitting these two concerns —
*fetching one item* vs. *deciding which items, batched and shuffled* — is
exactly the split between `torch.utils.data.Dataset` (the librarian:
`__getitem__(i)` and `__len__()`) and `torch.utils.data.DataLoader` (the
reading plan: batching, shuffling, optionally parallel fetching).

**Checkpointing:** imagine writing a long document with no autosave —
losing power means retyping everything from the start. Autosave doesn't
need to save *continuously*; it saves periodically (every few minutes, or
after every paragraph), and the document reopens exactly where it was.
Training a model is the same idea applied to a model's weights and an
optimizer's internal state: save both to disk periodically, and a crashed
run can reload the most recent save and continue, instead of restarting
at epoch 0.

## Why simpler approaches fail

- **One big in-memory array.** Every Phase 1/2 notebook in this repo used
  this pattern because every dataset used so far (XOR's 4 points, Breast
  Cancer Wisconsin's 569 rows, `make_moons`'s few hundred points) is tiny.
  The pattern itself — `X = np.array([...])`, `for x in X: ...` — has no
  way to represent "a dataset larger than RAM," no notion of fetching one
  example at a time from disk, and no built-in way to reshuffle the
  *order* examples are visited in without also copying the whole array.
  It is not wrong for small data; it simply has no answer for large data.
- **No checkpointing at all.** A training loop that keeps `model` and
  `optimizer` only as live Python objects in one process's memory loses
  both the instant that process dies, however far into training it was.
  For a training run measured in minutes (everything in this repo so far)
  that's a minor annoyance; for a run measured in hours or days, it can
  mean losing most of the compute already spent, at the exact moment
  (mid-run) a periodic save could have prevented it.

## Conceptual foundation

This topic's foundation is genuinely conceptual — an iterator/abstraction
pattern and a serialization discipline — rather than a derivation, so it
follows the "Algorithm" section directly rather than deriving equations
first (there is nothing new to derive: the forward/backward/update math is
exactly `02-nn-module-and-training-loop/notes.md`'s).

**`Dataset` + `DataLoader` as an iterator abstraction.** A PyTorch
`Dataset` is any object implementing `__len__` (how many examples exist)
and `__getitem__(i)` (fetch example `i`) — it says nothing about order or
batching. A `DataLoader` wraps a `Dataset` and is the actual iterator a
training loop consumes: each iteration it decides *which* indices to
fetch next (shuffled or not, per `shuffle=True/False`), groups them into
a batch of size `batch_size`, calls the `Dataset`'s `__getitem__` for
each index, and stacks the results into batch tensors. This mirrors
`08-mlops-deployment/07-cicd/notes.md`'s core idea — automating a
manual, error-prone, repeated-by-hand process (there: "run tests, build,
deploy" by hand each time; here: "shuffle, slice into batches, load each
item" by hand each epoch) into something that runs the same correct way
every time without a person re-implementing it.

**Checkpointing as periodic state serialization.** A checkpoint is a
snapshot, at one instant, of everything needed to resume training exactly
where it left off — not just the model's learned weights
(`model.state_dict()`), but also the optimizer's internal state
(`optimizer.state_dict()`, e.g. Adam's per-parameter momentum and
variance running averages) and bookkeeping like which epoch training
reached. `torch.save(...)` serializes this into one file at a chosen
cadence (e.g. every `N` epochs); `torch.load(...)` deserializes it back
into fresh objects. This connects directly to
[`08-mlops-deployment/04-model-packaging-versioning/notes.md`](../../08-mlops-deployment/04-model-packaging-versioning/notes.md)'s
content-addressing idea: a checkpoint file is exactly the kind of large,
opaque binary artifact that topic argues should be identified by a hash
of its content rather than a human-chosen filename like
`model_v2_final.pt` — the same "which file is actually the right one"
problem applies to a folder of `ckpt_epoch03.pt`, `ckpt_epoch06.pt`,
`ckpt_final_ACTUALLY.pt` as it does to a folder of manually-named model
files.

## Algorithm

**Manual batching/shuffling (from-scratch):**
1. Build an index list `[0, 1, ..., n-1]` for the `n` examples.
2. If shuffling, randomly reorder the index list (a fresh order each
   epoch).
3. Slice the (possibly reordered) index list into chunks of `batch_size`.
4. For each chunk, look up the corresponding data elements and yield them
   as one batch (the final chunk may be smaller than `batch_size` if `n`
   isn't evenly divisible).

**PyTorch `Dataset`/`DataLoader`:** identical shape, with `Dataset`
supplying `__getitem__`/`__len__` in place of raw list indexing, and
`DataLoader` internally performing steps 1–4 each time it's iterated.

**Checkpointed training loop, per epoch:**
1. Run the standard 5-step training loop (`02-nn-module-and-training-loop`)
   over every batch the `DataLoader` yields.
2. If `epoch % N == 0` (or it's the final epoch), serialize
   `{"epoch": epoch, "model_state_dict": model.state_dict(),
   "optimizer_state_dict": optimizer.state_dict(), "loss": current_loss}`
   to a file with `torch.save`.
3. To resume: construct a **fresh** model and optimizer, `torch.load` the
   checkpoint file, call `model.load_state_dict(...)` and
   `optimizer.load_state_dict(...)`, then continue the epoch loop starting
   at `checkpoint["epoch"] + 1`.

## From-scratch implementation

The companion notebook's first section (`manual_batch_generator`) is a
plain-Python generator — no `torch`, no `Dataset` class — that shuffles a
list of indices with `random.shuffle` and yields fixed-size slices, over
a toy list of 10 integers standing in for data points. It deliberately
implements only what `DataLoader` does at its core (shuffle indices,
slice into batches, handle a final partial batch); it does **not**
attempt parallel/background loading (`num_workers`) or pinned-memory GPU
transfer, since this environment has no GPU and a small in-memory toy
dataset needs no background workers — those are `DataLoader` features
this from-scratch version has no reason to reimplement.

## Practical implementation

The companion notebook (`03-datasets-dataloaders-checkpointing.ipynb`)
maps the from-scratch generator onto real PyTorch:

| From-scratch (`manual_batch_generator`) | PyTorch |
|---|---|
| Toy list `[0..9]` | `BreastCancerDataset` wrapping the 30-feature, 569-row Breast Cancer Wisconsin dataset |
| `list(range(n))` + `random.shuffle` | `DataLoader(..., shuffle=True)`'s internal index sampler |
| Manual slicing into chunks of `batch_size` | `DataLoader(..., batch_size=32)` |
| Plain Python indexing `data[i]` | `Dataset.__getitem__(i)` |

A small `MLP` (`nn.Linear(30, 16) → ReLU → nn.Linear(16, 1) → Sigmoid`) is
trained with `Adam` for 8 epochs (a stand-in for a training run that
later "crashes"), saving a checkpoint (model **and** optimizer state)
every 3 epochs via `torch.save`. A second part of the notebook then
constructs a **brand-new** model and optimizer — never touching the live
objects from the first run — and populates them *only* by `torch.load`-ing
the saved checkpoint file, exactly as a freshly started process resuming
a crashed job would have to. This is the honest way to test whether the
checkpoint actually contains everything needed: reusing the live
in-memory objects would not catch a checkpoint missing something.

## Experiment

**Hypothesis (stated before running):** if a checkpoint correctly saves
both model weights and optimizer state, then a model+optimizer
constructed entirely fresh and populated only from that checkpoint file
should (a) immediately reproduce the same loss the original run had at
that epoch, and (b) continue training with a loss curve that keeps
decreasing smoothly across the "crash" boundary, with no jump or spike.

**Setup:** train the `MLP` above for 8 epochs (Adam, `lr=1e-2`, batch size
32), checkpointing at epochs 3, 6, and 8; "simulate a crash" by simply
stopping there. In a fresh cell, construct new `MLP`/`Adam` objects, load
only from `checkpoints/ckpt_epoch08.pt`, verify the reloaded model's loss
matches the recorded epoch-8 loss, then resume training for epochs 9–16.

**Actual result (from the executed notebook, real output):**

```
[run 1] epoch  1  mean BCE loss = 0.3243
[run 1] epoch  2  mean BCE loss = 0.0997
[run 1] epoch  3  mean BCE loss = 0.0647   -> saved checkpoint: checkpoints/ckpt_epoch03.pt
[run 1] epoch  4  mean BCE loss = 0.0539
[run 1] epoch  5  mean BCE loss = 0.0468
[run 1] epoch  6  mean BCE loss = 0.0418   -> saved checkpoint: checkpoints/ckpt_epoch06.pt
[run 1] epoch  7  mean BCE loss = 0.0414
[run 1] epoch  8  mean BCE loss = 0.0370   -> saved checkpoint: checkpoints/ckpt_epoch08.pt

Loaded checkpoint from checkpoints/ckpt_epoch08.pt, saved loss = 0.0370
reloaded-model loss recomputed now = 0.0358  (run 1's recorded loss at epoch 8 = 0.0370)

[run 2 - resumed] epoch  9  mean BCE loss = 0.0332
[run 2 - resumed] epoch 10  mean BCE loss = 0.0321
[run 2 - resumed] epoch 11  mean BCE loss = 0.0341
[run 2 - resumed] epoch 12  mean BCE loss = 0.0328
[run 2 - resumed] epoch 13  mean BCE loss = 0.0365
[run 2 - resumed] epoch 14  mean BCE loss = 0.0312
[run 2 - resumed] epoch 15  mean BCE loss = 0.0305
[run 2 - resumed] epoch 16  mean BCE loss = 0.0320
```

**Interpretation:** the reloaded model's freshly recomputed loss (0.0358)
matches the originally recorded epoch-8 loss (0.0370) almost exactly —
confirming the deserialized weights are the same weights. The resumed
run's epoch-9 loss (0.0332) continues directly from epoch-8's 0.0370 with
no jump, and the loss keeps decreasing across the rest of the resumed
run — confirming the checkpoint restored everything needed to continue
training as if the process had never stopped.

**Limitations:** this experiment uses one small dataset, one small model,
and a short 8-then-8-epoch split; it demonstrates that checkpointing
*works* in this exact setup, not that every model/optimizer/PyTorch
version combination is checkpoint-compatible (see "Failure modes" below
for a documented exception).

## Failure modes

- **Forgetting optimizer state — demonstrated directly in the
  notebook.** A checkpoint holding only `model.state_dict()` restores the
  weights correctly but leaves the optimizer to start from scratch. For
  `Adam`, that means its per-parameter momentum and variance running
  averages reset to zero instead of continuing from where they were. The
  notebook reproduces this: reloading the same epoch-8 weights but into a
  **fresh, unrestored** `Adam` optimizer gives an epoch-9 loss of `0.0463`
  — visibly worse than the properly resumed run's `0.0332` — before
  settling back down over the next couple of epochs as Adam's state
  re-estimates itself. On a longer run or a more sensitive optimizer
  configuration, this kind of resume can show up as a much larger,
  destabilizing loss spike right at the resume point.
- **Checkpoint format incompatibility across PyTorch versions.** A
  `state_dict` is a dictionary of tensor names to tensor values; if a
  model's architecture changes (a renamed layer, an added layer) between
  saving and loading, `load_state_dict` raises a key-mismatch error rather
  than silently loading the wrong weights into the wrong layer —
  generally a safe failure, but it means checkpoints are tied to the
  *exact* architecture that produced them, not just "a similar model."
  Major PyTorch version upgrades can also change serialization internals;
  pinning the PyTorch version used to load a checkpoint to the version
  that saved it (or re-saving on upgrade) avoids this.
- **Not shuffling causes overfitting to batch order.** If `DataLoader` is
  constructed with `shuffle=False` (or the from-scratch generator is
  called with `shuffle=False` every epoch), the model sees examples in
  the exact same sequence every single epoch. Beyond the usual "some
  optimizers benefit from randomized example order" argument, a model can
  learn spurious patterns tied to *position within the epoch* rather than
  the actual features — particularly risky if the underlying data
  happens to be sorted by label or by time.

## Real-world usage

Every production PyTorch training job — vision models on ImageNet-scale
datasets, language models on web-scale text corpora — uses this exact
`Dataset`/`DataLoader` split, usually with `num_workers > 0` for
background parallel loading and lazy (streaming) `__getitem__`
implementations that read one file/row from disk per call rather than
holding the dataset in memory. Checkpointing is likewise universal: cloud
training jobs on preemptible/spot instances are specifically designed
around frequent checkpointing (the instance can be reclaimed at any
moment), and every major training framework (PyTorch Lightning, Hugging
Face `Trainer`, DeepSpeed) has checkpointing as a first-class,
configurable feature for exactly the reasons this topic covers.

## Mental model

`Dataset` is the librarian who can fetch item `i` on request and knows
how many items exist; `DataLoader` is the reading plan that decides which
items, batched and shuffled, to request each iteration — splitting "how
to fetch one item" from "what order to consume items in" cleanly.
Checkpointing is autosave for a training run: periodically writing down
everything needed to pick up exactly where training left off (weights
*and* optimizer state, not just weights), so a crash costs only the work
since the last save, not the whole run.

## Questions to think about

1. The from-scratch `manual_batch_generator` re-shuffles the *index* list
   each call, never the underlying `data` list itself. Why does shuffling
   indices rather than the data in place matter, especially once the data
   is something you can't cheaply copy (e.g. a dataset streamed from
   disk)?
2. The notebook's "bad resume" experiment reloaded model weights but used
   a fresh `Adam` optimizer. If the optimizer had been plain `SGD`
   (`02-nn-module-and-training-loop/notes.md`'s comparison optimizer)
   instead of `Adam`, would you expect the same kind of resume
   instability? Why or why not, in terms of what state each optimizer
   actually carries between steps.
3. A checkpoint is saved every `N=3` epochs, not every epoch. What is the
   tradeoff being made by picking a larger vs. smaller `N` — think in
   terms of disk usage, I/O time spent checkpointing vs. training, and
   how much progress could be lost on a crash between saves.
4. `Dataset.__getitem__` in the notebook returns already-loaded in-memory
   tensors (`self.X[idx]`). Sketch how `__getitem__` would need to change
   for a dataset of a million image files on disk, where loading all
   images into `__init__` up front is not possible — what does
   `__getitem__` do differently, and does anything about `DataLoader`'s
   usage from the training loop's point of view need to change at all?
5. The "reloaded-model loss recomputed now" check in the notebook compares
   `0.0358` (recomputed) against `0.0370` (originally recorded). These
   are close but not bit-identical. List at least two legitimate reasons
   they could differ even though the exact same weights were used both
   times, without concluding the checkpoint is broken.
