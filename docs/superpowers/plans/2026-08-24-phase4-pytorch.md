# Phase 4: PyTorch First-Principles Build-Out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a new `09-pytorch/` section teaching PyTorch by mapping every construct back to something already built from scratch in NumPy (Phase 1) or used via Keras (Phase 2) — tensors→NumPy arrays, autograd→manual backprop, `nn.Module`→the from-scratch MLP, `optimizer.step()`→manual gradient descent — so PyTorch reads as "the same ideas, a faster/more general engine," not a new framework to memorize.

**Architecture:** 4 topics, each following the 12-section AGENTS.md template. Unlike Phase 3, `09-pytorch`'s "from-scratch" step is usually already done — it lives in Phase 1's `06-deep-learning/01-ann` (manual MLP/backprop) and elsewhere — so most topics' From-scratch sections **cite** that prior work rather than re-deriving it, and spend their weight on the explicit NumPy↔PyTorch / manual-update↔optimizer mapping. 2 tasks: (1) tensors+autograd, nn.Module+training loop; (2) datasets/dataloaders/checkpointing, GPU/mixed-precision/profiling.

**Tech Stack:** PyTorch (CPU-only — no GPU available in this environment; GPU/mixed-precision content is written and reviewed but explicitly marked "not executed in this environment," same honesty discipline as Phase 3's un-executed Docker/DVC content), `.venv` (uv-managed).

**Spec:** `docs/superpowers/specs/2026-08-23-first-principles-curriculum-design.md` and `AGENTS.md` — read both first.

## Global Constraints

- `.venv/bin/python -c "import torch"` currently fails — PyTorch is not installed. Task 1 must `uv add torch` (CPU wheel; do not attempt a CUDA build, no GPU present).
- Every `notes.md`: 12-section template, `##` headers exactly as in AGENTS.md. Problem / Why-simpler-fails / Mathematical foundation / Mental model / Questions never skipped.
- From-scratch sections that cite prior work (e.g. `06-deep-learning/01-ann/ann-from-scratch-xor.ipynb`) must link to it by relative path and briefly restate the connecting idea — never just "see that other file" with no bridge.
- Practical implementation: real PyTorch code, actually executed, real output pasted — never fabricated. GPU/mixed-precision-specific code is the one exception: write and review it carefully, but mark explicitly as untested here (no GPU), same pattern as Phase 3.
- Datasets: reuse sklearn/keras built-ins already used elsewhere in this repo (Iris, breast cancer, fashion_mnist, IMDB) — no new downloads.
- Keep all training runs small (CPU, seconds-to-low-minutes) — this is a teaching demo, not a benchmark, same discipline as Phase 2's DL notebooks.
- Every topic gets an orientation-format README, ✅ Complete status.
- Commit granularity: one commit per task.
- Review level: **light** — a narrower reviewer pass (or, for small/mechanical fixes, direct controller verification) rather than the full deep-review process used in Phases 1-3's initial passes.

---

### Task 1: Tensors + Autograd, nn.Module + Training Loop

**Files:**
- Create: `09-pytorch/01-tensors-and-autograd/` (README.md, notes.md, notebook)
- Create: `09-pytorch/02-nn-module-and-training-loop/` (README.md, notes.md, notebook)

**Content requirements:**

- **`01-tensors-and-autograd`**: Problem = NumPy arrays don't track how they were computed, so getting a gradient requires deriving and coding backprop by hand for every new architecture (exactly what Phase 1's from-scratch MLP had to do). Why-simpler-fails = manual backprop doesn't scale past a couple of layers/architectures — cite the actual pain point from `06-deep-learning/01-ann/notes.md`'s from-scratch backprop derivation. Math = automatic differentiation as building a computation graph and applying the chain rule mechanically (reverse-mode AD) — derive why reverse-mode is efficient for the many-inputs-few-outputs case typical in ML (vs forward-mode). From-scratch = cite the existing manual backprop (`06-deep-learning/01-ann/ann-from-scratch-xor.ipynb`) as "this is what autograd automates" — don't re-derive, bridge to it explicitly. Practical = real PyTorch: `torch.Tensor` vs `np.ndarray` (same data, `requires_grad=True`), `.backward()`, inspect `.grad`, compare the autograd-computed gradient numerically against the from-scratch manual gradient on the same tiny example (`np.allclose`) — actually run this cross-check. Experiment = the cross-check above, hypothesis-first. Failure modes = forgetting `.zero_grad()` (accumulating gradients across steps), in-place ops breaking the graph, `.detach()`/`.item()` confusion. Real-world, Mental model, Questions.
- **`02-nn-module-and-training-loop`**: Problem = manually tracking every weight tensor and writing the update rule by hand (as the from-scratch MLP did) doesn't scale to real architectures with many layers. Why-simpler-fails = cite the from-scratch MLP's explicit weight-list management. Math/Conceptual = `nn.Module` as a structured container for parameters + a `forward()` method; the training-loop pattern (forward → loss → `loss.backward()` → `optimizer.step()` → `zero_grad()`) as the automated version of the from-scratch MLP's explicit loop. From-scratch = cite the from-scratch MLP again, bridge explicitly: "each of these 4 lines replaces one manual step you wrote by hand there." Practical = real PyTorch: define a small `nn.Module` (2-3 linear layers), train it on the same XOR-style toy problem the from-scratch demo used (or a slightly richer one, e.g. `sklearn`'s `make_moons`), actually train, plot the loss curve, real output. Experiment = compare `optimizer.step()`'s update rule (e.g. plain SGD) against the from-scratch MLP's manual gradient-descent update on the same problem — hypothesis that both converge to a similar loss, actually run and confirm. Failure modes = wrong loss/model-output-shape mismatches, forgetting `model.eval()`/`model.train()` mode switching, learning-rate sensitivity. Real-world, Mental model, Questions.

- [ ] **Step 1:** `uv add torch`; confirm `.venv/bin/python -c "import torch; print(torch.__version__)"` works.
- [ ] **Step 2:** Read `06-deep-learning/01-ann/notes.md` and `ann-from-scratch-xor.ipynb` to get the exact from-scratch derivation to bridge from.
- [ ] **Step 3:** Write both notes.md per the content requirements.
- [ ] **Step 4:** Write and execute both notebooks (real PyTorch code, real output, the autograd-vs-manual-gradient cross-check, the optimizer-vs-manual-update comparison).
- [ ] **Step 5:** Write both topic READMEs in orientation format, ✅ Complete.
- [ ] **Step 6:** `git add` both topic folders + `pyproject.toml`/`uv.lock`, commit: `git commit -m "Phase 4 Task 1: first-principles build-out — PyTorch tensors/autograd, nn.Module/training loop"`.

---

### Task 2: Datasets/DataLoaders/Checkpointing, GPU/Mixed-Precision/Profiling

**Files:**
- Create: `09-pytorch/03-datasets-dataloaders-checkpointing/` (README.md, notes.md, notebook)
- Create: `09-pytorch/04-gpu-mixed-precision-profiling/` (README.md, notes.md, notebook)

**Content requirements:**

- **`03-datasets-dataloaders-checkpointing`**: Problem = real training data doesn't fit in memory as one array, and a crashed training run loses all progress. Why-simpler-fails = loading a whole dataset into one NumPy array (as every Phase 1/2 notebook so far has done, since the toy datasets are small) breaks down at real scale; a training loop with no checkpointing has to restart from scratch on any interruption. Conceptual foundation = `Dataset`/`DataLoader` as an iterator abstraction over batches (with shuffling, batching, and optionally lazy loading) — connects to Phase 3's `07-cicd` idea of automating a manual process; checkpointing as periodically serializing model+optimizer state (connects to Phase 3's `04-model-packaging-versioning` content-addressing idea — cite it). From-scratch = a small manual batching/shuffling generator in plain Python (no `DataLoader`) over a toy dataset, to show what `DataLoader` automates. Practical = real PyTorch `Dataset`+`DataLoader` on a small dataset (e.g. breast cancer or a small synthetic set), a real training loop that saves a checkpoint (`torch.save`) every N epochs, actually run, then actually reload the checkpoint and resume/verify it continues correctly (real output both times). Experiment = interrupt-and-resume: train partway, checkpoint, "simulate a crash" (just don't continue in the same process), reload in a fresh script, resume, hypothesis that the reloaded model's loss picks up where it left off — actually run and confirm. Failure modes = forgetting to save optimizer state (not just model weights) breaking resumed training dynamics, checkpoint format incompatibility across PyTorch versions, not shuffling causing overfitting to batch order.
- **`04-gpu-mixed-precision-profiling`**: Problem = CPU training is too slow for real model sizes; naive full-precision (fp32) training wastes memory/time GPUs can save on. Why-simpler-fails = cite this repo's own experience — every Keras/PyTorch notebook so far ran on CPU because no GPU is available in this environment, which is fine for toy demos but doesn't scale. Conceptual foundation = what a GPU actually parallelizes (many small matrix-multiply-accumulate units vs a CPU's few powerful cores), why mixed precision (fp16/bf16 compute with fp32 master weights) speeds things up without destroying numerical stability — explain the mechanism, not just "it's faster." From-scratch = N/A or minimal — this is inherently a hardware/systems topic; document that choice explicitly. Practical = write REAL, CORRECT PyTorch code for `.to("cuda")`/`torch.autocast`/`torch.cuda.amp.GradScaler` and `torch.profiler` — carefully reviewed for correctness, but **explicitly marked "written and reviewed, not executed in this environment" since no GPU is available** (verify this claim first: `.venv/bin/python -c "import torch; print(torch.cuda.is_available())"`), same honesty discipline as Phase 3's un-executed Dockerfile. DO run the CPU-only parts (a `torch.profiler` CPU trace on the Task 2/Task 1 training loop is genuinely runnable without a GPU — do this for real). Experiment = the CPU profiler trace, hypothesis about which op dominates time, actually run and confirm. Failure modes = mixed-precision overflow/underflow without a loss scaler, profiler overhead skewing the very measurement it's taking, assuming GPU speedup without profiling first (premature optimization). Real-world, Mental model, Questions.

- [ ] **Step 1:** Confirm `torch.cuda.is_available()` is `False` in this environment (expected) — don't re-check assumptions, just confirm once and note it.
- [ ] **Step 2:** Write both notes.md per the content requirements.
- [ ] **Step 3:** Write and execute the checkpointing notebook (real save/reload/resume) and the CPU-portion of the profiling notebook (real `torch.profiler` trace); write the GPU/mixed-precision code carefully, mark honestly as unexecuted.
- [ ] **Step 4:** Write both topic READMEs in orientation format, ✅ Complete.
- [ ] **Step 5:** `git add` both topic folders, commit: `git commit -m "Phase 4 Task 2: first-principles build-out — PyTorch datasets/checkpointing, GPU/mixed-precision/profiling"`.

---

## Verification (after both tasks)

```bash
cd /home/yashwanth-aravind/ml-course/python-bootcamp
.venv/bin/python - <<'EOF'
import pathlib
for topic in sorted(pathlib.Path("09-pytorch").iterdir()):
    if not topic.is_dir(): continue
    print(topic.name, "notes.md" if (topic/"notes.md").exists() else "MISSING", "README.md" if (topic/"README.md").exists() else "MISSING")
EOF
```
