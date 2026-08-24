# Phase 9: Generative AI First-Principles Build-Out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** New `11-generative-ai/` section: GANs and diffusion models, first-principles, toy-scale throughout (owner's binding "no heavy training" constraint — see AGENTS.md).

**Architecture:** 2 topics, 1 task each (small section). Both bridge to `09-pytorch`'s `nn.Module`/training-loop foundation rather than re-deriving it.

**Tech Stack:** PyTorch (already installed, CPU-only). Tiny models, tiny/synthetic data, few epochs — the point is the mechanism, not sample quality.

**Spec:** `docs/superpowers/specs/2026-08-23-first-principles-curriculum-design.md`, `AGENTS.md`.

## Global Constraints

- **Binding: no heavy/long-running training.** Every training run in this section must complete in well under a few minutes on CPU. If a genuinely useful demo can't run at toy scale, write the real code, review it carefully, mark it honestly unexecuted — never fabricate.
- 12-section notes.md template. Real math throughout — both topics have real math (adversarial min-max objective, the diffusion forward/reverse process) — no Conceptual-foundation substitution needed here.
- Bridge to `09-pytorch/02-nn-module-and-training-loop` rather than re-deriving the training loop.
- Review level: light.

---

### Task 1: GANs

**Files:** Create `11-generative-ai/01-gans/` (README.md, notes.md, notebook)

**Content:** Problem = generating new, realistic-looking samples from a data distribution you only have examples of, not a closed-form description of. Why-simpler-fails = a model trained to minimize reconstruction error (an autoencoder) tends to produce blurry averages, not sharp, diverse samples. Mathematical foundation = the adversarial min-max objective $\min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$, derive the intuition (D learns to distinguish real from fake, G learns to fool D, the Nash-equilibrium framing), the mode-collapse failure mode explained mathematically (G finding one point that fools D and no incentive to diversify). From-scratch = cite `09-pytorch/02-nn-module-and-training-loop`'s training loop, then implement a REAL tiny GAN (a 2-3 layer generator + discriminator, both `nn.Module`) trained on a toy 2D distribution (e.g. a 2D Gaussian mixture or a simple shape like a circle) — actually train it for a small number of epochs, visualize generated points vs real points before/after training, real output. Experiment = hypothesis that generated points should approach the real distribution's shape after training — actually run, real before/after plots. Failure modes = mode collapse (demonstrate it concretely if feasible at toy scale — e.g. an undertrained/badly-tuned run collapsing to one point, contrasted with a working run), training instability (G/D loss oscillating rather than converging), vanishing gradients when D gets too good too fast. Real-world, Mental model, Questions.

- [ ] Write notes.md + notebook (real, toy-scale, fast). README. `git commit -m "Phase 9 Task 1: first-principles build-out — GANs"`.

### Task 2: Diffusion Models

**Files:** Create `11-generative-ai/02-diffusion-models/` (README.md, notes.md, notebook)

**Content:** Problem = GANs' adversarial training is notoriously unstable (cite Task 1's failure modes directly) — is there a way to train a generative model with a more stable, simpler objective? Why-simpler-fails = cite GAN instability explicitly. Mathematical foundation = the forward diffusion process (gradually adding Gaussian noise over T steps, derive the closed-form $q(x_t|x_0)$ using the reparameterization trick), the reverse process (a model learns to predict/remove noise at each step), the simplified training objective (predict the noise, not the full distribution — much more stable than GAN's adversarial game). From-scratch = a REAL, small implementation: forward-diffuse a toy 2D dataset (same one from Task 1, for direct comparison) over a small number of steps, train a tiny denoising network (`nn.Module`, cite Task 1's/`09-pytorch`'s training loop) to predict the added noise, actually run a small number of training steps, then run the reverse (sampling) process starting from pure noise and show it approaches the real data distribution — actually run, real before/after visualization, keep T (number of diffusion steps) and epochs small enough to finish in well under a few minutes. Experiment = hypothesis that the reverse process's output distribution approaches the training distribution, measured (e.g. by comparing generated-sample statistics to real-sample statistics), actually run. Failure modes = too few diffusion steps making denoising too hard per step (worse samples), too many steps making sampling slow (a real inference-time cost tradeoff, explain it), the classic tradeoff between sample quality and sampling speed that motivates later work (DDIM, distillation — mention by name, don't implement). Real-world, Mental model, Questions.

- [ ] Write notes.md + notebook (real, toy-scale, fast). README. `git commit -m "Phase 9 Task 2: first-principles build-out — diffusion models"`.

### Task 3: Section/root README

- [ ] Create `11-generative-ai/README.md` (both topics, ✅ Complete). Update root `README.md` (Curriculum table, roadmap, section blurb) per the established Phase 3-6 precedent. `git commit -m "Phase 9 Task 3: mark 11-generative-ai complete in section and root README"`.

## Verification

```bash
cd /home/yashwanth-aravind/ml-course/python-bootcamp
.venv/bin/python -c "
import pathlib
for t in sorted(pathlib.Path('11-generative-ai').iterdir()):
    if t.is_dir(): print(t.name, (t/'notes.md').exists(), (t/'README.md').exists())
"
```
