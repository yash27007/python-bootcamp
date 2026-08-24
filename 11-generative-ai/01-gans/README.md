# 01 – GANs (Generative Adversarial Networks)

Detailed notes (the reconstruction-error failure mode, the adversarial min-max objective derived,
Nash-equilibrium framing, mode collapse explained mathematically): [notes.md](notes.md)

Real, actually-executed, toy-scale GAN trained on an 8-mode 2D ring — before/after generated-sample
plots, real loss curves, and a concrete mode-collapse demo, all with real pasted output:
[01-gans.ipynb](01-gans.ipynb)

## What you'll learn

Why "generate a new, realistic sample from a distribution you only have examples of" is a different
problem than anything solved through `09-pytorch` — and why the natural first attempt
(an autoencoder trained on reconstruction error) produces blurry, off-distribution averages instead
of sharp, diverse samples. The adversarial min-max objective
$\min_G \max_D \mathbb{E}[\log D(x)] + \mathbb{E}[\log(1-D(G(z)))]$, derived from first principles,
its Nash-equilibrium interpretation, and exactly why and how mode collapse happens mathematically.

| Topic | Status |
|-------|--------|
| Problem: generating new samples from an unknown distribution | ✅ Complete |
| Why autoencoder reconstruction-error training produces blurry averages | ✅ Complete |
| Adversarial min-max objective, Nash-equilibrium framing, mode collapse (derived) | ✅ Complete |
| Real tiny GAN (`nn.Module` generator + discriminator) trained on a toy 2D ring | ✅ Complete |
| Real before/after generated-sample comparison | ✅ Complete |
| Real mode-collapse demo (imbalanced G:D training) | ✅ Complete |
| Training-instability / oscillating-loss and vanishing-gradient failure modes | ✅ Complete |

## Why it matters

Every prior topic trained a model against a fixed, known target (a label, a numeric value) using a
single descending loss. GANs are the first topic where the training signal itself is produced by a
second, simultaneously-training model — the point where "loss going down" stops being a reliable
signal that training is working, and where a wholly new failure mode (mode collapse) becomes
possible. Understanding this mechanism at toy scale, where every plot can be eyeballed against a
known ground-truth shape (the 8-mode ring), is what makes real GAN training runs — where the target
distribution is invisible and only sample quality can be judged — interpretable at all.

## Prerequisites

- `09-pytorch/02-nn-module-and-training-loop` — this topic reuses that topic's `nn.Module` +
  5-step training loop pattern directly (forward → loss → `.backward()` → `optimizer.step()` →
  `zero_grad()`), applied to two networks instead of one.
- `06-deep-learning/01-ann` — binary cross-entropy loss and MLP forward equations, used unchanged
  for both the generator's and discriminator's loss.

## What you'll build

- A toy 2D "8-mode Gaussian ring" dataset (fixed seed, no download) — a standard benchmark for
  eyeballing GAN mode coverage.
- A small `nn.Module` generator (noise → 2D point) and discriminator (2D point → real/fake
  probability), trained with two separate optimizers in an alternating loop for 2000 iterations
  (~3.5s on CPU).
- Real before/after visualizations: generated points before training (a tight, arbitrary blob) vs.
  after training (spread comparable in scale to the real ring, `std=[1.70, 0.89]` vs. real
  `std=[1.44, 1.39]`, up from an untrained `std=[0.07, 0.03]`).
- A deliberate, concrete mode-collapse run (imbalanced generator:discriminator update schedule)
  producing a measured `std=[0.0038, 0.0014]` — roughly 1000x less spread than real data — directly
  demonstrating the failure mode `notes.md` derives mathematically.

## Where it appears in real systems

The exact two-network, alternating-optimizer mechanism built here — scaled up with convolutional
architectures and stabilization tricks — underlies DCGAN, StyleGAN (photorealistic face synthesis),
CycleGAN (unpaired image-to-image translation), and GAN-based synthetic data augmentation for
domains with scarce labeled data (e.g. medical imaging).

## What's next

`02-diffusion-models` — motivated directly by this topic's training-instability failure mode: is
there a way to train a generative model with a simpler, more stable objective than an adversarial
game? Diffusion models answer yes, trained on the same toy 2D ring for a direct before/after
comparison against this topic's GAN.
