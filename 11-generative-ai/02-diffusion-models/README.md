# 02 – Diffusion Models

Detailed notes (the forward diffusion process derived via the reparameterization trick, the reverse
process, the simplified noise-prediction objective, why it sidesteps GAN instability): [notes.md](notes.md)

Real, actually-executed, toy-scale DDPM trained on the **same 8-mode 2D ring** dataset as
`01-gans` — real forward-diffusion visualization, a real training run with a monotonic loss curve,
real reverse sampling from pure noise, and a real too-few-vs-too-many-steps tradeoff experiment, all
with real pasted output: [02-diffusion-models.ipynb](02-diffusion-models.ipynb)

## What you'll learn

Why `01-gans`'s adversarial min-max game — with its documented oscillating losses and mode
collapse — isn't the only way to solve the generation problem, and how a single ordinary-regression
network trained to predict per-step noise sidesteps both GAN instability and the naive-autoencoder
blurry-average failure at once. The forward diffusion process $q(x_t\mid x_0)$ derived in closed
form via the reparameterization trick, the reverse (denoising) process, the simplified
noise-prediction training objective $\mathbb E[\lVert\epsilon-\epsilon_\theta(x_t,t)\rVert^2]$, and
a real, measured quality/speed tradeoff in the number of diffusion steps $T$.

| Topic | Status |
|-------|--------|
| Problem: is adversarial training necessary, given `01-gans`'s documented instability? | ✅ Complete |
| Why diffusion avoids both GAN instability and the autoencoder's blurry-average failure | ✅ Complete |
| Forward process closed form $q(x_t\mid x_0)$, derived via the reparameterization trick | ✅ Complete |
| Reverse process and the simplified noise-prediction objective (derived) | ✅ Complete |
| Real tiny denoising `nn.Module`, trained on the same toy 2D ring as `01-gans` | ✅ Complete |
| Real forward-diffusion visualization (ring dissolving into noise over `T=40` steps) | ✅ Complete |
| Real reverse sampling from pure noise, before/after statistics measured | ✅ Complete |
| Real too-few vs. too-many diffusion-steps tradeoff, both directions actually trained | ✅ Complete |

## Why it matters

`01-gans` proved adversarial training works but is unstable by construction (a two-player game with
no single descending loss, prone to mode collapse). Diffusion models are the direct answer to "can
we do better": an ordinary single-network regression objective, with a loss curve that behaves like
every earlier topic's supervised loss instead of oscillating. Seeing both approaches trained on the
*identical* toy dataset, in adjacent notebooks, with directly comparable measured statistics, is what
makes the stability difference concrete rather than assumed — and understanding the reverse
process's inherent sequential-step cost here is what makes later production speedups (DDIM,
distillation) intelligible as solving a real, measured problem rather than an arbitrary optimization.

## Prerequisites

- `01-gans` — this topic is framed as a direct answer to `01-gans/notes.md`'s documented instability
  and mode-collapse failure modes, and reuses its exact toy 2D dataset for comparison.
- `09-pytorch/02-nn-module-and-training-loop` — the denoising network's training loop is that
  topic's unmodified 5-step pattern (forward → loss → `.backward()` → `optimizer.step()` →
  `zero_grad()`), applied to a single network instead of GANs' two.

## What you'll build

- The same toy 2D "8-mode Gaussian ring" dataset as `01-gans` (identical seed-0 `sample_real`), for
  direct before/after and cross-topic comparison.
- A closed-form forward diffusion process (`q_sample`), derived via the reparameterization trick,
  visualized dissolving the ring into pure noise over `T=40` steps.
- A tiny `nn.Module` noise-prediction network, trained for real (3000 steps, ~1.4s on CPU) with an
  ordinary MSE loss that decreases roughly monotonically — unlike `01-gans`'s oscillating losses.
- A real reverse (ancestral) sampling loop, run from literal Gaussian noise, producing samples whose
  mean/std (`[-0.018, 0.013]` / `[1.413, 1.406]`) land within a few percent of the real data's
  (`[-0.023, 0.062]` / `[1.436, 1.392]`) — and whose nearest-mode distance drops ~2 orders of
  magnitude (`16.36` → `0.31`) from an untrained model's samples.
- A real too-few-steps-vs-too-many-steps experiment: `T=5` measurably worse quality (nearest-mode
  distance `0.65` vs. `0.31` for `T=40`); `T=150` comparable quality but ~3.8x slower sampling
  (`0.0141s` vs. `0.0037s` for 500 points) — the concrete tradeoff motivating DDIM and distillation.

## Where it appears in real systems

The forward-diffuse/learn-to-reverse mechanism built here — scaled up with U-Net architectures, text
conditioning, and far more training — is exactly what powers DDPM, Stable Diffusion, Imagen, and
DALL-E's diffusion-based decoders. The step-count quality/speed tradeoff measured here is the direct
motivation for DDIM and distillation-based fast samplers used in every production system that needs
diffusion generation at interactive latency.

## What's next

This is the last currently-built topic in `11-generative-ai`; see the section [README.md](../README.md)
for the section-level overview of both topics.
