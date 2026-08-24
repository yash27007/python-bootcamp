# 11 – Generative AI

Two families of generative model, first-principles, at toy scale (per this repo's no-heavy-training
constraint for sections 11-15): the adversarial game GANs play, and the alternative diffusion
takes — training a much simpler noise-prediction objective instead of an unstable minimax game.
Both trained on the same toy 2D dataset for a direct comparison.

| # | Topic | Status | Description |
|---|-------|--------|--------------|
| 01 | [GANs](./01-gans/) | ✅ Complete | Adversarial min-max objective derived, a real trained generator/discriminator on an 8-mode ring, a concretely reproduced mode-collapse failure |
| 02 | [Diffusion Models](./02-diffusion-models/) | ✅ Complete | Forward/reverse diffusion process derived, a real trained DDPM on the same dataset, a measured too-few-vs-too-many-steps quality/speed tradeoff |

## Prerequisites

- `09-pytorch/02-nn-module-and-training-loop` — both topics cite this training-loop pattern
  rather than re-deriving it.

## Environment note

Every training run in this section is genuinely toy-scale (seconds, not minutes) per this repo's
binding "no heavy training" constraint for sections 11-15. `torch.set_num_threads(1)` is set
early in both notebooks to avoid a real CPU thread-oversubscription slowdown observed on tiny
matrix ops in this environment.

## What's next

`12-reinforcement-learning` onward continue the toy-scale discipline established here.
