# 02 – Pretraining Objective: A Tiny GPT-Style Decoder

Detailed notes (causal self-attention derived from
`06-deep-learning/05-attention-transformers`'s scaled dot-product attention, the
autoregressive language-modeling objective derived from the chain rule of probability,
and an honest discussion of what a toy-scale model can and cannot do):
[notes.md](notes.md)

Real, actually-executed, from-scratch tiny GPT training (PyTorch, `torch.set_num_threads(1)`)
— a real causal-mask sanity check, a real 300-step training loss curve, real measured
validation perplexity against a uniform-random baseline, and real sampled text generation
from the trained model, all with real pasted output:
[001_tiny_gpt_pretraining.ipynb](001_tiny_gpt_pretraining.ipynb)

## What you'll learn

What changes, mechanically and mathematically, when you take
`06-deep-learning/05-attention-transformers`'s self-attention and Transformer encoder
block and turn them into the *decoder-only* architecture GPT-style models pretrain with:
a causal (triangular) mask on self-attention, derived to show exactly why it prevents
future-information leakage and makes next-token prediction well-defined, and the
autoregressive language-modeling objective itself — cross-entropy on next-token
prediction, derived from the chain rule of probability
($P(x_1,\ldots,x_T) = \prod_t P(x_t \mid x_{<t})$). Then: build a genuinely tiny
decoder-only Transformer, actually train it, and actually generate text from it — with an
honest account of why the output is nonsensical at this scale, and why that's the expected
result, not a bug.

## Why it matters

`06-deep-learning/05-attention-transformers` never generated a single token — it built an
encoder for classification. `13-llms-from-scratch/01-tokenizer-from-scratch` built the
vocabulary a language model would need, but never trained anything on it. This topic is
the first in the course to actually run the training loop every modern LLM (GPT, Llama,
Claude's own model family) is pretrained with, even if at drastically smaller scale — and
to be explicit about exactly which parts of that gap (capacity, data, steps) are
responsible for the difference between this notebook's output and a real model's.

## Prerequisites

- `06-deep-learning/05-attention-transformers` — this topic extends that topic's
  from-scratch scaled dot-product attention directly; it does not re-derive attention from
  zero.
- `13-llms-from-scratch/01-tokenizer-from-scratch` — read for contrast; this topic
  deliberately uses a simpler character-level tokenizer instead, and explains why.
- Basic PyTorch (`nn.Module`, `nn.Linear`, `nn.Embedding`, autograd, an optimizer step
  loop) — no new framework concepts beyond what `06-deep-learning` already used.

## What you'll build

- A `CausalSelfAttention` PyTorch module: `06-deep-learning/05-attention-transformers`'s
  exact scaled-dot-product formula, extended with a lower-triangular mask — verified with
  a standalone sanity check that query position 3 attends to positions 0–3 and exactly 0
  weight to positions 4–5 (the future).
- A `TinyGPT` decoder-only model (2 layers, 32-dim embeddings, 2 heads, 21,093 parameters
  total — about 5,900x smaller than GPT-2 small) built from that module.
- A real training run: 300 AdamW steps on a 1,486-character hand-written corpus,
  `torch.set_num_threads(1)`, **1.90 seconds** wall-clock, training loss **3.68 → 1.74**.
- A real measured experiment: validation perplexity **13.91** vs. a **37**-way
  uniform-random baseline, and a before/after next-character-prediction confidence probe
  (0.063 → 0.179 on the top prediction).
- Real sampled text generation from the trained model — honestly nonsensical, with the
  reasons why (capacity/data/steps, connected to Chinchilla/Kaplan-style scaling laws)
  spelled out in notes.md's "Failure modes."

## Where it appears in real systems

This is literally the pretraining stage of every GPT-family LLM in production use today —
same causal-mask mechanism, same cross-entropy-on-next-token objective — run at a scale of
billions of parameters, trillions of tokens, and enormous compute instead of this
notebook's toy numbers. Perplexity, measured here as a sanity check, is a standard
benchmark metric reported for real language models. The resulting model type — a raw,
next-token-predicting completer — is also the direct prerequisite for
`13-llms-from-scratch/03-instruction-tuning`: you cannot fine-tune a model to follow
instructions until you have a pretrained model to fine-tune.

## What's next

`03-instruction-tuning` — takes a pretrained, raw-completion model (this topic's `TinyGPT`,
or a similarly tiny model) and fine-tunes it on (instruction, response) pairs, showing a
real before/after comparison between rambling completion and structured, instruction-aware
generation.
