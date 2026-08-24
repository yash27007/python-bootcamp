# 13 – LLMs From Scratch

Tokenizer → pretraining objective → instruction tuning, first-principles, toy scale
throughout. Builds directly on `06-deep-learning/05-attention-transformers`'s from-scratch
self-attention and `07-nlp/05-transformers-and-huggingface`'s tokenization/fine-tuning
content — this section's job is specifically the *pretraining* objective and
instruction-tuning distinction those two didn't cover.

| # | Topic | Status | Description |
|---|-------|--------|--------------|
| 01 | [Tokenizer From Scratch](./01-tokenizer-from-scratch/) | ✅ Complete | Byte-Pair Encoding derived and trained from scratch in plain Python, real encode/decode round-trip, real vocabulary-size-vs-tokens-per-word experiment, real wrong-domain failure demo |
| 02 | [Pretraining Objective](./02-pretraining-objective/) | ✅ Complete | Causal self-attention derived, a real 21k-parameter TinyGPT actually trained (300 steps, loss 3.68→1.74) and actually sampled from, honestly nonsensical output explained by capacity/data/steps and connected to Chinchilla/Kaplan scaling laws |
| 03 | [Instruction Tuning](./03-instruction-tuning/) | ✅ Complete | Task 02's TinyGPT fine-tuned on 24 hand-written (instruction, response) pairs, real before/after generation comparison, a real observed catastrophic-forgetting artifact tied to `07-nlp/05`'s failure mode, RLHF/DPO named and marked Planned |

## Prerequisites

- `06-deep-learning/05-attention-transformers` — Topic 02 extends its from-scratch
  scaled-dot-product attention with a causal mask rather than re-deriving attention from
  zero.
- `07-nlp/05-transformers-and-huggingface` — Topic 01 derives the BPE algorithm that
  topic's pretrained tokenizer used without building; Topic 03 connects instruction tuning
  to that topic's fine-tuning and catastrophic-forgetting content directly.
- `09-pytorch/02-nn-module-and-training-loop` — Topics 02 and 03's training loops use this
  pattern.

## Environment note

Every training run in this section is toy-scale (seconds, not minutes) —
`torch.set_num_threads(1)` throughout, the same fix Phases 9-10 needed for tiny matrix ops
on this environment's CPU. This section is explicit throughout that its models are toy
mechanism demonstrations, not usable language models: real LLM pretraining at production
scale is a Planned direction, not something this repo teaches by doing it for real.

## What's next

`14-multi-agent-systems` and `15-agent-skills-and-mcp` continue the toy-scale discipline,
building on this section's language-model mechanics for agentic and tool-use systems.
