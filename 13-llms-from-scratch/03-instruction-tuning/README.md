# 03 – Instruction Tuning: From "Complete This Text" to "Follow This Request"

Detailed notes (instruction tuning as supervised fine-tuning on the same next-token
cross-entropy objective, reformatted around a structured prompt template; the explicit
mechanism connection to `07-nlp/05-transformers-and-huggingface`'s fine-tuning topic; and
RLHF/DPO named as the honestly-Planned next step beyond this topic's scope):
[notes.md](notes.md)

Real, actually-executed, from-scratch instruction tuning of a freshly pretrained tiny GPT
(PyTorch, `torch.set_num_threads(1)`) — a real pretraining stage, a real instruction
fine-tuning stage on 24 hand-written `(question, answer)` pairs, real before/after
generations, and a real measured similarity-to-target-response comparison, all with real
pasted output: [001_instruction_tuning.ipynb](001_instruction_tuning.ipynb)

## What you'll learn

Why a pretrained, next-token-prediction model like `13-llms-from-scratch/02-pretraining-objective`'s
`TinyGPT` completes text instead of following instructions — and why "just prompt it
better" cannot fix that for a model with no instruction-shaped text anywhere in its
training data. Then: instruction tuning as *the same* cross-entropy next-token objective
`02-pretraining-objective` already derived, reformatted around `(instruction, response)`
pairs and a fixed prompt template — explicitly connected to
`07-nlp/05-transformers-and-huggingface`'s fine-tuning topic as the same underlying
mechanism (gradient descent nudging a pretrained model's weights) applied to a different
data shape. RLHF and DPO are named as the next step beyond supervised instruction tuning,
honestly marked Planned rather than implemented.

## Why it matters

`02-pretraining-objective` built and trained a real pretraining run, and was explicit that
the result is a text *completer*, not an assistant. This topic is the first in the course
to actually close part of that gap — taking a pretrained model this repository trained
itself and turning gradient steps into a measurable shift toward request/response
behavior, with an honest report of how much that shift actually moved the needle at toy
scale (modest, not transformative) and where it visibly breaks (catastrophic forgetting of
plain continuation behavior, directly observed, not just cited).

## Prerequisites

- `13-llms-from-scratch/02-pretraining-objective` — this topic retrains that topic's exact
  `TinyGPT` architecture and pretraining recipe from scratch as its first stage; read that
  topic first for the causal-attention/next-token-objective derivation this topic builds
  on without re-deriving.
- `07-nlp/05-transformers-and-huggingface` — read for its fine-tuning and
  catastrophic-forgetting content; this topic connects to both explicitly rather than
  re-deriving them.
- Basic PyTorch (`nn.Module`, autograd, an optimizer step loop) — no new framework
  concepts beyond what `02-pretraining-objective` already used.

## What you'll build

- A **shared 41-character vocabulary**, built from the union of `02-pretraining-objective`'s
  lighthouse corpus and 24 hand-written instruction pairs about that same story — before
  any training happens, exactly how a real tokenizer is fixed once and reused across
  pretraining and fine-tuning.
- A freshly **pretrained `TinyGPT`** (21,353 parameters, identical architecture to Task 2),
  300 AdamW steps on the lighthouse corpus, **1.97s** wall-clock, training loss
  **3.82 → 1.69**.
- An **instruction-fine-tuned** version of that same model — 150 further AdamW steps at a
  3x lower learning rate on the 24 formatted `Q: ... \nA: ...` pairs, **0.58s** wall-clock,
  training loss **2.82 → 1.47**.
- A real before/after comparison: the pretrained-only model ignoring the `Q:`/`A:` template
  entirely versus the instruction-tuned model shifting its character statistics right after
  the `"A:"` marker — measured, not just eyeballed, via mean similarity-to-target-response
  (**0.311 → 0.345**, a real but modest ~11% relative increase).
- A directly observed instance of **catastrophic forgetting**: after instruction tuning, a
  plain story-continuation prompt (`"The old "`) starts producing stray `Q:`/`:` template
  fragments it never would have before — connected explicitly to
  `07-nlp/05-transformers-and-huggingface`'s catastrophic-forgetting failure mode.

## Where it appears in real systems

Every production chat/assistant LLM (ChatGPT-family, Claude, Llama-Instruct/Chat variants)
goes through a supervised instruction-tuning stage structurally identical to this topic's
Stage 2 — pretrained weights, continued training on `(instruction, response)` pairs with
the same next-token objective, at vastly larger scale and typically followed by RLHF/DPO
preference alignment (named, not implemented, here). The prompt-template discipline this
topic's "Failure modes" names (why real deployments enforce a fixed chat template) is the
direct production consequence of the format-mismatch failure mode this notebook
demonstrates at toy scale.

## What's next

Later `13-llms-from-scratch` topics (see the phase plan) build on the idea that a
pretrained-then-instruction-tuned model is still just the *supervised* half of a full
alignment pipeline — RLHF/DPO-style preference optimization is the natural next
conceptual step, named in this topic's notes.md but explicitly out of toy-scale reach for
an actual from-scratch implementation.
