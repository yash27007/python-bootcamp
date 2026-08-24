# Phase 11: LLMs From Scratch First-Principles Build-Out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** New `13-llms-from-scratch/` section: tokenizer → pretraining objective → instruction tuning, first-principles, toy scale throughout. Builds directly on `06-deep-learning/05-attention-transformers`'s from-scratch attention and `07-nlp/05-transformers-and-huggingface`'s tokenization/fine-tuning content — this section's job is specifically the *pretraining* objective and instruction-tuning distinction those two didn't cover.

**Architecture:** 3 topics, 1 task each.

**Tech Stack:** PyTorch (CPU, `torch.set_num_threads(1)`). A genuinely tiny GPT-style decoder trained on a small text corpus for a small number of steps — this is a toy that proves the mechanism, explicitly not a usable language model.

**Spec:** `docs/superpowers/specs/2026-08-23-first-principles-curriculum-design.md`, `AGENTS.md`.

## Global Constraints

- **Binding: no heavy/long-running training.** A genuinely tiny model (a few layers, small embedding dim), a small character-level or small-vocabulary corpus, few training steps — well under a few minutes on CPU. This is explicitly NOT a real LLM and notes.md must say so honestly (Learned/Currently-learning/Planned distinction from AGENTS.md — actual LLM pretraining at real scale is a Planned direction, not something this repo teaches by doing it for real).
- 12-section notes.md template. Real math throughout.
- Review level: light.

---

### Task 1: Tokenization Revisited — Building a BPE Tokenizer From Scratch

**Files:** Create `13-llms-from-scratch/01-tokenizer-from-scratch/` (README.md, notes.md, notebook)

**Content:** Problem = `07-nlp/05-transformers-and-huggingface` USED a pretrained tokenizer without showing how one is built. Why-simpler-fails = a fixed word-level vocabulary can't handle words it's never seen (cite `07-nlp/05-transformers-and-huggingface`'s OOV discussion). Mathematical/algorithmic foundation = derive the Byte-Pair Encoding algorithm precisely: start from individual characters/bytes, iteratively merge the most frequent adjacent pair into a new token, repeat until a target vocabulary size — this IS the algorithm, walk through it step by step. From-scratch = a REAL BPE tokenizer trained from scratch (plain Python, no library) on a small text corpus — actually run the merge loop, show the learned merges, tokenize a new sentence with the learned vocabulary, and CROSS-CHECK against `tokenizers`/HuggingFace's own BPE implementation on the same tiny corpus if feasible (or at minimum verify your from-scratch tokenizer round-trips correctly: encode then decode returns the original string). Practical = cite `07-nlp/05-transformers-and-huggingface`'s pretrained tokenizer for contrast (a real tokenizer trained on billions of tokens vs. this toy one trained on a few KB of text). Experiment = hypothesis about vocabulary size vs. average tokens-per-word, actually measured on the toy corpus. Failure modes = a tokenizer trained on the wrong domain/language producing pathological splits, vocabulary size tradeoffs (too small = long sequences, too large = huge embedding tables). Real-world, Mental model, Questions.

- [ ] Write notes.md + notebook (real BPE training + tokenization). README. `git commit -m "Phase 11 Task 1: first-principles build-out — tokenizer from scratch"`.

### Task 2: Pretraining Objective — A Tiny GPT-Style Decoder

**Files:** Create `13-llms-from-scratch/02-pretraining-objective/` (README.md, notes.md, notebook)

**Content:** Problem = `06-deep-learning/05-attention-transformers` built an attention mechanism and a Transformer *encoder* for classification — what's different about the *decoder*-only, next-token-prediction setup that GPT-style LLMs actually use for pretraining? Why-simpler-fails = an encoder sees the whole sequence at once (bidirectional attention) — that's wrong for generation, where you can only condition on what came before. Mathematical foundation = derive the causal (masked) self-attention formula — the same scaled-dot-product formula from `06-deep-learning/05-attention-transformers`, now with a triangular mask preventing attending to future positions, DERIVE why this mask makes next-token prediction well-defined (no information leakage from the future). The pretraining objective itself: autoregressive language modeling as maximizing $\prod_t P(x_t | x_{<t})$, i.e. cross-entropy loss on next-token prediction. From-scratch = cite `06-deep-learning/05-attention-transformers`'s attention implementation, extend it with the causal mask, build a genuinely tiny decoder-only Transformer (1-2 layers, small embedding dim, using Task 1's from-scratch tokenizer or a simple character-level tokenizer for speed) — ACTUALLY TRAIN on a small text corpus for a small number of steps (well under a few minutes, `torch.set_num_threads(1)`), show training loss decreasing, then ACTUALLY GENERATE text by sampling from the trained tiny model (the output will be nonsensical at this scale — say so honestly, the point is the mechanism works, not the output quality). Experiment = hypothesis that loss decreases and next-token predictions get less random over training (measured e.g. by perplexity or a simple "does it now put higher probability on the most-frequent-following-word for a common bigram in the training text" check), actually run. Failure modes = a model this tiny/undertrained will produce repetitive or nonsensical text — discuss WHY (insufficient capacity/data/steps) rather than treating it as a bug, connect to real-world scaling laws (mention by name, don't derive). Real-world, Mental model, Questions.

- [ ] Write notes.md + notebook (real tiny-GPT training + generation, honest about output quality). README. `git commit -m "Phase 11 Task 2: first-principles build-out — pretraining objective (tiny GPT decoder)"`.

### Task 3: Instruction Tuning — Pretraining vs. Fine-Tuning for Instructions

**Files:** Create `13-llms-from-scratch/03-instruction-tuning/` (README.md, notes.md, notebook)

**Content:** Problem = a pretrained (next-token-prediction) model like Task 2's completes text, it doesn't follow instructions or hold a conversation — cite Task 2's tiny model's raw-completion behavior explicitly. Why-simpler-fails = "just prompt it well" (few-shot prompting) has limits — the underlying model was never trained to distinguish "respond helpfully to a request" from "continue this text plausibly." Conceptual foundation = instruction tuning as supervised fine-tuning on (instruction, response) pairs, reformatting the same next-token-prediction objective around a structured prompt template — connect explicitly to `07-nlp/05-transformers-and-huggingface`'s fine-tuning topic (same underlying mechanism — gradient descent on a pretrained model's weights — different data/objective framing) and briefly to RLHF as the next step beyond supervised instruction tuning (mention DPO/RLHF by name, don't implement — real RLHF training is explicitly out of toy-scale reach and should be marked Planned, not attempted). From-scratch/Practical = take Task 2's tiny pretrained model (or a similarly tiny model, your call) and fine-tune it on a small set of REAL, hand-written (instruction, response) pairs (a few dozen simple templated examples is enough to show the effect) — ACTUALLY TRAIN, then show a real before/after comparison: the same prompt fed to the pretrained-only model (rambling completion) vs the instruction-tuned model (attempts a structured response) — real generated output for both, honestly assessed (at this toy scale the difference may be modest — report what you actually observe). Experiment = hypothesis that instruction-tuned generations look more like the target response format than pretrained-only generations, actually measured/compared. Failure modes = catastrophic forgetting of general language ability from over-fitting on a tiny instruction set (cite `07-nlp/05-transformers-and-huggingface`'s catastrophic-forgetting failure mode), instruction-response format mismatches at inference time. Real-world, Mental model, Questions.

- [ ] Write notes.md + notebook (real before/after instruction-tuning comparison). README. `git commit -m "Phase 11 Task 3: first-principles build-out — instruction tuning"`.

### Task 4: Section/root README

- [ ] Create `13-llms-from-scratch/README.md` (all 3 topics, ✅ Complete). Update root `README.md`. `git commit -m "Phase 11 Task 4: mark 13-llms-from-scratch complete in section and root README"`.

## Verification

```bash
cd /home/yashwanth-aravind/ml-course/python-bootcamp
.venv/bin/python -c "
import pathlib
for t in sorted(pathlib.Path('13-llms-from-scratch').iterdir()):
    if t.is_dir(): print(t.name, (t/'notes.md').exists(), (t/'README.md').exists())
"
```
