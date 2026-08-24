# Phase 6: Modern Transformers & HuggingFace Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** New `07-nlp/05-transformers-and-huggingface/` topic — extends `06-deep-learning/05-attention-transformers`'s existing from-scratch NumPy self-attention + Keras Transformer-encoder work into the practical HuggingFace ecosystem (tokenizers, pretrained models, `pipeline`, fine-tuning), per the design spec's explicit instruction that this NOT be a disconnected framework section.

**Architecture:** Single topic, 1 task (small scope — one new topic, not a new section). Bridges explicitly to the existing attention work rather than re-deriving self-attention.

**Tech Stack:** HuggingFace `transformers`, `datasets`, `tokenizers` (all installable, small pretrained models only — e.g. `distilbert-base-uncased`, not anything multi-GB).

**Spec:** `docs/superpowers/specs/2026-08-23-first-principles-curriculum-design.md`, `AGENTS.md`.

## Global Constraints

- `uv add transformers datasets tokenizers` — check install size/time; use small models only (distilbert/tinybert-scale, not full BERT/GPT-2 if a smaller option demonstrates the same point).
- 12-section notes.md template. From-scratch section CITES `06-deep-learning/05-attention-transformers`'s existing NumPy attention + Keras Transformer-encoder work rather than re-deriving — the new content here is specifically the HuggingFace layer (tokenization, pretrained weights, fine-tuning), not attention math again.
- Real HuggingFace code, actually executed, real output — a small pretrained model's inference and a small fine-tuning run (few epochs, small dataset, CPU, a few minutes max) must both actually run.
- Review level: light.

---

### Task 1: Tokenizers, Pretrained Models, Fine-tuning

**Files:** Create `07-nlp/05-transformers-and-huggingface/` (README.md, notes.md, notebook)

**Content:** Problem = Phase 2's from-scratch attention demo and Keras Transformer-encoder prove the mechanism works, but training a Transformer from random weights on a small dataset (as that notebook did) can't match what a model pretrained on billions of tokens already knows. Why-simpler-fails = training from scratch every time throws away all that pretrained knowledge — cite `07-nlp/04-deep-learning-nlp`'s and `06-deep-learning/05-attention-transformers`'s from-scratch-trained results as the "what you get without pretraining" baseline. Conceptual foundation = subword tokenization (BPE/WordPiece — why not word-level or char-level: OOV handling vs sequence length tradeoff, derive the tradeoff), the pretrain-then-fine-tune paradigm (transfer learning applied to NLP), what a `pipeline()` actually does under the hood (tokenize → model forward pass → decode). From-scratch = CITE `06-deep-learning/05-attention-transformers`'s from-scratch NumPy self-attention explicitly — this topic doesn't re-derive attention, it explains what's NEW: tokenization and pretrained weights. Practical = REAL HuggingFace code, ACTUALLY RUN: (a) a small pretrained model's tokenizer applied to a sentence, showing real subword splits; (b) `pipeline("sentiment-analysis")` or similar on a small model, real output; (c) a real, small fine-tuning run (few epochs, small labeled dataset — could reuse IMDB via `datasets`, but keep the subset small for speed) comparing pretrained-then-fine-tuned accuracy against `07-nlp/04-deep-learning-nlp`'s from-scratch-trained Embedding+LSTM accuracy on a comparable task — actually run both comparisons, real numbers. Experiment = the pretrained-vs-from-scratch accuracy comparison, hypothesis stated first (pretrained fine-tuning should win, especially on a small fine-tuning dataset, since it starts with real linguistic knowledge). Failure modes = catastrophic forgetting during fine-tuning, tokenizer mismatch (using a model with a tokenizer it wasn't trained with), fine-tuning on too little data overfitting fast. Real-world = why this is now the default approach in production NLP rather than training from scratch. Mental model, Questions.

- [ ] `uv add transformers datasets tokenizers`. Write notes.md + notebook (real tokenizer/pipeline/fine-tuning runs, real comparison). README in orientation format. Update `07-nlp/README.md` (add row 05) and root `README.md`'s section-07 blurb/table if it lists subtopics (check first — Phase 2's section-07 blurb may need a one-clause addition, not a full rewrite).
- [ ] `git add` everything, commit: `git commit -m "Phase 6: first-principles build-out — transformers and HuggingFace"`.

## Verification

```bash
cd /home/yashwanth-aravind/ml-course/python-bootcamp
ls 07-nlp/05-transformers-and-huggingface/{notes.md,README.md}
grep -c "^## " 07-nlp/05-transformers-and-huggingface/notes.md  # expect 12
```
