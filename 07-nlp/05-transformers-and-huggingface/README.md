# 05 – Transformers & HuggingFace

Detailed notes (subword tokenization and the OOV-vs-sequence-length tradeoff — BPE/WordPiece vs.
word-level vs. character-level, the pretrain-then-fine-tune transfer-learning paradigm, what
`pipeline()` does under the hood; Conceptual-foundation substitution documented inline, citing
[`06-deep-learning/05-attention-transformers`](../../06-deep-learning/05-attention-transformers/)
for the attention mathematics rather than re-deriving it): [notes.md](notes.md)

Real, actually-executed HuggingFace code — a pretrained tokenizer's real subword splits, a real
`pipeline("sentiment-analysis")` run, and a real small fine-tuning run on IMDB compared directly
against [`07-nlp/04-deep-learning-nlp`](../04-deep-learning-nlp/)'s from-scratch-trained
Embedding+LSTM baseline, all with real pasted output:
[transformers-and-huggingface.ipynb](transformers-and-huggingface.ipynb)

## What you'll learn

Why training a Transformer's attention weights from random initialization — what
`06-deep-learning/05-attention-transformers` and `07-nlp/04-deep-learning-nlp` both do — forces a
model to spend its entire labeled-data budget relearning basic language structure that a
pretrained model already has. How subword tokenization (BPE/WordPiece) resolves the
out-of-vocabulary-vs-sequence-length tradeoff that word-level and character-level tokenization
each fail differently. What `pipeline()` actually computes (tokenize → forward pass → decode) and
why it produces identical output to the three steps run manually. Why fine-tuning a pretrained
model on a small labeled dataset can reach competitive accuracy with a fraction of the labeled
data a from-scratch model needs — and why fine-tuning is not magic: it can still overfit fast on
too little data, forget its pretrained knowledge if pushed too hard, or silently break if paired
with the wrong tokenizer.

| Topic | Status |
|-------|--------|
| Problem: pretrained knowledge vs. training every task from scratch | ✅ Complete |
| Subword tokenization (BPE/WordPiece) and the OOV-vs-sequence-length tradeoff | ✅ Complete |
| Pretrain-then-fine-tune as transfer learning | ✅ Complete |
| What `pipeline()` does under the hood | ✅ Complete |
| Real tokenizer run: subword splits on a real sentence | ✅ Complete |
| Real `pipeline("sentiment-analysis")` run vs. manual tokenize→forward→decode | ✅ Complete |
| Real small fine-tuning run on IMDB vs. `07-nlp/04-deep-learning-nlp`'s from-scratch baseline | ✅ Complete |
| Failure modes: catastrophic forgetting, tokenizer/model mismatch, fast overfitting | ✅ Complete |

## Why it matters

Every production NLP system that needs to classify, extract, or generate text today starts from a
pretrained checkpoint and fine-tunes, rather than training a Transformer from random weights per
task — the reasons are exactly the ones measured in this topic's experiment: far less labeled data
needed, far fewer training epochs, and a starting point that already encodes general language
structure instead of none at all. Understanding fine-tuning as *reusing already-learned attention
weights*, rather than as a separate mechanism from the attention math in
`06-deep-learning/05-attention-transformers`, is what makes HuggingFace's `transformers` library
legible instead of a black box.

## Prerequisites

- `06-deep-learning/05-attention-transformers` — this topic cites, and does not re-derive, that
  topic's from-scratch self-attention (Q/K/V, scaled dot-product attention, multi-head attention)
  and Keras Transformer-encoder work.
- `07-nlp/04-deep-learning-nlp` — this topic's fine-tuning experiment is directly compared against
  that topic's from-scratch-trained Embedding+LSTM IMDB baseline (86.94% test accuracy).
- `uv add transformers datasets tokenizers accelerate` (already added to this repository's
  `pyproject.toml`).

## What you'll build

- A real DistilBERT tokenizer applied to a real sentence, printing the actual WordPiece subword
  split, token count vs. naive word count, `input_ids`, and a round-trip decode.
- A real `pipeline("sentiment-analysis")` call on a small pretrained model
  (`distilbert-base-uncased-finetuned-sst-2-english`), run side by side with the manual
  tokenize → forward-pass → decode steps it wraps, confirming both produce identical output.
- A real fine-tuning run: `distilbert-base-uncased` + a fresh classification head, fine-tuned with
  a plain PyTorch training loop on a small IMDB subset, evaluated every epoch, and compared
  directly against `07-nlp/04-deep-learning-nlp`'s from-scratch accuracy on the same task family —
  actual numbers, actual overfitting trend, no fabricated results.

## Where it appears in real systems

Sentiment analysis, intent classification, named-entity recognition, spam/toxicity detection, and
document classification in production overwhelmingly fine-tune a pretrained checkpoint
(BERT/DistilBERT/RoBERTa-family, or larger) rather than training from scratch — HuggingFace's
`transformers`/`datasets`/`tokenizers` stack, used exactly as in this topic's notebook, is the de
facto standard tooling for this pattern, backed by a public model hub so most teams never run the
(expensive) pretraining step themselves.

## What's next

This is the final planned topic in `07-nlp`. `08-mlops-deployment` picks up from here on the
engineering side (packaging, versioning, serving, monitoring models like the ones fine-tuned in
this topic); `11-generative-ai` (planned) and `13-llms-from-scratch` (planned) extend the
Transformer architecture toward generation and building a GPT-style decoder from scratch,
respectively.
