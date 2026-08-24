# 01 – Tokenizer From Scratch: Byte-Pair Encoding

Detailed notes (the OOV/sequence-length tradeoff that motivates subword tokenization,
Byte-Pair Encoding derived precisely as a pair-counting-and-merging algorithm, and its
scale-dependent limitations): [notes.md](notes.md)

Real, actually-executed, from-scratch BPE training (plain Python, no `tokenizers`
library) — real learned merges, a real encode/decode round-trip check, a real
vocabulary-size-vs-tokens-per-word experiment, and a real wrong-domain failure-mode demo,
all with real pasted output:
[001_bpe_tokenizer_from_scratch.ipynb](001_bpe_tokenizer_from_scratch.ipynb)

## What you'll learn

How the tokenizer `07-nlp/05-transformers-and-huggingface` used ready-made —
`AutoTokenizer.from_pretrained("distilbert-base-uncased")` — actually gets built: Byte-Pair
Encoding, derived from scratch as an iterative pair-counting-and-merging algorithm, starting
from individual characters and building up a vocabulary of increasingly large subword
pieces. Why a fixed word-level vocabulary has a hard out-of-vocabulary failure that BPE
resolves into a softer, measurable efficiency cost instead — and where BPE itself still
degrades when the input domain doesn't match the training domain.

| Topic | Status |
|-------|--------|
| Problem: how a pretrained tokenizer's vocabulary is actually built | ✅ Complete |
| Why fixed word-level vocabularies fail (OOV), cited from `07-nlp/05` | ✅ Complete |
| BPE derived precisely: pair counting, merge selection, stopping rule | ✅ Complete |
| Real from-scratch BPE training on a toy corpus, merges inspected | ✅ Complete |
| Real encode/decode round-trip verification on new text | ✅ Complete |
| Real experiment: vocabulary size vs. tokens-per-word, measured | ✅ Complete |
| Real wrong-domain pathological-split failure-mode demo | ✅ Complete |

## Why it matters

Every previous NLP topic in this course either used pre-integer-encoded data
(`07-nlp/04-deep-learning-nlp`'s Keras word index) or a pretrained tokenizer loaded with
one line (`07-nlp/05-transformers-and-huggingface`'s `AutoTokenizer`). Neither showed how
a subword vocabulary comes to exist. This topic is the first to build that vocabulary
from raw text, from scratch — which is also the first ingredient a from-scratch language
model (`13-llms-from-scratch/02-pretraining-objective`, next) needs before it can turn
text into anything a model can compute over at all.

## Prerequisites

- Basic Python (string/list manipulation, `collections.Counter`) — no NumPy, no PyTorch,
  no deep learning needed for this topic; BPE training here is plain data structures and
  loops.
- `07-nlp/05-transformers-and-huggingface`'s subword-tokenization discussion (OOV vs.
  sequence-length tradeoff) — this topic derives the algorithm that discussion cited but
  did not build.

## What you'll build

- A real BPE trainer (`word_freqs`, `get_pair_counts`, `merge_pair`, `train_bpe`) in
  plain Python, no tokenizer library, run on a small (462-character) toy corpus.
- A real `encode`/`decode` pair using the learned, ordered merge rules, verified with a
  genuine passing round-trip assertion (`decode(encode(text)) == text`) on a sentence
  containing words never seen verbatim during training.
- A measured experiment: BPE trained at 6 different target vocabulary sizes, tokens-per-word
  measured on held-out text at each, surfacing a real corpus-size-dependent vocabulary
  ceiling (training stops early once no pair repeats).
- A concrete wrong-domain failure-mode demo: the same trained tokenizer applied to Python
  source code, producing a measured 6.00 tokens/word (vs. 1.769 in-domain) — real output,
  not a described hypothetical.

## Where it appears in real systems

BPE (or a close variant — WordPiece, SentencePiece) is the tokenization algorithm behind
essentially every modern LLM: GPT-family, BERT-family, Llama, and the
`distilbert-base-uncased` tokenizer `07-nlp/05-transformers-and-huggingface` used
directly. Production tokenizer training runs this exact same pair-counting-and-merging
loop, at a training-corpus scale of billions of words rather than this topic's 462
characters — the algorithm doesn't change with scale, only the data it's trained on and
the engineering used to make each merge step fast at that scale.

## What's next

`02-pretraining-objective` — takes this topic's tokenizer (or a simple character-level
one, for training speed) and builds the next ingredient a language model needs: turning a
sequence of tokens into a next-token-prediction training objective, with a genuinely tiny
decoder-only Transformer actually trained on a small corpus.
