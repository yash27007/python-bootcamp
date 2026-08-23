# Phase 2: Deep Learning & NLP First-Principles Retrofit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrofit all 9 topics in `06-deep-learning/` (5) and `07-nlp/` (4) into the first-principles `notes.md` template defined in `AGENTS.md`, adding from-scratch NumPy implementations of the core mechanics (forward/backward pass, convolution, recurrence, attention) alongside the existing Keras/TensorFlow practical notebooks, plus measurable experiments, failure modes, and reasoning questions.

**Architecture:** Each topic's existing `notes.md` (all 9 already have solid derivations and LaTeX from an earlier plan) is restructured into the 12-section template (Problem, Intuition, Why-simpler-fails, Math, Algorithm, From-scratch, Practical, Experiment, Failure-modes, Real-world, Mental-model, Questions). Every topic gets a from-scratch NumPy demonstration of its core mechanic — a tiny forward+backward pass for ANN, a manual 2D convolution for CNN, a manual recurrence for RNN, manual LSTM/GRU gate equations, manual scaled-dot-product attention (already exists for `05-attention-transformers` from the earlier plan — verify and extend rather than rewrite) — connected explicitly to the existing Keras notebook as the practical step. 5 tasks, grouped by narrative family: (1) ANN+CNN, (2) RNN+LSTM/GRU, (3) Attention/Transformers, (4) NLP text-preprocessing+feature-extraction, (5) NLP word-embeddings+deep-learning-NLP. Tasks dispatched **sequentially** per subagent-driven-development's rule.

**Tech Stack:** Python 3.13, NumPy/SciPy (from-scratch), TensorFlow/Keras (practical — never PyTorch, per this repo's existing DL section), NLTK/gensim/scikit-learn (NLP practical), `.venv` (uv-managed), `.venv/bin/jupyter`.

**Spec:** `docs/superpowers/specs/2026-08-23-first-principles-curriculum-design.md` and `AGENTS.md` (binding template + quality bar) — read both before starting any task.

## Global Constraints

- Repo root: `/home/yashwanth-aravind/ml-course/python-bootcamp`. Python env: `.venv` (uv-managed) — `.venv/bin/python` / `.venv/bin/jupyter`.
- Every `notes.md` follows the 12-section template in `AGENTS.md` exactly, in order: Problem, Intuition, Why simpler approaches fail, Mathematical foundation, Algorithm, From-scratch implementation, Practical implementation, Experiment, Failure modes, Real-world usage, Mental model, Questions to think about. Problem / Why-simpler-fails / Mathematical foundation / Mental model / Questions are never skipped. LaTeX for all math, every symbol explained.
- **Preserve correct existing derivations.** All 9 topics already have substantial notes.md content with real LaTeX derivations from an earlier plan — read each file fully first, enumerate every formula/table/list item as a checklist BEFORE rewriting, and confirm every item lands somewhere in the new version as you write (not as an afterthought). This exact failure mode (silent content loss during retrofit) caused fix rounds in 5 of 7 Phase 1 tasks — do not repeat it.
- From-scratch implementation: plain Python/NumPy only (SciPy where genuinely useful). Small, illustrative, proving the mechanic works on a toy example — not a training-scale reimplementation. `06-deep-learning/05-attention-transformers` already has a from-scratch NumPy scaled-dot-product attention demo from an earlier plan — verify it's still present and correct, extend/reference it rather than rewriting.
- Practical implementation: the existing Keras/TensorFlow (DL) or NLTK/gensim/sklearn (NLP) notebook(s) for that topic, explicitly connected in notes.md prose to the from-scratch step.
- Experiment section: hypothesis stated before result. Reuse existing notebook results where they already constitute a valid experiment (most of these topics already have training-curve/accuracy experiments from the earlier plan) rather than manufacturing a redundant one.
- **Notebook execution gate (revised per Phase 1's lesson): every notebook a topic's `notes.md` cites as evidence — whether newly created, edited, or merely referenced as the practical/experiment step — must have non-null `execution_count` on every code cell before that topic is done.** Do not scope execution to "touched or created" only. If a cited notebook is already fully executed from an earlier plan, just verify it (don't re-run unnecessarily); if any cited notebook has any unexecuted cells, execute it: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace <path>.ipynb`.
- Datasets: reuse what's already in each notebook (fashion_mnist, imdb, sklearn built-ins, existing toy corpora). No new manual downloads.
- Each topic's own `README.md`: verify it follows the AGENTS.md orientation format (concise stub, not a notes.md duplicate) and its status doesn't contradict the section-level `06-deep-learning/README.md`/`07-nlp/README.md` (both already show all rows ✅ Complete — don't let a topic README still say "Coming soon").
- Commit granularity: one commit per task.
- No unit-test framework applies. "Done" means: notes.md has all 12 sections with real substantive content, every cited notebook executes cleanly, and nothing correct from the original was silently lost.

---

### Task 1: ANN + CNN — `06-deep-learning/01-ann`, `06-deep-learning/02-cnn`

**Files:**
- Modify: `06-deep-learning/01-ann/notes.md` (retrofit)
- Modify: `06-deep-learning/02-cnn/notes.md` (retrofit)

**Content requirements:**

- **`01-ann`**: Problem = approximate an arbitrary function mapping inputs to outputs when no closed-form relationship is known. Why-simpler-fails = a single linear layer (logistic/linear regression) can only represent linear decision boundaries — link to `05-machine-learning/07-svm/notes.md` or `06-logistic-regression/notes.md`'s linear-boundary limitation as the motivating prior case. Math = existing forward-propagation/backprop/chain-rule derivation (preserve, restructure), activation function formulas, loss functions, optimizer update rules. From-scratch = a tiny NumPy 2-layer MLP: forward pass, manual backward pass (chain rule through one hidden layer), gradient descent weight update, trained on a toy XOR-like dataset a single-layer model provably cannot solve — this is the single most important from-scratch demo in the DL section since it's the conceptual root everything else builds on. Practical = existing `ann-regression.ipynb`/`ann-classification.ipynb` (Keras). Experiment = existing training/validation loss curves — cite as Experiment, hypothesis-first framing. Failure modes = vanishing/exploding gradients, poor initialization, existing content preserved. Mental model, Questions.
- **`02-cnn`**: Problem = images have spatial structure a fully-connected ANN ignores (flattening loses locality, and full connectivity means an enormous, position-specific parameter count). Why-simpler-fails = explicitly show the parameter-count blowup of a dense layer on a modest image vs a conv layer with shared weights. Math = existing convolution/padding/pooling derivation (preserve), parameter-sharing and translation-invariance argument. From-scratch = manual 2D convolution in NumPy (nested loops or `np.lib.stride_tricks`, whichever is clearer) applying a small edge-detection kernel to a toy image, visualize the feature map, compare to `scipy.signal.convolve2d` or `tf.nn.conv2d`'s output on the same input to confirm they match. Practical = existing `cnn-image-classification.ipynb`. Experiment = existing training curves/sample predictions — cite. Failure modes, Real-world, Mental model, Questions.

- [ ] **Step 1:** Read both existing `notes.md` files and notebooks fully; build the preservation checklist for each.
- [ ] **Step 2:** Retrofit both `notes.md` into the 12-section template per the content requirements.
- [ ] **Step 3:** Add the from-scratch NumPy MLP (XOR) notebook/cell for `01-ann` and the from-scratch 2D convolution notebook/cell for `02-cnn`.
- [ ] **Step 4:** Execute every notebook cited by either topic's notes.md that isn't already fully executed (check `execution_count` on every cell first — the existing Keras notebooks from the earlier plan should already be executed; only new from-scratch notebooks likely need running).
- [ ] **Step 5:** Verify both topic `README.md` files match the orientation format and don't contradict "Complete" status.
- [ ] **Step 6:** `git add` both topic folders, commit: `git commit -m "Phase 2 Task 1: first-principles retrofit — ANN, CNN"`.

---

### Task 2: RNN + LSTM/GRU — `06-deep-learning/03-rnn`, `06-deep-learning/04-lstm-gru`

**Files:**
- Modify: `06-deep-learning/03-rnn/notes.md` (retrofit — already scored highest template-alignment of any pre-Phase-1 topic in the original audit, build on it)
- Modify: `06-deep-learning/04-lstm-gru/notes.md` (retrofit)

**Content requirements:**

- **`03-rnn`**: Problem = sequential data where order matters and length varies — neither ANN nor CNN naturally handles variable-length ordered input. Why-simpler-fails = padding/flattening a sequence into a fixed-size ANN input discards position information and doesn't share weights across time steps. Math = existing recurrence formula and BPTT derivation (preserve). From-scratch = manual NumPy `SimpleRNN` forward pass: a single recurrent cell applying the same weight matrix at each time step over a short toy sequence, unrolled by hand, showing the hidden state evolving. Practical = existing `rnn-imdb-sentiment.ipynb`. Experiment = existing training curves — cite. Failure modes = existing vanishing-gradient-in-BPTT content (preserve — this is what motivates the next topic). Mental model, Questions.
- **`04-lstm-gru`**: Problem = vanilla RNNs can't learn long-range dependencies because gradients vanish over many BPTT steps (explicitly link back to `03-rnn`'s failure mode). Why-simpler-fails = already covered by that link — the "simpler approach" here IS vanilla RNN. Math = existing LSTM gate equations (forget/input/output, cell state) and GRU gate equations (preserve), the additive cell-state-update argument for why gradients don't vanish the same way. From-scratch = manual NumPy LSTM cell: one forward step through the four gate equations on a toy input, showing the cell state and hidden state update — a single time step is enough to demonstrate the mechanic, doesn't need a full unroll. Practical = existing `lstm-gru-time-series.ipynb` (already compares LSTM vs GRU vs BiLSTM). Experiment = existing three-way comparison — cite. Failure modes, Real-world, Mental model, Questions.

- [ ] **Step 1:** Read both existing `notes.md` and notebooks; build preservation checklists.
- [ ] **Step 2:** Retrofit both into the 12-section template.
- [ ] **Step 3:** Add the from-scratch manual RNN-cell-unroll notebook/cell for `03-rnn` and the from-scratch manual LSTM-gate-step notebook/cell for `04-lstm-gru`.
- [ ] **Step 4:** Execute any newly-added or previously-unexecuted cited notebooks.
- [ ] **Step 5:** Verify both topic `README.md` files.
- [ ] **Step 6:** `git add` both topic folders, commit: `git commit -m "Phase 2 Task 2: first-principles retrofit — RNN, LSTM/GRU"`.

---

### Task 3: Attention & Transformers — `06-deep-learning/05-attention-transformers`

**Files:**
- Modify: `06-deep-learning/05-attention-transformers/notes.md` (retrofit — already has a from-scratch NumPy attention demo from an earlier plan, verify and extend rather than rewrite)

**Content requirements:**

Problem = LSTM/GRU still process sequentially and compress all history into one fixed-size hidden state, which bottlenecks long-range dependencies and prevents parallelization (explicitly link back to `04-lstm-gru`). Why-simpler-fails = a single fixed-size context vector (vanilla encoder-decoder) can't represent arbitrarily long input faithfully — this is the seq2seq bottleneck the existing notes.md likely already covers (preserve). Math = existing scaled-dot-product self-attention formula, multi-head attention, positional encoding (preserve and verify correctness of the QKV derivation). From-scratch = the existing NumPy self-attention demo — verify it's present, correct, and includes the attention-weight heatmap visualization; if it only covers single-head attention, note multi-head as a natural extension in prose rather than necessarily re-implementing it from scratch (use judgment per the Global Constraints' "small and illustrative" guidance). Practical = existing Keras Transformer-encoder text classifier on IMDB. Experiment = existing test-accuracy result — cite, hypothesis-first framing if not already present. Failure modes = existing content (quadratic attention cost in sequence length is worth confirming is covered). Real-world, Mental model, Questions.

- [ ] **Step 1:** Read the existing `notes.md` and notebook fully; build the preservation checklist.
- [ ] **Step 2:** Retrofit into the 12-section template.
- [ ] **Step 3:** Verify the existing from-scratch attention demo meets the "From-scratch implementation" section's bar; extend only if genuinely thin.
- [ ] **Step 4:** Execute any cited notebook not already fully executed.
- [ ] **Step 5:** Verify the topic `README.md`.
- [ ] **Step 6:** `git add` the topic folder, commit: `git commit -m "Phase 2 Task 3: first-principles retrofit — attention & transformers"`.

---

### Task 4: NLP foundations — `07-nlp/01-text-preprocessing`, `07-nlp/02-feature-extraction`

**Files:**
- Modify: `07-nlp/01-text-preprocessing/notes.md` (retrofit)
- Modify: `07-nlp/02-feature-extraction/notes.md` (retrofit)

**Content requirements:**

- **`01-text-preprocessing`**: Problem = raw text is unstructured and inconsistent (case, inflection, punctuation, stopwords) — models need a normalized, finite vocabulary. Why-simpler-fails = treating every distinct string as a unique token explodes the vocabulary and treats "running"/"run"/"runs" as unrelated. Math/Algorithm = existing tokenization, stemming (Porter algorithm's rule-based suffix stripping — show a couple of its actual rules, not just "it strips suffixes"), lemmatization (dictionary+POS lookup), stopword removal, POS tagging, NER (preserve existing content). From-scratch = a tiny hand-rolled suffix-stripping stemmer (3-4 rules, e.g. strip "-ing"/"-ed"/"-s") on a few example words, compared to `PorterStemmer`'s output on the same words to show where the toy version agrees/disagrees — illustrates that stemming is rule-based pattern matching, not magic. Practical = existing NLTK notebook. Experiment = existing stemming-vs-lemmatization comparison — cite. Failure modes = over-stemming/under-stemming (preserve). Real-world, Mental model, Questions.
- **`02-feature-extraction`**: Problem = ML models need numeric input, not strings. Why-simpler-fails = one-hot encoding of a vocabulary is enormous and sparse and encodes no notion of similarity between words (link forward to `03-word-embeddings` as the eventual fix, without duplicating that topic's content). Math = existing BOW, N-gram, TF-IDF formulas (preserve — derive the IDF formula's log dampening if not already explained). From-scratch = manual NumPy TF-IDF computation on a tiny 3-4 document toy corpus (term frequency matrix, document frequency, IDF, TF×IDF), compared to `TfidfVectorizer`'s output on the same corpus to confirm they match. Practical = existing notebook (one-hot, BOW, bigrams, TF-IDF). Experiment = existing DataFrame comparisons — cite. Failure modes, Real-world, Mental model, Questions.

- [ ] **Step 1:** Read both existing `notes.md`/notebooks; build preservation checklists.
- [ ] **Step 2:** Retrofit both into the 12-section template.
- [ ] **Step 3:** Add the from-scratch toy stemmer for `01-text-preprocessing` and the from-scratch manual TF-IDF computation for `02-feature-extraction`.
- [ ] **Step 4:** Execute any newly-added or previously-unexecuted cited notebooks.
- [ ] **Step 5:** Verify both topic `README.md` files.
- [ ] **Step 6:** `git add` both topic folders, commit: `git commit -m "Phase 2 Task 4: first-principles retrofit — text preprocessing, feature extraction"`.

---

### Task 5: Word embeddings + Deep Learning NLP — `07-nlp/03-word-embeddings`, `07-nlp/04-deep-learning-nlp`

**Files:**
- Modify: `07-nlp/03-word-embeddings/notes.md` (retrofit — currently the thinnest notes.md in NLP per the original audit, only 14 `$` signs; needs real expansion, not just restructuring)
- Modify: `07-nlp/04-deep-learning-nlp/notes.md` (retrofit — also thin, 8 `$` signs)

**Content requirements:**

- **`03-word-embeddings`**: Problem = sparse BOW/TF-IDF vectors have no notion of semantic similarity ("king" and "queen" are as unrelated as "king" and "banana" under TF-IDF) — link back to `02-feature-extraction`'s failure mode. Why-simpler-fails = a one-hot/TF-IDF vector's dimensionality equals vocabulary size and every pair of distinct words is equidistant. Math = the distributional hypothesis ("a word is characterized by the company it keeps"), Word2Vec's CBOW objective (predict center word from context) vs Skip-Gram objective (predict context from center word) — derive the softmax/negative-sampling objective at a level of detail beyond what's currently there (this topic needs real expansion, not just reformatting). From-scratch = a tiny NumPy CBOW forward pass on a toy 4-5 word vocabulary: one-hot context words → averaged embedding lookup → softmax over vocabulary — small enough to trace by hand, illustrating the mechanic Word2Vec's Keras/gensim implementation performs at scale. Practical = existing `gensim.Word2Vec` CBOW-vs-SkipGram notebook. Experiment = existing `most_similar` results + AvgWord2Vec classification accuracy — cite. Failure modes = out-of-vocabulary words, doesn't capture polysemy (one vector per word regardless of sense) — motivates contextual embeddings as a "planned/future" direction per AGENTS.md's Learned/Currently-learning/Planned honesty rule, don't claim it's covered. Real-world, Mental model, Questions.
- **`04-deep-learning-nlp`**: Problem = Word2Vec produces fixed, context-independent embeddings — how do we let a model use embeddings as input to something that also models sequence/context (link to `03-rnn`/`04-lstm-gru`). Why-simpler-fails = averaging word vectors (as in `03-word-embeddings`'s AvgWord2Vec classifier) discards word order entirely ("dog bites man" vs "man bites dog" average to the same vector). Math = the Keras `Embedding` layer as a learned lookup table (differentiable, trained end-to-end vs Word2Vec's separately-trained vectors) — explain this distinction clearly since it's the topic's core conceptual point and is currently thin. From-scratch = not much new mechanic needed here (LSTM already covered in `04-lstm-gru`, embeddings already covered from-scratch in `03-word-embeddings`) — the from-scratch section can legitimately be brief, explicitly noting it composes two already-derived mechanics (embedding lookup + recurrence), per the Global Constraints' "from-scratch only where it adds insight" judgment call — document this reasoning rather than skip the section. Practical = existing Keras Embedding+LSTM+IMDB notebook. Experiment = existing test accuracy + decoded-review predictions — cite. Failure modes, Real-world, Mental model, Questions.

- [ ] **Step 1:** Read both existing `notes.md`/notebooks; build preservation checklists. Note both files currently need substantive expansion, not just restructuring — budget accordingly.
- [ ] **Step 2:** Retrofit both into the 12-section template with the expanded math/from-scratch content above.
- [ ] **Step 3:** Add the from-scratch CBOW notebook/cell for `03-word-embeddings`.
- [ ] **Step 4:** Execute any newly-added or previously-unexecuted cited notebooks.
- [ ] **Step 5:** Verify both topic `README.md` files.
- [ ] **Step 6:** `git add` both topic folders, commit: `git commit -m "Phase 2 Task 5: first-principles retrofit — word embeddings, deep learning NLP"`.

---

## Verification (after all 5 tasks)

```bash
.venv/bin/python - <<'EOF'
import json, pathlib
paths = list(pathlib.Path("06-deep-learning").glob("*/*.ipynb")) + list(pathlib.Path("07-nlp").glob("*/*.ipynb"))
bad = []
for p in paths:
    d = json.load(open(p))
    for c in d["cells"]:
        if c["cell_type"] == "code" and not c.get("execution_count"):
            bad.append(str(p)); break
print(f"{len(paths)} notebooks checked")
print("NOT executed:", bad or "none")

for topic in sorted(list(pathlib.Path("06-deep-learning").iterdir()) + list(pathlib.Path("07-nlp").iterdir())):
    if not topic.is_dir(): continue
    nm = topic / "notes.md"
    if not nm.exists():
        print("MISSING notes.md:", topic); continue
EOF
```

Expect `NOT executed: none` and no `MISSING notes.md`.
