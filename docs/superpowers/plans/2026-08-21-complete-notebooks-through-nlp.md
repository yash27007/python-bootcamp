# Complete Course Notebooks Through NLP Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill in every remaining gap in `05-machine-learning`, `06-deep-learning`, and `07-nlp` with real concept notes (`notes.md`) and real, executed practical notebooks (`*.ipynb`), so the course is fully complete through NLP.

**Architecture:** Each topic folder gets (where missing) a `notes.md` (concept explanation, following the style of `05-machine-learning/01-introduction/notes.md`: headers, prose, LaTeX via `$...$`/`$$...$$`, no placeholders) and one or more `*.ipynb` notebooks that actually train/fit/run something real on a real (built-in) dataset, with cells executed so outputs/plots/metrics are baked in. Each topic's `README.md` status column flips from 🚧/📝 to ✅ Complete once its notes + notebook(s) exist. Work is organized into 3 tasks by section (ML gaps, Deep Learning, NLP) since sections are independent of each other, but per this skill's rule they are dispatched **sequentially**, not in parallel.

**Tech Stack:** Python 3.13, scikit-learn, pandas/numpy/matplotlib/seaborn (ML); TensorFlow/Keras (DL); NLTK, gensim, scikit-learn, TensorFlow/Keras (NLP). All already installed via `uv sync` (tensorflow, nltk, gensim, wordcloud added to `pyproject.toml`).

**Spec:** No separate spec file — this plan implements the design approved in chat during the `superpowers:brainstorming` session on this task (bounded-path brainstorm: cleanup of empty project folders, already committed at `f985b1a`, plus completing 14 remaining topics across 05/06/07 with real concept+practical content). The Global Constraints below are the distilled, binding requirements from that approved design.

## Global Constraints

- Repo root: `/home/yashwanth-aravind/ml-course/python-bootcamp`. Python env: `.venv` (uv-managed), invoke as `.venv/bin/python` / `.venv/bin/jupyter`.
- Every new/modified notebook MUST be executed end-to-end with no errors before being considered done. Use: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace <path>.ipynb`. A notebook with unexecuted cells (no `execution_count`, no outputs) fails task review.
- Deep learning notebooks use **TensorFlow/Keras** (`import tensorflow as tf` / `from tensorflow import keras`), never PyTorch.
- Datasets are **standard built-ins only** — no manual file downloads, no bundling new CSVs: `sklearn.datasets` (`load_iris`, `fetch_california_housing`, `load_breast_cancer`, `fetch_20newsgroups`), `tensorflow.keras.datasets` (`mnist`, `fashion_mnist`, `imdb`). Internet access is available (already verified) for the one-time Keras dataset cache download.
- Every `notes.md` follows the style of `05-machine-learning/01-introduction/notes.md`: a status table at the top listing the subtopics (copy the exact subtopic names from that folder's `README.md` table), then one `##` section per subtopic with real prose explanation and math in `$...$` / `$$...$$` LaTeX. No "TBD"/"Coming soon" text inside notes.md — it is written as if the topic is fully done, because it is.
- After a topic's notes + notebook(s) exist and execute cleanly, edit that topic's row in its section's `README.md` (`05-machine-learning/README.md`, `06-deep-learning/README.md`, `07-nlp/README.md`) status column from `🚧 Coming soon` / `📝 Notes only` to `✅ Complete`. Do not touch rows for topics outside this task.
- After all 3 tasks are done, the root `README.md` "Curriculum" table rows for sections 05 (already ✅), 06, and 07 must read `✅ Complete` instead of `🚧 Coming soon`, and the "Course Roadmap" ASCII diagram and the `### 06 –` / `### 07 –` prose blurbs must drop the `*(coming soon)*` marker. This edit belongs to Task 3 (the last task to finish), since it depends on both 06 and 07 being done.
- Commit after each topic's notes+notebook(s) are executed and verified (small, reviewable commits), not one giant commit per task.
- No unit-test framework applies to this content — the equivalent of "tests pass" is: notebook executes with `--execute` producing no error, and the notebook's markdown/code cells visibly cover every subtopic bullet from that topic's `README.md`.

---

### Task 1: Machine Learning gaps — `01-introduction`, `05-cross-validation`, `08-naive-bayes`, `09-knn`, `10-decision-tree`

**Files:**
- Modify: `05-machine-learning/01-introduction/` — create `05-machine-learning/01-introduction/instance-vs-model-based-and-geometry.ipynb`
- Modify: `05-machine-learning/05-cross-validation/` — create `05-machine-learning/05-cross-validation/cross-validation.ipynb`
- Modify: `05-machine-learning/08-naive-bayes/` — create `notes.md` and `naive-bayes.ipynb`; delete the now-superfluous `.gitkeep`
- Modify: `05-machine-learning/09-knn/` — create `notes.md` and `knn.ipynb`; delete `.gitkeep`
- Modify: `05-machine-learning/10-decision-tree/` — create `notes.md` and `decision-tree.ipynb`; delete `.gitkeep`
- Modify: `05-machine-learning/README.md` — flip rows 01, 05, 08, 09, 10 to `✅ Complete` (row 01 and 05 are already marked complete/notes-only in the table but must now actually have a notebook backing that status — read the current table first)

**Interfaces:** None (leaf content task, no code interfaces consumed/produced).

**Content requirements per topic** (read each folder's existing `README.md` first — it lists the exact subtopics the notebook/notes must cover):

- **`01-introduction`**: `notes.md` already exists and is complete — do not rewrite it, only add the notebook. Notebook must demonstrate: (a) the equation of a line/hyperplane in 2D and 3D with a matplotlib plot, (b) a worked comparison of instance-based learning (fit a `KNeighborsClassifier`) vs. model-based learning (fit a `LogisticRegression`) on the same small dataset (`sklearn.datasets.load_iris`, 2 features for plotting), showing decision boundaries side by side. Use markdown cells to narrate what's happening; this is a from-scratch demo, not a benchmark.
- **`05-cross-validation`**: `notes.md` already exists — read it, do not duplicate its prose in the notebook, just implement what it describes. Notebook must run: K-Fold, Stratified K-Fold, and Leave-One-Out CV (`sklearn.model_selection`) on `load_breast_cancer`, print per-fold scores and mean±std, then run `GridSearchCV` and `RandomizedSearchCV` over a small hyperparameter grid for a `RandomForestClassifier`, print best params/score.
- **`08-naive-bayes`**: The existing `README.md` already contains the Bayes' theorem / naive assumption / variants math — copy that content into `notes.md` verbatim as the base, then extend `notes.md` with a `## Cost Function Formulation` recap (already in README) and one worked numeric example (a tiny 2-feature toy table) computing a posterior by hand in the markdown. Notebook: fit `GaussianNB` on `load_iris`, and fit `MultinomialNB` on text data from `fetch_20newsgroups(categories=['sci.space','rec.sport.hockey'])` using `CountVectorizer`; print accuracy and a confusion matrix for both.
- **`09-knn`**: `notes.md` must cover: KNN intuition (instance-based/lazy learning), distance metrics (Euclidean, Manhattan, Minkowski), the effect of K (bias/variance), and KNN for both classification and regression. Notebook: `KNeighborsClassifier` on `load_iris` with a K-vs-accuracy sweep plot (elbow-style, using cross-validation scores) picking a best K, plus `KNeighborsRegressor` on `fetch_california_housing` reporting RMSE/R².
- **`10-decision-tree`**: `notes.md` must cover: tree structure (root/internal/leaf), splitting criteria (Gini impurity, entropy/information gain, MSE for regression), overfitting and pruning (`max_depth`, `min_samples_leaf`, `ccp_alpha` cost-complexity pruning). Notebook: `DecisionTreeClassifier` on `load_breast_cancer` with `sklearn.tree.plot_tree` visualization comparing an unpruned vs. a depth-limited tree (accuracy comparison), and `DecisionTreeRegressor` on `fetch_california_housing` reporting RMSE.

- [ ] **Step 1:** For each of the 5 topics above, write `notes.md` (where missing) per the content requirements.
- [ ] **Step 2:** For each of the 5 topics, write the notebook per the content requirements — code cells + markdown narration cells, no empty placeholder cells.
- [ ] **Step 3:** Execute every notebook created/modified in this task: `.venv/bin/jupyter nbconvert --to notebook --execute --inplace <path>` for each of the 5 notebook files. Confirm exit code 0 for each and that the resulting `.ipynb` has non-null `execution_count` and outputs on its code cells (spot check with `.venv/bin/python -c "import json; d=json.load(open(path)); assert all(c.get('execution_count') for c in d['cells'] if c['cell_type']=='code')"`).
- [ ] **Step 4:** Update `05-machine-learning/README.md` status column for rows 01, 05, 08, 09, 10 to `✅ Complete` (only if not already so).
- [ ] **Step 5:** `git add` the 5 topic folders + `05-machine-learning/README.md`, commit: `git commit -m "Complete ML notebooks: introduction geometry, cross-validation, naive bayes, KNN, decision tree"`.

---

### Task 2: Deep Learning — `01-ann`, `02-cnn`, `03-rnn`, `04-lstm-gru`, `05-attention-transformers`

**Files:**
- Modify: `06-deep-learning/01-ann/` — create `notes.md`, `ann-regression.ipynb`, `ann-classification.ipynb`
- Modify: `06-deep-learning/02-cnn/` — create `notes.md`, `cnn-image-classification.ipynb`
- Modify: `06-deep-learning/03-rnn/` — create `notes.md`, `rnn-imdb-sentiment.ipynb`
- Modify: `06-deep-learning/04-lstm-gru/` — create `notes.md`, `lstm-gru-time-series.ipynb`
- Modify: `06-deep-learning/05-attention-transformers/` — create `notes.md`, `attention-and-transformer.ipynb`
- Modify: `06-deep-learning/README.md` — flip all 5 rows to `✅ Complete`

**Interfaces:** None (leaf content task; independent of Task 1's files).

**Content requirements per topic** (each folder's `README.md` lists the exact subtopic bullets `notes.md` must cover — read it first):

- **`01-ann`**: `notes.md` covers: perceptron intuition, ANN architecture & forward propagation, backpropagation & weight updates (chain rule, gradient descent update rule), activation functions (Sigmoid, Tanh, ReLU, Leaky ReLU, ELU, Softmax — formulas + when to use each), loss vs. cost function, regression loss (MSE) vs. classification loss (binary/categorical cross-entropy), optimizers (SGD, Momentum, AdaGrad, RMSProp, Adam — one paragraph each), vanishing/exploding gradients, weight initialization (Xavier/He) and dropout. `ann-regression.ipynb`: Keras `Sequential` MLP on `fetch_california_housing` (StandardScaler features, Dense layers with ReLU, Adam, MSE loss, EarlyStopping), plot training/validation loss curves, report test MAE/RMSE. `ann-classification.ipynb`: Keras `Sequential` MLP on `load_breast_cancer` with Dropout layers, sigmoid output, binary cross-entropy, plot accuracy/loss curves, report test accuracy + confusion matrix.
- **`02-cnn`**: `notes.md` covers: convolution operation (kernels/filters, stride, padding), pooling (max/average), typical CNN architecture (conv→pool→...→dense), why CNNs beat plain ANNs on images (parameter sharing, translation invariance). Notebook: Keras `Sequential` CNN (`Conv2D`+`MaxPooling2D` stack) trained on `tf.keras.datasets.fashion_mnist`, plot training curves, report test accuracy, show a grid of sample predictions with true/predicted labels.
- **`03-rnn`**: `notes.md` covers: why sequence models (order matters, variable length), RNN architecture and the recurrence formula, backpropagation through time (BPTT), the vanishing gradient problem in vanilla RNNs (motivating LSTM/GRU in the next topic). Notebook: Keras `Embedding` + `SimpleRNN` on `tf.keras.datasets.imdb` (top 10k words, padded sequences) for binary sentiment classification, plot training curves, report test accuracy.
- **`04-lstm-gru`**: `notes.md` covers: LSTM gates (forget, input, output — formulas), GRU gates (reset, update) and how it simplifies LSTM, Bidirectional RNNs, when to prefer LSTM/GRU/Bidirectional. Notebook: generate a reproducible synthetic time series (e.g. noisy sine wave with `numpy`, fixed `np.random.seed`) framed as a windowed sequence-prediction task, train and compare a Keras `LSTM` model vs. a `GRU` model vs. a `Bidirectional(LSTM(...))` model on the same data, plot predicted-vs-actual for each, report test MSE for all three.
- **`05-attention-transformers`**: `notes.md` covers: the limitation of RNNs that motivates attention (long-range dependencies), self-attention mechanism (Query/Key/Value, scaled dot-product formula), multi-head attention, positional encoding, and the encoder-decoder Transformer architecture at a conceptual level (no need to derive backprop). Notebook: first implement scaled dot-product self-attention from scratch in NumPy on a tiny toy sequence (show the attention weight matrix as a heatmap) to build intuition; then build a small Keras Transformer-encoder-based text classifier (`MultiHeadAttention` + `LayerNormalization` + Dense layers, following the standard Keras "Text classification with Transformer" pattern) on `tf.keras.datasets.imdb`, report test accuracy.

- [ ] **Step 1:** For each of the 5 topics, write `notes.md` per the content requirements above, using the exact subtopic list from that folder's current `README.md`.
- [ ] **Step 2:** For each of the 5 topics, write the notebook(s) per the content requirements — real Keras models, real training runs, plots and metrics in markdown/code cells.
- [ ] **Step 3:** Execute every notebook created in this task with `.venv/bin/jupyter nbconvert --to notebook --execute --inplace <path>`, confirm exit 0 and populated outputs on every code cell (same check as Task 1 Step 3). Keep epoch counts small enough (5-15 epochs, small batch/hidden sizes) that each notebook finishes in a few minutes on CPU — this is a teaching demo, not a leaderboard model.
- [ ] **Step 4:** Update `06-deep-learning/README.md`: flip all 5 topic rows' status column to `✅ Complete`.
- [ ] **Step 5:** `git add` the 5 topic folders + `06-deep-learning/README.md`, commit: `git commit -m "Complete deep learning notebooks: ANN, CNN, RNN, LSTM/GRU, attention & transformers"`.

---

### Task 3: NLP — `01-text-preprocessing`, `02-feature-extraction`, `03-word-embeddings`, `04-deep-learning-nlp`, plus root README finalization

**Files:**
- Modify: `07-nlp/01-text-preprocessing/` — create `notes.md`, `text-preprocessing.ipynb`
- Modify: `07-nlp/02-feature-extraction/` — create `notes.md`, `feature-extraction.ipynb`
- Modify: `07-nlp/03-word-embeddings/` — create `notes.md`, `word-embeddings.ipynb`
- Modify: `07-nlp/04-deep-learning-nlp/` — create `notes.md`, `deep-learning-nlp.ipynb`
- Modify: `07-nlp/README.md` — flip all 4 rows to `✅ Complete`
- Modify: `README.md` (repo root) — per Global Constraints: Curriculum table rows for 06 and 07 to `✅ Complete`; Course Roadmap diagram and `### 06 –` / `### 07 –` blurbs drop `(coming soon)`

**Interfaces:** None (leaf content task; independent of Tasks 1-2's files, but its Step 5 root-README edit is sequenced last because it also finalizes the 06 row, which Task 2 completes).

**Content requirements per topic** (each folder's `README.md` lists the exact subtopic bullets `notes.md` must cover):

- **`01-text-preprocessing`**: `notes.md` covers: tokenization & basic terminology (corpus, vocabulary, document), stemming (Porter/Snowball — over-stemming pitfalls), lemmatization (dictionary-based, POS-aware, vs. stemming), stop word removal, POS tagging (tagset overview), named entity recognition (NER). Notebook: use `nltk` on a short real paragraph of text (2-3 sentences, hardcoded string) to run: `word_tokenize`, `PorterStemmer` vs `WordNetLemmatizer` side-by-side comparison table, stopword removal, `pos_tag`, and `ne_chunk` for NER — print/display results for each step. Any `nltk.download(...)` calls needed must be present as executable cells (idempotent, safe to re-run).
- **`02-feature-extraction`**: `notes.md` covers: one-hot encoding of text (and its sparsity/no-similarity problem), Bag of Words, N-Grams (bigrams/trigrams — what they capture that unigram BOW doesn't), TF-IDF (term frequency, inverse document frequency, formula). Notebook: on a small toy corpus (5-10 hardcoded sentences), build one-hot vectors manually with `pandas.get_dummies` on a token list, `CountVectorizer` for BOW, `CountVectorizer(ngram_range=(2,2))` for bigrams, and `TfidfVectorizer` — display the resulting matrices as DataFrames for comparison.
- **`03-word-embeddings`**: `notes.md` covers: why sparse BOW/TF-IDF vectors lose semantic meaning, dense word embeddings, Word2Vec CBOW vs. Skip-Gram (architecture and objective difference), and Average Word2Vec for representing a whole sentence/document. Notebook: use `gensim.models.Word2Vec` trained on a modest built-in/synthetic tokenized corpus (e.g. `nltk.corpus.gutenberg` sample or the 20-newsgroups text tokenized), train once with `sg=0` (CBOW) and once with `sg=1` (Skip-Gram), show `most_similar` results for a couple of words from both, then compute AvgWord2Vec sentence vectors and use them as features in a simple classifier (`LogisticRegression`) on a small labeled subset of `fetch_20newsgroups`, report accuracy.
- **`04-deep-learning-nlp`**: `notes.md` covers: the Keras `Embedding` layer (learned dense word vectors vs. pretrained), feeding embeddings into RNN/LSTM for text classification, padding/truncating sequences. Notebook: Keras `Embedding` + `LSTM` + `Dense` model trained on `tf.keras.datasets.imdb` for sentiment classification (this may reuse the same dataset as `06-deep-learning/03-rnn` — that's fine, the NLP angle here is the embedding-layer framing, not a new dataset), plot training curves, report test accuracy, and show 3-4 example predictions with the decoded review text (`imdb.get_word_index()`) next to true/predicted sentiment.

- [ ] **Step 1:** For each of the 4 NLP topics, write `notes.md` per the content requirements, using the exact subtopic list from that folder's current `README.md`.
- [ ] **Step 2:** For each of the 4 NLP topics, write the notebook per the content requirements.
- [ ] **Step 3:** Execute every notebook created in this task with `.venv/bin/jupyter nbconvert --to notebook --execute --inplace <path>`, confirm exit 0 and populated outputs on every code cell (same check as Task 1 Step 3).
- [ ] **Step 4:** Update `07-nlp/README.md`: flip all 4 topic rows' status column to `✅ Complete`.
- [ ] **Step 5:** Update root `README.md`: in the "Curriculum" table, change the 06 and 07 rows' Status column from `🚧 Coming soon` to `✅ Complete`; in the "Course Roadmap" ASCII diagram, no text change is needed there (it has no status markers) — skip it if so, but verify; in the `### 06 – Deep Learning` and `### 07 – NLP` prose headers, remove the `*(coming soon)*` suffix from the heading text.
- [ ] **Step 6:** `git add` the 4 NLP topic folders + `07-nlp/README.md` + root `README.md`, commit: `git commit -m "Complete NLP notebooks: text preprocessing, feature extraction, word embeddings, deep learning NLP; mark 06/07 complete in root README"`.

---

## Verification (after all 3 tasks)

Run once, from repo root, to confirm every notebook in scope executed cleanly and no topic was skipped:

```bash
.venv/bin/python - <<'EOF'
import json, pathlib
paths = list(pathlib.Path("05-machine-learning").glob("*/*.ipynb")) + \
        list(pathlib.Path("06-deep-learning").glob("*/*.ipynb")) + \
        list(pathlib.Path("07-nlp").glob("*/*.ipynb"))
bad = []
for p in paths:
    d = json.load(open(p))
    for c in d["cells"]:
        if c["cell_type"] == "code" and not c.get("execution_count"):
            bad.append(str(p))
            break
print(f"{len(paths)} notebooks checked")
print("NOT executed:", bad if bad else "none")
EOF
```

Expect `NOT executed: none` and the notebook count to equal 2 (ML: 01, 05) + 3 (ML: 08, 09, 10) + ... i.e. every topic folder under 05/06/07 that was in scope has at least one `.ipynb`.
