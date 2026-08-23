# 03 – Word Embeddings

## Problem

`07-nlp/02-feature-extraction` closed with an unresolved limitation: BOW and TF-IDF vectors are sparse (one dimension per vocabulary word) and, more importantly, carry **no notion of semantic similarity** — the vector for `"good"` is exactly as unrelated to `"great"` as it is to `"car"`, because every word occupies its own independent, orthogonal dimension. Under TF-IDF, `"king"` and `"queen"` are as unrelated as `"king"` and `"banana"`. **How do we represent a word as a numeric vector such that words with related meaning end up close together, so a model can generalize across synonyms and related concepts instead of treating every distinct word string as an isolated symbol?**

## Intuition

Read enough English text and a pattern emerges: words that mean similar things tend to show up in similar surroundings. `"cat"` and `"dog"` both appear near words like `"pet"`, `"feed"`, `"vet"`, `"leash"`; `"king"` and `"queen"` both appear near `"throne"`, `"crown"`, `"palace"`, `"reign"`. You don't need a dictionary to notice `"cat"` and `"dog"` are related — the *company they keep* gives it away. This is the **distributional hypothesis**: "a word is characterized by the company it keeps" (Firth, 1957). If two words tend to occur in the same contexts across a large corpus, they probably mean similar things.

**Word embeddings** operationalize this: learn a dense, low-dimensional vector (typically 50–300 dimensions, versus the ~10,000+ dimensions of a BOW vocabulary) for each word directly from co-occurrence patterns in raw text, with no manual labels. Words that share contexts end up with similar vectors (high cosine similarity), and — because the geometry is learned rather than hand-assigned — directions in the space can capture relationships: the classic literature example is $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$. This notebook's own toy corpus shows a milder version of the same idea empirically (see Experiment): Skip-Gram's nearest neighbours of `"queen"` include `"king"`, `"knave"`, and `"hearts"` — words a human would also associate with royalty and card-game context — recovered purely from word co-occurrence, with no hand-built thesaurus.

## Why simpler approaches fail

Formally, one-hot encoding assigns word $v_i$ the vector $\mathbf{e}_i \in \{0,1\}^{|V|}$ with a `1` at position $i$ and `0` elsewhere. For any two distinct words, $\mathbf{e}_i \cdot \mathbf{e}_j = 0$ — **every pair of distinct words is exactly as dissimilar as every other pair**, by construction, regardless of meaning. BOW and TF-IDF (topic 02) build document vectors as weighted sums of these same one-hot word vectors; reweighting by frequency (TF-IDF) changes *how much* a word's dimension counts, but never adds a shared axis between two different words' dimensions. No amount of reweighting one-hot counts can manufacture similarity structure that isn't there in the representation to begin with — the dimensionality is tied to vocabulary size ($|V|$, often tens of thousands), and semantic relatedness would have to be recovered downstream by the model from scratch, learning it separately for every task, every time. Word embeddings instead learn that similarity structure once, upfront, from the statistics of a large corpus, and hand a downstream model vectors that already encode it.

## Mathematical foundation

### The distributional hypothesis, formalized

Model each word $w$ by the distribution of words that occur in its context window across the corpus. Two words $w_1, w_2$ are semantically similar to the extent their context-word distributions are similar. Word2Vec (Mikolov et al., 2013) doesn't compute this distribution directly (that would still be a huge sparse co-occurrence matrix); instead it trains a shallow neural network on a **self-supervised prediction task** — predicting words from their context, or context from words — and treats the network's *learned weights*, not its predictions, as the actual product. If the network can predict context well, its internal embedding vectors must have absorbed the co-occurrence statistics.

Let $V$ be the vocabulary, $|V| = n$. Every word $w_i$ gets **two** embedding vectors during training: an "input" (center-word) vector $\mathbf{v}_i \in \mathbb{R}^d$ and an "output" (context-word) vector $\mathbf{u}_i \in \mathbb{R}^d$, stored as rows of two weight matrices $V_{in}, V_{out} \in \mathbb{R}^{n \times d}$ ($d \ll n$, e.g. $d=100$). Only $V_{in}$ is kept as "the" word embeddings after training (in practice `gensim` just returns $V_{in}$ via `model.wv`).

### CBOW objective — derived

CBOW predicts the **center word** $w_t$ from its surrounding **context words** $w_{t-k}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+k}$ within window size $k$.

1. **Input representation**: look up each context word's input embedding and average them:
$$\mathbf{h} = \frac{1}{2k} \sum_{-k \le j \le k,\, j \ne 0} \mathbf{v}_{w_{t+j}}$$
   ($\mathbf{h} \in \mathbb{R}^d$ — this averaging is exactly what "CBOW" (Continuous **Bag** Of Words) refers to: word order within the window is discarded, only the set of context words matters.)
2. **Score every vocabulary word** as a candidate center word, via a dot product with each word's output embedding:
$$z_i = \mathbf{u}_i^\top \mathbf{h}, \quad i = 1, \ldots, n$$
3. **Softmax** turns scores into a probability distribution over the whole vocabulary:
$$P(w_t = i \mid \text{context}) = \frac{\exp(z_i)}{\sum_{j=1}^{n} \exp(z_j)} = \frac{\exp(\mathbf{u}_i^\top \mathbf{h})}{\sum_{j=1}^{n} \exp(\mathbf{u}_j^\top \mathbf{h})}$$
4. **Loss**: cross-entropy against the true center word $t$ — i.e. maximize $\log P(w_t \mid \text{context})$, equivalently minimize
$$\mathcal{L}_{\text{CBOW}} = -\log P(w_t \mid w_{t-k}, \ldots, w_{t+k}) = -z_t + \log \sum_{j=1}^{n} \exp(z_j)$$

Training performs stochastic gradient descent on $\mathcal{L}_{\text{CBOW}}$ over every window in the corpus, updating $V_{in}$ and $V_{out}$. Because $\mathbf{h}$ is a single averaged vector regardless of which specific context words appeared, CBOW smooths over context — one gradient update touches all context words' embeddings identically through the shared average, which trains faster and works well on frequent words with abundant context examples.

### Skip-Gram objective — derived

Skip-Gram reverses the direction: predict each **context word** individually from the **center word** $w_t$.

1. **Input representation**: no averaging needed — just the center word's own input embedding, $\mathbf{h} = \mathbf{v}_{w_t}$.
2. **Score and softmax**, once per context position $j$:
$$P(w_{t+j} = i \mid w_t) = \frac{\exp(\mathbf{u}_i^\top \mathbf{v}_{w_t})}{\sum_{l=1}^{n} \exp(\mathbf{u}_l^\top \mathbf{v}_{w_t})}$$
3. **Loss** sums the negative log-probability over every context position in the window:
$$\mathcal{L}_{\text{SG}} = -\sum_{-k \le j \le k,\, j \ne 0} \log P(w_{t+j} \mid w_t)$$

Skip-Gram generates $2k$ training pairs per window (one per context position) instead of CBOW's one, so each occurrence of a rare center word gets $2k$ separate gradient updates on its embedding rather than one averaged-in update — this is why Skip-Gram tends to learn better vectors for rare words and small corpora, at the cost of more total training pairs and slower training.

### Making the softmax tractable: negative sampling

Both objectives above require $\sum_{j=1}^n \exp(z_j)$ — a sum over the **entire vocabulary** — for every single training example, which is computationally prohibitive when $n$ is tens of thousands to millions of words. **Negative sampling** replaces the $n$-way softmax classification with a much cheaper **binary classification**: for each true (center, context) pair, sample $m$ random "negative" words from the vocabulary (words that did *not* actually appear in that context) and train the model to assign high probability to the true pair and low probability to the negative pairs, using the sigmoid $\sigma(x) = 1/(1+e^{-x})$ instead of softmax:
$$\mathcal{L}_{\text{NS}} = -\log \sigma(\mathbf{u}_{w_O}^\top \mathbf{v}_{w_I}) - \sum_{i=1}^{m} \log \sigma(-\mathbf{u}_{w_i}^\top \mathbf{v}_{w_I})$$
where $w_I$ is the input word, $w_O$ the true output word, and $w_1, \ldots, w_m$ are sampled negatives (typically drawn from a unigram distribution raised to the $3/4$ power, which oversamples rare words relative to their raw frequency). This turns each update from $O(n)$ work into $O(m)$ work, $m$ typically 5–20 — the difference between infeasible and fast at web-corpus scale. (An alternative, hierarchical softmax, restructures the $n$-way softmax as a binary tree walk in $O(\log n)$ instead; both are implementation-level tricks for the same underlying CBOW/Skip-Gram objectives above, not different objectives.) `gensim`'s default `Word2Vec` uses negative sampling internally.

## Algorithm

1. Build the vocabulary $V$ from the tokenized corpus; assign each word an index.
2. Initialize $V_{in}, V_{out} \in \mathbb{R}^{|V| \times d}$ with small random values.
3. Slide a window of size $k$ over every sentence in the corpus. At each position $t$:
   - **CBOW**: average the context words' input embeddings → $\mathbf{h}$; compute softmax (or negative-sampling loss) over the center word; backpropagate into $V_{in}$ (context words' rows) and $V_{out}$.
   - **Skip-Gram**: for each context position $j$, use the center word's input embedding directly; compute softmax (or negative-sampling loss) for that one context word; backpropagate.
4. Update $V_{in}, V_{out}$ via SGD (or Adam) after each example/mini-batch.
5. Repeat over multiple epochs until convergence.
6. Discard $V_{out}$; keep $V_{in}$ as the final word embedding matrix (`model.wv` in `gensim`).

## From-scratch implementation

The notebook (`word-embeddings.ipynb`, "From-scratch: tiny NumPy CBOW forward pass" section) hand-traces exactly one CBOW forward pass on a 5-word toy vocabulary built from the sentence `"the cat sat on mat"`, with `vocab = {the: 0, cat: 1, sat: 2, on: 3, mat: 4}`, embedding dimension $d=3$, window size $k=1$, center word `"sat"` (index 2) with context words `["cat", "on"]`:

1. One-hot encode each context word: $\mathbf{e}_{cat} = [0,1,0,0,0]$, $\mathbf{e}_{on} = [0,0,0,1,0]$.
2. Small fixed $V_{in} \in \mathbb{R}^{5 \times 3}$ (hand-chosen integers, so the arithmetic is traceable by hand) — one-hot-encoding then matrix-multiplying by $V_{in}$ is exactly a row lookup, e.g. $\mathbf{e}_{cat}^\top V_{in} = V_{in}[1, :]$.
3. Average the two looked-up rows: $\mathbf{h} = \frac{1}{2}(V_{in}[1,:] + V_{in}[3,:])$.
4. Small fixed $V_{out} \in \mathbb{R}^{3 \times 5}$; compute scores $\mathbf{z} = \mathbf{h}^\top V_{out} \in \mathbb{R}^5$.
5. Softmax $\mathbf{z}$ into a probability distribution over all 5 vocabulary words; read off $P(\text{"sat"})$ — the value the training objective wants to push toward 1 — and compute the cross-entropy loss $-\log P(\text{"sat"})$ against the true center word.

This is the exact mechanic — one-hot context → embedding lookup → average → score → softmax — that `gensim.Word2Vec(sg=0)` performs internally, millions of times, with negative sampling in place of the full softmax and gradient descent updating $V_{in}, V_{out}$ instead of them being fixed by hand.

## Practical implementation

`gensim` provides a production-grade, heavily optimized `Word2Vec` implementation performing the same lookup → average/center → softmax(-approximation) computation as the from-scratch section, at corpus scale with negative sampling:

```python
from gensim.models import Word2Vec
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, sg=0)  # CBOW
model_sg = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, sg=1)  # Skip-Gram
```

| | CBOW | Skip-Gram |
|---|---|---|
| Predicts | center word from context | context words from center |
| Input to the network | averaged context embeddings ($\mathbf{h}$ above) | single center-word embedding |
| Training pairs per window | 1 | $2k$ |
| Training speed | faster | slower |
| Good for | frequent words, larger corpora | rare words, smaller corpora |
| `gensim` flag | `sg=0` | `sg=1` |

Key hyperparameters: `vector_size` ($d$, embedding dimensionality), `window` ($k$, context window size), `min_count` (ignore words below this frequency), and `sg` (0 = CBOW, 1 = Skip-Gram). Once trained, `model.wv.most_similar("word")` returns the nearest neighbours by cosine similarity in the learned $V_{in}$ space, and `model.wv["word"]` returns the raw embedding vector.

Word2Vec produces one dense vector *per word*, but many downstream tasks (classification, clustering, search) need a single fixed-length vector *per document*. **AvgWord2Vec** is the simplest bridge: represent a document as the element-wise average of its constituent words' Word2Vec vectors:
$$\vec{v}_{doc} = \frac{1}{n}\sum_{i=1}^{n} \vec{v}_{w_i}$$
where $w_1, \ldots, w_n$ are the (in-vocabulary) tokens of the document. This produces a dense, fixed-size feature vector regardless of document length — a large step up from sparse BOW/TF-IDF features while remaining simple to compute — but the averaging discards word order and can dilute the signal from a few important words in a long document. It remains a strong, cheap baseline before moving to sequence models (RNN/LSTM, `07-nlp/04-deep-learning-nlp`) that respect word order and don't collapse a sentence to a single averaged point.

## Experiment

**Hypothesis**: on a small, single-book corpus (Alice in Wonderland, ~1,500 sentences), Skip-Gram — which gives rare/content words more individual gradient updates — should surface more semantically coherent nearest neighbours than CBOW, which smooths everything through an averaged context vector; and AvgWord2Vec features, despite discarding word order, should still substantially outperform chance on a topically distinct two-category text classification task.

**Setup**: `word-embeddings.ipynb` trains CBOW (`sg=0`) and Skip-Gram (`sg=1`) Word2Vec models (100-dim, window 5) on `nltk.corpus.gutenberg`'s *Alice in Wonderland* (1,491 tokenized sentences, vocabulary size 1,014), then compares `most_similar` neighbours for `"alice"` and `"queen"`. Separately, it trains a fresh Word2Vec model on 20-Newsgroups documents from two categories (`rec.sport.hockey` vs `sci.space`, 1,193 train / 793 test documents), builds AvgWord2Vec features per document, and fits a logistic regression classifier on those features.

**Result**: For `"queen"`, CBOW's top neighbours (`with`, `his`, `voice`, `rabbit`, `white`) are largely generic high-frequency words, while Skip-Gram's top neighbours (`executioner`, `king`, `knave`, `hearts`, `pointing`) are thematically on-point — `"king"` and `"knave"` are royalty/playing-card terms that co-occur with `"queen"` throughout the book's Queen-of-Hearts scenes — confirming the hypothesis that Skip-Gram surfaces more semantically coherent neighbours on this small corpus. The AvgWord2Vec + logistic regression classifier reached **95.0% test accuracy** (precision/recall/F1 all ≈0.93–0.97 on both classes) distinguishing hockey from space articles, confirming that even order-blind averaged embeddings carry strong topical signal for tasks where individual salient words (e.g. `"puck"`, `"orbit"`) dominate the vocabulary difference between classes.

**Limitations**: a 1,491-sentence corpus is tiny by Word2Vec standards (production Word2Vec is typically trained on billions of tokens), so neighbour quality here is illustrative, not representative of large-scale embeddings; and the 95% classification accuracy reflects a relatively easy, topically well-separated two-class problem — it doesn't establish how AvgWord2Vec would perform on finer-grained or more ambiguous classification tasks.

## Failure modes

- **Out-of-vocabulary (OOV) words**: Word2Vec only has a vector for words seen during training (with frequency $\ge$ `min_count`). Any word absent from the training corpus — a typo, a new term, a rare proper noun — has no embedding at all, and downstream code must handle this explicitly (e.g. `avg_word2vec` in this topic's notebook skips OOV tokens when averaging).
- **Polysemy — one vector per word regardless of sense**: Word2Vec assigns each *word string* exactly one vector, so `"bank"` gets a single embedding that has to represent both "river bank" and "financial bank" senses simultaneously, pulled toward whichever sense's contexts are more frequent in the training corpus. This is a real, unresolved limitation of static embeddings — fixing it requires context-dependent ("contextual") embeddings that produce a different vector for the same word depending on its sentence (e.g. Transformer-based models). That is genuinely **out of scope for this repository at present** — a **planned**, not yet covered, direction — and this topic's embeddings should not be treated as if they already solve polysemy.
- **No compositionality beyond averaging**: nothing in Word2Vec itself represents phrases or sentences; AvgWord2Vec's word-order blindness (`"dog bites man"` and `"man bites dog"` average to the identical vector) is a direct consequence of using embeddings only via averaging, not a flaw in the embeddings themselves.
- **Frozen after training**: a trained Word2Vec model's vectors don't update as language evolves or as new domain-specific vocabulary appears, unless retrained on fresh data.

## Real-world usage

Pretrained word embeddings (Word2Vec, GloVe, FastText) are used as a fast, cheap semantic feature source: seeding the `Embedding` layer of downstream neural networks (especially useful when task-specific labeled data is limited, `07-nlp/04-deep-learning-nlp`), computing document similarity for search/retrieval and recommendation, detecting near-duplicate or paraphrased text, and as quick baseline features for text classification before investing in heavier sequence models. FastText extends the same CBOW/Skip-Gram machinery to subword n-grams, which partially addresses the OOV problem by composing an embedding for unseen words from their known character n-grams.

## Mental model

**A word embedding is a word's address in a map built entirely from who it hangs out with.** CBOW asks "given these neighbours, who lives here?"; Skip-Gram asks "given who lives here, who are the neighbours?" — same map, opposite question, and the map itself (not the answer to either question) is the useful artifact.

## Questions to think about

1. Why does averaging context-word embeddings (CBOW) versus using each context word individually (Skip-Gram) lead to CBOW training faster but Skip-Gram handling rare words better? Trace the argument back to how many gradient updates each rare word's embedding receives per corpus pass.
2. If you trained Word2Vec twice on the exact same corpus with the same hyperparameters but different random initializations, would `model.wv["king"]` be identical both times? Would `model.wv.most_similar("king")`'s *ranking* of neighbours be identical? Why or why not — and what does that imply about comparing raw embedding vectors across two independently trained models?
3. Negative sampling replaces an $O(|V|)$ softmax with an $O(m)$ binary-classification approximation. What would happen to training quality if $m=1$? What would happen to training speed if $m = |V|-1$ (i.e. every other word is a "negative")? What does this trade-off suggest about choosing $m$ in practice?
4. AvgWord2Vec reached 95% accuracy on hockey-vs-space classification by averaging away word order entirely. Construct a two-class classification task where you'd expect AvgWord2Vec to perform close to chance, and explain what property of your constructed task breaks the "salient words dominate the average" assumption that made hockey-vs-space easy.
5. A word embedding trained on news text is applied to medical text containing many domain-specific terms absent from the training vocabulary. Beyond simply "retrain on medical text," what does the from-scratch CBOW forward pass in this topic's notebook tell you about *why* those terms have no vector at all, rather than merely an inaccurate one?
