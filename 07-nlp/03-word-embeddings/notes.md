# 03 – Word Embeddings

| Topic | Status |
|-------|--------|
| Word Embeddings Intuition | ✅ Complete |
| Word2Vec (CBOW & Skip-Gram) | ✅ Complete |
| AvgWord2Vec | ✅ Complete |
| Word2Vec Implementation (Gensim) | ✅ Complete |

## Word Embeddings Intuition

Bag of Words and TF-IDF (topic 02) represent text with **sparse**, high-dimensional vectors — one dimension per vocabulary word — where every word is treated as an independent, unrelated symbol. This has two core weaknesses:

1. **Sparsity**: vectors are mostly zeros, wasting memory and compute, and growing with vocabulary size.
2. **No semantic meaning**: the representation cannot express that *"king"* and *"queen"* are related, or that *"good"* and *"great"* are near-synonyms — cosine similarity between any two one-hot/BOW word vectors is essentially meaningless (orthogonal by construction).

**Word embeddings** solve this by learning a **dense**, low-dimensional vector (typically 50–300 dimensions) for each word, such that words with similar meaning or usage context end up close together in vector space (by cosine similarity or Euclidean distance). These vectors are *learned* from data rather than hand-designed, based on the **distributional hypothesis**: "a word is characterized by the company it keeps" — words that occur in similar contexts tend to have similar meanings.

Because embeddings are dense and continuous, they can be fed directly into neural networks and support vector arithmetic that captures analogies (the classic example: $\vec{king} - \vec{man} + \vec{woman} \approx \vec{queen}$).

## Word2Vec (CBOW & Skip-Gram)

**Word2Vec** (Mikolov et al., 2013) learns word embeddings by training a shallow neural network on a *self-supervised* prediction task over a large text corpus — no manual labels are needed, the text itself supplies the supervision signal. It has two architectural variants that differ in what predicts what:

### CBOW (Continuous Bag of Words)

CBOW predicts a **target (center) word** given its **surrounding context words** within a fixed window. E.g. given the context `["the", "cat", "on", "the"]` (window around a missing word), CBOW tries to predict `"sat"`.

- Input: the (averaged/summed) embeddings of the context words.
- Output: a probability distribution over the vocabulary for the center word.
- Objective: maximize $P(w_t \mid w_{t-k}, \ldots, w_{t-1}, w_{t+1}, \ldots, w_{t+k})$.
- CBOW smooths over context (it averages multiple context words), which makes it **faster to train** and works well on **frequent words** and **larger corpora**.

### Skip-Gram

Skip-Gram does the reverse: it predicts the **surrounding context words** given a single **center word**. E.g. given `"sat"`, it tries to predict `"the"`, `"cat"`, `"on"`, `"the"` (each context word individually).

- Input: the embedding of the center word.
- Output: a probability distribution over the vocabulary for each context position.
- Objective: maximize $\sum_{-k \le j \le k, j \neq 0} P(w_{t+j} \mid w_t)$.
- Skip-Gram generates more training pairs per sentence (one per context word), so it is **slower to train** but tends to work **better with smaller datasets and rare words**, since each rare word gets multiple update opportunities as a center word.

Both architectures use a shallow network (an embedding lookup layer feeding into a softmax over the vocabulary, in practice approximated via negative sampling or hierarchical softmax for efficiency). The trained embedding weight matrix — not the prediction task itself — is the actual product; the prediction task is only a device for learning useful vectors.

| | CBOW | Skip-Gram |
|---|---|---|
| Predicts | center word from context | context words from center |
| Training speed | faster | slower |
| Good for | frequent words, larger corpora | rare words, smaller corpora |
| `gensim` flag | `sg=0` | `sg=1` |

## AvgWord2Vec

Word2Vec produces one dense vector *per word*, but many downstream tasks (classification, clustering, search) need a single fixed-length vector *per sentence or document*.

**Average Word2Vec (AvgWord2Vec)** is the simplest way to bridge that gap: represent a sentence/document as the **element-wise average** of the Word2Vec vectors of its constituent words:

$$\vec{v}_{doc} = \frac{1}{n}\sum_{i=1}^{n} \vec{v}_{w_i}$$

where $w_1, \ldots, w_n$ are the (in-vocabulary) tokens of the document and $\vec{v}_{w_i}$ is each word's learned embedding.

This produces a dense, fixed-size feature vector regardless of document length, which can be fed directly into any standard classifier (logistic regression, SVM, random forest, etc.) — a large step up from sparse BOW/TF-IDF features while still being simple to compute. Its main limitation is that averaging discards word order and can dilute the signal from a few important words in a long document, but it remains a strong, cheap baseline before moving to sequence models (RNN/LSTM, topic 04) that respect word order.

## Word2Vec Implementation (Gensim)

The `gensim` library provides an efficient `Word2Vec` implementation. Typical usage:

```python
from gensim.models import Word2Vec
model = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, sg=0)  # CBOW
model_sg = Word2Vec(sentences=tokenized_corpus, vector_size=100, window=5, min_count=1, sg=1)  # Skip-Gram
```

Key hyperparameters: `vector_size` (embedding dimensionality), `window` (context window size), `min_count` (ignore words below this frequency), and `sg` (0 = CBOW, 1 = Skip-Gram). Once trained, `model.wv.most_similar("word")` returns the nearest neighbours by cosine similarity, and `model.wv["word"]` returns the raw embedding vector — these are exactly the tools used in this topic's notebook to compare CBOW vs. Skip-Gram and to build AvgWord2Vec document features for a downstream classifier.
