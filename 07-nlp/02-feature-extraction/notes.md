# 02 – Feature Extraction

| Topic | Status |
|-------|--------|
| One-Hot Encoding (OHE) for Text | ✅ Complete |
| Bag of Words (BOW) | ✅ Complete |
| N-Grams | ✅ Complete |
| TF-IDF | ✅ Complete |

## One-Hot Encoding (OHE) for Text

The simplest way to turn text into numbers is **one-hot encoding**: build a vocabulary of all unique words in the corpus, then represent each word as a vector of length $|V|$ (vocabulary size) with a single `1` at the index of that word and `0` everywhere else.

A document can then be represented as a sequence (or, in a very crude form, a sum) of its words' one-hot vectors.

**Problems with one-hot encoding of text**:
- **Extreme sparsity**: for a vocabulary of tens of thousands of words, each vector has one non-zero entry — almost all the representation is wasted `0`s, which is memory-inefficient and computationally wasteful.
- **No notion of similarity**: the vectors for *"good"* and *"great"* are exactly as different (orthogonal, dot product = 0) as the vectors for *"good"* and *"car"*. One-hot encoding treats every word as equally unrelated to every other word — there is no way to recover synonymy or semantic closeness from the representation itself.
- **Fixed, unordered vocabulary size**: adding a new word to the corpus changes the dimensionality of every vector.

These problems motivate frequency-based representations (BOW, TF-IDF) for capturing document content, and eventually dense embeddings (Word2Vec, covered in topic 03) for capturing semantic similarity.

## Bag of Words (BOW)

**Bag of Words** represents a document as a vector of word *counts* over the corpus vocabulary, ignoring grammar, word order, and sentence structure — only "what words appear, and how often" matters (hence "bag").

Construction:
1. Build the vocabulary $V$ from all documents in the corpus.
2. For each document $d$, produce a vector $\mathbf{x}_d \in \mathbb{R}^{|V|}$ where $x_{d,i}$ is the count of vocabulary word $i$ in document $d$.

BOW is implemented in scikit-learn via `CountVectorizer`, which handles tokenisation, vocabulary building, and count-matrix construction in one step. It still produces a sparse matrix (most documents don't contain most vocabulary words), but unlike one-hot encoding it retains frequency information, which is a useful (weak) signal for tasks like topic classification.

Limitation: BOW discards word order entirely — *"dog bites man"* and *"man bites dog"* produce identical BOW vectors despite opposite meaning.

## N-Grams

An **n-gram** is a contiguous sequence of $n$ tokens. BOW as described above is really "1-gram" (unigram) BOW.

- **Unigrams** ($n=1$): individual words. Cannot capture any local word order or short phrases.
- **Bigrams** ($n=2$): pairs of consecutive words, e.g. *"not good"*, *"machine learning"*. Bigrams capture short-range context that unigrams cannot — critically, they can distinguish negation (*"not good"* vs. just seeing *"good"* and *"not"* as independent, unordered counts).
- **Trigrams** ($n=3$): triples of consecutive words, e.g. *"New York City"*, capturing slightly longer fixed phrases.

In scikit-learn, `CountVectorizer(ngram_range=(2,2))` extracts only bigrams; `ngram_range=(1,2)` extracts unigrams and bigrams together. The trade-off is dimensionality: the number of possible n-grams grows roughly exponentially with $n$, so higher-order n-gram vocabularies explode in size and sparsity while giving diminishing returns — in practice bigrams and trigrams are the most commonly used beyond unigrams.

## TF-IDF

**TF-IDF (Term Frequency – Inverse Document Frequency)** improves on raw BOW counts by down-weighting words that are common across *many* documents (and therefore not discriminative) and up-weighting words that are frequent in a *specific* document but rare across the corpus.

**Term Frequency** (how important a word is *within* a document):
$$\text{TF}(t, d) = \frac{\text{count of term } t \text{ in document } d}{\text{total number of terms in } d}$$

**Inverse Document Frequency** (how rare a word is *across* the corpus):
$$\text{IDF}(t) = \log\left(\frac{N}{\text{DF}(t)}\right)$$
where $N$ is the total number of documents and $\text{DF}(t)$ is the number of documents containing term $t$. (Scikit-learn's default adds smoothing: $\text{IDF}(t) = \log\left(\frac{1+N}{1+\text{DF}(t)}\right) + 1$ to avoid division by zero and always assign a positive weight.)

**TF-IDF score**:
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Interpretation: a word that appears often in one document but rarely elsewhere (e.g. a technical term specific to one article) gets a **high** TF-IDF score — it's characteristic of that document. A word that appears in nearly every document (e.g. *"the"*, *"is"*) gets a **low** IDF and thus a low TF-IDF score even if its raw count is high, because it doesn't help distinguish documents from each other.

In scikit-learn, `TfidfVectorizer` combines tokenisation, vocabulary building, and the TF-IDF weighting (plus L2-normalisation of each row) in a single transformer, and is generally a stronger default baseline for text classification and information retrieval than raw BOW counts.
