# 02 – Feature Extraction

## Problem

Topic 01's preprocessing pipeline turns raw text into a clean, normalized stream of tokens — but every machine learning model (linear models, tree ensembles, neural networks) computes on numbers, not strings. **How do we turn a normalized token stream into fixed-length numeric vectors a model can actually consume, while retaining as much of the text's informative content as possible?**

## Intuition

Suppose you have a corpus of short reviews about cats and cars, and want a classifier to tell the two topics apart. The simplest possible idea: build a vocabulary of every unique word across the corpus, then represent each document by which of those vocabulary words it contains, and how often. A review containing `"cat"`, `"purring"`, `"mat"` looks numerically different from one containing `"car"`, `"engine"`, `"road"`, simply because they light up different vocabulary positions. This is the core idea behind every representation in this topic — one-hot, bag-of-words, and TF-IDF are all different ways of turning "which words are present, and how much" into a numeric vector, each fixing a specific weakness in the one before it.

## Why simpler approaches fail

The most naive numeric representation is **one-hot encoding**: build a vocabulary of $|V|$ unique words, and represent each word as a vector of length $|V|$ with a single `1` at that word's index and `0` everywhere else. This fails in two concrete ways:

1. **Extreme sparsity.** For a vocabulary of even a few thousand words, each word's vector has exactly one non-zero entry among thousands of zeros — almost the entire representation is wasted space, which is memory-inefficient and computationally wasteful to store and multiply through a model.
2. **No notion of similarity.** The vectors for `"good"` and `"great"` are exactly as different (orthogonal, cosine similarity 0) as the vectors for `"good"` and `"car"` — one-hot encoding treats every word as equally unrelated to every other word, by construction, because each word occupies its own independent dimension with no shared structure. There is no way to recover synonymy or semantic closeness from the representation itself, no matter how the vectors are combined.

Frequency-based representations (bag-of-words, TF-IDF, this topic) fix the sparsity problem only partially (they're still sparse, but they at least carry frequency information as a weak topic signal) and don't fix the similarity problem at all — a BOW or TF-IDF vector for `"good"` is still orthogonal to the vector for `"great"`. **This is exactly the limitation that dense word embeddings (Word2Vec, `07-nlp/03-word-embeddings`) are built to solve**: instead of one dimension per vocabulary word, embeddings learn a small, dense vector per word from co-occurrence patterns in a large corpus, such that semantically related words end up close together in vector space. That is a separate, later topic — the representations in this notes.md are the frequency-based family that predates and motivates it, not a replacement for it.

## Mathematical foundation

### One-hot encoding

Given vocabulary $V = \{v_1, \ldots, v_{|V|}\}$, word $v_i$'s one-hot vector $\mathbf{e}_i \in \{0,1\}^{|V|}$ has $\mathbf{e}_i[i] = 1$ and $\mathbf{e}_i[j] = 0$ for $j \neq i$. Any two distinct words' vectors are orthogonal: $\mathbf{e}_i \cdot \mathbf{e}_j = 0$ for $i \neq j$, which is the formal statement of "no notion of similarity" above.

### Bag of Words (BOW)

For vocabulary $V$ built from all documents in the corpus, document $d$'s BOW vector $\mathbf{x}_d \in \mathbb{R}^{|V|}$ has $x_{d,i} = $ the count of vocabulary word $i$ in document $d$. Unlike one-hot per-word vectors, a document-level BOW vector retains frequency information but still ignores word order and grammar entirely — *"dog bites man"* and *"man bites dog"* produce identical BOW vectors despite opposite meaning, since only counts of individual words are tracked.

### N-grams

An n-gram is a contiguous sequence of $n$ tokens; BOW as defined above is unigram ($n=1$) BOW. Bigrams ($n=2$, e.g. *"not good"*) and trigrams ($n=3$, e.g. *"New York City"*) extend the same counting construction to sequences of tokens instead of single tokens, capturing local word-order context unigrams cannot (critically, bigrams can distinguish negation, since *"not good"* is one bigram feature rather than two independent, unordered unigram counts of `"not"` and `"good"`). The number of possible distinct n-grams grows roughly exponentially with $n$, so higher-order n-gram vocabularies explode in size and sparsity for diminishing returns — bigrams and trigrams are the most commonly used beyond unigrams in practice.

### TF-IDF

TF-IDF improves on raw BOW counts by down-weighting words common across *many* documents (not discriminative) and up-weighting words frequent in a *specific* document but rare across the corpus.

**Term Frequency** — how important a word is *within* a document:
$$\text{TF}(t, d) = \frac{\text{count of term } t \text{ in document } d}{\text{total number of terms in } d}$$

**Document Frequency** — how many documents (out of $N$ total) contain term $t$ at all:
$$\text{DF}(t) = |\{d \in \text{corpus} : t \in d\}|$$

**Inverse Document Frequency** — the reciprocal of document frequency, expressing rarity:
$$\text{IDF}(t) = \log\left(\frac{N}{\text{DF}(t)}\right)$$

**Why the log?** Document frequency itself grows linearly with corpus size, so an un-dampened ratio $N / \text{DF}(t)$ would let extremely rare terms dominate the score by orders of magnitude relative to moderately common ones, swamping the TF factor and making the weighting numerically unstable as the corpus grows. Taking the logarithm compresses this ratio: a term appearing in 1 of 1,000,000 documents ($\text{ratio}=10^6$) gets $\log(10^6) \approx 13.8$, not a million-fold multiplier — so IDF still rewards rarity but on a scale comparable to TF, and the *relative* ordering of IDF across terms is preserved ($\log$ is monotonic) while the *absolute* spread is compressed. Without the log, a single very rare term could dominate a document's TF-IDF vector regardless of how much of the document it actually made up.

scikit-learn's default adds smoothing to avoid division by zero and always assign a positive weight:
$$\text{IDF}(t) = \log\left(\frac{1+N}{1+\text{DF}(t)}\right) + 1$$

**TF-IDF score** — the product of the two factors:
$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \text{IDF}(t)$$

Interpretation: a word appearing often in one document but rarely elsewhere (e.g. a technical term specific to one article) gets a **high** TF-IDF score — it's characteristic of that document. A word appearing in nearly every document (e.g. *"the"*, *"is"*) gets a **low** IDF and thus a low TF-IDF score even if its raw count is high, because it doesn't help distinguish documents from each other.

## Algorithm

1. Preprocess and tokenize every document in the corpus (topic 01).
2. Build the vocabulary $V$: the set of unique tokens (or n-grams) across all documents.
3. For BOW: for each document $d$, count occurrences of each vocabulary term, producing $\mathbf{x}_d \in \mathbb{R}^{|V|}$.
4. For TF-IDF: compute the term-frequency matrix as in step 3 (normalized by document length), compute $\text{DF}(t)$ for every term across the whole corpus, compute $\text{IDF}(t) = \log(N/\text{DF}(t))$ (or the smoothed variant) for every term, then multiply each document's TF row element-wise by the IDF vector to get the TF-IDF matrix.
5. Optionally L2-normalize each document's row vector so document length doesn't dominate comparisons.

## From-scratch implementation

Implemented in `feature-extraction.ipynb`: a manual NumPy TF-IDF computation on a tiny 3-document toy corpus, following the Algorithm section step by step —

1. **Term frequency matrix**: tokenize each document by splitting on whitespace, build the vocabulary as the sorted set of unique tokens, and fill a `(num_docs, vocab_size)` count matrix, then divide each row by that document's total token count to get raw TF.
2. **Document frequency**: for each vocabulary term, count how many documents (rows) contain a non-zero count.
3. **IDF with log dampening**: apply $\text{IDF}(t) = \log\left(\frac{1+N}{1+\text{DF}(t)}\right) + 1$ (scikit-learn's smoothed formula, so the from-scratch numbers are directly comparable to `TfidfVectorizer`'s defaults) to the document-frequency vector.
4. **TF × IDF**: multiply the TF matrix by the IDF vector (broadcast across rows) to get the raw TF-IDF matrix, then L2-normalize each row (divide by its Euclidean norm) to match scikit-learn's default normalization.
5. **Cross-check**: run `TfidfVectorizer().fit_transform(...)` on the exact same 3-document corpus and compare its dense output to the manually-computed matrix with `np.allclose`, confirming they match to floating-point precision.

This confirms the Mathematical foundation section's formulas are not a simplification of what `TfidfVectorizer` actually computes — manually building the TF matrix, DF counts, smoothed IDF, and their product, then L2-normalizing, reproduces scikit-learn's output exactly. The only thing the library implementation adds is a much faster, sparse, vectorized version of the same arithmetic, plus consistent tokenization/vocabulary-building machinery.

## Practical implementation

`feature-extraction.ipynb` walks through all four representations on a small hardcoded 8-document toy corpus about cats and cars: one-hot encoding via `pandas.get_dummies` on a single document's tokens (demonstrating sparsity and computing cosine similarity between one-hot vectors to show they're always orthogonal), `CountVectorizer` for unigram BOW, `CountVectorizer(ngram_range=(2,2))` for bigrams (showing bigrams distinguish `"not happy"` from independent `"not"`/`"happy"` counts), and `TfidfVectorizer` for TF-IDF, with a final comparison table contrasting BOW and TF-IDF weights for a common word (`"the"`) versus a rare, document-distinctive word (`"playful"`).

The mapping back to "From-scratch implementation" is direct: `TfidfVectorizer`'s output on the toy corpus used for the from-scratch cross-check is produced by exactly the same TF/DF/IDF/L2-normalize pipeline built manually there — the 8-document corpus in this practical notebook is a larger instance of the same computation, run through the optimized sparse-matrix library implementation instead of dense NumPy loops.

## Experiment

**Hypothesis (stated before running):** TF-IDF weighting will systematically down-weight a word that appears in nearly every document (like `"the"`) relative to its raw BOW count, while up-weighting a word that appears in only one or two documents (like `"playful"`) relative to its raw count — because IDF penalizes high document frequency and rewards rarity, per the Mathematical foundation section.

**Setup:** `feature-extraction.ipynb`'s final comparison cell builds a DataFrame with four columns — raw BOW count and TF-IDF weight, for both `"the"` and `"playful"` — across all 8 documents in the toy corpus, allowing direct row-by-row comparison of how the two representations score the same words in the same documents.

**Result:** the comparison table shows `"the"` has a high BOW count in nearly every document but a correspondingly low TF-IDF weight, while `"playful"` (appearing in only one document) has a low BOW count but a comparatively high TF-IDF weight in the document where it occurs.

**Interpretation:** the results match the hypothesis directly — TF-IDF successfully separates "frequent everywhere, therefore uninformative" words from "frequent in one place, therefore distinctive" words, which is exactly the discriminative signal a downstream classifier benefits from that raw BOW counts don't provide on their own.

**Limitations:** this is a single small, hand-constructed 8-document corpus about two easily-separable topics (cats and cars); it demonstrates the qualitative IDF-down-weighting effect clearly, not TF-IDF's quantitative benefit on a large, noisy, real-world corpus with overlapping topics.

## Failure modes

- **Vocabulary mismatch at inference time**: BOW/TF-IDF vectorizers are fit on a training corpus's vocabulary; any word seen only at inference time that wasn't in the training vocabulary is silently dropped (out-of-vocabulary), losing potentially important signal.
- **No semantic similarity** (inherited from one-hot, not fixed by BOW or TF-IDF): `"good"` and `"great"` remain orthogonal under TF-IDF just as under one-hot — TF-IDF changes *weights*, not the underlying one-dimension-per-word structure, so it cannot recognize synonyms. This is the exact gap `03-word-embeddings` closes.
- **Word-order loss**: BOW and TF-IDF (beyond the specific bigrams/trigrams explicitly included) discard grammar and word order — *"dog bites man"* vs. *"man bites dog"* remain indistinguishable at the unigram level.
- **Sparsity and dimensionality at scale**: a real-world vocabulary can run into the hundreds of thousands of terms; even though BOW/TF-IDF matrices are stored sparsely, very high-cardinality vocabularies (e.g. from including many n-grams) can still blow up memory and slow down training.
- **IDF instability on tiny or skewed corpora**: with very few documents, or a corpus where nearly every term appears in nearly every document, IDF values compress toward a narrow range and TF-IDF degenerates toward looking like plain TF, losing its discriminative power.

## Real-world usage

TF-IDF remains a strong, cheap, interpretable baseline for text classification, document search/ranking (e.g. classic search engine relevance scoring), and information retrieval more broadly — it is often the first representation tried before moving to embeddings or transformer-based encoders, precisely because it requires no training beyond counting and produces directly interpretable per-term weights. Bag-of-words and n-gram features remain useful for tasks like spam detection and simple topic classification, where computational cheapness and interpretability outweigh the need for deep semantic understanding. Where similarity between words genuinely matters — search query expansion, recommendation from text, semantic clustering — the pipeline moves to dense embeddings (`03-word-embeddings`) precisely because of the "no notion of similarity" limitation identified in "Why simpler approaches fail."

## Mental model

Each representation in this topic answers "how much information about word *identity* and *frequency* can we encode without any notion of word *meaning*": one-hot encodes identity alone (maximally sparse, zero similarity), BOW adds frequency (still zero similarity), n-grams add a little local order, and TF-IDF re-weights frequency by how *discriminative* a word is across the corpus — but none of them, no matter how the weighting is tuned, can make `"good"` and `"great"` numerically close, because each word still occupies its own independent dimension. That last gap is precisely what dense embeddings are built to close.

## Questions to think about

1. The from-scratch TF-IDF cross-check uses scikit-learn's smoothed IDF formula rather than the plain $\log(N/\text{DF}(t))$ from the Mathematical foundation section. If you instead used the unsmoothed formula on the toy corpus, for which terms (if any) would the computation break, and why does the smoothing specifically avoid that?
2. If a document is duplicated many times in the corpus (e.g. accidental data duplication), how does that change IDF for the words it contains? Would raw BOW counts or TF-IDF weights be more distorted by this kind of duplication, and why?
3. Bigrams capture `"not good"` as a single feature distinct from independent `"not"` and `"good"` counts. Construct a phrase where a *trigram* captures meaning that even the corresponding bigrams miss — and explain what specifically about word order across three positions (not two) is required.
4. TF-IDF's IDF term is computed once from the training corpus and then applied to new documents. What happens to a new document's TF-IDF vector if it consists entirely of rare technical jargon that appeared in only a single training document — does the representation still meaningfully distinguish it from other documents? Why or why not?
5. One-hot, BOW, and TF-IDF all fail to encode word similarity, but for different structural reasons than "the numbers are wrong." Explain, in terms of vector *dimensionality and construction* (not weighting), why no amount of re-weighting a fixed one-dimension-per-vocabulary-word representation can ever make synonymous words close together — and what fundamentally different design choice (foreshadowing `03-word-embeddings`) would be required to fix it.
