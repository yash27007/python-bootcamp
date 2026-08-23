# 03 – Word Embeddings

Dense vector representations of words.

Detailed notes (distributional hypothesis, CBOW/Skip-Gram objective derivation, negative sampling, AvgWord2Vec): [notes.md](notes.md)

From-scratch: manual NumPy CBOW forward pass (one-hot context → embedding lookup → average → softmax) on a toy vocabulary, plus `gensim.Word2Vec` CBOW vs. Skip-Gram comparison and AvgWord2Vec document classification — [word-embeddings.ipynb](word-embeddings.ipynb)

| Topic | Status |
|-------|--------|
| Word Embeddings Intuition | ✅ Complete |
| Word2Vec (CBOW & Skip-Gram) | ✅ Complete |
| AvgWord2Vec | ✅ Complete |
| Word2Vec Implementation (Gensim) | ✅ Complete |
