# 01 – Text Preprocessing

| Topic | Status |
|-------|--------|
| Tokenisation & Basic Terminology | ✅ Complete |
| Stemming (NLTK) | ✅ Complete |
| Lemmatisation (NLTK) | ✅ Complete |
| Stop Words Removal | ✅ Complete |
| Parts of Speech Tagging | ✅ Complete |
| Named Entity Recognition (NER) | ✅ Complete |

## Tokenisation & Basic Terminology

**Tokenisation** is the process of splitting raw text into smaller units called **tokens** — typically words or punctuation marks. It is almost always the first step in any NLP pipeline, because downstream steps (stemming, tagging, vectorisation) operate on tokens, not on raw strings.

Basic terminology used throughout NLP:

- **Corpus**: A large, structured collection of text used for training or analysis. A corpus can be a single document repeated many times or a set of many documents (e.g. `nltk.corpus.gutenberg` is a corpus of full books).
- **Document**: One individual unit of text within a corpus — a sentence, paragraph, email, or review, depending on the granularity chosen for the task.
- **Vocabulary**: The set of unique tokens (words) that appear across a corpus. The vocabulary size directly affects the dimensionality of many text representations (one-hot, BOW).
- **Word / Token**: The atomic unit produced by tokenisation. A token is not always a dictionary word — it can be a number, punctuation mark, or sub-word piece depending on the tokenizer.

Two common levels of tokenisation:

1. **Sentence tokenisation** (`sent_tokenize`) — splits a document into sentences. Non-trivial because periods are ambiguous (`Dr.`, `U.S.A.`, decimal numbers).
2. **Word tokenisation** (`word_tokenize`) — splits a sentence (or document) into words and punctuation symbols. NLTK's `word_tokenize` uses the Punkt tokenizer, which is trained to handle contractions (`don't` → `do`, `n't`), abbreviations, and punctuation correctly.

Why tokenisation matters: every later stage (stemming, POS tagging, vectorisation) is defined over tokens. Poor tokenisation (e.g. naive `str.split()`) mishandles punctuation, contractions, and hyphenation, which propagates errors downstream.

## Stemming (NLTK)

**Stemming** reduces a word to its *stem* (root form) by chopping off suffixes using a fixed set of rules — it does **not** consult a dictionary and does **not** guarantee the output is a real word.

**Porter Stemmer**: The classic, widely-used algorithm (Porter, 1980). It applies a sequence of five phases of suffix-stripping rules (e.g. `-ing`, `-ed`, `-ational` → `-ate`). It is fast and simple but relatively aggressive.

**Snowball Stemmer**: An improved, more consistent successor to Porter (also by Martin Porter), supporting multiple languages. It fixes several edge cases where Porter over/under-stems and is generally preferred in modern pipelines when a stemmer (rather than a lemmatizer) is required.

**Over-stemming pitfalls**: Because stemming is purely rule-based, it can:
- **Over-stem**: strip too much and collapse unrelated words to the same stem. Example: *universal*, *university*, *universe* can all reduce to `univers`, even though they have different meanings.
- **Under-stem**: fail to reduce related words to the same stem. Example: *data* and *datum* may not be unified.
- Produce **non-words**: e.g. `studies` → `studi`, which is not a valid English word. This is acceptable for tasks like search indexing (where exact readability doesn't matter) but is a problem when the output needs to be human-readable.

Because of these pitfalls, stemming is best used for speed-sensitive tasks (information retrieval, search indexing) where crude normalisation is acceptable, rather than for tasks that need semantically precise, human-readable output.

## Lemmatisation (NLTK)

**Lemmatisation** reduces a word to its dictionary base form, the **lemma**, using vocabulary and morphological analysis (via WordNet in NLTK's `WordNetLemmatizer`) rather than blind suffix stripping. The output is always a valid word.

Key properties:
- **Dictionary-based**: The lemmatizer looks up the word in WordNet and returns its canonical form. `"studies"` → `"study"`, `"better"` → `"good"` (with the right POS).
- **POS-aware**: WordNet lemmatisation quality depends heavily on the part-of-speech supplied. Without a POS tag, `WordNetLemmatizer` assumes the word is a noun by default, so verbs are often left unchanged (e.g. `"running"` stays `"running"` unless tagged as a verb, in which case it becomes `"run"`). Correct results require mapping a POS tag (from `pos_tag`) to WordNet's tag set (`n`, `v`, `a`, `r`) before lemmatising.

**Lemmatisation vs. stemming**:

| Aspect | Stemming | Lemmatisation |
|---|---|---|
| Method | Rule-based suffix stripping | Dictionary + morphological analysis |
| Output | May not be a real word | Always a valid dictionary word |
| Speed | Fast | Slower (needs lookup, often POS) |
| Accuracy | Coarser, more collisions | More accurate, context-sensitive |
| Needs POS? | No | Ideally yes, for best results |

In practice: use stemming when speed matters more than precision (search engines); use lemmatisation when the output must be readable or semantically meaningful (chatbots, summarisation, feature engineering for downstream models).

## Stop Words Removal

**Stop words** are extremely common words (*the*, *is*, *in*, *and*, *a*, ...) that carry little discriminating power for many NLP tasks such as text classification or search. NLTK ships a built-in stop word list per language (`nltk.corpus.stopwords`).

Removing stop words:
- Reduces vocabulary size and noise, which shrinks the dimensionality of BOW/TF-IDF representations.
- Speeds up downstream processing.
- Can be **harmful** for tasks where function words carry meaning — e.g. sentiment analysis where negations (`not`, `no`, `never`) matter, or tasks needing exact phrase matching. In such cases, stop word lists must be customised (e.g. keep negations).

## Parts of Speech Tagging

**POS tagging** assigns each token a grammatical category — noun, verb, adjective, adverb, etc. — based on both its definition and its context in the sentence. NLTK's `pos_tag` uses the Penn Treebank tag set.

Common Penn Treebank tags:

| Tag | Meaning | Example |
|---|---|---|
| `NN` | Noun, singular | dog |
| `NNS` | Noun, plural | dogs |
| `NNP` | Proper noun, singular | London |
| `VB` | Verb, base form | run |
| `VBD` | Verb, past tense | ran |
| `VBG` | Verb, gerund/present participle | running |
| `JJ` | Adjective | quick |
| `RB` | Adverb | quickly |
| `IN` | Preposition/subordinating conjunction | in, of |
| `DT` | Determiner | the, a |
| `PRP` | Personal pronoun | he, she |
| `CC` | Coordinating conjunction | and, but |

POS tags feed directly into lemmatisation (choosing the correct lemma) and into NER and syntactic parsing (identifying candidate noun phrases).

## Named Entity Recognition (NER)

**Named Entity Recognition** identifies and classifies spans of text into predefined categories such as **PERSON**, **ORGANIZATION**, **GPE** (geo-political entity — countries, cities), **DATE**, **MONEY**, etc.

NLTK's pipeline for NER is: `word_tokenize` → `pos_tag` → `ne_chunk`. `ne_chunk` groups POS-tagged tokens into a tree, wrapping recognised entity spans in labelled subtrees (e.g. `(PERSON Barack/NNP)`), while leaving non-entity tokens as flat leaves.

NER is used for information extraction — pulling structured facts (who, where, when, how much) out of unstructured text — and is a building block for question answering, knowledge-graph construction, and document summarisation.
