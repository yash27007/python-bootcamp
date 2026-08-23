# 01 – Text Preprocessing

## Problem

Machine learning models operate on structured numeric input, but raw text is unstructured and highly inconsistent: the same underlying concept can appear as `"Run"`, `"run"`, `"running"`, `"ran"`, or `"runs"`; sentences carry punctuation, capitalization, and filler words that vary between writers; and a corpus of any real size contains a combinatorial explosion of surface forms for a comparatively small set of underlying concepts. Before text can be turned into numeric features at all (topic 02), it needs to be normalized into a finite, consistent vocabulary where related word forms are recognized as related. **How do we turn inconsistent, unbounded raw text into a normalized, finite representation that downstream steps can operate on?**

## Intuition

Imagine building a search engine over a collection of product reviews. A user searches `"running shoes"`, but reviews contain `"run"`, `"runs"`, `"running"`, and `"ran"` scattered across different documents. If the search engine treats each of these as a completely distinct, unrelated string, it misses the reviews that used a different inflection of the same word — even though a human reader immediately recognizes them as "the same idea." The fix is **normalization**: reduce every inflected form to a shared root or canonical form before comparing or counting words, so `"run"`, `"runs"`, and `"running"` all collapse to something the system can match against each other.

Text preprocessing is the pipeline of steps that gets raw text into this normalized state: split it into tokens, reduce those tokens to a canonical form (stemming or lemmatization), strip out tokens that carry little discriminating information (stop words), and optionally tag each token with grammatical or semantic information (POS tags, named entities) that later steps can use.

## Why simpler approaches fail

The naive approach is to treat every distinct string in the text as its own unique token with no relationship to any other token — i.e., skip normalization entirely and feed raw, un-stemmed, case-sensitive strings straight into a downstream vectorizer. This fails in two concrete ways:

1. **Vocabulary explosion.** Every inflected form of a word (`"run"`, `"runs"`, `"running"`, `"ran"`, `"Run"`) becomes a *separate* vocabulary entry. For a corpus with even a modest number of unique root words, the surface-form vocabulary is many times larger, because English regularly inflects nouns (singular/plural), verbs (tense, aspect, person), and adjectives (comparative/superlative). A larger vocabulary means every downstream representation (one-hot, BOW, TF-IDF — topic 02) has higher dimensionality, more sparsity, and needs more data to estimate reliably.
2. **Semantically identical words are treated as unrelated.** Without normalization, `"run"`, `"runs"`, and `"running"` are three arbitrary, unrelated strings as far as any word-count-based representation is concerned — there is nothing in the string `"running"` that tells a `CountVectorizer` it shares a root with `"run"`. A query for `"run"` will not match a document that only contains `"running"`, even though they refer to the same action. Naive `str.split()` tokenization compounds this by also mishandling punctuation and contractions (`"don't"` becomes the single garbled token `"don't"` instead of being split into a meaningful `"do"` + `"n't"`), silently corrupting the token stream before normalization even has a chance to run.

Normalizing inflected forms to a shared root (stemming, lemmatization) directly collapses the vocabulary-explosion problem and directly fixes the "unrelated strings" problem, at the cost of needing an explicit set of rules or a dictionary to decide what counts as "the same root."

## Mathematical foundation

Text preprocessing is rule-based and lookup-based rather than a numeric optimization problem, so "mathematical foundation" here means precisely specifying the *procedures* stemming, lemmatization, and tagging use, since correctness depends entirely on getting these procedures right.

**Porter Stemmer's rule structure.** The Porter algorithm (Porter, 1980) applies five ordered phases of suffix-stripping rules, each phase specified as `condition → suffix replacement`. Two concrete example rules (of many):

- **Step 1a**: `SSES → SS` (`caresses → caress`), `IES → I` (`ponies → poni`), `S → ""` (`cats → cat`, but a word ending in `SS` is left alone).
- **Step 1b**: if a word ends in `EED` and has at least one vowel-consonant measure ≥ 1, `EED → EE` (`agreed → agree`); if a word ends in `ED` or `ING` and the stem *before* the suffix contains a vowel, strip the suffix and clean up the result (`plastered → plaster`, `hopping → hop` — note the double-consonant cleanup that removes a doubled final consonant when appropriate).

Each phase's rules are ordered so only the first matching rule in that phase fires; the algorithm never consults a dictionary, so it can produce non-words. This is exactly why stemming is fast (constant-time string matching per phase, no lookup) but crude — the rule set is a fixed pattern-matcher, not a semantic process.

**Lemmatization as constrained lookup.** WordNet lemmatization is a function $\text{lemma}(w, p) \to \ell$ that maps a (word, part-of-speech) pair to the canonical dictionary form $\ell$, by looking up morphological exceptions and applying WordNet's own suffix rules only as a fallback within a known-word dictionary. Because the function is indexed by POS, `lemma("better", \text{adj}) = \text{"good"}` (an irreducible dictionary mapping) while `lemma("running", \text{noun}) = \text{"running"}$ but `lemma("running", \text{verb}) = \text{"run"}`. Without the correct POS argument, `WordNetLemmatizer` defaults to noun, which silently produces wrong lemmas for verbs and adjectives — this is why POS tagging (below) is a prerequisite for high-quality lemmatization, not an independent, optional step.

## Algorithm

The standard text preprocessing pipeline, in order:

1. **Tokenization**: split raw text into sentences (`sent_tokenize`), then each sentence into word/punctuation tokens (`word_tokenize`). Non-trivial because periods are ambiguous (`Dr.`, `U.S.A.`, decimal numbers) — a trained tokenizer (NLTK's Punkt) handles these and contractions correctly where naive `str.split()` does not.
2. **Case-folding / vocabulary construction**: lower-case tokens (usually) and collect the set of unique tokens across the corpus to define the vocabulary.
3. **Stemming or lemmatization**: reduce each token to a root/canonical form — stemming via fixed suffix-stripping rules (fast, no dictionary, may produce non-words), lemmatization via dictionary + POS lookup (slower, always a real word).
4. **Stop word removal**: drop tokens from a fixed high-frequency list (`the`, `is`, `and`, ...) that carry little discriminating signal for many downstream tasks — with the caveat that this list must be customized (e.g. keep negations) for tasks where function words matter.
5. **POS tagging**: assign each token a grammatical category (Penn Treebank tag set) using both the token itself and its sentence context.
6. **Named entity recognition (NER)**: group POS-tagged tokens into spans labeled with an entity type (PERSON, ORGANIZATION, GPE, DATE, ...), via `word_tokenize → pos_tag → ne_chunk` in NLTK.

Each step feeds the next: stemming/lemmatization operates on tokens (step 1's output); POS tagging determines lemmatization quality (step 3) and NER's chunking (step 6).

**Common Penn Treebank POS tags:**

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

**Stemming vs. lemmatization:**

| Aspect | Stemming | Lemmatization |
|---|---|---|
| Method | Rule-based suffix stripping | Dictionary + morphological analysis |
| Output | May not be a real word | Always a valid dictionary word |
| Speed | Fast | Slower (needs lookup, often POS) |
| Accuracy | Coarser, more collisions | More accurate, context-sensitive |
| Needs POS? | No | Ideally yes, for best results |

**Stemmer variants**: The **Porter Stemmer** is the classic algorithm described above — fast and simple but relatively aggressive. The **Snowball Stemmer** (also by Martin Porter) is an improved, more consistent successor supporting multiple languages, fixing several edge cases where Porter over/under-stems, and is generally preferred over Porter in modern pipelines when a stemmer is required.

## From-scratch implementation

Implemented in `text-preprocessing.ipynb`: a tiny hand-rolled stemmer using just four ordered rules — strip `"-ing"`, `"-ed"`, `"-s"`, `"-ly"` if the resulting stem is at least 3 characters long (to avoid mangling very short words like `"is"`) — applied to a handful of example words, and compared side-by-side against `PorterStemmer`'s output on the *same* words. The comparison table shows:

- **Agreement** on simple, regular inflections (e.g. both reduce `"jumps"` → `"jump"` and `"walked"` → `"walk"`), confirming that stemming's core mechanism really is just suffix pattern-matching — the toy 4-rule version reproduces Porter's behavior whenever the underlying pattern is simple.
- **Disagreement** on words with irregular spelling changes or where Porter's multi-phase rules do more (e.g. words needing a doubled-consonant cleanup, or a `y → i` conversion before a suffix, which the toy stemmer's naive suffix-strip does not handle), showing that Porter's five ordered phases exist precisely to handle cases a single flat rule list gets wrong.

This confirms the Mathematical foundation section's claim directly: stemming is pattern matching over a rule table, not a magic semantic process — a 4-rule toy version gets simple cases right and a real algorithm needs many more, carefully-ordered rules to handle English's actual irregularity, but the *kind* of computation (string suffix matching, no dictionary) is identical.

## Practical implementation

`text-preprocessing.ipynb` runs the full pipeline from the Algorithm section on a short real paragraph of text (about Barack Obama): `sent_tokenize`/`word_tokenize` for tokenization, a `PorterStemmer`-vs-`SnowballStemmer`-vs-`WordNetLemmatizer` (noun and verb mode) comparison table over sample words, `stopwords.words("english")`-based filtering, `nltk.pos_tag` over the full token list, and `nltk.ne_chunk` producing labeled entity spans (correctly identifying *Barack Obama* as PERSON, *Hawaii*/*United States*/*Chicago* as GPE, and *Harvard University* as ORGANIZATION).

The mapping back to "From-scratch implementation" is direct: `PorterStemmer.stem()` is exactly the same *kind* of suffix-stripping computation as the toy stemmer, just with the algorithm's real 5-phase rule table instead of 4 flat rules — nothing about the underlying mechanism changes between the toy version and NLTK's production implementation.

## Experiment

**Hypothesis (stated before running):** stemming (rule-based, no dictionary) will occasionally collapse semantically distinct words to the same output (over-stemming) or leave related words un-unified (under-stemming), while lemmatization (dictionary + POS-based) will always return valid dictionary words and will be more conservative about merging unrelated words — because it consults WordNet rather than blindly matching suffix patterns.

**Setup:** `text-preprocessing.ipynb`'s stemming-vs-lemmatization comparison cell runs `PorterStemmer`, `SnowballStemmer`, and `WordNetLemmatizer` (as noun and as verb) over the sample word list `["studies", "studying", "universal", "university", "better", "running", "presidency", "awarded", "nations", "flies"]` and tabulates the results side by side.

**Result:** the notebook's comparison table confirms the over-stemming pitfall directly — Porter reduces both `"universal"` and `"university"` toward the same `univers`-prefixed stem despite the words being semantically unrelated (one describes broad applicability, the other an institution), while `WordNetLemmatizer` leaves them as distinct, valid dictionary forms. The verb-mode lemma column also shows the POS-dependence claim from the Mathematical foundation section: `"running"` lemmatizes to `"running"` in noun mode but to `"run"` in verb mode.

**Interpretation:** the results match the hypothesis — stemming's purely rule-based mechanism has no way to know that `universal` and `university` are unrelated concepts, since it only looks at the string's suffix pattern, while lemmatization's dictionary lookup preserves the distinction because it treats them as separate WordNet entries.

**Limitations:** this is one short paragraph and one hand-picked word list; it demonstrates the over-stemming failure mode exists and is reproducible, not its frequency across a large, diverse corpus.

## Failure modes

- **Over-stemming**: the rule-based stemmer strips too much and collapses unrelated words to the same stem — e.g. *universal*, *university*, *universe* can all reduce to `univers`, even though they have different meanings. Demonstrated directly in the Experiment section above.
- **Under-stemming**: the stemmer fails to reduce genuinely related words to the same stem — e.g. *data* and *datum* may not be unified, because their surface forms don't share a suffix pattern the rule table recognizes.
- **Non-word output**: stemming can produce strings that are not valid English words (e.g. `studies → studi`). Acceptable for tasks like search indexing where exact readability doesn't matter, but a problem when output needs to be human-readable.
- **Lemmatization without correct POS**: `WordNetLemmatizer` assumes a noun by default; feeding it a verb without the right POS tag silently produces the wrong (unchanged) lemma, defeating the purpose of using a dictionary-based method in the first place.
- **Stop word removal erasing meaning**: blindly dropping a generic stop word list can remove negations (`not`, `no`, `never`) that are semantically critical for tasks like sentiment analysis, flipping or destroying the intended meaning of a sentence.

## Real-world usage

Stemming is used where speed matters more than precision and downstream consumers don't need human-readable output — search-engine indexing and information retrieval are the classic cases, since matching a query stem against an indexed document stem is a fast set-membership check regardless of whether the stem itself is a real word. Lemmatization is used where output must be readable or semantically precise — chatbots, summarization, and feature engineering for downstream models that a human will inspect. POS tagging and NER are foundational to information extraction pipelines: pulling structured facts (who, where, when, how much) out of unstructured text, and are building blocks for question answering, knowledge-graph construction, and document summarization. Every one of these normalization steps directly determines the vocabulary size and quality that topic 02's numeric representations (one-hot, BOW, TF-IDF) are built from — preprocessing mistakes here propagate into every downstream feature.

## Mental model

Text preprocessing is "collapsing the space of surface forms down to the space of underlying concepts, using either a fixed rule table (stemming — fast, no dictionary, sometimes wrong) or a dictionary lookup (lemmatization — slower, always correct-form output)": every step in the pipeline exists to reduce the vocabulary a downstream numeric representation has to deal with, and to make sure that reduction groups genuinely related words together rather than either under-grouping (missing matches) or over-grouping (creating false matches).

## Questions to think about

1. The from-scratch toy stemmer strips `"-ing"`, `"-ed"`, `"-s"`, `"-ly"` unconditionally (subject only to a minimum stem length). Construct an English word where this naive rule set produces a *worse* result than doing nothing at all — i.e., where stripping the suffix actively creates a false match with an unrelated word. What property of Porter's real rule set (condition checks beyond "ends with X") would prevent that specific failure?
2. Why does `WordNetLemmatizer` need a POS argument to work correctly, while `PorterStemmer` needs no linguistic context at all? What does this tell you about what each algorithm is actually using as its source of truth (a rule table vs. a dictionary)?
3. Stop word removal is described as both helpful (dimensionality reduction) and harmful (destroying negation). Design a preprocessing rule that keeps the benefit while avoiding the negation failure mode, without simply skipping stop-word removal for the whole pipeline.
4. If you were building a search engine over a small, technical corpus (e.g. legal contracts) where exact terminology matters and false-positive matches are costly, would you choose stemming or lemmatization for query/document normalization? Justify the choice using the over-stemming vs. under-stemming trade-off.
5. NER's pipeline (`word_tokenize → pos_tag → ne_chunk`) depends on every earlier stage being correct. If tokenization mis-splits a multi-word proper noun (e.g. splits "Harvard University" awkwardly around punctuation), what downstream failure would you expect in the NER stage, and at which stage would you actually need to fix it?
