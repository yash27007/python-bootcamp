# 01 – Tokenizer From Scratch: Byte-Pair Encoding

## Problem

[`07-nlp/05-transformers-and-huggingface`](../../07-nlp/05-transformers-and-huggingface/notes.md)
loaded a tokenizer with one line — `AutoTokenizer.from_pretrained("distilbert-base-uncased")` —
tokenized real sentences with it, and even showed the actual subword split it produces
(`"unbelievably"` → `un`, `##bel`, `##ie`, `##va`, `##bly`). What that topic never showed
is *how that tokenizer came to exist in the first place*: where its ~30,000-entry
vocabulary came from, why "the" is one token but "unbelievably" is five, and what
algorithm decided that split. This topic builds that algorithm from scratch: a real
Byte-Pair Encoding (BPE) tokenizer, trained on a small corpus, with every merge it learns
inspected directly.

## Intuition

Imagine building a vocabulary for a new language by watching people write, one letter at
a time, and every time you notice two adjacent letters (or letter-groups) appearing
together unusually often, you give that pair its own single symbol from then on. Do this
enough times and the symbols you end up with are not "letters" or "words" in the
traditional sense — they're whatever chunks actually recur often enough in the text to be
worth a symbol of their own. Very common whole words (`"the"`, `"and"`) get their own
single symbol quickly, because they recur constantly. Rare or novel words never get a
whole-word symbol — they stay assembled from smaller pieces that *did* recur often enough
individually (prefixes, suffixes, common letter clusters). Byte-Pair Encoding is exactly
this process, run automatically and iteratively over a training corpus's raw character
statistics, with no hand-written dictionary or grammar rules at all.

## Why simpler approaches fail

[`07-nlp/05-transformers-and-huggingface`](../../07-nlp/05-transformers-and-huggingface/notes.md#subword-tokenization-the-oov-vs-sequence-length-tradeoff)
already worked out this tradeoff in detail, comparing three strategies:

- **Word-level tokenization** (a fixed, capped vocabulary of whole words, `VOCAB_SIZE=10000`
  in `07-nlp/04-deep-learning-nlp`'s Keras `Embedding`) keeps sequences short but has a
  hard **out-of-vocabulary (OOV) problem**: any word not in the fixed vocabulary collapses
  to an opaque `<UNK>`/`<OOV>` token, losing all information about what that word actually
  was. Rare words, typos, and novel compounds are exactly the words most likely to fall
  outside a size-capped vocabulary — this is not a rare edge case, it is the routine
  behavior of word-level tokenization on any text that was not part of building the
  vocabulary.
- **Character-level tokenization** has zero OOV problem (any string is representable as
  characters from a small, fixed alphabet) but makes sequences dramatically longer, and
  because self-attention costs $O(n^2)$ in sequence length
  (`06-deep-learning/05-attention-transformers`, "Failure modes"), that length increase is
  expensive, not just inconvenient.

Byte-Pair Encoding is the resolution `07-nlp/05-transformers-and-huggingface` cited but
did not derive: a vocabulary of variable-length pieces, **learned from data** rather than
hand-designed, that gets the short-sequence benefit of word-level tokenization for common
words while never having an OOV problem at all — any string, however novel, always
decomposes into *some* sequence of known sub-pieces, in the worst case down to individual
characters, which are always in the vocabulary by construction. This topic derives and
implements exactly that algorithm.

## Mathematical foundation

**Setup.** Let a training corpus be a multiset of words $w_1, w_2, \ldots$ (from
whitespace pre-tokenization), each with an observed frequency $f(w)$. Represent each word
as an ordered sequence of *symbols*. Initially, every symbol is a single character, and an
explicit end-of-word marker `</w>` is appended to every word — this stops merges from
ever crossing a word boundary (without it, the last character of one word and the first
character of the next would look like an ordinary adjacent pair once whitespace is
dropped, and BPE would merge across words, which is not the intended behavior).

**Vocabulary.** The vocabulary $V$ starts as the set of all distinct symbols occurring in
the corpus — for a corpus using the Latin alphabet, this is small (on the order of 30
characters plus the end-of-word marker). This is why the algorithm is sometimes described
as starting from "bytes": at the most general level, the initial vocabulary is every
possible byte value (256 symbols), which guarantees *any* input string, in any script or
encoding, is representable from the start — no character can ever be unrepresentable. This
implementation uses characters, not raw bytes, for a single-language toy corpus; the
"bytes" framing matters for tokenizers meant to handle arbitrary Unicode/multilingual
text robustly, which is out of scope for this toy.

**Pair frequency.** For each adjacent pair of symbols $(s_i, s_{i+1})$ occurring within
any word's current symbol sequence, define its corpus-wide count as

$$
\text{count}(a, b) = \sum_{w} f(w) \cdot \big[\text{number of times } (a, b) \text{ occurs adjacently in } w\text{'s current symbol sequence}\big]
$$

— i.e. every occurrence of the pair is counted once per occurrence of the word it's in,
weighted by that word's frequency in the corpus. A pair that appears twice inside one very
common word contributes twice per instance of that word.

**The merge step.** At each iteration:

$$
(a^*, b^*) = \underset{(a,b)}{\arg\max} \; \text{count}(a, b)
$$

Every occurrence of $(a^*, b^*)$, in every word, is replaced by a single new symbol
$a^*{+}b^*$ (string concatenation). This new symbol is added to $V$, and the pair
$(a^*, b^*) \to a^*{+}b^*$ is recorded, **in order**, as a *merge rule*.

**Stopping condition.** Repeat the merge step until $|V|$ reaches a target vocabulary
size $K$, or until $\max_{(a,b)} \text{count}(a,b) < 2$ — i.e. no remaining pair occurs
more than once anywhere in the corpus, at which point any further "merge" would just be
memorizing one specific word rather than learning a pattern that generalizes, so training
stops early even if $K$ was never reached (this exact early stop is measured directly in
"Experiment," below).

**Why order matters for encoding.** The learned output is not just a final vocabulary —
it is an **ordered list of merge rules** $r_1, r_2, \ldots, r_M$ (in the order they were
learned, i.e. in decreasing "how corpus-wide-common was this pair" priority, since the
most frequent remaining pair is always merged first). To tokenize new text, the same rules
are applied **in that same order** to the new text's initial character sequence. Applying
them in a different order can produce a different, wrong split, because a later merge
rule was defined in terms of symbols that only existed *after* an earlier merge rule had
already been applied (e.g. the rule `('th', 'e</w>') -> 'the</w>'` only makes sense once
the earlier rule `('t', 'h') -> 'th'` has already run). This is why BPE tokenizers ship
their merges as an ordered list, not an unordered set.

## Algorithm

1. Pre-tokenize the training corpus on whitespace; count word frequencies $f(w)$.
2. Represent every word as a tuple of characters + an end-of-word marker `</w>`.
3. Initialize the vocabulary $V$ to the set of distinct symbols present.
4. **Repeat** until $|V| = K$ or no pair occurs more than once:
   a. Count every adjacent symbol pair across all words, weighted by word frequency.
   b. Find the single most frequent pair $(a^*, b^*)$.
   c. Replace every occurrence of $(a^*, b^*)$ with the merged symbol $a^*{+}b^*$; add it
      to $V$; record the merge rule, in order.
5. To **encode** new text: split into words (same pre-tokenization rule), represent each
   word as characters + `</w>`, then apply every recorded merge rule, in the order it was
   learned, to that word's symbol sequence.
6. To **decode**: concatenate the token strings and replace `</w>` with a space.

## From-scratch implementation

`001_bpe_tokenizer_from_scratch.ipynb` implements every step above in plain Python — no
`tokenizers`/HuggingFace library used anywhere in the training or encoding path — and
actually runs it on a small (462-character, 87-word, 42-distinct-word) toy corpus about
foxes, dogs, and tokenizers.

**Training, actually run** (`target_vocab_size=60`, real output, unedited):

```
merge # 1: ('e', '</w>')          -> 'e</w>'        (count= 17)  vocab_size=28
merge # 2: ('t', 'h')             -> 'th'           (count= 15)  vocab_size=29
merge # 3: ('th', 'e</w>')        -> 'the</w>'      (count= 15)  vocab_size=30
merge # 4: ('s', '</w>')          -> 's</w>'        (count= 12)  vocab_size=31
merge # 5: ('r', '</w>')          -> 'r</w>'        (count= 12)  vocab_size=32
merge # 6: ('e', 'r</w>')         -> 'er</w>'       (count= 10)  vocab_size=33
merge # 7: ('o', 'w')             -> 'ow'           (count=  9)  vocab_size=34
...
merge #13: ('to', 'k')            -> 'tok'          (count=  6)  vocab_size=40
merge #14: ('tok', 'en')          -> 'token'        (count=  6)  vocab_size=41
...
merge #21: ('token', 'i')         -> 'tokeni'       (count=  5)  vocab_size=48
merge #22: ('tokeni', 'z')        -> 'tokeniz'      (count=  5)  vocab_size=49
...
merge #27: ('quick', '</w>')      -> 'quick</w>'    (count=  4)  vocab_size=54
merge #28: ('fo', 'x')            -> 'fox'          (count=  4)  vocab_size=55
merge #29: ('fox', '</w>')        -> 'fox</w>'      (count=  4)  vocab_size=56
...
merge #33: ('lazy', '</w>')       -> 'lazy</w>'     (count=  4)  vocab_size=60

Final vocab size: 60
Number of merges learned: 33
```

The training loop discovers, purely from co-occurrence counts and with no dictionary, that
`"the"`, `"tokenizer"` (via `token` → `tokeni` → `tokeniz`), `"quick"`, `"fox"`, and
`"lazy"` are common enough whole words in this corpus to deserve their own single token —
exactly the behavior the mathematical foundation predicts.

**Encoding a new sentence and verifying the round trip** (real output):

```
Test sentence: the newest tokenizer lowers the lowest words
Encoded tokens: ['the</w>', 'n', 'e', 'w', 'e', 'st</w>', 'tokeniz', 'er</w>', 'low',
                 'e', 'r', 's</w>', 'the</w>', 'low', 'e', 'st</w>', 'w', 'o', 'r', 'd', 's</w>']
Decoded:        'the newest tokenizer lowers the lowest words'

Round-trip check PASSED: decode(encode(text)) == text
```

`assert decoded == test_sentence` is a genuine, passing assertion on real output, not
described behavior — see the notebook for the executed cell. Note the words chosen for
this test: `"newest"`, `"lowers"` never appeared verbatim in the training corpus (only
`"newer"`, `"lower"`, `"lowest"` did, as whole separate words) — the round trip checks
that *learned subword pieces* (`low`, `er</w>`, `st</w>`, ...) recombine correctly on
genuinely novel words, not that the tokenizer memorized whole training sentences.

## Practical implementation

[`07-nlp/05-transformers-and-huggingface`](../../07-nlp/05-transformers-and-huggingface/notes.md)
used `AutoTokenizer.from_pretrained("distilbert-base-uncased")` — a WordPiece tokenizer
(a close BPE variant that merges the pair maximizing training-corpus likelihood rather
than raw frequency) with a ~30,000-token vocabulary, learned from a training corpus of
English Wikipedia and BookCorpus — on the order of **billions of words**, not this topic's
462 characters. That scale difference is exactly why `distilbert-base-uncased`'s
tokenizer never produces the pathological near-character-level splits this topic's toy
tokenizer produces on out-of-domain text (see "Failure modes," below): a real production
tokenizer has seen orders of magnitude more text, so far more subword pieces — across far
more domains, languages, and vocabularies — have had the chance to recur often enough to
earn a merge. The *algorithm* connecting the two is the same one derived above (BPE and
WordPiece differ only in the merge-selection criterion, frequency vs. likelihood); the
practical tokenizer differs from this topic's from-scratch one entirely in *scale of
training data*, not in kind.

## Experiment

**Hypothesis (stated before running):** as target vocabulary size increases, more whole
words and larger subword chunks get merged into single tokens, so average tokens-per-word
on held-out text should decrease — until the corpus itself runs out of any repeated
adjacent pair to merge, at which point additional vocabulary budget stops helping.

**Setup:** train BPE from scratch on the same toy corpus at six target vocabulary sizes
(30, 45, 60, 80, 100, 130), then tokenize one fixed 13-word held-out sentence (words drawn
from the same domain, not copied verbatim from the training corpus) and measure
`len(tokens) / len(words)` at each vocabulary size.

**Result** (real, measured, unedited):

```
target_vocab  actual_vocab  merges  tokens   avg_tokens/word
          30            30       3      64             4.923
          45            45      18      47             3.615
          60            60      33      38             2.923
          80            80      53      31             2.385
         100            98      71      23             1.769
         130            98      71      23             1.769
```

**Interpretation:** average tokens-per-word fell monotonically from 4.923 to 1.769 as
target vocabulary size rose from 30 to 100, confirming the hypothesis directly. But at
`target_vocab=130`, the *actual* vocabulary size plateaus at 98 (not 130), and
tokens-per-word stops improving — the training loop's early-stopping rule
(`count(a,b) < 2`) triggered before the requested target was reached: once every
remaining adjacent pair in this 462-character corpus occurred only once, there was no
generalizable pattern left to merge, only individual words left to memorize one at a
time, so training correctly stopped early. This surfaces a real limitation the
mathematical foundation predicts but that only shows up by actually running the loop: a
target vocabulary size is a *ceiling* the corpus may or may not be able to support, not a
guarantee — how large a vocabulary a given corpus can meaningfully fill depends on how
much repeated structure that corpus actually contains.

**Limitations:** one 13-word held-out sentence, one toy corpus, a single deterministic
run (BPE training here is not randomized, so no seed variation to average over) — the
*trend* (larger vocabulary → fewer tokens/word, bounded by a corpus-size-dependent
ceiling) is real and measured, but the specific numbers are particular to this ~500-word
corpus and would look different, with a much higher ceiling, on real large-scale training
data (which is exactly why `distilbert-base-uncased`'s tokenizer, trained on billions of
words, supports a ~30,000-token vocabulary without hitting this early-stop limitation).

## Failure modes

- **Wrong-domain text produces pathological, near-character-level splits.** This topic's
  BPE was trained entirely on English prose about foxes, dogs, and tokenizers. Feeding it
  Python source code it never saw patterns from — `"def tokenize(x): return x.split()"` —
  produces (real, executed output):

  ```
  Tokens: ['d', 'e', 'f', '</w>', 'tokeniz', 'e', '(', 'x', ')', ':', '</w>',
           'r', 'e', 't', 'u', 'r', 'n</w>', 'x', '.', 'spli', 't', '(', ')', '</w>']
  24 tokens for 4 whitespace-separated words (6.00 tokens/word)
  ```

  vs. 1.769 tokens/word for this same vocabulary on in-domain held-out text — over 3x
  worse. `"tokenize"` partially survives (the learned `tokeniz` piece still fires,
  because it appeared literally in the training corpus), but `"def"`, `"return"`,
  `"split"`, and every punctuation character fall back to near single-character tokens,
  because Python keywords, parentheses-as-syntax, and snake_case identifiers never
  recurred often enough (or at all) in the training corpus to earn a merge. The result
  still round-trips correctly — `decode(encode(x)) == x` still holds, verified in the
  notebook — nothing decodes *wrong*, it is simply an inefficient, near-character-level
  split. This is the direct, measured version of the OOV-adjacent problem described in
  "Why simpler approaches fail": BPE never has a hard failure the way word-level
  tokenization's `<UNK>` does, but it does have a *soft* failure — long, inefficient
  sequences — whenever the input domain diverges from the training domain.
- **Vocabulary size is a real tradeoff, not a free parameter.** Too small a target
  vocabulary (this topic's `target_vocab=30`) means almost nothing beyond single
  characters gets merged, so sequences stay long (4.923 tokens/word, measured above) —
  paying the character-level tokenization cost (`06-deep-learning/05-attention-transformers`'s
  $O(n^2)$ attention cost in sequence length) without character-level tokenization's
  simplicity. Too large a target vocabulary means a proportionally larger embedding
  table (one row per vocabulary entry) that a real model must learn — parameters spent on
  rare subword combinations rather than deeper computation. Production tokenizers (BERT's
  ~30,000, GPT-2's ~50,000) sit in the middle of this tradeoff, calibrated against a large
  training corpus.

## Real-world usage

Every modern large language model — GPT-family, BERT-family, Llama, and effectively every
production LLM tokenizer — uses BPE or a close variant (WordPiece, SentencePiece) for
exactly the reason "Why simpler approaches fail" derives: no hard OOV failure, and
close-to-word-level sequence lengths for common text. The training loop this topic
implements — count pairs, merge the most frequent, repeat — is *the actual algorithm*,
not a simplification of it; production tokenizer training differs from this notebook only
in corpus scale (billions of words vs. 462 characters) and engineering (efficient
incremental pair-count updates instead of this notebook's readable-but-$O(n)$-per-merge
recount, called out explicitly as the from-scratch tradeoff this topic makes for clarity
over speed). `07-nlp/05-transformers-and-huggingface`'s `AutoTokenizer.from_pretrained(...)`
loads exactly this kind of tokenizer's output — a vocabulary plus an ordered merge-rule
list — trained once, offline, and reused unchanged for every subsequent use of that model.

## Mental model

**BPE builds a vocabulary the way repeated exposure builds familiarity: whatever chunks
of text recur often enough earn their own symbol, starting from individual characters and
working upward, with no dictionary and no notion of "word" built in.** The result is a
vocabulary with no hard failure case (any string decomposes into *some* known pieces, down
to individual characters in the worst case) but a real, measurable efficiency cost when
the input text's domain diverges from what the vocabulary was trained on — the same
tradeoff `07-nlp/05-transformers-and-huggingface`'s word-level-vocabulary discussion
identified for OOV, resolved from "a token becomes structurally unrepresentable" down to
"a token becomes a longer sequence of smaller known pieces."

## Questions to think about

1. This topic's `train_bpe` stopped early at `target_vocab=130` because no remaining pair
   occurred more than once in a 462-character corpus. If the training corpus were instead
   462 *megabytes*, would you expect the same early-stopping behavior at a target
   vocabulary of, say, 30,000? Why or why not, in terms of how pair counts scale with
   corpus size?
2. The "Failure modes" section showed Python code tokenized by an English-prose-trained
   BPE producing 6.00 tokens/word (vs. 1.769 in-domain). If you were building a code
   -completion model, what would this imply about training a *separate* tokenizer on a
   code corpus, versus reusing a general-English tokenizer like `distilbert-base-uncased`'s?
3. `apply_merges` in this notebook applies every learned merge rule, in learned order, to
   each new word. What would go wrong if the merge rules were applied in a different order
   (e.g. sorted by pair alphabetically instead of by learn-order)? Trace through
   `('t','h')` and `('th','e</w>')` from this topic's actual training trace as a concrete
   example.
4. The end-of-word marker `</w>` was introduced specifically to stop merges from crossing
   word boundaries. What specific (wrong) tokens might `train_bpe` learn on this topic's
   corpus if `</w>` were removed entirely and words were just concatenated with spaces
   as an ordinary character?
5. `07-nlp/05-transformers-and-huggingface`'s WordPiece tokenizer selects merges by
   which pair most increases training-corpus likelihood under a simple language model,
   rather than by raw frequency (this topic's BPE criterion). Both still produce an
   ordered list of merge rules applied the same way at encoding time. Under what
   corpus conditions might likelihood-based selection choose a *different* first merge
   than frequency-based selection would?
