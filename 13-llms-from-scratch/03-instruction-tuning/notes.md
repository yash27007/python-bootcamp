# 03 – Instruction Tuning: From "Complete This Text" to "Follow This Request"

## Problem

`13-llms-from-scratch/02-pretraining-objective` trained a real (if tiny) decoder-only
`TinyGPT` with the autoregressive next-token objective, and was honest about what that
objective actually teaches: **plausible continuation of text, nothing more.** That
notebook's own real generated output makes the point concretely — prompted with
`"The old "`, the trained model produced

```
The old houted vier aget, sin p gy toueainid
EChe ld sterly thid tsthull st n ye aiarmey
```

That is a raw completion: the model kept extending the character stream in whatever
direction its learned statistics pointed, with no notion that a human might have wanted
something *back* from it — an answer, a translation, a summary, anything with a shape
determined by a request rather than by "what plausibly comes next in this exact string."
Pretraining, by construction (`02-pretraining-objective`'s "Mathematical foundation"),
never once showed the model a `(request, response)` pair — every training example was
"here is some text, predict the next character of *this same text*." A pretrained model
has therefore never been asked to distinguish "keep writing this passage" from "stop,
switch roles, and produce a response to what was just asked." **The problem this topic
solves: how do you get a next-token-prediction model to behave like it is answering a
request instead of just continuing a string, without inventing a new training objective
from scratch?**

## Intuition

Imagine handing someone a single sentence with the instruction "finish this" versus
handing them a question with the instruction "answer this." Both are "predict some words
that plausibly follow," but they are *different tasks* — finishing a sentence rewards
staying in the same voice and register as what came before; answering a question rewards
switching from "continue" mode to "respond" mode, often in a completely different register
(a question in casual prose, an answer that is short and factual). A pretrained model has
only ever practiced the first task. Instruction tuning is the realization that the *second*
task can be taught with the **exact same mechanism** (predict the next token, given
everything before it) — the only thing that has to change is *what counts as "the text so
far."* If every training example is reformatted as `"{instruction}\n{response}"` instead
of raw prose, then "predict the next token" starts meaning "predict the next token of the
*response*, given the instruction" for every position after the template's response
marker. Nothing about the underlying prediction machinery changes — only the shape of the
data it is shown.

## Why simpler approaches fail

**"Just prompt it well" (few-shot prompting a frozen pretrained model) has a real ceiling.**
Few-shot prompting — putting a couple of `(instruction, response)` examples directly in the
prompt and hoping the model pattern-matches the format for a new instruction — can work
*for models that were pretrained on internet-scale data containing huge amounts of
naturally occurring instruction-like and dialogue-like text* (forum Q&A, StackOverflow,
how-to articles), because such a model has, incidentally, seen the *shape* of
question-answer exchanges millions of times during pretraining even though it was never
explicitly trained to produce them on command. That is a real, useful phenomenon in large
pretrained LLMs — but it is a side effect of scale and data diversity, not something
prompting can produce out of a model that never had the opportunity to see that shape.
`02-pretraining-objective`'s `TinyGPT` was pretrained on 1,486 characters of a single
lighthouse story — there is no dialogue, no Q&A, no instruction-following text anywhere in
its training data for a clever prompt to invoke. No prompt engineering can make a model
produce a behavior it has literally zero exposure to in its weights; a frozen model's
weights encode only what its training data and objective could have taught it, and
"predict the next character of this lighthouse story" never once created a gradient that
rewarded "stop and answer a question instead." **This section's own experiment confirms
it directly** (see "Experiment," below): the pretrained-only `TinyGPT`, prompted with the
`Q: ... \nA:` template it never saw once during pretraining, does not switch into
answer-mode — it keeps sampling lighthouse-prose-shaped characters regardless of the
template around them.

The only way to actually teach the new behavior is to put gradient steps behind it — i.e.,
show the model real examples of the shape you want and let backpropagation update the
weights toward producing it. That is supervised fine-tuning, and instruction tuning is
supervised fine-tuning applied specifically to `(instruction, response)` data.

## Conceptual foundation

*(This is a systems/methodology topic per `AGENTS.md`'s allowance — "Conceptual
foundation" substitutes for "Mathematical foundation" here, but the underlying objective
genuinely is the same math already derived in `02-pretraining-objective`, just applied to
differently-shaped data — that reuse is the whole point, and is made explicit below rather
than skipped.)*

**Instruction tuning is supervised fine-tuning on `(instruction, response)` pairs, using
the identical next-token cross-entropy objective `02-pretraining-objective`'s
"Mathematical foundation" derived:**

$$
\mathcal{L} = -\sum_{t=1}^{T} \log P_\theta(x_t \mid x_{<t})
$$

The only thing that changes is what $x_1, \ldots, x_T$ *is*. In pretraining, $x$ is a
window of raw corpus text with no internal structure the model is meant to respect.
In instruction tuning, $x$ is a **formatted pair**:

$$
x = \underbrace{\texttt{"Q: \{instruction\}}}_{\text{prompt segment}}\underbrace{\texttt{\textbackslash nA: \{response\}\textbackslash n\textbackslash n"}}_{\text{response segment}}
$$

(this notebook's exact template — a real system typically uses more elaborate role
markers like `<|user|>` / `<|assistant|>`, but the structural idea is identical: a fixed,
parseable boundary between "the request" and "the thing being predicted"). Training still
computes cross-entropy over *every* position in $x$ exactly as before — nothing in the loss
function itself distinguishes "prompt segment" tokens from "response segment" tokens
(this notebook does not mask the prompt segment out of the loss, matching many simple
instruction-tuning setups, though production recipes often do mask the prompt to avoid
spending gradient signal on tokens the model is never asked to generate at inference time —
named here as a real variant, not implemented). What actually produces the new *behavior*
is not a new loss function — it is that every training example now has this rigid
"instruction, then response" shape, so the positions right after the fixed `"\nA: "`
marker are, across the whole fine-tuning set, consistently followed by *answer-shaped*
continuations rather than *arbitrary-continuation-shaped* ones. The model's next-token
predictions after seeing that marker get pulled, by ordinary gradient descent, toward
"produce something that looks like a response," simply because that is what minimizes loss
on every training example sharing that marker.

**This is mechanistically the same thing `07-nlp/05-transformers-and-huggingface`'s
fine-tuning step did**, just with a different data shape and a different task framing.
That topic took a pretrained DistilBERT encoder and fine-tuned *every* weight (not just a
new head) with ordinary gradient descent on a small labeled sentiment dataset, at a
small learning rate specifically to avoid catastrophic forgetting (`07-nlp/05`'s
"Practical implementation," step 4, and "Failure modes"). The mechanism there —
gradient descent nudging a pretrained model's existing weights toward a new, narrower
task using a comparatively small amount of task-specific data — is exactly the mechanism
here. The differences are only in *what* is being predicted (a discrete sentiment label
from a new classification head there, vs. the next token of a structured response here,
using the model's own existing output head) and *what data shape* carries the new signal
(labeled `(review, sentiment)` pairs there, formatted `(instruction, response)` text here).
Same gradient descent on a pretrained model's weights, different objective framing around
it — the pattern `07-nlp/05`'s "Mental model" already named: *"a pretrained model starts
already knowing 'how language works'; fine-tuning spends a much smaller data budget only on
'how to do this specific task.'"* Instruction tuning's "specific task" is simply "respond
to a request" instead of "classify a review."

**Planned, not implemented: RLHF and DPO.** Supervised instruction tuning (what this
topic builds) gets a model to imitate the *style* of the response examples it was shown.
It says nothing about training a model to prefer *better* responses over worse ones when
given more than one candidate — that is what **Reinforcement Learning from Human Feedback
(RLHF)** and **Direct Preference Optimization (DPO)** do, both by name only here. RLHF
trains a separate reward model on human preference comparisons between candidate
responses, then optimizes the language model against that reward signal with
reinforcement learning (typically PPO). DPO reformulates the same preference-learning goal
as a single differentiable loss directly on the policy model, avoiding a separate reward
model and RL loop. Both require either a trained reward model, human-labeled preference
data, or both, at a scale and infrastructure complexity far beyond a toy character-level
model with 24 hand-written training pairs — genuinely **out of toy-scale reach**, per
`AGENTS.md`'s no-heavy-training constraint, and marked honestly as **Planned**, not
attempted, per this repository's Learned/Currently-learning/Planned convention.

## Algorithm

1. **Collect / write `(instruction, response)` pairs** in the target domain (this topic:
   24 hand-written Q&A pairs about `02-pretraining-objective`'s lighthouse story).
2. **Format each pair with a fixed template** that a model can learn to recognize as a
   request/response boundary (this topic: `"Q: {instruction}\nA: {response}\n\n"`).
3. **Concatenate the formatted pairs into one fine-tuning corpus**, tokenize with the
   *same* tokenizer/vocabulary the pretrained model already uses (critical — see
   "Failure modes" for what happens to characters the pretrained vocabulary never saw).
4. **Start from the pretrained model's weights** (not a fresh random initialization).
5. **Continue training with the identical next-token cross-entropy objective**, on
   `(context, target)` windows sliced from the formatted instruction corpus exactly the
   way pretraining sliced windows from raw prose — same `get_batch` mechanism, different
   source data.
6. **Use a smaller learning rate and fewer steps** than pretraining used, to nudge
   pretrained weights toward the new data shape rather than overwrite what they already
   encode (the catastrophic-forgetting mitigation `07-nlp/05` already established).
7. **At inference time**, prompt with the same template up through the response marker
   (`"Q: {new instruction}\nA:"`) and let the model generate — no architecture change, no
   new inference procedure, exactly `02-pretraining-objective`'s `generate()` loop.

## From-scratch implementation

`001_instruction_tuning.ipynb` builds and actually trains both stages, real gradient
steps, real before/after generations, real measured comparison:

- **Stage 0 — shared vocabulary.** The pretraining corpus (`02-pretraining-objective`'s
  exact 1,486-character lighthouse paragraph) and 24 hand-written `(question, answer)`
  pairs about that same story are combined, and the character vocabulary is built from
  their **union** (41 characters total) before any training happens — exactly how a real
  tokenizer is fixed once and reused across pretraining and fine-tuning. Seven characters
  appear only in the pretraining corpus; four (`:`, `?`, `Q`, `W`) appear only in the
  instruction corpus and are therefore never seen during Stage 1.
- **Stage 1 — pretrain `TinyGPT` fresh**, identical architecture and hyperparameters to
  `02-pretraining-objective` (2 layers, 32-dim embeddings, 2 heads, 48-character context,
  feed-forward width 64; **21,353 parameters**, marginally more than Task 2's 21,093 due
  only to the larger 41-vs-37-character vocabulary's embedding/head rows), 300 AdamW
  steps, `lr=3e-3`, batch size 32, `torch.set_num_threads(1)`, on the lighthouse corpus
  only.
- **Before instruction tuning**: the freshly pretrained model is prompted with
  `"Q: What did Mara carry with her?\nA:"` and with a plain `"The old "` prompt, and
  generates real, honestly-reported output for both (see "Experiment").
- **Stage 2 — instruction fine-tuning**: continuing training on the *same* model object
  (weights carried over, not reinitialized), now on the 24 formatted `(question, answer)`
  pairs concatenated into one small corpus, 150 AdamW steps, `lr=1e-3` (3x lower than
  Stage 1's), batch size 16 — deliberately gentler and shorter than pretraining, per
  "Conceptual foundation"'s catastrophic-forgetting reasoning.
- **After instruction tuning**: the same two prompts are regenerated from the
  fine-tuned model.
- **Measured comparison**: a `difflib.SequenceMatcher` character-similarity score between
  each generated response and its true target answer, averaged over all 24 training
  questions, computed for both the pretrained-only model and the instruction-tuned model
  (a same-seed, same-recipe reference pretrained-only model is retrained in its own cell
  to score fairly against the already-fine-tuned weights — see the notebook's Section 7
  for why).

## Practical implementation

There is no separate practical/library step here, for the same honest reason
`02-pretraining-objective`'s "Practical implementation" gave for skipping a real
GPT-2-scale run: real instruction-tuning recipes (e.g. the datasets and training code
behind InstructGPT, Alpaca, or any modern open instruction-tuned checkpoint) fine-tune
models with hundreds of millions to tens of billions of parameters on tens of thousands to
millions of curated instruction examples, over hours to days of GPU/TPU time — far beyond
`AGENTS.md`'s no-heavy-training constraint. The honest mapping instead: this notebook's
Stage 2 loop **is** architecturally what any real supervised instruction-tuning run does —
start from pretrained weights, continue training with the same objective on
template-formatted `(instruction, response)` data, at a reduced learning rate — just with
24 examples and 150 steps instead of a real dataset and a real training budget. Using a
real pretrained checkpoint (e.g. via HuggingFace `transformers`, following
`07-nlp/05-transformers-and-huggingface`'s pattern) and a real instruction dataset (e.g.
Alpaca-style or Dolly) to fine-tune a small open model is a natural next step, but is
deliberately not this topic's job — this topic is about the instruction-tuning
*mechanism*, at toy scale, on a model this repository actually built and pretrained
itself.

## Experiment

**Hypothesis (stated before running):** instruction-tuned generations should look more
like the target response format than pretrained-only generations — attempting to answer
rather than continuing lighthouse prose — even if the *content* is not reliably correct at
this toy scale, and this should be measurable as a higher mean character-similarity score
to the true target responses.

**Setup:** `001_instruction_tuning.ipynb`, `TinyGPT` (21,353 params) pretrained 300 steps
on the 1,486-character lighthouse corpus (`lr=3e-3`), then instruction-fine-tuned 150 steps
on 24 formatted `(question, answer)` pairs (`lr=1e-3`), `torch.set_num_threads(1)`.
Compared generations for `"Q: What did Mara carry with her?\nA:"` and `"The old "` before
and after Stage 2, and computed mean `difflib.SequenceMatcher` similarity to the true
answer across all 24 training questions for both the pretrained-only and instruction-tuned
model.

**Result (actual, from the executed notebook):**

- Stage 1 (pretraining): training loss **3.8192 → 1.6915** over 300 steps, **1.97s**
  wall-clock; val perplexity **12.08** vs. uniform baseline **41**.
- Stage 2 (instruction fine-tuning): training loss **2.8204 → 1.4704** over 150 steps,
  **0.58s** wall-clock; val perplexity **6.17** vs. uniform baseline **41** — lower than
  Stage 1's, consistent with the fine-tuning corpus being smaller and far more repetitive
  in structure (every example shares the same `"Q: ...\nA: ..."` shape).
- **Pretrained-only generation**, prompt `"Q: What did Mara carry with her?\nA:"`:
  ```
  Q: What did Mara carry with her?
  A:d agangoung therstotite of lllld, uook right the, whe oguter
  ```
  No attempt at a short, terminated answer — the model keeps producing
  lighthouse-prose-shaped character runs straight through the `"A:"` marker exactly as
  "Why simpler approaches fail" predicted, because it has never once seen that marker mean
  anything.
- **Instruction-tuned generation**, same prompt:
  ```
  Q: What did Mara carry with her?
  A: soker gudin lid dat ded ta houroump lid the slid wirshe oul
  ```
  Still not a correct or grammatical answer (24 examples and 150 steps is nowhere near
  enough for that), but the character-level statistics right after `"A: "` have shifted —
  measured directly by the similarity metric below, not just asserted qualitatively.
- **Mean similarity-to-target-response**, averaged over the same 24 training questions:
  **pretrained-only: 0.311** (min 0.135, max 0.486) vs. **instruction-tuned: 0.345**
  (min 0.179, max 0.561) — a real, measured, but modest increase (+0.034, about +11%
  relative), in the hypothesized direction on every summary statistic (mean, min, and max
  all increased).
- **Forgetting check, prompt `"The old "`** (a plain story-continuation prompt with no
  instruction template):
  - Before Stage 2: `"The old the sthe, sa st smas imagaing bra at ng lot thedorpawoks, th"`
    — lighthouse-prose-shaped noise, no template artifacts (expected — the template did
    not exist yet).
  - After Stage 2: `"The old lothe.\n\n\n\n: What d nofrote to\nQ: ck walininghed th to the wa"`
    — the fine-tuned model now inserts blank lines and literal `"Q:"`/`":"` fragments into
    a plain story-continuation prompt that never asked for a question-and-answer format.
    This is a real, directly observed instance of the instruction corpus's format bleeding
    into general continuation behavior — see "Failure modes."

**Interpretation:** the hypothesis holds, narrowly and honestly. The measured similarity
score moved in the predicted direction on every training example's summary statistic, and
the qualitative before/after generations show the same trend the metric captures — but the
effect is modest (roughly +11% relative), and neither model produces a coherent, correct
answer. This matches `02-pretraining-objective`'s own honest framing: the *mechanism*
(reformatted next-token prediction moving generation toward the target shape) is real and
measurable; the *capability* (an actually useful instruction-following model) requires far
more capacity, data, and steps than this toy notebook uses, exactly as
`02-pretraining-objective`'s "Failure modes" argued for raw pretraining.

**Limitations, stated directly:** one 24-pair hand-written instruction set (a real
instruction-tuning dataset has thousands to millions of examples), no hyperparameter
search over the fine-tuning learning rate or step count, the similarity metric is
evaluated in-sample (the same 24 questions used for training), so it measures
memorization-adjacent format adherence rather than generalization to unseen instructions,
and — most importantly — none of the above numbers claim the resulting model is a usable
instruction-follower; see "Failure modes" for exactly why not.

## Failure modes

- **Catastrophic forgetting, directly observed, not just cited.** The "forgetting check"
  above is a real, measured instance of `07-nlp/05-transformers-and-huggingface`'s
  catastrophic-forgetting failure mode: after Stage 2, a plain story-continuation prompt
  (`"The old "`) that never asked for a question-answer exchange now produces `"Q:"` and
  `":"` fragments bleeding into its output — 150 fine-tuning steps on a 24-example,
  strongly repetitive instruction corpus have visibly pulled the model's general
  next-character statistics toward the instruction template, at the expense of some of its
  ability to produce plain, template-free prose. `07-nlp/05`'s notes.md names the general
  mechanism ("fine-tuning updates every weight in the pretrained model... too many epochs
  on a narrow task can overwrite the general [ability] pretraining produced") and its
  mitigation (a reduced learning rate, `1e-3` here vs. Stage 1's `3e-3`, and few steps,
  150 here). This notebook's own numbers show that mitigation reducing but **not
  eliminating** the effect — a real, honest limitation, not a hypothetical one.
- **Instruction-response format mismatches at inference time.** Instruction tuning teaches
  the model to respond to *the specific template it was trained on*
  (`"Q: {x}\nA: {y}\n\n"` here). A prompt that deviates from that exact template — different
  marker text, missing the trailing `\n`, a different capitalization convention — gives the
  model input shaped unlike anything in its fine-tuning data, and there is no guarantee it
  generalizes to the new shape; at production scale this is exactly why real instruction-
  tuned model cards specify an exact expected prompt template (e.g. specific chat-role
  tokens) and why deviating from it degrades output quality even for large, well-trained
  models.
- **Vocabulary coverage gaps for characters unseen in pretraining.** Four characters
  (`:`, `?`, `Q`, `W`) exist only in the instruction corpus, per "From-scratch
  implementation" — their embedding rows started at random initialization and were touched
  by gradient updates *only* during Stage 2's 150 steps, versus the rest of the vocabulary's
  300 (Stage 1) + 150 (Stage 2) = 450 steps of exposure. This is a real, small-scale analog
  of a well-known practical issue in instruction tuning real models: special/role tokens
  (`<|user|>`, `<|assistant|>`, etc.) that were rare or absent in pretraining data can have
  under-trained embeddings relative to the rest of the vocabulary, and receive
  disproportionately little gradient signal during a typically much-shorter fine-tuning
  stage.
- **Small, hand-written instruction sets teach format more than capability.** With only 24
  examples, the measured similarity gain (+0.034) is almost entirely attributable to the
  model picking up *some* signal about "shorter, `A:`-anchored continuations" rather than
  learning to actually answer new, unseen questions correctly — the in-sample-only
  evaluation in "Experiment" cannot distinguish genuine instruction-following ability from
  memorizing surface statistics of 24 specific training examples, and at this scale it is
  almost certainly closer to the latter.

## Real-world usage

- **Every production chat/assistant LLM** (ChatGPT-family, Claude, Llama-Instruct/Chat
  variants, and effectively every "instruct" or "chat" model released today) goes through
  a supervised instruction-tuning stage structurally identical to this topic's Stage 2 —
  start from pretrained weights, continue training on `(instruction, response)` pairs with
  the same next-token objective, at a much larger scale (often hundreds of thousands to
  millions of curated or synthetically generated examples) and typically followed by RLHF
  or DPO (named in "Conceptual foundation," not implemented here) for preference alignment.
- Instruction tuning is also the mechanism behind most **domain-specific fine-tuned
  assistants** (customer-support bots, coding assistants, medical/legal Q&A tools) — the
  same recipe as this topic, with a domain-specific instruction dataset in place of this
  notebook's 24 lighthouse Q&A pairs.
- The prompt-template discipline this topic's "Failure modes" names (exact match between
  training-time and inference-time formatting) is why real instruction-tuned model
  deployments use a fixed chat template (often enforced by the serving framework, e.g.
  HuggingFace `transformers`' `apply_chat_template`) rather than leaving prompt formatting
  to each caller.

## Mental model

A pretrained model has only ever practiced one move: given some text, guess what comes
next in *that same text*. Instruction tuning does not teach it a second move — it teaches
it that a new *shape* of text (`"request, then response"`) exists, by showing enough
examples of that shape that "guess what comes next" starts meaning "guess what a response
looks like" once the model sees the request-response boundary. Same gradient descent, same
cross-entropy loss, same underlying prediction machinery as pretraining — the only thing
instruction tuning changes is *what the model is shown enough times to treat as normal*.
And because it is still ordinary fine-tuning on a pretrained model's full weight set, it
inherits fine-tuning's oldest problem for free: teach it a new normal too aggressively, and
some of the old normal gets overwritten along with it.

## Questions to think about

1. "Why simpler approaches fail" argues few-shot prompting only works on models whose
   pretraining data happened to contain naturally-occurring instruction-shaped text. If you
   pretrained a `TinyGPT`-scale model on a corpus that *did* contain some naturally
   dialogue-like text (e.g. a corpus of transcribed Q&A), would you expect few-shot
   prompting alone (no fine-tuning) to produce a measurably higher pretrained-only
   similarity score in this notebook's Section 7 metric than the 0.311 actually measured?
   Why or why not?
2. Stage 2 used a 3x lower learning rate and half as many steps as Stage 1. The
   "forgetting check" still showed template bleed-through into a plain prompt despite that
   mitigation. If you increased Stage 2's learning rate back to Stage 1's `3e-3` while
   keeping 150 steps, what would you predict happens to (a) the mean similarity score, and
   (b) the severity of the forgetting check's template bleed-through — and would those two
   predictions move in the same direction or opposite directions?
3. Four characters (`:`, `?`, `Q`, `W`) were seen only during Stage 2's 150 steps, not
   Stage 1's 300. If the instruction template were changed to avoid introducing any new
   characters at all (e.g. reusing only characters already in the pretraining vocabulary),
   what specifically would you expect to change about Stage 2's training loss curve and
   about the "vocabulary coverage gaps" failure mode?
4. This topic's "Conceptual foundation" says RLHF/DPO train a model to prefer *better*
   responses among candidates, which supervised instruction tuning alone cannot do. Given
   this notebook's Section 7 metric (similarity to a single fixed target response), could
   that metric distinguish "the model always outputs the exact training answer" from "the
   model outputs a fluent, reasonable, but differently-worded correct answer"? What does
   that limitation suggest about why real alignment pipelines need a preference-based
   objective in addition to supervised fine-tuning?
5. `07-nlp/05-transformers-and-huggingface`'s fine-tuning step masked nothing — it trained
   a fresh classification head against a scalar label. This topic's Stage 2 computes
   cross-entropy over *every* character position in `"Q: {q}\nA: {a}\n\n"`, including the
   question text itself, not just the answer. What would change about Stage 2's loss
   value and about the model's behavior if the loss were computed only over the answer
   segment (`{a}` and its trailing `\n\n`), masking the question and `"Q: "`/`"A: "`
   markers out of the loss entirely?
