# 01 — Agent Skills and Progressive Disclosure

## Problem

An agent that is genuinely useful across many domains — cleaning data, drafting emails,
reviewing code, writing SQL, writing commit messages, and hundreds of other specialized
workflows besides — cannot have the full instructions for every one of those workflows
sitting in its context window at once. A context window is finite (measured in tokens,
and every token used for instructions the current task doesn't need is a token not
available for the actual conversation, the actual code, the actual data). Most of an
agent's *potential* capabilities are irrelevant to any single task it is asked to do
right now. The problem this topic solves: how does an agent know a capability exists and
choose to use it, without paying the full cost of loading that capability's complete
instructions until it actually needs them?

## Intuition

Think of a large reference library versus a stack of books already open on your desk.
You don't read every book in the library before starting any task — you scan the spines
(title + one-line blurb) to find the one book relevant to today's problem, then you pull
just that one book off the shelf and actually read it. The spines are cheap to scan even
when there are thousands of them; the full text of a book is expensive to read and you
only pay that cost for the one book you actually need.

That's the whole idea behind an **Agent Skill**: a short name and one-line description
that stay visible all the time (cheap — like a spine), and a full instructions body that
only gets loaded into context when that specific skill is actually selected for use
(expensive, but paid for exactly once, for exactly the skill that matters right now).

Concretely, in this topic's from-scratch implementation (notebook, actually run): five
toy skills — `csv-cleaning`, `email-drafting`, `code-review-checklist`,
`sql-query-builder`, `commit-message-writer` — are registered. Their five one-line
descriptions together cost about 400 characters to keep always visible. Their five full
instruction bodies together cost roughly 3,700 characters. Given the task "clean this
messy CSV, the price column has dollar signs," a deterministic selector picks
`csv-cleaning` and only *that* skill's ~737-character body is loaded — not the other
four bodies (~2,988 characters saved, a measured 70.7% reduction at just 5 skills; see
Experiment).

## Why simpler approaches fail

**"Just put every skill's full instructions in the system prompt."** This is the naive
alternative, and it is tempting because it is simple to build: no selection logic
needed, nothing can be "missed." Two things break it as the skill library grows:

1. **It does not scale.** This topic's own measured numbers (Experiment, below): at 5
   registered skills, loading every full body costs 4,225 characters versus 1,237 for
   progressive disclosure (a 70.7% saving). At 20 skills: 20,229 vs. 2,668 characters
   (86.8% saving). At 50 skills: 54,276 vs. 5,319 characters (90.2% saving). The flat
   approach's cost grows roughly linearly with the number of registered skills; the
   progressive-disclosure cost grows far more slowly, because its dominant term stays
   "all the short descriptions plus exactly one full body," regardless of how many
   skills exist. The savings percentage *increases* with N — this is not a fixed
   constant-factor cost, it is a scaling problem that gets strictly worse the more
   capabilities an agent is given.

2. **Even where it technically still fits in the window, it dilutes attention.** A
   context window that could in principle hold 50 skills' full bodies is not "free" just
   because it fits — every irrelevant instruction competing for the model's attention on
   the current task is a real cost that a raw token-budget argument misses entirely.
   Forty-nine sets of instructions the current task doesn't need are forty-nine
   distractions the model has to implicitly filter past on every single turn, not just
   once at setup. A `csv-cleaning` task does not benefit from having `sql-query-builder`
   and `commit-message-writer` instructions sitting in context; at best they're inert
   weight, at worst they're a source of the model blending unrelated instructions
   together.

The natural fix is not "compress the instructions" (that just delays the same scaling
problem) but "don't load instructions that weren't asked for" — which requires a
mechanism for the agent to know a skill *exists* and roughly *what it's for*, without
paying to load *how it works*, until it decides that skill is the one it needs. That
mechanism is progressive disclosure.

## Conceptual foundation (documented substitution for "Mathematical foundation")

This is a systems/architecture topic — like `14-multi-agent-systems`, it doesn't have a
derivation in the traditional sense. What it has instead is a precise structural
definition, stated here in place of a mathematical derivation, matching the convention
`14-multi-agent-systems/01-communication-protocols/notes.md` established for this
category of topic.

**Progressive disclosure, formalized:** a skill is a pair of tiers with different
loading conditions.

$$
\\text{Skill} = (\\underbrace{\\text{name}, \\text{description}}_{\\text{Tier 1 — always loaded}},\\ \\underbrace{\\text{body}}_{\\text{Tier 2 — loaded only if selected}})
$$

Formally, for a registry of $n$ skills $S = \\{s_1, \\dots, s_n\\}$, define:

- $d(s_i)$ = character (or token) length of skill $s_i$'s one-line description.
- $b(s_i)$ = character (or token) length of skill $s_i$'s full instructions body.
- $\\text{select}(t, S) \\to s_k$ = a selection function that, given a task $t$, returns
  exactly one skill $s_k \\in S$ (or, in a real system, possibly a small explicit
  ranked shortlist — but never "all of them").

Then the two loading strategies cost, in units of context consumed:

$$
\\text{Cost}_{\\text{flat}}(S) = \\sum_{i=1}^{n} \\big(d(s_i) + b(s_i)\\big)
$$

$$
\\text{Cost}_{\\text{progressive}}(t, S) = \\sum_{i=1}^{n} d(s_i) \\;+\\; b\\big(\\text{select}(t, S)\\big)
$$

The flat strategy's cost grows with $\\sum b(s_i)$ — it pays for *every* skill's full
body on *every* task, regardless of relevance. The progressive strategy's cost is
$\\sum d(s_i)$ (the always-visible tier, small per skill and summed once) plus exactly
one $b(s_k)$ — the one skill actually selected. As $n \\to \\infty$ with
$d(s_i) \\ll b(s_i)$ (true of every skill in this topic's toy registry — descriptions
run 75–95 characters, bodies run 644–902 characters), the ratio
$\\text{Cost}_{\\text{progressive}} / \\text{Cost}_{\\text{flat}} \\to 0$: progressive
disclosure's *relative* advantage strictly improves as the skill library grows, which
is exactly the trend measured in the Experiment section below (70.7% → 86.8% → 90.2%
savings at $n = 5, 20, 50$).

This is a real, two-tier structure, not an invented one for this notebook. Anthropic's
Agent Skills feature and this very repository's own Claude Code plugin system
(`superpowers`, whose skills this session is running under) use exactly this shape: a
`SKILL.md` file with YAML frontmatter (`name:`, `description:` — always shown to the
model) followed by a Markdown body (the full instructions — loaded into context only
when that skill is invoked). Contrast this precisely with a **flat model**: one in
which every registered capability's full body is always present in context regardless
of whether the current task touches it — the model in "why simpler approaches fail,"
above, and the `flat_baseline_chars()` measurement in the notebook.

## Algorithm

**Registration (build the always-visible tier):**
1. For each skill, parse a frontmatter-style source (`---\\nname: ...\\ndescription:
   ...\\n---\\n<body>`) into a `Skill(name, description, body)` object.
2. Add it to a `SkillRegistry`. The registry's `all_descriptions()` — every registered
   skill's name and one-line description — is what stays in context at all times.

**Selection (deterministic, non-LLM — see substitution note below):**
1. Given a task string $t$, normalize it into a token set (lowercase, strip
   punctuation, drop stopwords and very short tokens).
2. For each registered skill $s_i$, normalize its *description* (not its body) into a
   token set the same way.
3. Score $s_i$ as $|\\text{tokens}(t) \\cap \\text{tokens}(\\text{description}(s_i))|$ —
   the size of the token overlap.
4. Select $\\arg\\max_i \\text{score}(s_i)$.

**Loading (progressive disclosure in effect):**
1. Only after step 4 picks a specific $s_k$, call `load_full_body(s_k)` — this is the
   only point in the whole pipeline where a skill's *body* (the expensive tier) is
   touched at all. Every skill that was not selected never has its body read.

## From-scratch implementation

Built and actually run in
[001_progressive_disclosure_skill_selection.ipynb](001_progressive_disclosure_skill_selection.ipynb):

- A `Skill` dataclass (`name`, `description`, `body`) with a minimal frontmatter parser
  (`Skill.from_markdown`) and serializer (`to_markdown`) — round-trip tested for real
  (`Round-trip parse OK.` — actual printed output).
- Five hand-written toy skills with realistic descriptions and multi-step bodies:
  `csv-cleaning`, `email-drafting`, `code-review-checklist`, `sql-query-builder`,
  `commit-message-writer`.
- A `SkillRegistry` exposing `all_descriptions()` (the always-visible tier),
  `select(task)` (the deterministic keyword-overlap heuristic — **explicitly not an LLM
  call**, per this whole section's binding constraint; see honest statement below), and
  `load_full_body(name)` (the second tier, called only post-selection).
- Selection actually run against five toy task prompts, one aimed at each skill. All
  five were routed correctly (real output, e.g. `TASK: Please clean this messy customer
  export CSV... -> selected: csv-cleaning (scores: [('csv-cleaning', 6), ...others 0])`).

**Honest substitution statement:** a real Agent Skills system selects among skills by
having the model itself read every skill's name + description and reason about which
one applies — genuine semantic understanding, not bag-of-words overlap. This section's
binding constraint (no live LLM calls of any kind, no network calls) makes that
approach unavailable in this environment. The keyword-overlap heuristic implemented
here is a deliberately simple, fully deterministic, fully inspectable stand-in that
demonstrates the *mechanism* (two tiers, selection gates which body loads) faithfully,
while its *quality* is visibly worse than what a real model-driven selector would
achieve — see Failure modes, below, for exactly where and how it breaks.

## Practical implementation

The from-scratch `Skill`/`SkillRegistry` pair above *is* the practical implementation
for this topic — there is no separate "production library" step, because the two-tier
file format (`SKILL.md` with frontmatter `name`/`description` + Markdown body) is
already the real, production format used by Anthropic's Agent Skills feature and by
this repository's own Claude Code plugin system. The mapping back to the from-scratch
step is direct rather than approximate:

| From-scratch (this notebook) | Real system (Agent Skills / `superpowers` plugin) |
|---|---|
| `Skill.from_markdown()` frontmatter parser | The real `SKILL.md` loader that reads `name:`/`description:` frontmatter |
| `SkillRegistry.all_descriptions()` | The list of installed skills' names + descriptions shown to the model at all times |
| `SkillRegistry.select(task)` (keyword overlap) | The model itself reading names + descriptions and choosing which skill to invoke |
| `SkillRegistry.load_full_body(name)` | The harness loading a specific `SKILL.md`'s full body into context only on invocation |

The one substituted piece — selection — is exactly the piece this section's no-live-LLM
constraint requires substituting, and is called out above rather than left implicit.

## Experiment

**Hypothesis:** as the number of registered skills $n$ grows, the flat "load every
body" baseline's context cost grows roughly linearly in $n$, while progressive
disclosure's cost grows far more slowly — dominated by one full body regardless of
$n$ — so progressive disclosure's *relative* savings should increase with $n$, not stay
constant.

**Setup:** starting from the 5 real hand-written skills, 45 additional synthetic toy
skills were generated programmatically (`make_synthetic_skill`) by recombining the
description/body vocabulary of the 5 real skills at realistic lengths — clearly labeled
as synthetic, since hand-writing 50 plausible skills was impractical. Registries of
size $n = 5, 20, 50$ were built by taking the first $n$ skills from (5 real + 45
synthetic). For each registry, `flat_baseline_chars()` summed every skill's full
name+description+body; `progressive_disclosure_chars()` summed every skill's
name+description (always-visible tier) plus the body of just the one skill selected
for a fixed task (the csv-cleaning-flavored task prompt).

**Expected result:** progressive disclosure noticeably cheaper at every $n$, with the
percentage saved increasing as $n$ grows.

**Actual result (real measured output from the notebook):**

| N skills | flat (all bodies) | progressive disclosure | savings | savings % |
|---:|---:|---:|---:|---:|
| 5 | 4,225 chars | 1,237 chars | 2,988 chars | 70.7% |
| 20 | 20,229 chars | 2,668 chars | 17,561 chars | 86.8% |
| 50 | 54,276 chars | 5,319 chars | 48,957 chars | 90.2% |

**Interpretation:** the hypothesis held. The flat baseline's cost grew by roughly
12.8x from $n=5$ to $n=50$ (4,225 → 54,276), tracking the growth in $n$ almost
exactly, because it pays for every body every time. Progressive disclosure's cost grew
by only about 4.3x over the same range (1,237 → 5,319) — most of that growth is just
the sum of many short descriptions, not bodies; the one full body it pays for stays a
constant-size term. This directly confirms the scaling argument in "Why simpler
approaches fail": the flat approach isn't merely a fixed inefficiency, it's a
strategy whose *relative* disadvantage compounds as an agent is given more
capabilities.

**Limitations:** character count is used as a token-count proxy, not an actual model
tokenizer (real tokenization would differ in exact numbers but not in the qualitative
scaling trend). The task prompt was held fixed across all three registry sizes rather
than varied. The 45 synthetic skills are recombinations of the 5 real skills'
vocabulary, not independently authored realistic skill text, so absolute character
counts should be read as illustrative of the scaling *shape*, not as a benchmark of
what a real 50-skill production library's typical body length would be.

## Failure modes

**1. Keyword-overlap heuristic picks the wrong skill — concrete, reproduced.** Task:
*"Write a status update email summarizing the pull request review comments and index
changes for the SQL migration."* The task's real intent is drafting an email
(`email-drafting`), but the sentence also contains vocabulary that overlaps heavily with
two unrelated skills. Actual measured scores (real notebook output):
`code-review-checklist=3, sql-query-builder=2, email-drafting=1,
commit-message-writer=1, csv-cleaning=0` — the heuristic selected
`code-review-checklist`, the **wrong** skill. This happens because the heuristic counts
raw token overlap with no notion of which words carry the sentence's actual verb
("write a ... email") versus which words are merely topical nouns the email happens to
discuss ("pull request," "review," "SQL," "index," "migration"). A bag-of-words match
cannot distinguish "an email *about* a PR and a SQL migration" from "a task *that is* a
PR/SQL task." This is exactly the class of error the honest substitution statement
above predicted: a real model-driven selector reads intent; this deterministic stand-in
counts words.

**2. Overlapping/ambiguous skill descriptions — concrete, reproduced.** Two
legitimately related skills, `sql-query-builder` ("optimize SQL SELECT queries -
joins, aggregations, window functions, indexes") and `sql-schema-reviewer` ("Review SQL
schema design - joins, indexes, normalization, and query performance"), were registered
together. Task: *"Look at these SQL joins and indexes and tell me if the query will be
fast."* Actual measured scores: `sql-schema-reviewer=4, sql-query-builder=3` — a
near-tie, not an exact one, which is arguably worse than a clean tie for a production
system: a near-tie gives false confidence that the top-scored skill is clearly right,
when in fact `sql-query-builder` — the skill actually written around "optimiz[ing]...
query performance," the concern the task states — was the runner-up purely because
`sql-schema-reviewer`'s description happens to repeat "joins" and "indexes" more
literally. An exact tie is easy to detect and escalate programmatically (e.g. `if
top_score == second_score: ask_for_clarification()`); a near-tie like this one is
silently swallowed by `max()` and looks like an ordinary, confident selection. This is
a genuine authoring problem too, not just a heuristic weakness: when two real skills'
purposes actually overlap, no selection mechanism — keyword-based or model-based — can
fully resolve the ambiguity without either more distinguishing descriptions or reading
further than the description tier.

## Real-world usage

- **Anthropic's Agent Skills feature** — the direct real-world instance of this exact
  two-tier pattern: a skill is packaged as a folder with a `SKILL.md` file (YAML
  frontmatter `name` + `description`, always visible to the model; Markdown body,
  loaded only when the skill is invoked), letting an agent be handed a large library of
  specialized capabilities without paying full context cost for all of them
  simultaneously.
- **This repository's own tooling** — the Claude Code session that authored this
  section runs under the `superpowers` plugin, whose skills (`test-driven-development`,
  `systematic-debugging`, `writing-plans`, and dozens more) are stored as real
  `SKILL.md` files with exactly this frontmatter/body split — inspected directly while
  writing this topic (e.g.
  `~/.claude/plugins/cache/claude-plugins-official/superpowers/6.3.0/skills/test-driven-development/SKILL.md`)
  to ground this notes.md's description in a real file rather than an invented one.
- **Any plugin/extension marketplace with many optional capabilities** — browser
  extensions, IDE plugins, app stores — faces the identical shape of problem (many
  possible capabilities, one user/session needs only a few at a time), and the common
  solution (a short always-visible listing plus a full payload fetched only on
  install/activation) is the same progressive-disclosure structure at a different
  layer.

## Mental model

**A skill is a business card, not a résumé, until you actually call the person.** The
card (name + one-line description) sits on the table for every skill you have, all the
time — cheap to keep around, cheap to scan. You only ask for someone's full résumé
(the complete instructions body) once you've decided, from the card alone, that this is
the person you need for the task in front of you right now.

## Questions to think about

1. The measured savings percentage (70.7% → 86.8% → 90.2%) *increases* with the number
   of registered skills. Using the cost formulas in "Conceptual foundation," explain
   algebraically why this trend is guaranteed to continue as $n$ grows further, given
   that $d(s_i) \\ll b(s_i)$ holds for every skill.
2. The wrong-skill failure mode (email task routed to `code-review-checklist`) happened
   because the task text contained vocabulary borrowed from *other* skills' domains. If
   you were only allowed to change the *skill descriptions* (not the selection
   algorithm), what would you change about `email-drafting`'s and
   `code-review-checklist`'s descriptions to reduce the chance of this specific
   misroute — and could any wording change fully eliminate this class of failure, or
   only reduce its likelihood?
3. This topic's `select()` only ever returns exactly one skill. Real systems sometimes
   return a small ranked shortlist instead of a single winner, and let a further step
   (a person, or the model reading full bodies) disambiguate. What would
   `Cost_progressive` look like if `select()` returned the top-$k$ skills and all $k$
   full bodies were loaded — and at what value of $k$ does progressive disclosure stop
   being meaningfully cheaper than the flat baseline?
4. The near-tie failure mode (`sql-schema-reviewer` vs. `sql-query-builder`) was called
   out as arguably worse than an exact tie, because it hides ambiguity behind an
   apparently confident answer. Design a concrete, checkable rule (not "ask a human"
   vaguely, an actual threshold or comparison you could put in code) that would flag
   this specific near-tie as ambiguous rather than silently returning the top score.
5. Anthropic's real Agent Skills feature and this section's toy implementation share
   the two-tier *structure* but differ in the selection *mechanism* (model reasoning
   vs. keyword overlap). Name one thing a token/keyword-overlap selector can guarantee
   that a model-based selector cannot (hint: think about determinism, cost, and
   auditability), and one thing it can never do that a model-based selector can.
