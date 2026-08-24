# 01 – Agent Skills and Progressive Disclosure

Detailed notes (progressive disclosure formalized precisely as a two-tier structure —
always-visible name + one-line description vs. a full instructions body loaded only on
selection — contrasted with a "flat" always-load-everything model, plus an honest,
explicit statement of the no-live-LLM-selection substitution this whole section makes):
[notes.md](notes.md)

Real, actually-executed, from-scratch `Skill`/`SkillRegistry` implementation (plain
Python, no network, no LLM calls) — five hand-written toy skills, a deterministic
keyword-overlap selection heuristic correctly routing five toy task prompts, a real
measured context-character-cost experiment at 5/20/50 registered skills, and two
concrete, reproduced selection failures (a wrong-skill misroute and a near-tie between
overlapping skill descriptions), all with real pasted output:
[001_progressive_disclosure_skill_selection.ipynb](001_progressive_disclosure_skill_selection.ipynb)

## What you'll learn

Why an agent with access to hundreds of possible capabilities cannot simply have all of
their instructions loaded into context at once, and what "progressive disclosure" —
a short, always-visible summary per capability plus a full body loaded only when that
capability is actually selected — buys instead of the naive "put everything in the
system prompt" approach. Then: a from-scratch `Skill`/`SkillRegistry` pair built and
run for real, a deterministic (non-LLM) keyword-overlap selection heuristic, a measured
context-cost comparison against a flat baseline at growing registry sizes, and two
concrete failure modes of that heuristic.

## Why it matters

Every later topic in `15-agent-skills-and-mcp` assumes an agent can be handed a large,
growing library of specialized capabilities without every one of them permanently
occupying context — this topic is where that mechanism is built and stress-tested
first. It is also the first topic in the curriculum to work with a real, currently
running instance of the pattern it teaches: this very session runs under a Claude Code
plugin (`superpowers`) whose skills are stored as real two-tier `SKILL.md` files,
inspected directly while writing this topic rather than only described secondhand.

## Prerequisites

- Comfort with plain Python classes/dataclasses and basic string processing (regex
  tokenization) — no ML framework knowledge is required for this topic.
- `14-multi-agent-systems/01-communication-protocols` (optional but useful) — for the
  shape of a "systems/architecture topic with a documented math substitution," the same
  convention this topic's notes.md follows for its "Conceptual foundation" section.

## What you'll build

- A minimal `Skill` dataclass (`name`, `description`, `body`) with a frontmatter-style
  parser/serializer round-tripped for real (`Round-trip parse OK.`).
- Five hand-written toy skills — `csv-cleaning`, `email-drafting`,
  `code-review-checklist`, `sql-query-builder`, `commit-message-writer` — with
  realistic one-line descriptions and multi-step bodies.
- A `SkillRegistry` exposing the always-visible tier (`all_descriptions()`), a
  deterministic keyword-overlap `select(task)` heuristic (no LLM call, per this
  section's binding constraint), and `load_full_body(name)` (the second tier, only
  reached after selection).
- A real measured experiment: total context characters loaded under progressive
  disclosure vs. a flat "load everything" baseline at $n \\in \\{5, 20, 50\\}$
  registered skills — measured savings grow from 70.7% at $n=5$ to 90.2% at $n=50$,
  exactly as the scaling argument in notes.md predicts.
- Two concrete, reproduced selection failures: a real wrong-skill misroute caused by
  superficial keyword overlap, and a real near-tie between two legitimately
  overlapping skill descriptions.

## Where it appears in real systems

Anthropic's Agent Skills feature uses exactly this two-tier `SKILL.md` structure in
production — YAML frontmatter (`name`, `description`) always visible to the model, a
Markdown instructions body loaded only on invocation. This repository's own tooling
runs on the same pattern: the `superpowers` plugin's skills
(`test-driven-development`, `systematic-debugging`, `writing-plans`, and others) are
real files with this exact shape, one of which was read directly while writing this
topic's notes.md to ground the description in a real file rather than an invented one.
No live LLM call was made anywhere in this section — selection is a deterministic
keyword heuristic, an honest, explicitly stated substitution for the model-driven
selection a real system uses (see notes.md's "Real-world usage" and "Honest
substitution statement").

## What's next

Later `15-agent-skills-and-mcp` topics build on this topic's `Skill`/`SkillRegistry`
primitives and its progressive-disclosure vocabulary to tackle the Model Context
Protocol (MCP) — how an agent discovers and calls *external tools*, not just loads
*instructions*, using a related but distinct disclosure-and-selection problem.
