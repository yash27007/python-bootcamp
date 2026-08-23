# AGENTS.md — Repository Constitution

This file is the durable, binding statement of what this repository is and how work on it is done. Any agent (or human) making changes here should read this first. The full rationale lives in `docs/superpowers/specs/2026-08-23-first-principles-curriculum-design.md` — this file is the operating summary.

## What this repository is

A first-principles, hands-on ML/AI engineering curriculum covering Python, statistics, data science, machine learning, deep learning, NLP, MLOps, distributed data, and — eventually — Edge AI. It teaches the reasoning underneath algorithms and tools, then applies that reasoning to increasingly realistic systems.

It is **not**: a collection of copied course notes, an API reference, a framework-tutorial dump, or a technology showcase for a résumé.

## The learning philosophy

> Problem → Intuition → Concept → Mathematics → Tiny Implementation → Experiment → Failure → Explanation → Practical Tool → Production Application

Every important concept must answer:
1. What problem does this solve?
2. Why does that problem exist?
3. What happens if we do nothing?
4. What's the simplest possible solution?
5. What assumptions does that solution make?
6. Where does the simple solution fail?
7. What idea improves it?
8. What tradeoffs does the improvement introduce?
9. How is the concept implemented mathematically?
10. How does the industry/framework implementation map back to the underlying idea?

Write: *"We have this problem. A simpler method fails because of this limitation. Therefore we need this idea. That idea leads to X."* Never: *"X is an algorithm used for Y."*

Frameworks (PyTorch, PySpark, MLflow, Kafka, ...) are introduced only after the underlying concept exists, with an explicit mapping back to it (e.g. manual gradient descent → NumPy → PyTorch `optimizer.step()`). No technology is added just because it looks good on a résumé.

Distinguish **Learned** / **Currently learning** / **Planned** honestly. Never write a topic as though the reader already understands something they're still learning. Write in durable, timeless language — no "today I learned," no diary content, no course-specific phrasing. This repo should still teach someone correctly years from now.

## Standard topic structure

```
topic/
├── README.md   — what you'll learn, why it matters, prerequisites, what you'll build, where it shows up in real systems, what's next
└── notes.md    — the durable conceptual explanation
```

### `notes.md` sections (in order)

1. **Problem** — the real problem, first.
2. **Intuition** — no jargon; concrete/numerical examples.
3. **Why simpler approaches fail** — never skip this; it's the motivation for the concept.
4. **Mathematical foundation** — derive, don't dump. LaTeX (`$...$` / `$$...$$`), every symbol explained.
5. **Algorithm** — actual steps, pseudocode where useful.
6. **From-scratch implementation** — Python/NumPy/SciPy, when it adds real insight. Don't reimplement mature production systems for their own sake.
7. **Practical implementation** — the production/library version, explicitly mapped back to the from-scratch step.
8. **Experiment** — hypothesis, setup, expected result, actual result, interpretation, limitations.
9. **Failure modes** — where the concept breaks.
10. **Real-world usage** — where it matters in actual engineering.
11. **Mental model** — one compact, memorable closer.
12. **Questions to think about** — reasoning questions that require applying the concept, not trivia.

Every conceptual topic needs Problem, Why-simpler-fails, Math, Mental Model, and Questions at minimum. Pure tooling/setup topics (e.g. Git basics) get a lighter treatment scaled to their nature.

### Notebooks

Laboratories, not the application. Each notebook answers one question — numbered (`001_dataset_exploration.ipynb`), never `final.ipynb` / `final2.ipynb`. Reusable logic that emerges from an experiment moves into Python modules, not more notebook cells.

## Repository structure

No renumbering of existing sections for aesthetics. New tracks are appended; existing numbering gaps (e.g. `08-mlops-deployment` skipping `03`/`04`) get filled in place. Current top-level map:

```
01-python-foundation  02-statistics  03-data-analysis  04-feature-engineering
05-machine-learning   06-deep-learning  07-nlp  08-mlops-deployment
09-pytorch (planned)          10-distributed-data (planned)
11-generative-ai (planned)    12-reinforcement-learning (planned)
13-llms-from-scratch (planned) 14-multi-agent-systems (planned)
15-agent-skills-and-mcp (planned)
projects/  docs/
```

Sections 11–15 (generative AI, RL, LLMs-from-scratch, multi-agent systems, agent skills/MCP) carry one extra constraint: no heavy/long-running training. Every from-scratch and practical step in these sections runs at toy scale (seconds to low minutes). Where a genuinely useful demo can't run at toy scale, write and review the real script but mark it honestly as not executed in this environment — never fabricate output, same discipline as an un-executed Dockerfile.

`projects/` root holds only external-project index cards (problem, why it matters, concepts learned, tech, prerequisites, link to the real repo, expected outcomes) — no full implementations in-repo. `projects/beginner/*` is the exception: small, self-contained starter notebooks that don't warrant their own external repo.

Out of scope for this repository: Go, DSA (tracked elsewhere).

## Quality bar

Before any topic is marked complete:
- Explanation starts from a problem, intuition is jargon-free, assumptions are explicit, math is explained, at least one concrete example exists.
- A from-scratch implementation exists where it adds insight; framework usage is explained, not copy-pasted; the practical code actually runs.
- A measurable experiment exists with recorded results and stated limitations.
- Failure modes are discussed; real-world context is given.
- Prerequisites and the next topic are clear enough for an independent learner to follow.

## How work happens here

- Design work uses `superpowers:brainstorming` (architectural changes get a written spec in `docs/superpowers/specs/`).
- Implementation plans live in `docs/superpowers/plans/` and are executed via `superpowers:subagent-driven-development` — one plan per phase, tasks reviewed before being marked done.
- Never a single giant destructive rewrite. Existing good notebooks/explanations are improved, not discarded.
- Every new/modified notebook is executed end-to-end (`jupyter nbconvert --execute --inplace`) before being considered done — no notebook with unexecuted cells ships.
