# First-Principles Curriculum — Design Spec

**Status:** Approved (owner sign-off in chat, 2026-08-23). Binding authority for every phase plan under this initiative.

## 1. Identity

This repository is a first-principles, hands-on ML/AI engineering curriculum: Python, statistics, data science, machine learning, deep learning, NLP, MLOps, distributed data, and — eventually — Edge AI. It teaches the reasoning underneath algorithms and tools, then applies that reasoning to increasingly realistic systems. It is not course notes, not an API reference, not a technology showcase.

**Central learning philosophy:**

> Problem → Intuition → Concept → Mathematics → Tiny Implementation → Experiment → Failure → Explanation → Practical Tool → Production Application

Every important concept answers: what problem does it solve; why does that problem exist; what happens if we do nothing; what's the simplest solution; what assumptions does it make; where does it fail; what idea improves it; what tradeoffs does the improvement introduce; how is it implemented mathematically; how does the industry/framework implementation map back to the underlying idea.

Avoid "X is an algorithm used for Y." Prefer "We have this problem. A simpler method fails because of this limitation. Therefore we need this idea. That idea leads to X."

Frameworks are taught only after the underlying concept exists (§7 of the owner's brief) — never inserted as disconnected tutorials. No technology is added because it looks good on a résumé; if PyTorch, Kafka, PySpark, MLflow, etc. appear, their purpose must be understandable from the problem they solve.

The repository distinguishes **Learned** / **Currently learning** / **Planned** honestly (never writes advanced material as if already understood), and is written in durable, timeless language — no "today I learned," no course-specific phrasing, no diary content. It should be discoverable and useful years from now.

## 2. Standard topic structure

```
topic/
├── README.md      — orientation: what/why/prerequisites/what you'll build/where it appears in real systems/what's next
└── notes.md        — durable conceptual explanation (see template below)
```

Notebooks are laboratories, not the application — they answer one question each, numbered (`001_dataset_exploration.ipynb`, not `final.ipynb`/`final2.ipynb`). Reusable logic that emerges from an experiment moves into Python modules, not more notebook cells. The repo should visibly teach the distinction between experiment / reusable code / production system.

### `notes.md` template (binding for every topic under this initiative)

1. **Problem** — the real problem, stated first.
2. **Intuition** — no jargon; concrete/numerical examples.
3. **Why simpler approaches fail** — the limitation that motivates the concept. Critical section; never skipped.
4. **Mathematical foundation** — derive, don't dump. LaTeX (`$...$` / `$$...$$`). Every symbol explained.
5. **Algorithm** — actual steps; pseudocode where useful.
6. **From-scratch implementation** — Python/NumPy/SciPy only, when pedagogically valuable. Not a reimplementation of a production system — the goal is understanding, not replacing mature libraries. Skipped when it wouldn't add insight (§8 of owner's brief: build from scratch *selectively*).
7. **Practical implementation** — the corresponding production/common library, explicitly mapped back to the from-scratch step (e.g. manual gradient descent → NumPy → PyTorch/Keras).
8. **Experiment** — hypothesis, setup, expected result, actual result, interpretation, limitations. Measurable.
9. **Failure modes** — where the concept breaks (overfitting, leakage, distribution shift, numerical instability, class imbalance, etc., as applicable).
10. **Real-world usage** — where this matters in actual engineering systems.
11. **Mental model** — one compact, memorable closer.
12. **Questions to think about** — reasoning questions, not trivia (require applying the concept, e.g. "you doubled batch size, throughput rose but p99 latency got worse — why?").

Not every topic needs all 12 sections at full weight — e.g. a topic with no meaningful failure mode doesn't force one — but Problem, Why-simpler-fails, Math, Mental Model, and Questions are the load-bearing sections that must not be skipped for any topic classified "conceptual" (as opposed to a pure tooling/setup topic like `08-mlops-deployment/02-git`, which gets a lighter treatment scaled to its nature).

### Quality bar (every topic, before it's marked complete)

- Explanation starts from a problem; intuition is jargon-free; assumptions explicit; math explained; at least one concrete example.
- A from-scratch implementation exists where useful; framework usage is explained, not copied; practical code actually runs.
- A measurable experiment exists with recorded results and discussed limitations.
- Failure modes discussed; real-world context given; connection to larger systems shown.
- Prerequisites and next-topic are clear enough for an independent learner to follow.

## 3. Repository structure — additive only

No renumbering of existing sections `01`–`08`. New tracks are appended; gaps in existing numbering (`08-mlops-deployment` currently has `01,02,05,06` — `03,04` reserved) are filled in place, not renumbered around.

**Existing (retrofit in place):**
```
01-python-foundation   02-statistics   03-data-analysis   04-feature-engineering
05-machine-learning    06-deep-learning   07-nlp   08-mlops-deployment
```

**New, appended:**
```
09-pytorch             — after 06 is retrofitted; maps every NumPy/Keras concept already built to autograd/nn.Module
10-distributed-data     — PySpark (local mode) + streaming/Kafka concepts, single-machine-bottleneck-first framing
07-nlp/05-transformers-and-huggingface — extends the existing from-scratch attention work (06/05), not a disconnected section
08-mlops-deployment/03-testing-ci, /04-model-packaging-versioning, /07-cicd, /08-monitoring — fills the existing gap + extends the progression
11-generative-ai        — GANs and diffusion models, first-principles (added 2026-08-23, owner request — see §7)
12-reinforcement-learning — MDPs → Q-learning → policy gradients, first-principles (added 2026-08-23, owner request)
13-llms-from-scratch    — tokenizer → pretraining objective → instruction tuning, conceptual + toy-scale only (added 2026-08-23, owner request)
14-multi-agent-systems  — agent communication/orchestration patterns, multi-agent swarms (added 2026-08-23, owner request)
15-agent-skills-and-mcp — Agent Skills and Model Context Protocol tooling, first-principles (added 2026-08-23, owner request)
```

**Explicitly out of scope for this repository:** Go, DSA (owner confirmed — tracked elsewhere). Edge AI / robotics: acknowledged as a future direction (owner roadmap item 9) but gets no content yet — only an honest "Planned" marker, per the Learned/Currently-learning/Planned distinction in §1.

**`projects/`:** root level holds only external-project index cards (`spi.md`, `sentinel.md`, `demandpulse.md`, `edgesense.md`, …) — problem / why it matters / concepts learned / technologies / prerequisites / link to the actual repo / expected learning outcomes. No full implementations in-repo. `projects/beginner/*` (Titanic EDA, Iris classifier, house price, student performance) stays as lightweight in-repo notebooks — these are simple enough that they don't need their own external repos, and are explicitly distinct in kind from the external "real" engineering projects indexed at the `projects/` root.

## 4. Repository constitution

`AGENTS.md` at repo root is the canonical, durable statement of this philosophy (this spec's content, condensed into an operating document for anyone — human or agent — working in the repo). `CLAUDE.md` and `GEMINI.md` are thin pointers to `AGENTS.md` so any assistant picks up the same rules.

## 5. Phased execution

Each phase is its own plan (`docs/superpowers/plans/`), executed via `superpowers:subagent-driven-development`, reviewed independently. Phases are dispatched **sequentially** (implementers never run in parallel, per that skill's rule — "work on multiple phases at once" means the phases queue and each gets full treatment, not concurrent implementer dispatch).

| Phase | Scope |
|---|---|
| 0 | `AGENTS.md`/`CLAUDE.md`/`GEMINI.md` + this spec (repo constitution) |
| 1 | Retrofit `05-machine-learning` (17 topics: 13 rewrite-in-place, 3 write-from-scratch — `03-polynomial-regression`, `07-svm`, `18-pca` — plus a new `bias-variance-tradeoff` topic) into the new template |
| 2 | Retrofit `06-deep-learning` (5) + `07-nlp` (4) into the new template |
| 3 | Complete `08-mlops-deployment` (`01,02,05,06` retrofit + new `03,04,07,08`) with the problem-first MLOps progression (§14 of owner's brief) |
| 4 | New `09-pytorch` |
| 5 | New `10-distributed-data` (PySpark + Kafka/streaming) |
| 6 | New `07-nlp/05-transformers-and-huggingface` |
| 7 | Retrofit `01-python-foundation`, `02-statistics`, `03-data-analysis`, `04-feature-engineering` (foundational, lowest urgency) |
| 8 | `projects/` restructure to index-cards |
| 9 | New `11-generative-ai` (GANs, diffusion models) |
| 10 | New `12-reinforcement-learning` (MDPs, Q-learning, policy gradients) |
| 11 | New `13-llms-from-scratch` (tokenizer → pretraining objective → instruction tuning) |
| 12 | New `14-multi-agent-systems` (communication protocols, orchestration, swarms) |
| 13 | New `15-agent-skills-and-mcp` (Agent Skills, Model Context Protocol) |

Phases 9–13 were added 2026-08-23 at the owner's explicit request, after reviewing github.com/rohitg00/ai-engineering-from-scratch for structural (not content) ideas — see §7. Phase order beyond that point is not fixed; the owner may reprioritize which of 4–13 goes next at any time.

## 6. Non-goals

Not a giant destructive rewrite in one pass. Not deleting working notebooks to make the structure look cleaner — existing good notebooks/explanations are improved, not discarded. Not adding a technology without a legible conceptual on-ramp. Not writing content as though the owner already knows something they're still learning.

**Phases 9–13 specifically:** no heavy/long-running training. RL agents, GANs/diffusion models, and LLM pretraining are conceptually expensive to train for real — every from-scratch and practical-implementation step in these phases uses toy-scale data and small enough models/iteration counts to run in seconds-to-low-minutes, the same discipline already established in Phase 3 (e.g. `08-monitoring`'s drift-detection demo). Where a real, useful demonstration genuinely cannot run at toy scale (e.g. an actual GPT-2-scale pretraining run), the content is written and reviewed like Phase 3's un-executed Dockerfile/DVC sections — explicit derivation and a correct, realistic script — but honestly marked as not executed in this environment, never fabricated as if it ran.

## 7. Amendment log

- **2026-08-23:** owner asked to review github.com/rohitg00/ai-engineering-from-scratch (511-lesson, 20-phase curriculum: math foundations → classical ML → DL → CV/NLP/speech → transformers → generative AI → RL → LLMs-from-scratch → LLM engineering → multimodal → agent systems → production). Comparison: its core lesson shape (Problem → Concept → from-scratch build → framework use) converges with this spec's Problem → Why-simpler-fails → From-scratch → Practical sequence — independently arrived at, not copied. Its "Ship It" step (every lesson produces a reusable artifact — a prompt, skill, agent, or MCP server) doesn't fit classical ML/DL/MLOps topics but is a natural fit for the new `15-agent-skills-and-mcp` phase. No content, prose, or code was copied from that repo — only the structural comparison above informed this amendment. Owner then explicitly requested Phases 9–13 be added to this repo's own scope, overriding this spec's earlier "not become an everything-AI-dump" caution for these five specific topics; the "no heavy training" non-goal in §6 is the binding constraint that keeps this expansion honest rather than resume-padding.
