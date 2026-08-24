# Phase 7: Foundational Sections Retrofit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Retrofit `01-python-foundation` (13 topics), `02-statistics` (3 topics, already has notes.md — restructure into template), `03-data-analysis` (8 topics), and `04-feature-engineering` (4 flat notebooks — folderize first) into the 12-section AGENTS.md template. Lowest urgency phase (foundational material the owner already knows) but completing it for repo consistency and for future learners.

**Architecture:** 10 tasks, 2-3 topics each, grouped by section and natural theme. `01-python-foundation`'s topics are software-engineering fundamentals, not ML — "from-scratch implementation" here means implementing the underlying mechanism by hand where one exists (a manual linked-list to motivate why Python's list/dict have the complexity they do, a hand-rolled context-manager to motivate `with`, a manual thread-safety bug reproduced before showing `threading.Lock`) rather than a NumPy/statistics derivation. `02-statistics` already has real notes.md content — pure restructure, no new content needed beyond the 12-section reshuffling and any missing sections (Experiment, Failure modes, Questions likely need adding). `03-data-analysis` and `04-feature-engineering` are tool topics (NumPy/Pandas/Matplotlib/Seaborn/SQLite) — "from-scratch" means the underlying data structure/algorithm the tool wraps (e.g. NumPy's broadcasting rules derived from array-shape arithmetic, Pandas' `groupby` as split-apply-combine implemented by hand in plain Python once, SQLite's B-tree index intuition).

**Tech Stack:** stdlib only for `01-python-foundation`; NumPy/Pandas/Matplotlib/Seaborn/SQLite (already installed) for `03-data-analysis`/`04-feature-engineering`.

**Spec:** `docs/superpowers/specs/2026-08-23-first-principles-curriculum-design.md`, `AGENTS.md`.

## Global Constraints

- 12-section notes.md template throughout. For topics with no meaningful math (most of `01-python-foundation`), use "Conceptual foundation" — MUST be documented inline (this exact requirement has needed fix rounds multiple times already, don't repeat).
- **Preserve existing content.** `02-statistics`'s 3 topics already have real notes.md — read fully, enumerate every formula/concept as a checklist before rewriting, confirm every item lands somewhere (same discipline as Phase 1). The other sections have no notes.md to preserve, but DO have existing notebooks/`.py` files — preserve/extend those, don't discard working code.
- `04-feature-engineering` currently has 4 flat notebook files with no per-topic folders — Task 10 must first create 4 topic folders (`01-missing-values/`, `02-handling-outliers/`, `03-data-encoding/`, `04-handling-imbalanced-dataset/`) and move each notebook into its folder (`git mv`, preserving history) before adding notes.md/README.md.
- Real code, actually executed, real output — never fabricate. All examples/demos here are fast (stdlib or small data) — no execution-time concerns like Phases 4-6 had.
- Every topic gets an orientation-format README; last task per section updates the section README + root README.
- Review level: light.
- Commit granularity: one commit per task.

---

### Task 1: Python Basics + Control Flow — `01-python-foundation/01-basics`, `02-control-flow`

**Files:** Modify both topics (add notes.md, extend existing notebook if useful)

Content, terse (implementer has full latitude within the template): Problem = what problem does a general-purpose language with these constructs solve vs. e.g. a calculator or a fixed script. Why-simpler-fails = e.g. for control-flow, "just write every branch as a separate script" doesn't compose. Conceptual foundation = Python's execution/evaluation model relevant to the topic (e.g. for basics: names as references, not boxes — mutable vs immutable distinction, derive why `a = b` behaves differently for a list vs an int). From-scratch = e.g. a manual state machine implemented without `if/elif` (using a dict-of-functions dispatch) to show what control flow automates. Practical = the existing notebook's content, extended if thin. Experiment, Failure modes (e.g. mutable-default-argument gotcha, off-by-one loop bugs), Real-world, Mental model, Questions.

- [ ] `git commit -m "Phase 7 Task 1: first-principles retrofit — Python basics, control flow"`

### Task 2: Data Structures + Functions — `03-data-structures`, `04-functions`

Content: data-structures' math/conceptual foundation IS real — Big-O complexity analysis of list/dict/set/tuple operations, derived (why dict lookup is O(1) amortized — hashing), not just stated. From-scratch = a tiny hand-rolled hash table (open addressing or chaining) to show what `dict` does underneath. Functions: closures/scope (LEGB), first-class functions, `*args`/`**kwargs` — from-scratch = manually implementing a decorator without `@` syntax to show what it desugars to.

- [ ] `git commit -m "Phase 7 Task 2: first-principles retrofit — data structures, functions"`

### Task 3: Modules/Packages + File/Exception Handling + OOP — `05-modules-packages`, `06-file-exception`, `07-oops`

Content: modules — the import system and namespace isolation problem it solves. File/exception — resource cleanup (why `finally`/`with` exist — from-scratch: a manual try/finally-based context manager before showing `contextlib`). OOP — encapsulation/inheritance/polymorphism with a real worked example showing WHY (not just what), composition-vs-inheritance tradeoff explicitly discussed (a genuine design decision, not just vocabulary).

- [ ] `git commit -m "Phase 7 Task 3: first-principles retrofit — modules, file/exception handling, OOP"`

### Task 4: Advanced Concepts + Logging + Multithreading — `08-advanced-concepts`, `09-logging`, `10-multithreading`

Content: iterators/generators — from-scratch a manual iterator protocol (`__iter__`/`__next__`) before `yield`, memory-efficiency argument (why generators beat building a full list) with a real measured memory/time comparison. Logging — why `print` debugging fails at scale (no levels, no structure, can't be turned off centrally). Multithreading — the GIL, race conditions (from-scratch: REPRODUCE a real race condition without a lock, actually run it enough times to show the failure, then fix with `threading.Lock`, show it's now correct) — this is the most valuable from-scratch demo in this task, don't skip it.

- [ ] `git commit -m "Phase 7 Task 4: first-principles retrofit — iterators/generators, logging, multithreading"`

### Task 5: Memory Management + Flask + Streamlit — `11-memory-management`, `12-flask`, `13-streamlit`

Content: memory management — reference counting + cycle-detecting GC, from-scratch a manual reference-count simulation showing why cycles need the cycle collector. Flask/Streamlit — these are "practical implementation" heavy topics with less from-scratch depth available (they ARE the practical tool) — for Flask, from-scratch = a minimal WSGI-adjacent request-handler using stdlib `http.server` before showing Flask's routing automates it (this connects nicely to `08-mlops-deployment/06-bentoml`'s from-scratch HTTP server — cite it). Streamlit — from-scratch = N/A/minimal, document why (it's a UI framework, the "from scratch" version would just be raw HTML/CSS which isn't the pedagogical point) — this is a legitimate judgment call, document it.

- [ ] `git commit -m "Phase 7 Task 5: first-principles retrofit — memory management, Flask, Streamlit"`. Also create `01-python-foundation/README.md` section index (all 13 topics) as part of this task since it's the last one for this section.

---

### Task 6: Statistics — `02-statistics/01-descriptive-statistics`, `02-probability`, `03-inferential-statistics`

**Preservation-critical** — real existing content, follow the checklist discipline from Phase 1's lesson exactly. Restructure into the 12-section template; the math is already there, add Problem/Why-simpler-fails framing, From-scratch implementations (e.g. computing variance/std from the definition in NumPy vs. `np.var`, a from-scratch bootstrap confidence interval, a from-scratch permutation test for the inferential-statistics topic), Experiment sections with real executed comparisons, Failure modes (p-hacking, misinterpreting confidence intervals, Simpson's paradox), Mental model, Questions. Update `02-statistics/README.md` per section (currently has no per-topic READMEs — check).

- [ ] `git commit -m "Phase 7 Task 6: first-principles retrofit — descriptive stats, probability, inferential statistics"`

---

### Task 7: NumPy + Pandas — `03-data-analysis/01-numpy`, `02-pandas`

Content: NumPy — why Python lists are slow for numeric work (per-element type overhead), broadcasting rules derived from shape arithmetic (not just stated), from-scratch = a manual element-wise loop timed against a vectorized NumPy op on the same data, real measured speedup. Pandas — `DataFrame` as a labeled, columnar extension of NumPy arrays; `groupby` as split-apply-combine, from-scratch = implement split-apply-combine manually with plain dicts/loops on a toy dataset, compare to `.groupby().agg()`'s real output on the same data (must match).

- [ ] `git commit -m "Phase 7 Task 7: first-principles retrofit — NumPy, Pandas"`

### Task 8: Data Manipulation + Reading + Matplotlib + Seaborn — `03-data-manipulation`, `04-data-reading`, `05-matplotlib`, `06-seaborn`

Content: data manipulation/reading — mostly practical-implementation-heavy, keep from-scratch sections proportionate (e.g. a from-scratch CSV parser demonstrating what `pd.read_csv` handles for you — delimiters, quoting, type inference). Matplotlib/Seaborn — the grammar-of-graphics-adjacent conceptual foundation (figure/axes hierarchy for Matplotlib, Seaborn as a statistical-plotting layer on top) — from-scratch = a manual bar-chart-from-rectangles demo using raw Matplotlib patches to show what `plt.bar` automates, connects the two topics.

- [ ] `git commit -m "Phase 7 Task 8: first-principles retrofit — data manipulation, data reading, matplotlib, seaborn"`

### Task 9: SQLite + EDA Projects — `07-sqlite`, `08-eda-projects`

Content: SQLite — why a flat file/CSV doesn't scale to concurrent/relational access, B-tree index intuition (why indexed lookup beats full scan — derive the log(n) argument), from-scratch = a manual linear-scan search timed against an indexed SQL query on the same data, real measured difference. EDA projects — these are practical capstones for the section; write a short notes.md tying the 3 existing project notebooks (wine quality, flight price, Google Play Store) back to the section's concepts as a synthesis, rather than a full 12-section treatment (document this as a deliberate scope choice — a synthesis topic, not a new concept).

- [ ] `git commit -m "Phase 7 Task 9: first-principles retrofit — SQLite, EDA projects"`. Also create/update `03-data-analysis/README.md` (all 8 topics) since this is the last task for this section.

---

### Task 10: Feature Engineering — folderize + retrofit all 4 topics

**Files:** First, `git mv` each flat notebook into a new topic folder: `01-missing-values.ipynb` → `01-missing-values/01-missing-values.ipynb`, same pattern for the other 3.

Content: missing values — MCAR/MAR/MNAR taxonomy (real conceptual distinction, not just imputation methods), from-scratch = manual mean/median imputation vs `sklearn`'s, real comparison; also KNN-imputation motivated by connecting to `05-machine-learning/09-knn`. Outliers — from-scratch = manual IQR/Z-score computation vs `sklearn`. Encoding — why raw categorical strings break most ML models (from-scratch = manual one-hot encoding with plain Python before `pd.get_dummies`/`OneHotEncoder`). Imbalanced data — why accuracy is misleading on imbalanced data (real worked example computing accuracy on a naive majority-class predictor to make the point concrete), SMOTE's actual synthesis mechanism explained (not just named) — connects to `05-machine-learning/09-knn`'s distance-based reasoning.

- [ ] Folderize (git mv, preserving history). Write all 4 notes.md + extend notebooks with from-scratch cells. Write section README (`04-feature-engineering/README.md`, all 4 topics) since none exists yet in per-topic form. `git commit -m "Phase 7 Task 10: first-principles retrofit — feature engineering (folderized + retrofitted)"`.

---

### Final task: section and root README sanity pass

- [ ] Confirm `01-python-foundation/README.md`, `02-statistics/README.md`, `03-data-analysis/README.md`, `04-feature-engineering/README.md` all correctly list their topics (should already be current from Tasks 5/6/9/10's own README updates — this is a final consistency check, not new work). Root `README.md`'s sections 01-04 already say ✅ Complete (they were marked complete before this phase even started, since notebooks existed) — verify the prose blurbs are still accurate given the added first-principles depth, update only if genuinely stale. `git commit` if anything changed.

## Verification

```bash
cd /home/yashwanth-aravind/ml-course/python-bootcamp
for s in 01-python-foundation 02-statistics 03-data-analysis 04-feature-engineering; do
  echo "=== $s ==="
  .venv/bin/python -c "
import pathlib
for t in sorted(pathlib.Path('$s').iterdir()):
    if t.is_dir(): print(' ', t.name, (t/'notes.md').exists(), (t/'README.md').exists())
"
done
```
