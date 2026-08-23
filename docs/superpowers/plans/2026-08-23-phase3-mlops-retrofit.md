# Phase 3: MLOps & Deployment First-Principles Build-Out Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Complete `08-mlops-deployment/` — currently 4 empty stub topics (`01-docker`, `02-git`, `05-mlflow-dagshub`, `06-bentoml`, all "🚧 Coming soon") plus 2 gaps in the numbering (`03`, `04`) — into a first-principles MLOps curriculum following the problem-first progression: *"I trained a model → how do I reproduce it? → package it? → know it works? → track experiments? → version data/models? → serve it? → monitor it? → update it safely?"*

**Architecture:** Unlike Phases 1-2 (which retrofitted existing rich `notes.md` content), this phase writes genuinely new content for most topics — there is nothing to preserve here except each topic's existing subtopic-name status table (which becomes the new `notes.md`'s status table, per the established template). Every topic still follows the 12-section template, but "From-scratch implementation" here means something different than in ML/DL: it means implementing the underlying *mechanism* by hand before reaching for the tool (e.g. a manual content-hash-based cache to motivate Docker layer caching, a tiny file-diff/pointer scheme to motivate Git's object model, a plain-JSON run-log to motivate MLflow's tracking store, a hand-rolled REST endpoint via `http.server`/Flask to motivate BentoML). 6 tasks: (1) Docker+Git (reproduce), (2) new Testing/CI + new Model packaging/versioning (quality gate + package), (3) MLflow/DagsHub incl. DVC (track+version), (4) BentoML (serve), (5) new CI/CD + new Monitoring (automate+observe), (6) section/root README finalization. Sequential dispatch per subagent-driven-development's rule.

**Tech Stack:** Docker (CLI + Dockerfile, no cloud registry required — local builds only, per the free/local constraint), Git, MLflow (local tracking server / SQLite backend, no hosted DagsHub account required for the from-scratch/practical parts — DagsHub referenced conceptually as the hosted option), DVC (local remote, no cloud storage required), BentoML, GitHub Actions (as the concrete CI/CD example, since this repo already lives on GitHub), a minimal monitoring approach using Python's `logging` + a simple drift-detection statistic (no dedicated monitoring service required). `.venv` (uv-managed) for anything Python; system-level tools (Docker, Git) documented as prerequisites the learner installs separately, same as any other systems topic.

**Spec:** `docs/superpowers/specs/2026-08-23-first-principles-curriculum-design.md` and `AGENTS.md` (binding template + quality bar) — read both before starting any task.

## Global Constraints

- Repo root: `/home/yashwanth-aravind/ml-course/python-bootcamp`. Python env: `.venv` (uv-managed) — `.venv/bin/python` / `.venv/bin/jupyter`.
- Every `notes.md` follows the 12-section template in `AGENTS.md` exactly, in order: Problem, Intuition, Why simpler approaches fail, Mathematical foundation, Algorithm, From-scratch implementation, Practical implementation, Experiment, Failure modes, Real-world usage, Mental model, Questions to think about. Problem / Why-simpler-fails / Mental model / Questions are never skipped. **Mathematical foundation is legitimately thin-to-absent for pure-systems topics (Docker, Git, CI/CD)** — where there's no real math, replace it with a "Conceptual foundation" treatment covering the underlying model (e.g. Git's content-addressable DAG, Docker's union filesystem/layer model) with precision but without forcing LaTeX where none is warranted; MLflow/DVC/monitoring topics DO have real math (versioning as a DAG, drift-detection statistics) and should use it. Use judgment per topic, document the choice in notes.md.
- Follow the MLOps progression narrative explicitly: each topic's "Why simpler approaches fail" section should name the specific problem the *previous* topic in the sequence leaves unsolved (Docker solves "reproduce on another machine," but not "reproduce the exact code state" → Git; Git solves code reproducibility, but not "did this experiment actually work, and can I compare it to the last one?" → testing/CI as a gate, then MLflow for tracking; tracking doesn't version the *data* → DVC; a tracked, versioned model still isn't something anyone else can *call* → BentoML for serving; a served model can still silently degrade → monitoring; and none of this happens automatically → CI/CD ties the whole pipeline together).
- Preserve each topic's existing subtopic list (from its stub README's status table) as the new `notes.md`'s status table — these subtopic names were already chosen deliberately, don't invent new ones arbitrarily, though minor renaming for accuracy is fine (document why in the report).
- From-scratch implementation: a small, real, runnable demonstration of the *underlying mechanism*, not the production tool itself — see per-task guidance below. Where genuinely nothing meaningful can be built from scratch for a pure-CLI-tool topic (e.g. `02-git`'s day-to-day commands), the from-scratch section instead builds a tiny toy version of the *underlying data structure* (a content-addressable blob store) rather than being skipped — document this choice.
- Practical implementation: real, runnable code/config using the actual tool (a real `Dockerfile` that builds, a real MLflow run that logs metrics, a real BentoML service file) — not just described in prose. Where a tool needs an external service (Docker daemon, a running MLflow server) that may not be available in the execution environment, the practical step still includes the real file/script; if it can be executed in this environment, execute it and show real output; if it genuinely cannot (e.g. no Docker daemon in the CI sandbox), the topic must say so explicitly and mark the artifact "written and reviewed, not executed in this environment" rather than silently claiming it ran — never fabricate output.
- Experiment section: hypothesis stated before result — for systems topics, this can be something like "hypothesis: rebuilding the Docker image after only changing application code will be faster than after changing a dependency, because of layer caching" with a real timed before/after.
- Failure modes: for MLOps topics this is one of the most important sections — training/serving skew, data leakage into the tracking store, container bloat, non-reproducible builds from unpinned base images, DVC remote misconfiguration, model staleness undetected by monitoring, etc.
- Every topic's own `README.md`: rewrite from the current "🚧 Coming soon" stub into the AGENTS.md orientation format once its `notes.md` exists, and flip status. Update `08-mlops-deployment/README.md` (section index) and root `README.md`'s row 08 + roadmap/heading only in the LAST task, once all topics are done.
- Notebook execution gate (Phase 1/2 lesson, carried forward): any notebook this phase creates and any topic's notes.md cites as evidence must have non-null `execution_count` on every code cell.
- Content-preservation discipline (Phase 1/2 lesson, carried forward): for topics with pre-existing stub content (the subtopic list in each README), enumerate every subtopic before writing, confirm each is actually covered in the new notes.md.
- Commit granularity: one commit per task.
- No unit-test framework applies. "Done" means: notes.md has all 12 sections (or the documented Math→Conceptual-foundation substitution) with real substantive content, every cited notebook/script executes cleanly where the environment allows, and every original subtopic is covered.

---

### Task 1: Docker + Git — `08-mlops-deployment/01-docker`, `08-mlops-deployment/02-git`

**Files:**
- Create: `08-mlops-deployment/01-docker/notes.md`, a real `Dockerfile` for a minimal ML app (e.g. containerize one of the repo's existing simple sklearn model-serving snippets), `docker-compose.yml` if the Docker Compose subtopic warrants a worked example
- Modify: `08-mlops-deployment/01-docker/README.md` (orientation format, flip status)
- Create: `08-mlops-deployment/02-git/notes.md`
- Modify: `08-mlops-deployment/02-git/README.md` (orientation format, flip status)

**Content requirements:**

- **`01-docker`**: Problem = "it works on my machine" — code that depends on the exact OS/library versions of the machine it was developed on. Why-simpler-fails = a written list of "install these exact versions" instructions rots the moment any dependency changes upstream, and doesn't capture OS-level dependencies at all. Conceptual foundation = union/layered filesystem model (an image is a stack of read-only layers, a container adds one writable layer on top), why this makes layer caching possible, image vs container distinction. Algorithm/Practical = a real, working `Dockerfile` (base image → copy requirements → install → copy code → entrypoint) that actually builds (`docker build`) if a Docker daemon is available in this environment — check first; if not available, write and review the Dockerfile carefully and state clearly in notes.md that it wasn't executed here (never fabricate build output). From-scratch = a tiny Python script implementing a content-hash-based build cache (hash each "layer" of a fake build script, skip re-running unchanged steps) to demonstrate exactly what Docker's layer cache is doing underneath, executed and timed. Experiment = the from-scratch cache demo timed with/without a change to an early "layer" vs a late one. Failure modes = unpinned base image tags causing non-reproducible builds, image bloat from not cleaning build artifacts, secrets baked into image layers. Real-world = containerized training jobs, containerized model-serving. Mental model, Questions.
- **`02-git`**: Problem = coordinating changes to code (and this course's own content) across time and collaborators without overwriting each other's work or losing history. Why-simpler-fails = numbered-copy files (`script_v2_final_FINAL.py`) don't merge, don't show who changed what/why, and don't let you cheaply branch to try something risky. Conceptual foundation = Git's object model as a content-addressable DAG (blobs/trees/commits, each addressed by the SHA-1/SHA-256 hash of its content — same idea as Docker's layer addressing in the prior topic, worth calling out explicitly) — this is real conceptual depth even without "math" in the calculus sense. From-scratch = a tiny Python script implementing a minimal content-addressable blob store (hash a string, store it under `.mini-git/objects/<hash>`, retrieve by hash) to demonstrate the core primitive Git commits are built from. Practical = real `git init`/`add`/`commit`/`log` walkthrough, then a real merge conflict deliberately created and resolved (two branches editing the same line, merge, show the conflict markers, resolve). Experiment = hypothesis about what a 3-way merge does with a genuine conflicting-edit scenario, actually run. Failure modes = force-pushing over shared history, committing secrets, merge conflicts from long-lived branches. Real-world, Mental model, Questions.

- [ ] **Step 1:** Read both stub READMEs' subtopic lists.
- [ ] **Step 2:** Write both `notes.md` per the content requirements.
- [ ] **Step 3:** Write the real Docker/Git artifacts (Dockerfile, from-scratch scripts) and execute what can be executed in this environment (check Docker daemon availability first with `docker info`; Git is always available).
- [ ] **Step 4:** Update both topic READMEs to orientation format + flipped status.
- [ ] **Step 5:** `git add` both topic folders, commit: `git commit -m "Phase 3 Task 1: first-principles build-out — Docker, Git"`.

---

### Task 2: Testing/CI + Model Packaging/Versioning (new topics, fill the `03`/`04` gap)

**Files:**
- Create: `08-mlops-deployment/03-testing-ci/` (README.md, notes.md, at least one real pytest-style test file demonstrating the concepts)
- Create: `08-mlops-deployment/04-model-packaging-versioning/` (README.md, notes.md, a small script demonstrating model serialization + a naive versioning scheme)

**Content requirements:**

- **`03-testing-ci`**: Problem = how do you know a change to a training pipeline or model didn't silently break something, before you ship it. Why-simpler-fails = "I ran it once and the numbers looked fine" doesn't catch regressions, doesn't run automatically, and doesn't scale past one person's memory. Conceptual foundation = the test pyramid adapted to ML (unit tests for data-transform functions, tests for model input/output shape and value-range invariants, a smoke test that the whole pipeline runs end-to-end on a tiny fixture dataset) — explicitly note what's genuinely different about testing ML code vs regular software (non-determinism, no single "correct" output, testing *properties* of a model rather than exact values). From-scratch/Practical = a real small `pytest` test file testing (a) a data-preprocessing function's output shape/dtype, (b) a trained toy model's prediction range is sane (e.g. probabilities sum to 1), (c) a regression test pinning a metric to not regress below a threshold on a fixed fixture — actually run with `.venv/bin/pytest` and show the real pass/fail output. Experiment = deliberately break one function, show the test catching it (hypothesis: the test suite fails; run it; confirm). Failure modes = flaky tests from unseeded randomness, tests that pin exact floating-point values instead of tolerances, no test for the untested "it trains" path. Real-world = CI pipelines gating merges. Mental model, Questions.
- **`04-model-packaging-versioning`**: Problem = a trained model is a large binary artifact (weights) plus metadata (architecture, preprocessing, library versions it needs) — none of which is a "file" Git handles well, and none of which is captured by an experiment-tracking log alone. Why-simpler-fails = committing model binaries directly to Git bloats the repo and Git's diffing/merging model doesn't apply to opaque binaries; a folder of `model_v1.pkl`, `model_v2_final.pkl` has the same problems as `02-git`'s motivating example, now for artifacts instead of code. Conceptual foundation = serialization formats (pickle/joblib vs framework-native `SavedModel`/`state_dict` vs interchange formats like ONNX) and their tradeoffs (portability, security of unpickling untrusted data, framework lock-in); model registries as "Git for models" — content-addressed or version-numbered storage with metadata. From-scratch = a tiny Python "model registry" — save a model with `joblib`, compute its content hash, write a JSON metadata sidecar (hash, training date, metric, library versions), register it in a simple append-only JSON log — actually implemented and run on one of this repo's existing trained models from Phase 1. Practical = mention MLflow's Model Registry as the production version of this exact idea (forward-pointer to `05-mlflow-dagshub`, don't duplicate its content). Experiment = save two versions of a toy model, show the registry log correctly distinguishing them by hash even if filenames collide. Failure modes = unpickling untrusted pickle files (security), no metadata capturing which code/data produced a model (irreproducibility), silent format incompatibility across library versions. Real-world, Mental model, Questions.

- [ ] **Step 1:** Confirm neither folder currently exists (`08-mlops-deployment/03-testing-ci`, `08-mlops-deployment/04-model-packaging-versioning`) — create fresh.
- [ ] **Step 2:** Write both `notes.md` per the content requirements.
- [ ] **Step 3:** Write and execute the real pytest file and the real from-scratch model-registry script.
- [ ] **Step 4:** Write both topic READMEs in orientation format with ✅ Complete status.
- [ ] **Step 5:** `git add` both new topic folders, commit: `git commit -m "Phase 3 Task 2: first-principles build-out — testing/CI, model packaging & versioning"`.

---

### Task 3: MLflow & DagsHub (incl. DVC) — `08-mlops-deployment/05-mlflow-dagshub`

**Files:**
- Create: `08-mlops-deployment/05-mlflow-dagshub/notes.md`, a real Python script/notebook logging actual runs to a local MLflow tracking store, a small DVC-tracked toy dataset demo
- Modify: `08-mlops-deployment/05-mlflow-dagshub/README.md`

**Content requirements:**

Problem = once you're iterating on models (Task 2's registry helps with the *final* artifact, but says nothing about the 50 experiments that led to it), how do you know which hyperparameters/data/code produced which result, and compare runs systematically. Why-simpler-fails = a spreadsheet of "run 1: lr=0.01, acc=0.82" manually copy-pasted doesn't scale, has no link back to the actual code/data version, and can't be queried. Conceptual foundation = experiment tracking as a structured, queryable log (params, metrics, artifacts, code version, all keyed by a run ID) — this is the same "why manual bookkeeping fails at scale" argument as `02-git` and `04`'s registry, now applied to experiments; DVC's data-versioning model (content-addressed data files, a small `.dvc` pointer file committed to Git instead of the large data file itself — directly reuses Task 1's Git object-model concept and Task 4's content-addressing idea). From-scratch = a plain-JSON-file run logger (append a dict of params/metrics/timestamp per run to a JSONL file) to demonstrate the core idea before showing MLflow's richer version. Practical = a REAL Python script using `mlflow` to log several actual training runs (reuse an existing simple model from Phase 1, e.g. logistic regression with varying `C`) to a local `mlflow.db` SQLite tracking store, executed, with real logged metrics — and a small DVC walkthrough (`dvc init`, `dvc add` on a toy CSV, show the generated `.dvc` pointer file) if `dvc` is installed/available; if not available, document the exact commands and expected output without fabricating execution. Experiment = compare several real logged runs, hypothesis about which hyperparameter wins, confirmed against actual MLflow-logged results. Failure modes = tracking store bloat from logging every trivial run, not versioning the data alongside the code+params (irreproducible experiments), DagsHub/remote credentials in logged artifacts. Real-world, Mental model, Questions.

- [ ] **Step 1:** Check whether `mlflow` and `dvc` are installed in `.venv`; if not, add `mlflow` to `pyproject.toml` dependencies via `uv add mlflow` (DVC is a separate CLI tool, not a Python-env dependency in the same way — check availability with `which dvc`).
- [ ] **Step 2:** Write `notes.md` per the content requirements.
- [ ] **Step 3:** Write and execute the from-scratch JSON run-logger and the real MLflow tracking script (multiple real runs, real local tracking store). Do the DVC walkthrough if available; document-only if not.
- [ ] **Step 4:** Update the topic README.
- [ ] **Step 5:** `git add` the topic folder (and `pyproject.toml`/`uv.lock` if `mlflow` was added), commit: `git commit -m "Phase 3 Task 3: first-principles build-out — MLflow, DagsHub, DVC"`.

---

### Task 4: BentoML — `08-mlops-deployment/06-bentoml`

**Files:**
- Create: `08-mlops-deployment/06-bentoml/notes.md`, a real BentoML service file wrapping a trained model, a from-scratch minimal HTTP serving script
- Modify: `08-mlops-deployment/06-bentoml/README.md`

**Content requirements:**

Problem = a trained, versioned, tracked model (Tasks 2-3) still isn't something another program or person can *use* — it's a file on disk. Why-simpler-fails = "just import the model and call `.predict()` in whatever script needs it" doesn't work across languages/processes, has no concurrency handling, no input validation, no versioned API contract. Conceptual foundation = model serving as wrapping inference behind a network API (request→preprocess→predict→postprocess→response), the batching/concurrency considerations that differ from a one-off script call. From-scratch = a minimal HTTP model-serving endpoint using Python's stdlib `http.server` (or plain Flask, already a repo dependency) — load a pickled model, accept a JSON POST, run `.predict()`, return JSON — actually implemented, actually run, actually hit with a real request (e.g. via `requests` or `curl` in the same script/notebook) showing a real response. Practical = a real BentoML `service.py` file (`bentoml.Service`, an API endpoint, `@bentoml.api` decorator) wrapping the same or a similar model — write it correctly even if `bentoml serve` itself isn't run in this environment (document that clearly if so); if BentoML is installed and can run headless enough to demonstrate, show real output. Experiment = load-test the from-scratch endpoint with a handful of sequential requests, measure latency, hypothesis about per-request overhead. Failure modes = training/serving skew (preprocessing done differently at train vs serve time — this is one of the most important ML-specific failure modes, give it real weight), no input validation letting malformed requests crash the service, cold-start latency. Real-world, Mental model, Questions.

- [ ] **Step 1:** Check whether `bentoml` is installed; if not, add via `uv add bentoml` if it installs cleanly (it's a heavier dependency — if installation fails/is impractical in this environment, note this in the report and proceed with the from-scratch section plus a carefully-written-but-unexecuted BentoML service file).
- [ ] **Step 2:** Write `notes.md` per the content requirements.
- [ ] **Step 3:** Write and execute the from-scratch HTTP serving demo (including a real client request/response). Write the BentoML service file; execute if feasible.
- [ ] **Step 4:** Update the topic README.
- [ ] **Step 5:** `git add` the topic folder (and dependency files if changed), commit: `git commit -m "Phase 3 Task 4: first-principles build-out — BentoML model serving"`.

---

### Task 5: CI/CD + Monitoring (new topics, extend the progression past `06`)

**Files:**
- Create: `08-mlops-deployment/07-cicd/` (README.md, notes.md, a real `.github/workflows/*.yml` example — can reference/reuse this very repo's own CI needs, e.g. "lint the notebooks" or "run the Task 2 pytest suite on push")
- Create: `08-mlops-deployment/08-monitoring/` (README.md, notes.md, a from-scratch drift-detection script)

**Content requirements:**

- **`07-cicd`**: Problem = Tasks 1-4 gave you the pieces (containerize, version, track, test, serve) but none of it runs automatically — a human has to remember to do each step, in order, every time. Why-simpler-fails = a manual checklist ("1. run tests, 2. build image, 3. push, 4. deploy") gets skipped under time pressure and doesn't block a broken change from shipping. Conceptual foundation = a CI/CD pipeline as a DAG of automated stages triggered by a code-repository event (a `02-git` push/PR), each stage gating the next — explicitly ties together `03-testing-ci`'s tests as the first real gate. Practical = a real, valid GitHub Actions workflow YAML file (this repo is already on GitHub) that runs `03-testing-ci`'s pytest suite on push/PR — write it correctly, and if feasible, verify its YAML syntax is valid (`yamllint` or a Python YAML parse) even though the workflow itself only truly "runs" once pushed to GitHub (document this honestly). From-scratch = a tiny bash/Python script that manually chains "run tests → if pass, build → if pass, print 'would deploy'" to demonstrate the DAG-of-gated-stages idea before showing the YAML version. Experiment = run the from-scratch pipeline script twice — once with passing tests, once with a deliberately broken test — showing the pipeline halts on failure. Failure modes = a CI pipeline that's slow enough people bypass it, secrets in workflow files, no rollback strategy on a bad deploy. Real-world, Mental model, Questions.
- **`08-monitoring`**: Problem = a model that was accurate at training/serving time can silently degrade as the real-world data distribution shifts — nothing in Tasks 1-7 detects this after deployment. Why-simpler-fails = "check accuracy occasionally by hand" doesn't scale and usually means you find out from a downstream complaint, not the monitoring system. Conceptual foundation/Math = distribution-shift detection — a real statistical treatment of at least one drift metric (e.g. population stability index or a KS-test comparing a reference distribution to a live-window distribution for one feature), derived and explained, not just named. From-scratch = a real NumPy/SciPy implementation of the chosen drift statistic, run on a toy "reference" vs "shifted" synthetic dataset (fixed seed) to show it correctly flags shift when injected and stays quiet when there's none — a genuine hypothesis-driven experiment. Practical = connect to what a production system does with this signal (alerting, automatic retraining triggers) in prose, referencing real tools (Evidently, WhyLabs) by name without needing to install them. Experiment = the from-scratch drift-statistic demo above, hypothesis stated first (the shifted distribution should trigger the threshold, the unshifted one should not). Failure modes = alert fatigue from too-sensitive thresholds, monitoring the wrong thing (input drift without checking output/label drift), no monitoring of the monitoring system itself. Real-world, Mental model, Questions.

- [ ] **Step 1:** Confirm neither folder exists — create fresh.
- [ ] **Step 2:** Write both `notes.md` per the content requirements.
- [ ] **Step 3:** Write and execute the from-scratch pipeline-gating script and the from-scratch drift-detection script (both should actually run and produce real output). Write the GitHub Actions YAML (valid syntax, referencing this repo's actual `03-testing-ci` test path).
- [ ] **Step 4:** Write both topic READMEs in orientation format with ✅ Complete status.
- [ ] **Step 5:** `git add` both new topic folders, commit: `git commit -m "Phase 3 Task 5: first-principles build-out — CI/CD, monitoring"`.

---

### Task 6: Section and root README finalization

**Files:**
- Modify: `08-mlops-deployment/README.md` (section index — currently lists only `01,02,05,06`; add all 8 topics `01`-`08` in order, flip section status)
- Modify: root `README.md` (row 08 in the Curriculum table: `🚧 Coming soon` → `✅ Complete`; Course Roadmap diagram if it references section 08's status; `### 08 –` heading: drop `*(coming soon)*`)

**Content requirements:** Purely mechanical — no new conceptual content. Read the current `08-mlops-deployment/README.md` and root `README.md` first. This mirrors exactly what the earlier NLP-completion plan's Task 3 Step 5 did for sections 06/07 — follow that precedent.

- [ ] **Step 1:** Read both files' current state.
- [ ] **Step 2:** Update `08-mlops-deployment/README.md`'s topic table to include all 8 topics with correct descriptions and ✅ Complete status.
- [ ] **Step 3:** Update root `README.md` per the content requirements.
- [ ] **Step 4:** `git add` both files, commit: `git commit -m "Phase 3 Task 6: mark 08-mlops-deployment complete in section and root README"`.

---

## Verification (after all 6 tasks)

```bash
.venv/bin/python - <<'EOF'
import pathlib
for topic in sorted(pathlib.Path("08-mlops-deployment").iterdir()):
    if not topic.is_dir(): continue
    nm = topic / "notes.md"
    rm = topic / "README.md"
    print(topic.name, "notes.md" if nm.exists() else "MISSING notes.md", "README.md" if rm.exists() else "MISSING README.md")
EOF
grep -c "Coming soon" 08-mlops-deployment/README.md README.md 08-mlops-deployment/*/README.md
```

Expect all 8 topics present with both files, and the `grep -c "Coming soon"` count to be 0 everywhere in scope.
