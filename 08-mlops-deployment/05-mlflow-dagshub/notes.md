# MLflow, DagsHub & DVC — experiment tracking and data versioning

## Problem

`04-model-packaging-versioning` solved a real problem: given a trained
model, uniquely identify it by content hash, attach metadata, and never
lose track of which artifact is which. But that registry only ever sees
the *winning* model — the one you actually decided to call `register()`
on. In practice, getting to that one model took dozens of experiments:
different learning rates, different regularization strengths, different
feature sets, different random seeds, most of them failures or
near-misses. **How do you know which combination of hyperparameters, data
version, and code produced a given result — for all 50 experiments, not
just the one that won — and how do you compare them systematically instead
of relying on memory or scattered print statements?** A second, related
problem shows up the moment training data itself changes: a dataset is
often gigabytes of binary rows, not something Git's line-based diffing was
built for (`02-git`'s "Why simpler approaches fail" made exactly this
argument about model binaries; the same argument applies to data). **How do
you version a large dataset the same rigorous way Git versions source
code, without bloating the Git repository itself?**

## Intuition

Picture a week of experimentation on a classifier, tracked the way most
people start out: a spreadsheet, or worse, a scroll of terminal output.

```
run 1: lr=0.01, epochs=10, acc=0.82
run 2: lr=0.001, epochs=10, acc=0.79
run 3: lr=0.01, epochs=20, acc=0.85  <- best so far, used the new cleaned dataset
run 4: lr=0.01, epochs=20, acc=0.71  <- wait, was this the old dataset or new?
```

By run 4, the record has already broken down — nobody wrote down *which
version of the data* run 4 used, so the 0.71 number is unexplained and the
whole log is now suspect. Nothing here is queryable ("show me every run
with `lr=0.01` and the cleaned dataset") without manually re-reading every
line, and nothing links a row in the spreadsheet back to a specific,
verifiable file on disk. Two things fix this, and they attack different
halves of the problem:

1. **Experiment tracking** replaces the spreadsheet with a structured
   record — one entry per run, with params/metrics/timestamp captured
   automatically at training time, addressable by a generated run ID and
   queryable afterward. This is `05-mlflow-dagshub`'s first idea.
2. **Data versioning (DVC)** replaces "was this the old dataset or new?"
   with the same content-addressing idea `02-git` used for code and `04`
   used for model binaries, applied to the data files themselves — so a
   run's metadata can record *exactly* which data version produced it, and
   that claim is independently verifiable, not just remembered.

## Why simpler approaches fail

**Experiment tracking, the manual way:** a spreadsheet (or a text file) of
"run N: params, metric" copy-pasted by hand after each experiment fails
for structural reasons, not just tedium:

- **It doesn't scale.** Ten runs, fine. A hyperparameter sweep of 200 runs
  across three model families, and manual transcription either stops
  happening (so recent runs are simply missing) or starts happening
  wrong (transposed digits, a metric pasted into the wrong row).
- **It has no link back to code or data version.** A row that says
  "acc=0.85" is a number floating in space unless something also records
  the exact Git commit of the training script and the exact version of the
  training data at that moment — and a human typing a spreadsheet row
  essentially never remembers to capture both, every time, under deadline
  pressure.
- **It can't be queried.** "Show me every run where `C < 1` and accuracy
  beat the current production model" means a human visually scanning rows,
  which gets slower and more error-prone as the log grows — exactly the
  same "linear scan by a human instead of an indexed lookup" problem that
  motivated `04`'s move from filenames to content hashes.

**Data versioning, the naive way:** either committing a large dataset file
directly to Git, or not versioning it at all and just overwriting
`train.csv` in place whenever the data changes. Both fail:

- **Committing the raw file to Git repeats `04`'s core problem one level
  up.** Git's diffing and delta-compression are built for text; every new
  version of a large binary/CSV data dump gets stored close to its full
  size in Git's history, forever (Git objects, once pushed, are never
  cleanly removed without a history rewrite) — precisely the argument
  `04`'s "Why simpler approaches fail" made about model binaries, now
  applied to training data.
- **Overwriting the file in place with no versioning at all** means a run
  logged last Tuesday against `train.csv` is *silently* unreproducible the
  moment `train.csv` changes under it — there is no way to get back the
  exact bytes that run actually trained on, and no record that anything
  changed at all.

## Conceptual foundation

*(This section is titled "Conceptual foundation" rather than
"Mathematical foundation," following the same documented substitution
`02-git` and `04-model-packaging-versioning` use: experiment tracking and
data versioning have no closed-form math to derive — the foundational
ideas here are conceptual (a structured queryable log; content-addressed
data storage), not mathematical.)*

### Experiment tracking as a structured, queryable log

The core idea is simple to state and easy to underestimate: **an
experiment-tracking system is a structured log, keyed by a generated run
ID, where every entry records the parameters that went in, the metrics
that came out, any produced artifacts, and (ideally) the exact code
version — captured automatically at training time, not typed by hand
afterward.** This is the exact same "stop trusting human bookkeeping, make
the record structural" move `02-git` made for code history and `04` made
for model identity, now applied to the process of experimentation itself
rather than to any single artifact. A run ID is to an experiment what a
content hash is to a model file in `04`: an identity assigned by the
system, not a name a human chose, so nothing can be silently overwritten
or mislabeled.

### DVC's data-versioning model: content-addressing, reused

DVC (Data Version Control) applies **the same content-addressed storage
idea `02-git`'s object model and `04`'s model registry already
established**, to data files:

- The large data file itself (a CSV, a directory of images) is moved into
  DVC's own content-addressed cache, stored under a path derived from its
  hash — not committed to Git at all.
- In its place, Git tracks a tiny **`.dvc` pointer file** — plain YAML,
  a few hundred bytes regardless of how large the actual data is —
  recording the data file's hash, size, and path. Git's line-based diffing
  works perfectly well on this small text pointer even though it says
  nothing sensible about the binary content it points to.
- Checking out an old Git commit checks out the old `.dvc` pointer (a
  cheap, ordinary Git operation); a separate `dvc checkout` step then uses
  that pointer's hash to pull the *matching* version of the actual data
  from DVC's cache (or a remote like DagsHub/S3) into the working
  directory.

This is precisely `02-git`'s blob/tree/commit model and `04`'s
`register()` content-addressing step (see
[`../02-git/notes.md`](../02-git/notes.md) and
[`../04-model-packaging-versioning/notes.md`](../04-model-packaging-versioning/notes.md)
for the two prior derivations of this idea) — a two-part system where
identity is derived from content, and a small, cheap, human-readable
pointer stands in for a large, opaque blob wherever line-based tools need
to track "what changed." DagsHub is, for the purposes of this topic, an
MLflow-tracking-server-plus-DVC-remote-storage hosting service — it
doesn't introduce a new idea, it hosts the same two systems (a queryable
run log, a content-addressed data store) so a team can share them instead
of each person running a local `mlflow.db` and a local DVC cache.

## Algorithm

**Experiment tracking, per run:**
1. Start a run: generate (or receive) a unique run ID.
2. Log parameters (hyperparameters, dataset identifier, code
   version/commit) under that run ID.
3. Train, evaluating whatever metrics matter.
4. Log metrics (and, optionally, artifacts — a plot, a serialized model)
   under the same run ID.
5. End the run: the record is now immutable and queryable by run ID or by
   filtering on any logged param/metric.

**DVC, per data file:**
1. `dvc init` — sets up DVC's metadata directory (`.dvc/`) inside a Git
   repository, analogous to `git init` but for data.
2. `dvc add <file>` — hashes the file's content, moves it into DVC's
   content-addressed cache (`.dvc/cache/`, keyed by hash), replaces the
   original path with a lightweight checked-out copy (or symlink) from the
   cache, and writes a `<file>.dvc` pointer file recording the hash.
3. `git add <file>.dvc .gitignore` then `git commit` — Git now tracks the
   tiny pointer (and a `.gitignore` entry DVC adds so Git never
   accidentally tracks the real data file directly); the actual data
   content lives in DVC's cache, optionally pushed to a remote
   (`dvc push`, e.g. to DagsHub) the same way `git push` shares commits.

## From-scratch implementation

`json_run_logger.py` (this directory) — the entire "tracking store" is one
function: hash nothing, just append one JSON object per run to a JSONL
file, and a linear-scan query function to filter it back out. This makes
the core idea of experiment tracking (a structured, run-ID-keyed record)
visible with zero infrastructure, before MLflow's real client/server/DB
version below:

```python
def log_run(params: dict, metrics: dict) -> str:
    run_id = uuid.uuid4().hex[:12]
    record = {
        "run_id": run_id,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "params": params,
        "metrics": metrics,
    }
    with open(LOG_PATH, "a") as f:
        f.write(json.dumps(record) + "\n")
    return run_id


def query_runs(min_accuracy: float | None = None) -> list[dict]:
    runs = []
    with open(LOG_PATH) as f:
        for line in f:
            record = json.loads(line)
            if min_accuracy is None or record["metrics"].get("accuracy", 0) >= min_accuracy:
                runs.append(record)
    return runs
```

Actually executed (`.venv/bin/python json_run_logger.py`, three real
`LogisticRegression` fits on the breast-cancer dataset with varying `C`,
scaled features, held-out test split) — real captured output:

```
=== Logging 3 real training runs to a plain JSONL file ===
logged run dad556024802: C=0.01 accuracy=0.9649
logged run 4c05a84bc411: C=1.0 accuracy=0.9737
logged run 3b6bb1a1fc18: C=100.0 accuracy=0.9386

=== Raw contents of runs.jsonl ===
{"run_id": "dad556024802", "timestamp": "2026-08-23T18:20:09Z", "params": {"C": 0.01, "model": "LogisticRegression"}, "metrics": {"accuracy": 0.9649122807017544}}
{"run_id": "4c05a84bc411", "timestamp": "2026-08-23T18:20:09Z", "params": {"C": 1.0, "model": "LogisticRegression"}, "metrics": {"accuracy": 0.9736842105263158}}
{"run_id": "3b6bb1a1fc18", "timestamp": "2026-08-23T18:20:09Z", "params": {"C": 100.0, "model": "LogisticRegression"}, "metrics": {"accuracy": 0.9385964912280702}}

=== Querying runs with accuracy >= 0.97 ===
{'run_id': '4c05a84bc411', 'timestamp': '2026-08-23T18:20:09Z', 'params': {'C': 1.0, 'model': 'LogisticRegression'}, 'metrics': {'accuracy': 0.9736842105263158}}
```

This already delivers the core benefit over a spreadsheet — every run has
a unique, system-generated ID, the record is structured (not free text),
and it's programmatically queryable (`query_runs`) — but notice what it
still *doesn't* do: no code-version capture, no artifact storage, no
concurrent-writer safety, no UI, and no comparison view across runs beyond
what you write ad hoc in Python. That gap is exactly what MLflow's real
client fills.

## Practical implementation

`mlflow_tracking_demo.py` (this directory) uses the real `mlflow` Python
client against a local SQLite tracking store (`sqlite:///mlflow.db`,
created in this directory), logging real training runs of the same model
family (`LogisticRegression`, breast-cancer dataset, scaled features) that
`json_run_logger.py` used — so the two can be compared directly. Each
`mlflow.start_run()` block maps one-to-one onto `json_run_logger.py`'s
`log_run()` call, except the run ID, timestamp, and storage are now
handled by MLflow's client and a real relational database instead of by
hand:

```python
mlflow.set_tracking_uri(f"sqlite:///{HERE / 'mlflow.db'}")
mlflow.set_experiment("breast-cancer-logistic-regression")

for C in (0.001, 0.01, 0.1, 1.0, 10.0, 100.0):
    with mlflow.start_run(run_name=f"logreg-C={C}") as run:
        model = LogisticRegression(C=C, max_iter=2000)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        mlflow.log_param("C", C)
        mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
        mlflow.log_metric("f1_score", f1_score(y_test, preds))
```

Actually executed (`.venv/bin/python mlflow_tracking_demo.py`, from this
directory) — real captured output:

```
2026/08/23 18:20:10 INFO mlflow.store.db.utils: Creating initial MLflow database tables...
2026/08/23 18:20:10 INFO mlflow.store.db.utils: Updating database tables
2026/08/23 18:20:13 INFO mlflow.tracking.fluent: Experiment with name 'breast-cancer-logistic-regression' does not exist. Creating a new experiment.
=== Logging real training runs to MLflow (sqlite:///mlflow.db) ===
run_id=94d6a9cee1c44217a2b27350b8e0b9c1 C=0.001    accuracy=0.8860 f1=0.9161
run_id=28769711c4b5480bbbdc5159b28e31f3 C=0.01     accuracy=0.9649 f1=0.9726
run_id=df98ec6d46784ae7bbfff41a6d443406 C=0.1      accuracy=0.9825 f1=0.9861
run_id=a4c6cce3f8454ffcb386dc42c2ea2744 C=1.0      accuracy=0.9737 f1=0.9790
run_id=e3110404827b4040a7d12a8317616b12 C=10.0     accuracy=0.9737 f1=0.9787
run_id=8a4c799b38e94d7680012f50ed702a2c C=100.0    accuracy=0.9386 f1=0.9489

=== Querying logged runs back via MlflowClient ===
run_id                            C         accuracy    f1_score
df98ec6d46784ae7bbfff41a6d443406  0.1       0.9825      0.9861
e3110404827b4040a7d12a8317616b12  10.0      0.9737      0.9787
a4c6cce3f8454ffcb386dc42c2ea2744  1.0       0.9737      0.9790
28769711c4b5480bbbdc5159b28e31f3  0.01      0.9649      0.9726
8a4c799b38e94d7680012f50ed702a2c  100.0     0.9386      0.9489
94d6a9cee1c44217a2b27350b8e0b9c1  0.001     0.8860      0.9161

Best run by accuracy: C=0.1 accuracy=0.9825
```

Mapping back to the from-scratch version: `mlflow.start_run()` /
`log_param` / `log_metric` are the richer equivalents of
`json_run_logger.py`'s `log_run()`; `MlflowClient().search_runs(...,
order_by=[...])` is the richer equivalent of `query_runs()` — but now
backed by a real SQL database (queryable with arbitrary filters, not just
the one condition hand-coded into `query_runs`), with a run ID assigned
and namespaced under an experiment automatically, and (not exercised
here, since no artifacts were logged) support for storing model files,
plots, and other run outputs alongside the metrics.

**DVC and DagsHub — written and reviewed, not executed in this
environment** (the `dvc` CLI is not installed here; the commands and
expected output below are documented precisely, not fabricated by
actually running them). Given a toy dataset committed at
`data/toy_dataset.csv` in this directory (5 rows, 86 bytes, MD5
`e0e86c56618fbd0e45c2554726c806c2`):

```
$ dvc init
Initialized DVC repository.

$ dvc add data/toy_dataset.csv
To track the changes with git, run:

        git add data/.gitignore data/toy_dataset.csv.dvc

$ cat data/toy_dataset.csv.dvc
outs:
- md5: e0e86c56618fbd0e45c2554726c806c2
  size: 86
  hash: md5
  path: toy_dataset.csv

$ git add data/.gitignore data/toy_dataset.csv.dvc
$ git commit -m "Track toy_dataset.csv with DVC"
```

Note what actually got committed to Git: not the 86-byte CSV's *content*
directly (DVC's own `.gitignore` entry excludes it), but the 4-line
`.dvc` pointer file — the same trade Git itself makes for large binaries
in `04`'s "Why simpler approaches fail," just implemented by DVC's own
cache instead of Git's object store. Pushing that cached content to a
shared remote — `dvc remote add origin https://dagshub.com/<user>/<repo>.dvc
&& dvc push` — is DagsHub's role here: hosting both a DVC remote (for the
data) and an MLflow tracking server (for the runs above), so a team shares
one source of truth for both instead of each person running local-only
`mlflow.db` and DVC caches.

## Experiment

**Hypothesis:** for `LogisticRegression` on the breast-cancer dataset with
standardized features, extremely small `C` (heavy regularization) should
underfit, extremely large `C` (weak regularization) should start
overfitting the training split, and a **middling `C` value should win** —
neither extreme should be best. Concretely: predict the winner will be
somewhere in the `0.1`–`10` range, not at either end (`0.001` or `100`).

**Setup:** `mlflow_tracking_demo.py`, exactly as shown above — six real
runs, `C ∈ {0.001, 0.01, 0.1, 1.0, 10.0, 100.0}`, same train/test split
(`random_state=42`) and same standardized features across all six runs, so
`C` is the only thing that varies.

**Actual result** (from the real logged MLflow runs above, sorted by
accuracy): `C=0.1` won outright at **0.9825** accuracy / 0.9861 F1. The
two extremes were the two worst runs — `C=0.001` (heaviest regularization)
scored lowest at 0.8860, and `C=100.0` (weakest regularization) was
second-worst at 0.9386. The middle of the range (`0.1`, `1.0`, `10.0`) all
scored ≥0.9737, clearly separated from both extremes.

**Interpretation:** confirmed. `C=0.001` underfits — regularization this
strong shrinks the decision boundary's coefficients so aggressively the
model can't separate the classes well even on this fairly linearly
separable dataset. `C=100.0` lets the model fit training-set noise more
than it should, and unlike a huge, high-dimensional dataset where this
would show up as a large train/test accuracy gap, here it shows up
directly as worse held-out accuracy. The winner (`C=0.1`) sits in between,
which matches the hypothesis's shape even though the exact winning value
(`0.1` rather than, say, `1.0`) wasn't predicted more precisely than "the
middle of the range" — a genuinely blind pick between `0.1` and `1.0`
would have had even odds, and this is exactly the kind of question logged,
queryable MLflow runs answer directly (`search_runs(order_by=[...])`)
instead of by memory.

**Limitations:** six runs on one fixed train/test split is enough to
demonstrate the *tracking mechanism* and get a plausible, reproducible
answer for this dataset, not a rigorous hyperparameter search — a proper
comparison would use cross-validation (to know whether `0.1` beating `1.0`
by 0.0088 accuracy is a real effect or split-specific noise) and a finer
grid around the winning region.

## Failure modes

- **Tracking-store bloat from logging every trivial run.** MLflow makes
  logging so cheap that it's tempting to log every debug run, every typo'd
  hyperparameter, every accidental re-run — and unlike the from-scratch
  JSONL logger (where bloat is at least visible as a growing text file),
  a SQLite or remote-DB-backed tracking store can silently accumulate
  thousands of near-duplicate, low-value runs that make the useful signal
  (the handful of runs that actually mattered) harder to find in the UI or
  in `search_runs()` results. Treat experiment names and run tagging as a
  real hygiene practice, not an afterthought — and periodically archive or
  delete runs that added no information.
- **Not versioning the data alongside the code and params means
  irreproducible experiments.** `mlflow_tracking_demo.py` above logs `C`,
  `model_type`, and `max_iter`, but not a hash of the exact training data
  used — this is precisely the gap DVC closes. A logged run that says
  "accuracy=0.9825" without a data version pointer is only as trustworthy
  as the assumption that `data/toy_dataset.csv` (or whatever dataset a
  real project uses) never silently changed between then and now; without
  a `.dvc` pointer's hash checked at training time, that assumption is
  unverifiable months later, exactly the "run 4: wait, which dataset was
  this?" problem from the Intuition section.
- **Credentials/secrets in logged artifacts.** MLflow's `log_artifact`
  can capture entire files — a training script, a config file, a notebook
  — and it is easy to accidentally log a file that embeds an API key,
  database connection string, or cloud credential (e.g. a `.env` file
  swept up by a glob, or a config dict logged via `log_dict` that includes
  a secret field). Once logged to a shared tracking server (DagsHub or
  otherwise), that artifact is visible to everyone with read access to the
  experiment, and — exactly like `02-git`'s "committing secrets" failure
  mode — deleting it later doesn't guarantee the secret wasn't already
  seen, copied, or cached; it must be treated as compromised and rotated.

## Real-world usage

- Any team running more than a handful of training experiments needs a
  systematic way to answer "which run produced this number, with what
  configuration, on what data" — that's the direct real-world use of
  experiment tracking, at the scale of hundreds or thousands of runs
  rather than the six run in this topic's experiment.
- MLflow's Model Registry (built on top of the same tracking store this
  topic covers) is the production version of `04`'s from-scratch
  registry — see `04-model-packaging-versioning`'s "Practical
  implementation" for that explicit mapping; a registered model there
  links back to the exact run (and, transitively, the exact params/data)
  that produced it.
- DVC pipelines (`dvc.yaml`, not covered in depth here) extend the same
  content-addressing idea to entire multi-stage data pipelines — "rerun
  only the stages whose inputs actually changed," the same principle
  `make` uses for build systems, applied to data processing.
- DagsHub (or a self-hosted MLflow server + object storage) is what lets a
  team share one tracking store and one DVC remote instead of everyone's
  experiments and data versions living only on their own laptop.

## Mental model

**A spreadsheet of runs is a label a human typed; an MLflow run ID is an
identity the system assigned. A `.dvc` pointer file is to a dataset what a
content hash was to a model file in `04` — a tiny, Git-friendly stand-in
for a large blob, so "which exact data produced this number" is answerable
by hash comparison, not memory.**

## Questions to think about

1. `mlflow_tracking_demo.py`'s runs logged `C`, `model_type`, and
   `max_iter`, but no dataset version identifier. Sketch how you'd extend
   it to log a DVC data hash (or, absent DVC, a hash of the training data
   file computed with `04`'s `hash_file` function) as an MLflow param on
   every run — what would that buy you six months from now that today's
   version doesn't?
2. `query_runs()` in the from-scratch logger does a full linear scan of
   the JSONL file on every call. At what rough scale (hundreds of runs?
   millions?) does that stop being fine, and what does MLflow's SQLite/
   Postgres-backed store give you instead that makes the same query fast
   at that scale?
3. The Experiment section found `C=0.1` won by a margin of under 1% over
   `C=1.0` and `C=10.0` on a single train/test split. If you only had the
   from-scratch JSONL logger (no MLflow, no built-in comparison view), what
   would you need to add to `query_runs()` to answer "is this difference
   likely real, or within noise" — and why can't the six logged accuracy
   numbers alone answer that?
4. DVC's `.dvc` pointer file records an MD5 hash, size, and path — no
   author, no message, no parent pointer the way a Git commit has. What
   information does a `.dvc` file *not* capture that a full Git commit of
   the same file would, and why does DVC rely on being paired with a
   Git commit (rather than standing alone) to make up that gap?
5. This topic's Failure modes section warns about tracking-store bloat
   from logging every trivial run. Propose a concrete policy (e.g. what to
   log automatically vs. only on request, what to prune and when) that
   balances "capture everything, in case it matters later" against that
   bloat risk for a team running automated hyperparameter sweeps of
   hundreds of runs per day.
