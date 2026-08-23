# Model packaging & versioning

## Problem

A trained model is not "a file" the way source code is. It's a large
binary artifact (the learned weights) bundled with metadata that has to
travel with it to be usable at all: the exact preprocessing it expects, the
architecture/hyperparameters that produced it, and the library versions it
was serialized with. Git handles neither part well — it diffs and merges
text, not opaque binary blobs, and it has no concept of "this binary was
produced by this code, on this data, with this metric." An experiment
log (Phase 1/`05-machine-learning` notebooks, or a spreadsheet of runs)
records numbers, but doesn't itself store or version the artifact those
numbers describe. The problem this topic answers: **how do you save a
trained model such that, six months from now, you (or a teammate) can find
the exact right version, trust it hasn't silently changed, and know what
produced it** — without relying on human memory or a folder full of
similarly-named files?

## Intuition

Imagine a folder of trained models saved over a few weeks of
experimentation:

```
model.pkl
model_v2.pkl
model_v2_final.pkl
model_v2_final_ACTUALLY.pkl
model_new_dataset.pkl
```

Which one is in production right now? Which one was trained on the old,
buggy data cleaning step? If two of these files happen to be byte-for-byte
identical (saved twice by accident), does anyone notice, or do both get
kept and confused for different things? A human-chosen filename carries no
guarantee about what's actually inside the file — it's a label, not an
identity. What *does* uniquely identify a file's content, with mathematical
certainty, is a cryptographic hash of its bytes: two files with the same
SHA-256 hash are (for all practical purposes) the same file; two files that
differ by even one byte get completely different hashes. That's the whole
idea this topic builds on.

## Why simpler approaches fail

Committing model binaries directly into Git — `git add model_v2_final.pkl`
— has the exact same shape of problem `02-git`'s motivating example solved
for code, just moved to artifacts:

- **Git's object model doesn't fit large binaries well.** Git computes
  diffs and stores deltas efficiently for text; for an opaque binary blob,
  every version is stored close to its full size, and there's no
  meaningful line-by-line diff to review — the repository bloats with
  every retrained model, and `git clone` gets slower for everyone forever
  (Git objects are, by design, never deleted from history without a rewrite).
- **Filenames aren't identity.** As in the intuition above, `model_v2.pkl`
  vs `model_v2_final.pkl` tells you nothing about whether they're actually
  different, which one is better, or what produced either of them. This is
  exactly the "which version is real" problem uncontrolled filenames caused
  for source code before version control — except an experiment log alone
  (a spreadsheet of run metrics) doesn't fix it either, because the log and
  the artifact are two separate things that can drift out of sync: the log
  says "run 47 got 94% accuracy" but nothing ties that log row to a
  specific, verifiable file on disk.
- **No linkage to what produced it.** Even if the file is findable, without
  captured metadata there's no way to know which commit of the training
  code, which version of the data, and which library versions produced it
  — so it can't be reliably reproduced or trusted later.

## Conceptual foundation

### Serialization formats and their tradeoffs

| Format | What it does | Tradeoffs |
|---|---|---|
| **pickle** (Python stdlib) | Serializes arbitrary Python objects, including full sklearn estimators, by reference to their class + `__dict__`. | Convenient, but **unpickling executes arbitrary code** if the file is untrusted (see Failure modes) — never unpickle a file from an unverified source. Also tightly coupled to the exact library version that wrote it (see below). |
| **joblib** | Built on pickle, optimized for objects containing large NumPy arrays (most sklearn models) — faster and more compact for that case. | Same security and version-coupling caveats as pickle; it's a pickle variant, not a different trust model. |
| **Framework-native** (`SavedModel` for TensorFlow, `state_dict` for PyTorch) | Serializes weights (and, for `SavedModel`, the computation graph) in the framework's own format. | More portable *within* that framework's ecosystem and across some version gaps than raw pickle, but still framework lock-in — a PyTorch `state_dict` is useless without PyTorch and the matching model class definition to load weights into. |
| **ONNX** (interchange format) | A framework-neutral graph representation — export from PyTorch/TensorFlow/sklearn, run in a different runtime (e.g. a C++ or mobile inference engine) without the original framework installed. | Solves framework lock-in and is generally safer to load (no arbitrary code execution), but not every operation in every framework has a lossless ONNX equivalent — conversion can silently change numerical behavior for unusual model architectures. |

The from-scratch implementation below uses joblib (matching what
`01-docker`'s `train_and_pickle.py` already does in this repo) precisely
because it's the common case and its risks are worth understanding
directly, not because it's the only correct choice.

### Model registries as "Git for models"

A **model registry** takes the same core idea Git uses for code — a
content-addressed store, where an object's identity *is* a hash of its
content, plus metadata layered on top (Git's commit messages/author/parent
pointers; a registry's training date/metric/library versions) — and applies
it to model artifacts instead of source files. "Content-addressed" is the
load-bearing idea: storage location is derived from the content itself
(its hash), not from an arbitrary name a human picked, which is exactly
what makes filename collisions harmless (see Experiment).

## Algorithm

The from-scratch registry (`registry.py`) implements this sequence for
each `register()` call:

1. Serialize the model object to a temporary file with `joblib.dump`.
2. Compute the SHA-256 hash of that file's raw bytes (`hash_file`).
3. Rename the temp file to a path derived from the hash
   (`registry/<hash-prefix>.pkl`) — this is the content-addressing step;
   whatever filename the caller *asked* to save under is never used as the
   storage key.
4. Build a metadata dict: hash, the caller-requested name (kept only as a
   label, not an identity), registration timestamp, a named metric and its
   value, and the exact library versions (`sklearn`, `joblib`, `numpy`,
   `python`) active at save time.
5. Write that metadata as a JSON sidecar file next to the model
   (`<hash-prefix>.json`) — so metadata travels with the artifact even if
   the log below is lost.
6. Append the same metadata to `registry_log.json` — an append-only JSON
   array, never rewritten or truncated, so every registration is a
   permanent, queryable record (mirroring Git's own append-only commit
   history).

## From-scratch implementation

`registry.py` (full file in this directory) — the core content-addressing
step:

```python
def register(model, requested_name, metric_name, metric_value, extra_metadata=None):
    REGISTRY_DIR.mkdir(exist_ok=True)
    tmp_path = REGISTRY_DIR / f"_tmp_{requested_name}"
    joblib.dump(model, tmp_path)
    content_hash = hash_file(tmp_path)

    short_hash = content_hash[:12]
    final_model_path = REGISTRY_DIR / f"{short_hash}.pkl"
    tmp_path.replace(final_model_path)          # storage key = hash, not requested_name

    metadata = {
        "requested_name": requested_name,
        "hash": content_hash,
        "short_hash": short_hash,
        "model_file": final_model_path.name,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "library_versions": _library_versions(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    (REGISTRY_DIR / f"{short_hash}.json").write_text(json.dumps(metadata, indent=2))
    _append_log(metadata)
    return metadata
```

`hash_file` reads in chunks (safe for model files far larger than fit in
memory at once):

```python
def hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()
```

`train_and_register.py` trains two real sklearn models on Iris and
registers both — see Experiment below for the actual executed output.

## Practical implementation

The production version of this exact idea is **MLflow's Model Registry**:
it content-addresses and versions registered models, attaches metadata
(metrics, parameters, the run that produced them), supports stage
transitions (Staging → Production → Archived), and integrates with the
tracking server so a registered model links back to the exact run, code
version, and data that produced it — everything `extra_metadata` in this
from-scratch version does by hand, at production scale. The full, real
MLflow integration is built in `05-mlflow-dagshub` (not duplicated here);
this topic's job is to make sure the *idea* underneath MLflow's registry —
content-addressing plus a metadata log — is understood before that
framework is introduced.

## Experiment

**Hypothesis:** if two different trained models are registered under the
exact same *requested* filename (`"model.pkl"` for both), a naive
filename-based save (`model_v1.pkl`, `model_v2.pkl` chosen by hand, or
worse, both literally named `model.pkl` and one overwriting the other)
would either collide (second overwrites first, data lost) or require a
human to manually invent distinguishing names. The content-addressed
registry should instead store both without collision or any human
naming decision, distinguishing them purely by their content hash.

**Setup** (`train_and_register.py`): train two `LogisticRegression` models
on the same Iris train/test split — version A with `C=1.0`, version B with
`C=0.01` — and call `register(model, requested_name="model.pkl", ...)` for
both.

**Actual result** (`.venv/bin/python train_and_register.py`, run from this
directory):

```
=== Training version A: LogisticRegression(C=1.0) ===
accuracy = 1.0000
registered as 2578e567ae76.pkl (hash 2578e567ae76)

=== Training version B: LogisticRegression(C=0.01) ===
accuracy = 0.8667
registered as 2a0165f30a0f.pkl (hash 2a0165f30a0f)

=== Registry log (both entries, same requested_name, different hash) ===
{
  "requested_name": "model.pkl",
  "short_hash": "2578e567ae76",
  "model_file": "2578e567ae76.pkl",
  "metric_value": 1.0,
  "hyperparameters": {
    "C": 1.0
  }
}
{
  "requested_name": "model.pkl",
  "short_hash": "2a0165f30a0f",
  "model_file": "2a0165f30a0f.pkl",
  "metric_value": 0.8666666666666667,
  "hyperparameters": {
    "C": 0.01
  }
}

CONFIRMED: identical requested_name ('model.pkl') for both versions did NOT cause a collision —
they are stored as distinct files (2578e567ae76.pkl vs 2a0165f30a0f.pkl) because their content hashes differ.

=== Loading version A back by hash and verifying integrity ===
reloaded model predictions match original: True
```

**Interpretation:** exactly as hypothesized. Both models were requested to
save as `"model.pkl"` — the naive-filename scenario this topic set out to
recreate — and neither overwrote the other, because the registry never
used the requested name as a storage key. The log correctly shows two
distinct entries with the same `requested_name` but different `short_hash`
and `model_file`, each carrying its own metric and hyperparameters.
Reloading version A back by hash (`load_model_by_hash`) reproduced
identical predictions to the in-memory original, confirming the
content-hash round-trips correctly and the integrity check
(re-hash-on-load) passes.

**Limitations:** two logistic regressions on a 150-row toy dataset is
enough to demonstrate the *mechanism* (hash-based identity beats
name-based identity), not to stress-test it at production scale — a real
registry also needs to handle much larger files efficiently (chunked
hashing here already accounts for that), concurrent writers to the log
(this JSON-array implementation is not safe for concurrent processes
writing simultaneously — a production registry needs a real database or
file-locking), and retention/cleanup policy for old versions, none of
which this from-scratch version addresses.

## Failure modes

- **Unpickling untrusted pickle files is a security risk.** `pickle.load`
  (and `joblib.load`, which is pickle underneath) can execute arbitrary
  code embedded in the file during deserialization — a malicious
  `.pkl` is a code-execution vector, not just a data file. Never load a
  pickle/joblib file from a source you don't trust and haven't verified
  (this is exactly why the registry's hash-based integrity check matters —
  it detects tampering, but only *after* trusting the original source).
- **No metadata linking a model to the code/data that produced it means
  irreproducibility.** A `.pkl` file with no sidecar recording which Git
  commit trained it, which dataset version, and which hyperparameters were
  used is a dead end six months later — the number is known, but not how
  to get back to it or verify it. `extra_metadata` in this registry
  (hyperparameters, a description of the training data) is a minimal stand-in
  for what a real system would capture via a Git commit hash and a data
  version identifier.
- **Silent format incompatibility across library versions.** A model
  pickled with `scikit-learn==1.3` is not guaranteed to unpickle correctly
  with `scikit-learn==1.8` — sklearn's own documentation explicitly warns
  that pickled estimators are not guaranteed to be compatible across
  versions, and a failure can be a hard crash *or*, worse, a silent
  behavior change with no error at all. This is exactly why
  `library_versions` is captured in every metadata entry here — not to
  prevent the incompatibility, but to make it diagnosable when it happens.

## Real-world usage

- Any team that retrains models regularly needs an answer to "which
  version is currently serving in production, and can I get back the exact
  model that produced last month's numbers" — that's what a registry is
  for, at the scale of hundreds or thousands of registered models rather
  than two.
- Model registries are the natural companion to the CI pipelines from
  `03-testing-ci`: a CI run that trains a candidate model can register it
  automatically, tagging the registry entry with the commit hash and test
  results that produced it, closing the loop between "this code passed
  tests" and "this exact artifact is what got deployed."
- `05-mlflow-dagshub` builds the production-grade version of this idea.

## Mental model

**A filename is a label a human chose; a content hash is an identity
nothing can fake. A model registry is what you get when you stop trusting
the label and start indexing by the identity — plus a metadata sidecar
recording everything needed to trust and reproduce what that identity
points to.**

## Questions to think about

1. If two teammates independently train what they believe are "different"
   models but happen to use identical code, data, and random seed, the
   registry would assign both runs the exact same content hash. Is that a
   bug in this design or a correct behavior? What would you want the
   registry to do differently, if anything?
2. The metadata sidecar captures `library_versions` at save time but not
   at load time. Sketch how you'd extend `load_model_by_hash` to warn (not
   necessarily block) when the currently installed `scikit-learn` version
   differs from the one recorded at registration — what would you compare,
   and what would you do if the load itself failed with a version-related
   error?
3. This registry's `registry_log.json` is a single JSON file rewritten in
   full on every `register()` call. What goes wrong if two training
   processes call `register()` at nearly the same time? Propose a minimal
   fix (you don't have to implement it).
4. `extra_metadata` here records hyperparameters and a description of the
   training data, but not the exact Git commit of the training code. Why
   does that matter for reproducibility specifically — what could change
   between two runs with identical hyperparameters and data that only a
   commit hash would capture?
