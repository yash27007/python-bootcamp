# CI/CD for ML pipelines

## Problem

Tasks 1-4 of this section built real pieces: a Dockerfile that
containerizes the environment (`01-docker`), Git as the source of truth
for changes (`02-git`), a real `pytest` suite that catches regressions
(`03-testing-ci`), and a versioned, hashed model artifact
(`04-model-packaging-versioning`). Each piece works when run by hand. But
nothing *connects* them: after someone edits `pipeline.py`, they still
have to remember, in order, every time: run the tests, build the image,
push the versioned artifact, and only then consider deploying. **How do
you make "test, then build, then deploy" happen automatically, and make a
failure at any step actually stop the ones after it**, instead of relying
on a person to run the right commands in the right order, every time?

## Intuition

Picture the four topics so far as stations on an assembly line, each one
useful on its own but currently operated by hand: someone has to walk the
part from station to station. CI/CD is the conveyor belt: it moves the
part (a code change) from station to station automatically, and — this is
the important part — each station has a gate that only lets the part
through if it passed inspection. If the part fails inspection at the
"test" station, the belt stops there. It never reaches "build" or
"deploy." Nobody has to remember to stop it; the belt is built to stop
itself.

Concretely: a developer pushes a commit. That push is an *event*. The
event triggers a *pipeline*: a sequence of *stages* (test → build →
deploy), where each stage's success is required before the next stage
starts. If stage 1 (tests) fails, stages 2 and 3 simply never run — the
same "half-broken thing never gets closer to production" guarantee a
manual checklist is supposed to provide, except automatic and unskippable.

## Why simpler approaches fail

The simpler approach is a checklist — written down, maybe even pinned to
a wiki page: "1. Run tests. 2. Build the image. 3. Push it. 4. Deploy."
This fails in a specific, predictable way, not a random one:

1. **It gets skipped under time pressure.** A checklist is only followed
   when someone chooses to follow it. The moment there's a deadline, an
   urgent hotfix, or simple fatigue, step 1 ("run the tests, it'll take a
   minute") is exactly the step that gets silently dropped — *especially*
   because skipping it doesn't cause an immediate visible error. The
   pipeline still "works" (a human runs the build and deploy commands
   fine); it's just no longer gated on anything.
2. **A skipped gate doesn't announce itself.** If a human forgets to run
   the tests before deploying, there's no signal anywhere that this
   happened — no red flag, no blocked merge, nothing in the commit
   history. The first sign of trouble is a production incident, at which
   point the question "did anyone even run the tests on this?" often
   can't be answered.
3. **It doesn't scale past one careful person.** A checklist followed
   diligently by one disciplined engineer says nothing about whether a
   teammate, a contractor, or that same engineer six months from now,
   under different pressure, follows it identically every time. Nothing
   *enforces* the order; it's a suggestion, not a gate.

The structural fix isn't "write a stricter checklist" — a checklist is
still something a human chooses to follow. The fix is to make the
computer run the checklist, so following it isn't a choice.

## Conceptual foundation

*(This section is titled "Conceptual foundation" rather than
"Mathematical foundation," following the same substitution
`03-testing-ci` documents: a CI/CD pipeline's structure has no closed-form
math to derive — the foundational idea is conceptual (a DAG of gated
stages triggered by an event), not mathematical. `08-monitoring`, the
next topic, does have a real statistical derivation.)*

**A CI/CD pipeline is a directed acyclic graph (DAG) of automated stages,
triggered by a code-repository event, where each stage gates the next.**
Unpacking each part of that:

- **Directed acyclic graph (DAG).** The stages have a strict order —
  test → build → deploy — and no stage depends on a later one (no
  cycles). This is the same DAG-of-dependent-steps idea that shows up in
  a Makefile or a data pipeline: nothing downstream should start until
  everything it depends on has finished successfully.
- **Triggered by a code-repository event.** The pipeline doesn't run on a
  schedule or "whenever someone remembers" — it runs automatically the
  moment a specific Git event happens: a `push` to a branch, or a pull
  request being opened/updated against one (the exact events `02-git`
  produces). This is what removes the human trigger from the equation.
- **Each stage gates the next.** "Gating" means the next stage's
  *existence* is conditional on the previous one's exit status. In
  `pipeline_gate.sh` below, `pytest`'s exit code is that gate mechanically
  (`$? -ne 0` stops the script); in the GitHub Actions YAML, a failed step
  stops the job by default and GitHub reports the whole workflow run as
  failed — which a branch-protection rule can then use to physically
  block the "Merge" button.
- **`03-testing-ci`'s tests are the first real gate.** This is the direct
  link back: `07-cicd` doesn't invent new checks. It automates *running*
  the exact `pytest` suite `03-testing-ci` already built
  (`08-mlops-deployment/03-testing-ci/test_pipeline.py`), and makes its
  pass/fail result the thing that decides whether "build" and "deploy"
  are even attempted.

CI (Continuous Integration) is usually used for the "test" part — every
change is automatically integrated and verified. CD (Continuous
Delivery/Deployment) extends the same DAG to "build" and "deploy." They're
presented together here because they're the same mechanism (gated,
event-triggered automation) applied to more stages, not two different
ideas.

## Algorithm

The pipeline this topic builds, in both its from-scratch and YAML forms:

1. **Trigger**: a `git push` or pull-request event against the
   repository (see `02-git`).
2. **Stage 1 — Test**: run the `pytest` suite. Capture its exit code.
   - Exit code `0` → continue to Stage 2.
   - Exit code non-zero → **halt**. Stages 2 and 3 never execute. Report
     failure back to wherever the trigger came from (a failed GitHub
     Actions check on a PR, or a non-zero shell exit locally).
3. **Stage 2 — Build**: only reached if Stage 1 passed. Package the code
   and/or model artifact (in a real system: build a Docker image from
   `01-docker`'s `Dockerfile`, or produce a versioned artifact per
   `04-model-packaging-versioning`).
   - Success → continue to Stage 3.
   - Failure → halt before Stage 3.
4. **Stage 3 — Deploy gate**: only reached if Stage 2 passed. In this
   topic's demo, this stage only *prints* "would deploy" — a real system
   would push the built image to a registry and roll it out (BentoML's
   serving layer from `06-bentoml` is a natural target).
5. **Report**: the overall pipeline status (pass/fail) is the single
   answer to "is this change safe to ship" — visible on the commit, the
   PR, and (with branch protection configured) blocking the merge button
   directly.

## From-scratch implementation

`pipeline_gate.sh` (this directory) — a real, runnable bash script
implementing exactly the three stages above by chaining shell exit codes:

```bash
"$PYTEST" "$TARGET_DIR/test_pipeline.py" -v
TEST_EXIT=$?

if [ "$TEST_EXIT" -ne 0 ]; then
    echo "PIPELINE HALTED at STAGE 1 (tests failed, exit code $TEST_EXIT)"
    echo "-> build and deploy stages were never reached."
    exit 1
fi

echo "STAGE 2/3: BUILD ..."
echo "STAGE 3/3: DEPLOY GATE -> would deploy"
exit 0
```

It takes a directory (containing a `pipeline.py` + `test_pipeline.py` pair
in `03-testing-ci`'s layout) as its argument, so it can be pointed at
either a working pipeline or a broken one — which is exactly the
experiment below.

## Practical implementation

`.github/workflows/testing-ci.yml` (repository root — GitHub only
discovers workflows there, so this is placed at the real location rather
than documented as an example) — the same three-gate idea, but triggered
automatically by GitHub on a Git event instead of run by hand:

```yaml
name: Testing CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    name: Run 03-testing-ci pytest suite
    runs-on: ubuntu-latest
    steps:
      - name: Check out repository
        uses: actions/checkout@v4
      - name: Set up Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.13"
      - name: Install uv
        uses: astral-sh/setup-uv@v3
      - name: Install dependencies (from pyproject.toml / uv.lock)
        run: uv sync --frozen
      - name: Run pytest on 03-testing-ci's pipeline suite
        run: uv run pytest 08-mlops-deployment/03-testing-ci/test_pipeline.py -v
```

This runs the *exact* pytest path documented and executed by hand in
`08-mlops-deployment/03-testing-ci/notes.md`
(`08-mlops-deployment/03-testing-ci/test_pipeline.py`) — same command,
same suite. What changes going from the from-scratch script to this YAML
is *who* runs it (a GitHub-hosted VM, not a person) and *when*
(automatically, on every `push`/PR to `main`, not "when someone
remembers"). This workflow only implements Stage 1 (test) for real —
Stages 2/3 (build/deploy) are left as the from-scratch script's job here,
since this repository doesn't ship a container registry or deployment
target to push to; a real project would add `docker build`/`docker push`
steps gated the same way, using the exact `Dockerfile` from `01-docker`.

**Honesty about execution**: like `01-docker`'s `Dockerfile`, this
workflow only truly *runs* once it's pushed to GitHub and a matching event
(a push or PR to `main`) fires — it was not executed by a GitHub Actions
runner in this local environment, because no such runner exists here.
What *was* verified locally is that the file is syntactically valid YAML:

```
$ .venv/bin/python -c "import yaml; yaml.safe_load(open('.github/workflows/testing-ci.yml'))" && echo "YAML parsed OK"
YAML parsed OK
```

One notable and worth-documenting quirk from that check: PyYAML (which
implements the older YAML 1.1 spec) parses the bare key `on:` as the
boolean `True`, not the string `"on"` — a well-known YAML 1.1 gotcha
(`on`/`off`/`yes`/`no` are boolean literals in 1.1). The file is still
syntactically valid YAML either way, and GitHub Actions' own parser
special-cases the `on:` key correctly as the trigger keyword regardless
of this YAML-1.1/1.2 discrepancy — this is a parser-library quirk, not a
bug in the workflow file.

## Experiment

**Hypothesis:** running `pipeline_gate.sh` against a pipeline whose tests
pass will proceed through all three stages and print "would deploy."
Running it against a pipeline with a deliberately broken `preprocess()`
(the same variance-instead-of-std bug `03-testing-ci` introduces) will
halt at Stage 1 and never reach Stage 2 or 3.

**Setup:** two directories, both containing `pipeline.py` +
`test_pipeline.py` in the layout `pipeline_gate.sh` expects:

- The real, correct `08-mlops-deployment/03-testing-ci/` directory
  (unmodified).
- A copy with the same one-character bug `03-testing-ci`'s own Experiment
  section introduces — dividing by `variance` instead of `std` in
  `preprocess()` — in a scratch directory, so the real `03-testing-ci`
  files are never touched by this demo.

**Actual result — passing run**
(`./pipeline_gate.sh 08-mlops-deployment/03-testing-ci`, run from the repo
root):

```
==================================================================
 STAGE 1/3: TEST  (gate: pytest exit code)
==================================================================
============================= test session starts ==============================
platform linux -- Python 3.13.9, pytest-9.1.1, pluggy-1.6.0 -- .../.venv/bin/python3
collecting ... collected 3 items

08-mlops-deployment/03-testing-ci/test_pipeline.py::test_preprocess_output_shape_and_dtype PASSED [ 33%]
08-mlops-deployment/03-testing-ci/test_pipeline.py::test_prediction_probabilities_are_valid PASSED [ 66%]
08-mlops-deployment/03-testing-ci/test_pipeline.py::test_accuracy_does_not_regress PASSED [100%]

============================== 3 passed in 0.73s ===============================

==================================================================
 STAGE 2/3: BUILD  (gate: tests passed)
==================================================================
  packaging pipeline.py + trained model artifact ...
  build OK

==================================================================
 STAGE 3/3: DEPLOY GATE  (gate: build succeeded)
==================================================================
  all gates passed -> would deploy

==================================================================
 PIPELINE SUCCEEDED: test -> build -> would-deploy
==================================================================
EXIT_CODE=0
```

**Actual result — failing run** (same script, pointed at the scratch copy
with the variance bug):

```
==================================================================
 STAGE 1/3: TEST  (gate: pytest exit code)
==================================================================
============================= test session starts ==============================
platform linux -- Python 3.13.9, pytest-9.1.1, pluggy-1.6.0 -- .../.venv/bin/python3
collecting ... collected 3 items

.../broken_pipeline/test_pipeline.py::test_preprocess_output_shape_and_dtype FAILED [ 33%]
.../broken_pipeline/test_pipeline.py::test_prediction_probabilities_are_valid PASSED [ 66%]
.../broken_pipeline/test_pipeline.py::test_accuracy_does_not_regress PASSED [100%]

=================================== FAILURES ===================================
____________________ test_preprocess_output_shape_and_dtype ____________________
    assert np.allclose(out.mean(axis=0), 0.0, atol=1e-8)
>   assert np.allclose(out.std(axis=0), 1.0, atol=1e-8)
E   assert False
E    +  where False = <function allclose at 0x...>(array([0.40824829, 0.40824829, 0.34874292]), 1.0, atol=1e-08)

========================= 1 failed, 2 passed in 0.78s ==========================

==================================================================
 PIPELINE HALTED at STAGE 1 (tests failed, exit code 1)
 -> build and deploy stages were never reached.
==================================================================
EXIT_CODE=1
```

**Interpretation:** exactly as hypothesized. The passing run walked
through all three stages and reached "would deploy"; the failing run
stopped dead after Stage 1 — the literal shell lines for Stage 2 (BUILD)
and Stage 3 (DEPLOY GATE) never printed, because `exit 1` inside the
`if` block returned control to the shell before reaching them. This is
the mechanical proof of the "gate" claim in the Conceptual foundation
section: it's not that build/deploy *also ran and happened to look fine*
— they structurally could not run.

**Limitations:** this demo uses a shell script with `exit` codes as the
gating mechanism, which is simple and transparent but not what a real CD
system uses for the build/deploy stages (those would be real `docker
build` / `docker push` / a deployment API call, each of which can fail
for reasons unrelated to code correctness — network errors, registry
auth, disk space — that this toy demo doesn't model).

## Failure modes

- **A CI pipeline slow enough that people bypass it.** If the test suite
  takes 40 minutes, engineers under deadline pressure start pushing
  directly to `main` with `--no-verify`, disabling the required check, or
  merging before CI finishes "because it always passes anyway." A slow
  gate reintroduces exactly the human-discipline failure mode this topic
  exists to remove — the fix is keeping the suite fast (this repo's suite
  runs in under a second) and parallelizing/splitting slow suites rather
  than letting people route around them.
- **Secrets committed into workflow files.** A workflow YAML often needs
  credentials (a registry password, a cloud API key, a DagsHub/MLflow
  token from `05-mlflow-dagshub`) to build and deploy. Hardcoding them
  directly in the `.yml` file puts them in Git history permanently — even
  deleting them later doesn't remove them from old commits. The correct
  pattern is GitHub's encrypted "Secrets" store, referenced in the
  workflow as `${{ secrets.SOME_TOKEN }}`, never as a literal value.
- **No rollback strategy on a bad deploy.** A pipeline that only knows how
  to move forward (test → build → deploy) has no answer for "the tests
  passed, the build succeeded, but the new version is bad in production
  in a way the tests didn't catch." Without a documented, ideally
  automated way to redeploy the previous known-good version quickly, a
  passing CI pipeline can give false confidence that shipping is always
  safe — CI/CD reduces the *chance* of shipping a broken change, it
  doesn't eliminate the need for a way to undo one.

## Real-world usage

Every major software team building anything deployed regularly uses this
pattern — GitHub Actions, GitLab CI, Jenkins, CircleCI, and cloud-native
options (AWS CodePipeline, Google Cloud Build) are all implementations of
the same DAG-of-gated-stages idea with different syntax. For ML
specifically, the pattern extends past "test the code": a mature MLOps
pipeline also gates on model-quality checks (does the newly trained model
beat the currently deployed one on a held-out set?), data-validation
checks (did the input schema change unexpectedly?), and — the subject of
the next topic — post-deployment monitoring signals feeding back into
whether to trigger a retraining pipeline automatically.

## Mental model

**A CI/CD pipeline is a checklist a human can no longer skip: an event
triggers a DAG of stages, and each stage's exit status is a hard gate on
whether the next stage exists at all — turning "please remember to test
before you deploy" from a request into a mechanical guarantee.**

## Questions to think about

1. `pipeline_gate.sh`'s Stage 2 (build) is currently just `echo`
   statements — it can't actually fail. If it were replaced with a real
   `docker build` command per `01-docker`'s `Dockerfile`, what new failure
   modes could Stage 2 introduce that Stage 1 (tests) structurally cannot
   catch?
2. The GitHub Actions workflow in this topic only automates Stage 1
   (test). Sketch what a `build` job would need to add to `testing-ci.yml`
   to also automate Stage 2, and what GitHub Actions feature would make
   `build` wait for `test` to pass first (hint: look at the `needs:` key).
3. Branch protection rules can require a CI check to pass before a PR can
   be merged, on GitHub. Why is *enforcing this at the platform level*
   (not just "the CI runs and shows a red X") the part that actually
   solves the "checklist gets skipped under pressure" problem from this
   topic's Why-simpler-approaches-fail section?
4. Suppose a workflow's `test` job passes reliably but takes 45 minutes
   because it retrains a full model every run instead of running the
   fast `pytest` unit/invariant suite from `03-testing-ci`. Connect this
   back to this topic's Failure modes section: what two different bad
   outcomes could this slowness produce, and are they the same failure
   mode or different ones?
