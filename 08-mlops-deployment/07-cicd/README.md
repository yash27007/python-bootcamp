# 07 – CI/CD

## What you'll learn

Why having a Dockerfile (`01`), Git history (`02`), a `pytest` suite
(`03`), and a versioned model artifact (`04`) still isn't enough — nothing
runs any of it automatically. CI/CD as a DAG of automated, gated stages
triggered by a Git event: a from-scratch bash script chaining
test → build → deploy-gate by shell exit codes, then the same idea as a
real GitHub Actions workflow running `03-testing-ci`'s actual `pytest`
suite on every push/PR.

| Topic | Status |
|-------|--------|
| CI/CD as a DAG of gated stages | ✅ Complete |
| A from-scratch pipeline-gating script, run pass and fail | ✅ Complete |
| A real GitHub Actions workflow for this repo | ✅ Complete |

## Why it matters

A manual checklist ("run tests, then build, then deploy") is a request, not
a guarantee — it gets skipped under deadline pressure, and a skipped gate
doesn't announce itself until something breaks in production. CI/CD turns
"please remember to test before deploying" into a mechanical property of
the system: a failed test genuinely, structurally prevents the build and
deploy stages from ever running.

## Prerequisites

- `02-git` — the pipeline is triggered by the exact Git events (`push`,
  pull request) this topic covers.
- `03-testing-ci` — this topic automates *running* that topic's `pytest`
  suite; it assumes that suite already exists and passes.
- `01-docker` (helpful, not required) — a real CD pipeline's build stage
  would use `01-docker`'s `Dockerfile`.

## What you'll build

- `pipeline_gate.sh` — a real, runnable bash script implementing
  test → build → deploy-gate as three shell stages chained by exit codes.
  Actually run twice: once against `03-testing-ci`'s real passing suite
  (reaches "would deploy"), once against a deliberately broken copy
  (halts at the test stage, build/deploy never run) — real captured
  output for both in `notes.md`.
- `.github/workflows/testing-ci.yml` (repo root — GitHub only discovers
  workflows there) — a real, syntactically-valid GitHub Actions workflow
  that runs `03-testing-ci`'s actual `test_pipeline.py` on every push/PR
  to `main`. YAML syntax verified locally with a Python `yaml.safe_load`;
  honestly documented as un-executed here (it only truly runs once
  pushed to GitHub).

## Where it appears in real systems

- Every team shipping software on a schedule uses some form of this:
  GitHub Actions, GitLab CI, Jenkins, CircleCI — different syntax, same
  DAG-of-gated-stages mechanism.
- `08-monitoring`'s "real-world usage" section connects this same
  mechanism to *automatic retraining*: a drift alert becomes the trigger
  event for a pipeline shaped exactly like this one, with "retrain →
  validate → deploy" as its gated stages.

## What's next

`08-monitoring` — once a model is actually deployed via a pipeline like
this one, the next question is what tells you it needs to be redeployed:
detecting that the live data it's serving on has drifted away from what
it was trained on.
