# 08 – Monitoring

## What you'll learn

Why a model that was accurate at deploy time can silently degrade as
real-world input data drifts away from what it was trained on — and why
"check accuracy occasionally by hand" doesn't catch this early (it needs
labels that arrive late, if ever). A real statistical derivation of the
Population Stability Index (PSI) from first principles (symmetrized KL
divergence), implemented from scratch in NumPy, and run on a synthetic
reference-vs-shifted dataset to show it correctly flags injected drift
and stays quiet when there's none.

| Topic | Status |
|-------|--------|
| Why post-deployment drift is invisible without monitoring | ✅ Complete |
| PSI derived from KL divergence, real math | ✅ Complete |
| From-scratch PSI implementation, run on synthetic shift | ✅ Complete |
| Connecting to real tools (Evidently, WhyLabs) and retraining triggers | ✅ Complete |

## Why it matters

Every prior topic in this section ends at "the model is deployed and
serving requests." None of them says anything about what happens next.
The real world doesn't hold still — user behavior changes, upstream data
sources change — and a model doesn't crash when its inputs drift, it just
keeps confidently answering questions about a world that no longer
matches what it was trained on. Detecting that early, from the input side
and without needing ground-truth labels, is what turns "we found out from
a support ticket" into "an alert fired before it mattered."

## Prerequisites

- `04-model-packaging-versioning` and `05-mlflow-dagshub` — monitoring
  compares live traffic against a *reference* distribution, which is
  naturally the training-time data captured by those topics' tracking.
- `07-cicd` — the "real-world usage" section here connects a drift alert
  to an automatic retraining pipeline, shaped exactly like `07-cicd`'s
  gated-stages DAG.
- Basic probability (distributions, histograms) and NumPy familiarity.

## What you'll build

- `drift_detection.py` — a from-scratch NumPy implementation of the
  Population Stability Index, derived in `notes.md` as the discretized,
  binned Jeffreys divergence (symmetrized KL divergence) between a
  reference and a live-window distribution.
- A real, fixed-seed synthetic experiment: PSI computed between a
  reference distribution and both an unshifted and a deliberately shifted
  live window, hypothesis stated first, real output captured showing the
  detector correctly stays quiet on the unshifted window and flags the
  shifted one — runs in well under a second.

## Where it appears in real systems

- Production ML monitoring tools — Evidently (open-source) and WhyLabs
  (hosted) — compute this exact class of statistic continuously over
  production traffic and drive alerting/dashboards from it.
- A significant, sustained drift alert is the trigger event for an
  automatic retraining pipeline in mature MLOps setups — the same
  gated-DAG mechanism `07-cicd` builds, with "drift alert" instead of
  "git push" as the trigger.

## What's next

This is the last topic in `08-mlops-deployment`'s current build-out: the
section now covers the full lifecycle from containerizing an environment
(`01`) through source control (`02`), testing (`03`), model versioning
(`04`), experiment tracking (`05`), serving (`06`), automated pipelines
(`07`), to post-deployment monitoring (`08`).
