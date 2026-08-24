# EdgeSense

**Status:** 🗓 Planned — no repository yet, tied to this repo's longer-term Edge AI direction.

## Problem

Running inference under real hardware constraints — limited memory, no GPU, tight latency
budgets — rather than assuming a model can always call out to a cloud endpoint.

## Why it matters

Every model built elsewhere in this repo runs on a normal development machine with no resource
constraints. Edge deployment is a different problem: the model itself often has to change
(quantization, pruning, distillation) to fit the target device, not just get packaged
differently.

## Concepts learned (curriculum cross-references, planned)

- `09-pytorch` — the training/serialization foundation any edge-bound model still starts from
- `08-mlops-deployment` — packaging and versioning ideas apply, but the deployment target is a
  device, not a server
- A future Edge AI section of this curriculum (not yet built — see the design spec's "Learned /
  Currently learning / Planned" distinction: this is explicitly a Planned direction, not
  something this repo teaches yet)

## Technologies (anticipated, not yet built)

Not yet decided — likely a quantization/pruning toolchain and an on-device or embedded runtime,
to be determined once this direction is actually started.

## Prerequisites

`09-pytorch` complete; a dedicated Edge AI curriculum section would need to exist first for this
project to be attempted with the same first-principles rigor as the rest of this repo.

## Link to project repository

None yet.

## Expected learning outcomes

Understand what changes when "does it run" becomes "does it run in 50ms on 256MB of RAM" — the
constraint-driven engineering that separates edge deployment from server deployment.
