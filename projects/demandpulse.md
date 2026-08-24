# DemandPulse

**Status:** 🗓 Planned — no repository yet, name and scope only loosely fixed.

## Problem

Time-series demand forecasting — predicting future demand for a product/resource from historical
patterns, seasonality, and external signals.

## Why it matters

Regression on i.i.d. rows (what `05-machine-learning` teaches by default) and forecasting a
time-ordered sequence are genuinely different problems — a forecast has to respect the arrow of
time (no leaking future information into training) in a way a standard train/test split doesn't
enforce automatically.

## Concepts learned (curriculum cross-references, planned)

- `05-machine-learning` — regression fundamentals, generalized to a time-respecting evaluation
  scheme (rolling-origin cross-validation, not a random split)
- `06-deep-learning/03-rnn` and `04-lstm-gru` — sequence models, if the eventual approach uses one
- `08-mlops-deployment/07-cicd` and `08-monitoring` — a forecasting model needs periodic retraining
  as the underlying demand pattern shifts, which is exactly what those two topics teach

## Technologies (anticipated, not yet built)

Not yet decided — will be filled in once the project starts.

## Prerequisites

`05-machine-learning` and `06-deep-learning`'s sequence-model topics.

## Link to project repository

None yet.

## Expected learning outcomes

Practice the specific discipline time-series problems demand: leakage-free evaluation, handling
seasonality/trend, and deciding when a classical statistical model is enough versus when a
learned sequence model earns its added complexity.
