# Student Performance Indicator

**Status:** ✅ Built — [github.com/yash27007/ml-spi](https://github.com/yash27007/ml-spi)

## Problem

Predict a student's exam performance from demographic, socioeconomic, and preparation-related
features (parental education, test preparation course, lunch type, etc.), and expose that
prediction as a servable web application rather than a one-off notebook result.

## Why it matters

Every course project up to `05-machine-learning`/`08-mlops-deployment` in this repo teaches one
concept at a time. A real project forces integrating several of them into one coherent pipeline —
the gap between "I can fit a regressor in a notebook" and "I can ship a model someone else can
actually use" is most of the real engineering work.

## Concepts learned (curriculum cross-references)

- `02-statistics` — descriptive statistics and inferential reasoning during EDA
- `03-data-analysis` — Pandas-based EDA, visualization
- `04-feature-engineering` — encoding categorical features, handling the dataset's specific
  feature set
- `05-machine-learning` — model selection and evaluation for a regression target
- `08-mlops-deployment` — packaging and serving the trained model (this repo's `06-bentoml`/
  `01-docker` topics cover the same underlying ideas this project applies)

## Technologies

Python, scikit-learn, Flask (or similar) for serving — see the project repo's own README for the
exact stack.

## Prerequisites

Comfortable with `05-machine-learning` (regression) and `04-feature-engineering` (categorical
encoding) before attempting to reproduce this project from scratch.

## Link to project repository

**[github.com/yash27007/ml-spi](https://github.com/yash27007/ml-spi)** — the actual
implementation lives there, not in this repo. This file is an index card, not a copy.

## Expected learning outcomes

Practice taking a model from "trained in a notebook" to "packaged and servable" — the exact
transition `08-mlops-deployment`'s topic progression (package → track → version → serve) teaches
conceptually, applied here to a real dataset end to end.
