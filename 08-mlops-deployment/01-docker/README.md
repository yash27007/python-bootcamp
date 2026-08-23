# 01 – Docker

## What you'll learn

Why an application that runs correctly on the machine that built it can fail on any other machine (the "works on my machine" problem), and how Docker solves this by packaging an application together with its entire runtime environment into a portable, content-addressed image. Covers containers vs. virtual machines, the union/layered filesystem model that makes image layers cacheable and shareable, the image-vs-container distinction, building a real Dockerfile end to end, and Docker Compose for declaring a service instead of typing `docker run` flags by hand.

| Topic | Status |
|-------|--------|
| What Are Containers? | ✅ Complete |
| Docker Images vs Containers | ✅ Complete |
| Docker vs Virtual Machines | ✅ Complete |
| Docker Installation & Basic Commands | ✅ Complete |
| Creating a Docker Image | ✅ Complete |
| Pushing to Docker Hub | ✅ Complete |
| Docker Compose | ✅ Complete |

## Why it matters

Every later topic in this section assumes an application can be reliably reproduced somewhere other than the machine it was written on — a CI runner, a teammate's laptop, a production cluster. Docker is the standard mechanism for that reproducibility in modern ML/software engineering: training environments, serving endpoints, and CI jobs are almost always run inside containers precisely so that "it passed in CI" and "it works in production" mean the same thing as "it worked when I tested it."

## Prerequisites

- Basic command-line familiarity (`01-python-foundation`).
- A pickled/serialized model artifact and a minimal serving script — this topic builds one from scratch (`train_and_pickle.py`, `app.py`) rather than assuming one.
- No prior containerization knowledge assumed.

## What you'll build

- A trained `LogisticRegression` (Iris) pickled to `model.pkl`, served by a minimal Flask app (`app.py`) with `/health` and `/predict` endpoints — verified working by actually starting it and hitting it with `curl`.
- A complete, reviewed `Dockerfile` that packages that app: pinned base image, dependency-layer-before-code-layer ordering for cache efficiency, a non-root runtime user, `EXPOSE`/`HEALTHCHECK`, and an exec-form entrypoint — written and carefully reviewed for correctness, but **not executed with `docker build` in this environment** (no working Docker daemon reachable from this WSL2 setup; see `notes.md` for the verified `docker info` failure and full explanation).
- A `docker-compose.yml` wrapping the build+run into one declarative service definition.
- `build_cache.py` — a real, executed, from-scratch Python implementation of a content-hash-based build cache, demonstrating exactly what Docker's layer cache is doing underneath `docker build`, timed across cold/warm/early-change/late-change scenarios.

## Where it appears in real systems

- Containerized training jobs, where the exact library/CUDA versions used to produce a model matter for reproducing results later.
- Containerized model-serving endpoints — this topic's `app.py` + `Dockerfile` is the minimal version of a pattern used at production scale.
- CI/CD pipelines (`08-mlops-deployment/07-cicd`, planned) that build and test inside containers so tests run in the same environment code will actually ship in.

## What's next

`02-git` — version control for the code (and, from here on, the containers/configuration) this topic produces; its "Conceptual foundation" section explicitly connects Git's content-addressed object model back to Docker's content-addressed image layers covered here.
