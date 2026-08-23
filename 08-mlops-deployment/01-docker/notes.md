# 01 – Docker

## Problem

A model trains and serves correctly on the machine that built it, then breaks the moment it moves anywhere else — a teammate's laptop, a CI runner, a production server. The training script that produced 97% accuracy this morning fails to even import on a colleague's machine because their `scikit-learn` is a minor version behind. A Flask app that serves predictions locally throws a cryptic `ImportError` in production because the production host's system Python is 3.9 and the code uses a 3.11-only syntax feature. This is the classic **"works on my machine"** problem: the code is not actually self-contained — it silently depends on a huge amount of context (the exact interpreter version, exact library versions, OS-level shared libraries, environment variables, even filesystem layout) that lives on the developer's machine and nowhere else, and none of that context travels with the code itself.

**How do we package an application together with the *entire* environment it needs to run, so that "it works here" and "it works everywhere" become the same statement?**

## Intuition

Think about shipping physical cargo before the standardized shipping container existed: every item was loaded by hand, in whatever shape it came in, and every port needed different equipment and expertise to handle wildly different cargo (barrels, crates, sacks, machinery). The shipping container didn't change what was inside it — it changed the fact that *any* port, ship, or truck could move it without knowing or caring what was inside, because the container's outer interface was standardized.

Docker does the same thing for software. A **container image** packages an application together with everything it needs — the exact interpreter, the exact library versions, OS-level dependencies, configuration — into one standardized, portable unit. Whoever runs that image, on whatever machine, with whatever Docker installed, gets byte-for-byte the same environment the image was built with. "It works on my machine" stops being a meaningful distinction, because the image *is* the machine (the relevant parts of it) — it travels with the code instead of being left behind on one developer's laptop.

## Why simpler approaches fail

The obvious lighter-weight fix is: write down the setup instructions. A `SETUP.md` that says "install Python 3.11, run `pip install -r requirements.txt`, install `libpq-dev` for the Postgres driver, set these three environment variables..." This fails for two structural reasons, not just because people are lazy about reading docs:

1. **It rots continuously and silently.** The moment any upstream dependency publishes a new version — a transitive dependency of a transitive dependency bumps a version and subtly changes behavior — the written instructions are still "correct" (they still describe *a* way to get a working environment as of the day they were written) but no longer *reproduce the same environment*. Nothing signals that the instructions have gone stale; they just quietly stop producing what they used to produce, and the next person to follow them gets a different, possibly broken, result with no error message pointing at why.
2. **It cannot capture OS-level dependencies at all.** `pip install -r requirements.txt` only ever manages Python packages. It says nothing about the C libraries some of those packages link against (`libpq` for Postgres drivers, `libgomp` for some numeric libraries, specific CUDA/cuDNN versions for GPU work), nothing about the OS itself, nothing about locale settings, nothing about system-level binaries the code shells out to. A written instruction list is fundamentally scoped to "things one package manager knows how to install," while the actual runtime environment is much bigger than that — and every gap between what's written down and what's actually required becomes a "works on my machine" bug for the next person.

A written instruction list is a description of an environment, re-interpreted by a human, at build time, against whatever the ambient OS state happens to be. It is not the environment itself, and that gap is exactly where reproducibility breaks.

## Conceptual foundation

*(This is a systems/tooling topic without calculus-style mathematics behind it, per AGENTS.md's documented substitution — the real conceptual depth here is Docker's storage/execution model, not an equation.)*

### Images are a stack of read-only layers

A Docker image is not a single monolithic snapshot of a filesystem. It is a stack of **layers**, where each layer is the filesystem *diff* produced by one instruction in the build (one `FROM`, `COPY`, or `RUN` line). Docker uses a **union filesystem** (technically overlayfs on modern Linux) to present these stacked, read-only diffs as if they were one single, coherent filesystem to anything running inside the container — the union filesystem is the mechanism that merges N separate immutable layers into what looks like one ordinary directory tree, without physically copying any of their contents together.

Concretely, for this topic's Dockerfile:

```
Layer 1: FROM python:3.11.9-slim-bookworm   (the entire base OS + Python install)
Layer 2: COPY requirements.txt .            (adds one file)
Layer 3: RUN pip install -r requirements.txt (adds installed packages)
Layer 4: COPY app.py .                      (adds one file)
Layer 5: COPY model.pkl .                   (adds one file)
```

Each layer is content-addressed by a hash of its own contents (this is precisely the same idea `02-git/notes.md` covers for Git objects — worth reading side by side). Two images that share an identical base-image layer physically share that layer's data on disk; Docker never stores it twice.

### Why this makes layer caching possible

Because each layer's identity is a hash of "the instruction plus everything before it in the build," Docker can ask, before running any instruction: *have I already produced a layer with exactly this hash?* If yes, it reuses the existing layer instead of re-running the instruction. This is why Dockerfile instruction **order matters enormously** for build speed: this topic's Dockerfile deliberately copies `requirements.txt` and runs `pip install` *before* copying `app.py`, because `requirements.txt` changes far less often than application code — so a routine code-only change reuses the (often slow) dependency-install layer's cache untouched, and only re-runs the fast `COPY app.py` step and everything after it.

`build_cache.py` in this same folder is a small, real, executed implementation of exactly this caching mechanism, stripped down to its essence — see "From-scratch implementation" below.

### Image vs container

An **image** is the read-only layer stack itself — a template, sitting on disk (or in a registry), that does nothing on its own. A **container** is a *running instance* of an image: Docker takes the image's read-only layers, adds exactly one additional **writable layer** on top (the "container layer"), and starts a process inside that combined filesystem view. Any file the running process writes, deletes, or modifies lands only in that thin writable layer; the underlying image layers are never touched.

This is why you can run the *same* image many times, as many independent containers, without them interfering with each other's writes — and why deleting a container throws away only that thin writable layer, leaving the (much larger, expensively-built) image intact and ready to spin up a fresh container instantly. It is the same relationship a class has to its instances: one image, many containers, each with its own throwaway state layered on top of shared, immutable, cacheable foundations.

## Algorithm / Practical implementation

The "algorithm" here is procedural rather than mathematical — how a Dockerfile turns into a running service:

1. **Choose and pin a base image** (`FROM python:3.11.9-slim-bookworm`) — an exact tag, not a moving one like `latest`, so the build is reproducible months later.
2. **Copy dependency manifests first** (`COPY requirements.txt .`) and **install dependencies** (`RUN pip install ...`) — in their own layer, ahead of application code, for the caching reason explained above.
3. **Copy application code and artifacts** (`COPY app.py .`, `COPY model.pkl .`) — last, since these change most often.
4. **Drop root privileges** for the running process (`USER appuser`) — defense in depth.
5. **Declare the port** the app listens on (`EXPOSE 5000`) and a **health check** the daemon/orchestrator can poll.
6. **Set the entrypoint** (`CMD ["python", "app.py"]`) — what actually runs when a container starts from this image.

This folder contains a complete, real implementation of this progression:

- **`train_and_pickle.py`** — trains a `LogisticRegression` on the Iris dataset and pickles it to `model.pkl`. Executed for real in this environment:
  ```
  $ .venv/bin/python train_and_pickle.py
  Trained LogisticRegression on Iris, saved to model.pkl
  Train accuracy: 0.9733
  ```
- **`app.py`** — a minimal Flask app that loads `model.pkl` and serves predictions at `POST /predict` and a health check at `GET /health`. **Verified working outside the container** by actually starting it and hitting it with `curl` in this environment:
  ```
  $ curl -s http://localhost:5000/health
  {"status":"ok"}
  $ curl -s -X POST http://localhost:5000/predict \
      -H "Content-Type: application/json" -d '{"features":[5.1,3.5,1.4,0.2]}'
  {"class_name":"setosa","prediction":0}
  ```
- **`requirements.txt`** — pinned `flask==3.1.3`, `scikit-learn==1.8.0` (the exact versions verified installed in this repo's `.venv`).
- **`Dockerfile`** — the full image build recipe described above, with inline comments explaining every instruction's purpose (pinned base tag, layer ordering for caching, non-root user, `EXPOSE`, `HEALTHCHECK`, exec-form `CMD`).
- **`.dockerignore`** — excludes `.venv/`, `.git/`, caches, and logs from the build context, directly addressing the "image bloat" and "secrets baked into layers" failure modes below.
- **`docker-compose.yml`** — a single-service Compose file (covers this folder's original "Docker Compose" subtopic) that wraps `docker build` + `docker run -p 5000:5000` into one declarative `docker compose up`.

**Explicit, verified environment fact — this Dockerfile was NOT executed with `docker build` in this environment.** `docker info` was checked directly:
```
$ docker info
The command 'docker' could not be found in this WSL 2 distro.
We recommend to activate the WSL integration in Docker Desktop settings.
```
The `docker` CLI binary exists on the Windows side but there is no working Docker daemon reachable from this WSL2 environment. The Dockerfile above has been **written and carefully reviewed** for correctness (verified against real, tested `app.py`/`requirements.txt`/`model.pkl` artifacts, and against standard, well-established Docker practice), but its build has **not been executed here**, and no build output anywhere in this notes.md is fabricated. A reader with a working Docker daemon can run, from this folder:
```
docker build -t iris-predictor .
docker run -p 5000:5000 iris-predictor
```

## From-scratch implementation

`build_cache.py` (in this folder) implements, in plain Python with no Docker involved, exactly the mechanism described in "Conceptual foundation" for layer caching: a "build" is a list of `(step_name, step_content, run_fn)` steps; each step's cache key is `sha256(previous_step's_hash : step_name : step_content)` — chaining the running hash forward is what makes changing an *early* step invalidate every *later* step's cache key too, even though those later steps' own content didn't change, mirroring exactly how a change to an early Dockerfile instruction invalidates every layer built after it. If a step's hash has already been computed and cached on disk, `run_fn` (which stands in for potentially expensive work — pulling a base image, running `pip install`) is **skipped entirely** and the cached result is reused.

This was actually executed with `.venv/bin/python build_cache.py`; real captured output:

```
=== Build 1: cold cache (everything is a MISS) ===
[CACHE MISS] 'FROM python:3.11-slim'                                 hash=71762f8465b8 (0.3002s)
[CACHE MISS] 'COPY requirements.txt .'                               hash=c15ac3392f9a (0.0502s)
[CACHE MISS] 'RUN pip install -r requirements.txt'                   hash=5bbbb2ca1d20 (1.0002s)
[CACHE MISS] 'COPY . .'                                              hash=fa3bb186cc1d (0.0502s)
Total: 1.4064s

=== Build 2: rerun with NO changes (everything should HIT) ===
[CACHE HIT ] 'FROM python:3.11-slim'                                 hash=71762f8465b8 (0.0001s)
[CACHE HIT ] 'COPY requirements.txt .'                               hash=c15ac3392f9a (0.0000s)
[CACHE HIT ] 'RUN pip install -r requirements.txt'                   hash=5bbbb2ca1d20 (0.0000s)
[CACHE HIT ] 'COPY . .'                                              hash=fa3bb186cc1d (0.0000s)
Total: 0.0003s

=== Build 3: change LATE layer only (app code changes) ===
[CACHE HIT ] 'FROM python:3.11-slim'                                 hash=71762f8465b8 (0.0000s)
[CACHE HIT ] 'COPY requirements.txt .'                               hash=c15ac3392f9a (0.0000s)
[CACHE HIT ] 'RUN pip install -r requirements.txt'                   hash=5bbbb2ca1d20 (0.0000s)
[CACHE MISS] 'COPY . .'                                              hash=d302c8b71059 (0.0502s)
Total: 0.0506s

=== Build 4: change EARLY layer only (requirements.txt changes) ===
[CACHE HIT ] 'FROM python:3.11-slim'                                 hash=71762f8465b8 (0.0001s)
[CACHE MISS] 'COPY requirements.txt .'                               hash=8f71e0440b16 (0.0502s)
[CACHE MISS] 'RUN pip install -r requirements.txt'                   hash=e7ea58aa92ec (1.0003s)
[CACHE MISS] 'COPY . .'                                              hash=5d5d1adff403 (0.0503s)
Total: 1.1022s
```

This is the mechanism Docker's build cache uses underneath `docker build`, made visible: content hashing, a chained hash so earlier changes cascade forward, and a disk-backed cache of prior step results keyed by that hash.

## Experiment

**Hypothesis:** changing an *early* build step should invalidate (force a re-run of) every step after it — including expensive ones — while changing a *late* build step should invalidate only itself, leaving every earlier step's cache untouched and fast.

**Setup:** `build_cache.py`'s `__main__` block runs the same 4-step pseudo-build ("pull base image" 0.3s, "copy requirements.txt" 0.05s, "install dependencies" 1.0s [the expensive step], "copy app code" 0.05s) four times: (1) cold cache, (2) identical rerun, (3) rerun with only the *last* step's content changed, (4) rerun with the *second* step's content changed (which cascades into the third step too, since `RUN pip install` depends on `requirements.txt`'s content).

**Actual result** (from the captured run above):

| Run | What changed | Total time | Steps re-run |
|---|---|---|---|
| 1 (cold) | everything | 1.4064s | all 4 |
| 2 (no change) | nothing | 0.0003s | none — 4 cache hits |
| 3 (late change) | last step only | 0.0506s | 1 (last step only; the expensive 1.0s install step stayed cached) |
| 4 (early change) | 2nd step (cascades to 3rd, 4th) | 1.1022s | 3 (2nd, 3rd, 4th — including the expensive 1.0s install step again) |

**Interpretation:** Run 3 finishes in ~0.05s — 28× faster than the 1.4s cold build — because the expensive "install dependencies" step's cache entry was untouched by a change to a later step. Run 4 costs almost as much as a fully cold build (1.10s vs 1.41s) even though only the requirements file changed, because that change sits *before* the expensive install step in the chain, and the chained-hash design (each step's key depends on every prior step's key) correctly and unavoidably invalidates everything downstream of the change. This numerically confirms the ordering advice in "Algorithm/Practical implementation": put whatever changes least often as early as possible in the Dockerfile.

**Limitations:** this is a simulated build (steps sleep for a fixed duration rather than doing real I/O/compute), so the absolute timings are illustrative, not a benchmark of real Docker; a real `pip install` layer's cost varies with network speed and package count, and a real Docker build cache also accounts for build-context digest and multi-platform concerns this toy script ignores. What transfers faithfully is the *qualitative* mechanism: content-hash caching with forward-chained invalidation.

## Failure modes

- **Unpinned base image tags.** `FROM python:latest` (or `python:3.11` without a full patch version) resolves to whatever that tag currently points to — which changes over time. A build that succeeded last month can produce a subtly different (or outright broken) image today, from an unchanged Dockerfile, because the *tag* moved underneath it. This directly reproduces the "works on my machine" problem this topic exists to solve, just one layer down. Fix: pin an exact tag (or, for maximum reproducibility, a digest — `FROM python@sha256:...`).
- **Image bloat from uncleaned build artifacts.** Leaving compiler toolchains, package manager caches (`pip`'s download cache, `apt`'s package lists), or test/debug files in the final image inflates its size — slower pulls, slower deploys, more attack surface. Multi-stage builds (build in one stage, copy only the final artifacts into a slim final stage) and flags like `pip install --no-cache-dir` (used in this topic's Dockerfile) address this directly.
- **Secrets baked into image layers.** `COPY .env .` or `ARG API_KEY=...` embeds that secret's value into a layer's content-addressed hash permanently — even if a *later* layer deletes the file, the secret is still recoverable from the earlier layer's data, because layers are immutable and layer history ships with the image. Secrets belong in runtime injection (environment variables at `docker run` time, mounted secret files, or a secrets manager), never in a `COPY` or `ARG` baked into a layer.

## Real-world usage

- **Containerized training jobs.** A training script's exact library/CUDA/cuDNN versions matter enormously for numerical reproducibility; packaging the training environment as an image means "rerun this experiment" reproduces the same environment on any machine (or cluster node) that can run the image, not just the one it was developed on.
- **Containerized model serving.** This topic's `app.py` + `Dockerfile` is the minimal version of a pattern used everywhere in production ML: a served model wrapped in a lightweight HTTP layer, packaged as an image, deployed identically to a laptop, a staging environment, or a Kubernetes cluster — the same image, unmodified, at every stage.
- **CI/CD pipelines** build and test inside containers precisely so "tests passed in CI" means the same environment the code will actually run in, not a coincidentally-similar one.

## Mental model

**An image is a frozen, content-addressed stack of filesystem diffs; a container is that stack plus one throwaway writable layer, running as a process.** Order your Dockerfile so the layers that change least sit at the top (cached, reused) and the layers that change most sit at the bottom (rebuilt often, cheaply) — because a layer's cache validity depends on every layer before it, not just itself.

## Questions to think about

1. If `build_cache.py`'s Experiment showed that changing `requirements.txt` re-runs `pip install` *and* every step after it, what would happen if you swapped the order — put `COPY app.py .` *before* `COPY requirements.txt .` / `RUN pip install ...` in a real Dockerfile? Which everyday change (editing application code vs. adding a new dependency) would become expensive that wasn't before?
2. This topic's Dockerfile copies a pre-trained `model.pkl` into the image rather than training the model as part of the build. What would change (build time, image reproducibility, ability to retrain without rebuilding the image) if `RUN python train_and_pickle.py` were a build step instead?
3. Two different images both `FROM python:3.11.9-slim-bookworm` and both `RUN pip install flask==3.1.3 scikit-learn==1.8.0` with identical `requirements.txt` content. Given the "Conceptual foundation" section's explanation of content-addressed layers, do these two images share any actual layer data on disk, or does Docker store two independent copies? What has to be *exactly* identical for that sharing to happen?
4. `HEALTHCHECK` in this topic's Dockerfile polls `GET /health` every 30 seconds. Why is a container-level health check useful even though the process inside (`python app.py`) is still technically "running" (not crashed) even when it's stuck, deadlocked, or the model failed to load?
