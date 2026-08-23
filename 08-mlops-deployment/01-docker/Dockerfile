# Dockerfile for a minimal sklearn model-serving app (app.py + model.pkl).
#
# IMPORTANT (see notes.md "Algorithm/Practical implementation"): this file
# has been written and carefully reviewed for correctness, but it has NOT
# been executed with `docker build` in this environment -- there is no
# working Docker daemon here (the `docker` CLI is present but WSL2
# integration with Docker Desktop is not configured; `docker info` fails
# with "could not be found in this WSL 2 distro"). Every instruction below
# is standard, well-established Dockerfile practice; none of its build
# output is fabricated anywhere in this topic's notes.

# --- Base image ------------------------------------------------------------
# Pinned to an exact tag (not `python:3.11` or, worse, `python:latest`).
# An unpinned tag is a moving target: "python:3.11" today might resolve to
# a different underlying minor version/patch tomorrow, silently changing
# what "works on my machine" even means. `slim` keeps the base layer small
# (no build toolchains, docs, etc. that a full image ships).
FROM python:3.11.9-slim-bookworm

# Metadata -- who/what this image is, useful when many images accumulate
# in a registry.
LABEL maintainer="ml-course" \
      description="Minimal Flask + scikit-learn model-serving app (Iris classifier)"

# --- Working directory -------------------------------------------------
WORKDIR /app

# --- Dependency layer (deliberately BEFORE copying the rest of the code) ---
# This ordering is the entire point of layer caching (see notes.md
# "Conceptual foundation" and the from-scratch build_cache.py demo): the
# requirements.txt file changes far less often than app.py during normal
# development. By copying and installing dependencies in their own layer
# ahead of the application code, a code-only change (the common case)
# reuses this layer's cache entirely and skips re-running `pip install`
# -- exactly the "change a LATE layer, EARLY layers stay cached" scenario
# demonstrated in build_cache.py's Build 3.
COPY requirements.txt .

# --no-cache-dir: don't let pip's own download cache bloat this layer
# (there is no benefit to a package cache inside a throwaway build layer).
RUN pip install --no-cache-dir -r requirements.txt

# --- Application layer ---------------------------------------------------
# Only now do we copy the application code and the pre-trained model
# artifact. A change to app.py or model.pkl invalidates ONLY this layer
# and everything after it -- the base image and dependency layers above
# stay cached (Build 3 in build_cache.py's Experiment demonstrates this
# exact cascade in isolation).
COPY app.py .
COPY model.pkl .

# --- Runtime user ---------------------------------------------------------
# Run as a non-root user inside the container. Defense in depth: if the
# app or one of its dependencies is compromised, the attacker doesn't
# automatically get root inside the container (and, on some
# misconfigurations, root-equivalent access on the host).
RUN useradd --no-create-home --uid 1000 appuser
USER appuser

# --- Network + entrypoint -------------------------------------------------
# Documents (does not itself publish) which port the app listens on --
# actual host<->container port mapping happens at `docker run -p` time.
EXPOSE 5000

# Container-native healthcheck: lets `docker ps` and orchestrators (e.g.
# Kubernetes liveness probes built on the same idea) know if the process
# is up but not actually serving.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:5000/health', timeout=2)" || exit 1

# Exec form (not shell form) so the Flask process is PID 1 and receives
# signals (SIGTERM on `docker stop`) directly instead of through a shell.
CMD ["python", "app.py"]
