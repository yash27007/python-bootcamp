"""
build_cache.py -- a tiny, real content-hash-based build cache.

This is NOT Docker. It demonstrates the *mechanism* Docker's image layer
cache uses underneath `docker build`: a "build" is a sequence of steps run
in order; each step is hashed together with the hash of everything before
it (so a change to an early step invalidates every step after it, exactly
like Docker layers); if a step's combined hash matches a hash we've already
computed and cached, we skip re-running that step and reuse its cached
result instead of re-executing potentially expensive work.

Run directly: `.venv/bin/python build_cache.py`
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Callable

CACHE_DIR = Path(__file__).parent / ".build_cache"
CACHE_DIR.mkdir(exist_ok=True)


def _hash_step(previous_hash: str, step_name: str, step_content: str) -> str:
    """Hash a step's content together with the running hash of everything
    before it. This chaining is exactly why changing an EARLY layer forces
    every LATER layer to be considered changed too, even if their own
    content is byte-for-byte identical to last time: their hash INPUT
    (previous_hash) changed even though their own step_content didn't."""
    payload = f"{previous_hash}:{step_name}:{step_content}".encode()
    return hashlib.sha256(payload).hexdigest()


def _cache_path(layer_hash: str) -> Path:
    return CACHE_DIR / f"{layer_hash}.json"


def run_build(steps: list[tuple[str, str, Callable[[], str]]], verbose: bool = True) -> list[dict]:
    """
    steps: list of (step_name, step_content, run_fn)
      - step_name / step_content together stand in for a Dockerfile
        instruction, e.g. ("RUN pip install -r requirements.txt",
        <the file contents of requirements.txt>, <function that actually
        does the (simulated) work and returns a result string>).
      - run_fn is only CALLED when the step's hash is not already cached
        -- this is the "skip re-running unchanged steps" behaviour.

    Returns a log of what happened for each step (hit/miss, hash, timing).
    """
    running_hash = "root"
    log = []
    for step_name, step_content, run_fn in steps:
        layer_hash = _hash_step(running_hash, step_name, step_content)
        cache_file = _cache_path(layer_hash)

        start = time.perf_counter()
        if cache_file.exists():
            cached = json.loads(cache_file.read_text())
            elapsed = time.perf_counter() - start
            log.append({
                "step": step_name,
                "hash": layer_hash[:12],
                "cache": "HIT",
                "result": cached["result"],
                "seconds": round(elapsed, 4),
            })
            if verbose:
                print(f"[CACHE HIT ] {step_name!r:55s} hash={layer_hash[:12]} ({elapsed:.4f}s)")
        else:
            result = run_fn()
            elapsed = time.perf_counter() - start
            cache_file.write_text(json.dumps({"result": result}))
            log.append({
                "step": step_name,
                "hash": layer_hash[:12],
                "cache": "MISS",
                "result": result,
                "seconds": round(elapsed, 4),
            })
            if verbose:
                print(f"[CACHE MISS] {step_name!r:55s} hash={layer_hash[:12]} ({elapsed:.4f}s)")

        # Chain: the next step's hash depends on THIS step's hash, not just
        # its own content -- mirrors how Docker layer N's cache key depends
        # on the digest of layer N-1.
        running_hash = layer_hash

    return log


def clear_cache():
    for f in CACHE_DIR.glob("*.json"):
        f.unlink()


# --- simulated expensive build steps ------------------------------------

def _slow(seconds: float, label: str) -> Callable[[], str]:
    def _fn():
        time.sleep(seconds)
        return f"{label} done"
    return _fn


BASE_IMAGE = ("FROM python:3.11-slim", "python:3.11-slim", _slow(0.3, "pulled base image"))
COPY_REQS = ("COPY requirements.txt .", "flask==3.1.3\nscikit-learn==1.8.0\n", _slow(0.05, "copied requirements.txt"))
PIP_INSTALL = ("RUN pip install -r requirements.txt", "flask==3.1.3\nscikit-learn==1.8.0\n", _slow(1.0, "installed dependencies"))
COPY_CODE = ("COPY . .", "app.py contents v1", _slow(0.05, "copied app code"))


if __name__ == "__main__":
    print("=== Build 1: cold cache (everything is a MISS) ===")
    clear_cache()
    steps = [BASE_IMAGE, COPY_REQS, PIP_INSTALL, COPY_CODE]
    t0 = time.perf_counter()
    run_build(steps)
    print(f"Total: {time.perf_counter() - t0:.4f}s\n")

    print("=== Build 2: rerun with NO changes (everything should HIT) ===")
    t0 = time.perf_counter()
    run_build(steps)
    print(f"Total: {time.perf_counter() - t0:.4f}s\n")

    print("=== Build 3: change LATE layer only (app code changes) ===")
    steps_late_change = [
        BASE_IMAGE,
        COPY_REQS,
        PIP_INSTALL,
        ("COPY . .", "app.py contents v2 -- changed!", _slow(0.05, "copied app code")),
    ]
    t0 = time.perf_counter()
    run_build(steps_late_change)
    print(f"Total: {time.perf_counter() - t0:.4f}s\n")

    print("=== Build 4: change EARLY layer only (requirements.txt changes) ===")
    steps_early_change = [
        BASE_IMAGE,
        ("COPY requirements.txt .", "flask==3.1.3\nscikit-learn==1.8.0\nnumpy==2.1.0\n", _slow(0.05, "copied requirements.txt")),
        ("RUN pip install -r requirements.txt", "flask==3.1.3\nscikit-learn==1.8.0\nnumpy==2.1.0\n", _slow(1.0, "installed dependencies")),
        ("COPY . .", "app.py contents v2 -- changed!", _slow(0.05, "copied app code")),
    ]
    t0 = time.perf_counter()
    run_build(steps_early_change)
    print(f"Total: {time.perf_counter() - t0:.4f}s\n")
