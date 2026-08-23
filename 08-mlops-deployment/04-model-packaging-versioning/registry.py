"""
A tiny, from-scratch "model registry" — the minimal version of what tools
like MLflow's Model Registry (see notes.md "Practical implementation") do
in production: content-address a serialized model, attach metadata, and
keep an append-only log so every registered version is discoverable and
distinguishable, regardless of what filename the caller asked to save it
under.

Core idea: don't trust a human-chosen filename ("model_v2_final.pkl") to
uniquely identify a model. Hash the serialized bytes instead — identical
models always get the same hash, and any two models that differ by even
one bit get different hashes, no matter what name was requested.
"""

from __future__ import annotations

import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy
import sklearn

REGISTRY_DIR = Path(__file__).parent / "registry"
LOG_PATH = REGISTRY_DIR / "registry_log.json"


def hash_file(path: Path) -> str:
    """SHA-256 of a file's raw bytes, read in chunks (safe for large model files)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _library_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "joblib": joblib.__version__,
        "numpy": numpy.__version__,
    }


def load_log() -> list[dict[str, Any]]:
    if LOG_PATH.exists():
        return json.loads(LOG_PATH.read_text())
    return []


def _append_log(entry: dict[str, Any]) -> None:
    log = load_log()
    log.append(entry)
    LOG_PATH.write_text(json.dumps(log, indent=2))


def register(model: Any, requested_name: str, metric_name: str, metric_value: float,
             extra_metadata: dict[str, Any] | None = None) -> dict[str, Any]:
    """Serialize `model` with joblib, content-address it, write a JSON metadata
    sidecar, and append an entry to the append-only registry log.

    `requested_name` is exactly the filename the caller would naively want to
    save under (e.g. "model.pkl") — the registry deliberately does NOT use it
    as the storage key, precisely to demonstrate that filename collisions
    don't cause version collisions (see Experiment in notes.md).
    """
    REGISTRY_DIR.mkdir(exist_ok=True)

    # 1. Serialize to a temp path first — we need the bytes on disk to hash them.
    tmp_path = REGISTRY_DIR / f"_tmp_{requested_name}"
    joblib.dump(model, tmp_path)
    content_hash = hash_file(tmp_path)

    # 2. Rename to the content-addressed final path. If this exact model was
    #    already registered, this is a no-op overwrite of identical bytes.
    short_hash = content_hash[:12]
    final_model_path = REGISTRY_DIR / f"{short_hash}.pkl"
    tmp_path.replace(final_model_path)

    # 3. Write the metadata sidecar.
    metadata = {
        "requested_name": requested_name,
        "hash": content_hash,
        "short_hash": short_hash,
        "model_file": final_model_path.name,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "metric_name": metric_name,
        "metric_value": metric_value,
        "library_versions": _library_versions(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)

    sidecar_path = REGISTRY_DIR / f"{short_hash}.json"
    sidecar_path.write_text(json.dumps(metadata, indent=2))

    # 4. Append to the log — this is what makes the registry queryable without
    #    needing to know a hash in advance.
    _append_log(metadata)
    return metadata


def load_model_by_hash(short_hash: str) -> Any:
    """Load a registered model back by its (short) content hash, verifying
    integrity — the loaded bytes must still hash to the same value."""
    model_path = REGISTRY_DIR / f"{short_hash}.pkl"
    actual_hash = hash_file(model_path)
    if not actual_hash.startswith(short_hash):
        raise ValueError(
            f"integrity check failed: {model_path} now hashes to {actual_hash[:12]}, "
            f"expected prefix {short_hash} — file was modified after registration"
        )
    return joblib.load(model_path)
