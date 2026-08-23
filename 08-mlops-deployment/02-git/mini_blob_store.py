"""
mini_blob_store.py -- a tiny, real content-addressable blob store.

This is NOT Git. It demonstrates the core primitive every Git object
(blob, tree, commit) is built from: content is hashed, and the hash
becomes the object's permanent address. Store and retrieve are both keyed
purely by the hash of the content -- never by a filename or a path. This
is exactly what `git hash-object -w` does for a blob (Git additionally
prefixes the content with a header like "blob <size>\\0" before hashing,
and zlib-compresses it on disk; both are omitted here to keep the
mechanism visible, and are noted, not reimplemented).

Run directly: `.venv/bin/python mini_blob_store.py`
"""

from __future__ import annotations

import hashlib
from pathlib import Path

STORE_DIR = Path(__file__).parent / ".mini-git" / "objects"


def hash_content(content: str) -> str:
    """Git hashes a header-prefixed version of the content; we hash the
    raw bytes directly to keep the mechanism minimal and visible."""
    return hashlib.sha256(content.encode()).hexdigest()


def store(content: str) -> str:
    """Write `content` under its own hash, if not already stored.
    Returns the hash (the object's permanent address)."""
    digest = hash_content(content)
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    obj_path = STORE_DIR / digest
    if obj_path.exists():
        # Content-addressing means storing identical content twice is a
        # no-op -- this is *why* Git dedupes identical file content across
        # every commit in history for free, with zero extra bookkeeping.
        return digest
    obj_path.write_text(content)
    return digest


def retrieve(digest: str) -> str:
    """Look content up purely by its hash -- there is no other index."""
    obj_path = STORE_DIR / digest
    if not obj_path.exists():
        raise KeyError(f"no object with hash {digest}")
    return obj_path.read_text()


def verify_integrity(digest: str) -> bool:
    """Re-hash the stored content and confirm it still matches its own
    address. If the file on disk were tampered with, this fails --
    exactly how `git fsck` detects corruption."""
    content = retrieve(digest)
    return hash_content(content) == digest


if __name__ == "__main__":
    print("=== Storing two different blobs ===")
    h1 = store("def train_model():\n    return model.fit(X, y)\n")
    h2 = store("def train_model():\n    return model.fit(X, y, sample_weight=w)\n")
    print(f"blob 1 hash: {h1}")
    print(f"blob 2 hash: {h2}")
    print(f"hashes differ (different content): {h1 != h2}\n")

    print("=== Storing the SAME content twice (deduplication) ===")
    h3 = store("def train_model():\n    return model.fit(X, y)\n")
    print(f"blob 3 hash: {h3}")
    print(f"h1 == h3 (identical content -> identical, deduped address): {h1 == h3}")
    files_on_disk = sorted(p.name for p in STORE_DIR.glob("*"))
    print(f"objects actually on disk ({len(files_on_disk)}): {files_on_disk}\n")

    print("=== Retrieving by hash alone ===")
    print(f"retrieve(h1) ->\n{retrieve(h1)!r}\n")

    print("=== Changing content by even one character changes the address ===")
    h4 = store("def train_model():\n    return model.fit(X, y)")  # no trailing newline
    print(f"blob 4 hash (no trailing newline): {h4}")
    print(f"h1 != h4 (one-byte diff -> completely different hash): {h1 != h4}\n")

    print("=== Integrity check ===")
    print(f"verify_integrity(h1): {verify_integrity(h1)}")
