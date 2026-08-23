# 02 – Git & GitHub

## What you'll learn

Why version control exists at all — coordinating changes to code and this course's own content across time and collaborators without overwriting each other's work or losing history — and how Git actually does it under the hood: a content-addressable graph of hashed, immutable objects (blobs, trees, commits), the same core idea `01-docker` uses for image layers. Covers `git init`/`add`/`commit`/`log`, branching and merging, and deliberately creating and resolving a real merge conflict.

| Topic | Status |
|-------|--------|
| Git Basics (init, add, commit, push) | ✅ Complete |
| Git Merge, Checkout, Log | ✅ Complete |
| Resolving Merge Conflicts | ✅ Complete |

## Why it matters

Every topic in this course, and every real ML codebase, lives in Git. Understanding *why* Git behaves the way it does (why merges usually just work, why they occasionally can't, why force-pushing is dangerous, why deleting a committed secret doesn't actually remove it from history) requires understanding the content-addressable object model underneath the porcelain commands — not just memorizing which command to type.

## Prerequisites

- `01-docker` — its "Conceptual foundation" (content-addressed, layered image storage) is the direct conceptual precursor to this topic's content-addressed object model; read in order.
- Basic command-line familiarity (`01-python-foundation`).

## What you'll build

- `mini_blob_store.py` — a real, executed, from-scratch Python implementation of a minimal content-addressable blob store (hash a string, store it under `.mini-git/objects/<hash>`, retrieve by hash), demonstrating the core primitive every Git object is built from, including a real deduplication and integrity-check demonstration.
- A real `git init` → `add` → `commit` → `log` walkthrough, actually run in a scratch directory outside this repository's own history, with genuine captured terminal output.
- A real merge conflict, deliberately created by two branches editing the same line of the same file, then resolved by hand — with the actual conflict markers and resolution output captured, not narrated.

## Where it appears in real systems

- Every collaborative codebase, including this one.
- CI/CD pipelines (`08-mlops-deployment/07-cicd`, planned) trigger directly off Git events (a push, a pull request).
- Model/data versioning tools (DVC, `08-mlops-deployment/03`–`04`, planned) extend the same content-addressing idea to large binary artifacts that don't belong directly in a Git repository.

## What's next

`03-testing-ci` (planned) — testing and continuous integration workflows that trigger off the Git commit graph this topic covers.
