# Memory Management

## What you'll learn

How CPython decides when an object's memory can be reclaimed — reference counting as the primary,
always-on mechanism — and the specific case that mechanism cannot handle on its own: a reference
cycle, where two objects reference only each other. That gap is why CPython also runs a separate,
periodic cycle-detecting garbage collector (`gc`).

## Why it matters

A long-running program that leaks memory eventually gets killed by the OS, or degrades until it
does. Reference counting frees most objects deterministically and immediately, but a program that
creates reference cycles (parent/child back-references, doubly-linked structures, observer
patterns) needs the cycle collector too — understanding exactly why is what lets you diagnose a
real leak instead of guessing at it.

## Prerequisites

- `07-oops` (`__init__`, `__del__`, and object attributes — used directly in the cycle demo)
- `08-advanced-concepts` (generators as the memory-efficient alternative to building a full list,
  referenced in this topic's Practical implementation)

## What you'll build

- A real, observed reference-counting walkthrough (`sys.getrefcount` going up and down as names are
  bound and deleted)
- A real reference cycle, actually created and actually collected — refcounts shown NOT reaching
  zero after `del`, and `gc.collect()` shown freeing the objects (`__del__` firing) only once it
  runs

See [`notes.md`](notes.md) for the full write-up including real captured output,
[`refcount_and_cycle_demo.py`](refcount_and_cycle_demo.py) for the from-scratch demo, and
[`memory_management.ipynb`](memory_management.ipynb) (all cells executed) for the practical tour —
`gc` module controls, `gc.get_stats()`, `gc.garbage`, generators for memory efficiency, and
`tracemalloc` for real allocation profiling.

## Where it shows up in real systems

Every long-running Python service (web servers, training loops, notebook kernels left open for
days) depends on this working correctly. `tracemalloc` and `objgraph` are the standard tools for
diagnosing a real production leak. `10-multithreading`'s GIL exists partly because refcount
increments/decrements are not thread-safe on their own.

## What's next

`12-flask` — building a web server, one of the long-running-process categories where memory
management (and the leaks this topic demonstrates how to diagnose) actually matters in practice.
