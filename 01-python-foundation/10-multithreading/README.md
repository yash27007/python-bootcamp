# Multithreading

## What you'll learn

How to do multiple things "at once" from one Python program using threads, why the Global
Interpreter Lock (GIL) means that only helps for I/O-bound work, and why sharing mutable state
across threads without a lock is a real, reproducible correctness bug — not a theoretical one.

## Why it matters

Programs that wait on I/O (network requests, disk reads) waste real time if they wait
sequentially. Threading reclaims that time by overlapping the *waiting* — but the same threads
that make this possible can also silently corrupt shared state if two of them touch it at the same
time without coordination. Recognizing and fixing that failure mode is one of the most important
correctness skills in concurrent programming.

## Prerequisites

- `06-file-exception` (context managers — `with lock:` guarantees release exactly like `with
  open(...)` guarantees a file gets closed)
- `11-memory-management` (reference counting — the reason the GIL exists in the first place)

## What you'll build

- A real, observed, non-deterministic race condition — a shared counter corrupted by unsynchronized
  threads, actually run and captured — fixed with `threading.Lock` to show it's now always correct
- A real deadlock (two locks acquired in opposite orders) and its fix (a single global acquisition
  order)

See [`notes.md`](notes.md) for the full write-up including the real race-condition output,
[`race-condition-demo.py`](race-condition-demo.py) and [`deadlock_demo.py`](deadlock_demo.py) for
the from-scratch demos, and the existing `.py` files (`multi-threading.py`,
`multi-processing.py`, `advanced-multi-threading.py`, `advanced-multi-processing.py`,
`usecase-multi-threading.py`, `usecase-multi-processing.py`) for the practical I/O-bound and
CPU-bound patterns.

## Where it shows up in real systems

Web servers handling concurrent requests, producer-consumer pipelines, database connection pools,
and `08-mlops-deployment`'s serving layers all depend on getting this right — and on avoiding the
exact race-condition and deadlock failure modes demonstrated in this topic.

## What's next

`11-memory-management` — how Python actually manages the memory threads are sharing: reference
counting, and why cyclic references need a separate garbage collector.
