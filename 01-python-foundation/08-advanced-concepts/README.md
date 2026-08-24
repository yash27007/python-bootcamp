# Iterators and Generators

## What you'll learn

How Python's `for` loop actually works underneath — the iterator protocol
(`__iter__`/`__next__`) — and how generators (`yield`) automate that protocol so a sequence can be
produced one value at a time instead of built entirely in memory upfront.

## Why it matters

Every large or unbounded sequence a program touches — a huge file, a streamed dataset, an infinite
counter — cannot be fully materialized into a list first. Understanding the iterator protocol is
what makes it possible to reason about *why* `for line in file:` doesn't load the whole file, and
to build the same lazy behavior into custom code.

## Prerequisites

- `01-basics` (names, mutability), `04-functions` (functions, first-class functions, closures —
  the same "function that keeps state across calls" mechanism generators build on)

## What you'll build

- A manual class-based iterator (`__iter__`/`__next__`, no `yield`) proven identical in behavior
  to the equivalent generator function
- A real, measured memory/time comparison between a full list and a generator over 10 million
  items
- A demonstration of generator exhaustion (why a generator can only be iterated once)

See [`notes.md`](notes.md) for the full write-up, [`manual_iterator_protocol.py`](manual_iterator_protocol.py)
and [`memory_comparison_demo.py`](memory_comparison_demo.py) for the from-scratch demos, and
[`iterators_generators.ipynb`](iterators_generators.ipynb) for the practical notebook.

## Where it shows up in real systems

Streaming large datasets, database cursors, paginated APIs, and `06-deep-learning`'s
`DataLoader`-style batch generation all rely on this same lazy, on-demand production of values.

## What's next

`09-logging` — a different kind of "what happened, when" tracking: not iterating over data, but
recording what a running program did.
