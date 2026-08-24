# 01 – Basics

Detailed notes (name-binding/object model, mutable vs. immutable, aliasing, Conceptual-foundation
substitution documented inline): [notes.md](notes.md)

Real, actually-executed notebook covering syntax and semantics, variables, all core data types,
operators, string formatting, the reference-semantics demos, a real aliasing-bug experiment, and
the mutable-default-argument failure mode, all with real pasted output: [basics.ipynb](basics.ipynb)

## What you'll learn

What problem a general-purpose language's names/types/operators solve that a calculator or a fixed
script can't (remembering and reusing intermediate results, representing different kinds of data,
adapting to runtime-only-known values). Python's actual name-binding model — a name is a reference
to an object, not a labeled box holding a value — and why that model, not "assignment copies
things," correctly predicts every aliasing surprise involving mutable objects.

| Topic | Status |
|-------|--------|
| Syntax & semantics, comments, docstrings | ✅ Complete |
| Variables, multiple assignment, augmented assignment | ✅ Complete |
| Data types: int/float/str/bool/None, casting | ✅ Complete |
| Operators: arithmetic, comparison, logical, membership, identity | ✅ Complete |
| String formatting: f-strings, `.format()`, `%` | ✅ Complete |
| Conceptual foundation: names as references, mutable vs. immutable | ✅ Complete |
| From-scratch: box-model simulation vs. real reference semantics | ✅ Complete |
| Experiment: real aliasing bug (buggy vs. fixed) | ✅ Complete |
| Failure mode: mutable default arguments (buggy vs. fixed) | ✅ Complete |

## Why it matters

Almost every "impossible" bug a Python beginner hits traces back to a wrong mental model of what
`=` does: expecting an assignment or a function call to copy a mutable value, when Python instead
shares the same object across every name pointing at it. Getting this right early avoids a whole
category of aliasing bugs later, and it's the same mechanism (shared mutable state across
references) that makes multithreaded code need locks — this topic builds the foundation
`10-multithreading` later builds on directly.

## Prerequisites

None — this is the first topic in the curriculum. No external dependencies; standard library only.

## What you'll build

- A side-by-side comparison of a simulated "box model" (deep-copying `Box` class) against real
  Python's actual reference semantics, on identical operations, with real executed output showing
  where they diverge.
- A real, reproduced aliasing bug: a function that unintentionally mutates its caller's list
  in place, followed by the fixed version that returns a genuinely new list.
- A real, reproduced mutable-default-argument bug (`def f(x=[])`), with real executed output
  showing state silently accumulating across unrelated calls, followed by the `None`-sentinel fix.

## Where it appears in real systems

Library API design decisions (does a method mutate in place and return `None`, like `list.sort()`,
or return a new object, like `sorted()`) are exactly this distinction made explicit. Static
analysis tools (`pylint`, `ruff`) flag mutable default arguments by default because this is a real,
common bug, not a theoretical one. The same aliasing mechanism is the root cause of race conditions
in concurrent code, covered later in `10-multithreading`.

## What's next

`02-control-flow` — conditionals and loops, the mechanisms that let one program body adapt its
behavior to runtime data rather than executing one fixed sequence of statements.
