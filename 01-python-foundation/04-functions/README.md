# 04 – Functions

Detailed notes (LEGB scope resolution, closures as a live-variable-capture mechanism, decorator
desugaring, Conceptual-foundation substitution documented inline): [notes.md](notes.md)

Real, actually-executed notebook covering function definition and calling, parameter types
(default, `*args`, `**kwargs`), the mutable-default-argument bug, LEGB scope and
`global`/`nonlocal`, a real reproduced late-binding closure bug (and two fixes), a hand-desugared
decorator (built without `@`, then rewritten with `@`), `lambda`/`map`/`filter`/`reduce`, and
docstrings/type hints, all with real pasted output: [functions.ipynb](functions.ipynb)

## What you'll learn

Why functions exist (reuse and abstraction, not just "avoiding repetition"), Python's exact name
resolution order (LEGB) for reading a name a function didn't assign locally, what a closure
actually captures (the enclosing *variable*, not a snapshot of its value — and why that one fact
explains both `make_counter`'s working state and the late-binding closure gotcha), and what
`@decorator` syntax literally desugars to (`func = decorator(func)`), built and verified by hand
before using the shorthand.

| Topic | Status |
|-------|--------|
| Function definition, calling, return values (including implicit `None`, tuple returns) | ✅ Complete |
| Parameters: positional, default, `*args`, `**kwargs`, combined | ✅ Complete |
| Mutable default argument bug (buggy vs. fixed) | ✅ Complete |
| Conceptual foundation: LEGB scope resolution, closures, first-class functions | ✅ Complete |
| `global` / `nonlocal` | ✅ Complete |
| Failure mode, reproduced: late-binding closures in a loop (2 real fixes) | ✅ Complete |
| From-scratch: manual `@decorator` desugaring, `functools.wraps` gotcha | ✅ Complete |
| `lambda`, `map`/`filter`/`functools.reduce`, docstrings, type hints | ✅ Complete |

## Why it matters

Closures and first-class functions are the mechanism underneath decorators, callbacks, and
memoization — three patterns that show up constantly in real frameworks (Flask routes, pytest
fixtures, `functools.lru_cache`). The late-binding closure bug demonstrated here is one of the
most common real bugs involving closures created inside a loop (event handlers, deferred
callbacks) — understanding *why* it happens (a shared variable, not independent values) is what
makes the fix obvious instead of magical.

## Prerequisites

`01-basics` (names as references — directly relevant to how closures capture variables, not
values) and `02-control-flow` (loops, used in the closure-gotcha demo).

## What you'll build

- A real reproduction of the late-binding closure bug: three lambdas created in a loop all
  returning the loop's final value (`[2, 2, 2]`, not `[0, 1, 2]`), fixed two independent ways
  (default-argument capture, and a factory function) and verified correct.
- A decorator built the long way first — an ordinary function that wraps another function, applied
  via a plain function call and name rebinding, no `@` syntax — then rewritten with `@` to confirm
  identical behavior, including the real `functools.wraps` gotcha (a wrapper silently shadowing the
  original function's `__name__` unless explicitly fixed).

## Where it appears in real systems

Decorators in real frameworks (Flask's `@app.route`, pytest's `@pytest.fixture`,
`functools.lru_cache`, `@property`/`@staticmethod`/`@classmethod`) are all this exact mechanism.
Callback-heavy and event-driven code is the most common real-world trigger for the late-binding
closure bug demonstrated here. Configuration/factory functions that return a closure
(`make_counter`-shaped code) are the standard way to build a family of related callables that each
carry their own private state without a full class.

## What's next

`05-modules-packages` — the import system and namespace isolation, extending this topic's scope
model (LEGB) to a fourth, module-boundary-crossing dimension.
