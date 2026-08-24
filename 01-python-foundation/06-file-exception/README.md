# 06 – File Handling & Exception Management

Detailed notes (the acquire/release problem, the context-manager protocol derived from the `with`
statement's desugaring, a manual `try`/`finally` fix built and measured before the reusable
`__enter__`/`__exit__` class): [notes.md](notes.md)

Real, actually-executed notebook covering text/CSV/JSON file I/O, `try`/`except`/`else`/`finally`,
the built-in exception hierarchy, custom exceptions, `contextlib.contextmanager`, a from-scratch
resource-leak reproduction (naive → manual `try`/`finally` → hand-rolled context-manager class),
and a reproduced bare-`except:` silent-swallowing bug, all with real pasted output:
[file_exception.ipynb](file_exception.ipynb)

## What you'll learn

Why "just remember to close it" fails the moment an exception happens between acquire and
release, and what `with` automates: the context-manager protocol (`__enter__`/`__exit__`),
derived from the `with` statement's own desugaring and rebuilt by hand before `with open(...)`
is used at all.

| Topic | Status |
|-------|--------|
| Text/CSV/JSON file I/O | ✅ Complete |
| `try`/`except`/`else`/`finally`, exception hierarchy, custom exceptions | ✅ Complete |
| Conceptual foundation: the context-manager protocol derived from `with`'s desugaring | ✅ Complete |
| From-scratch: naive leak → manual `try`/`finally` → hand-rolled `__enter__`/`__exit__` class | ✅ Complete |
| Experiment: resource-leak observed/fixed across all three stages, real pasted output | ✅ Complete |
| Failure modes: bare `except:` silently swallowing a real bug, unclosed resource on error path | ✅ Complete |

## Why it matters

Resource cleanup that only runs on the happy path is a real, common bug — a leaked file handle,
an unreleased lock, an exhausted connection pool — that doesn't crash loudly; it degrades quietly
over time. Understanding *why* `with` guarantees cleanup (it's calling `__exit__`
unconditionally, the same guarantee `finally` gives, just moved into a reusable object) makes it
obvious when hand-written cleanup code is missing that guarantee.

## Prerequisites

`01-basics`/`02-control-flow` (functions, `try`/`except` needs conditionals) and `05-modules-packages`
(the `csv`/`json` standard-library modules used throughout).

## What you'll build

- A naive resource wrapper that provably leaks when an exception is raised mid-use (an
  `open_resources` list tracks what's still "open").
- The manual `try`/`finally` fix for the same resource, verified to close correctly even as the
  exception still propagates.
- A hand-rolled context-manager class (`__enter__`/`__exit__`) wrapping the same fix into a
  reusable form — used with `with`, before `with open(...)` appears anywhere else in the notebook.
- A reproduced bare-`except:` bug: a function meant to guard against one specific error silently
  returns a wrong-looking-valid answer for an unrelated real bug instead of raising it.

## Where it appears in real systems

File and database I/O (`with open(...)`, DB cursors), locking in concurrent code
(`threading.Lock`, covered in `10-multithreading` — the same protocol, higher stakes since an
unreleased lock deadlocks other threads), input validation via custom exception hierarchies at
API/pipeline boundaries, and logging inside `except` blocks (`09-logging`) as the production fix
for the silent-swallowing failure mode covered here.

## What's next

`07-oops` — bundling state with the functions that operate on it (encapsulation), building
directly on this topic's `__enter__`/`__exit__` classes as a first taste of custom classes with
real behavior.
</content>
