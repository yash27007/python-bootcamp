# File Handling & Exception Management

## Problem

Almost any program that touches the outside world — reading a file, writing a config, hitting the
network, acquiring a lock — has an *acquire* step and a *release* step: open the file, eventually
close it; take the lock, eventually release it. Between those two steps, something can go wrong —
a bad line of data, a divide-by-zero, a missing key. The question this topic answers: how does a
program guarantee the release step still happens when the code in between raises, without the
programmer having to remember to write that guarantee correctly at every single place a resource
is used?

## Intuition

Picture borrowing a library book. The "happy path" is: take the book, read it, return it. But what
if you drop the book, or it turns out to be the wrong book, partway through reading it? "Return the
book" as the very last line of your evening's plan never runs if something interrupts you before
you get there — the book stays checked out under your name indefinitely. A librarian who says "no
matter what happens while you have it, bring it back to the front desk before you leave the
building" has moved the guarantee to a *rule about leaving*, not a *step in your plan* — it now
holds even when the plan itself falls apart.

`finally` is that librarian's rule applied to code: whatever happens inside a `try` block —
success, a caught exception, an uncaught exception propagating past this function entirely — the
`finally` block runs before control actually leaves. `with` is the same guarantee, pre-packaged:
instead of writing `try`/`finally` correctly around every single file/lock/connection by hand, a
`with` block delegates "acquire, and guarantee release" to an object that knows how to do both.

## Why simpler approaches fail

**"Just remember to call `.close()` at the end of the function."** This works exactly as long as
nothing between "open" and "close" ever raises — which is precisely the situation exceptions exist
to handle. The moment an exception is raised between acquire and the hand-written release line,
that release line is skipped: control jumps straight past it to the nearest matching `except` (or,
if there isn't one, out of the function entirely). Reproduced for real below: a resource opened,
then an operation on it raises, with `.close()` written as the function's last line — the resource
is provably still "open" afterward.

Writing `try`/`finally` by hand around every acquire/release pair *does* fix this — but it has to
be written correctly, every time, at every call site that touches the resource. It's easy to get
right once and easy to forget the tenth time, especially once a function has several early
returns or several resources open at once. The fix that removes the "remember to do this
correctly, every time" burden is to put the guarantee inside a reusable object instead of
repeating it at every call site — a context manager.

## Conceptual foundation

*(Substituting for "Mathematical foundation" — this topic has no derivation; the underlying
mechanism to make explicit is the context-manager protocol.)*

**The context-manager protocol** is two methods, `__enter__` and `__exit__`, and the `with`
statement is syntax that calls them at fixed points:

```
with EXPR as NAME:
    BODY
```

desugars, roughly, to:

```
manager = EXPR
NAME = manager.__enter__()
try:
    BODY
finally:
    manager.__exit__(exc_type, exc_value, traceback)
```

- `__enter__()` runs once, before `BODY`, and its return value becomes `NAME`.
- `__exit__(exc_type, exc_value, traceback)` runs once, after `BODY`, **unconditionally** — whether
  `BODY` completed normally (all three arguments `None`) or raised (the exception's type, value,
  and traceback are passed in). This is exactly the `finally` guarantee, just relocated from "a
  block the caller has to remember to write" to "a method the object always provides."
- `__exit__`'s return value matters: `True` (or any truthy value) tells `with` to *suppress* the
  exception — swallow it, as if `BODY` had completed normally; `False`/`None` lets it propagate
  after `__exit__` finishes. This is the one piece of behavior a hand-written `try`/`finally`
  doesn't give for free — `finally` never suppresses, it only guarantees the cleanup runs.

`open()` implements exactly this protocol: it returns a file object whose `__enter__` returns
`self` and whose `__exit__` calls `.close()` and returns a falsy value (never suppresses). That's
the entire mechanism behind "always use `with open(...)`."

## Algorithm

Building the context-manager protocol from a resource that needs guaranteed cleanup:

1. Define `acquire()`/`release()` (or `open()`/`close()`) on the resource.
2. **Naive:** call `acquire()`, do work, call `release()` as the last line — broken the moment the
   work raises.
3. **Manual fix:** wrap the work in `try: ... finally: release()` — correct, but has to be
   rewritten at every call site.
4. **Reusable fix — the context-manager class:** move `acquire()` into `__enter__`, move
   `release()` into `__exit__`, return `False` from `__exit__` to keep exceptions propagating.
   Every caller now just writes `with Resource(...) as r:` and gets step 3's guarantee for free,
   without rewriting it.

## From-scratch implementation

All three stages of the Algorithm above, built and run against a real observable resource-leak
check (`open_resources`, a list a stand-in "resource" appends to on open and removes itself from
on close). See [`file_exception.ipynb`](file_exception.ipynb), section "0. From-scratch: what
`with` automates."

**Stage 1 — the leak, reproduced for real** (`.close()` as the unprotected last line):

```
opened naive.txt
caught: something went wrong while using naive.txt
leaked resources after naive_use: ['naive.txt']
```

**Stage 2 — the manual fix** (`try`/`finally`, correct but call-site-local):

```
opened safe.txt
closed safe.txt
caught: something went wrong while using safe.txt
leaked resources after safe_use: []
```

**Stage 3 — the reusable fix** (a hand-rolled `__enter__`/`__exit__` class, used with `with`):

```
opened cm.txt
closed cm.txt
caught: something went wrong while using cm.txt
leaked resources after the `with` block: []
```

Stages 2 and 3 both close the resource correctly even though the exception still propagates out of
the `with`/`try` block (each is caught one level up, by an outer `except RuntimeError`) — the
guarantee holds independent of whether the exception is ultimately caught. Stage 3 is the one that
generalizes: the guarantee now lives once, inside `ManagedResourceCM`, instead of being retyped at
every place the resource is used.

## Practical implementation

The full practical notebook — [`file_exception.ipynb`](file_exception.ipynb) — covers, with real
executed examples: reading/writing text files (`.read()`, line iteration, `.readlines()`, file
modes), CSV via `csv.DictWriter`/`DictReader`, JSON via `json.dump`/`json.load`,
`try`/`except`/`else`/`finally` mechanics, catching multiple exception types, the built-in
exception hierarchy (`ValueError`, `TypeError`, `IndexError`, `KeyError`, `AttributeError`,
`FileNotFoundError`, `ZeroDivisionError`, `OverflowError`), catching specific exceptions before
general ones, `raise`-ing exceptions for invalid input, custom exception classes (subclassing
`Exception`/`ValueError`), and `contextlib.contextmanager` — which generates the same
`__enter__`/`__exit__` pair from a single generator function instead of a full class. This maps
directly back to the Conceptual foundation: every `with open(...)` in the notebook is running the
exact `__enter__`/`__exit__` sequence built by hand in section 0, and `@contextmanager` is a
second, more concise way to produce that same pair.

## Experiment

This topic's "experiment" is the from-scratch reproduction itself, structured as a controlled
before/after: the *same* stand-in resource, the *same* failing operation, run three times under
three different cleanup strategies (none, manual `try`/`finally`, context-manager class), with
`open_resources` as the observable outcome. The result (naive: leaked; both fixes: not leaked) is
pasted verbatim above rather than asserted, and it directly demonstrates the claim that `with`
provides no *new* capability over hand-written `try`/`finally` — only reusability, since stages 2
and 3 arrive at an identical outcome.

## Failure modes

- **Swallowing exceptions silently — bare `except:`.** A bare `except:` (or a reflexive
  `except Exception:`) catches everything, including bugs unrelated to the situation the handler
  was written for. Reproduced for real: `bad_average` is written to guard only against
  `ZeroDivisionError` on an empty list, but a bare `except:` also silently catches a `TypeError`
  from genuinely malformed input (a string mixed into a list of numbers) and returns the same `0`
  as the legitimate empty-list case:

  ```
  bad_average([]) = 0
  bad_average([10, '20', 30]) = 0
  ```

  Restricting the handler to the specific exception it's meant for lets the real bug surface
  instead of being reported as a plausible-looking wrong answer:

  ```
  good_average([]) = 0
  good_average([10, '20', 30]) raised (as it should): unsupported operand type(s) for +: 'int' and 'str'
  ```

  This is strictly worse than a crash — a crash is loud and gets fixed; a silently wrong `0` looks
  like valid output and can propagate through the rest of a pipeline unnoticed.
- **Not closing resources on the error path.** Exactly the Stage-1 failure in the From-scratch
  section above: `.close()`/`.release()` written as an unprotected last line is skipped whenever
  an exception is raised earlier in the same function — a leak that only manifests as a resource
  count creeping upward over time (too many open file handles, an exhausted connection pool), not
  as an immediate, obvious crash at the point of the mistake.

## Real-world usage

- **File and database I/O** almost universally use `with` (`open(...)`, database cursors,
  `tempfile.TemporaryDirectory`) specifically for the guarantee derived here — resources release
  even when the code using them raises partway through.
- **Locking in concurrent code** (`threading.Lock`, covered later in
  `01-python-foundation/10-multithreading`) is a context manager for exactly this reason: a lock
  acquired but never released on an exception path deadlocks every other thread waiting on it — a
  far more severe consequence than a leaked file handle.
- **Validating input at API/pipeline boundaries** relies on `raise`-ing specific, well-named
  exceptions (custom exception classes, as in `ModelNotTrainedError`/`DataValidationError` in the
  practical notebook) so that a caller several layers up can catch precisely the failure it knows
  how to handle, rather than an undifferentiated "something went wrong."
- **Logging in `except` blocks**, covered next in `01-python-foundation/09-logging`, exists
  specifically to avoid the silent-swallowing failure mode above at production scale — a caught
  exception that's neither re-raised nor logged leaves no trace that anything went wrong at all.

## Mental model

**`finally` (and the `with` statement built on top of it) moves a guarantee from "a step in the
plan, which is skipped if something earlier in the plan fails" to "a rule that holds regardless of
how the plan ends."** A hand-written `try`/`finally` gives that guarantee once, at one call site;
a context-manager class gives it once, permanently, to every call site that uses `with
Resource(...)`. Catching an exception is a separate decision from cleaning up after one — conflate
them (a bare `except:` that both hides the bug *and* implicitly "handles" cleanup) and both jobs
get done badly.

## Questions to think about

1. Stage 3's `__exit__` returns `False`. What would change, concretely, in the `with
   ManagedResourceCM("cm.txt") as r: r.do_work(should_fail=True)` demo if `__exit__` returned
   `True` instead — walk through what the surrounding `try`/`except RuntimeError` would see.
2. `bad_average([10, "20", 30])` returns `0` instead of raising. Why is this a worse outcome for
   whoever calls `bad_average` later than `bad_average([])` legitimately returning `0` — what
   information has been lost, and how would you notice the bug without reading `bad_average`'s
   source?
3. `contextlib.contextmanager`'s `timer` function in the practical notebook uses `try: yield
   finally: ...` internally. Using the `with EXPR as NAME: BODY` desugaring from the Conceptual
   foundation, explain what part of that desugaring the generator-based `@contextmanager`
   decorator is producing automatically, compared to writing a full `__enter__`/`__exit__` class
   by hand.
4. The Failure modes section separates "swallowing exceptions silently" from "not closing
   resources on the error path" as two different failure modes, but a single bare `except: pass`
   wrapped around resource-using code can cause both at once. Construct that combined case and
   explain why it's harder to detect than either failure occurring alone.
5. `open_resources` in the from-scratch cells is a plain Python list used purely to make a leak
   observable in a notebook. In a real long-running program (a web server handling thousands of
   requests), what would an actual, unnoticed version of the Stage-1 leak look like from the
   outside, before anyone reads the code closely enough to find the missing `finally`?
</content>
