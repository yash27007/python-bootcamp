# Functions — Reuse, Scope, Closures, Decorators

> **Note on template substitution:** like `01-basics`, this is a software-engineering foundations
> topic, not a numerical-modeling one. Per this section's plan
> (`docs/superpowers/plans/2026-08-24-phase7-foundations-retrofit.md`, Global Constraints), the
> "Mathematical foundation" section below is replaced with a **Conceptual foundation** section —
> Python's scope-resolution and closure mechanism — which plays the same structural role: it's the
> precise mechanism the rest of this document's examples and demos rest on. Documented here inline,
> as required.

## Problem

Without functions, any piece of logic used more than once must be either copy-pasted everywhere
it's needed (a maintenance hazard — fixing a bug means finding and fixing every copy) or inlined
into one giant script that can't be reasoned about or tested in pieces. Functions solve two
separate problems at once: **reuse** (write the logic once, call it by name anywhere) and
**abstraction** (a caller only needs to know a function's name, inputs, and outputs — not how it's
implemented — to use it correctly). Both are necessary for a codebase to grow past a few dozen
lines without becoming unmanageable.

## Intuition

A function is a labeled, reusable *recipe*, not a single-use calculation. `def square(x): return
x ** 2` doesn't compute a number — it records a procedure, under the name `square`, that can be
handed any `x` later and will compute `x ** 2` for that specific `x`. Calling `square(5)` doesn't
re-type the recipe; it *runs* the recorded recipe with `x` bound to `5`.

Two extra ideas build on top of that recipe picture and matter for everything downstream in this
topic:

- **Closures**: a function defined *inside* another function can "remember" variables from the
  outer function's scope even after the outer function has finished running — the inner function
  carries a reference to those variables along with it, not a snapshot of their values at some
  fixed moment. This is why `make_counter()` in the practical notebook can hand back a `counter`
  function that keeps incrementing `n` across separate calls, long after `make_counter` itself has
  returned.
- **First-class functions**: in Python, a function is a value like any other — it can be assigned
  to a variable, stored in a list or dict, passed as an argument, and returned from another
  function. `@decorator` syntax (see From-scratch implementation) and `map`/`filter`/`sort(key=...)`
  all rely on exactly this — a function being something that can be *passed around*, not just
  called by its own fixed name.

## Why simpler approaches fail

**"Just copy-paste the code wherever it's needed."** Works once. The moment the logic needs a bug
fix or a behavior change, every copy must be found and updated identically — a linear, error-prone
task that grows with the number of copies, and one that silently fails whenever a copy is missed
(the codebase now has several subtly different versions of "the same" logic, no longer actually
the same).

**"Just use global variables instead of parameters/closures."** A function that reads and writes
globals instead of taking parameters and returning values *appears* simpler at first (fewer things
to pass around explicitly) but couples every caller to the exact current value of shared global
state — two calls to the same function can now produce different results depending on what else
ran in between, calling the function from a different part of the program can have invisible
side effects elsewhere, and testing it in isolation requires reconstructing the entire global
state first. Parameters and return values make a function's dependencies and effects explicit and
callable independently of everything else; closures give a narrower, deliberate version of "shared
state across calls" (see `make_counter` above) — scoped to exactly the variables the closure
chose to capture, not the entire global namespace.

## Conceptual foundation (substituting for Mathematical foundation)

**The LEGB rule — how Python resolves a bare name.** When code inside a function reads a name that
isn't a parameter and wasn't assigned in that function's own body, Python searches outward through
exactly four scopes, in this fixed order, and uses the *first* match:

1. **L — Local**: names assigned anywhere in the current function's body (including the
   parameters).
2. **E — Enclosing**: names in any enclosing function's local scope, for a nested function — the
   scope of the function that *textually* contains this one, not the scope of whatever function
   happened to call it.
3. **G — Global**: names assigned at module level (top of the `.py` file / notebook).
4. **B — Built-in**: names Python itself provides (`len`, `print`, `range`, ...).

Demonstrated for real in the notebook's `outer`/`inner` example: `inner` prints its own local `x`
("local"), then `outer` — with its nested `inner` finished running — prints its own `x`
("enclosing"), and the module-level `print("global:", x)` prints the outermost `x` ("global") —
three different bindings of the same name `x`, resolved correctly at each of the three call sites
by walking L → E → G in order.

**Assignment inside a function is local by default — and that's why `global`/`nonlocal` exist.**
If a function's body contains `x = ...` anywhere, Python treats `x` as local to that function for
its *entire* body (even before that line runs) — it will not silently fall through to the
enclosing/global `x` for reads either. `global x` and `nonlocal x` are explicit opt-outs from this
default: they tell Python "when this function assigns to `x`, mutate the global/enclosing binding,
don't create a new local one." Without `global count` in the notebook's `increment()` example,
`count += 1` would raise `UnboundLocalError` rather than silently reading the outer `count`,
because the assignment inside the function makes Python treat `count` as local for that whole
function body.

**A closure is a function plus the enclosing-scope variables it actually reads/writes, kept
alive.** Normally, a function's local variables are discarded once it returns. If a nested function
defined inside it is *returned* (or otherwise escapes, e.g. stored in a list), and that nested
function references a variable from the outer function, Python keeps that specific variable alive
— attached to the returned function object — for as long as the returned function exists. Critically,
what's captured is the *variable itself* (a mutable cell Python can still write to via `nonlocal`),
not a copy of whatever value it held at definition time. This single fact is both why
`make_counter()`'s returned `counter` function can keep incrementing `n` across calls, and why the
late-binding closure gotcha below (several closures over the same loop variable) happens — every
closure created in the loop shares the *one* loop variable, not an independent snapshot per
iteration.

## Algorithm

Not applicable in the numerical sense — the "procedure" worth internalizing here is name
resolution, as a checklist for reading any function body that references a name it didn't
assign locally:

1. Is the name assigned anywhere in the current function's body? If yes, it's Local for the
   *whole* function (reads before the assignment line raise `UnboundLocalError`, they don't fall
   through) — unless the function explicitly declared `global`/`nonlocal` for that name.
2. If not Local, is this function nested inside another function that assigned this name? If yes,
   Enclosing.
3. If not, was the name assigned at module level? If yes, Global.
4. If not, is it a Python builtin? If yes, Built-in. If none of the four match, `NameError`.

## From-scratch implementation

`@decorator` syntax above a `def` is sugar for one specific rewrite. Built the long way first (no
`@` at all — an ordinary function call that rebinds a name), then rewritten with `@` to confirm
they behave identically. See [`functions.ipynb`](functions.ipynb), sections "7. From-scratch: what
`@decorator` desugars to."

Real executed output, manual version (no `@`):

```python
def timing_decorator(func):
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

slow_add = timing_decorator(slow_add)   # <- the entire thing @ automates
print(slow_add(2, 3))
```
```
slow_add took 0.0101s
manual desugared call: 5
```

Rewritten with `@` — identical behavior, confirmed by real output:

```
slow_multiply took 0.0101s
'@' syntax call:       20
```

`@timing_decorator` above `def slow_multiply(...):` is exactly `slow_multiply =
timing_decorator(slow_multiply)`, evaluated immediately after the `def` — nothing more.

**A real gotcha this from-scratch build surfaces**: the returned `wrapper` shadows the original
function's identity. Real executed proof: `slow_multiply.__name__` is `'wrapper'`, not
`'slow_multiply'`, after decoration — confirmed in the notebook. `functools.wraps` fixes this by
copying `__name__`/`__doc__`/etc. from the original function onto the wrapper; with
`@wraps(func)` added, `slow_divide.__name__` correctly prints `'slow_divide'`.

## Practical implementation

The full practical notebook — [`functions.ipynb`](functions.ipynb) — covers, with real executed
examples: basic function definition and calling, implicit `None` return, multiple return values
via tuple unpacking, positional/default/`*args`/`**kwargs` parameters (and combining all of them
in one signature), the mutable-default-argument bug and its `None`-sentinel fix, the LEGB scope
demo, `global`/`nonlocal`, `lambda`, `map`/`filter`/`functools.reduce` (and list-comprehension
equivalents), and docstrings/type hints. This maps back to the Conceptual foundation directly:
every scope example is LEGB in action, and the decorator section's `wrapper` closure is a concrete
instance of "a nested function keeping an enclosing variable (`func`) alive after the outer call
finished."

## Experiment

**Hypothesis (stated before running):** creating several closures inside a `for` loop, each one
capturing the *loop variable* `i` by reference (not its value at creation time), will make every
one of those closures return the loop's *final* value of `i` when called later — not the distinct
value each closure "looked like" it was capturing at its own iteration — because all of them share
one enclosing variable, and the loop has already finished running by the time any closure is
called.

**Setup:** `buggy_funcs = [lambda: i for i in range(3)]` builds three closures inside a list
comprehension (which, like a `for` loop, shares one `i` across iterations); each closure is then
called after the comprehension has finished.

**Actual result (real executed output):**

```
buggy (all see final i): [2, 2, 2]
fixed via default arg:   [0, 1, 2]
fixed via factory func:  [0, 1, 2]
```

The buggy version returns `[2, 2, 2]`, not the naively expected `[0, 1, 2]` — every closure reads
the *same* `i`, and by the time any of them is called, the comprehension has finished with `i == 2`.
Two independent fixes were verified against this same failure: (1) the default-argument trick
(`lambda i=i: i`) works because default argument *values* are evaluated once, at `def`/`lambda`
creation time, capturing that iteration's value of `i` into a fresh parameter rather than
referencing the shared outer `i`; (2) a factory function (`make_returner(value)`) works because
each call creates a genuinely new local scope with its own `value`, not a shared one.

**Interpretation:** the bug is not "closures are broken" — it's that a closure captures a
*variable*, not a value, exactly as stated in the Conceptual foundation. The fix is always the
same shape: force the value to be captured into something that won't keep changing after the
closure is created (a fresh default argument, or a fresh function-call scope) rather than a
variable the surrounding loop keeps reassigning.

**Limitations:** demonstrated with `lambda` in a list comprehension; the identical failure occurs
with `def`-defined closures inside an ordinary `for` loop, and with any mutable loop variable, not
just an integer counter — the shape of the bug (shared variable, not shared value) is unrelated to
which of those forms is used.

## Failure modes

- **Late-binding closures in loops (demonstrated above)** — the single most common closure-related
  bug in Python. Any time closures are created inside a loop (`lambda`s in a list, event-handler
  callbacks registered in a loop, `functools.partial`-style deferred calls that reference a loop
  variable directly), every closure sees the loop's *final* value, not the value at its own
  iteration, unless a fix like the default-argument trick or a factory function is used.
- **Mutable default arguments** — covered fully in `01-basics/notes.md`'s Failure modes section
  and reproduced again in this topic's notebook (`append_wrong` vs. `append_correct`): a default
  argument value is evaluated once, at `def` time, so a mutable default (`def f(x=[])`) is shared
  and silently accumulates state across unrelated calls.
- **`UnboundLocalError` from an unintended local.** Assigning to a name anywhere in a function body
  makes Python treat it as local for the *entire* function — so a function that reads a global
  variable and then, later in the same body, assigns to a variable of the same name (without
  declaring `global`) raises `UnboundLocalError` on the read, not a graceful fallback to the
  global value. This surprises anyone expecting Python to "notice" the read happens before the
  assignment.
- **Overusing `lambda` for anything beyond a single expression.** `lambda` can only contain one
  expression (no statements, no assignments) — reaching for nested/conditional lambdas to work
  around that produces code that's harder to read than the equivalent `def`, which is why the
  notebook's own comment favors list comprehensions over `map`/`filter` + `lambda` once the logic
  gets past a trivial one-liner.

## Real-world usage

- **Decorators in real frameworks**: Flask's `@app.route(...)`, pytest's `@pytest.fixture`,
  `functools.lru_cache`, and Python's own `@property`/`@staticmethod`/`@classmethod` are all this
  exact mechanism (a function wrapping another function, applied via `@`) — understanding the
  manual desugaring above is what makes framework decorator behavior (e.g. why a decorated
  function's `__name__` needs `functools.wraps` to survive) predictable instead of magical.
- **Callback-heavy code and event handlers**: GUI/async callback registration in a loop is exactly
  the shape that triggers the late-binding closure bug — this is a real, recurring class of bug in
  UI code and asynchronous job scheduling, not a Python trivia question.
- **Memoization/caching**: `functools.lru_cache` is a decorator whose wrapper closes over a cache
  dict, using closures precisely to keep private state alive across calls without exposing it as a
  global.
- **Configuration/factory functions**: `make_counter`-shaped code (a function that returns a
  configured function/closure) is the standard way to build a family of related callables that
  each carry their own private state, without needing a full class for each one.

## Mental model

**A function is a reusable recipe; a closure is that recipe plus a live reference to specific
ingredients from the kitchen it was written in — and "live reference" is the whole story: it sees
those ingredients' *current* state whenever it's actually used, not a snapshot from when the
recipe was written.** Before relying on a closure capturing a loop variable, ask: will this
closure be *called* after the loop has moved on? If yes, it will see the loop's current (possibly
final) value, not the value from "its" iteration — force a fresh binding (default argument or a
factory call) whenever that's not the intended behavior.

## Questions to think about

1. `increment()` in the notebook uses `global count`. What error would `count += 1` raise without
   that declaration, and why does the error happen even though `count` clearly exists at module
   level — walk through LEGB's local-detection rule to explain it.
2. `make_counter()` returns a `counter` closure that keeps incrementing `n` across separate calls.
   Why doesn't `n` get reset to `0` on each call to `counter()`, given that `n = 0` only appears
   once, inside `make_counter`, which has already returned?
3. The late-binding closure demo shows `[lambda: i for i in range(3)]` producing `[2, 2, 2]`
   instead of `[0, 1, 2]`. Would the same bug occur with a `for` loop appending closures to a list
   one at a time (rather than a list comprehension)? Why or why not?
4. `functools.wraps` fixes `wrapper.__name__` being `'wrapper'` instead of the original function's
   name. What's a concrete scenario where a decorated function's wrong `__name__`/`__doc__` would
   cause a real, user-visible problem (not just an aesthetic one) — think about tools that inspect
   functions by name.
5. Both the default-argument trick (`lambda i=i: i`) and the mutable-default-argument bug from
   `01-basics` rely on the exact same underlying fact about Python (default values are evaluated
   once, at `def` time). Explain why that one fact makes default arguments the *fix* for the
   closure-capture bug but the *cause* of the mutable-default bug — what's the difference between
   the two situations that flips it from fix to bug?
