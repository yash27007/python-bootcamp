# Basics — Syntax, Variables, Types, Operators

> **Note on template substitution:** this is a software-engineering foundations topic, not a
> numerical-modeling one — there is no loss function or derivative to derive here. Per this
> `AGENTS.md`'s content standards, the "Mathematical foundation" section below is replaced with a **Conceptual
> foundation** section that plays the same structural role: it precisely defines the mechanism
> (Python's name-binding / object model) that the rest of this document's examples and demos rest
> on. This substitution is documented here inline, as required.

## Problem

A calculator evaluates one fixed expression and forgets it. A single hard-coded script computes
one fixed sequence of operations on one fixed input. Neither can: remember an intermediate result
under a name and reuse it later, represent different *kinds* of data (a name, a price, a
true/false flag) without the programmer manually tracking what each raw byte pattern means, or
adapt what it computes based on values only known once the program is running.

A general-purpose language solves this with three primitives, all present from the first line of
code in this topic's notebook: **names** (variables) that can be bound to a value and reused,
**types** (int, float, str, bool, None, ...) that give a value a well-defined set of operations
and a well-defined meaning, and **operators** that combine values of those types into new values.
Every later topic in this curriculum — control flow, functions, data structures, and everything
built on top of them — is these three primitives, composed.

## Intuition

Sticky notes on a shelf, not labeled boxes. `age = 25` doesn't create a box called `age` and pour
`25` into it — it writes `age` on a sticky note and presses that note onto the `25` object sitting
somewhere in memory (Python already had an `int` object worth `25`, or created one). `age = 30`
later doesn't overwrite anything inside a box — it peels the sticky note off `25` and presses it
onto a *different* object, `30`. The `25` object doesn't change; the note just moved.

This single picture explains several things that look unrelated at first: why `type(x)` can change
across two lines even though `x` is "the same variable" (dynamic typing — the note just moved to
an object of a different type), why `x is y` can be `True` even though `x` and `y` were "assigned
separately" (two notes on the same object), and why mutating a list through one name can be seen
through another name pointing at the same list (the object changed; every note pointing at it sees
the change, because there is only one object).

## Why simpler approaches fail

**"Just track everything as raw values, no types."** Every operation would need the programmer to
remember, by hand, what a value's bit pattern means — is this `4` the number 4, or the character
`'4'`, or one bit of a flags field? C's `void*` and untyped assembly work exactly this way, and the
result is a well-known category of bugs (treating a string's bytes as a number, or vice versa) that
a type system exists specifically to prevent. Python's dynamic type system tags every object with
its type at creation and enforces the tag on every operation (`"a" + 1` raises `TypeError`
immediately, rather than silently reinterpreting bytes).

**"Just copy values on assignment, always."** This is the box model from the Intuition section —
it's simpler to explain, but it's not how Python works, and pretending it is produces wrong
predictions the moment a mutable object and two names both reference it (see Experiment below).
Some languages *do* copy-by-value by default (C's structs, or explicitly-cloned values elsewhere)
— but that has its own cost: copying a large object on every assignment is wasteful when the
common case doesn't need an independent copy, and it must be paired with an explicit "share this
without copying" mechanism (pointers/references) for the cases that do. Python instead makes
reference-by-default the *only* behavior and pushes the cost/benefit decision onto whether a type
is mutable or immutable (see Conceptual foundation).

## Conceptual foundation (substituting for Mathematical foundation)

**Every value in Python is an object.** Every object has three properties: an **identity** (a
fixed memory address, exposed by `id(x)`; never changes for the object's lifetime), a **type**
(fixed at creation, exposed by `type(x)`; determines which operations the object supports), and a
**value** (which may or may not be allowed to change — this is the mutable/immutable distinction).

**A name is a binding, not storage.** `x = <expr>` evaluates `<expr>` to get an object (existing or
newly created), then makes the name `x`, in the current namespace, point at that object. It never
copies the object, and it never modifies the target of any *other* name that happened to point at
the same object before the assignment — `x = ...` only ever affects what `x` itself points at.

**Immutable types** (`int`, `float`, `str`, `bool`, `tuple`, `frozenset`, `None`, ...): once
created, the object's value can never change. Every operation that looks like it "changes" an
immutable object (`x += 1`, `s.upper()`, `t + (4,)`) actually computes a brand-new object and
(for `+=`) rebinds the name to it, leaving the original object — and any other name still
pointing at it — untouched. This is why aliasing an immutable value (`b = a` where `a` is an int)
is always safe: no operation exists that could mutate the shared object out from under `b`.

**Mutable types** (`list`, `dict`, `set`, and user-defined objects by default): the object's value
*can* change after creation, in place, without creating a new object or changing its identity.
`list.append`, `list[i] = ...`, `dict[k] = ...`, and `set.add` are all in-place mutations — the
object at that `id()` is altered directly. Critically, **every name currently bound to that
object observes the mutation**, because there was only ever one object; the names are just labels
on it. This is why aliasing a mutable value (`b = a` where `a` is a list) is *not* automatically
safe — any mutation reachable through `b` is also visible through `a`, and vice versa, whether
that sharing was intended or not.

**Function calls bind parameters the same way `=` does.** Passing an argument to a function does
not copy it — the parameter name inside the function body is bound to the *same object* the
caller's argument name was bound to. This is why a function that mutates a mutable parameter
in place (rather than building and returning a new object) silently mutates the caller's object —
demonstrated for real in the Experiment section below.

## Algorithm

Not applicable in the algorithmic-steps sense — this topic is about a semantic model (how names,
objects, and assignment relate), not a procedure with steps. The "procedure" worth internalizing
as a checklist when reading any line of code that reassigns or mutates a name:

1. Is the right-hand side creating a new object, or referring to an existing one (a bare name, a
   function's return value that might be the same object it was given)?
2. Is the left-hand side a plain name (`x = ...`, a rebind — only affects `x`) or an in-place
   mutation (`x[i] = ...`, `x.append(...)`, — affects the object, visible through every name
   pointing at it)?
3. If in doubt, check `id(x)` before and after: unchanged `id` means the object was mutated in
   place; changed `id` means `x` was rebound to a different object.

## From-scratch implementation

Python's actual name-binding behavior is a language primitive, not something implemented in
Python code that can be reimplemented from scratch in the usual sense. Instead, the notebook
demonstrates the *wrong* mental model (a `Box` class that deep-copies its value on assignment,
simulating "variable = independent box") side by side with real Python's actual behavior on the
identical operations, so the difference is directly visible rather than asserted. See
[`basics.ipynb`](basics.ipynb), the "Names as references, not boxes" section, cells simulating the
box model followed immediately by the same operations in real Python.

Real executed output from that comparison:

```
box model  -> box_a.value: [1, 2, 3]  box_b.value: [1, 2, 3, 4]
real Python -> real_a: [1, 2, 3, 4]  real_b: [1, 2, 3, 4]
```

Under the box model, `box_a` (simulating the original list) is untouched after "copying" it into
`box_b` and mutating `box_b`. Under real Python's actual reference semantics, `real_a` changes too
— confirming the box model gives the wrong prediction the moment a mutable object and two names
share it.

## Practical implementation

The full practical notebook — [`basics.ipynb`](basics.ipynb) — covers, with real executed
examples: comments and docstrings, variable assignment (including multiple assignment, swap,
augmented assignment), dynamic typing, all core numeric/text/boolean/`None` types (including
arbitrary-precision integers, IEEE-754 float gotchas like `0.1 + 0.2 != 0.3`, raw/multi-line
strings), implicit and explicit type casting (and the `ValueError`/`TypeError`s that come from
casting failures), the full arithmetic/comparison/logical/membership operator set, chained
comparisons, short-circuit evaluation, identity (`is`) vs. equality (`==`), and string formatting
(f-strings, `.format()`, legacy `%`-formatting). The name-binding demos above extend that notebook
directly — same file, same executed-inline discipline.

## Experiment

**Hypothesis (stated before running):** a function that receives a mutable list argument and
mutates it in place (instead of building and returning a new list) will silently change the
caller's original list — because the function's parameter is bound to the same object the caller
passed, not a copy — even though nothing in the call site (`tag_names_BUGGY(original_roster)`)
looks like it should touch `original_roster` directly.

**Setup:** `tag_names_BUGGY` loops over its `names` parameter and reassigns each element in place
(`names[i] = names[i] + suffix`), intending (incorrectly) to build a new tagged list.

**Actual result (real executed output):**

```
before call: ['alice', 'bob', 'carol']
tagged:          ['alice_reviewed', 'bob_reviewed', 'carol_reviewed']
original_roster: ['alice_reviewed', 'bob_reviewed', 'carol_reviewed']
same object?  True
```

Confirmed: `original_roster` was mutated by a call that never referenced it after the call
started, and `tagged is original_roster` is `True` — there was only ever one list object.
`tag_names_FIXED`, which builds and returns a genuinely new list via a list comprehension, was
run on the same input and left `original_roster` unchanged (`same object? False`), confirming the
fix.

**Interpretation:** the bug is not "list operations are broken" — it's that the programmer's
mental model (box model: "the caller's list is safe because I'm working inside a function") was
wrong for a mutable argument. The fix is always the same shape: build a new object and return it,
rather than mutating the one you were handed, whenever the caller's object should be left alone.

**Limitations:** this experiment demonstrates the failure with a list; the same reasoning applies
identically to `dict`, `set`, and any mutable custom object, but the specific mutating
operation differs by type — the pattern to watch for is any in-place method or item assignment on
an object you did not create inside the current scope.

## Failure modes

- **Aliasing bugs (demonstrated above)** — mutating a shared mutable object through one reference
  silently affects every other reference to it. Most common where an object is passed into a
  function, stored in another structure, or assigned to a second name "for convenience," and the
  code later mutates it in place rather than building a new value.
- **Mutable default arguments (demonstrated in the notebook)** — a default argument value is
  evaluated exactly **once**, at `def` time, not per call. `def add_item(item, basket=[])` creates
  one `[]` object that is reused as the default for *every* call that omits `basket` — mutating it
  in one call leaves the accumulated state visible on the next call. Real executed proof:
  `add_item_BUGGY('apple')` → `['apple']`, then `add_item_BUGGY('banana')` → `['apple', 'banana']`
  — the second call's "default" list already contained the first call's data. Fix: default to
  `None` and construct the mutable object fresh inside the function body (`if basket is None:
  basket = []`), which was verified to give `['apple']` then `['banana']` — independent per call.
- **Confusing `==` and `is`.** `==` asks "do these objects have equal *value*"; `is` asks "are
  these the exact same object." `a == b` can be `True` while `a is b` is `False` (equal-valued but
  distinct lists) — using `is` where `==` was intended silently breaks comparisons for any type
  where equal-valued objects are commonly distinct objects (almost every mutable type, and even
  some immutable ones outside small-integer/short-string caching ranges).
- **Floating-point comparison.** `0.1 + 0.2 == 0.3` is `False` (IEEE-754 double precision cannot
  represent `0.1`, `0.2`, or `0.3` exactly) — comparing floats for exact equality after arithmetic
  is a reliable source of "impossible" bugs; round or use a tolerance instead.

## Real-world usage

- **API/library design**: whether a function mutates its argument or returns a new object is a
  first-class design decision documented in real libraries' docstrings — e.g. `list.sort()`
  mutates in place and returns `None` (a deliberate signal: "don't chain this expecting the
  sorted list"), while `sorted(list)` returns a new list and leaves the original untouched. Both
  exist because both are sometimes the right tool, and confusing them is a common real bug.
- **Default mutable arguments** are a real, common code-review finding — linters (`pylint`,
  `ruff`) flag `def f(x=[])` by default specifically because of the failure mode demonstrated
  above; this is not a theoretical gotcha.
- **Concurrency and shared mutable state**: the same aliasing mechanism that causes accidental
  bugs in single-threaded code is also the *entire reason* multithreaded code needs locks — two
  threads holding references to the same mutable object can race to mutate it (this curriculum's
  `10-multithreading` topic reproduces exactly this class of bug deliberately, with a real race
  condition, before showing `threading.Lock`).
- **Deep copy vs. shallow copy** (`copy.copy` / `copy.deepcopy`) exist in the standard library
  specifically to let a programmer *opt into* the "independent copy" behavior the box model
  assumes by default, for the cases where sharing genuinely isn't wanted.

## Mental model

**A name is a sticky note, not a box — and whether the object it points at can be mutated in
place through any note pointing at it is the single fact that predicts every aliasing surprise in
Python.** Before writing `b = a`, ask: is `a`'s object mutable? If yes, `a` and `b` now share one
object, and any in-place change through either name is visible through both. If no, sharing is
free and safe — the only way `a` and `b` diverge is if one of them is later rebound to a different
object entirely, which never touches the other.

## Questions to think about

1. `a = [1, 2]; b = a; a = a + [3]` — does `b` end up as `[1, 2]` or `[1, 2, 3]`? Walk through
   which operation (`+`) creates a new object versus which name gets rebound, using the sticky-note
   model, then verify with `id()`.
2. Why does `tag_names_BUGGY` (mutating in a loop with `names[i] = ...`) count as an in-place
   mutation of the *same* list object, while `tag_names_FIXED`'s list comprehension does not, even
   though both eventually produce a list of tagged strings? What's the object-identity difference?
3. A colleague argues `def f(basket=[])`'s bug "doesn't matter" if the function is only ever called
   without a caller relying on accumulation across calls. Construct a concrete two-call sequence
   where this argument fails even for a caller who never intended to share state.
4. Tuples are immutable, but `t = ([1, 2], 3)` is a tuple containing a mutable list. Is `t` itself
   safe to alias (`t2 = t`) the way an int is? What can and can't change through `t2` if so — and
   why doesn't tuple immutability prevent it?
5. `is` compares identity, `==` compares value. Two separately-created strings with identical
   content are sometimes `is`-equal in Python (CPython interns some strings/small ints) and
   sometimes not. Why is relying on `is` for value equality risky even when it happens to "work" in
   a quick test?
