# Control Flow — Conditionals and Loops

> **Note on template substitution:** this is a software-engineering foundations topic, not a
> numerical-modeling one — there is no loss function or derivative to derive here. Per this
> section's plan (`docs/superpowers/plans/2026-08-24-phase7-foundations-retrofit.md`, Global
> Constraints), the "Mathematical foundation" section below is replaced with a **Conceptual
> foundation** section that plays the same structural role: it precisely defines the mechanisms
> (truthiness, short-circuit evaluation) that the rest of this document's examples and the
> from-scratch dispatch table rest on. This substitution is documented here inline, as required.

## Problem

A program that always executes the exact same sequence of statements, on every run, regardless of
input, can only ever do one thing. Real programs need to do different things depending on data —
grade a score differently depending on its value, process a list whose length isn't known until
runtime, keep retrying an operation until it succeeds. Without a way to express "run this block
only if some condition holds" and "run this block some number of times determined by the data,"
every distinct case a program needs to handle would need its own separately written, separately
maintained program — and the number of cases a real program handles (every possible input value,
every possible list length) is usually unbounded. Control flow is what lets one program body cover
an unbounded set of concrete situations.

## Intuition

A recipe that says "add sugar" always adds sugar. A recipe that says "if the batter tastes bland,
add more sugar" adapts to a fact discovered while cooking — that's a conditional. A recipe that
says "knead the dough until it's smooth" repeats an action an unknown number of times, determined
by a condition observed each time — that's a loop. Both are ways of writing one instruction that
covers many concrete situations, rather than needing a separate recipe for "bland batter,"
"slightly bland batter," "very bland batter," and so on.

## Why simpler approaches fail

**"Write a separate script for every case."** For a grading function with 4 letter grades, this
means 4 separate scripts (or 4 separate hard-coded branches of logic duplicated across files) —
already unwieldy. For anything indexed by a value known only at runtime (loop over however many
items are in *this* user's shopping cart), it's not just unwieldy, it's impossible — there is no
way to know at write-time how many scripts to write, because the count depends on data the
program doesn't see until it runs. Control flow expressions don't just make branching/repetition
more convenient than writing separate scripts — for a runtime-determined repeat count, separate
scripts can't express the logic *at all*.

**"Duplicate the branch logic inline everywhere it's needed."** Even once a program has an
`if/elif` or a loop, copy-pasting the same conditional logic at every call site instead of writing
it once creates a maintenance liability: a rule change (e.g. the grade cutoff for a "B" moves from
80 to 82) now has to be found and updated at every duplicated copy, and missing even one produces
silently inconsistent behavior. This is the same "doesn't compose" failure at a smaller scale —
the fix (write the branching logic once, call it from everywhere it's needed) is exactly why
control flow belongs inside reusable functions, previewed here and made explicit in the next
topic (`03-data-structures`/`04-functions`).

## Conceptual foundation (substituting for Mathematical foundation)

**Truthiness.** Every `if`/`while` condition in Python is evaluated by converting its value to a
`bool` via `bool(value)`, even if the value isn't already a `bool`. Python defines this conversion
for every built-in type via a small, fixed rule set rather than requiring an explicit `== True`
everywhere: `0`, `0.0`, `None`, and every *empty* container (`""`, `[]`, `{}`, `set()`, `()`) are
**falsy**; every other value is **truthy** (including every nonzero number and every non-empty
container, regardless of its actual contents). This is why `if my_list:` is idiomatic Python for
"if the list is non-empty" rather than `if len(my_list) > 0:` — both give the same answer, but the
former is checking truthiness directly rather than routing through a length comparison.

**Short-circuit evaluation.** `and` and `or` do not always evaluate both operands. `a and b`:
if `a` is falsy, the result is `a` and `b` is **never evaluated**; only if `a` is truthy is `b`
evaluated, and the result is `b`. `a or b`: if `a` is truthy, the result is `a` and `b` is never
evaluated; only if `a` is falsy is `b` evaluated. This is not a performance detail — it changes
program *behavior*: `0 and 1/0` never raises `ZeroDivisionError`, because `1/0` is never reached
(`0` is falsy, so `and` returns `0` immediately). This is the mechanism behind a very common real
pattern, `value = user_input or default` — if `user_input` is falsy (e.g. `""` or `None`),
`default` is evaluated and returned; otherwise `user_input` itself is returned without `default`
ever being touched.

**Structural pattern matching (`match`/`case`)**, Python's newer addition, is a different kind of
conditional: rather than testing one boolean expression per branch, it tests whether a value's
*shape* (literal value, tuple structure, class structure, ...) matches a pattern, binding names
from the matched structure along the way. It composes with truthiness and short-circuiting the
same way `if` does for any guard condition inside a `case`, but its core mechanism — structural
matching — is distinct.

## Algorithm

The evaluation order Python actually follows (not a summary — this is what runs):

**`if`/`elif`/`else`:**
```
evaluate condition_1
if truthy: run block_1, skip everything else, done
else: evaluate condition_2
    if truthy: run block_2, skip everything else, done
    else: ...
if no condition was truthy and an else exists: run else block
```
Every condition is checked in order, top to bottom, and evaluation stops at the first truthy one
— later conditions (and their blocks) are never touched once one matches.

**`for` loop:** requests an iterator from the iterable (`iter(sequence)`), then repeatedly calls
`next()` on it, running the loop body once per value returned, until `next()` raises
`StopIteration`.

**`while` loop:** evaluate the condition; if truthy, run the body, then re-evaluate the condition;
repeat until the condition is falsy (or a `break` exits early). Nothing about a `while` loop
guarantees the condition will ever become falsy — that depends entirely on whether the body
changes something the condition depends on (see Failure modes: infinite loops).

## From-scratch implementation

Control flow's `if`/`elif` automates a **sequential scan**: check each condition in order, stop at
the first truthy one. To see what that scan is doing — and where it can be replaced — the notebook
implements the same case-dispatch task two ways: a normal `if`/`elif` chain, and a **dispatch
table** (a `dict` mapping each known input value directly to the function that should handle it,
with `.get(value, default_handler)` doing the lookup) that produces identical output with zero
`if`/`elif` statements. See [`control_flow.ipynb`](control_flow.ipynb), "From-scratch: a dispatch
table without `if`/`elif`."

Real executed output, confirming both approaches agree on every input:
```
200: if/elif -> 'OK'   dispatch -> 'OK'   match (both agree)
404: if/elif -> 'Not Found'   dispatch -> 'Not Found'   match (both agree)
503: if/elif -> 'Unknown'   dispatch -> 'Unknown'   match (both agree)
```

**What this reveals:** `if/elif` is a linear scan — worst case, $O(n)$ comparisons for $n$
branches, since Python checks them in written order and stops at the first match (or falls through
to `else` after checking all of them). A dict-based dispatch table replaces the scan with one hash
lookup — $O(1)$ regardless of branch count — but only for *exact-value* matches; it cannot express
a range condition (`status >= 500`) without extra logic to bucket the value first. This is exactly
why real URL routers and CLI subcommand dispatchers use dict-like structures keyed on an exact
route/command string rather than a giant `if/elif` chain, and why `match`'s `case _:` catch-all
still needs real conditional logic for anything beyond an exact literal.

## Practical implementation

The full practical notebook — [`control_flow.ipynb`](control_flow.ipynb) — covers, with real
executed examples: `if`/`elif`/`else` grading logic, ternary (conditional) expressions including
nested ternaries, truthy/falsy values across every built-in type, `for` loops over lists/strings/
dicts/`range`/`enumerate`/`zip`, nested loops, `while` loops (including a binary-search example and
`while`/`else`), `break`/`continue`/`pass`, list/dict/set comprehensions (including nested/flatten
and a memory-efficient generator expression), and structural pattern matching (`match`/`case`) on
both literal values and tuple structure. The dispatch-table and failure-mode demos below extend
that notebook directly.

## Experiment

**Hypothesis (stated before running):** `range(1, 10)` will *not* include `10` — because `range`'s
`stop` argument is exclusive — producing a 9-element sequence when 10 elements (`1` through `10`
inclusive) were intended; correcting `stop` to `11` will produce the intended 10-element sequence.

**Setup:** compute `list(range(1, 10))` (intended: 1 through 10) and `list(range(1, 11))` side by
side.

**Actual result (real executed output):**
```
buggy  (want 1..10): [1, 2, 3, 4, 5, 6, 7, 8, 9]  len = 9
fixed  (want 1..10): [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  len = 10
```
Confirmed: the buggy range silently produced 9 elements, missing `10` entirely, with no error —
this is precisely what makes off-by-one bugs dangerous: the code runs without crashing, and the
output looks superficially plausible unless the count is checked. A second variant in the notebook
confirmed the same failure shape for manual list indexing: `range(1, len(data))` skips index `0`
(the first element), and `range(0, len(data) - 1)` skips the last index — both real, common
mistakes when hand-indexing instead of iterating directly.

**Interpretation:** `range`'s exclusive `stop` is a consistent, documented rule (mirrors slice
semantics elsewhere in Python), not itself a bug — the bug is a mismatch between what the
programmer intends ("through 10 inclusive") and what they wrote (`stop=10` instead of `stop=11`).
The fix is mechanical once the rule is internalized: for an inclusive upper bound `N`, `stop` must
be `N + 1`.

**Limitations:** this experiment covers `range`'s exclusive-stop convention specifically; other
off-by-one shapes exist (fencepost errors in cumulative sums, `<` vs. `<=` in a manual loop
condition) that follow the same "boundary mismatch" root cause but aren't all directly `range`-
based — the mental check ("does my stop condition include or exclude the boundary I actually
want?") generalizes even though this specific demo doesn't cover every variant.

## Failure modes

- **Off-by-one errors (demonstrated above, real executed output)** — an exclusive-`stop` range or
  a manual index bound one off from what's intended silently produces a sequence missing (or
  including one extra of) its first or last element, with no error raised.
- **Infinite loops (demonstrated in the notebook, with a real safety-capped run)** — a `while`
  loop's condition depends on a value the loop body never updates, so the condition never becomes
  falsy. Real executed proof: a loop with `while n > 0:` and no `n -= 1` (or equivalent) inside the
  body ran to a 1,000,000-iteration safety cap with `n` still at its original value of `5`,
  proving it would never terminate on its own; adding `n -= 1` inside the body (the actual fix, not
  the safety cap) terminated normally after exactly 5 iterations. The safety cap in the demo is a
  defensive engineering pattern for *catching* this class of bug in production (a max-iteration or
  timeout guard) — it is not a substitute for the real fix, which is ensuring the loop body always
  moves the state toward the condition being false.
- **Dangling `elif` / missing `else` assumptions.** An `if/elif` chain with no final `else`
  silently does nothing when no condition matches — easy to miss if the writer assumed the listed
  conditions were exhaustive when they weren't (e.g. handling grades `90+`/`80+`/`70+` with no
  branch for anything below 70 leaves `grade` unset or stale from a previous iteration, rather than
  raising an error).
- **Relying on truthiness for a value that can legitimately be `0` or `""`.** `if count:` is `False`
  for `count = 0` even when `0` is a perfectly valid, meaningful value (e.g. "zero items in stock"
  is a real state, not "no count provided") — conflating "falsy" with "missing/invalid" causes
  valid zero/empty values to be silently treated as absent. The fix is an explicit check
  (`if count is not None:`) when zero/empty is a legitimate value distinct from "no value."

## Real-world usage

- **Request routing** in every web framework (Flask, Django, FastAPI) is a dispatch table exactly
  like the from-scratch demo above: a URL path maps to a handler function via a lookup structure,
  not a giant `if path == "/a": ... elif path == "/b": ...` chain — this curriculum's `12-flask`
  topic makes that mapping explicit.
- **State machines** (order processing, connection lifecycle handling, game logic) are commonly
  implemented as dict-of-functions dispatch tables keyed by current state, for the same $O(1)$-
  lookup and easy-to-extend reasons demonstrated above, rather than deeply nested `if/elif`.
- **Guard clauses and short-circuiting** are a standard defensive-programming pattern:
  `if obj is not None and obj.is_valid():` relies on short-circuit evaluation to avoid ever calling
  `.is_valid()` on `None` — this is not incidental, it's the mechanism that makes the guard clause
  safe.
- **Off-by-one and infinite-loop bugs** are consistently among the most common real-world bug
  categories in code review and postmortems — pagination logic (`offset`/`limit` boundaries),
  batch-processing loops, and retry logic (a `while` loop retrying an operation "until it
  succeeds," with no max-attempt fallback) are the most frequent concrete sites.

## Mental model

**Control flow is what lets a fixed program body cover an unbounded set of runtime situations —
conditionals cover unbounded *possible values*, loops cover unbounded *possible counts*.** Under
the hood, `if/elif` is a sequential scan through conditions (stop at first truthy), while a
dict-dispatch table replaces that scan with a single lookup when every case is an exact value —
know which one a given piece of code needs, since only one of them handles ranges/predicates.
Every `while` loop is a bet that its body will eventually make its condition false — check that
bet explicitly whenever writing one, because Python will never check it for you.

## Questions to think about

1. The dispatch-table demo showed `if/elif` and dict-dispatch producing identical output. Under
   what condition would they stop being interchangeable — i.e. what kind of branch condition can
   `if/elif` express that a plain dict-key lookup cannot?
2. `value = user_input or default` relies on short-circuit evaluation and on truthiness. Construct
   a concrete `user_input` value where this idiom gives a *wrong* result (hint: think about a
   legitimate, meaningful falsy input value from the Failure modes section).
3. The off-by-one experiment showed `range(1, 10)` producing 9 elements with no error. Why is a
   silent wrong-count bug like this more dangerous in practice than a bug that raises an exception
   immediately? What would you add to code using `range` to catch this class of mistake earlier?
4. The infinite-loop demo used a 1,000,000-iteration safety cap to make the bug observable without
   hanging the notebook. In a real production service, what's the practical difference between
   "the loop is a true infinite loop" (would never terminate under any circumstance) and "the loop
   terminates, but only after an unacceptably large number of iterations for realistic input"? Are
   both bugs, and would the same fix address both?
5. `while` `/`else` runs the `else` block only if the loop finished without hitting a `break`.
   Sketch a realistic scenario (e.g. searching for an item) where this is meaningfully different
   from just putting the same code immediately after the loop with no `else`.
