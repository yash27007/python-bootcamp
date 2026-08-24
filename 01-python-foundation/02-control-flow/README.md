# 02 – Control Flow

Detailed notes (truthiness, short-circuit evaluation, dispatch-table from-scratch demo,
Conceptual-foundation substitution documented inline): [notes.md](notes.md)

Real, actually-executed notebook covering conditionals, ternaries, loops, comprehensions,
structural pattern matching, the dispatch-table demo, and both off-by-one and infinite-loop
failure-mode reproductions, all with real pasted output: [control_flow.ipynb](control_flow.ipynb)

## What you'll learn

Why a program that always runs the same fixed sequence of statements can only ever do one thing,
and why "write a separate script per case" doesn't scale (some cases aren't even known until
runtime — a loop count determined by data). Python's truthiness rules and short-circuit evaluation
as the concrete mechanisms every conditional and boolean expression rests on. What `if`/`elif`
automates under the hood (a sequential scan) by building the same case-dispatch logic by hand with
a dict-based dispatch table instead — zero `if`/`elif` statements, same output, $O(1)$ lookup
instead of an $O(n)$ scan.

| Topic | Status |
|-------|--------|
| Conditionals: `if`/`elif`/`else`, ternary expressions | ✅ Complete |
| Truthiness across every built-in type | ✅ Complete |
| Loops: `for`, `while`, `range`, `enumerate`, `zip`, nested loops | ✅ Complete |
| `break`/`continue`/`pass`, `while`/`else` | ✅ Complete |
| List/dict/set comprehensions, generator expressions | ✅ Complete |
| Structural pattern matching (`match`/`case`) | ✅ Complete |
| Conceptual foundation: truthiness, short-circuit evaluation | ✅ Complete |
| From-scratch: dict-dispatch table replacing `if`/`elif` | ✅ Complete |
| Failure mode: off-by-one loop errors (real executed demo) | ✅ Complete |
| Failure mode: infinite loop (real executed, safety-capped demo) | ✅ Complete |

## Why it matters

Every branching or repeating decision a real program makes — grading logic, retry loops,
pagination, request routing — is built from exactly the primitives in this topic. Off-by-one and
infinite-loop bugs are among the most common real bug categories in production code (pagination
boundaries, retry logic with no termination guarantee); reproducing both here, deliberately and
safely, builds the habit of checking for them before they ship.

## Prerequisites

- `01-basics` — variables, types, operators, and especially truthiness's reliance on the
  mutable/immutable and type-conversion rules covered there.
- Standard library only, no external dependencies.

## What you'll build

- A dict-based dispatch table (`HTTP_DISPATCH`) that replicates an `if`/`elif` chain's output
  exactly, verified with an assertion across every test input, demonstrating the scan-vs-lookup
  tradeoff `if`/`elif` and dict-dispatch each make.
- A real off-by-one bug: `range(1, 10)` silently producing 9 elements instead of the intended 10,
  compared side by side with the corrected `range(1, 11)`, plus a manual-indexing variant showing
  the same failure shape.
- A real infinite loop: a `while` loop whose condition never becomes false, run to a
  1,000,000-iteration safety cap to prove it, followed by the actual fix (updating the loop
  variable inside the body) terminating normally after 5 iterations.

## Where it appears in real systems

Web framework URL routing (Flask, Django, FastAPI) is a dispatch table exactly like the one built
from scratch here, covered explicitly in this curriculum's `12-flask` topic. State machines
(order processing, connection lifecycles) are commonly implemented the same way. Off-by-one bugs
in pagination (`offset`/`limit`) and infinite/runaway retry loops with no termination guarantee are
among the most frequently cited real-world bug categories in code review and postmortems.

## What's next

`03-data-structures` — the containers (`list`, `dict`, `set`, `tuple`) that control-flow loops most
often iterate over, including the real Big-O complexity analysis of their operations.
