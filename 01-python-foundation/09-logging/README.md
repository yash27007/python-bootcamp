# Logging

## What you'll learn

Why `print()` debugging stops working once a program runs unattended, and how Python's `logging`
module solves it with severity levels, named loggers, handlers, and formatters that compose
together.

## Why it matters

Production programs run without a debugger attached. The log a program writes while running is
often the only evidence available after something goes wrong — and a program that logs everything
at the same, unfiltered severity is nearly as hard to diagnose from as a program that logs nothing
at all.

## Prerequisites

- `06-file-exception` (file handling, exception handling — logging routinely captures caught
  exceptions and writes to files)

## What you'll build

- A real comparison of the same log calls at two different levels, proving level filtering changes
  output with zero call-site edits
- A multi-handler, multi-destination logger (console + file) with a real logged division-by-zero
  error

See [`notes.md`](notes.md) for the full write-up, [`level_toggle_demo.py`](level_toggle_demo.py)
for the level-filtering demo, [`app.py`](app.py) for a full worked module, and
[`main.ipynb`](main.ipynb) / [`logs/`](logs/) for further practical examples.

## Where it shows up in real systems

Web services, ML training runs, and MLOps pipelines all depend on structured, leveled logging to
diagnose failures without direct access to a running process — see `12-flask`,
`06-deep-learning`, and `08-mlops-deployment`.

## What's next

`10-multithreading` — doing multiple things at once from one program, and the very real
correctness bugs (race conditions) that arise when threads share state without coordination.
