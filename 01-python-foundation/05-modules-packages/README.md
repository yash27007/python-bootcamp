# 05 – Modules & Packages

Detailed notes (the import system derived step-by-step — locate, read, execute in a fresh
namespace, cache — a from-scratch manual import, circular-import and stdlib-shadowing failures
reproduced): [notes.md](notes.md)

Real, actually-executed notebook covering `import` styles, the `__name__ == '__main__'` guard,
the standard library tour (`os`/`pathlib`, `math`/`random`, `datetime`, `json`, `re`,
`collections`, `itertools`), a from-scratch manual module loader compared against `importlib`, a
namespace-isolation demo, a `sys.path`/`sys.modules` caching demo, and reproduced circular-import
and stdlib-shadowing failures, all with real pasted output:
[modules_packages.ipynb](modules_packages.ipynb)

## What you'll learn

Why splitting code across files needs more than "just put it in another file" — namespace
isolation, and what `import` actually does underneath (`sys.path` search → read → execute into a
fresh namespace → cache in `sys.modules`), derived and then rebuilt by hand with `exec()`.

| Topic | Status |
|-------|--------|
| `import` styles: whole-module, aliased, specific-name, wildcard | ✅ Complete |
| `__name__ == '__main__'` guard | ✅ Complete |
| Standard library: `os`/`pathlib`, `math`/`random`, `datetime`, `json`, `re`, `collections`, `itertools` | ✅ Complete |
| Conceptual foundation: the import system derived (locate → read → execute → cache) | ✅ Complete |
| From-scratch: manual module loader (`exec()` into a fresh namespace), vs. `importlib` | ✅ Complete |
| Experiment: namespace-isolation and `sys.modules`-caching demonstrated directly | ✅ Complete |
| Failure modes: circular imports, stdlib name shadowing — both reproduced for real | ✅ Complete |

## Why it matters

Every non-trivial Python project is organized as modules and packages, and the standard library
itself is nothing but modules discovered through the exact `sys.path` mechanism covered here.
Understanding *why* two modules' names don't collide (namespace isolation) and *why* a module's
top-level code runs only once (`sys.modules` caching) is what makes circular imports and
stdlib-shadowing bugs predictable instead of mysterious.

## Prerequisites

`01-basics` and `02-control-flow` (functions, `if`/`for`, used throughout the standard-library
tour and the from-scratch loader).

## What you'll build

- A minimal manual module loader — read a `.py` file's text, `exec()` it into a fresh namespace
  dict — verified against `importlib`'s real mechanism on the identical file.
- A namespace-isolation demo: two files that both define `X` don't collide when imported
  separately, but do collide (silently) when `exec()`-ed into one shared namespace.
- A `sys.path`/`sys.modules` caching demo showing a module's top-level code runs once, not on
  every `import`.
- A reproduced circular-import `ImportError` (two real `.py` files that import from each other).
- A reproduced stdlib-shadowing failure — a local `random.py` silently shadowing the real
  `random` module, breaking `random.randint(...)` at first use.

## Where it appears in real systems

Every Python project's structure (this repo included); the standard library itself, discovered via
`sys.path` like any other module; virtual environments (`uv`, `venv`), which work by manipulating
`sys.path`; and `sys.modules` caching, which is why a library's import-time side effects (opening
a connection, loading a config) only ever happen once per process.

## What's next

`06-file-exception` — resource cleanup and error handling, building on functions/control flow the
same way this topic built on them for code organization.
</content>
