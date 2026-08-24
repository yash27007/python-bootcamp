# Modules & Packages

## Problem

A single `.py` script works fine until a project grows past "one file." At some point functions
written for one task turn out to be useful for another, a script gets long enough that scrolling
through it to find one function is itself a cost, and more than one person (or more than one
notebook) needs the same helper code without copy-pasting it. The question this topic answers:
how does a program spread its code across many files while still being able to *use* code defined
in one file from another, without every file's names colliding with every other file's names?

## Intuition

Think of a single script as one big room where every name — every variable, every function —
lives on the same shelf. Add a second script that also defines a function called `process`, and
if both scripts' code ever runs in the same room, the second `process` overwrites the first with
no warning.

A **module** (any `.py` file) is instead a room with its own shelf. Nothing on one module's shelf
is visible from another module's shelf by default. `import` is the act of reaching into another
room and either bringing back the whole shelf under a label (`import math` → everything is now
reachable as `math.something`) or bringing back specific items by name
(`from math import pi, sqrt`). A **package** is a folder of rooms — modules grouped under one
name, with an `__init__.py` file marking the folder itself as importable (in modern Python this
file can even be empty; its presence, or the folder being on `sys.path` as a namespace package,
is what makes `import mypackage.mymodule` work at all).

## Why simpler approaches fail

**"Just keep everything in one file, or copy-paste the function into every script that needs
it."** Both work at small scale and both break down the same way as a project grows:

- One giant file makes every name globally visible to every other name in the file — a `total`
  variable used for one calculation on line 40 can be silently reused (and corrupted) by another
  calculation on line 400 that also happens to use `total`. There's no boundary.
- Copy-pasting a function into five scripts means a bug fix has to be applied five times, and the
  fifth copy someone forgets to update becomes a real, hard-to-find inconsistency — five things
  that were supposed to be "the same function" have quietly diverged.

The module system solves both at once: each file gets its own private namespace (no accidental
collisions), and a function defined once in one file is *reused*, not *copied*, everywhere it's
imported — one definition, one place to fix a bug.

## Conceptual foundation

*(Substituting for "Mathematical foundation" — this topic has no derivation; the underlying
mechanism to make explicit is the import system itself: `sys.path`, namespace isolation, and the
module cache.)*

**What `import module_name` actually does, in order:**

1. **Locate.** Python searches an ordered list of directories, `sys.path`, for a file matching
   the module name (`module_name.py`, or a package directory `module_name/` containing
   `__init__.py`). `sys.path` starts with the running script's own directory (or `''`, meaning
   "current working directory," for `python -c` / a REPL / a notebook), then the standard
   library's directories, then any installed third-party packages' locations.
2. **Read.** The file's source text is read from disk.
3. **Execute in a fresh namespace.** The module's top-level code — every `def`, every
   `class`, every bare assignment — runs exactly once, inside a namespace (conceptually a dict)
   that belongs only to that module. This is namespace isolation: two modules that both define
   `X = 1` never collide, because each `X` lives in its own module's dict, not in one shared dict.
4. **Cache.** The resulting namespace is stored in `sys.modules`, keyed by module name. Every
   subsequent `import module_name` anywhere in the program — even from a completely different
   file — returns the *same* cached namespace instead of re-reading and re-executing the file.
   This is why a module's top-level `print` statement, if it has one, is only ever seen once per
   process, on the first import.

**The `__name__ == '__main__'` guard follows directly from step 3.** Every module's namespace gets
a `__name__` variable set during execution: `'__main__'` if the file is being run directly (as a
script), or the module's own name if it's being imported. `if __name__ == '__main__':` is checking
which of those two situations produced the current execution — letting one file serve as both a
standalone script and an importable module without its script-only code running on import.

**A package adds one more layer on top of steps 1 and 3**: `import mypackage.mymodule` locates
and executes `mypackage/__init__.py` first (the package's own namespace, which can re-export names
from its submodules), then locates and executes `mypackage/mymodule.py` within that package's
namespace.

## Algorithm

`import` as a sequence of steps (this is what the from-scratch cell below reimplements a small,
incomplete slice of):

1. Check `sys.modules` for the requested name — if present, return the cached namespace
   immediately (steps 2–4 below never run again).
2. Search `sys.path`, in order, for a file/directory matching the name.
3. Read the located file's source text.
4. Create a fresh namespace, execute the source text inside it.
5. Store the namespace in `sys.modules` under the module's name.
6. Bind the requested name(s) in the *importing* module's namespace (the whole module for
   `import x`, or specific names for `from x import y`).

## From-scratch implementation

A minimal, deliberately incomplete stand-in for steps 3–4 above — read a `.py` file's text and
`exec()` it into a fresh namespace dict — compared against the real mechanism (`importlib`) on the
identical file. See [`modules_packages.ipynb`](modules_packages.ipynb), section "1b. From-scratch:
what `import` actually does."

Real executed output:

```
manual import  -> hello from mini module
manual import  -> hello from mini module, Ada!
real importlib -> hello from mini module
real importlib -> hello from mini module, Ada!
```

Namespace isolation, demonstrated directly — two files that both define `X` never collide when
each is `manual_import`-ed into its own fresh dict, but do collide (silently) when both are
`exec()`-ed into one shared dict:

```
ns_a['X'] = from module A
ns_b['X'] = from module B
same dict object? False
shared['X'] after both = from module B (module B silently overwrote module A's X)
```

`sys.path`/`sys.modules` caching, demonstrated directly — a module's top-level `print` only fires
on the *first* import:

```
mod_c is executing (top-level code runs once, at first import)
first import:  mod_c.Y = 42
second import: mod_c.Y = 42
cached in sys.modules? True
```

## Practical implementation

The full practical notebook — [`modules_packages.ipynb`](modules_packages.ipynb) — covers, with
real executed examples: `import` styles (whole-module, aliased, specific-name, wildcard — and why
wildcard is discouraged), the `__name__ == '__main__'` guard, and a tour of the standard library
modules most used elsewhere in this repo: `os`/`pathlib` (filesystem paths), `math`/`random`,
`datetime`, `json`, `re`, `collections` (`Counter`, `defaultdict`, `deque`), `itertools`
(`chain`, `product`, `combinations`, `permutations`), and `uv` for installing/managing packages at
the project level. This maps directly back to the Conceptual foundation: every one of these
`import`s runs exactly the locate → read → execute → cache sequence derived above; the from-scratch
section makes that sequence explicit before the rest of the notebook uses it at speed.

## Experiment

This topic has no timed performance experiment — the import system's *mechanism* is the subject,
not its speed. The "experiment" here is closer to a controlled reproduction: 1b's `sys.modules`
caching demo is set up with an explicit before/after (a module executes its top-level code once,
not twice, across two `import` calls) and the module-collision demo has an explicit contrast
(isolated dicts don't collide; one shared dict does) — both confirmed by the real output pasted
above rather than asserted.

## Failure modes

- **Circular imports.** If `circ_a.py` contains `from circ_b import Y` and `circ_b.py` contains
  `from circ_a import X`, importing `circ_a` starts executing it, reaches the `from circ_b import
  Y` line, and starts executing `circ_b` — which immediately tries `from circ_a import X`.
  `circ_a` is already present in `sys.modules` (it's mid-execution, only its very first line has
  run), but its `X = 1` line hasn't executed yet, so the name isn't there. Reproduced for real:

  ```
  returncode: 1
  last stderr line: ImportError: cannot import name 'X' from 'circ_a' (consider renaming
  '/tmp/circ_demo/circ_a.py' if it has the same name as a library you intended to import)
  ```

  The fix is structural, not syntactic: break the cycle (move the shared piece into a third
  module both import from, or import inside the function that needs it rather than at module
  top-level, deferring the lookup until both modules have finished executing).

- **Shadowing a standard-library module name.** `sys.path` search order (step 2 above) means a
  local file literally named `random.py`, sitting in a directory ahead of the real standard
  library on `sys.path`, is found *first* — `import random` binds to the local file silently, with
  no error at the import line itself. The failure only surfaces later, wherever the real module's
  functionality is actually used. Reproduced for real (a throwaway `random.py` containing only
  `MY_CONSTANT = 7`, run from the directory containing it):

  ```
  --- run from inside shadow_demo/ (local random.py shadows stdlib) ---
  returncode: 1
  last stderr line: AttributeError: module 'random' has no attribute 'randint' (consider renaming
  '/tmp/shadow_demo/random.py' since it has the same name as the standard library module named
  'random' and prevents importing that standard library module)
  --- run from /tmp (no shadowing file present) ---
  returncode: 0
  stdout: 3
  ```

  The identical `import random; random.randint(...)` line succeeds or fails purely based on which
  directory happens to be searched first — a strong argument for never naming a project file after
  a standard-library module.

## Real-world usage

- **Project structure.** Every non-trivial Python codebase (this repo included) is organized as
  packages of modules specifically so that related code lives together, unrelated code stays
  isolated, and one function fix propagates everywhere that function is used instead of needing to
  be repeated.
- **The standard library itself** is nothing but modules and packages — `os`, `json`, `re`,
  `collections` are all ordinary `.py` files (or C-implemented equivalents) discovered via the
  exact `sys.path` mechanism derived above; there is no special-case machinery for "built-in"
  modules beyond being pre-installed on `sys.path`.
- **Virtual environments** (`uv`, `venv`, `conda`) work by manipulating `sys.path` — activating an
  environment changes which directories are searched first, which is why the same `import numpy`
  can resolve to entirely different installed versions depending on which environment is active.
- **`sys.modules` caching** is why a module-level side effect (opening a database connection,
  loading a config file) that a library performs at import time only happens once per process, no
  matter how many other modules import that library — a deliberate design choice that both saves
  redundant work and can surprise anyone expecting a fresh run on every import.

## Mental model

**`import` is: locate a file on `sys.path`, read it, execute it once into its own private
namespace, cache that namespace by name, and hand back either the whole namespace or specific
names from it.** Every module gets its own shelf; nothing collides unless two things are
deliberately imported into the same namespace under the same name. Circular imports and stdlib
shadowing are not separate bugs to memorize — they're both direct, predictable consequences of
this exact sequence (a partially-populated namespace read too early; the wrong file found first on
`sys.path`).

## Questions to think about

1. `sys.modules` caches a module after its first import, and a module's top-level code runs
   exactly once as a result. What would break if Python instead re-executed a module's top-level
   code on every single `import` statement that named it?
2. The circular-import demo fails on `from circ_a import X` specifically, not on the earlier
   `import circ_a` inside `circ_b`. Why does a bare `import circ_a` (without `from ... import`)
   *not* immediately fail the same way, even while `circ_a` is still mid-execution?
3. The shadowing demo shows `import random` binding to a local `random.py` with no error at the
   import line — the `AttributeError` only appears later, at `random.randint(...)`. Why is a
   failure that happens later, at first use, harder to debug than one that would happen
   immediately at the `import` line — and where would you actually look, given only the
   `AttributeError`, to find the real cause?
4. `from math import *` pulls every public name from `math` directly into the importing module's
   own namespace, rather than requiring the `math.` prefix. Using the namespace-isolation argument
   from the Conceptual foundation, explain concretely what this specific import style gives up
   compared to `import math`.
5. A package's `__init__.py` runs as part of importing the package itself (step of the Algorithm
   above, applied to a package rather than a plain module). What does that imply about where it's
   safe to put expensive setup code (e.g., loading a large file) versus where it should instead be
   deferred into a function that only runs when actually called?
</content>
