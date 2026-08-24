# Memory Management

## Problem

A long-running program (a web server, a training loop, a notebook kernel left open for hours)
creates objects continuously — lists, dicts, custom class instances. If nothing ever reclaimed the
memory behind an object once nothing in the program could reach it any more, memory usage would
grow without bound until the process is killed by the OS or simply runs out of RAM. The question
this topic answers: **how does Python know when an object is safe to free, and what happens when
that "safe to free" signal never actually fires even though the object really is unreachable?**

## Intuition

Every Python object carries a small counter: how many places in the program currently hold a
reference to it. Assign a list to a second variable name and the counter goes up by one; delete one
of those names and the counter goes down by one. The moment the counter hits zero, nothing in the
program can possibly reach the object any more, so CPython frees it immediately, deterministically,
the instant that last reference disappears — no waiting for a periodic sweep. This is **reference
counting**, and it is the primary, always-on mechanism.

But there is a case that breaks the "counter hits zero" story: two objects that reference *each
other*, with nothing else in the program referencing either one. Delete the outside variables that
pointed at them and each object's refcount drops by one — but never to zero, because each is still
holding a reference to the other. The two objects are unreachable from anywhere the program can
actually get to, yet by pure refcounting arithmetic they'll sit in memory forever. This is a
**reference cycle**, and it's why CPython also runs a separate, periodic **cycle-detecting garbage
collector** (the `gc` module) that specifically looks for and frees groups of objects that only
reference each other.

## Why simpler approaches fail

**"Just rely on refcounting for everything."** This is refcounting's actual failure mode, not a
hypothetical: a reference cycle like `n1.other = n2; n2.other = n1` genuinely cannot reach a
refcount of zero through deletion alone, no matter how many outside references are removed — see
the executed demo below, where refcounts stay above zero after `del` and the objects are only freed
once `gc.collect()` runs.

**"Just never create cycles."** Reasonable advice, but not fully controllable in practice: cycles
arise naturally from common patterns — a parent object holding a list of children, each child
holding a `.parent` back-reference; a doubly-linked list; an observer registered on the object it
observes. Forbidding all of these outright would make normal object-oriented code harder to write,
so Python instead detects and collects cycles automatically rather than requiring the programmer to
avoid them.

**"Just periodically restart the process instead of fixing leaks."** A real, sometimes-used
workaround in production (some server frameworks recycle worker processes on a schedule
specifically to paper over slow leaks), but it treats the symptom, not the cause, and hides genuine
bugs (an accidentally-growing global cache, a cycle involving `__del__` — see Failure modes) behind
a restart cadence instead of surfacing and fixing them.

## Conceptual foundation

*(Substituting for "Mathematical foundation" — the mechanism here is a counting invariant and a
graph-reachability argument, not a numeric derivation.)*

**Reference counting.** Every Python object has a hidden integer field, its reference count.
`sys.getrefcount(obj)` reports it (always one *higher* than intuition suggests, because passing
`obj` into the function call itself creates one more temporary reference for the call's duration).
Every time a new name, container slot, or attribute is made to point at an object, its count goes
up; every time one of those is removed (`del`, reassignment, a container element being overwritten,
a function returning and its local variables going out of scope), the count goes down. The instant
the count reaches zero, CPython frees the object immediately — this is why simple, acyclic Python
objects are freed in a fully deterministic, predictable way, unlike garbage-collected languages
that only free on an unpredictable sweep.

**Why a cycle defeats pure counting.** Reference counting is a purely *local* rule: it only asks
"is my own count zero," never "can anything in the whole program still reach me." Two objects that
reference only each other satisfy neither object's local zero-check, even though *globally*,
nothing outside the pair can reach either one. Refcounting has no way to see that global fact — it
would need to trace reachability across the whole object graph, which is exactly what it was
designed not to have to do on every reference change (that would be far too slow to run on every
single assignment).

**The cycle-detecting garbage collector.** `gc` runs a separate algorithm, not on every reference
change but periodically (triggered by allocation-count thresholds, or manually via `gc.collect()`):
it looks specifically at *container* objects (objects that can hold references to other objects —
lists, dicts, class instances, but not plain ints or strings, which can't participate in a cycle)
and finds groups whose only remaining references are to each other, unreachable from any global or
stack variable. It frees those groups as a unit. This is graph-reachability applied to the subset
of objects that can form cycles, run only when needed — a global check the local refcounting rule
was never built to do.

## Algorithm

Reference counting (implicit, on every reference change):
1. New reference created (assignment, container insert, function argument, ...) -> count += 1.
2. Reference removed (`del`, reassignment, scope exit, container element overwritten) -> count -= 1.
3. If count reaches 0 -> free the object immediately, and recursively decrement the counts of
   everything *it* referenced (which may cascade further frees).

Cycle collection (`gc.collect()`, periodic or manual):
1. Walk all container objects known to the collector.
2. For each, subtract references that originate *from within the same candidate group* (i.e. count
   only references coming from outside the group being examined).
3. Any object whose *externally-originating* reference count is still zero after that subtraction
   is unreachable from outside the group — the whole group is cyclic garbage.
4. Free every object in that group.

## From-scratch implementation

See [`refcount_and_cycle_demo.py`](refcount_and_cycle_demo.py) — real, actually executed, both
parts of the mechanism observed directly rather than merely described.

**Part 1 — reference counting observed via `sys.getrefcount`:**

```python
a = []
print(sys.getrefcount(a))   # the extra +1 is the temporary reference from the call itself
b = a
print(sys.getrefcount(a))   # up by 1
del b
print(sys.getrefcount(a))   # back down
```

Real executed output:

```
refs to a right after creation: 2
refs to a after b = a:          3
refs to a after del b:          2
```

**Part 2 — a real reference cycle that refcounting alone cannot free:**

```python
class Node:
    def __init__(self, name):
        self.name = name
        self.other = None
    def __del__(self):
        print(f"  Node {self.name} __del__ called -> memory actually freed")

gc.disable()                  # isolate refcounting from the cycle collector
n1, n2 = Node("A"), Node("B")
n1.other = n2
n2.other = n1                 # the cycle
del n1
del n2                        # local names gone, but the cycle still references both objects
```

Real executed output (`refcount_and_cycle_demo.py`, `gc` disabled so only refcounting is active
until `gc.collect()` is called explicitly):

```
=== 2. A reference cycle refcounting alone cannot free ===
created cycle: n1.other -> n2, n2.other -> n1
refcount n1 before del: 3
refcount n2 before del: 3
deleted local names n1, n2 -- refcount dropped by 1 each, but not to 0
are the cycle objects still alive (found by gc.get_objects())? True
calling gc.collect() -- the cycle-detecting collector...
  Node A __del__ called -> memory actually freed
  Node B __del__ called -> memory actually freed
gc.collect() collected 2 objects
are the cycle objects still alive now? False
```

This is the whole argument made concrete: after `del n1; del n2`, `gc.get_objects()` still finds
both nodes alive (refcounting alone did not free them — their mutual references kept each other's
count above zero), and their `__del__` messages have not printed yet. Only after `gc.collect()` are
they actually freed — the `__del__` messages print *at that moment*, not at `del` time, and
`gc.get_objects()` no longer finds them.

The existing notebook, [`memory_management.ipynb`](memory_management.ipynb) (cell 7), contains an
equivalent from-scratch cycle demo with `gc` left enabled — real executed output:

```
Object obj1 created
Object obj2 created
Object obj1 destroyed
Object obj2 destroyed
```

(followed by `gc.collect()` returning `2`) — with `gc` enabled, CPython's automatic generation-0
collection threshold can trigger a collection during the `del` calls themselves, which is why both
`destroyed` messages already appear before the explicit `gc.collect()` line runs in that cell; the
disabled-`gc` version above isolates the two mechanisms so the "refcounting alone doesn't do it"
step is unambiguous.

## Practical implementation

The existing notebook, [`memory_management.ipynb`](memory_management.ipynb), all cells now
executed:

- Cell 1 — `sys.getrefcount` on a real list, matching the from-scratch demo above.
- Cells 2-4 — `gc.enable()` / `gc.disable()` / `gc.collect()`, the collector's on/off/manual-trigger
  controls.
- Cell 5 — `gc.get_stats()`, real per-generation collection counts from this run (three
  generations, CPython's generational collector — younger generations are swept more often than
  older ones, on the empirical assumption that most objects die young).
- Cell 6 — `gc.garbage`, the (empty, on this modern CPython) list of objects the collector found
  uncollectable — see Failure modes for when this historically was not empty.
- Cell 7 — the cycle demo described above.
- Cell 8 — a generator (`yield`) as the memory-efficient alternative to building a full list, when
  only one item is needed at a time; connects to `08-advanced-concepts`' iterator/generator topic.
- Cell 9 — `tracemalloc`, real per-line memory-allocation profiling on a 100,000-element list
  comprehension, the practical tool for actually finding where a real leak's allocations originate.

## Experiment

**Hypothesis (stated before running):** a reference cycle's objects will not be freed by `del`
alone (refcounts will still be above zero and the objects will still be found alive by
`gc.get_objects()`), but will be freed the moment `gc.collect()` runs.

**Setup:** `refcount_and_cycle_demo.py`, `gc` disabled to isolate refcounting from the automatic
collector, `__del__` used as a real, observable "was this object actually freed" signal, `id()`
checked against `gc.get_objects()` before and after `gc.collect()`.

**Actual result:** confirmed exactly, real captured output above — both nodes' refcounts were 3
before `del` (name + `.other` from the other node + the `getrefcount` call's own temporary
reference), still findable via `gc.get_objects()` after both `del` statements, and only freed
(`__del__` printed, no longer found in `gc.get_objects()`) once `gc.collect()` ran.

**Limitations:** this demo uses a minimal 2-object cycle for clarity; real leaks typically involve
longer chains or larger structures (e.g. a cache holding many cyclic entries), where the same
mechanism applies but is harder to spot without a tool like `tracemalloc` or `objgraph`. It also
disables `gc` to isolate the effect cleanly — with `gc` left enabled (the default, and the
notebook's cell 7 configuration), CPython's automatic threshold-triggered collection can free a
cycle "on its own" without an explicit `gc.collect()` call, which is the normal, desired behavior in
production; the disabled version exists here purely to make the "refcounting alone is insufficient"
step unambiguous and directly observable.

## Failure modes

- **Reference cycles involving `__del__` — historically uncollectable before Python 3.4.** Before
  PEP 442 (Python 3.4), the cycle collector refused to break a cycle if *any* object in it defined
  `__del__`, because the collector couldn't determine a safe order to call the finalizers in (one
  object's `__del__` might depend on another object in the cycle still being valid). Such cycles
  were placed in `gc.garbage` instead of being freed — a real, silent memory leak in long-running
  Python 2/early Python 3 programs that used `__del__` on objects that could form cycles. Python 3.4
  onward calls finalizers in a well-defined way and *does* collect these cycles (as demonstrated
  above — this repo's environment collects the `Node`/`MyObject` cycles cleanly), but the historical
  failure mode is worth knowing: any codebase or tutorial written against pre-3.4 semantics that
  warns "never use `__del__` on cyclic objects, it will leak" was correct for its era.
- **Holding references in a global/module-level cache accidentally.** The single most common real
  leak in long-running Python services: a module-level `dict` or `list` used as a cache that objects
  get added to but never removed from. Refcounting works exactly as designed here — the cache
  really does hold a live reference, so the objects are, correctly, not garbage. The "leak" is a
  logic bug (forgetting to evict), not a refcounting or `gc` failure; no garbage collector can free
  memory that a live, reachable data structure still legitimately references. `tracemalloc` (cell 9
  above) is the practical tool for finding this: a growing cache shows up as a specific line
  allocating steadily more memory over time, snapshot to snapshot.
- **Manually disabling `gc` for performance and never re-enabling it.** `gc.disable()` (cell 3) is a
  real, sometimes-used optimization (the collector's periodic sweep has a real cost, and some
  workloads create no cycles at all), but doing it without confidence that the program creates no
  cycles reintroduces exactly the leak this topic demonstrates, silently.

## Real-world usage

- **Web servers and long-running services** are the primary place memory leaks matter in practice —
  a script that runs for a second and exits never accumulates enough leaked memory to notice, but a
  server handling requests for weeks will eventually be killed by an out-of-memory error if it leaks
  even a small amount per request.
- **`tracemalloc` and `objgraph`** (a third-party tool built on the same `gc.get_objects()`
  machinery used in the demo above) are the standard practical tools for diagnosing a real
  production leak — find which line is allocating memory that isn't being freed, and which objects
  are keeping a suspected leaked object reachable.
- **`10-multithreading`'s GIL** exists partly because of this topic: refcount increments/decrements
  are not thread-safe by default (two threads incrementing the same object's refcount simultaneously
  could corrupt it), and the GIL is CPython's solution to that specific problem, among others —
  see `10-multithreading/notes.md`.
- **Deep learning training loops** (covered later in this repo) are a common place to accidentally
  create leaks — appending a full computation graph or GPU tensor into a Python list "for logging"
  every step, without detaching it, keeps every step's entire graph alive for the whole run.

## Mental model

**Refcounting frees an object the instant nothing can reach it — a purely local, deterministic
rule — but two objects that only reference each other never satisfy that local rule even when
globally unreachable; the cycle-detecting `gc` module is the periodic, global check that catches
exactly that gap.** `del` removes a name, not necessarily an object — the object is freed only when
its refcount (or its cycle's external reachability) actually reaches zero.

## Questions to think about

1. `sys.getrefcount(a)` reported `2` for a freshly created list bound to one variable name. Explain
   precisely where the second reference comes from, and predict what `sys.getrefcount(a)` would
   report immediately after `l = [a, a]` (the same object appearing twice in a new list).
2. The from-scratch cycle demo disabled `gc` before creating the cycle. If `gc` had been left
   enabled instead (the default), is it still guaranteed that `del n1; del n2` alone would leave the
   objects un-freed until some collection runs — or could the automatic collector free them as part
   of those very `del` statements? What does `memory_management.ipynb`'s cell 7 output suggest about
   this?
3. A module-level dict is used as a cache: `_cache[key] = expensive_result`. Entries are never
   removed. Is this a reference-counting bug, a cycle the `gc` module should be catching, or neither?
   Justify the answer using the Failure modes section.
4. Why can't reference counting itself just periodically ask "is anything in the whole program still
   able to reach this object" instead of only tracking a local counter — what would that cost on
   every single assignment, and why did CPython's designers choose the local-counter-plus-periodic-
   cycle-sweep split instead?
5. Before Python 3.4, a cycle containing an object with a `__del__` method was left uncollected in
   `gc.garbage` rather than freed. Using the description of what the collector needs to determine to
   safely free a cycle, explain concretely what made `__del__` specifically the problem (as opposed
   to, say, a cycle with no finalizers at all).
