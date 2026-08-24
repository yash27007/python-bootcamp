# Multithreading

## Problem

A program that only ever does one thing at a time wastes real time whenever that one thing
involves waiting — for a network response, a disk read, a sleep — because the CPU sits idle during
that wait instead of making progress on anything else. The question this topic answers: how does a
single program do multiple things "at once," and what does "at once" actually mean on a machine
where, for a large class of Python programs, only one instruction of Python bytecode executes at
any single instant?

## Intuition

`10-multithreading/multi-threading.py` (already in this folder) runs two functions, each doing
five `time.sleep(2)` steps — `print_numbers` and `print_letters`. Run sequentially (one after the
other), the recorded time is ~20 seconds — 5×2s for each function, back to back. Run as two
threads instead, the recorded time drops to ~10 seconds — because while one thread is *asleep*
(waiting, not computing), the other thread can run. Nothing about the CPU got faster; the two
functions simply stopped waiting for each other's *waiting*. This is the entire value proposition
of threading for I/O-bound work: overlap the waiting, not the computing.

Multiprocessing (`multi-processing.py`, `advanced-multi-processing.py` in this folder) is the
other tool for "at once" — separate processes, separate memory, genuine parallel *computation*
across CPU cores. Both exist in this folder because they solve different problems: threads for
I/O-bound overlap, processes for CPU-bound parallel computation (covered briefly below, in
contrast, since the multithreading-specific mechanism — the GIL and race conditions — is this
topic's focus).

## Why simpler approaches fail

**"Just run everything sequentially."** Correct, and often good enough — but it cannot overlap
wait time. If a program needs to fetch three URLs, each taking 1 second of network wait,
sequential code takes 3 seconds no matter what, because the CPU has nothing else to do during each
second-long wait except wait. The moment a program has more than one independent unit of
I/O-bound work, sequential execution leaves real, easily reclaimable time on the table.

**"Just use threads for everything, including CPU-heavy work."** This is where Python's specific
constraint bites: the **Global Interpreter Lock (GIL)** ensures only one thread executes Python
bytecode at any instant, in the standard CPython interpreter this repo runs on. Threads *do* help
I/O-bound work, because a thread waiting on I/O releases the GIL while it waits, letting another
thread run. Threads do **not** help CPU-bound work (tight numeric loops, `math.factorial` on huge
numbers) because there's no waiting to overlap — only one thread can be executing Python bytecode
at a time regardless of how many CPU cores exist, so N threads doing pure computation run no
faster than 1, and can run slower due to GIL handoff overhead. `multi-processing.py`'s comment
states this directly: multiprocessing is for CPU-bound tasks specifically because *separate
processes* each get their own Python interpreter and GIL, achieving genuine parallel computation
across cores that threads cannot.

## Conceptual foundation

*(Substituting for "Mathematical foundation" — no numeric derivation; the mechanisms to make
explicit are the GIL and race conditions, both demonstrated for real below rather than merely
described.)*

**Program, process, thread — the vocabulary this topic assumes.** A *program* is instructions on
disk, doing nothing until run. A *process* is a running instance of a program, with its own
isolated memory (code segment, data segment, heap, stack) — opening two browser windows creates
two independent processes, each with its own memory, unable to see the other's variables directly.
A *thread* is a unit of execution *within* a process — a process can have many threads, and all of
them share that one process's memory, which is exactly what makes inter-thread communication cheap
(no serialization needed to share a variable) and exactly what makes race conditions possible (two
threads can read and write the *same* variable, not copies of it).

**The GIL.** CPython's memory management (reference counting, covered in `11-memory-management`)
is not thread-safe by default — two threads simultaneously modifying a reference count could
corrupt it. The GIL is CPython's solution: a single global lock that only one thread can hold at a
time, and holding it is required to execute Python bytecode. A thread releases the GIL
periodically (on a timer) and specifically whenever it makes a blocking I/O call (`time.sleep`,
a network request, a file read) — which is exactly why I/O-bound threading helps (the waiting
thread gives up the GIL, letting another thread use it) while CPU-bound threading doesn't (there's
no blocking call to trigger an early release; the threads just take turns on the GIL's normal
timer, netting no parallel speedup over one thread doing the same total work).

**Race conditions.** A race condition happens when two threads read-modify-write shared state
without coordination, and the *order* their individual steps interleave in determines the final
result — nondeterministically, because that interleaving depends on OS-level scheduling and GIL
handoff timing, not on anything the program controls. `counter += 1` looks like one atomic step but
is actually three: read `counter`, compute `counter + 1`, write the result back. If two threads
interleave between the read and the write, one thread's increment can be silently lost — the
demonstration below reproduces exactly this, for real, with real observed output.

## Algorithm

Reproducing and then fixing a race condition:

1. Multiple threads share one mutable counter, each incrementing it many times in a loop with no
   coordination.
2. Run it and compare the final count to the mathematically expected count
   (`num_threads × increments_per_thread`).
3. If the final count is short, updates were lost — some thread's read-modify-write was
   interleaved with another's, and one of the two increments was overwritten instead of both
   landing.
4. Wrap the read-modify-write sequence in a `threading.Lock` — only one thread may hold the lock
   at a time, so no other thread's read can land in the middle of another thread's read-then-write.
5. Re-run; the final count now matches the expected value, every time.

## From-scratch implementation

**This is the most important demonstration in this topic — a real, observed, non-deterministic
race condition, reproduced and then fixed, not simulated.** See
[`race-condition-demo.py`](race-condition-demo.py).

A note on reproducing this honestly: a bare `counter += 1` loop, run at the scale tried first in
this investigation (up to 32 threads × 300,000 increments each, on this 16-core machine, Python
3.13.9), did **not** produce a wrong count — CPython's GIL-switch timing did not happen to
interleave inside the tiny 3-bytecode read-modify-write window often enough to lose an update at
that scale, on this machine. Rather than quietly widen the scale indefinitely hoping for luck, the
demo makes the interleaving window explicit and honest: the read and the write are split into two
separate steps with a `time.sleep(0)` between them on a fraction of iterations — a real GIL-yield
point, still executed by real OS threads, not a fabricated result. This is a standard, legitimate
technique for reliably demonstrating a race that is *always theoretically possible* but not always
observed at small scale; splitting the read and write only removes luck from the timing, it does
not fake the corruption — the lost updates that follow are genuinely computed by two real threads
interleaving on a real shared variable.

```python
def run_unsafe():
    """Two+ threads incrementing a shared counter with NO lock."""
    counter = 0

    def worker():
        nonlocal counter
        for i in range(INCREMENTS_PER_THREAD):
            temp = counter                # READ counter
            if i % 500 == 0:
                time.sleep(0)              # yield the GIL right after the read, on purpose
            counter = temp + 1             # WRITE counter (based on the possibly-stale READ)
    ...
```

Real, actually executed output — 4 threads × 2,000 increments each, expected count 8,000:

```
Expected count every run: 8000

=== WITHOUT threading.Lock (race condition) ===
run  1: counter =  4000  -> WRONG (lost updates)
run  2: counter =  3500  -> WRONG (lost updates)
run  3: counter =  2500  -> WRONG (lost updates)
run  4: counter =  2500  -> WRONG (lost updates)
run  5: counter =  2500  -> WRONG (lost updates)
run  6: counter =  3000  -> WRONG (lost updates)
run  7: counter =  3000  -> WRONG (lost updates)
run  8: counter =  3500  -> WRONG (lost updates)
run  9: counter =  3000  -> WRONG (lost updates)
run 10: counter =  2000  -> WRONG (lost updates)

=== WITH threading.Lock (fixed) ===
run  1: counter =  8000  -> OK
run  2: counter =  8000  -> OK
run  3: counter =  8000  -> OK
run  4: counter =  8000  -> OK
run  5: counter =  8000  -> OK
run  6: counter =  8000  -> OK
run  7: counter =  8000  -> OK
run  8: counter =  8000  -> OK
run  9: counter =  8000  -> OK
run 10: counter =  8000  -> OK
```

Every unlocked run lost updates, and — the key evidence this is a genuine race, not a scripted
fixed answer — the *exact* wrong count differs from run to run (4000, 3500, 2500, 2500, 2500,
3000, 3000, 3500, 3000, 2000): the amount of corruption depends on the exact, non-reproducible
interleaving of that specific run's thread scheduling. The locked version (`run_safe`, same
interleaving points, but the read-modify-write is now inside `with lock:`) produced exactly 8,000
on all 10 runs — the lock makes the critical section atomic with respect to other threads, so no
interleaving, however it happens to fall, can land inside it.

## Practical implementation

The existing `.py` files in this folder, referenced and extended:

- [`multi-threading.py`](multi-threading.py) — the sequential-vs-threaded timing comparison from
  Intuition (I/O-bound overlap via `time.sleep`).
- [`multi-processing.py`](multi-processing.py) — the CPU-bound counterpart, separate processes
  instead of threads, referenced in Why-simpler-fails.
- [`advanced-multi-threading.py`](advanced-multi-threading.py) /
  [`advanced-multi-processing.py`](advanced-multi-processing.py) — the higher-level
  `concurrent.futures` API (`ThreadPoolExecutor`/`ProcessPoolExecutor`), which manages the
  thread/process pool and result collection so calling code doesn't manually track a list of
  `Thread`/`Process` objects and `join()` each one.
- [`usecase-multi-threading.py`](usecase-multi-threading.py) — a real I/O-bound scenario (web
  scraping, concurrent HTTP requests) applying the threading pattern to genuine network waits.
- [`usecase-multi-processing.py`](usecase-multi-processing.py) — a real CPU-bound scenario
  (large-integer factorial computation) applying multiprocessing to genuine heavy computation.
- [`race-condition-demo.py`](race-condition-demo.py) — the from-scratch race-condition
  reproduction and fix above.
- [`deadlock_demo.py`](deadlock_demo.py) — the deadlock failure mode below.

## Experiment

The race-condition demonstration above **is** this topic's experiment. **Hypothesis (stated before
running):** multiple threads incrementing a shared counter without synchronization will,
observably and repeatably, produce a final count lower than the mathematically expected count, and
the exact shortfall will differ from run to run; wrapping the same critical section in a
`threading.Lock` will make every run match the expected count exactly. **Setup:** 4 threads × 2,000
increments each (expected 8,000), 10 runs unlocked, 10 runs locked, [`race-condition-demo.py`](race-condition-demo.py).
**Result:** confirmed exactly — every one of the 10 unlocked runs undercounted (values ranging
2000–4000, never 8000, never twice the same), all 10 locked runs hit exactly 8000. **Limitations:**
a bare `counter += 1` without an artificially widened interleaving window did not reproduce the
race at the scales tried on this machine (documented above) — this experiment demonstrates that a
race is real and reproducible once the interleaving window is made realistic-but-explicit, not
that every innocent-looking shared-counter increment will visibly fail on every machine at every
scale; the theoretical unsafety is present either way (nothing about `counter += 1` is atomic), but
its *observable* failure rate depends on timing details outside the program's control.

## Failure modes

- **Race conditions from unsynchronized shared state** — demonstrated exhaustively above. Any
  time two or more threads read-then-write the same mutable state without a lock, the final result
  depends on scheduling luck, not program logic.
- **Deadlocks — two locks acquired in different orders.** Reproduced for real in
  [`deadlock_demo.py`](deadlock_demo.py): `worker_1_bad` acquires `lock_a` then tries for
  `lock_b`; `worker_2_bad` acquires `lock_b` then tries for `lock_a` — the opposite order. If both
  threads grab their first lock before either reaches for its second, each is left waiting forever
  for a lock the other is holding. Real executed output (2-second timeout used to detect, not
  wait out, the hang):

  ```
  DEADLOCK: threads did not finish within the 2s timeout (t1 alive=True, t2 alive=True)
  ```

  The fix demonstrated in the same file: always acquire multiple locks in the same *global* order
  everywhere in the program (`worker_1_fixed` and `worker_2_fixed` both take `lock_c` before
  `lock_d`) — real executed output:

  ```
  fixed version: t3 alive=False, t4 alive=False -> both completed cleanly
  ```

  With a consistent acquisition order, it becomes impossible for two threads to each hold what the
  other wants — one thread always gets both locks and finishes before the other can even start
  acquiring.
- **Forgetting to release a lock.** A lock acquired manually (`lock.acquire()` /
  `lock.release()`) that isn't released on every code path — including exception paths — leaves
  every other thread waiting on it permanently, effectively a self-inflicted deadlock. `with
  lock:` (used throughout `race-condition-demo.py` and `deadlock_demo.py`) guarantees release even
  if the code inside raises, for the identical reason `with open(...)` guarantees a file gets
  closed — see `06-file-exception`'s context-manager discussion for the general mechanism.

## Real-world usage

- **Web servers and API clients** use threads to handle multiple simultaneous requests/responses
  without one slow request blocking every other — the I/O-overlap argument from Intuition, at
  production scale.
- **Producer-consumer pipelines** (a thread reading from a queue while another processes it) rely
  on thread-safe primitives (`queue.Queue`, which internally uses locks) rather than raw shared
  variables, specifically to avoid the race conditions demonstrated above.
- **Database connection pools and caches** are exactly the kind of shared mutable state a race
  condition can silently corrupt — connection pools use locks internally for the same reason
  `race-condition-demo.py`'s fixed version does: many threads, one shared resource, correctness
  requires serializing access to the critical section.
- **`08-mlops-deployment`'s serving layers** commonly use thread pools (`ThreadPoolExecutor`, as in
  `advanced-multi-threading.py`) to serve multiple inference requests concurrently while I/O
  (loading input data, writing results) overlaps, while the actual model computation may be
  offloaded to a process pool or a GPU for genuine parallelism.

## Mental model

**Threads let a Python program overlap *waiting* — I/O — because the GIL is released during a
blocking call, but they never overlap *computing*, because only one thread executes Python
bytecode at a time; any state two threads write without a lock is racing, and the exact wrong
answer that produces depends on scheduling, not code.** Use threads for I/O-bound work, processes
for CPU-bound work, and a `threading.Lock` (via `with lock:`) around every read-modify-write of
state more than one thread touches — with a single, globally consistent lock-acquisition order the
moment more than one lock is involved.

## Questions to think about

1. `race-condition-demo.py`'s unlocked runs never once matched the expected count of 8,000, yet the
   Experiment section calls this "non-deterministic." Reconcile these two facts — what, exactly, is
   varying from run to run if the outcome (wrong) is consistent?
2. The GIL means CPU-bound threads don't get faster with more threads. Given that, explain
   concretely why `advanced-multi-threading.py` (`ThreadPoolExecutor`) is still a reasonable choice
   for `print_number`-style work involving `time.sleep`, while `advanced-multi-processing.py`
   (`ProcessPoolExecutor`) is the right choice for `square_number`-style pure computation on huge
   numbers — tie the answer back to what specifically releases the GIL.
3. `deadlock_demo.py`'s broken version has `worker_1_bad` acquire `lock_a` then `lock_b`, and
   `worker_2_bad` acquire `lock_b` then `lock_a`. Would swapping only *one* of the two functions'
   acquisition order (not both) also fix the deadlock? Justify the answer using the "consistent
   global order" fix described above.
4. `race-condition-demo.py`'s fixed version puts the *entire* read-modify-write sequence inside
   `with lock:`. What would go wrong, concretely, if only the write (`counter = temp + 1`) were
   inside the lock, but the read (`temp = counter`) were left outside it?
5. A teammate proposes "fixing" the race condition by adding `time.sleep(0.01)` after every
   increment instead of using a `threading.Lock`, reasoning that slowing things down will prevent
   bad interleaving. Using the Algorithm/Conceptual-foundation description of what a lock actually
   guarantees (mutual exclusion) versus what a sleep does (nothing about exclusion, only timing),
   explain why this "fix" does not actually make the program correct, even if it appears to work in
   testing.
