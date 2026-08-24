# Iterators and Generators

## Problem

A program often needs to process a sequence of items one at a time — the lines of a file, the
rows of a dataset, the integers up to some large `N` — without knowing or caring, at the call
site, how "where am I in this sequence" is tracked from one step to the next. Two bad options
present themselves without a shared abstraction for this: build the *entire* sequence in memory
before touching a single item (fine for five items, impossible for a 50 GB log file or a
genuinely infinite sequence), or hand-write the position-tracking state (an index variable, a
"have I reached the end" flag) fresh, correctly, inside every function that needs to walk a
sequence. The question this topic answers: what does Python provide so that `for x in thing:`
works identically whether `thing` is a list already sitting in memory, a file being read line by
line, or a sequence with no fixed end at all — and who is responsible for remembering the current
position?

## Intuition

Picture reading a 10-million-line log file two ways. Way one: `lines = file.readlines()` reads
every line into a Python list *first*, then the `for` loop begins — the whole file exists twice
over (once on disk, once in RAM) before any processing starts, and if the file doesn't fit in
memory the program dies before it does anything useful. Way two: `for line in file:` reads and
hands over exactly one line at a time, processes it, discards it, and reads the next one — total
memory use stays roughly constant no matter whether the file has 10 lines or 10 billion. Both
loops look identical at the call site (`for x in thing:`); the difference is entirely in whether
`thing` hands over a value it already built, or computes/fetches the next value *on demand*, the
instant it's asked. That "on demand, one at a time" behavior is what an iterator is, and Python's
`for` loop is built to work with it uniformly regardless of which strategy the object underneath
actually uses.

## Why simpler approaches fail

**"Just build the full list, then loop over it."** Works fine at small scale (the practical
notebook's `my_list = [1,2,3,4]`), and is in fact simpler to reason about — but it doesn't scale.
Building a full list of every value before processing any of them means paying the *entire*
memory cost up front, even if the consumer only ever needs one value at a time, or even if the
sequence has no natural end (there is no list of "all positive integers" that could ever finish
being built). The Experiment section below measures this cost directly, in bytes and seconds, for
10 million items.

**"Hand-write the position tracking every time."** A function that walks a sequence without
Python's iterator support has to invent its own way to say "where am I" — an index into a list, a
byte offset into a file, a manually incremented counter — and every caller that wants to consume
that sequence has to know which convention that particular function used. The iterator protocol
below is Python's answer: one uniform interface (`__iter__`/`__next__`) that every kind of
sequence — lists, files, dict views, generators, custom classes — implements the same way, so
`for x in thing:` never needs to know or care which specific position-tracking strategy `thing`
uses underneath.

## Conceptual foundation

*(Substituting for "Mathematical foundation" — there's no derivation here; the mechanism to make
explicit is the iterator protocol itself: two dunder methods, and what `for` actually compiles
down to.)*

An **iterable** is anything `iter()` can be called on — it implements `__iter__`, which returns an
**iterator**. An iterator is anything `next()` can be called on — it implements `__next__`, which
either returns the next value or raises `StopIteration` when there is nothing left. `for x in
thing:` is syntactic sugar for exactly this: call `iter(thing)` once to get an iterator, then call
`next()` on it repeatedly, catching `StopIteration` to know when to stop:

```python
it = iter(my_list)
while True:
    try:
        x = next(it)
    except StopIteration:
        break
    # loop body with x
```

The practical notebook demonstrates this manually — `iterator = iter(my_list)` then four explicit
`next(iterator)` calls that walk the list one element at a time, followed by a fifth call that
raises `StopIteration`, caught to print "You have reached the end of the list." That explicit
version is exactly what the `for` loop above it does implicitly, every single iteration.

A **generator** (a function using `yield`, or a generator expression `(x for x in ...)`) is a
shortcut that produces an object which already satisfies the iterator protocol — `__iter__` and
`__next__` are both implemented for you, automatically, by the language, from a function body that
reads like ordinary imperative code. Every time `next()` is called on a generator, the function
body resumes running from wherever it last paused (at the previous `yield`), runs until the next
`yield`, and pauses there again — the function's local variables and exact execution point persist
across calls, which is what makes "resume where you left off" possible without any hand-written
state-tracking variable at all.

## Algorithm

Building a class-based iterator manually, then observing what a generator automates:

1. Define `__iter__(self)` returning `self` (or a fresh iterator, for a reusable iterable).
2. Define `__next__(self)`: compute and return the next value, or raise `StopIteration` when
   exhausted. State that "where am I" position must be tracked as ordinary instance attributes
   (`self.current`, `self.limit` below), set and updated by hand.
3. `for x in obj:` then works automatically — Python calls `iter(obj)` once, `next()` repeatedly.
4. A generator function replaces steps 1–2 with `yield` — the interpreter itself keeps the
   "current position" (the function's paused execution state) instead of the class needing an
   explicit `self.current` attribute.

## From-scratch implementation

The manual class-based iterator, written before any `yield`, alongside the equivalent generator —
see [`manual_iterator_protocol.py`](manual_iterator_protocol.py), executed for real:

```python
class CountUpTo:
    """Manual iterator protocol: __iter__ + __next__, no yield."""
    def __init__(self, limit):
        self.limit = limit
        self.current = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.current >= self.limit:
            raise StopIteration
        self.current += 1
        return self.current
```

Real executed output — the hand-written class and the `yield`-based generator produce identical
results, confirming the generator is not a different behavior, only a shorter way to get the same
protocol:

```
Manual __iter__/__next__ class:
1
2
3
4
5

Same thing with a generator function (yield):
1
2
3
4
5
```

The instance-attribute state (`self.current`) in `CountUpTo` is exactly the hand-written version
of what the generator's paused-function-frame does automatically — nothing conceptually new, only
who is responsible for remembering "where am I."

### Real measured memory/time comparison: full list vs. generator, N = 10,000,000

See [`memory_comparison_demo.py`](memory_comparison_demo.py). Real executed output on this
machine:

```
N = 10,000,000
List build time:      0.3365s
Generator build time: 0.00000691s
List sys.getsizeof:      89,095,160 bytes (84.97 MB)
Generator sys.getsizeof: 200 bytes
Ratio: list is 445,476x the size of the generator object

sum(full_list) result:            333333283333335000000
sum(generator expr) result:       333333283333335000000
Time to sum from prebuilt list:    0.0869s
Time to sum via generator (no list build needed at all): 0.2669s
```

Interpretation: building the list of 10 million squared integers costs ~85 MB and ~0.34s *before*
any processing happens — `sys.getsizeof` on the generator object itself reports a constant 200
bytes regardless of how many values it will eventually produce, because it holds only the paused
function state, not the values. The list is over 400,000× larger in memory for the same logical
sequence. The tradeoff shows up in the timing too: summing the pre-built list is faster
(0.087s, because the values already exist) than summing directly from the generator expression
(0.267s, because computing `i*i` for all 10M values happens *during* the sum, not before it) — the
generator trades a bit of per-element overhead for never holding the whole sequence in memory at
once. When the full sequence must be visited more than once, or holding it all is cheap, a list
can be faster; when memory is the constraint (or the sequence might never fully materialize), the
generator's constant memory footprint is the entire point.

## Practical implementation

The existing notebook — [`iterators_generators.ipynb`](iterators_generators.ipynb) — covers, with
real executed output: manual iteration via `iter()`/`next()` on a list through to `StopIteration`;
a generator function (`square`) and a hand-written multi-`yield` generator (`my_generator`);
lazily reading a large file line-by-line with `read_large_file` (the direct practical payoff of
the memory argument above — a file too large to fit in memory can still be processed one line at a
time); and closures/decorators (`@decorator`, `@repeat(n)`), which are covered in depth from the
scope/first-class-functions angle in `04-functions`'s notes — referenced here because the notebook
builds them as a natural extension of "a function that returns a function," the same mechanism
generators rely on internally.

## Experiment

**Hypothesis (stated before running):** for a large `N`, materializing a full list of `N` computed
values will cost measurably more memory than a generator over the same logical sequence, because
the generator only ever holds one paused function frame while the list holds every value
simultaneously. **Setup:** `N = 10,000,000`; build a full list of `i*i` for `i in range(N)`
versus a generator expression over the identical computation; measure with `sys.getsizeof` and
`time.perf_counter`. **Result:** confirmed — 84.97 MB for the list vs. 200 bytes for the generator
object (pasted above), a ratio of over 445,000×. **Limitations:** `sys.getsizeof` on a container
reports the container's own overhead, not necessarily every referenced object's size transitively
— for a list of small cached ints this measurement is close to the true marginal cost, but it
would undercount a list of large mutable objects each holding their own separate allocations. The
timing numbers are single-run wall-clock measurements on one machine, not averaged across many
runs, and are meant to show the right *order of magnitude and direction*, not a precise benchmark.

## Failure modes

- **Exhausting a generator — it can only be iterated once.** Reproduced for real in
  [`manual_iterator_protocol.py`](manual_iterator_protocol.py):

  ```
  Exhaustion failure mode: a generator can only be iterated once
  first pass: [1, 2, 3]
  second pass on the SAME generator object: []
  ```

  Once a generator's `__next__` has raised `StopIteration`, it stays exhausted — calling `list()`
  or looping over it again returns nothing, silently, with no error. This differs sharply from a
  list, which can be iterated any number of times. Code that assumes "I can loop over this twice"
  needs either a fresh generator each time (call the generator function again) or an actual list.
- **Accidentally materializing a generator into a list, defeating the memory benefit.** Writing
  `list(read_large_file(path))` instead of `for line in read_large_file(path):` reintroduces the
  exact full-list-in-memory cost the generator existed to avoid — the generator's laziness is only
  useful as long as something actually consumes it lazily. This is an easy, silent mistake: the
  code still runs and still produces the right *values*, just without the memory guarantee that
  was the entire point of choosing a generator over a list comprehension in the first place.

## Real-world usage

- **Streaming large datasets.** Reading a dataset far larger than available RAM (a multi-GB CSV, a
  training corpus) one record/line/batch at a time via a generator is the standard technique —
  connects directly to `06-file-exception`'s file-handling and to data-loading patterns used
  throughout `06-deep-learning`.
- **`itertools`** (`itertools.count`, `itertools.chain`, `itertools.islice`) is a standard-library
  toolkit of generator-based building blocks for exactly the composition patterns this topic
  motivates — infinite sequences, lazy chaining, lazy slicing — all without ever materializing a
  full list.
- **Database cursors and API pagination** commonly return iterators/generators rather than full
  result lists, for the identical reason: the full result set might be too large, too slow to
  fetch entirely, or not yet fully known (a paginated API) before processing needs to start.
- **PyTorch's `DataLoader`/`Dataset`** and generator-based data pipelines in `06-deep-learning`
  apply this exact lazy-materialization idea to training data — batches are produced on demand
  during training, not all held in memory before training starts.

## Mental model

**An iterable knows how to produce an iterator; an iterator knows how to produce the next value,
one at a time, until it says stop — a generator is the language automating the bookkeeping
(`__iter__`/`__next__`, and the "where am I" state) that a hand-written class has to track
explicitly.** Reach for a generator (or keep something as a generator, resisting `list(...)`)
whenever the sequence is large, potentially infinite, or only ever needs to be walked once in
order; reach for an actual list only when the data must be indexed, re-iterated, or is small enough
that the memory cost genuinely doesn't matter.

## Questions to think about

1. `CountUpTo.__next__` raises `StopIteration` based on `self.current >= self.limit`. If two
   separate `for` loops both did `c = CountUpTo(5)` and then looped over `c`, what would the second
   loop print, given `__iter__` returns `self` rather than a fresh iterator — and how does this
   connect to the exhausted-generator failure mode above?
2. The Experiment section shows summing directly from a generator expression (0.267s) was *slower*
   than summing a pre-built list (0.087s), even though the generator used far less memory. Under
   what concrete condition would you accept that extra time cost, and under what condition would
   you not?
3. `read_large_file` in the practical notebook opens the file inside the generator function body,
   after the first `yield` is ever reached (lazily, on the first `next()` call, not when
   `read_large_file(path)` is called). What would go wrong if the file were opened, read
   completely, and closed *before* any `yield` — and does that version still deserve to be called a
   generator in the memory-saving sense this topic cares about?
4. `sys.getsizeof` reported 200 bytes for the generator object regardless of whether `N` was
   10,000 or 10,000,000. Explain concretely why the generator's size is independent of `N`, in
   terms of what the generator object actually stores.
5. Given the exhaustion failure mode, write out the concrete difference between a function that
   returns a generator (`def make_gen(): return (x for x in range(5))`) called fresh each time it's
   needed, versus a module-level generator object created once and reused everywhere it's
   referenced — which pattern would you choose for a function meant to be called from multiple
   places in a program, and why?
