# Logging

## Problem

Once a program is running somewhere other than a developer's own terminal — a server, a scheduled
job, someone else's machine — there is no debugger attached and no one watching the screen live.
When something goes wrong (or even when everything is going right and someone just needs to
confirm it), the only evidence available afterward is whatever the program itself recorded while
it ran. The question this topic answers: how does a program record what it did, with enough
detail to diagnose a problem later, without either drowning that record in noise or leaving out
the one line that would have explained the failure?

## Intuition

Compare two ways of tracking what `divide(334, 123)` and `divide(23423, 0)` do inside
[`app.py`](app.py). With `print()`: every message looks the same on the screen — an informational
"here's the result" and a genuine "division by zero happened" both show up as identical-looking
text, with no way to tell them apart programmatically, no way to silence the routine ones while
keeping the important ones, and no record left behind once the terminal is closed. With
`logging`: the same events are tagged by *severity* (`DEBUG` for the routine arithmetic, `ERROR`
for the zero-division), routed to *multiple destinations at once* (the terminal and a file,
simultaneously, from one call), and stamped with *when* and *which logger* produced them — real
output from `app.py`, run for real:

```
2026-08-24 02:05:05-ArithmeticApp-DEBUG-Adding 3 and 5 = 8
2026-08-24 02:05:05-ArithmeticApp-DEBUG-Subracting 34 and 21 = 13
2026-08-24 02:05:05-ArithmeticApp-DEBUG-Multiplying 19 and 13 = 247
2026-08-24 02:05:05-ArithmeticApp-DEBUG-Diving 334 by 123 = 2.7154471544715446
2026-08-24 02:05:05-ArithmeticApp-ERROR-Division by Zero Error
```

Every line carries a timestamp, a logger name, and a level — three pieces of structure `print()`
never provides, and all three turn out to matter the moment more than one person, more than one
module, or more than a few hours of running time are involved.

## Why simpler approaches fail

**`print()` debugging** is genuinely fine for a five-line script run once at a terminal — but it
breaks down for anything that runs unattended or grows past a single file, for four concrete
reasons:

1. **No severity levels.** A `print()` call can't distinguish "routine status update" from
   "something is actually wrong" — both are just text on stdout. Filtering later means grepping
   text and hoping the wording is consistent, not asking "show me only the errors."
2. **Can't be selectively enabled or disabled.** Removing or adding `print()` statements means
   editing and redeploying code. There is no built-in way to say "show me the detailed internals
   of this one module, but nothing else" without hand-adding a flag and threading it through every
   call site.
3. **Not structured.** `print(f"Adding {a} and {b} = {a+b}")` is a one-off string with no
   consistent machine-parseable shape across an entire codebase — one developer's `print`
   statements look nothing like another's, and nothing enforces a shared format for timestamp,
   severity, or source.
4. **Clutters production output, with no separation of concerns.** `print()` always writes to
   stdout — there's no way for one call to simultaneously go to a rotating log file *and* the
   console, at different verbosity levels for each, the way `logging`'s handler system does out of
   the box (see [`logs/logger.py`](logs/logger.py), which routes everything to `logs/app.log`
   rather than the terminal at all).

## Conceptual foundation

*(Substituting for "Mathematical foundation" — there's no derivation; the mechanism to make
explicit is the logging hierarchy: levels, loggers, handlers, and formatters, and how they
compose.)*

**Levels** are an ordered severity scale — `DEBUG < INFO < WARNING < ERROR < CRITICAL`. A logger
configured at `level=logging.WARNING` silently drops every `DEBUG` and `INFO` call made against
it; raising or lowering that one number changes what's recorded everywhere that logger is used,
with zero changes to any call site. This is the direct fix for "can't be selectively enabled" above
— demonstrated for real in [`level_toggle_demo.py`](level_toggle_demo.py):

```
--- with level=WARNING (production-like setting) ---
WARNING:demo:cache miss rate above 30 percent
ERROR:demo:payment gateway timeout

--- after raising the level to DEBUG (same call sites, zero edits) ---
DEBUG:demo:connecting to db at host=10.0.0.5
INFO:demo:request served in 42ms
WARNING:demo:cache miss rate above 30 percent
ERROR:demo:payment gateway timeout
```

The exact same four `logger.debug/info/warning/error(...)` calls appear twice in the source with
no edits between the two blocks — only `logging.getLogger("demo").setLevel(logging.DEBUG)` changed
what got through.

**Loggers** are named channels (`logging.getLogger("ArithmeticApp")`, `logging.getLogger("module1")`)
— the practical notebook's [`main.ipynb`](main.ipynb) shows two loggers (`module1` at `DEBUG`,
`module2` at `WARNING`) coexisting with independent levels, so one module can be noisy for
debugging while another stays quiet, without a single global on/off switch controlling both.

**Handlers** decide *where* a logger's records go — `app.py` attaches both a `FileHandler`
(writes to `app1.log`) and a `StreamHandler` (writes to the console) to the same logger, so one
`logger.debug(...)` call produces output in two places simultaneously, at potentially different
levels per handler.

**Formatters** decide the *shape* of each record — `'%(asctime)s-%(name)s-%(levelname)s-%(message)s'`
is what turns a bare message into the structured, timestamped, leveled, source-tagged line shown
in Intuition above.

## Algorithm

How a call like `logger.debug("...")` is actually processed:

1. The call creates a `LogRecord` carrying the message, the level (`DEBUG`), a timestamp, and the
   logger's name.
2. The logger checks its own effective level. If `DEBUG < logger.level`, the record is dropped
   here and nothing further happens — this is the level-based filtering from Conceptual foundation.
3. If the record survives, it's passed to every handler attached to the logger. Each handler
   independently checks its *own* level (a handler can be stricter than the logger) before
   deciding whether to emit.
4. Each handler that accepts the record runs it through its formatter, then writes the formatted
   string to its destination (a file, the console, a network socket, ...).

## From-scratch implementation

**N/A/minimal — a deliberate judgment call, documented here rather than skipped silently.** Unlike
the iterator protocol in `08-advanced-concepts` (a small mechanism worth rebuilding by hand to
show what it automates), the `logging` module's actual internals — thread-safe record dispatch,
handler/formatter composition, the propagation hierarchy across parent/child logger names — are
themselves already close to the minimal correct implementation of "leveled, routed, formatted
event records." Hand-rebuilding a smaller version (e.g., a class wrapping a `list` of `(level,
message)` tuples with a `min_level` filter) would demonstrate the *level-filtering* idea in
isolation, but would not add insight beyond what the from-scratch-level line already shows:
`level_toggle_demo.py`'s comparison **is** effectively that minimal mechanism, expressed with the
real module instead of a toy reimplementation, because the real module's public behavior at this
scope (level compare, then act) genuinely is the from-scratch version. Building a second, smaller
logger class purely to have "from-scratch code" would not teach anything `level_toggle_demo.py`
doesn't already show directly.

## Practical implementation

- [`app.py`](app.py) — a real multi-function module (`add`/`sub`/`mul`/`divide`) logging at
  `DEBUG` for routine operations and `ERROR` for the caught `ZeroDivisionError`, with both a
  `FileHandler` and `StreamHandler` attached — exact output pasted in Intuition above.
- [`logs/logger.py`](logs/logger.py) / [`logs/test.py`](logs/test.py) — a minimal reusable pattern:
  a shared `logger.py` module configures `logging.basicConfig` once (writing to `logs/app.log`),
  and any other module (`test.py`) does `from logger import logging` to inherit that
  configuration automatically, without repeating the setup.
- [`main.ipynb`](main.ipynb) — multiple independent loggers (`module1` at `DEBUG`, `module2` at
  `WARNING`) demonstrating that level configuration is per-logger, not global.
- [`level_toggle_demo.py`](level_toggle_demo.py) — the from-scratch-substitute comparison above,
  real executed output pasted in Conceptual foundation.

## Experiment

**Hypothesis (stated before running):** raising a logger's level from `WARNING` to `DEBUG`, with
no changes to any `logger.debug/info/warning/error(...)` call site, will change exactly which of
those calls produce visible output — `DEBUG`/`INFO` calls will start appearing that were silent
before, while `WARNING`/`ERROR` calls remain visible throughout. **Setup:**
[`level_toggle_demo.py`](level_toggle_demo.py) — the identical four calls issued twice, once at
each level. **Result:** confirmed exactly, real output pasted above — two lines visible at
`WARNING`, all four visible at `DEBUG`, source unchanged between the two blocks. **Limitations:**
this experiment demonstrates level filtering in isolation on one process; it does not exercise
handler-level filtering (a handler stricter than its logger), multi-process log aggregation, or
log rotation under sustained volume — all real production concerns not measured here.

## Failure modes

- **Logging sensitive data — passwords, tokens, PII.** A `logger.debug(f"login attempt: user={user}
  password={password}")` written for convenience during development is easy to forget to remove,
  and unlike a `print()` statement that vanishes when the terminal closes, a log line written to a
  file or shipped to a log-aggregation service can persist indefinitely, get backed up, get
  indexed for search, and be visible to anyone with log access — turning a debugging convenience
  into a real data-exposure incident. The fix is discipline at the call site (never log raw
  credentials, tokens, or personal data — log an identifier or a redacted form instead), not a
  logging-configuration setting.
- **Log volume overwhelming storage or cost at scale.** A logger left at `DEBUG` in production,
  or a hot code path logging on every iteration of a tight loop, can generate gigabytes of log
  data per hour — this is a real operational cost (storage, log-aggregation-service ingestion
  pricing, and the human cost of a signal buried in noise) directly caused by the level-filtering
  mechanism in Conceptual foundation being set too permissively, or not being used at all in favor
  of unconditional output. The fix is the same mechanism this topic is about: set production levels
  deliberately (`WARNING` or `INFO`, not `DEBUG`), and use `DEBUG` only for the specific
  module/timeframe being actively investigated.

## Real-world usage

- **Web frameworks and APIs** (Flask in `12-flask`, any production service) log request/response
  metadata, errors, and timing at `INFO`/`WARNING`/`ERROR` — the exact level-based triage this
  topic covers is how an on-call engineer filters a flood of request logs down to the handful that
  actually indicate a problem.
- **ML training runs** (`06-deep-learning`) log epoch/loss/metric progress at `INFO` and
  numerically unstable conditions (`NaN` loss, exploding gradients) at `WARNING`/`ERROR` — a long
  unattended training job is exactly the "no debugger attached" scenario from Problem.
- **MLOps pipelines** (`08-mlops-deployment`) rely on structured, leveled logs feeding into
  centralized log-aggregation systems (the same handler/formatter mechanism here, pointed at a
  network destination instead of a local file) so that a failure in one service, among many, can
  be traced without SSHing into a specific machine.

## Mental model

**A log line is a triaged, timestamped, routable fact about what a program did — the level says
how urgent it is, the logger says who's reporting it, the handler says where it goes, and the
formatter says what shape it takes — four independent knobs `print()` collapses into "always show
this text on this one screen, right now."** Set the level deliberately for the environment (loud
in development, quiet in production, temporarily loud for one module under investigation); never
let sensitive data reach a log call, because unlike a closed terminal, a log line can outlive the
process that wrote it.

## Questions to think about

1. `level_toggle_demo.py` changes what's visible by calling `setLevel` — no call site changed. If a
   teammate instead "fixed" noisy production logs by deleting the `logger.debug(...)` calls
   entirely, what capability would that team permanently lose that keeping the calls (and just
   lowering the level when needed) would have preserved?
2. `app.py` attaches both a `FileHandler` and a `StreamHandler` to the same logger. Design a
   concrete scenario where you'd want the file handler to record `DEBUG` and above, but the console
   handler to show only `WARNING` and above — from the Algorithm section, at what step would that
   filtering actually happen, and is it the logger or the handler that would need the stricter
   level?
3. `logs/test.py` does `from logger import logging` to inherit `logs/logger.py`'s configuration.
   What happens if two different modules in the same program both call `logging.basicConfig(...)`
   with different settings — which configuration wins, and what does that imply about where
   `basicConfig` should be called from in a larger, multi-module application?
4. The Failure modes section says logging sensitive data is a discipline problem, not a
   configuration one. Propose one concrete, mechanical safeguard (something checkable in code
   review or by a tool, not "remember not to") that would catch a `password=` argument being passed
   into a log call before it reaches a production log file.
5. A service logs at `INFO` in production and receives 10,000 requests per minute, each producing
   one `INFO` line. Using the Failure-modes reasoning about volume, what would you change first —
   the level, the message content, or the destination — and what would you need to know about how
   those logs get used (searched interactively? aggregated into metrics? rarely read at all?)
   before deciding?
