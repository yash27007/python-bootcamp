# Matplotlib

## What you'll learn

The Figure/Axes hierarchy underneath every Matplotlib plot, why `plt.plot()`'s implicit
"current axes" behavior breaks down once more than one Axes is alive, and a manual bar chart
built directly from `matplotlib.patches.Rectangle` — showing exactly what `ax.bar()` automates
(geometry + view-limit autoscaling) — plus the full practical surface: line/bar/scatter/
histogram/pie plots, multi-panel dashboards, styling, and saving figures.

## Why it matters

Nearly every other Python plotting tool — Pandas' `.plot()`, Seaborn (`06-seaborn`) — is built as
a layer on top of Matplotlib's Figure/Axes object model, so understanding it here pays off
everywhere else. This topic's Failure modes (implicit "current axes" state silently changing
inside a loop, a `figsize`/`dpi` mismatch making a saved figure unreadably small) are both real,
executed reproductions of bugs that produce no error — just a wrong or illegible plot.

## Prerequisites

- `02-pandas` (plotted data typically comes from a DataFrame/Series)

## What you'll build

- A manual bar chart built from raw `Rectangle` patches added via `ax.add_patch()`, rendered
  side-by-side with `ax.bar()`'s output on the same data
- A real, executed reproduction of "current axes" silently changing after creating a new figure
  (`plt.gca()` pointing at a different Axes with no call naming it), and the explicit-`ax.`-
  reference fix
- A real, executed measurement of the same figure saved at `dpi=20` vs `dpi=150` — 160×100 vs
  1200×750 actual pixels for an identical `figsize`

See [`notes.md`](notes.md) for the full write-up including real captured output, and
[`matplotlib_basics.ipynb`](matplotlib_basics.ipynb) (all cells executed) for the practical tour
— the OO `fig, ax = plt.subplots()` pattern, line/bar/scatter/histogram/pie plots, a 4-panel
dashboard, styling, annotations, `fig.savefig`, and the from-scratch/failure-mode sections above.

## Where it shows up in real systems

The Figure/Axes model underpins every multi-panel report and dashboard figure. Getting the
current-axes gotcha right matters most in scripts/notebooks that generate many plots in a loop
(one figure per experiment run, or per model); the `dpi` gotcha matters for any figure destined
for a document, slide, or paper — a common cause of "why does my saved chart look blurry" bug
reports.

## What's next

`06-seaborn` — the statistical-plotting layer built directly on top of the Figure/Axes model
covered here; Seaborn's plotting functions return the same Matplotlib `Axes` objects.