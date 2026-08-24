# Matplotlib

## Problem

Numbers in a table are hard for a human to reason about at scale; a plot converts a column of
numbers into a shape a human's visual system parses almost instantly (a trend, a cluster, an
outlier, a gap). Producing that plot programmatically — reproducibly, from code, as part of an
analysis pipeline rather than by hand in a GUI — needs a library that maps data arrays onto
visual primitives (lines, bars, points, text) on a 2-D canvas. Matplotlib is the foundational
Python library that does this; nearly every other Python plotting tool (Pandas' `.plot()`,
Seaborn — see `../06-seaborn/notes.md`) is built as a layer on top of it.

## Intuition

A plot is a physical drawing: there's the whole sheet of paper (the **Figure**), and there's one
or more individual drawing areas on that sheet (the **Axes** — each with its own x/y coordinate
system, title, and gridlines). `plt.plot()` — the quick, "implicit" interface — is like saying
"draw on whatever sheet and area happen to be open right now" without naming either; the
object-oriented interface (`fig, ax = plt.subplots()`, then `ax.plot(...)`) is like naming the
exact sheet and drawing area you mean, every time — no ambiguity about which one gets the next
line you draw.

## Why simpler approaches fail

`plt.plot()` (and `plt.title()`, `plt.xlabel()`, …) all operate on Matplotlib's notion of the
**current Axes** — a single piece of global, mutable state that changes every time a new Axes is
created (e.g. by `plt.subplots()`) or explicitly selected. This works fine for one quick plot in
one cell. It stops working the moment there is more than one Axes alive at once — inside a loop
that creates several figures, or a function called from multiple places — because "which Axes is
current" is invisible at the call site of `plt.plot()`; nothing in that line of code says *which*
Axes it will land on. A bug where a plotting call silently lands on the wrong (stale) Axes
produces no error — the code runs, a plot appears, and it's simply the wrong plot, which is a far
worse failure than a crash because nothing signals it happened.

## Conceptual foundation

*(Substituting for "Mathematical foundation" per the template's documented substitution
allowance — this topic's foundation is the Figure/Axes object model, not a numeric derivation.)*

**Figure/Axes hierarchy.**

```
Figure  <- the whole canvas (can be saved to a file, has an overall size and dpi)
  └── Axes  <- one plot (a Figure can hold many, in a grid via plt.subplots(rows, cols))
        ├── x-axis / y-axis  <- ticks, tick labels, scale (linear/log)
        ├── title, xlabel, ylabel
        └── Artists  <- the actual drawn objects: Line2D, Rectangle (bars), PathCollection
                         (scatter points), Text
```

A `Figure` is the container; an `Axes` is one coordinate system living inside it; an "Artist" is
any object actually drawn (a line, a bar, a point). Every high-level call — `ax.plot()`,
`ax.bar()`, `ax.scatter()` — is a convenience method that constructs the appropriate low-level
Artist objects and adds them to that Axes.

**Why the object-oriented style is more robust.** `fig, ax = plt.subplots()` returns explicit
references to the Figure and Axes; every subsequent call goes through `ax.` (`ax.plot()`,
`ax.set_title()`, `ax.bar()`) rather than `plt.`, so which Axes receives each call is determined
by which variable name appears in the code, not by invisible global state. This is exactly what
makes multi-panel figures (`fig, axes = plt.subplots(2, 2)`) tractable — each subplot's code
addresses `axes[0, 0]`, `axes[0, 1]`, etc. explicitly, with no ambiguity about which panel is
being drawn on, and it composes correctly inside loops and functions where `plt.plot()`'s
implicit-current-axes behavior would not.

## Algorithm

What `ax.bar()` does, generically, per bar:
1. Compute the bar's rectangle geometry from the category's numeric position (`x`), a `width`,
   and the value (`height`): bottom-left corner at `(x - width/2, 0)`, size `(width, height)`.
2. Construct a `matplotlib.patches.Rectangle` Artist with that geometry and the requested
   colors.
3. Add the Rectangle to the Axes (`ax.add_patch(...)`).
4. After all bars are added, auto-scale the Axes' view limits (`ax.set_xlim`/`set_ylim`) so every
   bar is visible.

## From-scratch implementation

A manual bar chart built directly from `matplotlib.patches.Rectangle`, compared to `ax.bar()`'s
output on the same data (`matplotlib_basics.ipynb`, "3b. From-Scratch"):

```python
from matplotlib.patches import Rectangle

manual_categories = ['Laptop', 'Phone', 'Tablet', 'Watch']
manual_values = [450, 820, 310, 560]
bar_width = 0.6

fig, ax = plt.subplots(figsize=(7, 4))
for i, val in enumerate(manual_values):
    rect = Rectangle((i - bar_width / 2, 0), bar_width, val,
                      facecolor='steelblue', edgecolor='white')
    ax.add_patch(rect)   # ax.bar() does exactly this, once per bar, under the hood

# ax.bar() also auto-scales the view limits to fit its bars -- must be done by hand here
ax.set_xlim(-0.5, len(manual_categories) - 0.5)
ax.set_ylim(0, max(manual_values) * 1.15)
```

Both the manual `Rectangle`-based chart and `ax.bar(manual_categories, manual_values, ...)` on
the same `manual_categories`/`manual_values` data render as visually identical bar charts (both
cells run and rendered in the notebook) — confirming `ax.bar()` is exactly "create one Rectangle
per bar, add it to the Axes, auto-scale the limits," not a categorically different drawing
mechanism. The one piece `ax.bar()` handles that the manual version has to do explicitly is the
view-limit auto-scaling (`set_xlim`/`set_ylim`) — a raw `Rectangle` added via `add_patch()` does
not automatically expand the Axes' view to include it.

## Practical implementation

`matplotlib_basics.ipynb` covers the practical surface: the OO `fig, ax = plt.subplots()`
pattern used throughout, line plots (multi-series, styled), bar charts (grouped and horizontal),
scatter plots (colored by category, with a fitted trend line), histograms (single and
overlapping), a 4-panel dashboard (`plt.subplots(2, 2)`, mixing line/bar/pie in one figure),
style contexts (`plt.style.context(...)`), annotations (`ax.annotate`), and `fig.savefig(...)`
— plus, added in this pass, the from-scratch `Rectangle` bar chart above and the two
failure-mode demonstrations below.

## Experiment

**Hypothesis:** a manual `Rectangle`-based bar chart and `ax.bar()` on the same category/value
data are visually indistinguishable once the manual version's Axes limits are set to match what
`ax.bar()` computes automatically.

**Setup:** the 4-category `manual_categories`/`manual_values` toy data above, plotted both ways
in adjacent cells at the same figure size.

**Actual result:** both renders show four bars at the same x-positions, same widths, same
heights, same colors — the manual version required one extra explicit step (`set_xlim`/
`set_ylim`) that `ax.bar()` performs automatically.

**Interpretation:** confirms the conceptual claim that `ax.bar()` is a thin, well-understood
convenience wrapper around `Rectangle` + view-limit autoscaling, not a fundamentally different
rendering path.

**Limitations:** this toy comparison doesn't cover `ax.bar()`'s additional features (error bars,
log-scale bar handling, stacked bars) which involve more than the geometry shown here.

## Failure modes

- **Mutating shared state via `plt.gca()`/`plt.plot()` unexpectedly in a loop.** Measured:
  after `fig1, ax1 = plt.subplots()` followed by `plt.plot(...)`, `plt.gca() is ax1` was `True`;
  immediately after creating a *second* figure (`fig2, ax2 = plt.subplots()`), `plt.gca() is ax2`
  became `True` with no call naming `ax2` anywhere — "current axes" changed as a side effect of
  merely creating a new figure. A loop that calls `plt.plot()` (rather than an explicit
  `ax.plot()`) after creating several figures will silently plot onto whichever Axes happens to
  be current at that point in the loop, not necessarily the one the code author intended — and
  it produces no error, just a plot on the wrong panel. The fix demonstrated: hold the Axes
  reference explicitly (`fig3, (axA, axB) = plt.subplots(1, 2)`) and call `axA.plot()`/
  `axB.plot()` — this never depends on global "current axes" state.
- **Mismatched figure/dpi settings making saved images unreadable.** Measured: the identical
  `figsize=(8, 5)` figure saved at `dpi=20` produced a **160×100 pixel**, 2,645-byte PNG; saved
  at `dpi=150` it produced a **1200×750 pixel**, 36,329-byte PNG — same on-screen appearance via
  `plt.show()` (`figsize` controls the *displayed* size), radically different actual pixel
  resolution once saved to a file. A figure that "looked fine" in the notebook can be
  unreadably small once exported at a mismatched `dpi`, because `dpi` — not `figsize` alone — is
  what determines the output file's pixel dimensions (`pixels = figsize_inches × dpi`).

## Real-world usage

The Figure/Axes model underpins every multi-panel report and dashboard figure; getting the
current-axes gotcha right matters most in scripts/notebooks that generate many plots in a loop
(e.g. one figure per experiment run, or per model). The `dpi` gotcha matters for any figure
destined for a document, slide, or paper — a plot that looks fine in an interactive session can
be genuinely illegible once exported at a default/low `dpi`, a common cause of "why does my
saved chart look blurry" bug reports.

## Mental model

A Figure is the canvas, an Axes is one coordinate system on it, and every high-level plotting
call (`ax.bar()`, `ax.scatter()`, …) is a convenience wrapper that builds low-level Artists
(`Rectangle`, `Line2D`, …) and adds them to a specific Axes — the object-oriented style
(`fig, ax = plt.subplots()`) is worth the extra explicitness because it names *which* Axes every
call targets, closing off the entire class of "silently drew on the wrong panel" bugs that the
implicit `plt.`-level interface allows.

## Questions to think about

1. The from-scratch `Rectangle` bar chart needed an explicit `ax.set_xlim`/`set_ylim` call that
   `ax.bar()` doesn't. Why does adding a `Rectangle` via `ax.add_patch()` not automatically
   expand the Axes' view the way `ax.plot()`/`ax.scatter()`/`ax.bar()` do?
2. In the current-axes failure-mode demo, `plt.gca()` pointed at `ax2` immediately after `fig2`
   was created, before any plotting call was made on it. What does that imply about *when*
   "current axes" changes — is it tied to plotting calls, or to Axes creation?
3. Given `pixels = figsize_inches × dpi`, what `dpi` would be needed for an `figsize=(8, 5)`
   figure to reach a specific target resolution such as 1920×1200 pixels — and does the 8×5
   aspect ratio actually divide evenly into 1920×1200?
4. Would the current-axes failure mode still be possible if every plotting call in a codebase
   used `ax.` methods instead of `plt.` — i.e. does the object-oriented style eliminate the bug
   class entirely, or only make it easier to avoid?
5. `ax.bar()` computes each bar's rectangle from a numeric x-position and a width. For a bar
   chart with string category labels (`['Laptop', 'Phone', ...]`) rather than numbers, where do
   the numeric x-positions actually come from, and what does `ax.set_xticklabels(categories)` do
   to make the plot show the category names instead of `0, 1, 2, 3`?
