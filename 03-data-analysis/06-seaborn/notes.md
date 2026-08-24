# Seaborn

## Problem

Producing a good statistical plot from raw Matplotlib requires manually implementing the
statistics first — binning for a histogram, kernel density estimation for a smooth distribution,
computing a mean and a confidence interval for a grouped bar chart, fitting a regression line —
*before* any drawing happens, and separately handling how to facet or color the result by a
categorical column. Seaborn exists to remove that repeated statistical-computation step for the
common cases (a distribution, a category comparison, a correlation matrix, a pairwise
relationship), while still handing back a plain Matplotlib object for anything Seaborn doesn't
cover.

## Intuition

If Matplotlib is a set of drawing primitives (draw a line, draw a rectangle, draw a point),
Seaborn is a set of *statistical recipes* built from those primitives, pre-wired to work directly
with a Pandas `DataFrame`'s columns: "give me the distribution of this column, split by that
column" (`sns.histplot(data=df, x=..., hue=...)`) instead of manually binning the data and
looping over categories to draw one histogram per group. Seaborn decides the statistics and the
layout; Matplotlib still does the actual pixel-level drawing underneath.

## Why simpler approaches fail

Building a grouped-mean-with-confidence-interval bar chart directly in Matplotlib means computing
each group's mean and confidence interval by hand (bootstrap or normal-approximation), then
manually positioning grouped bars with the correct offsets and drawing error bars at the right
height — a categorical scatter plot with jitter means writing the jitter-offset logic yourself. None
of this is hard in isolation, but re-deriving it for every plot, every project, means re-testing
the same statistical/layout logic repeatedly instead of writing it once. Seaborn exists so this
statistical-plus-layout logic is implemented once, correctly, with sensible defaults, while still
returning the underlying Matplotlib Axes for further customisation when the defaults aren't
enough (see below).

## Conceptual foundation

*(Substituting for "Mathematical foundation" per the template's documented substitution
allowance — this topic's foundation is Seaborn's relationship to Matplotlib's object model, not
a numeric derivation.)*

**Seaborn as a statistical layer on top of Matplotlib.** Every Seaborn plotting function does its
statistical work first — binning, KDE estimation, mean+CI aggregation, regression fitting — and
then draws the result through Matplotlib's Figure/Axes API underneath (see
`../05-matplotlib/notes.md` for that hierarchy). Almost every Seaborn function **returns the
Matplotlib `Axes` object it drew onto** (or a `FacetGrid`/`PairGrid` wrapping one or more Axes) —
confirmed directly: `sns.boxplot(data=tips, x='day', y='total_bill', ax=ax)` returned
`<class 'matplotlib.axes._axes.Axes'>`, and that returned object *was* the same `ax` passed in
(`returned is ax` → `True`). This is exactly why `ax=axes[0]` works as an argument throughout
this topic's notebook, and why any Seaborn result can be further customised with plain Matplotlib
calls (`ax.set_title(...)`, `ax.axhline(...)`, …) — the two libraries share one object model, not
two separate ones bridged by conversion.

## Algorithm

What a Seaborn categorical/statistical plotting call does, generically:
1. Group the input `DataFrame` by the requested categorical column(s) (`x`/`hue`/`col`/`row`).
2. Compute the relevant statistic per group — a mean and confidence interval (`barplot`), a
   kernel density estimate (`kdeplot`), a set of quartiles (`boxplot`), a fitted regression line
   (`regplot`).
3. Draw the result onto a Matplotlib Axes using Matplotlib's own primitives underneath (bars,
   patches, lines) — the same primitives covered in `../05-matplotlib/notes.md`.
4. Return that Axes (or a Grid object wrapping several Axes) so the caller can continue
   customising it with plain Matplotlib calls.

## From-scratch implementation — N/A (documented)

Seaborn *is* the higher-level convenience layer described in the Conceptual Foundation above —
there is no meaningfully "lower level" reimplementation to build here that isn't just the
Matplotlib topic's raw-`Rectangle` bar chart demo (`../05-matplotlib/notes.md`, "From-scratch
implementation"). That demo already shows exactly what a categorical Seaborn plot such as
`sns.barplot`/`sns.countplot` automates underneath: computing bar geometry and creating
`Rectangle` patches on a Matplotlib Axes. Re-deriving it again here would duplicate that section
rather than add insight — this is a deliberate scope decision, documented per the plan's
allowance for a topic where the "from scratch" version is genuinely just the prerequisite topic's
demo, not an omission.

## Practical implementation

`seaborn_basics.ipynb` covers the practical surface: distribution plots (`histplot`, `kdeplot`,
`ecdfplot`), categorical plots (`boxplot`, `violinplot`, `stripplot`, `barplot`, `countplot`),
relationship plots (`scatterplot`, `regplot`, `lmplot`), matrix plots (`heatmap` for correlation
matrices and pivoted frequency tables), `pairplot`, and themes/palettes — plus, added in this
pass, the Conceptual-Foundation `Axes`-return demonstration above and the aggregation-hides-data
failure-mode demonstration below. (This pass also fixed a pre-existing bug surfaced by executing
every cell: `sns.palplot()` does not accept an `ax` keyword argument in this environment's
Seaborn version — the palette-swatch cell was rewritten to draw swatches with `ax.imshow()`
directly, which does compose with an existing multi-panel `Axes` grid.)

## Experiment

**Hypothesis:** `sns.barplot`'s default mean+CI aggregation can look like an unremarkable single
value while the underlying data is actually bimodal — and overlaying the raw points on the same
Axes will make the bimodality visible where the bar alone did not.

**Setup:** 300 synthetic values for one category, drawn from two well-separated normal
subpopulations (`N(10, 1.5)`, `N(50, 1.5)`, 150 points each) concatenated into a single
`'category': 'A'` column — a barplot's mean sees one category, one bar.

**Actual result:**

```
Mean of the aggregated bar: 29.97
That single mean (~30) does not correspond to any real observation --
every data point clusters near 10 or near 50, none near 30.
```

The `sns.barplot` alone rendered one bar at ≈30 with a tight confidence interval (both
subpopulations have low individual variance, so the CI around the *mean* is narrow even though
the *data* is nowhere near that mean) — a plot that reads as "confident, unremarkable single
value." Overlaying `sns.stripplot`'s raw points on the same Axes immediately revealed two tight
clusters near 10 and near 50, with a visible gap around the plotted mean.

**Interpretation:** confirms the conceptual risk concretely — the aggregated statistic (mean+CI)
is a valid summary of the numbers, but is actively misleading about the *shape* of the
underlying distribution whenever that shape is not unimodal-and-roughly-symmetric.

**Limitations:** this is a deliberately extreme synthetic case (two cleanly separated clusters)
chosen to make the effect unambiguous; real bimodality is often subtler and the raw-point overlay
correspondingly less immediately obvious, though the same diagnostic (always look at the raw
points, not just the aggregate) still applies.

## Failure modes

- **Seaborn's default aggregation hides the underlying data distribution.** Demonstrated above:
  a 300-point bimodal distribution (two clusters near 10 and near 50) aggregates to a single,
  confident-looking bar at ≈30 under `sns.barplot`'s default mean+CI — a value that corresponds
  to *no actual observation* in the data. The fix demonstrated: overlay `sns.stripplot`'s raw
  points on the same Axes (`ax=` shared between both calls, exploiting the Axes-return property
  from the Conceptual Foundation above) to make the true shape visible alongside the summary
  statistic.

## Real-world usage

`sns.barplot`/similar aggregating plots are extremely common in dashboards and reports precisely
because they're compact — one bar per category — which is exactly what makes the hidden-bimodality
failure mode dangerous in practice: A/B test results, per-segment metrics, or before/after
comparisons that are actually bimodal (e.g. "two different user cohorts behaving very
differently, averaged together") will look like one clean number in a `barplot` and hide the real
finding entirely. The standard defense — used throughout real EDA workflows — is exactly the one
demonstrated here: never trust an aggregated categorical plot without also looking at
`stripplot`/`swarmplot`/`histplot` on the same data at least once.

## Mental model

Seaborn computes the statistics, Matplotlib draws the pixels, and the returned object is *always*
a real Matplotlib Axes you can keep customising — which is also exactly why a Seaborn aggregate
plot can lie by omission: the statistic it computed (a mean, a KDE, a regression fit) is real and
correctly computed, but a single summary number cannot, by construction, tell you whether the
data behind it was unimodal or not — you have to look at the raw points to know.

## Questions to think about

1. `sns.boxplot` shows quartiles rather than a mean+CI. Would a boxplot have revealed the
   bimodal structure in the failure-mode demo as clearly as the raw-point overlay did? Why or
   why not?
2. The failure-mode demo used `sns.stripplot` with `jitter` to overlay raw points. What would
   `sns.swarmplot` (non-overlapping point placement) show differently on the same 300-point
   bimodal data that `stripplot`'s random jitter might not?
3. `sns.barplot`'s confidence interval was described as "narrow... even though the data is
   nowhere near that mean." Walk through why a bimodal distribution with two low-variance
   clusters can still produce a *narrow* CI around a mean that sits between both clusters.
4. If `sns.histplot(data=bimodal_df, x='value')` had been the first plot drawn instead of
   `sns.barplot`, would the bimodality failure mode have occurred at all? What does that suggest
   about which Seaborn plot type to reach for first during initial EDA on an unfamiliar column?
5. The Conceptual Foundation demonstrated that `sns.boxplot(..., ax=ax)` returns the same `ax`
   object passed in. What does that guarantee let you do that calling `sns.boxplot(...)` with no
   `ax` argument at all would not?
