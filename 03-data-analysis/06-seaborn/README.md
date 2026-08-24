# Seaborn

## What you'll learn

Seaborn as a statistical layer built directly on top of Matplotlib — confirmed by checking that
`sns.boxplot(..., ax=ax)` returns the very same Matplotlib `Axes` object passed in — plus the
full practical surface: distribution, categorical, relationship, and matrix (heatmap/pairplot)
plots. This topic documents a deliberate from-scratch scope decision: the "from scratch" version
of a Seaborn bar/count plot is `05-matplotlib`'s raw-`Rectangle` demo, cited rather than
re-derived.

## Why it matters

Seaborn's default aggregation (mean + confidence interval, e.g. in `sns.barplot`) is compact and
common in dashboards — which is exactly what makes it dangerous: an aggregated plot can look
perfectly unremarkable while hiding a bimodal or otherwise non-unimodal underlying distribution.
This topic's Failure mode demonstrates that concretely and shows the standard defense — overlay
the raw points on the same Axes.

## Prerequisites

- `05-matplotlib` (Seaborn draws through Matplotlib's Figure/Axes API and returns Matplotlib
  Axes objects — the Figure/Axes hierarchy from that topic is the foundation this one builds on)

## What you'll build

- A confirmed demonstration that `sns.boxplot(..., ax=ax)` returns the exact same `ax` object
  passed in, enabling further customisation with plain Matplotlib calls
- A real, executed reproduction of `sns.barplot`'s mean+CI hiding a genuinely bimodal
  300-point distribution (two clusters near 10 and near 50 averaging to an unoccupied ≈30), fixed
  by overlaying `sns.stripplot`'s raw points on the same Axes

See [`notes.md`](notes.md) for the full write-up including real captured output, and
[`seaborn_basics.ipynb`](seaborn_basics.ipynb) (all cells executed) for the practical tour —
distribution plots, categorical plots, relationship plots, heatmaps, pair plots, themes/palettes,
and the Conceptual-Foundation/failure-mode sections above.

## Where it shows up in real systems

`sns.heatmap` on a correlation matrix is one of the most common first steps in real EDA before
modeling. The aggregation-hides-distribution failure mode covered here is a recurring risk in
A/B test dashboards and per-segment metric reports — a result that's actually bimodal (e.g. two
very different user cohorts averaged together) looks like one clean number in a bar chart and
can hide the real finding entirely.

## What's next

`07-sqlite` — querying relational data directly, including the indexing intuition behind why a
query beats a full linear scan.