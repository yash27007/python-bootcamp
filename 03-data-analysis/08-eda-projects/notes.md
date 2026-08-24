# EDA Projects

## Scope note — why this topic doesn't follow the full 12-section template

This topic is a **synthesis/capstone**, not a new concept. Every technique used across the three
projects below — array/vectorised operations, `DataFrame` filtering and `groupby`, statistical
plotting, correlation heatmaps, relational querying — was already introduced, derived, and
from-scratch-implemented in this section's earlier topics (`01-numpy` through `07-sqlite`). Forcing
a fabricated "From-scratch implementation" or "Mathematical foundation" section here would either
duplicate that earlier work verbatim or invent depth that doesn't exist for "run an end-to-end EDA
on a real dataset." That would violate the same discipline `06-seaborn` and `07-sqlite` already
applied when a section legitimately doesn't need a component (see `06-seaborn/notes.md`'s
documented from-scratch scope decision).

Per the plan's explicit judgment-call allowance, this notes.md instead uses a **reduced
structure**: Problem, a per-project synthesis mapping back to where each technique's real
foundation lives, Failure modes actually encountered in these projects, Real-world usage, Mental
model, and Questions. Where the full template would ask "From-scratch implementation" or
"Mathematical foundation," the answer is: **that work already happened** — in `01-numpy` (array
statistics), `02-pandas`/`03-data-manipulation` (`groupby`, missing-value handling, merges),
`05-matplotlib`/`06-seaborn` (distribution and correlation plotting, including the from-scratch
`Rectangle`-patch bar chart and the aggregation-hides-distribution failure mode), and `07-sqlite`
(the B-tree index derivation and measured linear-scan-vs-indexed comparison). This topic's job is
to show those tools **composed together** on real, messy data, not re-derive any one of them.

## Problem

Every prior topic in this section works on one tool at a time, on data that's already close to
analysis-ready. Real datasets never arrive that way — they have missing values, duplicate rows,
wrong dtypes, outliers, and skewed distributions, and answering any real question about them
("does class affect flight price?", "why do some apps have a rating but no reviews?", "which
chemical properties correlate with wine quality?") requires *all* of cleaning, transformation,
visualization, and statistical reasoning in sequence — not any one of them in isolation. The
reason to do a full EDA pass before jumping straight to modeling is that a model trained on
unexamined data silently inherits every one of those problems: a model can't distinguish "this
column is genuinely `NaN`-heavy because it's rarely collected" from "this column is missing
because of a scraping bug," but a human doing EDA first can catch the difference and decide what
to do about it before it becomes a modeling artifact.

## The three projects, and what each ties back to

### Wine Quality (`wine-quality/redwine.ipynb`, `whitewinequality.ipynb`)

UCI's red/white wine physicochemical datasets (acidity, sugar, sulfur dioxide, alcohol, …) against
a quality score. Ties back to:
- **`01-numpy`** — the underlying statistics (mean, std, correlation) that every summary and
  heatmap cell is computed from.
- **`02-pandas`** — `.describe()`, boolean filtering, dtype inspection on both wine variants.
- **`06-seaborn`** — `sns.heatmap` on the full feature correlation matrix (the exact recipe
  `06-seaborn/notes.md` describes: bin/aggregate first, draw second), with markdown cells in the
  notebook explicitly interpreting what each correlation value means for quality prediction —
  this is the "Practical implementation, mapped back to the underlying idea" step from
  `AGENTS.md`'s learning philosophy applied to a real dataset instead of a synthetic one.

### Flight Price (`flight-price-prediction/flight_price_prediction_v1.ipynb`, `_v2.ipynb`)

~300,000-row flight booking dataset (Ease My Trip) — airline, route, class, days-left, price. Ties
back to:
- **`02-pandas`/`03-data-manipulation`** — one-hot encoding categorical columns (`airline`,
  `class`, `stops`), concatenation (`pd.concat`) of the encoded columns back onto the frame.
- **`05-matplotlib`/`06-seaborn`** — a `pairplot` over a sampled subset (`n=1000`, sampled rather
  than plotted in full — a real, necessary compromise: `sns.pairplot` on 300k rows would be both
  visually unreadable and slow to render) to inspect pairwise relationships between the numeric
  features and price before any modeling step.
- **`v2` is an intentionally incomplete earlier pass** (only reaches `.info()`/`.describe()`) kept
  alongside `v1`'s complete analysis — a realistic artifact of iterative EDA, not a defect: real
  analysis work accumulates exploratory drafts, and `v1` is the notebook that reaches a full
  conclusion.

### Google Play Store (`google-playstore-dataset/googleplaystore.ipynb`)

Play Store app listings — category, rating, reviews, installs, price. Ties back to:
- **`02-pandas`** — `.isnull().sum()`-driven missing-value diagnosis, duplicate-row detection and
  removal, dtype coercion (the `Installs`/`Price` columns arrive as strings with `+`/`,`/`$`
  characters that must be stripped before any numeric aggregation is possible).
- **`03-data-manipulation`** — `groupby('Category')['Install'].sum()` to rank categories by total
  installs.
- **`05-matplotlib`/`06-seaborn`** — bar charts of category popularity, distribution plots of
  rating and price, each with an explicit markdown "Observation" cell interpreting the result
  (e.g. "Family category has the most apps... suggests apps targeting families are a larger,
  more established market") — the Problem→Intuition→Explanation chain from `AGENTS.md` applied to
  a real business question instead of an abstract one.

## Failure modes (encountered, not hypothetical)

- **A `to_csv` call assumed a `data/` directory that didn't exist.** The Google Play Store
  notebook's cleaning step wrote `df_copy.to_csv('data/google_cleaned.csv')` against a relative
  path with no `data/` subdirectory present — this is the same class of "looks fine until you run
  it fresh" bug that motivates `AGENTS.md`'s rule that every notebook is executed end-to-end
  before being considered done; it was caught only by actually executing the notebook in this
  retrofit pass, not by reading the code.
- **Raw category columns arriving as unusable strings.** `Installs` values like `"10,000+"` and
  `Price` values like `"$4.99"` are not numeric until the `+`/`,`/`$` characters are stripped —
  a `groupby(...).sum()` or `.mean()` on the raw string column would either error or silently
  produce a nonsensical string-concatenation result depending on dtype inference. This is the same
  "assumptions the simple approach makes" failure `AGENTS.md` asks every topic to surface — here
  it's encountered directly in messy real-world data rather than constructed as a demo.
- **Plotting all 300,000 flight rows in a `pairplot` is impractical**, not just slow — every panel
  becomes an indistinguishable smear of overplotted points. The fix used (`sample(1000,
  random_state=42)`) is the standard real-world compromise: a fixed-seed random sample large
  enough to show the relationship's shape, small enough to render and read.

## Real-world usage

This is what EDA looks like in practice, industry-wide: before any model is trained, an analyst
or ML engineer inspects missingness, dtypes, duplicates, distributions, and correlations, and
writes down what each finding implies — exactly the pattern in all three notebooks' markdown
"Observation" cells. Skipping this step is one of the most common causes of models that perform
well in a notebook and fail in production, because the model was never shown a dataset an analyst
had actually verified was clean.

## Mental model

A from-scratch implementation shows you *how one tool works*; an EDA capstone shows you *how to
decide which tool to reach for, in what order, on data nobody has cleaned for you yet* — the skill
this section has been building toward across NumPy, Pandas, Matplotlib, Seaborn, and SQLite is
exactly the skill exercised, together, in these three projects.

## Questions to think about

1. The Google Play Store `Installs` column required string-cleaning before any aggregation. What
   would `groupby('Category')['Installs'].sum()` have silently done if `Installs` were left as a
   string dtype containing values like `"10,000+"` — error, or produce a wrong-but-plausible-
   looking answer? Which failure mode is more dangerous in practice, and why?
2. The flight price `pairplot` sampled 1,000 of 300,153 rows. What property would a sample need to
   have (beyond just "random") to reliably preserve the pairwise relationships visible in the full
   dataset, and how would you check whether a given sample has that property?
3. Wine quality's heatmap shows feature-to-feature correlations, not feature-to-target causation.
   Pick one strongly correlated pair from that dataset (e.g. alcohol content and quality) and
   describe a plausible confound that would make the correlation misleading as a causal claim.
4. The Google Play Store notebook's `to_csv('data/google_cleaned.csv')` failed on a fresh
   checkout because the `data/` directory didn't exist. What convention (e.g. creating output
   directories at the top of a notebook, using `Path.mkdir(exist_ok=True)`) would make a
   notebook's file-writing cells robust to being run in a clean environment?
5. All three projects end their cleaning steps with an "Observation" markdown cell before moving
   on. What is lost, specifically, if that interpretation step is skipped and the notebook goes
   straight from a chart to the next code cell?
