# 03 – Data Analysis

The daily tools of data science: numerical computing, data manipulation, relational access, and
visualisation — first-principles retrofitted (see each topic's `notes.md` for the
Problem → Intuition → Why-simpler-fails → Foundation → From-scratch → Practical → Experiment →
Failure-modes chain per `AGENTS.md`).

| # | Topic | Status | Description |
|---|-------|--------|-------------|
| 01 | [NumPy](./01-numpy/) | ✅ Complete | Arrays, vectorised ops, broadcasting, linear algebra, random — from-scratch vectorisation speedup measured against a pure-Python loop |
| 02 | [Pandas](./02-pandas/) | ✅ Complete | Series, DataFrame, loc/iloc, filtering, missing values, str/dt |
| 03 | [Data Manipulation](./03-data-manipulation/) | ✅ Complete | GroupBy, merge/join, concat, pivot, melt, rolling windows |
| 04 | [Reading Data](./04-data-reading/) | ✅ Complete | CSV, JSON, Excel, SQLite, Parquet, HTML tables |
| 05 | [Matplotlib](./05-matplotlib/) | ✅ Complete | Line, bar, scatter, histogram, pie, subplots, annotations — from-scratch `Rectangle`-patch bar chart |
| 06 | [Seaborn](./06-seaborn/) | ✅ Complete | histplot, boxplot, violin, heatmap, pairplot, regplot — measured failure mode: aggregation hiding a bimodal distribution |
| 07 | [SQLite3](./07-sqlite/) | ✅ Complete | CREATE, CRUD, GROUP BY, JOIN, Pandas integration, parameterised queries — B-tree index intuition derived and measured (linear scan vs. indexed query, 100k rows); real SQL-injection vulnerable-vs-fixed demo |
| 08 | [EDA Projects](./08-eda-projects/) | ✅ Complete | Capstone synthesis — Wine Quality, Flight Price, Google Play Store: three real end-to-end EDA projects composing every tool above |

## Prerequisites

- Section 01 (Python Foundation)
- Section 02 (Statistics)

## Running the Notebooks

```bash
# From repo root
uv sync
source .venv/bin/activate
jupyter lab
```

Navigate to `03-data-analysis/` and open any `.ipynb` file.

> All notebooks use **built-in or auto-downloaded datasets** — no manual data downloads required.
