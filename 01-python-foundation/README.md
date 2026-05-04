# 01 – Python Foundation

Core Python skills needed before diving into data science and machine learning.

| # | Topic | Type | Description |
|---|-------|------|-------------|
| 01 | [OOP](./01-oops/) | Notebook | Classes, inheritance, encapsulation, polymorphism |
| 02 | [Advanced Concepts](./02-advanced-concepts/) | Notebooks | Iterators, generators, memory management |
| 03 | [Logging](./03-logging/) | Notebook + Scripts | Python `logging` module, handlers, formatters |
| 04 | [Multi-threading & Multiprocessing](./04-multi-threading/) | Scripts | Threads, processes, GIL, use-cases |
| 05 | [Flask](./05-flask/) | Scripts | REST APIs with Flask |
| 06 | [Streamlit](./06-streamlit/) | Scripts | Interactive data apps with Streamlit |

## Running the Code

All notebooks and scripts use the **root virtual environment**. From the repo root:

```bash
# First time only
uv sync

# Activate
source .venv/bin/activate

# Open any notebook
jupyter lab
```

> Flask and Streamlit are standalone web apps — run them directly after activating the root venv.
