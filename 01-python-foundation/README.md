# 01 – Python Foundation

Core Python skills needed before diving into data science and machine learning.

| # | Topic | Status | Description |
|---|-------|--------|-------------|
| 01 | [Basics](./01-basics/) | ✅ Complete | Syntax, variables, data types, operators, f-strings |
| 02 | [Control Flow](./02-control-flow/) | ✅ Complete | if/elif/else, for/while loops, comprehensions, match/case |
| 03 | [Data Structures](./03-data-structures/) | ✅ Complete | Lists, tuples, sets, dictionaries, Counter, deque |
| 04 | [Functions](./04-functions/) | ✅ Complete | Functions, *args/**kwargs, scope, lambda, map/filter |
| 05 | [Modules & Packages](./05-modules-packages/) | ✅ Complete | Imports, os/pathlib/json/re/itertools, uv |
| 06 | [File Handling & Exceptions](./06-file-exception/) | ✅ Complete | File I/O, CSV/JSON, try/except, custom exceptions |
| 07 | [OOP](./07-oops/) | ✅ Complete | Classes, inheritance, encapsulation, polymorphism |
| 08 | [Advanced Concepts](./08-advanced-concepts/) | ✅ Complete | Iterators, generators, closures, decorators |
| 09 | [Logging](./09-logging/) | ✅ Complete | Python `logging` module, handlers, formatters |
| 10 | [Multi-threading & Multiprocessing](./10-multithreading/) | ✅ Complete | Threads, processes, GIL, use-cases |
| 11 | [Memory Management](./11-memory-management/) | ✅ Complete | Allocation, deallocation, garbage collection |
| 12 | [Flask](./12-flask/) | ✅ Complete | REST APIs with Flask |
| 13 | [Streamlit](./13-streamlit/) | ✅ Complete | Interactive data apps with Streamlit |

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
