# Contributing Guide

Thank you for wanting to make this a better learning resource for everyone!  
All skill levels are welcome — from fixing a typo to adding a full project.

---

## Ways to Contribute

| Type | Examples |
|------|---------|
| Fix typos / improve notes | Clearer explanations, grammar, formatting |
| Add code examples | More illustrative examples to notebooks |
| Add a new project | Any difficulty tier (see [projects/README.md](./projects/README.md)) |
| Fill in a "Coming soon" section | Any section marked 🚧 in the curriculum |
| Add resources | Good books, blogs, courses to [RESOURCES.md](./RESOURCES.md) |
| Report a bug | Broken code, wrong output, confusing explanation |

---

## Getting Started

### 1. Fork and clone

```bash
git clone https://github.com/<your-username>/python-bootcamp.git
cd python-bootcamp
```

### 2. Set up the environment

```bash
# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Install all dependencies
uv sync

# Activate the venv
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### 3. Create a branch

```bash
git checkout -b feature/add-numpy-section
# or
git checkout -b fix/typo-in-statistics-notes
```

Use a descriptive branch name:
- `feature/` — new content or section
- `fix/` — bug or error correction
- `docs/` — documentation only

### 4. Make your changes

Follow the conventions below, then commit and push.

### 5. Open a Pull Request

- Fill in the PR template
- Link any related issue (e.g., `Closes #42`)
- Keep PRs focused — one topic per PR

---

## Conventions

### Notebook Style

- **One notebook per sub-topic** (e.g., `numpy_basics.ipynb`)
- Start with a Markdown cell: `# Topic Title` + bullet list of what's covered
- Explain the **why** in Markdown cells, show the **how** in code cells
- Include expected output where helpful
- Use `# comment` to explain non-obvious lines
- End with a **Quick Summary** table + link to next section

### Section Structure

```
XX-section-name/
├── README.md        ← topic table with status, prerequisites
├── notes.md         ← optional theory notes / formulas
└── topic.ipynb      ← practical notebook
```

### Code Style

- PEP 8 formatting (use `ruff` or `black` if you have them)
- Descriptive variable names — no `x1`, `temp2`
- Prefer `pathlib.Path` over `os.path`
- Prefer f-strings over `.format()` or `%`

### Writing Style

- Write for a **beginner who just finished the previous section**
- Prefer simple words over jargon; define jargon when you use it
- Use tables for comparisons
- Use callout boxes for common mistakes (quote block with `> ⚠️ ...`)

---

## Reporting Issues

Open a [GitHub Issue](https://github.com/yash27007/python-bootcamp/issues) with:

- **Bug**: What you expected vs. what happened, Python version, error message
- **Suggestion**: What you'd like to see added and why
- **Question**: Ask in Discussions, not Issues

---

## Code of Conduct

Be kind. Everyone was a beginner once. Constructive feedback only.

---

## Thank You

Every contribution, no matter how small, helps someone learn.  
If this resource helped you, please ⭐ the repo!
