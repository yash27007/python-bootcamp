# Projects

Small end-to-end projects that apply the concepts taught in the earlier sections of this course.

> Projects are the best way to learn. Pick one that interests you, build it, break it, and rebuild it better.

---

## Beginner Projects

| # | Project | Skills | Dataset |
|---|---------|--------|---------|
| 01 | [Titanic Survival EDA](./beginner/01-titanic-eda/) | Pandas, Matplotlib, Seaborn | [Kaggle Titanic](https://www.kaggle.com/c/titanic) |
| 02 | [Iris Flower Classifier](./beginner/02-iris-classifier/) | scikit-learn, classification | Built-in sklearn |
| 03 | [House Price Prediction](./beginner/03-house-price-prediction/) | Regression, feature engineering | [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| 04 | [Student Performance Analysis](./beginner/04-student-performance/) | EDA, visualisation | [UCI ML Repo](https://archive.ics.uci.edu/ml/datasets/Student+Performance) |

---

## Larger / Production-Grade Projects

Bigger, production-style ML projects (full pipelines, MLOps, deployment) live in a separate repo so this course stays focused on learning material rather than sprawling project code:

**→ [github.com/yash27007/ml-spi](https://github.com/yash27007/ml-spi)**

---

## How to Contribute a Beginner Project

1. Pick an idea or open an [issue](https://github.com/yash27007/python-bootcamp/issues) to propose one
2. Create your project folder under `beginner/`
3. Use the template below
4. Open a PR — we'll review and merge!

### Project Folder Template

```
projects/beginner/XX-your-project/
├── README.md          ← objective, dataset, approach, results
├── notebook.ipynb     ← full exploratory notebook
├── requirements.txt   ← any extra packages (if needed)
└── data/
    └── .gitkeep       ← add raw data or a download script
```

### README Template for a Project

```markdown
# Project Name

**Difficulty:** Beginner
**Domain:** Classification / Regression / EDA
**Dataset:** Name + link

## Objective
One paragraph describing what problem this solves.

## Approach
- Step 1
- Step 2

## Results
| Metric | Value |
|--------|-------|
| Accuracy | 0.94 |

## How to Run
\`\`\`bash
uv sync
jupyter lab notebook.ipynb
\`\`\`
```
