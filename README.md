# Zero to Hero: Python, Data Science, ML & AI

A structured, hands-on course repository covering everything from Python basics to MLOps and LLM applications — following the [Complete Machine Learning, NLP Bootcamp (MLOps & Deployment)](https://www.udemy.com/course/complete-machine-learning-nlp-bootcamp-mlops-deployment/) curriculum, extended with GenAI and LLM application projects.

> A one-stop learning resource for ML, DS, and AI. If you find this helpful, please leave a ⭐

---

## Course Roadmap

```
01-python-foundation  →  02-statistics  →  03-data-analysis
       ↓
04-feature-engineering  →  05-machine-learning  →  06-deep-learning
       ↓
07-nlp  →  08-mlops-deployment  →  projects (beginner → advanced → ai-llm)
```

---

## Curriculum

| Section | Topic | Status | Contents |
|---------|-------|--------|----------|
| [01](./01-python-foundation/) | **Python Foundation** | ✅ Complete | Basics, Control Flow, Data Structures, Functions, Modules, File I/O, OOP, Advanced, Logging, Threading, Memory, Flask, Streamlit |
| [02](./02-statistics/) | **Statistics** | ✅ Complete | Descriptive Stats, Probability, Inferential Statistics |
| [03](./03-data-analysis/) | **Data Analysis** | 🚧 Coming soon | NumPy, Pandas, Data Manipulation, Matplotlib, Seaborn, SQLite, EDA Projects |
| [04](./04-feature-engineering/) | **Feature Engineering** | ✅ Complete | Missing Values, Outliers, Encoding, Imbalanced Data |
| [05](./05-machine-learning/) | **Machine Learning** | 🚧 Coming soon | Linear→Polynomial Regression, Regularization, Logistic, SVM, Naive Bayes, KNN, Trees, Ensembles, Boosting, Unsupervised, Anomaly Detection |
| [06](./06-deep-learning/) | **Deep Learning** | 🚧 Coming soon | ANN, CNN, RNN, LSTM/GRU, Attention & Transformers |
| [07](./07-nlp/) | **NLP** | 🚧 Coming soon | Text Preprocessing, BOW/TF-IDF, Word2Vec, Deep Learning NLP |
| [08](./08-mlops-deployment/) | **MLOps & Deployment** | 🚧 Coming soon | Docker, Git, End-to-End Projects, MLflow/DVC, BentoML |
| [projects](./projects/) | **Projects** | 🏗️ Building | Beginner EDA → Intermediate ML → Advanced DL → AI/LLM (RAG, Agents) |

---

## Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/getting-started/installation/) (fast Python package manager)

### Setup (one-time)

```bash
git clone https://github.com/yash27007/python-bootcamp.git
cd python-bootcamp

# Install all dependencies and create the virtual environment
uv sync

# Activate the virtual environment
source .venv/bin/activate        # Linux / macOS
# .venv\Scripts\activate         # Windows
```

### Running Notebooks

```bash
# After activating .venv
jupyter lab
```

Navigate to any section folder and open a `.ipynb` file.

### Running Scripts

```bash
# Example: multi-threading demo
python 01-python-foundation/04-multi-threading/multi-threading.py

# Example: Flask API
python 01-python-foundation/05-flask/app.py

# Example: Streamlit app
streamlit run 01-python-foundation/06-streamlit/main.py
```

---

## Section Structure

Each section follows a consistent layout:

```
XX-section-name/
├── README.md               ← what's covered and prerequisites
├── notes.md                ← theory notes with formulas and diagrams
└── topic.ipynb             ← practical code with comments
```

For practical-only topics (scripts, web apps) there may be `.py` files instead of notebooks.

---

## What's Inside Each Section

### 01 – Python Foundation
13 sub-topics from basic syntax to web frameworks: variables, control flow, data structures, functions, modules, file I/O, OOP, iterators/generators, logging, concurrency, memory management, Flask, Streamlit.

### 02 – Statistics
The mathematical bedrock of ML:
- **Descriptive Statistics** – central tendency, dispersion, correlation
- **Probability** – rules, distributions, Bayes' theorem, CLT
- **Inferential Statistics** – CIs, hypothesis testing, t-tests, ANOVA, chi-square

### 03 – Data Analysis *(coming soon)*
NumPy, Pandas, data manipulation, reading from multiple sources, Matplotlib, Seaborn, SQLite, and three real-world EDA projects (Red Wine, Flight Price, Google Play Store).

### 04 – Feature Engineering
Preparing raw data for ML models:
- Handling missing values (mean/median/KNN/MICE imputation)
- Detecting and treating outliers (IQR, Z-score, Winsorization)
- Encoding categorical features (label, one-hot, ordinal, target)
- Dealing with class imbalance (SMOTE, class weights)

### 05 – Machine Learning *(coming soon)*
17 sub-topics covering every major algorithm: linear through polynomial regression, regularisation, logistic regression, SVM, Naïve Bayes, KNN, decision trees, random forest, AdaBoost, gradient boosting, XGBoost, PCA, K-Means, DBSCAN, and anomaly detection.

### 06 – Deep Learning *(coming soon)*
ANN from scratch (activations, optimisers, dropout), CNN for images, RNN for sequences, LSTM/GRU for long-range dependencies, and the full Transformer architecture (self-attention through decoder).

### 07 – NLP *(coming soon)*
Text preprocessing with NLTK, classical feature extraction (BOW, N-Grams, TF-IDF), dense word embeddings (Word2Vec, AvgWord2Vec), deep learning-based NLP, and four end-to-end projects.

### 08 – MLOps & Deployment *(coming soon)*
Docker, Git/GitHub, two full-scale projects (Student Performance + Sensor Fault Detection with MongoDB/MLflow/DVC/GitHub Actions), and BentoML for serving models as APIs.

---

## Dependencies

All dependencies are managed via [uv](https://docs.astral.sh/uv/). The `pyproject.toml` at the root includes everything needed:

- **Core:** numpy, pandas, scipy, statsmodels
- **Visualisation:** matplotlib, seaborn
- **ML:** scikit-learn, imbalanced-learn
- **Notebooks:** jupyter, ipykernel
- **Web:** flask, streamlit
- **Utilities:** requests, beautifulsoup4

A single `uv sync` installs them all.

---

## Projects

End-to-end projects organised by difficulty — from beginner EDA to production MLOps and LLM applications.

| Tier | Projects |
|------|---------|
| [Beginner](./projects/beginner/) | Titanic EDA, Iris Classifier, House Price Prediction, Student Performance |
| [Intermediate](./projects/intermediate/) | Customer Churn, Fraud Detection, Time Series, Sentiment Analysis |
| [Advanced](./projects/advanced/) | Image CNN, Transformer from Scratch, MLOps Pipeline, Sensor Fault Detection |
| [AI / LLM](./projects/ai-llm/) | Document Q&A (RAG), Chatbot with Memory, Multi-doc Summariser, LLM Fine-tuning |

See [projects/README.md](./projects/README.md) for full details and how to add your own project.

---

## Resources

Curated free books, blogs, courses, cheatsheets, and YouTube channels to go deeper:

→ **[RESOURCES.md](./RESOURCES.md)**

Highlights:
- [ISLR](https://www.statlearning.com) — free ML textbook PDF
- [fast.ai](https://course.fast.ai) — best practical DL course
- [Karpathy's Zero to Hero](https://karpathy.ai/zero-to-hero.html) — build GPT from scratch
- [Jay Alammar's Blog](https://jalammar.github.io) — best visual transformer explainers
- [HuggingFace Learn](https://huggingface.co/learn) — NLP, LLM, Agents courses

---

## Contributing

Contributions are welcome! Whether it's fixing a typo, adding examples, filling in a "coming soon" section, or adding a project:

→ **[CONTRIBUTING.md](./CONTRIBUTING.md)**

Quick steps:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/add-numpy-section`)
3. Commit your changes (follow the notebook conventions in CONTRIBUTING.md)
4. Open a pull request using the PR template

GitHub issue templates are available for [bug reports](./.github/ISSUE_TEMPLATE/bug-report.md) and [content requests](./.github/ISSUE_TEMPLATE/content-request.md).

---

## License

[MIT License](./LICENSE)
