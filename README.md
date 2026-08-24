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
07-nlp  →  08-mlops-deployment  →  09-pytorch  →  10-distributed-data  →  11-generative-ai  →  12-reinforcement-learning
       ↓
13-llms-from-scratch  →  14-multi-agent-systems  →  15-agent-skills-and-mcp  →  projects (beginner)
```

---

## Curriculum

| Section | Topic | Status | Contents |
|---------|-------|--------|----------|
| [01](./01-python-foundation/) | **Python Foundation** | ✅ Complete | Basics, Control Flow, Data Structures, Functions, Modules, File I/O, OOP, Advanced, Logging, Threading, Memory, Flask, Streamlit |
| [02](./02-statistics/) | **Statistics** | ✅ Complete | Descriptive Stats, Probability, Inferential Statistics |
| [03](./03-data-analysis/) | **Data Analysis** | ✅ Complete | NumPy, Pandas, Data Manipulation, Reading Data, Matplotlib, Seaborn, SQLite, EDA Projects |
| [04](./04-feature-engineering/) | **Feature Engineering** | ✅ Complete | Missing Values, Outliers, Encoding, Imbalanced Data |
| [05](./05-machine-learning/) | **Machine Learning** | ✅ Complete | Linear→Polynomial Regression, Regularization, Logistic, SVM, Naive Bayes, KNN, Trees, Ensembles, Boosting, XGBoost, PCA, K-Means, Hierarchical, DBSCAN, Isolation Forest, LOF |
| [06](./06-deep-learning/) | **Deep Learning** | ✅ Complete | ANN, CNN, RNN, LSTM/GRU, Attention & Transformers |
| [07](./07-nlp/) | **NLP** | ✅ Complete | Text Preprocessing, BOW/TF-IDF, Word2Vec, Deep Learning NLP, Transformers & HuggingFace |
| [08](./08-mlops-deployment/) | **MLOps & Deployment** | ✅ Complete | Docker, Git, Testing & CI, Model Packaging & Versioning, MLflow/DagsHub, DVC, BentoML, CI/CD, Monitoring |
| [09](./09-pytorch/) | **PyTorch** | ✅ Complete | Tensors & Autograd, nn.Module & Training Loop, Datasets/DataLoaders & Checkpointing, GPU/Mixed Precision/Profiling |
| [10](./10-distributed-data/) | **Distributed Data** | ✅ Complete | Why Distributed Processing, PySpark Local Mode, Streaming Fundamentals, Kafka |
| [11](./11-generative-ai/) | **Generative AI** | ✅ Complete | GANs, Diffusion Models |
| [12](./12-reinforcement-learning/) | **Reinforcement Learning** | ✅ Complete | MDPs & Bellman Equation, Q-Learning, Policy Gradients |
| [13](./13-llms-from-scratch/) | **LLMs From Scratch** | ✅ Complete | Tokenizer (BPE) From Scratch, Pretraining Objective (Tiny GPT), Instruction Tuning |
| [14](./14-multi-agent-systems/) | **Multi-Agent Systems** | ✅ Complete | Communication Protocols, Orchestration Patterns, Swarm Coordination (PSO) |
| [15](./15-agent-skills-and-mcp/) | **Agent Skills & MCP** | 🚧 In progress | Agent Skills, Model Context Protocol |
| [projects](./projects/) | **Projects** | 🔄 Growing | Beginner: Titanic EDA, Iris Classifier, House Prices, Student Performance (all with notebooks) — bigger projects: [ml-spi](https://github.com/yash27007/ml-spi) |

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
python 01-python-foundation/10-multithreading/multi-threading.py

# Example: Flask API
python 01-python-foundation/12-flask/app.py

# Example: Streamlit app
streamlit run 01-python-foundation/13-streamlit/main.py
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

### 03 – Data Analysis
NumPy, Pandas, data manipulation, reading from multiple sources, Matplotlib, Seaborn, SQLite, and three real-world EDA projects (Red Wine, Flight Price, Google Play Store).

### 04 – Feature Engineering
Preparing raw data for ML models:
- Handling missing values (mean/median/KNN/MICE imputation)
- Detecting and treating outliers (IQR, Z-score, Winsorization)
- Encoding categorical features (label, one-hot, ordinal, target)
- Dealing with class imbalance (SMOTE, class weights)

### 05 – Machine Learning
18 sub-topics covering every major algorithm: linear through polynomial regression, regularisation, logistic regression, SVM, Naïve Bayes, KNN, decision trees, random forest, AdaBoost, gradient boosting, XGBoost, PCA (with eigen decomposition), K-Means (K-Means++, elbow method), hierarchical clustering (Ward linkage, dendrograms), DBSCAN, silhouette analysis, isolation forest, local outlier factor, and DBSCAN-based anomaly detection.

### 06 – Deep Learning
ANN from scratch (activations, optimisers, dropout), CNN for images, RNN for sequences, LSTM/GRU for long-range dependencies, and the full Transformer architecture (self-attention through decoder).

### 07 – NLP
Text preprocessing with NLTK, classical feature extraction (BOW, N-Grams, TF-IDF), dense word embeddings (Word2Vec, AvgWord2Vec), deep learning-based NLP, and Transformers/HuggingFace fine-tuning.

### 08 – MLOps & Deployment
Docker and Git/GitHub fundamentals, testing & CI, model packaging & versioning, MLflow/DagsHub (with DVC) for experiment tracking, BentoML for serving models as APIs, CI/CD pipelines, and monitoring deployed models.

### 09 – PyTorch
First-principles PyTorch, each topic bridged from the from-scratch NumPy MLP in `06-deep-learning`: tensors and reverse-mode autograd, `nn.Module` and the standard training loop, `Dataset`/`DataLoader` pipelines with checkpointing (real crash-and-resume), and the hardware/systems side of training at scale — GPU parallelism, mixed precision, and profiling.

### 10 – Distributed Data
Why one machine stops being enough before any framework syntax: partitioning, shuffle, and fault tolerance (with a real timed single-thread vs. multiprocessing vs. PySpark comparison), real PySpark local-mode jobs, then the streaming equivalent — producer/consumer, backpressure, and partitioned logs — before Kafka (real reviewed code, honestly marked unexecuted since no broker is available in this environment).

### 11 – Generative AI
GANs and diffusion models, first-principles, at toy scale: the adversarial min-max objective derived and a real trained generator/discriminator with a reproduced mode-collapse failure, then diffusion's forward/reverse process derived and a real trained DDPM on the same toy dataset, with a measured too-few-vs-too-many-steps quality/speed tradeoff.

### 12 – Reinforcement Learning
MDPs → Q-learning → policy gradients: the Bellman equation derived and solved by value iteration on a real stochastic grid-world, model-free Q-learning on the same grid with a real measured 85.71% policy match against value iteration, then REINFORCE derived and trained on a continuous-state environment a Q-table can't represent.

### 13 – LLMs From Scratch
Tokenizer → pretraining objective → instruction tuning, first-principles: Byte-Pair Encoding derived and trained from scratch in plain Python with a real round-trip check, causal self-attention derived and a real 21k-parameter TinyGPT actually trained and sampled from (honestly nonsensical output, explained by capacity/data/steps and connected to Chinchilla/Kaplan scaling laws), then that same tiny model fine-tuned on hand-written (instruction, response) pairs with a real before/after comparison and an observed catastrophic-forgetting artifact.

### 14 – Multi-Agent Systems
Communication protocols → orchestration patterns → swarm coordination, in increasing decentralization: a from-scratch `Message`/`MessageBus` (direct/broadcast/blackboard) with a real scripted negotiation and measured message-count scaling, a manager/worker orchestrator built on top of it with real fan-out/fan-in vs. sequential comparisons and concrete worker-failure demos, then fully decentralized Particle Swarm Optimization derived from scratch in NumPy with a real convergence run and a real measured premature-convergence failure. No live external LLM API calls anywhere in this section — agents are deterministic, honestly-labeled stand-ins.

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

Small beginner projects that apply the earlier sections live here:

| Tier | Projects |
|------|---------|
| [Beginner](./projects/beginner/) | Titanic EDA, Iris Classifier, House Price Prediction, Student Performance |

Bigger, production-style ML projects live in a separate repo so this course stays focused on learning material: **[github.com/yash27007/ml-spi](https://github.com/yash27007/ml-spi)**.

See [projects/README.md](./projects/README.md) for full details and how to add your own beginner project.

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
