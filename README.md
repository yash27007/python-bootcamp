# Zero to Hero: Python, Data Science, ML & AI

A structured, hands-on, first-principles curriculum covering everything from Python basics through MLOps, PyTorch, generative AI, reinforcement learning, LLMs built from scratch, and multi-agent systems. Every topic follows the same chain: Problem → Intuition → Why simpler approaches fail → Math derived from scratch → Tiny working implementation → Real experiment → Failure modes → Production usage (see [AGENTS.md](./AGENTS.md) for the full standard).

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

Every topic below links directly to its folder (`README.md` + `notes.md` + notebook/script). All 15 sections are  Complete.

1. **[Python Foundation](./01-python-foundation/)**
   - [Basics](./01-python-foundation/01-basics/)
   - [Control Flow](./01-python-foundation/02-control-flow/)
   - [Data Structures](./01-python-foundation/03-data-structures/)
   - [Functions](./01-python-foundation/04-functions/)
   - [Modules & Packages](./01-python-foundation/05-modules-packages/)
   - [File I/O & Exceptions](./01-python-foundation/06-file-exception/)
   - [OOP](./01-python-foundation/07-oops/)
   - [Advanced Concepts (Iterators/Generators)](./01-python-foundation/08-advanced-concepts/)
   - [Logging](./01-python-foundation/09-logging/)
   - [Multithreading & Concurrency](./01-python-foundation/10-multithreading/)
   - [Memory Management](./01-python-foundation/11-memory-management/)
   - [Flask](./01-python-foundation/12-flask/)
   - [Streamlit](./01-python-foundation/13-streamlit/)

2. **[Statistics](./02-statistics/)**
   - [Descriptive Statistics](./02-statistics/01-descriptive-statistics/)
   - [Probability](./02-statistics/02-probability/)
   - [Inferential Statistics](./02-statistics/03-inferential-statistics/)

3. **[Data Analysis](./03-data-analysis/)**
   - [NumPy](./03-data-analysis/01-numpy/)
   - [Pandas](./03-data-analysis/02-pandas/)
   - [Data Manipulation](./03-data-analysis/03-data-manipulation/)
   - [Reading Data (multiple sources)](./03-data-analysis/04-data-reading/)
   - [Matplotlib](./03-data-analysis/05-matplotlib/)
   - [Seaborn](./03-data-analysis/06-seaborn/)
   - [SQLite](./03-data-analysis/07-sqlite/)
   - [EDA Projects](./03-data-analysis/08-eda-projects/)

4. **[Feature Engineering](./04-feature-engineering/)**
   - [Missing Values](./04-feature-engineering/01-missing-values/)
   - [Handling Outliers](./04-feature-engineering/02-handling-outliers/)
   - [Data Encoding](./04-feature-engineering/03-data-encoding/)
   - [Handling Imbalanced Data](./04-feature-engineering/04-handling-imbalanced-dataset/)

5. **[Machine Learning](./05-machine-learning/)**
   - [Introduction](./05-machine-learning/01-introduction/)
   - [Linear Regression](./05-machine-learning/02-linear-regression/)
   - [Polynomial Regression](./05-machine-learning/03-polynomial-regression/)
   - [Regularization](./05-machine-learning/04-regularization/)
   - [Cross-Validation](./05-machine-learning/05-cross-validation/)
   - [Bias-Variance Tradeoff](./05-machine-learning/05b-bias-variance-tradeoff/)
   - [Logistic Regression](./05-machine-learning/06-logistic-regression/)
   - [SVM](./05-machine-learning/07-svm/)
   - [Naive Bayes](./05-machine-learning/08-naive-bayes/)
   - [KNN](./05-machine-learning/09-knn/)
   - [Decision Tree](./05-machine-learning/10-decision-tree/)
   - [Random Forest](./05-machine-learning/11-random-forest/)
   - [AdaBoost](./05-machine-learning/12-adaboost/)
   - [Gradient Boosting](./05-machine-learning/13-gradient-boosting/)
   - [XGBoost](./05-machine-learning/14-xgboost/)
   - [Unsupervised Learning (K-Means, Hierarchical, DBSCAN)](./05-machine-learning/15-unsupervised-learning/)
   - [Anomaly Detection (Isolation Forest, LOF)](./05-machine-learning/16-anomaly-detection/)
   - [PCA](./05-machine-learning/18-pca/)

6. **[Deep Learning](./06-deep-learning/)**
   - [ANN](./06-deep-learning/01-ann/)
   - [CNN](./06-deep-learning/02-cnn/)
   - [RNN](./06-deep-learning/03-rnn/)
   - [LSTM/GRU](./06-deep-learning/04-lstm-gru/)
   - [Attention & Transformers](./06-deep-learning/05-attention-transformers/)

7. **[NLP](./07-nlp/)**
   - [Text Preprocessing](./07-nlp/01-text-preprocessing/)
   - [Feature Extraction (BOW/TF-IDF)](./07-nlp/02-feature-extraction/)
   - [Word Embeddings (Word2Vec)](./07-nlp/03-word-embeddings/)
   - [Deep Learning NLP](./07-nlp/04-deep-learning-nlp/)
   - [Transformers & HuggingFace](./07-nlp/05-transformers-and-huggingface/)

8. **[MLOps & Deployment](./08-mlops-deployment/)**
   - [Docker](./08-mlops-deployment/01-docker/)
   - [Git](./08-mlops-deployment/02-git/)
   - [Testing & CI](./08-mlops-deployment/03-testing-ci/)
   - [Model Packaging & Versioning](./08-mlops-deployment/04-model-packaging-versioning/)
   - [MLflow / DagsHub (with DVC)](./08-mlops-deployment/05-mlflow-dagshub/)
   - [BentoML](./08-mlops-deployment/06-bentoml/)
   - [CI/CD](./08-mlops-deployment/07-cicd/)
   - [Monitoring](./08-mlops-deployment/08-monitoring/)

9. **[PyTorch](./09-pytorch/)**
   - [Tensors & Autograd](./09-pytorch/01-tensors-and-autograd/)
   - [nn.Module & Training Loop](./09-pytorch/02-nn-module-and-training-loop/)
   - [Datasets/DataLoaders & Checkpointing](./09-pytorch/03-datasets-dataloaders-checkpointing/)
   - [GPU / Mixed Precision / Profiling](./09-pytorch/04-gpu-mixed-precision-profiling/)

10. **[Distributed Data](./10-distributed-data/)**
    - [Why Distributed Processing](./10-distributed-data/01-why-distributed-processing/)
    - [PySpark (Local Mode)](./10-distributed-data/02-pyspark-local-mode/)
    - [Streaming Fundamentals](./10-distributed-data/03-streaming-fundamentals/)
    - [Kafka](./10-distributed-data/04-kafka/)

11. **[Generative AI](./11-generative-ai/)**
    - [GANs](./11-generative-ai/01-gans/)
    - [Diffusion Models](./11-generative-ai/02-diffusion-models/)

12. **[Reinforcement Learning](./12-reinforcement-learning/)**
    - [MDPs & Bellman Equation](./12-reinforcement-learning/01-mdps-and-bellman-equation/)
    - [Q-Learning](./12-reinforcement-learning/02-q-learning/)
    - [Policy Gradients](./12-reinforcement-learning/03-policy-gradients/)

13. **[LLMs From Scratch](./13-llms-from-scratch/)**
    - [Tokenizer From Scratch (BPE)](./13-llms-from-scratch/01-tokenizer-from-scratch/)
    - [Pretraining Objective (Tiny GPT)](./13-llms-from-scratch/02-pretraining-objective/)
    - [Instruction Tuning](./13-llms-from-scratch/03-instruction-tuning/)

14. **[Multi-Agent Systems](./14-multi-agent-systems/)**
    - [Agent Communication Protocols](./14-multi-agent-systems/01-communication-protocols/)
    - [Orchestration Patterns](./14-multi-agent-systems/02-orchestration-patterns/)
    - [Swarm Coordination (PSO)](./14-multi-agent-systems/03-swarm-coordination/)

15. **[Agent Skills & MCP](./15-agent-skills-and-mcp/)**
    - [Agent Skills & Progressive Disclosure](./15-agent-skills-and-mcp/01-agent-skills/)
    - [Model Context Protocol (MCP)](./15-agent-skills-and-mcp/02-model-context-protocol/)
    - [Skills+MCP Agent Loop](./15-agent-skills-and-mcp/03-skills-and-mcp-agent-loop/)

**[Projects](./projects/)**  Growing — Beginner: Titanic EDA, Iris Classifier, House Prices, Student Performance (all with notebooks) — bigger projects: [ml-spi](https://github.com/yash27007/ml-spi)

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

### 15 – Agent Skills and MCP
Knowledge disclosure → action standardization → the loop that ties them together: a real `Skill`/`SkillRegistry` with deterministic keyword-overlap selection and a measured progressive-disclosure context-savings experiment, a real MCP-shaped JSON-RPC 2.0 server/client over a genuine local subprocess-stdio boundary with real tool discovery/invocation and schema validation, then both reused verbatim in a real 4-stage agent loop with a measured skill-narrowed vs. consider-everything tool-selection experiment — the curriculum's capstone connection back to `14-multi-agent-systems`'s orchestration pattern. No live LLM or external service call anywhere in this section.

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
