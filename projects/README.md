# Projects

End-to-end projects that apply the concepts taught throughout this course. Projects are organised by difficulty and domain.

> Projects are the best way to learn. Pick one that interests you, build it, break it, and rebuild it better.

---

## Structure

```
projects/
├── beginner/          ← Python + EDA + basic ML
├── intermediate/      ← Full ML pipelines, feature engineering
├── advanced/          ← Deep learning, NLP, production-grade
└── ai-llm/            ← LangChain, RAG, agents, fine-tuning
```

---

## Beginner Projects

| # | Project | Skills | Dataset |
|---|---------|--------|---------|
| 01 | [Titanic Survival EDA](./beginner/01-titanic-eda/) | Pandas, Matplotlib, Seaborn | [Kaggle Titanic](https://www.kaggle.com/c/titanic) |
| 02 | [Iris Flower Classifier](./beginner/02-iris-classifier/) | scikit-learn, classification | Built-in sklearn |
| 03 | [House Price Prediction](./beginner/03-house-price-prediction/) | Regression, feature engineering | [Kaggle House Prices](https://www.kaggle.com/c/house-prices-advanced-regression-techniques) |
| 04 | [Student Performance Analysis](./beginner/04-student-performance/) | EDA, visualisation | [UCI ML Repo](https://archive.ics.uci.edu/ml/datasets/Student+Performance) |

---

## Intermediate Projects

| # | Project | Skills | Dataset |
|---|---------|--------|---------|
| 01 | [Customer Churn Prediction](./intermediate/01-customer-churn/) | Classification, SMOTE, pipelines | [Telco Customer Churn](https://www.kaggle.com/blastchar/telco-customer-churn) |
| 02 | [Credit Card Fraud Detection](./intermediate/02-fraud-detection/) | Imbalanced data, anomaly detection | [Kaggle Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud) |
| 03 | [Time Series: Sales Forecasting](./intermediate/03-sales-forecasting/) | ARIMA, Prophet, LSTM | [Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales) |
| 04 | [Sentiment Analysis (Traditional ML)](./intermediate/04-sentiment-analysis/) | NLP, TF-IDF, Naive Bayes | [IMDB Reviews](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) |

---

## Advanced Projects

| # | Project | Skills | Notes |
|---|---------|--------|-------|
| 01 | [Image Classifier with CNN](./advanced/01-image-cnn/) | PyTorch/TensorFlow, transfer learning | CIFAR-10 or custom |
| 02 | [Transformer from Scratch](./advanced/02-transformer-scratch/) | Attention, PyTorch | Educational build |
| 03 | [End-to-End MLOps Pipeline](./advanced/03-mlops-pipeline/) | MLflow, DVC, Docker, GitHub Actions | Any dataset |
| 04 | [Sensor Fault Detection](./advanced/04-sensor-fault-detection/) | MongoDB, MLflow, production deployment | Industrial IoT |

---

## AI / LLM Projects

| # | Project | Skills | Stack |
|---|---------|--------|-------|
| 01 | [Document Q&A with RAG](./ai-llm/01-document-qa-rag/) | LangChain, FAISS, embeddings | OpenAI / HuggingFace |
| 02 | [Chatbot with Memory](./ai-llm/02-chatbot-memory/) | LangChain, conversation history | Streamlit + LangChain |
| 03 | [Multi-document Summariser](./ai-llm/03-multi-doc-summariser/) | LangChain, map-reduce chains | Any LLM |
| 04 | [LLM Fine-tuning (LoRA)](./ai-llm/04-llm-finetuning/) | PEFT, LoRA, HuggingFace | LLaMA / Mistral |

---

## How to Contribute a Project

1. Pick an idea or open an [issue](https://github.com/yash27007/python-bootcamp/issues) to propose one
2. Create your project folder under the appropriate difficulty tier
3. Use the project template below
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

**Difficulty:** Beginner / Intermediate / Advanced  
**Domain:** Classification / Regression / NLP / CV / LLMs  
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
