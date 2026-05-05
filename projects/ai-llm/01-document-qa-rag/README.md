# 01 – Document Q&A with RAG

**Difficulty:** Advanced  
**Domain:** LLMs / Retrieval-Augmented Generation  
**Stack:** LangChain · FAISS · OpenAI / HuggingFace Embeddings · Streamlit

## Objective

Build a system that answers questions about a collection of PDF/text documents by combining semantic search (retrieval) with an LLM (generation). This is the most common real-world LLM application pattern.

## What You'll Learn

- What RAG is and why it matters (LLMs have a knowledge cutoff; documents don't)
- Document loading and chunking strategies
- Creating and storing vector embeddings
- Similarity search with FAISS
- Building a retrieval chain with LangChain
- Wrapping everything in a Streamlit UI

## Architecture

```
User Question
      ↓
[Embedding Model] → Question vector
      ↓
[Vector Store (FAISS)] → Top-k relevant chunks
      ↓
[LLM] + relevant chunks → Grounded answer
```

## Prerequisites

Sections 01–08 of the course. Basic understanding of how LLMs work (see Section 07 – NLP).

## Stack

| Component | Tool |
|-----------|------|
| Document loading | `langchain.document_loaders` |
| Text splitting | `RecursiveCharacterTextSplitter` |
| Embeddings | `OpenAIEmbeddings` or `HuggingFaceEmbeddings` |
| Vector store | `FAISS` |
| LLM | `ChatOpenAI` / `HuggingFaceHub` |
| Chain | `RetrievalQA` or LCEL |
| UI | `streamlit` |

## How to Run

```bash
# Install extra dependencies
uv add langchain langchain-openai langchain-community faiss-cpu streamlit pypdf

# Set your API key (or use HuggingFace free models)
export OPENAI_API_KEY="sk-..."

# Run the Streamlit app
streamlit run app.py
```

## Resources

- [LangChain RAG Tutorial](https://python.langchain.com/docs/tutorials/rag/)
- [DeepLearning.AI — Chat with Your Data](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/)
- [Pinecone RAG Guide](https://www.pinecone.io/learn/retrieval-augmented-generation/)
