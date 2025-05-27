# Compliance-Ready Contract Clause Finder

[![CI](https://github.com/rohit-shinde-03/contract-clause-finder/actions/workflows/ci.yml/badge.svg)](https://github.com/rohit-shinde-03/contract-clause-finder/actions/workflows/ci.yml)  
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)

---

## 🚀 Project Overview

A Retrieval-Augmented Generation (RAG) system to ingest PDF contracts, index their clauses, and serve a Q&A API for fast, precise legal clause lookup.

**Key Features**  
- Bulk ingest and parse PDF contracts  
- Chunk text into semantic passages  
- Embed passages into a vector store  
- Serve search & Q&A via a REST API  
- Minimal front-end for clause lookup

---

## 🧰 Tech Stack

- **Language & Frameworks:** Python & [FastAPI](https://fastapi.tiangolo.com/)  
- **PDF Parsing:** Apache Tika (Docker) / PyPDF2  
- **Chunking & Embeddings:** [LangChain](https://github.com/langchain-ai/langchain) + `sentence-transformers` (`all-MiniLM` or legal-BERT)  
- **Vector Store:** [Chroma](https://github.com/chroma-core/chroma) (or FAISS)  
- **Local LLMs:** [Llama 2](https://huggingface.co/meta-llama) / [GPT4All](https://github.com/nomic-ai/gpt4all)  
- **Frontend (MVP):** [Streamlit](https://streamlit.io/) → (later: React + MUI)  
- **CI/CD:** GitHub Actions (lint + pytest)  
- **Infrastructure:** Docker Compose → (later: AWS ECS / Kubernetes)

---

## 📐 Architecture Diagram

![Architecture Diagram](docs/architecture.png)  
> *High-level flow: PDF ingest → parsing → chunking → embedding → vector DB → API → UI*

---

## 🛠️ Getting Started

### Prerequisites

- Docker & Docker Compose  
- Python 3.9+  
- Git

### Local Setup

1. **Clone repo**  
   ```bash
   git clone https://github.com/<your-username>/contract-clause-finder.git
   cd contract-clause-finder
   git checkout develop
   ```
   
2. **Build & run infra**
   ```
   docker-compose up -d
   ```
   
3. **Create & activate virtualenv**
   python -m venv .venv
   .venv\Scripts\activate
   
5. **Install dependencies**
   ```
   pip install -r requirements.txt
   ```
6. **Run the API**
   ```
   uvicorn backend.main:app --reload
   ```
## 💡 Usage Examples
1. **Ingest a contract**
  ```
  curl -X POST http://localhost:8000/ingest -F "file=@/path/to/your/contract.pdf"
  ```
2. **Search for a clause**
  ```
  curl "http://localhost:8000/search?q=termination"
  ```
## 🤝 Contributing
We follow Git Flow:

1. Fork & clone this repo

2. Create a feature branch:
   ```
   git checkout -b feature/<area>/<short-desc>
   ```
3. Commit your changes & open a PR against develop

4. Ensure all checks pass (lint, tests)

5. Upon approval, merge into develop
