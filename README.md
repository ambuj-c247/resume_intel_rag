# Resume Intelligence AI

A production-quality learning project built to teach the complete **Retrieval-Augmented Generation (RAG)** pipeline while following clean architecture and modern software engineering principles.

The application runs entirely from the terminal. There is no web interface.

---

## Table of Contents
1. [Learning Objectives & AI Concepts](#learning-objectives--ai-concepts)
2. [RAG Architecture](#rag-architecture)
3. [Folder Structure](#folder-structure)
4. [Installation](#installation)
5. [Docker Setup (PostgreSQL + pgvector)](#docker-setup-postgresql--pgvector)
6. [Environment Variables](#environment-variables)
7. [Running Ingestion](#running-ingestion)
8. [Running Interactive Chat](#running-interactive-chat)
9. [Running Ragas Evaluation](#running-ragas-evaluation)
10. [Example Questions](#example-questions)
11. [Future Improvements](#future-improvements)

---

## Learning Objectives & AI Concepts

This project is built as an educational blueprint for understanding modern AI engineering. Throughout the codebase and documentation, key AI concepts are explained:

* **Why Chunking Exists**: Large Language Models (LLMs) have context window limits. Moreover, trying to embed a multi-page resume as a single vector averages out specific details. By splitting documents into smaller, semantically coherent segments using `RecursiveCharacterTextSplitter`, we retain granular details (like specific project bullet points or languages) and reduce prompt costs.
* **Why Chunk Overlap Matters**: When documents are split, critical sentences can get cut in half. Overlapping text between consecutive chunks ensures that context and transitions are preserved at boundaries.
* **Why Embeddings are Needed**: Machines cannot natively compare strings for meaning. Embeddings map text chunks to a high-dimensional vector space (e.g., 768 dimensions for Google's `gemini-embedding-001`). Vectors pointing in similar directions indicate semantic similarity, enabling keyword-agnostic searches.
* **How pgvector Stores and Queries Vectors**: The `pgvector` extension adds a native `vector` data type and distance operators to PostgreSQL. In this project, we store embeddings in a `Vector(768)` column and use the Cosine Distance operator (`<=>` in SQL, mapped to `cosine_distance()` in SQLAlchemy) to perform fast, nearest-neighbor vector search directly inside relational tables.
* **Why RAG Improves Accuracy**: LLMs are frozen in time and prone to hallucination. RAG (Retrieval-Augmented Generation) retrieves verified facts from a document store and injects them as prompt context, shifting the LLM's task from *retrieving knowledge from parameters* to *performing reading comprehension*.
* **How Ragas Evaluates RAG Pipelines**: Standard software assertions cannot validate natural language. Ragas uses LLM-as-a-judge prompts to score metrics (Faithfulness, Context Precision, Context Recall, Answer Relevancy) from `0.0` to `1.0`.

---

## RAG Architecture

```mermaid
graph TD
    A[Raw Resume: PDF / TXT / MD] --> B[LoaderFactory]
    B --> C[RecursiveChunker]
    C --> D[GeminiEmbeddings]
    D --> E[PostgreSQL + pgvector]
    F[User Query] --> G[PostgresVectorRetriever]
    E --> G
    G --> H[Prompt Templates]
    H --> I[Gemini 2.5 Flash]
    I --> J[Terminal Chat / CLI Report]
```

The application strictly follows **Clean Architecture**:
* **`src/ingestion/`**: Handles parsing and chunking.
* **`src/embeddings/`**: Decouples the application from Gemini using a `BaseEmbeddings` abstraction.
* **`src/retrieval/`**: Abstracts semantic querying using a `BaseRetriever` interface.
* **`src/llm/`**: Decouples the generator using a `BaseLLM` interface.
* **`src/database/`**: Configures tables and sessions.

---

## Folder Structure

```
resume-intelligence-ai/
├── alembic/                 # Alembic DB migration environment
├── docker/
│   ├── postgres/
│   │   └── init.sql         # Automatically enables pgvector on startup
│   └── docker-compose.yml   # PostgreSQL + pgvector service definition
├── evaluation/
│   ├── dataset.json         # Ragas QA evaluation dataset
│   └── reports/             # Generated markdown and json evaluation reports
├── resumes/                 # Directory holding candidate resumes (PDF, TXT, MD)
├── src/
│   ├── cli/
│   ├── config/              # Configuration & env variable settings (Pydantic)
│   ├── database/            # Connection sessions and SQLAlchemy models
│   ├── embeddings/          # Embedding abstraction and Gemini implementations
│   ├── ingestion/           # File loaders and recursive chunkers
│   ├── llm/                 # LLM generation abstraction and Gemini wrapper
│   ├── prompts/             # Prompt library for Resume Intelligence features
│   ├── retrieval/           # pgvector semantic query retriever
│   ├── evaluation/          # Ragas evaluator engine setup
│   └── utils/
├── ingest.py                # Main document loading pipeline script
├── chat.py                  # Interactive CLI chat application
├── evaluate.py              # Ragas pipeline evaluation runner
├── requirements.txt         # Project dependencies
├── .env.example             # Configuration settings template
└── README.md
```

---

## Installation

1. **Clone the project** and open the workspace.
2. **Create a virtual environment** and install dependencies using Python 3.12+ (Python 3.13 is fully supported):
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```

---

## Docker Setup (PostgreSQL + pgvector)

Start the PostgreSQL database service with `pgvector` enabled:
```bash
docker compose -f docker/docker-compose.yml up -d
```
*Note: The container maps the host port `5433` to container port `5432` to prevent conflicts if you have a local PostgreSQL server running on your machine.*

---

## Environment Variables

Copy `.env.example` to `.env` and enter your credentials:
```bash
cp .env.example .env
```

Edit the `.env` file:
```env
# Google Gemini Configuration
GEMINI_API_KEY=your_actual_gemini_api_key
GEMINI_MODEL=gemini-2.5-flash
EMBEDDING_MODEL=gemini-embedding-001

# Database Configuration
DATABASE_URL=postgresql+psycopg://postgres:postgres@localhost:5433/resume_intelligence

# Chunking Parameters
CHUNK_SIZE=1000
CHUNK_OVERLAP=200

# Retrieval & Generation settings
TOP_K=5
TEMPERATURE=0.0
```

---

## Running Ingestion

To ingest the resumes inside the `resumes/` folder (such as the preloaded test resumes: `alice_developer.pdf`, `jane_smith.md`, `john_doe.txt`):

```bash
python ingest.py
```

The script performs the following pipeline stages:
1. Scans `resumes/` for supported files.
2. Performs SHA-256 hash checks to ensure it doesn't re-ingest duplicate/unchanged files.
3. Extracts text based on file format.
4. Splits text recursively using the configured chunk size and overlap.
5. Invokes Google Gemini API to generate embeddings.
6. Persists the resume record and chunk records (with vectors) into PostgreSQL.

---

## Running Interactive Chat

Start the terminal Q&A interface:
```bash
python chat.py
```

1. Select one of the ingested resumes from the numeric list.
2. Start chatting! The CLI prints every pipeline stage for educational tracing:
   - **Similarity Search**: Queries the database using the pgvector operator.
   - **Retrieved Context**: Previews the fetched chunks.
   - **Prompt Construction**: Merges context with templates.
   - **Gemini Response**: Invoking LLM for synthesis.
   - **Final Answer**: Displays clean markdown.

### Interactive Slash Command Shortcuts

Inside the chat, you can type `/help` or use the following command shortcuts:
* `/summary` — Generates a professional candidate summary.
* `/skills` — Extracts technical skills grouped by category.
* `/experience` — Outlines work experience and bullet points.
* `/projects` — Extracts projects, roles, and technologies.
* `/education` — Extracts education history.
* `/certifications` — Lists training and professional certifications.
* `/strengths` — Evaluates candidate's top strengths.
* `/weaknesses` — Reviews profile gaps and areas of improvement.
* `/questions` — Generates tailored technical/behavioral interview questions.
* `/match` — Paste a Job Description to calculate a Match Score and obtain a detailed alignment explanation.
* `/exit` or `/quit` — Ends the chat session.

---

## Running Ragas Evaluation

To run Ragas evaluation against the preloaded QA dataset `evaluation/dataset.json` (which targets the `alice_developer.pdf` profile):

```bash
python evaluate.py
```

The runner:
1. Loads the QA dataset containing query and ground truth pairs.
2. Performs retrieval and LLM generation for each question to gather predictions.
3. Evaluates predictions using Google Gemini (2.5 Flash and embedding-001) as the judge.
4. Outputs a Rich table showing scores for **Faithfulness**, **Context Precision**, **Context Recall**, and **Answer Relevancy**.
5. Persists a detailed markdown report (`eval_report_<timestamp>.md`) and JSON log (`eval_report_<timestamp>.json`) in `evaluation/reports/`.

---

## Example Questions

Once you select a resume in `chat.py`, you can test it with these prompts:
* *What programming languages does this candidate know?*
* *Does the candidate have experience with RAG or AI embeddings?*
* *Where did they go to school and when did they graduate?*
* *Did the candidate work with Docker or AWS?*
* *Has the candidate ever led a migration project?*
* *Can you tell me about their experience with Kubernetes?* (Tests grounding - should respond with *'The requested information is not available in the resume.'* if it's not present).

---

## Future Improvements

The clean architecture is designed to support:
* **Alternative Vector Databases**: Swapping pgvector with Pinecone, Qdrant, or Weaviate by implementing `BaseRetriever`.
* **Alternate LLM Providers**: Supporting OpenAI, Claude, or local Ollama instances by inheriting `BaseLLM`.
* **Advanced Retrieval**: Implementing Hybrid Search, MMR, or Parent Document Retrieval within `PostgresVectorRetriever`.
