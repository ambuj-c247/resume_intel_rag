# Resume Intelligence AI
## RAG Learning Project

---

# Project Overview

Build a terminal-based Resume Intelligence application using Retrieval-Augmented Generation (RAG).

The purpose of this project is to understand and implement the complete RAG lifecycle while following clean architecture and modern software engineering practices.

The project should demonstrate how an AI system can understand resumes, retrieve relevant context using semantic search, and generate accurate responses grounded only in the resume content.

This is a learning-focused project, but it should be structured like a production-ready application.

No web interface is required.

The application will run entirely from the terminal.

---

# Learning Objectives

This project should help understand:

- Retrieval-Augmented Generation (RAG)
- Document Processing
- Chunking Strategies
- Embedding Generation
- Vector Databases
- Semantic Search
- Prompt Engineering
- Grounded LLM Responses
- AI Evaluation
- Clean AI Architecture

---

# Technology Stack

## Language

Python 3.12+

## AI Framework

LangChain

## LLM

Google Gemini API

Default Model

gemini-2.5-flash

## Embeddings

Google Gemini Embedding API

Default Model

gemini-embedding-001

## Database

PostgreSQL

## Vector Extension

pgvector

## Database Driver

psycopg3

## ORM

SQLAlchemy

## Database Migration

Alembic

## Evaluation

Ragas

## CLI

Typer

## Logging

Rich

## Configuration

python-dotenv

---

# Architecture

The application should be modular.

Each layer must have a single responsibility.

The application should be easy to extend with:

- Different LLM providers
- Different embedding providers
- Different retrieval strategies
- Different vector databases

Business logic should not depend directly on Gemini or PostgreSQL implementations.

---

# Functional Requirements

## Resume Loading

Support

- PDF
- TXT
- Markdown

Resumes should be stored inside

resumes/

---

## Document Processing

Extract text from resumes.

Clean unnecessary whitespace.

Preserve logical document structure where possible.

---

## Chunking

Use

RecursiveCharacterTextSplitter

Configuration

- Chunk Size
- Chunk Overlap

Chunking configuration should be editable without changing code.

---

## Embedding Generation

Generate embeddings using

gemini-embedding-001

Store embeddings in PostgreSQL using pgvector.

Embedding logic should be isolated behind an interface.

---

## Database

Use PostgreSQL with pgvector.

Store

- Resume metadata
- Document chunks
- Chunk embeddings

The database should support persistence across application restarts.

---

## Retrieval

Implement semantic similarity search.

Configuration

Top K

Design retrieval so future implementations can support

- Hybrid Search
- MMR
- Parent Document Retrieval
- Context Compression

---

## LLM

Use

Gemini 2.5 Flash

Generate answers only from retrieved context.

Prevent hallucinations by instructing the model not to invent information.

---

# Resume Intelligence Features

The assistant should answer questions such as

- Summarize this resume
- Extract technical skills
- Extract frontend skills
- Extract backend skills
- Extract cloud technologies
- Extract education
- Extract certifications
- Extract work experience
- Calculate years of experience
- Extract projects
- Find React experience
- Find AI experience
- Find AWS experience
- Generate interview questions
- Generate strengths
- Generate weaknesses
- Compare resume against a Job Description
- Calculate resume match score
- Explain why the score was assigned

---

# Terminal Commands

## Start Database

docker compose up -d

---

## Ingest Documents

python ingest.py

Expected

Loading resumes...

Processing documents...

Chunking...

Generating embeddings...

Saving to PostgreSQL...

Done.

---

## Interactive Chat

python chat.py

Example

User

What programming languages does this candidate know?

Assistant

Python

JavaScript

TypeScript

---

## Evaluation

python evaluate.py

Expected

Faithfulness ............ 0.95

Context Precision ....... 0.93

Context Recall .......... 0.91

Answer Relevancy ........ 0.96

---

# Logging

Display every stage of the pipeline.

Example

Loading Resume

↓

Text Extraction

↓

Chunking

↓

Embedding Generation

↓

Saving to PostgreSQL

↓

Similarity Search

↓

Retrieved Context

↓

Prompt Construction

↓

Gemini Response

↓

Final Answer

This logging is mandatory for learning.

---

# Docker

Docker should only be used for PostgreSQL.

Python should run locally inside a virtual environment.

Create

docker-compose.yml

The compose file should include

- PostgreSQL
- pgvector extension
- Persistent Docker volume
- Database initialization script

---

# Project Structure

resume-intelligence-ai/

├── docker/
│   ├── postgres/
│   │   └── init.sql
│   └── docker-compose.yml
│
├── resumes/
│
├── evaluation/
│   ├── dataset.json
│   └── reports/
│
├── src/
│   ├── config/
│   ├── database/
│   ├── ingestion/
│   ├── embeddings/
│   ├── retrieval/
│   ├── llm/
│   ├── prompts/
│   ├── evaluation/
│   ├── cli/
│   └── utils/
│
├── ingest.py
├── chat.py
├── evaluate.py
├── requirements.txt
├── README.md
├── .env.example
└── .gitignore

---

# Configuration

Environment variables

- GEMINI_API_KEY
- GEMINI_MODEL
- EMBEDDING_MODEL
- DATABASE_URL
- CHUNK_SIZE
- CHUNK_OVERLAP
- TOP_K
- TEMPERATURE

---

# Evaluation

Use Ragas.

Evaluate

- Faithfulness
- Context Precision
- Context Recall
- Answer Relevancy

Create a sample evaluation dataset.

Generate a readable report.

---

# Code Quality

The project must include

- Modular Architecture
- SOLID Principles
- Type Hints
- Docstrings
- Error Handling
- Proper Logging
- Configuration Driven Design
- Small Reusable Components

---

# Documentation

README should include

- Project Overview
- Complete RAG Architecture
- Folder Structure
- Installation
- Docker Setup
- PostgreSQL Setup
- Environment Variables
- Running Ingestion
- Running Chat
- Running Evaluation
- Example Questions
- Future Improvements

---

# Future Improvements

The architecture should make it easy to add

- Local LLMs
- Ollama
- OpenAI
- Claude
- Azure OpenAI
- Pinecone
- Qdrant
- Weaviate
- LangGraph
- LangSmith
- REST API
- Web Interface
- Multi Resume Search
- Resume Ranking

These features do not need to be implemented now.

Only design the architecture to support them.

---

# Success Criteria

A user should be able to

1. Start PostgreSQL using Docker.
2. Ingest resumes.
3. Generate embeddings.
4. Store vectors inside PostgreSQL.
5. Ask questions.
6. Receive grounded responses.
7. Run Ragas evaluation.
8. Experiment with chunk size, Top-K, and embedding settings.

The project should prioritise readability, modularity, extensibility, and learning over production optimisation.