"""
Configuration Module for Resume Intelligence AI.

This module defines the configuration-driven settings for the RAG pipeline using Pydantic Settings.
Environment variables are loaded from the project's `.env` file.
"""

import os
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

# Determine the project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


class Settings(BaseSettings):
    """
    Application settings for the RAG pipeline.
    
    Includes credentials, model selections, database connection strings, 
    and parameters for chunking and retrieval.
    """
    
    # --- LLM and Embeddings Configuration ---
    # GEMINI_API_KEY is required to generate embeddings and run the generator LLM.
    # Why RAG improves LLM accuracy:
    # Large Language Models (LLMs) are trained on static historical data and can hallucinate
    # when asked about private, corporate, or recent documents. RAG (Retrieval-Augmented Generation)
    # improves accuracy by fetching relevant, factual context from a trusted database and injecting
    # it into the LLM's prompt, turning the LLM's role from "generating knowledge from memory" 
    # to "reading comprehension and synthesis of the provided text."
    gemini_api_key: str = Field(alias="GEMINI_API_KEY", default="")
    gemini_model: str = Field(alias="GEMINI_MODEL", default="gemini-2.5-flash")
    embedding_model: str = Field(alias="EMBEDDING_MODEL", default="gemini-embedding-001")
    
    # --- Groq LLM Configuration ---
    groq_api_key: str = Field(alias="GROQ_API_KEY", default="")
    groq_model: str = Field(alias="GROQ_MODEL", default="llama-3.3-70b-versatile")

    # --- Database Configuration ---
    # Connection URL for PostgreSQL. Must use postgresql+psycopg schema for psycopg3.
    database_url: str = Field(
        alias="DATABASE_URL", 
        default="postgresql+psycopg://postgres:postgres@localhost:5432/resume_intelligence"
    )

    # --- Ingestion & Chunking Configuration ---
    # Why chunking exists:
    # LLMs have context window limits, and embedding models have input token limits (e.g., 2048 tokens).
    # Furthermore, generating a single vector embedding representing an entire multi-page document
    # dilutes semantic granularity—specific details get "averaged out."
    # Chunking splits large documents into smaller, semantically coherent segments, ensuring
    # that:
    # 1. We stay within model input limits.
    # 2. Vector search can pinpoint exact sentences or paragraphs matching the query.
    # 3. Only the most relevant sections are sent to the LLM, reducing noise and prompt costs.
    chunk_size: int = Field(alias="CHUNK_SIZE", default=1000)
    chunk_overlap: int = Field(alias="CHUNK_OVERLAP", default=200)

    # --- Retrieval & LLM Generation Configuration ---
    # How vector similarity search works:
    # A vector similarity search compares the high-dimensional embedding vector of a user's query
    # with the stored chunk embeddings using similarity metrics (such as Cosine distance or Inner Product).
    # The database ranks chunks by distance and returns the 'Top K' closest chunks.
    # Tuning TOP_K is a trade-off:
    # - Too small: The LLM might miss critical context (low recall).
    # - Too large: The prompt size increases, introducing noise and potential model distraction (low precision).
    top_k: int = Field(alias="TOP_K", default=5)
    temperature: float = Field(alias="TEMPERATURE", default=0.0)

    # Load from the local .env file in the project root
    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore"
    )


# Instantiate settings to be imported across the app
settings = Settings()
