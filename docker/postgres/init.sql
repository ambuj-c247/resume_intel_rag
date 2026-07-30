-- Initialize vector extension for the database.
-- 
-- How pgvector stores embeddings:
-- pgvector is an open-source extension for PostgreSQL that adds a custom data type: `vector`.
-- This data type represents a point in a high-dimensional space (e.g., 768 dimensions for Google's gemini-embedding-001).
-- Rather than storing arrays of floats as text or JSON, pgvector stores them as binary structures on disk,
-- allowing highly optimized distance computations (like Cosine distance, Inner Product, and L2 distance) 
-- to be executed natively by PostgreSQL.
-- It also supports advanced indexes (such as IVFFlat or HNSW) to accelerate query speeds 
-- for nearest-neighbor searches in large-scale datasets.

CREATE EXTENSION IF NOT EXISTS vector;
